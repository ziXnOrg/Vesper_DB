/** \file query_planner.cpp
 *  \brief Cost-based query planning for optimal index selection.
 */

#include "vesper/index/index_manager.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>

namespace vesper::index {

/** \brief Cost model for different index types. */
struct CostModel {
    // Base costs (milliseconds)
    float hnsw_base_cost{0.1f};
    float ivf_pq_base_cost{0.5f};
    float diskann_base_cost{2.0f};
    
    // Scaling factors
    float hnsw_log_factor{0.01f};      // O(log N)
    float ivf_linear_factor{0.001f};   // O(nprobe * posting_size)
    float diskann_io_factor{0.1f};     // O(L * disk_reads)
    
    // Recall penalties (ms per 1% recall loss)
    float recall_penalty{0.5f};
    
    // Cache effects
    float cache_hit_reduction{0.5f};   // 50% time reduction on cache hit
    float cache_miss_penalty{1.5f};    // 50% time increase on cache miss
};

/** \brief Historical performance data for adaptive tuning. */
struct PerformanceHistory {
    struct Entry {
        IndexType index;
        QueryConfig config;
        float actual_time_ms;
        float estimated_time_ms;
        float actual_recall;
        std::chrono::steady_clock::time_point timestamp;
    };
    
    std::vector<Entry> entries;
    std::size_t max_entries{1000};
    
    void add(const Entry& entry) {
        if (entries.size() >= max_entries) {
            // Remove oldest entry
            entries.erase(entries.begin());
        }
        entries.push_back(entry);
    }
    
    float get_avg_error(IndexType index) const {
        float total_error = 0;
        int count = 0;
        
        for (const auto& e : entries) {
            if (e.index == index) {
                total_error += std::abs(e.actual_time_ms - e.estimated_time_ms);
                count++;
            }
        }
        
        return count > 0 ? total_error / count : 0;
    }
    
    float get_avg_recall(IndexType index) const {
        float total_recall = 0;
        int count = 0;
        
        for (const auto& e : entries) {
            if (e.index == index) {
                total_recall += e.actual_recall;
                count++;
            }
        }
        
        return count > 0 ? total_recall / count : 0.95f;
    }
};

/** \brief Implementation class for QueryPlanner. */
class QueryPlanner::Impl {
public:
    explicit Impl(const IndexManager& manager)
        : manager_(manager) {
        // Initialize cost model with default values
        cost_model_ = CostModel{};
        
        // Get initial statistics from manager
        auto stats = manager_.get_stats();
        for (const auto& s : stats) {
            index_stats_[s.type] = s;
        }
    }
    
    ~Impl() = default;
    
    auto plan(const float* query, const QueryConfig& config) -> QueryPlan {
        QueryPlan best_plan;
        best_plan.estimated_cost_ms = std::numeric_limits<float>::max();
        
        // Get available indexes
        auto active_indexes = manager_.get_active_indexes();
        
        if (active_indexes.empty()) {
            best_plan.explanation = "No indexes available";
            return best_plan;
        }
        
        // Evaluate each index
        for (auto index_type : active_indexes) {
            QueryPlan plan = evaluate_index(index_type, query, config);
            
            // Select plan with best cost/recall tradeoff
            float score = compute_plan_score(plan, config);
            float best_score = compute_plan_score(best_plan, config);
            
            if (score < best_score) {
                best_plan = plan;
            }
        }
        
        // Adapt parameters based on history
        adapt_parameters(best_plan);
        
        // Update statistics
        plans_generated_++;
        
        return best_plan;
    }
    
    auto update_stats(const QueryPlan& plan, float actual_time_ms, 
                     std::optional<float> actual_recall) -> void {
        // Record performance
        PerformanceHistory::Entry entry;
        entry.index = plan.index;
        entry.config = plan.config;
        entry.actual_time_ms = actual_time_ms;
        entry.estimated_time_ms = plan.estimated_cost_ms;
        entry.actual_recall = actual_recall.value_or(plan.estimated_recall);
        entry.timestamp = std::chrono::steady_clock::now();
        
        history_.add(entry);
        
        // Update cost model based on error
        float error_ratio = actual_time_ms / plan.estimated_cost_ms;
        adapt_cost_model(plan.index, error_ratio);
        
        // Update statistics
        plans_executed_++;
        total_estimation_error_ += std::abs(actual_time_ms - plan.estimated_cost_ms);
        
        if (actual_recall.has_value()) {
            total_recall_error_ += std::abs(*actual_recall - plan.estimated_recall);
        }
    }
    
    auto get_stats() const -> PlannerStats {
        PlannerStats stats;
        stats.plans_generated = plans_generated_;
        stats.plans_executed = plans_executed_;
        
        if (plans_executed_ > 0) {
            stats.avg_estimation_error_ms = total_estimation_error_ / plans_executed_;
            stats.avg_recall_error = total_recall_error_ / plans_executed_;
        }
        
        return stats;
    }
    
private:
    QueryPlan evaluate_index(IndexType index, const float* query, 
                           const QueryConfig& config) {
        QueryPlan plan;
        plan.index = index;
        plan.config = config;
        
        // Get index statistics
        auto it = index_stats_.find(index);
        if (it == index_stats_.end()) {
            plan.estimated_cost_ms = 1000.0f; // Default high cost
            plan.estimated_recall = 0.5f;
            plan.explanation = "No statistics available";
            return plan;
        }
        
        const auto& stats = it->second;
        
        // Estimate cost based on index type
        switch (index) {
            case IndexType::HNSW:
                plan = evaluate_hnsw(stats, config);
                break;
            case IndexType::IVF_PQ:
                plan = evaluate_ivf_pq(stats, config);
                break;
            case IndexType::DiskANN:
                plan = evaluate_diskann(stats, config);
                break;
        }
        
        // Adjust for historical performance
        float avg_error = history_.get_avg_error(index);
        if (avg_error > 0) {
            // Adjust estimate based on historical error
            float adjustment = 1.0f + (avg_error / plan.estimated_cost_ms) * 0.1f;
            plan.estimated_cost_ms *= adjustment;
        }
        
        return plan;
    }
    
    QueryPlan evaluate_hnsw(const IndexStats& stats, const QueryConfig& config) {
        QueryPlan plan;
        plan.index = IndexType::HNSW;
        plan.config = config;
        
        // HNSW cost: O(ef_search * log(N))
        float n = static_cast<float>(stats.num_vectors);
        float cost = cost_model_.hnsw_base_cost;
        cost += config.ef_search * cost_model_.hnsw_log_factor * std::log2(n + 1);
        
        // Adjust for memory residency
        if (stats.memory_usage_bytes < 1024 * 1024 * 1024) { // < 1GB
            cost *= cost_model_.cache_hit_reduction;
        }
        
        plan.estimated_cost_ms = cost;
        
        // Estimate recall based on ef_search
        float recall = std::min(0.99f, 0.85f + 0.001f * config.ef_search);
        plan.estimated_recall = recall;
        
        // Optimize ef_search for target recall
        if (config.ef_search == 64) { // Default value
            float optimal_ef = estimate_optimal_ef(stats.num_vectors, config.k);
            plan.config.ef_search = static_cast<std::uint32_t>(optimal_ef);
        }
        
        plan.explanation = "HNSW: Fast in-memory search, O(log N) complexity";
        
        return plan;
    }
    
    QueryPlan evaluate_ivf_pq(const IndexStats& stats, const QueryConfig& config) {
        QueryPlan plan;
        plan.index = IndexType::IVF_PQ;
        plan.config = config;
        
        // IVF-PQ cost: O(nprobe * posting_size + PQ_distance_cost)
        float n = static_cast<float>(stats.num_vectors);
        float nlist = std::sqrt(n); // Typical nlist
        float posting_size = n / nlist;
        
        float cost = cost_model_.ivf_pq_base_cost;
        cost += config.nprobe * posting_size * cost_model_.ivf_linear_factor;
        
        // PQ distance computation is fast
        cost += config.nprobe * posting_size * 0.0001f;
        
        plan.estimated_cost_ms = cost;
        
        // Estimate recall based on nprobe
        float recall = std::min(0.98f, 0.70f + 0.03f * config.nprobe);
        plan.estimated_recall = recall;
        
        // Optimize nprobe for target recall
        if (config.nprobe == 8) { // Default value
            float optimal_nprobe = estimate_optimal_nprobe(nlist, config.k);
            plan.config.nprobe = static_cast<std::uint32_t>(optimal_nprobe);
        }
        
        plan.explanation = "IVF-PQ: Memory-efficient with PQ compression";
        
        return plan;
    }
    
    QueryPlan evaluate_diskann(const IndexStats& stats, const QueryConfig& config) {
        QueryPlan plan;
        plan.index = IndexType::DiskANN;
        plan.config = config;
        
        // DiskANN cost: O(L * disk_read_cost)
        float cost = cost_model_.diskann_base_cost;
        cost += config.l_search * cost_model_.diskann_io_factor;
        
        // Cache effects
        if (stats.query_count > 100) {
            // Warm cache reduces cost
            cost *= 0.7f;
        } else {
            // Cold cache increases cost
            cost *= cost_model_.cache_miss_penalty;
        }
        
        plan.estimated_cost_ms = cost;
        
        // Estimate recall based on L
        float recall = std::min(0.99f, 0.80f + 0.001f * config.l_search);
        plan.estimated_recall = recall;
        
        // Optimize L for target recall
        if (config.l_search == 128) { // Default value
            float optimal_l = estimate_optimal_l(stats.num_vectors, config.k);
            plan.config.l_search = static_cast<std::uint32_t>(optimal_l);
        }
        
        plan.explanation = "DiskANN: Billion-scale with SSD storage";
        
        return plan;
    }
    
    float compute_plan_score(const QueryPlan& plan, const QueryConfig& config) {
        // Score = cost + recall_penalty
        float score = plan.estimated_cost_ms;
        
        // Penalize if recall is below target
        float target_recall = 0.95f; // Default target
        if (plan.estimated_recall < target_recall) {
            float recall_gap = target_recall - plan.estimated_recall;
            score += recall_gap * 100 * cost_model_.recall_penalty;
        }
        
        return score;
    }
    
    void adapt_parameters(QueryPlan& plan) {
        // Adjust parameters based on historical performance
        float avg_recall = history_.get_avg_recall(plan.index);
        
        if (avg_recall < 0.95f) {
            // Increase search effort
            switch (plan.index) {
                case IndexType::HNSW:
                    plan.config.ef_search = static_cast<std::uint32_t>(
                        plan.config.ef_search * 1.2f);
                    break;
                case IndexType::IVF_PQ:
                    plan.config.nprobe = static_cast<std::uint32_t>(
                        plan.config.nprobe * 1.2f);
                    break;
                case IndexType::DiskANN:
                    plan.config.l_search = static_cast<std::uint32_t>(
                        plan.config.l_search * 1.2f);
                    break;
            }
        }
    }
    
    void adapt_cost_model(IndexType index, float error_ratio) {
        // Simple exponential smoothing
        const float alpha = 0.1f; // Learning rate
        
        switch (index) {
            case IndexType::HNSW:
                cost_model_.hnsw_base_cost *= (1 - alpha) + error_ratio * alpha;
                break;
            case IndexType::IVF_PQ:
                cost_model_.ivf_pq_base_cost *= (1 - alpha) + error_ratio * alpha;
                break;
            case IndexType::DiskANN:
                cost_model_.diskann_base_cost *= (1 - alpha) + error_ratio * alpha;
                break;
        }
    }
    
    float estimate_optimal_ef(std::size_t n, std::uint32_t k) {
        // Heuristic: ef = k * log2(n) / 10
        return std::max(64.0f, static_cast<float>(k * std::log2(n + 1) / 10.0));
    }
    
    float estimate_optimal_nprobe(float nlist, std::uint32_t k) {
        // Heuristic: nprobe = sqrt(nlist) * (k / 10)
        return std::max(8.0f, static_cast<float>(std::sqrt(nlist) * (k / 10.0)));
    }
    
    float estimate_optimal_l(std::size_t n, std::uint32_t k) {
        // Heuristic: L = k * log2(n) / 5
        return std::max(128.0f, static_cast<float>(k * std::log2(n + 1) / 5.0));
    }
    
private:
    const IndexManager& manager_;
    CostModel cost_model_;
    PerformanceHistory history_;
    std::unordered_map<IndexType, IndexStats> index_stats_;
    
    // Statistics
    std::uint64_t plans_generated_{0};
    std::uint64_t plans_executed_{0};
    float total_estimation_error_{0};
    float total_recall_error_{0};
};

// QueryPlanner public interface implementation

QueryPlanner::QueryPlanner(const IndexManager& manager)
    : impl_(std::make_unique<Impl>(manager)) {}

QueryPlanner::~QueryPlanner() = default;

auto QueryPlanner::plan(const float* query, const QueryConfig& config) -> QueryPlan {
    return impl_->plan(query, config);
}

auto QueryPlanner::update_stats(const QueryPlan& plan, float actual_time_ms, 
                               std::optional<float> actual_recall) -> void {
    impl_->update_stats(plan, actual_time_ms, actual_recall);
}

auto QueryPlanner::get_stats() const -> PlannerStats {
    return impl_->get_stats();
}

} // namespace vesper::index