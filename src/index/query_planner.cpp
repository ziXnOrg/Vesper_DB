// Proper includes for implementation below
#include "vesper/index/index_manager.hpp"
#include "vesper/core/platform_utils.hpp" // safe_getenv
#include <atomic>
#include <shared_mutex>
#include <optional>
#include <vector>
#include <cmath>

// Minimal, thread-safe QueryPlanner::Impl implementation (H2 + H3 frozen mode)
class vesper::index::QueryPlanner::Impl {
public:
    explicit Impl(const IndexManager& manager)
        : manager_(manager)
        , frozen_mode_([] {
              // Cache frozen toggle once for determinism and perf
              auto v = vesper::core::safe_getenv("VESPER_PLANNER_FROZEN");
              return (v && !v->empty() && ((*v)[0] == '1'));
          }()) {}

    auto plan(const float* /*query*/, const QueryConfig& config) -> QueryPlan {
        // Deterministic, fixed logic in both modes (no adaptive reads here yet)
        QueryPlan out{};
        out.config = config;
        out.estimated_cost_ms = 0.0f;
        out.estimated_recall = 1.0f;
        out.explanation = "Minimal planner (thread-safe)";

        // Pick first active index if available; else default to HNSW
        auto active = manager_.get_active_indexes();
        if (!active.empty()) {
            out.index = active.front();
        } else {
            out.index = IndexType::HNSW;
            out.explanation = "No indexes available";
        }

        // Maintain read-phase lock pattern for future shared-state reads
        std::shared_lock<std::shared_mutex> _(state_mutex_);
        (void)_; // silence unused in minimal implementation

        plans_generated_.fetch_add(1, std::memory_order_acq_rel);
        return out;
    }

    auto update_stats(const QueryPlan& plan, float actual_time_ms,
                      std::optional<float> actual_recall) -> void {
        // In frozen mode, skip adaptive aggregate updates but keep counters
        if (frozen_mode_) {
            (void)plan; (void)actual_time_ms; (void)actual_recall;
            plans_executed_.fetch_add(1, std::memory_order_acq_rel);
            return; // no locking needed since no shared state changes
        }

        // Write-phase: update aggregates
        std::unique_lock<std::shared_mutex> _(state_mutex_);
        (void)plan;
        total_estimation_error_ += std::abs(actual_time_ms - 0.0f);
        if (actual_recall.has_value()) {
            total_recall_error_ += std::abs(*actual_recall - 1.0f);
        }
        _.unlock();
        plans_executed_.fetch_add(1, std::memory_order_acq_rel);
    }

    auto get_stats() const -> PlannerStats {
        PlannerStats s{};
        s.plans_generated = plans_generated_.load(std::memory_order_acquire);
        s.plans_executed = plans_executed_.load(std::memory_order_acquire);
        std::shared_lock<std::shared_mutex> _(state_mutex_);
        if (s.plans_executed > 0) {
            s.avg_estimation_error_ms = total_estimation_error_ / static_cast<float>(s.plans_executed);
            s.avg_recall_error = total_recall_error_ / static_cast<float>(s.plans_executed);
        }
        return s;
    }

private:
    const IndexManager& manager_;
    const bool frozen_mode_;
    mutable std::shared_mutex state_mutex_;
    std::atomic<std::uint64_t> plans_generated_{0};
    std::atomic<std::uint64_t> plans_executed_{0};
    float total_estimation_error_{0.0f};
    float total_recall_error_{0.0f};
};

// QueryPlanner public interface implementation

vesper::index::QueryPlanner::QueryPlanner(const IndexManager& manager)
    : impl_(std::make_unique<Impl>(manager)) {}

vesper::index::QueryPlanner::~QueryPlanner() = default;

auto vesper::index::QueryPlanner::plan(const float* query, const QueryConfig& config) -> QueryPlan {
    return impl_->plan(query, config);
}

auto vesper::index::QueryPlanner::update_stats(const QueryPlan& plan, float actual_time_ms,
                               std::optional<float> actual_recall) -> void {
    impl_->update_stats(plan, actual_time_ms, actual_recall);
}

auto vesper::index::QueryPlanner::get_stats() const -> PlannerStats {
    return impl_->get_stats();
}

