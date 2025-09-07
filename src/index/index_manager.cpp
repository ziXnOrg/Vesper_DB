/** \file index_manager.cpp
 *  \brief Implementation of unified index management for multiple index types.
 */

#include "vesper/index/index_manager.hpp"
#include "vesper/kernels/dispatch.hpp"
#include <expected>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <numeric>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>

namespace vesper::index {

/** \brief Implementation class for IndexManager. */
class IndexManager::Impl {
public:
    explicit Impl(std::size_t dimension)
        : dimension_(dimension)
        , memory_budget_mb_(1024)
        , total_vectors_(0) {
    }
    
    ~Impl() = default;
    
    auto build(const float* vectors, std::size_t n, const IndexBuildConfig& config)
        -> std::expected<void, core::error> {
        
        using core::error;
        using core::error_code;
        
        // Validate input
        if (!vectors || n == 0) {
            return std::vesper_unexpected(error{
                error_code::precondition_failed,
                "Invalid input vectors",
                "index_manager"
            });
        }
        
        // Store configuration
        build_config_ = config;
        total_vectors_ = n;
        
        // Determine which indexes to build based on strategy
        std::vector<IndexType> indexes_to_build;
        
        if (config.strategy == SelectionStrategy::Manual) {
            indexes_to_build.push_back(config.type);
        } else if (config.strategy == SelectionStrategy::Auto) {
            // Auto-select based on data size and memory budget
            indexes_to_build.push_back(select_optimal_index(n, config));
        } else { // Hybrid
            // Build multiple indexes for query-time selection
            if (n < 1000000) {
                indexes_to_build.push_back(IndexType::HNSW);
            }
            if (n >= 10000 && n <= 10000000) {
                indexes_to_build.push_back(IndexType::IVF_PQ);
            }
            if (n >= 100000 && config.memory_budget_mb >= 512) {
                indexes_to_build.push_back(IndexType::DiskANN);
            }
        }
        
        // Build selected indexes
        for (auto index_type : indexes_to_build) {
            auto result = build_index(index_type, vectors, n, config);
            if (!result) {
                return result;
            }
        }
        
        return {};
    }
    
    auto add(std::uint64_t id, const float* vector)
        -> std::expected<void, core::error> {
        
        using core::error;
        using core::error_code;
        
        if (!vector) {
            return std::vesper_unexpected(error{
                error_code::precondition_failed,
                "Invalid vector",
                "index_manager"
            });
        }
        
        std::unique_lock lock(mutex_);
        
        // Add to all active indexes
        if (hnsw_index_) {
            auto result = hnsw_index_->add(id, vector);
            if (!result) return result;
        }
        
        if (ivf_pq_index_) {
            // IVF-PQ only supports batch add
            std::uint64_t ids[] = {id};
            auto result = ivf_pq_index_->add(&ids[0], vector, 1);
            if (!result) return result;
        }
        
        if (disk_graph_index_) {
            // DiskANN typically requires rebuild for additions
            // For now, mark as needing optimization
            needs_optimization_ = true;
        }
        
        total_vectors_++;
        return {};
    }
    
    auto add_batch(const std::uint64_t* ids, const float* vectors, std::size_t n)
        -> std::expected<void, core::error> {
        
        using core::error;
        using core::error_code;
        
        if (!ids || !vectors || n == 0) {
            return std::vesper_unexpected(error{
                error_code::precondition_failed,
                "Invalid batch input",
                "index_manager"
            });
        }
        
        std::unique_lock lock(mutex_);
        
        // Batch add to all active indexes
        if (hnsw_index_) {
            for (std::size_t i = 0; i < n; ++i) {
                auto result = hnsw_index_->add(ids[i], vectors + i * dimension_);
                if (!result) return result;
            }
        }
        
        if (ivf_pq_index_) {
            auto result = ivf_pq_index_->add(ids, vectors, n);
            if (!result) return result;
        }
        
        if (disk_graph_index_) {
            needs_optimization_ = true;
        }
        
        total_vectors_ += n;
        return {};
    }
    
    auto search(const float* query, const QueryConfig& config)
        -> std::expected<std::vector<std::pair<std::uint64_t, float>>, core::error> {
        
        using core::error;
        using core::error_code;
        
        if (!query) {
            return std::vesper_unexpected(error{
                error_code::precondition_failed,
                "Invalid query vector",
                "index_manager"
            });
        }
        
        std::shared_lock lock(mutex_);
        
        // Select index based on configuration or query planner
        IndexType selected_index = IndexType::HNSW;
        
        if (config.preferred_index.has_value()) {
            selected_index = *config.preferred_index;
        } else if (config.use_query_planner && query_planner_) {
            // Use query planner to select optimal index
            auto plan = query_planner_->plan(query, config);
            selected_index = plan.index;
        } else {
            // Simple heuristic: use fastest available index
            if (hnsw_index_) {
                selected_index = IndexType::HNSW;
            } else if (ivf_pq_index_) {
                selected_index = IndexType::IVF_PQ;
            } else if (disk_graph_index_) {
                selected_index = IndexType::DiskANN;
            }
        }
        
        // Execute search on selected index
        switch (selected_index) {
            case IndexType::HNSW:
                if (hnsw_index_) {
                    HnswSearchParams params;
                    params.k = config.k;
                    params.ef = config.ef_search;
                    return hnsw_index_->search(query, params);
                }
                break;
                
            case IndexType::IVF_PQ:
                if (ivf_pq_index_) {
                    // IVF-PQ search not yet fully implemented
                    // IvfSearchParams params;
                    // params.k = config.k;
                    // params.nprobe = config.nprobe;
                    // return ivf_pq_index_->search(query, params);
                    return std::vector<std::pair<std::uint64_t, float>>();
                }
                break;
                
            case IndexType::DiskANN:
                if (disk_graph_index_) {
                    VamanaSearchParams params;
                    params.k = config.k;
                    params.search_L = config.l_search;
                    return disk_graph_index_->search(std::span<const float>(query, dimension_), params);
                }
                break;
        }
        
        return std::vesper_unexpected(error{
            error_code::not_found,
            "No suitable index available",
            "index_manager"
        });
    }
    
    auto search_batch(const float* queries, std::size_t nq, const QueryConfig& config)
        -> std::expected<std::vector<std::vector<std::pair<std::uint64_t, float>>>, core::error> {
        
        std::vector<std::vector<std::pair<std::uint64_t, float>>> results;
        results.reserve(nq);
        
        for (std::size_t i = 0; i < nq; ++i) {
            auto result = search(queries + i * dimension_, config);
            if (!result) {
                return std::vesper_unexpected(result.error());
            }
            results.push_back(std::move(*result));
        }
        
        return results;
    }
    
    auto remove(std::uint64_t id) -> std::expected<void, core::error> {
        std::unique_lock lock(mutex_);
        
        // Remove from all active indexes
        if (hnsw_index_) {
            // HNSW doesn't support removal, mark for rebuild
            needs_optimization_ = true;
        }
        
        if (ivf_pq_index_) {
            // IVF-PQ can use tombstones
            // removed_ids_.insert(id);  // TODO: Implement tombstones
            needs_optimization_ = true;
        }
        
        if (disk_graph_index_) {
            // DiskANN requires rebuild
            needs_optimization_ = true;
        }
        
        return {};
    }
    
    auto get_stats() const -> std::vector<IndexStats> {
        std::shared_lock lock(mutex_);
        std::vector<IndexStats> stats;
        
        if (hnsw_index_) {
            IndexStats s;
            s.type = IndexType::HNSW;
            s.num_vectors = total_vectors_;
            s.memory_usage_bytes = estimate_hnsw_memory();
            s.measured_recall = hnsw_stats_.measured_recall;
            s.avg_query_time_ms = hnsw_stats_.total_time_ms / 
                                 std::max(1ULL, hnsw_stats_.query_count.load());
            s.query_count = hnsw_stats_.query_count.load();
            stats.push_back(s);
        }
        
        if (ivf_pq_index_) {
            IndexStats s;
            s.type = IndexType::IVF_PQ;
            s.num_vectors = total_vectors_;
            s.memory_usage_bytes = estimate_ivf_pq_memory();
            s.measured_recall = ivf_stats_.measured_recall;
            s.avg_query_time_ms = ivf_stats_.total_time_ms / 
                                 std::max(1ULL, ivf_stats_.query_count.load());
            s.query_count = ivf_stats_.query_count.load();
            stats.push_back(s);
        }
        
        if (disk_graph_index_) {
            IndexStats s;
            s.type = IndexType::DiskANN;
            s.num_vectors = total_vectors_;
            s.memory_usage_bytes = estimate_diskann_memory();
            s.disk_usage_bytes = estimate_diskann_disk();
            s.measured_recall = diskann_stats_.measured_recall;
            s.avg_query_time_ms = diskann_stats_.total_time_ms / 
                                 std::max(1ULL, diskann_stats_.query_count.load());
            s.query_count = diskann_stats_.query_count.load();
            stats.push_back(s);
        }
        
        return stats;
    }
    
    auto get_active_indexes() const -> std::vector<IndexType> {
        std::shared_lock lock(mutex_);
        std::vector<IndexType> active;
        
        if (hnsw_index_) active.push_back(IndexType::HNSW);
        if (ivf_pq_index_) active.push_back(IndexType::IVF_PQ);
        if (disk_graph_index_) active.push_back(IndexType::DiskANN);
        
        return active;
    }
    
    auto optimize(bool force) -> std::expected<void, core::error> {
        std::unique_lock lock(mutex_);
        
        if (!needs_optimization_ && !force) {
            return {};
        }
        
        // Rebuild indexes if needed
        // TODO: Implement incremental optimization
        
        needs_optimization_ = false;
        return {};
    }
    
    auto save(const std::string& path) const -> std::expected<void, core::error> {
        // TODO: Implement save
        (void)path;
        return {};
    }
    
    auto load(const std::string& path) -> std::expected<void, core::error> {
        // TODO: Implement load
        (void)path;
        return {};
    }
    
    auto memory_usage() const -> std::size_t {
        std::shared_lock lock(mutex_);
        std::size_t total = 0;
        
        if (hnsw_index_) total += estimate_hnsw_memory();
        if (ivf_pq_index_) total += estimate_ivf_pq_memory();
        if (disk_graph_index_) total += estimate_diskann_memory();
        
        return total;
    }
    
    auto disk_usage() const -> std::size_t {
        std::shared_lock lock(mutex_);
        
        if (disk_graph_index_) {
            return estimate_diskann_disk();
        }
        
        return 0;
    }
    
    auto set_memory_budget(std::size_t budget_mb) -> std::expected<void, core::error> {
        std::unique_lock lock(mutex_);
        memory_budget_mb_ = budget_mb;
        
        // TODO: Evict or convert indexes if over budget
        
        return {};
    }
    
    void set_query_planner(std::shared_ptr<QueryPlanner> planner) {
        query_planner_ = planner;
    }
    
private:
    IndexType select_optimal_index(std::size_t n, const IndexBuildConfig& config) const {
        // Simple heuristic based on data size and memory budget
        const std::size_t memory_per_vector_hnsw = dimension_ * 4 + 64; // float + graph
        const std::size_t memory_per_vector_ivfpq = 32; // PQ codes
        const std::size_t memory_per_vector_diskann = 128; // cache + metadata
        
        const std::size_t hnsw_memory_mb = (n * memory_per_vector_hnsw) / (1024 * 1024);
        const std::size_t ivfpq_memory_mb = (n * memory_per_vector_ivfpq) / (1024 * 1024);
        
        // Small datasets: use HNSW if fits in memory
        if (n < 100000 && hnsw_memory_mb < config.memory_budget_mb) {
            return IndexType::HNSW;
        }
        
        // Medium datasets: use IVF-PQ for compression
        if (n < 10000000 && ivfpq_memory_mb < config.memory_budget_mb) {
            return IndexType::IVF_PQ;
        }
        
        // Large datasets: use DiskANN
        return IndexType::DiskANN;
    }
    
    auto build_index(IndexType type, const float* vectors, std::size_t n, 
                    const IndexBuildConfig& config)
        -> std::expected<void, core::error> {
        
        switch (type) {
            case IndexType::HNSW: {
                hnsw_index_ = std::make_unique<HnswIndex>();
                auto result = hnsw_index_->init(dimension_, config.hnsw_params, n);
                if (!result) return result;
                
                // Add vectors
                for (std::size_t i = 0; i < n; ++i) {
                    auto add_result = hnsw_index_->add(i, vectors + i * dimension_);
                    if (!add_result) return add_result;
                }
                break;
            }
            
            case IndexType::IVF_PQ: {
                ivf_pq_index_ = std::make_unique<IvfPqIndex>();
                auto result = ivf_pq_index_->train(vectors, dimension_, n, 
                                                   config.ivf_params, config.pq_params);
                if (!result) return result;
                
                // Add vectors
                std::vector<std::uint64_t> ids(n);
                std::iota(ids.begin(), ids.end(), 0);
                auto add_result = ivf_pq_index_->add_batch(ids.data(), vectors, n);
                if (!add_result) return add_result;
                break;
            }
            
            case IndexType::DiskANN: {
                disk_graph_index_ = std::make_unique<DiskGraphIndex>(dimension_);
                auto result = disk_graph_index_->build(vectors, n, config.vamana_params);
                if (!result) return result;
                break;
            }
        }
        
        return {};
    }
    
    std::size_t estimate_hnsw_memory() const {
        if (!hnsw_index_) return 0;
        // Estimate: vectors + graph structure
        return total_vectors_ * (dimension_ * sizeof(float) + 64);
    }
    
    std::size_t estimate_ivf_pq_memory() const {
        if (!ivf_pq_index_) return 0;
        // Estimate: PQ codes + centroids
        return total_vectors_ * 32 + 256 * dimension_ * sizeof(float);
    }
    
    std::size_t estimate_diskann_memory() const {
        if (!disk_graph_index_) return 0;
        // Estimate: cache + routing
        return build_config_.cache_size_mb * 1024 * 1024;
    }
    
    std::size_t estimate_diskann_disk() const {
        if (!disk_graph_index_) return 0;
        // Estimate: graph + PQ codes
        return total_vectors_ * (128 + 32);
    }
    
private:
    std::size_t dimension_;
    std::size_t memory_budget_mb_;
    std::size_t total_vectors_;
    bool needs_optimization_{false};
    
    IndexBuildConfig build_config_;
    
    // Index instances
    std::unique_ptr<HnswIndex> hnsw_index_;
    std::unique_ptr<IvfPqIndex> ivf_pq_index_;
    std::unique_ptr<DiskGraphIndex> disk_graph_index_;
    
    // Statistics
    mutable struct {
        std::atomic<std::uint64_t> query_count{0};
        float total_time_ms{0};
        float measured_recall{0};
    } hnsw_stats_, ivf_stats_, diskann_stats_;
    
    // Tombstones for removal
    // std::unordered_set<std::uint64_t> removed_ids_; // TODO: Implement
    
    // Query planner (optional)
    std::shared_ptr<QueryPlanner> query_planner_;
    
    // Thread safety
    mutable std::shared_mutex mutex_;
};

// IndexManager public interface implementation

IndexManager::IndexManager(std::size_t dimension)
    : impl_(std::make_unique<Impl>(dimension)) {}

IndexManager::~IndexManager() = default;

IndexManager::IndexManager(IndexManager&&) noexcept = default;

IndexManager& IndexManager::operator=(IndexManager&&) noexcept = default;

auto IndexManager::build(const float* vectors, std::size_t n, const IndexBuildConfig& config)
    -> std::expected<void, core::error> {
    return impl_->build(vectors, n, config);
}

auto IndexManager::add(std::uint64_t id, const float* vector)
    -> std::expected<void, core::error> {
    return impl_->add(id, vector);
}

auto IndexManager::add_batch(const std::uint64_t* ids, const float* vectors, std::size_t n)
    -> std::expected<void, core::error> {
    return impl_->add_batch(ids, vectors, n);
}

auto IndexManager::search(const float* query, const QueryConfig& config)
    -> std::expected<std::vector<std::pair<std::uint64_t, float>>, core::error> {
    return impl_->search(query, config);
}

auto IndexManager::search_batch(const float* queries, std::size_t nq, const QueryConfig& config)
    -> std::expected<std::vector<std::vector<std::pair<std::uint64_t, float>>>, core::error> {
    return impl_->search_batch(queries, nq, config);
}

auto IndexManager::remove(std::uint64_t id) -> std::expected<void, core::error> {
    return impl_->remove(id);
}

auto IndexManager::get_stats() const -> std::vector<IndexStats> {
    return impl_->get_stats();
}

auto IndexManager::get_active_indexes() const -> std::vector<IndexType> {
    return impl_->get_active_indexes();
}

auto IndexManager::optimize(bool force) -> std::expected<void, core::error> {
    return impl_->optimize(force);
}

auto IndexManager::save(const std::string& path) const -> std::expected<void, core::error> {
    return impl_->save(path);
}

auto IndexManager::load(const std::string& path) -> std::expected<void, core::error> {
    return impl_->load(path);
}

auto IndexManager::memory_usage() const -> std::size_t {
    return impl_->memory_usage();
}

auto IndexManager::disk_usage() const -> std::size_t {
    return impl_->disk_usage();
}

auto IndexManager::set_memory_budget(std::size_t budget_mb) -> std::expected<void, core::error> {
    return impl_->set_memory_budget(budget_mb);
}

} // namespace vesper::index