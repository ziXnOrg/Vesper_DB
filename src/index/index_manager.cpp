/** \file index_manager.cpp
 *  \brief Implementation of unified index management for multiple index types.
 */

#include "vesper/index/index_manager.hpp"
#include "vesper/index/hnsw.hpp"
#include "vesper/index/ivf_pq.hpp"
#include "vesper/index/disk_graph.hpp"
#include "vesper/tombstone/tombstone_manager.hpp"
#include "vesper/metadata/metadata_store.hpp"
#include "vesper/kernels/dispatch.hpp"
#include "incremental_repair.cpp"  // Include repair implementations
#include <expected>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <numeric>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <filesystem>
#include <fstream>

namespace vesper::index {

/** \brief Implementation class for IndexManager. */
class IndexManager::Impl {
public:
    explicit Impl(std::size_t dimension)
        : dimension_(dimension)
        , memory_budget_mb_(1024)
        , total_vectors_(0) {
        // Initialize tombstone manager
        tombstone::TombstoneConfig tombstone_config;
        tombstone_config.compaction_threshold = 10000;
        tombstone_config.compaction_ratio = 0.2;
        tombstone_config.use_compression = true;
        tombstone_manager_ = std::make_unique<tombstone::TombstoneManager>(tombstone_config);
        
        // Initialize metadata store
        metadata::MetadataIndexConfig metadata_config;
        metadata_config.enable_bitmap_index = true;
        metadata_config.enable_range_index = true;
        metadata_config.max_cardinality = 10000;
        metadata_config.cache_size_mb = 64;
        metadata_store_ = std::make_unique<metadata::MetadataStore>(metadata_config);
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
        std::vector<std::pair<std::uint64_t, float>> results;
        
        switch (selected_index) {
            case IndexType::HNSW:
                if (hnsw_index_) {
                    HnswSearchParams params;
                    params.k = config.k;
                    params.efSearch = config.ef_search;
                    auto raw_results = hnsw_index_->search(query, params);
                    if (!raw_results) {
                        return std::vesper_unexpected(raw_results.error());
                    }
                    results = std::move(*raw_results);
                }
                break;
                
            case IndexType::IVF_PQ:
                if (ivf_pq_index_) {
                    IvfPqSearchParams params;
                    params.k = config.k;
                    params.nprobe = config.nprobe > 0 ? config.nprobe : 8;  // Default to 8 if not specified
                    params.use_exact_rerank = false;  // Can be made configurable later
                    
                    auto raw_results = ivf_pq_index_->search(query, params);
                    if (!raw_results) {
                        return std::vesper_unexpected(raw_results.error());
                    }
                    results = std::move(*raw_results);
                }
                break;
                
            case IndexType::DiskANN:
                if (disk_graph_index_) {
                    VamanaSearchParams params;
                    params.k = config.k;
                    params.beam_width = config.l_search;
                    auto result = disk_graph_index_->search(std::span<const float>(query, dimension_), params);
                    if (!result) return std::vesper_unexpected(result.error());
                    // Convert from <float, uint32_t> to <uint64_t, float>
                    results.reserve(result->size());
                    for (const auto& [score, id] : *result) {
                        results.emplace_back(static_cast<std::uint64_t>(id), score);
                    } 
                }
                break;
        }
        
        if (results.empty() && !hnsw_index_ && !ivf_pq_index_ && !disk_graph_index_) {
            return std::vesper_unexpected(error{
                error_code::not_found,
                "No suitable index available",
                "index_manager"
            });
        }
        
        // Apply metadata filtering if specified
        if (config.filter.has_value() && metadata_store_) {
            auto filter_result = metadata_store_->evaluate_filter(*config.filter);
            if (filter_result) {
                // Filter results to only include IDs in the bitmap
                std::vector<std::pair<std::uint64_t, float>> filtered;
                filtered.reserve(results.size());
                
                for (const auto& [id, score] : results) {
                    if (filter_result->contains(static_cast<std::uint32_t>(id))) {
                        filtered.emplace_back(id, score);
                    }
                }
                
                results = std::move(filtered);
            }
        }
        
        // Filter out tombstoned results
        results = tombstone_manager_->filter_results(results);
        
        // If we need more results due to filtering, search again with larger k
        if (results.size() < config.k) {
            auto oversample_factor = tombstone_manager_->get_oversampling_factor();
            if (oversample_factor > 1.0f) {
                // Re-search with oversampling
                QueryConfig oversample_config = config;
                oversample_config.k = static_cast<std::uint32_t>(config.k * oversample_factor * 1.5f);
                
                // Recursively search with larger k (limit recursion depth)
                static thread_local int recursion_depth = 0;
                if (recursion_depth < 2) {
                    recursion_depth++;
                    auto oversample_result = search(query, oversample_config);
                    recursion_depth--;
                    
                    if (oversample_result && oversample_result->size() >= config.k) {
                        results = std::move(*oversample_result);
                    }
                }
            }
        }
        
        // Truncate to requested k
        if (results.size() > config.k) {
            results.resize(config.k);
        }
        
        return results;
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
    
    auto update(std::uint64_t id, const float* vector)
        -> std::expected<void, core::error> {
        
        using core::error;
        using core::error_code;
        
        if (!vector) {
            return std::vesper_unexpected(error{
                error_code::precondition_failed,
                "Invalid vector",
                "index_manager.update"
            });
        }
        
        std::unique_lock lock(mutex_);
        
        // Strategy: For most indexes, update = remove + add
        // This ensures consistency across all index types
        
        // Track if vector exists in any index
        bool exists_in_any = false;
        
        // Check HNSW
        if (hnsw_index_) {
            // HNSW doesn't have an exists check, so we'll try to update
            // For now, mark for optimization and do remove+add
            needs_optimization_ = true;
            exists_in_any = true;
        }
        
        // Check IVF-PQ
        if (ivf_pq_index_) {
            // IVF-PQ can handle updates via remove+add
            exists_in_any = true;
        }
        
        // Check DiskGraph
        if (disk_graph_index_) {
            // DiskGraph requires rebuild for updates
            needs_optimization_ = true;
            exists_in_any = true;
        }
        
        if (!exists_in_any) {
            return std::vesper_unexpected(error{
                error_code::not_found,
                "Vector ID not found in any index",
                "index_manager.update"
            });
        }
        
        // Store vector temporarily for atomic update
        std::vector<float> temp_vector(vector, vector + dimension_);
        
        // Phase 1: Mark old vector as deleted (soft delete)
        auto tombstone_result = tombstone_manager_->mark_deleted(static_cast<std::uint32_t>(id));
        if (!tombstone_result) {
            return std::vesper_unexpected(tombstone_result.error());
        }
        
        // Phase 2: Add new vector with same ID
        // This creates a new version while the old one is tombstoned
        
        if (hnsw_index_) {
            // HNSW: Add new version (old version is tombstoned)
            auto result = hnsw_index_->add(id, temp_vector.data());
            if (!result) {
                // Rollback tombstone on failure
                tombstone_manager_->unmark_deleted(static_cast<std::uint32_t>(id));
                return result;
            }
        }
        
        if (ivf_pq_index_) {
            // IVF-PQ: Add new version
            std::uint64_t ids[] = {id};
            auto result = ivf_pq_index_->add(&ids[0], temp_vector.data(), 1);
            if (!result) {
                // Rollback tombstone on failure
                tombstone_manager_->unmark_deleted(static_cast<std::uint32_t>(id));
                return result;
            }
        }
        
        if (disk_graph_index_) {
            // DiskGraph: Mark for rebuild with new vector
            // In production, we'd queue this update for batch processing
            needs_optimization_ = true;
        }
        
        // Phase 3: Remove tombstone after successful update
        tombstone_manager_->unmark_deleted(static_cast<std::uint32_t>(id));
        
        return {};
    }
    
    auto update_batch(const std::uint64_t* ids, const float* vectors, std::size_t n)
        -> std::expected<void, core::error> {
        
        using core::error;
        using core::error_code;
        
        if (!ids || !vectors || n == 0) {
            return std::vesper_unexpected(error{
                error_code::precondition_failed,
                "Invalid batch input",
                "index_manager.update_batch"
            });
        }
        
        std::unique_lock lock(mutex_);
        
        // Batch update strategy: mark all as deleted, then add new versions
        
        // Phase 1: Mark all old vectors as deleted
        std::vector<std::uint32_t> tombstoned_ids;
        tombstoned_ids.reserve(n);
        
        for (std::size_t i = 0; i < n; ++i) {
            auto result = tombstone_manager_->mark_deleted(static_cast<std::uint32_t>(ids[i]));
            if (!result) {
                // Rollback previous tombstones
                for (auto tid : tombstoned_ids) {
                    tombstone_manager_->unmark_deleted(tid);
                }
                return std::vesper_unexpected(result.error());
            }
            tombstoned_ids.push_back(static_cast<std::uint32_t>(ids[i]));
        }
        
        // Phase 2: Add new versions
        if (hnsw_index_) {
            for (std::size_t i = 0; i < n; ++i) {
                auto result = hnsw_index_->add(ids[i], vectors + i * dimension_);
                if (!result) {
                    // Rollback all tombstones
                    for (auto tid : tombstoned_ids) {
                        tombstone_manager_->unmark_deleted(tid);
                    }
                    return result;
                }
            }
        }
        
        if (ivf_pq_index_) {
            auto result = ivf_pq_index_->add(ids, vectors, n);
            if (!result) {
                // Rollback all tombstones
                for (auto tid : tombstoned_ids) {
                    tombstone_manager_->unmark_deleted(tid);
                }
                return result;
            }
        }
        
        if (disk_graph_index_) {
            needs_optimization_ = true;
        }
        
        // Phase 3: Remove tombstones after successful update
        for (auto tid : tombstoned_ids) {
            tombstone_manager_->unmark_deleted(tid);
        }
        
        return {};
    }
    
    auto remove(std::uint64_t id) -> std::expected<void, core::error> {
        std::unique_lock lock(mutex_);
        
        // Mark as deleted in tombstone manager
        auto result = tombstone_manager_->mark_deleted(static_cast<std::uint32_t>(id));
        if (!result) {
            return std::vesper_unexpected(result.error());
        }
        
        // Remove from all active indexes
        if (hnsw_index_) {
            // HNSW doesn't support removal, mark for rebuild
            needs_optimization_ = true;
        }
        
        if (ivf_pq_index_) {
            // IVF-PQ can use tombstones - it will check tombstone_manager during search
            // No need to physically remove, just mark as deleted
        }
        
        if (disk_graph_index_) {
            // DiskANN requires rebuild for physical removal
            needs_optimization_ = true;
        }
        
        // Check if compaction is needed
        if (tombstone_manager_->needs_compaction(total_vectors_)) {
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
        
        // Determine optimization strategy based on deletion ratio and index types
        auto deletion_ratio = tombstone_manager_->get_stats().deletion_ratio(total_vectors_);
        
        // Phase 1: Tombstone compaction if needed
        if (tombstone_manager_->needs_compaction(total_vectors_)) {
            auto compact_result = tombstone_manager_->compact();
            if (!compact_result) {
                return compact_result;
            }
        }
        
        // Phase 2: Index-specific optimization
        
        // Use incremental repair coordinator for all indexes
        auto repair_result = index::repair::IncrementalRepairCoordinator::repair_all(
            hnsw_index_.get(),
            ivf_pq_index_.get(),
            disk_graph_index_.get(),
            tombstone_manager_.get(),
            dimension_,
            force
        );
        
        if (!repair_result) {
            return repair_result;
        }
        
        // The repair coordinator handles index-specific optimizations above
        
        // Phase 3: Index conversion based on memory pressure
        if (!force && memory_budget_mb_ > 0) {
            auto current_mem = memory_usage() / (1024 * 1024);
            
            // If over budget, convert to more compact representations
            if (current_mem > memory_budget_mb_) {
                // Convert HNSW to IVF-PQ if possible
                if (hnsw_index_ && !ivf_pq_index_) {
                    // Extract vectors from HNSW for conversion
                    std::vector<float> extracted_vectors;
                    std::vector<std::uint64_t> extracted_ids;
                    extracted_vectors.reserve(total_vectors_ * dimension_);
                    extracted_ids.reserve(total_vectors_);
                    
                    // Extract all non-deleted vectors from HNSW
                    auto extract_result = hnsw_index_->extract_all_vectors(extracted_ids, extracted_vectors);
                    if (!extract_result) {
                        // Log extraction error but continue
                        return extract_result.error();
                    }
                    
                    if (!extracted_ids.empty()) {
                        // Train new IVF-PQ index
                        ivf_pq_index_ = std::make_unique<IvfPqIndex>();
                        auto train_result = ivf_pq_index_->train(
                            extracted_vectors.data(), 
                            dimension_, 
                            extracted_ids.size(),
                            build_config_.ivf_params
                        );
                        
                        if (train_result) {
                            // Add vectors to IVF-PQ
                            auto add_result = ivf_pq_index_->add(
                                extracted_ids.data(),
                                extracted_vectors.data(), 
                                extracted_ids.size()
                            );
                            
                            if (add_result) {
                                // Successfully converted, clear HNSW
                                hnsw_index_.reset();
                            } else {
                                // Conversion failed, keep HNSW and clear partial IVF-PQ
                                ivf_pq_index_.reset();
                            }
                        } else {
                            ivf_pq_index_.reset();
                        }
                    } else {
                        // No vectors to convert, just clear HNSW
                        hnsw_index_.reset();
                    }
                }
                
                // Convert IVF-PQ to DiskGraph if still over budget
                if (current_mem > memory_budget_mb_ && ivf_pq_index_ && !disk_graph_index_) {
                    // Extract vectors from IVF-PQ for conversion
                    std::vector<float> extracted_vectors;
                    std::vector<std::uint64_t> extracted_ids;
                    extracted_vectors.reserve(total_vectors_ * dimension_);
                    extracted_ids.reserve(total_vectors_);
                    
                    // Extract all non-deleted vectors from IVF-PQ
                    for (std::uint64_t id = 0; id < total_vectors_; ++id) {
                        if (!tombstone_manager_->is_deleted(static_cast<std::uint32_t>(id))) {
                            extracted_ids.push_back(id);
                            // In production, IVF-PQ would provide vector reconstruction
                            // For now, allocate space for reconstructed vectors
                            extracted_vectors.resize(extracted_vectors.size() + dimension_);
                        }
                    }
                    
                    if (!extracted_ids.empty()) {
                        // Build new DiskGraph index
                        disk_graph_index_ = std::make_unique<DiskGraphIndex>(dimension_);
                        auto build_result = disk_graph_index_->build(
                            std::span<const float>(extracted_vectors.data(), extracted_vectors.size()),
                            build_config_.vamana_params
                        );
                        
                        if (build_result) {
                            // Successfully converted, clear IVF-PQ
                            ivf_pq_index_.reset();
                        } else {
                            // Conversion failed, keep IVF-PQ and clear partial DiskGraph
                            disk_graph_index_.reset();
                            needs_optimization_ = true;
                        }
                    } else {
                        // No vectors to convert, just clear IVF-PQ
                        ivf_pq_index_.reset();
                    }
                }
            }
        }
        
        // Phase 4: Statistics update
        if (deletion_ratio < 0.05) {
            // Low deletion ratio, optimization complete
            needs_optimization_ = false;
        } else if (deletion_ratio > 0.3 && !force) {
            // High deletion ratio, schedule full rebuild
            needs_optimization_ = true;
        }
        
        return {};
    }
    
    auto save(const std::string& path) const -> std::expected<void, core::error> {
        namespace fs = std::filesystem;
        
        // Create directory if it doesn't exist
        std::error_code ec;
        fs::create_directories(path, ec);
        if (ec) {
            return std::vesper_unexpected(core::error{
                core::error_code::io_failed,
                "Failed to create directory: " + ec.message(),
                "index_manager.save"
            });
        }
        
        // Save manifest file
        std::string manifest_path = path + "/index_manager.manifest";
        std::ofstream manifest(manifest_path, std::ios::binary);
        if (!manifest) {
            return std::vesper_unexpected(core::error{
                core::error_code::io_failed,
                "Failed to create manifest file",
                "index_manager.save"
            });
        }
        
        // Write version header
        const std::uint32_t version = 1;
        manifest.write(reinterpret_cast<const char*>(&version), sizeof(version));
        
        // Write basic metadata
        manifest.write(reinterpret_cast<const char*>(&dimension_), sizeof(dimension_));
        manifest.write(reinterpret_cast<const char*>(&total_vectors_), sizeof(total_vectors_));
        manifest.write(reinterpret_cast<const char*>(&memory_budget_mb_), sizeof(memory_budget_mb_));
        manifest.write(reinterpret_cast<const char*>(&needs_optimization_), sizeof(needs_optimization_));
        
        // Write build configuration
        manifest.write(reinterpret_cast<const char*>(&build_config_.strategy), sizeof(build_config_.strategy));
        manifest.write(reinterpret_cast<const char*>(&build_config_.type), sizeof(build_config_.type));
        manifest.write(reinterpret_cast<const char*>(&build_config_.memory_budget_mb), sizeof(build_config_.memory_budget_mb));
        manifest.write(reinterpret_cast<const char*>(&build_config_.cache_size_mb), sizeof(build_config_.cache_size_mb));
        
        // Write HNSW params
        manifest.write(reinterpret_cast<const char*>(&build_config_.hnsw_params.M), sizeof(build_config_.hnsw_params.M));
        manifest.write(reinterpret_cast<const char*>(&build_config_.hnsw_params.efConstruction), sizeof(build_config_.hnsw_params.efConstruction));
        manifest.write(reinterpret_cast<const char*>(&build_config_.hnsw_params.seed), sizeof(build_config_.hnsw_params.seed));
        
        // Write IVF-PQ params
        manifest.write(reinterpret_cast<const char*>(&build_config_.ivf_params.nlist), sizeof(build_config_.ivf_params.nlist));
        manifest.write(reinterpret_cast<const char*>(&build_config_.ivf_params.m), sizeof(build_config_.ivf_params.m));
        manifest.write(reinterpret_cast<const char*>(&build_config_.ivf_params.nbits), sizeof(build_config_.ivf_params.nbits));
        
        // Write Vamana params
        manifest.write(reinterpret_cast<const char*>(&build_config_.vamana_params.degree), sizeof(build_config_.vamana_params.degree));
        manifest.write(reinterpret_cast<const char*>(&build_config_.vamana_params.alpha), sizeof(build_config_.vamana_params.alpha));
        manifest.write(reinterpret_cast<const char*>(&build_config_.vamana_params.L), sizeof(build_config_.vamana_params.L));
        
        // Write active index flags
        bool has_hnsw = (hnsw_index_ != nullptr);
        bool has_ivf_pq = (ivf_pq_index_ != nullptr);
        bool has_disk_graph = (disk_graph_index_ != nullptr);
        
        manifest.write(reinterpret_cast<const char*>(&has_hnsw), sizeof(has_hnsw));
        manifest.write(reinterpret_cast<const char*>(&has_ivf_pq), sizeof(has_ivf_pq));
        manifest.write(reinterpret_cast<const char*>(&has_disk_graph), sizeof(has_disk_graph));
        
        manifest.close();
        
        // Save individual indexes
        if (hnsw_index_) {
            auto result = hnsw_index_->save(path + "/hnsw");
            if (!result) return result;
        }
        
        if (ivf_pq_index_) {
            auto result = ivf_pq_index_->save(path + "/ivf_pq");
            if (!result) return result;
        }
        
        if (disk_graph_index_) {
            auto result = disk_graph_index_->save(path + "/disk_graph");
            if (!result) return result;
        }
        
        // Save tombstone manager
        if (tombstone_manager_) {
            auto result = tombstone_manager_->save(path + "/tombstones.bin");
            if (!result) return result;
        }
        
        // Save metadata store
        if (metadata_store_) {
            auto result = metadata_store_->save(path + "/metadata.bin");
            if (!result) return result;
        }
        
        return {};
    }
    
    auto load(const std::string& path) -> std::expected<void, core::error> {
        namespace fs = std::filesystem;
        
        // Check if directory exists
        if (!fs::exists(path)) {
            return std::unexpected(core::error{
                core::error_code::file_not_found,
                "Index directory not found: " + path,
                "index_manager.load"
            });
        }
        
        // Load manifest file
        std::string manifest_path = path + "/index_manager.manifest";
        std::ifstream manifest(manifest_path, std::ios::binary);
        if (!manifest) {
            return std::unexpected(core::error{
                core::error_code::file_not_found,
                "Manifest file not found: " + manifest_path,
                "index_manager.load"
            });
        }
        
        // Read and validate version
        std::uint32_t version = 0;
        manifest.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (version != 1) {
            return std::unexpected(core::error{
                core::error_code::version_mismatch,
                "Unsupported manifest version: " + std::to_string(version),
                "index_manager.load"
            });
        }
        
        // Read metadata
        manifest.read(reinterpret_cast<char*>(&metadata_.dimension), sizeof(metadata_.dimension));
        manifest.read(reinterpret_cast<char*>(&metadata_.metric), sizeof(metadata_.metric));
        manifest.read(reinterpret_cast<char*>(&metadata_.num_vectors), sizeof(metadata_.num_vectors));
        manifest.read(reinterpret_cast<char*>(&metadata_.build_time), sizeof(metadata_.build_time));
        manifest.read(reinterpret_cast<char*>(&metadata_.index_flags), sizeof(metadata_.index_flags));
        
        // Read build config
        manifest.read(reinterpret_cast<char*>(&build_config_.hnsw_config.M), sizeof(build_config_.hnsw_config.M));
        manifest.read(reinterpret_cast<char*>(&build_config_.hnsw_config.ef_construction), sizeof(build_config_.hnsw_config.ef_construction));
        manifest.read(reinterpret_cast<char*>(&build_config_.hnsw_config.max_M), sizeof(build_config_.hnsw_config.max_M));
        manifest.read(reinterpret_cast<char*>(&build_config_.hnsw_config.max_M0), sizeof(build_config_.hnsw_config.max_M0));
        manifest.read(reinterpret_cast<char*>(&build_config_.hnsw_config.ml), sizeof(build_config_.hnsw_config.ml));
        manifest.read(reinterpret_cast<char*>(&build_config_.hnsw_config.seed), sizeof(build_config_.hnsw_config.seed));
        manifest.read(reinterpret_cast<char*>(&build_config_.hnsw_config.extend_candidates), sizeof(build_config_.hnsw_config.extend_candidates));
        manifest.read(reinterpret_cast<char*>(&build_config_.hnsw_config.keep_pruned_connections), sizeof(build_config_.hnsw_config.keep_pruned_connections));
        
        manifest.read(reinterpret_cast<char*>(&build_config_.ivfpq_config.nlist), sizeof(build_config_.ivfpq_config.nlist));
        manifest.read(reinterpret_cast<char*>(&build_config_.ivfpq_config.subvector_count), sizeof(build_config_.ivfpq_config.subvector_count));
        manifest.read(reinterpret_cast<char*>(&build_config_.ivfpq_config.pq_bits), sizeof(build_config_.ivfpq_config.pq_bits));
        manifest.read(reinterpret_cast<char*>(&build_config_.ivfpq_config.centroid_sample_rate), sizeof(build_config_.ivfpq_config.centroid_sample_rate));
        manifest.read(reinterpret_cast<char*>(&build_config_.ivfpq_config.encode_residual), sizeof(build_config_.ivfpq_config.encode_residual));
        manifest.read(reinterpret_cast<char*>(&build_config_.ivfpq_config.use_opq), sizeof(build_config_.ivfpq_config.use_opq));
        manifest.read(reinterpret_cast<char*>(&build_config_.ivfpq_config.balanced_clustering), sizeof(build_config_.ivfpq_config.balanced_clustering));
        
        manifest.read(reinterpret_cast<char*>(&build_config_.disk_config.max_degree), sizeof(build_config_.disk_config.max_degree));
        manifest.read(reinterpret_cast<char*>(&build_config_.disk_config.search_list_size), sizeof(build_config_.disk_config.search_list_size));
        manifest.read(reinterpret_cast<char*>(&build_config_.disk_config.prune_threshold), sizeof(build_config_.disk_config.prune_threshold));
        manifest.read(reinterpret_cast<char*>(&build_config_.disk_config.max_occlusion_size), sizeof(build_config_.disk_config.max_occlusion_size));
        manifest.read(reinterpret_cast<char*>(&build_config_.disk_config.num_pq_chunks), sizeof(build_config_.disk_config.num_pq_chunks));
        manifest.read(reinterpret_cast<char*>(&build_config_.disk_config.use_compression), sizeof(build_config_.disk_config.use_compression));
        manifest.read(reinterpret_cast<char*>(&build_config_.disk_config.use_opq), sizeof(build_config_.disk_config.use_opq));
        
        manifest.read(reinterpret_cast<char*>(&build_config_.memory_budget_gb), sizeof(build_config_.memory_budget_gb));
        manifest.read(reinterpret_cast<char*>(&build_config_.num_build_threads), sizeof(build_config_.num_build_threads));
        
        // Read index flags
        bool has_hnsw = false, has_ivfpq = false, has_disk = false, has_tombstone = false;
        manifest.read(reinterpret_cast<char*>(&has_hnsw), sizeof(has_hnsw));
        manifest.read(reinterpret_cast<char*>(&has_ivfpq), sizeof(has_ivfpq));
        manifest.read(reinterpret_cast<char*>(&has_disk), sizeof(has_disk));
        manifest.read(reinterpret_cast<char*>(&has_tombstone), sizeof(has_tombstone));
        
        if (!manifest.good()) {
            return std::unexpected(core::error{
                core::error_code::io_failed,
                "Failed to read manifest file",
                "index_manager.load"
            });
        }
        
        manifest.close();
        
        // Load individual indexes
        if (has_hnsw) {
            hnsw_index_ = std::make_unique<index::HNSW>(
                metadata_.dimension,
                metadata_.metric,
                build_config_.hnsw_config
            );
            auto result = hnsw_index_->load(path + "/hnsw");
            if (!result) {
                return std::unexpected(core::error{
                    core::error_code::io_failed,
                    "Failed to load HNSW index: " + result.error().message,
                    "index_manager.load"
                });
            }
        }
        
        if (has_ivfpq) {
            ivf_pq_index_ = std::make_unique<index::IVF_PQ>(
                metadata_.dimension,
                metadata_.metric,
                build_config_.ivfpq_config
            );
            auto result = ivf_pq_index_->load(path + "/ivfpq");
            if (!result) {
                return std::unexpected(core::error{
                    core::error_code::io_failed,
                    "Failed to load IVF-PQ index: " + result.error().message,
                    "index_manager.load"
                });
            }
        }
        
        if (has_disk) {
            disk_graph_index_ = std::make_unique<index::DiskGraph>(
                metadata_.dimension,
                metadata_.metric,
                build_config_.disk_config
            );
            auto result = disk_graph_index_->load(path + "/disk_graph");
            if (!result) {
                return std::unexpected(core::error{
                    core::error_code::io_failed,
                    "Failed to load DiskGraph index: " + result.error().message,
                    "index_manager.load"
                });
            }
        }
        
        // Load tombstone manager if present
        if (has_tombstone) {
            tombstone_manager_ = std::make_shared<tombstone::TombstoneManager>();
            auto result = tombstone_manager_->load(path + "/tombstones.bin");
            if (!result) {
                return std::unexpected(core::error{
                    core::error_code::io_failed,
                    "Failed to load tombstone data: " + result.error().message,
                    "index_manager.load"
                });
            }
        }
        
        // Load metadata store if present
        if (fs::exists(path + "/metadata.bin")) {
            if (!metadata_store_) {
                metadata::MetadataIndexConfig metadata_config;
                metadata_store_ = std::make_unique<metadata::MetadataStore>(metadata_config);
            }
            auto result = metadata_store_->load(path + "/metadata.bin");
            if (!result) {
                return std::unexpected(core::error{
                    core::error_code::io_failed,
                    "Failed to load metadata: " + result.error().message,
                    "index_manager.load"
                });
            }
        }
        
        // Update index state
        is_built_ = (has_hnsw || has_ivfpq || has_disk);
        
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
        
        // Check current memory usage and enforce budget
        auto result = enforce_memory_budget();
        if (!result) {
            return result;
        }
        
        return {};
    }
    
    auto set_metadata(std::uint64_t id, 
                     const std::unordered_map<std::string, metadata::MetadataValue>& metadata)
        -> std::expected<void, core::error> {
        if (!metadata_store_) {
            return std::unexpected(core::error{
                core::error_code::not_initialized,
                "Metadata store not initialized",
                "index_manager.set_metadata"
            });
        }
        
        metadata::DocumentMetadata doc;
        doc.id = id;
        doc.attributes = metadata;
        
        // Check if document exists to decide between add and update
        if (metadata_store_->exists(id)) {
            return metadata_store_->update(doc);
        } else {
            return metadata_store_->add(doc);
        }
    }
    
    auto get_metadata(std::uint64_t id) const
        -> std::expected<std::unordered_map<std::string, metadata::MetadataValue>, core::error> {
        if (!metadata_store_) {
            return std::unexpected(core::error{
                core::error_code::not_initialized,
                "Metadata store not initialized",
                "index_manager.get_metadata"
            });
        }
        
        auto result = metadata_store_->get(id);
        if (!result) {
            return std::unexpected(result.error());
        }
        
        return result->attributes;
    }
    
    auto remove_metadata(std::uint64_t id) -> std::expected<void, core::error> {
        if (!metadata_store_) {
            return std::unexpected(core::error{
                core::error_code::not_initialized,
                "Metadata store not initialized",
                "index_manager.remove_metadata"
            });
        }
        
        return metadata_store_->remove(id);
    }
    
    void set_query_planner(std::shared_ptr<QueryPlanner> planner) {
        query_planner_ = planner;
    }
    
    auto enforce_memory_budget() -> std::expected<void, core::error> {
        // Calculate current memory usage
        std::size_t current_memory_mb = memory_usage() / (1024 * 1024);
        
        if (current_memory_mb <= memory_budget_mb_) {
            return {};  // Within budget
        }
        
        // Need to evict indexes to meet budget
        std::size_t excess_mb = current_memory_mb - memory_budget_mb_;
        
        // Priority for eviction: HNSW (largest) -> IVF-PQ -> DiskGraph (already disk-based)
        // Also consider query statistics to keep frequently used indexes
        
        struct IndexPriority {
            IndexType type;
            std::size_t memory_mb;
            float usage_score;  // Based on query count and recall
        };
        
        std::vector<IndexPriority> priorities;
        
        if (hnsw_index_) {
            priorities.push_back({
                IndexType::HNSW,
                estimate_hnsw_memory() / (1024 * 1024),
                static_cast<float>(hnsw_stats_.query_count.load()) * hnsw_stats_.measured_recall
            });
        }
        
        if (ivf_pq_index_) {
            priorities.push_back({
                IndexType::IVF_PQ,
                estimate_ivf_pq_memory() / (1024 * 1024),
                static_cast<float>(ivf_stats_.query_count.load()) * ivf_stats_.measured_recall
            });
        }
        
        // Sort by usage_score (ascending, so least used is first)
        std::sort(priorities.begin(), priorities.end(), 
                 [](const auto& a, const auto& b) {
                     return a.usage_score < b.usage_score;
                 });
        
        // Evict indexes until we're under budget
        for (const auto& priority : priorities) {
            if (excess_mb <= 0) break;
            
            switch (priority.type) {
                case IndexType::HNSW:
                    // Convert HNSW to IVF-PQ (more compact)
                    if (!ivf_pq_index_ && total_vectors_ > 0) {
                        // Save HNSW data temporarily
                        std::string temp_path = "/tmp/vesper_hnsw_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
                        auto save_result = hnsw_index_->save(temp_path);
                        if (!save_result) {
                            return save_result;
                        }
                        
                        // Clear HNSW to free memory
                        hnsw_index_.reset();
                        excess_mb -= priority.memory_mb;
                        
                        // Attempt conversion to IVF-PQ
                        // Save vectors for conversion (would be extracted from HNSW in production)
                        std::string vectors_path = temp_path + ".vectors";
                        // In production: extract_vectors_from_hnsw(hnsw_index_.get(), vectors_path);
                        
                        // Mark for deferred conversion
                        needs_optimization_ = true;
                    } else {
                        // Just evict HNSW
                        hnsw_index_.reset();
                        excess_mb -= priority.memory_mb;
                    }
                    break;
                    
                case IndexType::IVF_PQ:
                    // Move IVF-PQ to disk
                    if (!disk_graph_index_ && total_vectors_ > 0) {
                        // Save IVF-PQ data
                        std::string temp_path = "/tmp/vesper_ivfpq_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
                        auto save_result = ivf_pq_index_->save(temp_path);
                        if (!save_result) {
                            return save_result;
                        }
                        
                        // Clear IVF-PQ
                        ivf_pq_index_.reset();
                        excess_mb -= priority.memory_mb;
                        
                        // Mark for conversion to DiskGraph
                        needs_optimization_ = true;
                    } else {
                        // Just evict IVF-PQ
                        ivf_pq_index_.reset();
                        excess_mb -= priority.memory_mb;
                    }
                    break;
                    
                case IndexType::DiskANN:
                    // DiskGraph uses minimal memory, just reduce cache if possible
                    if (disk_graph_index_ && build_config_.cache_size_mb > 64) {
                        // Reduce cache size
                        std::size_t old_cache = build_config_.cache_size_mb;
                        build_config_.cache_size_mb = std::max(64UL, build_config_.cache_size_mb / 2);
                        excess_mb -= (old_cache - build_config_.cache_size_mb);
                        
                        // Apply new cache size to disk_graph_index_
                        if (disk_graph_index_) {
                            // DiskGraph would have a method to update cache size
                            // disk_graph_index_->set_cache_size_mb(build_config_.cache_size_mb);
                        }
                    }
                    break;
            }
        }
        
        // Final check
        current_memory_mb = memory_usage() / (1024 * 1024);
        if (current_memory_mb > memory_budget_mb_) {
            return std::unexpected(core::error{
                core::error_code::out_of_memory,
                "Unable to meet memory budget after eviction. Current: " + 
                std::to_string(current_memory_mb) + "MB, Budget: " + 
                std::to_string(memory_budget_mb_) + "MB",
                "index_manager.enforce_memory_budget"
            });
        }
        
        return {};
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
                                                   config.ivf_params);
                if (!result) {
                    return std::vesper_unexpected(result.error());
                }
                
                // Add vectors
                std::vector<std::uint64_t> ids(n);
                std::iota(ids.begin(), ids.end(), 0);
                auto add_result = ivf_pq_index_->add(ids.data(), vectors, n);
                if (!add_result) return add_result;
                break;
            }
            
            case IndexType::DiskANN: {
                disk_graph_index_ = std::make_unique<DiskGraphIndex>(dimension_);
                auto result = disk_graph_index_->build(
                    std::span<const float>(vectors, n * dimension_), 
                    config.vamana_params);
                if (!result) {
                    return std::vesper_unexpected(result.error());
                }
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
    
    // Tombstone manager for soft deletion
    std::unique_ptr<tombstone::TombstoneManager> tombstone_manager_;
    
    // Metadata store for filtering
    std::unique_ptr<metadata::MetadataStore> metadata_store_;
    
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

auto IndexManager::update(std::uint64_t id, const float* vector)
    -> std::expected<void, core::error> {
    return impl_->update(id, vector);
}

auto IndexManager::update_batch(const std::uint64_t* ids, const float* vectors, std::size_t n)
    -> std::expected<void, core::error> {
    return impl_->update_batch(ids, vectors, n);
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

auto IndexManager::set_metadata(std::uint64_t id, 
                                const std::unordered_map<std::string, metadata::MetadataValue>& metadata)
    -> std::expected<void, core::error> {
    return impl_->set_metadata(id, metadata);
}

auto IndexManager::get_metadata(std::uint64_t id) const
    -> std::expected<std::unordered_map<std::string, metadata::MetadataValue>, core::error> {
    return impl_->get_metadata(id);
}

auto IndexManager::remove_metadata(std::uint64_t id) -> std::expected<void, core::error> {
    return impl_->remove_metadata(id);
}

} // namespace vesper::index
