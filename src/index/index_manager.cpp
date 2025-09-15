/** \file index_manager.cpp
 *  \brief Implementation of unified index management for multiple index types.
 */

#include "vesper/index/index_manager.hpp"
#include "vesper/index/hnsw.hpp"
#include "vesper/index/ivf_pq.hpp"
#include "vesper/index/disk_graph.hpp"
#include "vesper/index/rabitq_quantizer.hpp"
#include "vesper/index/matryoshka.hpp"
#include "vesper/tombstone/tombstone_manager.hpp"
#include "vesper/metadata/metadata_store.hpp"
#include "vesper/kernels/dispatch.hpp"
#include "vesper/core/platform_utils.hpp"

#include <expected>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <numeric>
#include <limits>

#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <filesystem>
#include <fstream>
#include <cstdio>


#include <cstring>

#include <execution>
#if defined(__AVX2__) || defined(_M_AVX2)
#include <immintrin.h>
#endif

namespace vesper::index {
namespace {
inline float l2_sqr_scalar(const float* a, const float* b, int d) {
    float sum = 0.0f;
    for (int i = 0; i < d; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

#if defined(__AVX2__) || defined(_M_AVX2)
inline float l2_sqr_avx2(const float* a, const float* b, int d) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= d; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        __m256 mul = _mm256_mul_ps(diff, diff);
        acc = _mm256_add_ps(acc, mul);
    }
    // Horizontal sum of acc's 8 lanes
    __m256 t1 = _mm256_hadd_ps(acc, acc);
    __m256 t2 = _mm256_hadd_ps(t1, t1);
    __m128 low = _mm256_castps256_ps128(t2);
    __m128 high = _mm256_extractf128_ps(t2, 1);
    __m128 sum128 = _mm_add_ps(low, high);
    float sum = _mm_cvtss_f32(sum128);
    for (; i < d; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}
#endif

inline float l2_sqr_fast(const float* a, const float* b, int d) {
#if defined(__AVX2__) || defined(_M_AVX2)
    return l2_sqr_avx2(a, b, d);
#else
    return l2_sqr_scalar(a, b, d);
#endif
}

// Environment override helpers for rerank controls
inline std::optional<std::string> getenv_nonempty(const char* key) noexcept {
    auto v = vesper::core::safe_getenv(key);
    if (v && !v->empty()) return v;
    return std::nullopt;
}

inline bool parse_bool_ci(std::string_view s) noexcept {
    auto eq_ci = [](char a, char b){
        if (a >= 'A' && a <= 'Z') a = static_cast<char>(a - 'A' + 'a');
        if (b >= 'A' && b <= 'Z') b = static_cast<char>(b - 'A' + 'a');
        return a == b;
    };
    if (s.size() == 1 && (s[0] == '1' || s[0] == '0')) return s[0] == '1';
    if (s.size() == 4 && eq_ci(s[0],'t') && eq_ci(s[1],'r') && eq_ci(s[2],'u') && eq_ci(s[3],'e')) return true;
    if (s.size() == 5 && eq_ci(s[0],'f') && eq_ci(s[1],'a') && eq_ci(s[2],'l') && eq_ci(s[3],'s') && eq_ci(s[4],'e')) return false;
    return false;
}

inline void apply_rerank_env_overrides(bool& use_exact, std::uint32_t& rerank_k,
                                       float& alpha, std::uint32_t& ceil_val) noexcept {
    if (auto v = getenv_nonempty("VESPER_USE_EXACT_RERANK")) {
        use_exact = parse_bool_ci(*v);
    }
    if (auto v = getenv_nonempty("VESPER_RERANK_K")) {
        try {
            unsigned long long x = std::stoull(*v);
            rerank_k = static_cast<std::uint32_t>(x);
        } catch (...) { /* ignore */ }
    }
    if (auto v = getenv_nonempty("VESPER_RERANK_ALPHA")) {
        try {
            alpha = std::stof(*v);
        } catch (...) { /* ignore */ }
    }
    if (auto v = getenv_nonempty("VESPER_RERANK_CEIL")) {
        try {
            unsigned long long x = std::stoull(*v);
            ceil_val = static_cast<std::uint32_t>(x);
        } catch (...) { /* ignore */ }
    }
    // Clamp/validate
    if (!(alpha > 0.0f) || !std::isfinite(alpha)) alpha = 2.0f;
}

} // anonymous namespace


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

        // Initialize quantizers (will be configured during build)
        rabitq_quantizer_ = std::make_unique<RaBitQuantizer>();
        matryoshka_embedding_ = std::make_unique<MatryoshkaEmbedding>();
    }

    ~Impl() = default;

    // Accessors
    auto get_memory_budget() const -> std::size_t { return memory_budget_mb_; }

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
            // DiskGraph handles additions through batch rebuilds
            // Queue the vector for batch processing
            if (!pending_diskgraph_additions_) {
                pending_diskgraph_additions_ = std::make_unique<std::vector<std::pair<std::uint64_t, std::vector<float>>>>();
            }
            pending_diskgraph_additions_->emplace_back(id, std::vector<float>(vector, vector + dimension_));

            // Trigger rebuild when batch is large enough
            if (pending_diskgraph_additions_->size() >= 1000) {
                apply_diskgraph_batch();
            }
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
            // Queue vectors for batch rebuild
            if (!pending_diskgraph_additions_) {
                pending_diskgraph_additions_ = std::make_unique<std::vector<std::pair<std::uint64_t, std::vector<float>>>>();
            }
            for (std::size_t i = 0; i < n; ++i) {
                pending_diskgraph_additions_->emplace_back(
                    ids[i],
                    std::vector<float>(vectors + i * dimension_, vectors + (i + 1) * dimension_)
                );
            }

            // Apply batch when large enough
            if (pending_diskgraph_additions_->size() >= 5000) {
                apply_diskgraph_batch();

            }
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


        // Apply environment overrides for rerank controls (per-call, optional)
        bool cfg_use_exact = config.use_exact_rerank;
        std::uint32_t cfg_rerank_k = config.rerank_k;
        float cfg_rerank_alpha = config.rerank_alpha;
        std::uint32_t cfg_rerank_ceil = config.rerank_cand_ceiling;
        apply_rerank_env_overrides(cfg_use_exact, cfg_rerank_k, cfg_rerank_alpha, cfg_rerank_ceil);

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
                    // Request larger candidate pool from IVF-PQ when reranking is desired
                    std::uint32_t cand_k = 0;
                    if (cfg_use_exact) {
                        if (cfg_rerank_k > 0) {
                            cand_k = cfg_rerank_k;
                        } else {
                            const double base = static_cast<double>(config.k);
                            const double probe = static_cast<double>(params.nprobe);
                            const double heuristic = std::max(base, static_cast<double>(cfg_rerank_alpha) * base * std::log2(1.0 + probe));
                            cand_k = static_cast<std::uint32_t>(std::ceil(heuristic));
                            if (cfg_rerank_ceil > 0) {
                                cand_k = std::min<std::uint32_t>(cand_k, cfg_rerank_ceil);
                            }
                        }
                    }
                    params.cand_k = cand_k; // 0 => default to k
                    params.use_exact_rerank = cfg_use_exact;
                    params.rerank_k = (cfg_rerank_k > 0 ? cfg_rerank_k : cand_k);

                    auto raw_results = ivf_pq_index_->search(query, params);
                    if (!raw_results) {
                        return std::vesper_unexpected(raw_results.error());
                    }

                    // Optional exact rerank using raw vectors if available (e.g., via HNSW store)
                    if (cfg_use_exact && hnsw_index_) {
                        const std::uint32_t desired = (params.rerank_k > 0) ? params.rerank_k : ((params.cand_k > 0) ? params.cand_k : config.k);
                        const auto shortlist = std::min<std::uint32_t>(
                            desired,
                            static_cast<std::uint32_t>(raw_results->size())
                        );
                        if (shortlist > 0) {
#ifdef VESPER_ENABLE_TIMING
                            const auto t_start = std::chrono::high_resolution_clock::now();
#endif
                            // Gather shortlist IDs
                            std::vector<std::uint64_t> ids;
                            ids.reserve(shortlist);
                            for (std::size_t i = 0; i < shortlist; ++i) {
                                ids.push_back((*raw_results)[i].first);
                            }
#ifdef VESPER_ENABLE_TIMING
                            const auto t_after_ids = std::chrono::high_resolution_clock::now();
#endif
                            // Compute exact distances in parallel
                            std::vector<float> distances(shortlist, std::numeric_limits<float>::infinity());
                            std::vector<std::size_t> idxs(shortlist);
                            std::iota(idxs.begin(), idxs.end(), 0);
                            const int d = static_cast<int>(dimension_);
                            std::for_each(std::execution::par, idxs.begin(), idxs.end(), [&](std::size_t idx){
                                auto vec = hnsw_index_->get_vector(ids[idx]);
                                if (vec) {
                                    distances[idx] = l2_sqr_fast(query, vec->data(), d);
                                }
                            });
#ifdef VESPER_ENABLE_TIMING
                            const auto t_after_dist = std::chrono::high_resolution_clock::now();
#endif
                            // Build merged candidate set: exact reranked head + untouched tail
                            std::vector<std::pair<std::uint64_t, float>> merged;
                            merged.reserve(raw_results->size());
                            for (std::size_t i = 0; i < shortlist; ++i) {
                                merged.emplace_back(ids[i], distances[i]);
                            }
                            for (std::size_t i = shortlist; i < raw_results->size(); ++i) {
                                merged.push_back((*raw_results)[i]);
                            }

                            // Select final top-k by distance
                            const std::size_t topk = std::min<std::size_t>(config.k, merged.size());
                            std::partial_sort(merged.begin(), merged.begin() + topk, merged.end(), [](const auto& a, const auto& b){ return a.second < b.second; });
#ifdef VESPER_ENABLE_TIMING
                            const auto t_after_select = std::chrono::high_resolution_clock::now();
                            const auto ids_us = std::chrono::duration_cast<std::chrono::microseconds>(t_after_ids - t_start).count();
                            const auto dist_us = std::chrono::duration_cast<std::chrono::microseconds>(t_after_dist - t_after_ids).count();
                            const auto select_us = std::chrono::duration_cast<std::chrono::microseconds>(t_after_select - t_after_dist).count();
                            std::printf("[vesper][rerank] gather_us=%lld dist_us=%lld select_us=%lld shortlist=%u k=%u nprobe=%u\n",
                                        static_cast<long long>(ids_us),
                                        static_cast<long long>(dist_us),
                                        static_cast<long long>(select_us),
                                        static_cast<unsigned>(shortlist),
                                        static_cast<unsigned>(config.k),
                                        static_cast<unsigned>(params.nprobe));
#endif
                            results.assign(merged.begin(), merged.begin() + topk);
                            return results;
                        }
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
            // HNSW update via remove and re-add
            auto existing_vec = hnsw_index_->get_vector(id);
            if (existing_vec) {
                exists_in_any = true;
                // Mark as deleted, then add the new version
                auto mark_result = hnsw_index_->mark_deleted(id);
                if (mark_result) {
                    auto add_result = hnsw_index_->add(id, vector);
                    if (!add_result) {
                        return add_result;
                    }
                }
            }
        }

        // Check IVF-PQ
        if (ivf_pq_index_) {
            // IVF-PQ can handle updates via remove+add
            exists_in_any = true;
        }

        // Check DiskGraph
        if (disk_graph_index_) {
            // Check if vector exists in DiskGraph
            auto existing_vec = disk_graph_index_->get_vector(id);
            if (existing_vec) {
                exists_in_any = true;
            }
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
                tombstone_manager_->restore(static_cast<std::uint32_t>(id));
                return result;
            }
        }

        if (ivf_pq_index_) {
            // IVF-PQ: Add new version
            std::uint64_t ids[] = {id};
            auto result = ivf_pq_index_->add(&ids[0], temp_vector.data(), 1);
            if (!result) {
                // Rollback tombstone on failure
                tombstone_manager_->restore(static_cast<std::uint32_t>(id));
                return result;
            }
        }

        if (disk_graph_index_) {
            // DiskGraph: Queue update for batch processing
            if (!pending_diskgraph_additions_) {
                pending_diskgraph_additions_ = std::make_unique<std::vector<std::pair<std::uint64_t, std::vector<float>>>>();
            }
            // Remove old version from pending if exists
            auto it = std::remove_if(pending_diskgraph_additions_->begin(),
                                    pending_diskgraph_additions_->end(),
                                    [id](const auto& p) { return p.first == id; });
            pending_diskgraph_additions_->erase(it, pending_diskgraph_additions_->end());

            // Add updated version
            pending_diskgraph_additions_->emplace_back(id, std::vector<float>(vector, vector + dimension_));

            // Apply batch if large enough
            if (pending_diskgraph_additions_->size() >= 1000) {
                apply_diskgraph_batch();
            }
        }

        // Phase 3: Remove tombstone after successful update
        tombstone_manager_->restore(static_cast<std::uint32_t>(id));

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
                    tombstone_manager_->restore(tid);
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
                        tombstone_manager_->restore(tid);
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
                    tombstone_manager_->restore(tid);
                }
                return result;
            }
        }

        if (disk_graph_index_) {
            // DiskGraph doesn't support direct deletion
            // Mark in tombstones and schedule rebuild if needed
            if (tombstone_manager_->needs_compaction(total_vectors_)) {
                // Queue rebuild with non-deleted vectors
                if (!pending_diskgraph_additions_) {
                    pending_diskgraph_additions_ = std::make_unique<std::vector<std::pair<std::uint64_t, std::vector<float>>>>();
                }
                // Flag for rebuild at next opportunity
                needs_optimization_ = true;
            }
        }

        // Phase 3: Remove tombstones after successful update
        for (auto tid : tombstoned_ids) {
            tombstone_manager_->restore(tid);
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

    auto apply_diskgraph_batch() -> std::expected<void, core::error> {
        if (!disk_graph_index_ || !pending_diskgraph_additions_ || pending_diskgraph_additions_->empty()) {
            return {};
        }

        // Rebuild DiskGraph with all vectors including new additions
        std::vector<float> all_vectors;
        std::vector<std::uint64_t> all_ids;

        // Get existing vectors from DiskGraph
        auto num_existing = disk_graph_index_->size();
        all_vectors.reserve((num_existing + pending_diskgraph_additions_->size()) * dimension_);
        all_ids.reserve(num_existing + pending_diskgraph_additions_->size());

        // Extract existing vectors
        for (std::uint64_t id = 0; id < num_existing; ++id) {
            if (!tombstone_manager_->is_deleted(static_cast<std::uint32_t>(id))) {
                auto vec_result = disk_graph_index_->get_vector(id);
                if (vec_result) {
                    all_ids.push_back(id);
                    all_vectors.insert(all_vectors.end(), vec_result.value().begin(), vec_result.value().end());
                }
            }
        }

        // Add pending vectors
        for (const auto& [id, vec] : *pending_diskgraph_additions_) {
            all_ids.push_back(id);
            all_vectors.insert(all_vectors.end(), vec.begin(), vec.end());
        }

        // Rebuild the index
        VamanaBuildParams params = build_config_.vamana_params;
        auto new_index = std::make_unique<DiskGraphIndex>(dimension_);
        auto build_result = new_index->build(
            std::span<const float>(all_vectors.data(), all_vectors.size()),
            params
        );

        if (!build_result) {
            return std::vesper_unexpected(build_result.error());
        }

        // Swap in the new index
        disk_graph_index_ = std::move(new_index);
        pending_diskgraph_additions_->clear();

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

        // TODO: Use incremental repair coordinator for all indexes
        // auto repair_result = index::repair::IncrementalRepairCoordinator::repair_all(
        //     hnsw_index_.get(),
        //     ivf_pq_index_.get(),
        //     disk_graph_index_.get(),
        //     tombstone_manager_.get(),
        //     dimension_,
        //     force
        // );
        //
        // if (!repair_result) {
        //     return repair_result;
        // }

        // For now, just return success
        return {};

        // The repair coordinator handles index-specific optimizations above

        // Phase 3: Index conversion based on memory pressure
        if (!force && memory_budget_mb_ > 0) {
            auto current_mem = memory_usage_internal() / (1024 * 1024);

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
                    // TODO: Implement extract_all_vectors for HNSW
                    // For now, iterate through vectors manually
                    for (std::uint64_t id = 0; id < total_vectors_; ++id) {
                        if (!tombstone_manager_->is_deleted(static_cast<std::uint32_t>(id))) {
                            auto vec_result = hnsw_index_->get_vector(id);
                            if (vec_result) {
                                extracted_ids.push_back(id);
                                const auto& vec = vec_result.value();
                                extracted_vectors.insert(extracted_vectors.end(), vec.begin(), vec.end());
                            }
                        }
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
                            // Reconstruct vector from IVF-PQ
                            auto vec_result = ivf_pq_index_->get_vector(id);
                            if (vec_result) {
                                extracted_ids.push_back(id);
                                const auto& vec = vec_result.value();
                                extracted_vectors.insert(extracted_vectors.end(), vec.begin(), vec.end());
                            }
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
            return std::vesper_unexpected(core::error{
                core::error_code::not_found,
                "Index directory not found: " + path,
                "index_manager.load"
            });
        }

        // Load manifest file
        std::string manifest_path = path + "/index_manager.manifest";
        std::ifstream manifest(manifest_path, std::ios::binary);
        if (!manifest) {
            return std::vesper_unexpected(core::error{
                core::error_code::not_found,
                "Manifest file not found: " + manifest_path,
                "index_manager.load"
            });
        }

        // Read and validate version
        std::uint32_t version = 0;
        manifest.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (version != 1) {
            return std::vesper_unexpected(core::error{
                core::error_code::config_invalid,
                "Unsupported manifest version: " + std::to_string(version),
                "index_manager.load"
            });
        }

        // Read basic metadata
        manifest.read(reinterpret_cast<char*>(&dimension_), sizeof(dimension_));
        manifest.read(reinterpret_cast<char*>(&total_vectors_), sizeof(total_vectors_));
        manifest.read(reinterpret_cast<char*>(&memory_budget_mb_), sizeof(memory_budget_mb_));
        manifest.read(reinterpret_cast<char*>(&needs_optimization_), sizeof(needs_optimization_));

        // Read build configuration
        manifest.read(reinterpret_cast<char*>(&build_config_.strategy), sizeof(build_config_.strategy));
        manifest.read(reinterpret_cast<char*>(&build_config_.type), sizeof(build_config_.type));
        manifest.read(reinterpret_cast<char*>(&build_config_.memory_budget_mb), sizeof(build_config_.memory_budget_mb));
        manifest.read(reinterpret_cast<char*>(&build_config_.cache_size_mb), sizeof(build_config_.cache_size_mb));

        // Read HNSW params
        manifest.read(reinterpret_cast<char*>(&build_config_.hnsw_params.M), sizeof(build_config_.hnsw_params.M));
        manifest.read(reinterpret_cast<char*>(&build_config_.hnsw_params.efConstruction), sizeof(build_config_.hnsw_params.efConstruction));
        manifest.read(reinterpret_cast<char*>(&build_config_.hnsw_params.seed), sizeof(build_config_.hnsw_params.seed));

        // Read IVF-PQ params
        manifest.read(reinterpret_cast<char*>(&build_config_.ivf_params.nlist), sizeof(build_config_.ivf_params.nlist));
        manifest.read(reinterpret_cast<char*>(&build_config_.ivf_params.m), sizeof(build_config_.ivf_params.m));
        manifest.read(reinterpret_cast<char*>(&build_config_.ivf_params.nbits), sizeof(build_config_.ivf_params.nbits));

        // Read Vamana params
        manifest.read(reinterpret_cast<char*>(&build_config_.vamana_params.degree), sizeof(build_config_.vamana_params.degree));
        manifest.read(reinterpret_cast<char*>(&build_config_.vamana_params.alpha), sizeof(build_config_.vamana_params.alpha));
        manifest.read(reinterpret_cast<char*>(&build_config_.vamana_params.L), sizeof(build_config_.vamana_params.L));

        // Read active index flags
        bool has_hnsw = false;
        bool has_ivf_pq = false;
        bool has_disk_graph = false;

        manifest.read(reinterpret_cast<char*>(&has_hnsw), sizeof(has_hnsw));
        manifest.read(reinterpret_cast<char*>(&has_ivf_pq), sizeof(has_ivf_pq));
        manifest.read(reinterpret_cast<char*>(&has_disk_graph), sizeof(has_disk_graph));

        if (!manifest.good()) {
            return std::vesper_unexpected(core::error{
                core::error_code::io_failed,
                "Failed to read manifest file",
                "index_manager.load"
            });
        }

        manifest.close();

        // Load individual indexes
        if (has_hnsw) {
            auto result = HnswIndex::load(path + "/hnsw");
            if (!result) {
                return std::vesper_unexpected(core::error{
                    core::error_code::io_failed,
                    "Failed to load HNSW index: " + result.error().message,
                    "index_manager.load"
                });
            }
            hnsw_index_ = std::make_unique<HnswIndex>(std::move(*result));
        }

        if (has_ivf_pq) {
            auto result = IvfPqIndex::load(path + "/ivf_pq");
            if (!result) {
                return std::vesper_unexpected(core::error{
                    core::error_code::io_failed,
                    "Failed to load IVF-PQ index: " + result.error().message,
                    "index_manager.load"
                });
            }
            ivf_pq_index_ = std::make_unique<IvfPqIndex>(std::move(*result));
        }

        if (has_disk_graph) {
            disk_graph_index_ = std::make_unique<DiskGraphIndex>(dimension_);
            auto result = disk_graph_index_->load(path + "/disk_graph");
            if (!result) {
                return std::vesper_unexpected(core::error{
                    core::error_code::io_failed,
                    "Failed to load DiskGraph index: " + result.error().message,
                    "index_manager.load"
                });
            }
        }

        // Load tombstone manager
        tombstone_manager_ = std::make_unique<tombstone::TombstoneManager>();
        if (fs::exists(path + "/tombstones.bin")) {
            auto result = tombstone_manager_->load(path + "/tombstones.bin");
            if (!result) {
                return std::vesper_unexpected(core::error{
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
                return std::vesper_unexpected(core::error{
                    core::error_code::io_failed,
                    "Failed to load metadata: " + result.error().message,
                    "index_manager.load"
                });
            }
        }

        // Successfully loaded

        return {};
    }

    // Internal version that doesn't acquire lock (for use when lock is already held)
    auto memory_usage_internal() const -> std::size_t {
        std::size_t total = 0;

        if (hnsw_index_) total += estimate_hnsw_memory();
        if (ivf_pq_index_) total += estimate_ivf_pq_memory();
        if (disk_graph_index_) total += estimate_diskann_memory();

        return total;
    }

    auto memory_usage() const -> std::size_t {
        std::shared_lock lock(mutex_);
        return memory_usage_internal();
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
            return std::vesper_unexpected(core::error{
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
            return std::vesper_unexpected(core::error{
                core::error_code::not_initialized,
                "Metadata store not initialized",
                "index_manager.get_metadata"
            });
        }

        auto result = metadata_store_->get(id);
        if (!result) {
            return std::vesper_unexpected(result.error());
        }

        return result->attributes;
    }

    auto remove_metadata(std::uint64_t id) -> std::expected<void, core::error> {
        if (!metadata_store_) {
            return std::vesper_unexpected(core::error{
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
        // Note: This is called with lock already held, so use internal version
        std::size_t current_memory_mb = memory_usage_internal() / (1024 * 1024);

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
                        // Extract vectors from HNSW before clearing
                        std::vector<float> extracted_vectors;
                        std::vector<std::uint64_t> extracted_ids;
                        extracted_vectors.reserve(hnsw_index_->size() * dimension_);
                        extracted_ids.reserve(hnsw_index_->size());

                        for (std::uint64_t id = 0; id < hnsw_index_->size(); ++id) {
                            auto vec_result = hnsw_index_->get_vector(id);
                            if (vec_result) {
                                extracted_ids.push_back(id);
                                const auto& vec = vec_result.value();
                                extracted_vectors.insert(extracted_vectors.end(), vec.begin(), vec.end());
                            }
                        }

                        // Clear HNSW to free memory
                        hnsw_index_.reset();
                        excess_mb -= priority.memory_mb;

                        // Build IVF-PQ with extracted vectors
                        if (!extracted_ids.empty()) {
                            ivf_pq_index_ = std::make_unique<IvfPqIndex>();
                            IvfPqTrainParams params = build_config_.ivf_params;
                            auto build_result = ivf_pq_index_->train(
                                extracted_vectors.data(),
                                dimension_,
                                extracted_ids.size(),
                                params
                            );
                            if (build_result) {
                                auto add_result = ivf_pq_index_->add(
                                    extracted_ids.data(),
                                    extracted_vectors.data(),
                                    extracted_ids.size()
                                );
                                if (!add_result) {
                                    ivf_pq_index_.reset();
                                }
                            } else {
                                ivf_pq_index_.reset();
                            }
                        }
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
                        build_config_.cache_size_mb = std::max(static_cast<std::size_t>(64), build_config_.cache_size_mb / 2);
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
        current_memory_mb = memory_usage_internal() / (1024 * 1024);
        if (current_memory_mb > memory_budget_mb_) {
            return std::vesper_unexpected(core::error{
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

        // Initialize quantizers if enabled
        if (config.enable_rabitq) {
            RaBitQTrainParams rabitq_params;
            rabitq_params.bits = static_cast<QuantizationBits>(config.quantization_bits);
            rabitq_params.use_rotation = true;
            auto result = rabitq_quantizer_->train(vectors, n, dimension_, rabitq_params);
            if (!result) return result;
        }

        if (config.enable_matryoshka) {
            MatryoshkaConfig matryoshka_config;
            if (!config.matryoshka_dims.empty()) {
                matryoshka_config.dimensions = config.matryoshka_dims;
            }
            matryoshka_config.target_recall = config.target_recall;
            auto result = matryoshka_embedding_->init(dimension_, matryoshka_config);
            if (!result) return result;

            // Analyze embeddings for optimal truncation
            matryoshka_embedding_->analyze(vectors, n);
        }

        switch (type) {
            case IndexType::HNSW: {
                hnsw_index_ = std::make_unique<HnswIndex>();

                // Enable parallel construction
                HnswBuildParams hnsw_params = config.hnsw_params;
                if (config.use_parallel_construction) {
                    hnsw_params.num_threads = 0; // Auto-detect thread count
                }

                auto result = hnsw_index_->init(dimension_, hnsw_params, n);
                if (!result) return result;

                // Use batch addition for parallel construction
                if (config.use_parallel_construction && n > 1000) {
                    std::vector<std::uint64_t> ids(n);
                    std::iota(ids.begin(), ids.end(), 0);
                    auto add_result = hnsw_index_->add_batch(ids.data(), vectors, n);
                    if (!add_result) return add_result;
                } else {
                    // Sequential addition for small datasets
                    for (std::size_t i = 0; i < n; ++i) {
                        auto add_result = hnsw_index_->add(i, vectors + i * dimension_);
                        if (!add_result) return add_result;
                    }
                }
                break;
            }

            case IndexType::IVF_PQ: {
                ivf_pq_index_ = std::make_unique<IvfPqIndex>();

                // Apply sensible IVF-PQ defaults when not provided (esp. under Auto)
                IvfPqTrainParams params = config.ivf_params;
                if (config.strategy != SelectionStrategy::Manual) {
                    if (params.nlist == 0) {
                        params.nlist = (n >= 1000000) ? 4096 : (n >= 100000 ? 2048 : 512);
                    }
                    if (params.m == 0) {
                        if (dimension_ % 16 == 0) params.m = 16;
                        else if (dimension_ % 8 == 0) params.m = 8;
                        else params.m = 4;
                    }
                    if (params.nbits == 0) {
                        params.nbits = 8;
                    }
                }
                // Persist the effective params in build_config_ for save/load
                build_config_.ivf_params = params;

                // Train on a subsample to keep k-means tractable on very large n.
                // Heuristic: cap training set to at most 200k points, evenly spaced.
                std::size_t train_n = std::min<std::size_t>(n, 200000);
                const float* train_ptr = vectors;
                std::vector<float> train_buf;
                if (train_n < n) {
                    train_buf.resize(train_n * dimension_);
                    const double step = static_cast<double>(n) / static_cast<double>(train_n);
                    for (std::size_t t = 0; t < train_n; ++t) {
                        std::size_t i = static_cast<std::size_t>(t * step);
                        if (i >= n) i = n - 1;
                        std::memcpy(train_buf.data() + t * dimension_,
                                    vectors + i * dimension_,
                                    dimension_ * sizeof(float));
                    }
                    train_ptr = train_buf.data();
                }
                // Slightly reduce iterations in Auto to speed up without hurting recall much
                if (config.strategy != SelectionStrategy::Manual) {
                    params.max_iter = std::min<std::uint32_t>(params.max_iter, 20);
                }

                auto result = ivf_pq_index_->train(train_ptr, dimension_, train_n, params);
                if (!result) {
                    return std::vesper_unexpected(result.error());
                }

                // Add all vectors
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

    // Pending operations for batch processing
    std::unique_ptr<std::vector<std::pair<std::uint64_t, std::vector<float>>>> pending_diskgraph_additions_;

    // Quantization support
    std::unique_ptr<RaBitQuantizer> rabitq_quantizer_;
    std::unique_ptr<MatryoshkaEmbedding> matryoshka_embedding_;

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

auto IndexManager::get_memory_budget() const -> std::size_t {
    return impl_->get_memory_budget();
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
