#include "vesper/search/hybrid_searcher.hpp"
#include "vesper/search/fusion_algorithms.hpp"
#include <algorithm>
#include <execution>
#include <future>
#include <numeric>
#include <unordered_set>
#include <unordered_map>

namespace vesper::search {

// HybridSearcher::Impl class definition
class HybridSearcher::Impl {
public:
    Impl(std::shared_ptr<index::IndexManager> index_manager,
         std::shared_ptr<index::BM25Index> bm25_index)
        : index_manager_(std::move(index_manager))
        , bm25_index_(std::move(bm25_index)) {
    }

    auto search(const search::HybridQuery& query, const search::HybridSearchConfig& config)
        -> std::expected<std::vector<search::HybridResult>, core::error>;

    auto batch_search(const std::vector<search::HybridQuery>& queries,
                     const search::HybridSearchConfig& config)
        -> std::expected<std::vector<std::vector<search::HybridResult>>, core::error>;

    auto get_stats() const noexcept -> search::HybridSearchStats;

private:
    std::shared_ptr<index::IndexManager> index_manager_;
    std::shared_ptr<index::BM25Index> bm25_index_;
    mutable search::HybridSearchStats stats_;

    // Helper functions
    auto execute_dense_search(const std::vector<float>& embedding,
                             std::uint32_t k,
                             const roaring::Roaring* filter)
        -> std::expected<std::vector<std::pair<std::uint64_t, float>>, core::error>;

    auto execute_sparse_search(const std::string& text,
                              std::uint32_t k,
                              const roaring::Roaring* filter)
        -> std::expected<std::vector<std::pair<std::uint64_t, float>>, core::error>;

    auto fuse_results(const std::vector<std::pair<std::uint64_t, float>>& dense_results,
                     const std::vector<std::pair<std::uint64_t, float>>& sparse_results,
                     const search::HybridSearchConfig& config)
        -> std::vector<search::HybridResult>;
};

// HybridSearcher implementation

HybridSearcher::HybridSearcher(std::shared_ptr<index::IndexManager> index_manager,
                               std::shared_ptr<index::BM25Index> bm25_index)
    : impl_(std::make_unique<Impl>(std::move(index_manager), std::move(bm25_index))) {
}

HybridSearcher::~HybridSearcher() = default;

auto HybridSearcher::search(const HybridQuery& query, const HybridSearchConfig& config)
    -> std::expected<std::vector<HybridResult>, core::error> {
    return impl_->search(query, config);
}

auto HybridSearcher::batch_search(const std::vector<HybridQuery>& queries,
                                  const HybridSearchConfig& config)
    -> std::expected<std::vector<std::vector<HybridResult>>, core::error> {
    return impl_->batch_search(queries, config);
}

auto HybridSearcher::get_stats() const noexcept -> HybridSearchStats {
    // Return a copy that's constructed from atomic values
    return impl_->get_stats();
}

auto HybridSearcher::is_initialized() const noexcept -> bool {
    return impl_ != nullptr;
}


// HybridSearcher::Impl implementation

auto HybridSearcher::Impl::execute_dense_search(
    const std::vector<float>& embedding,
    std::uint32_t k,
    const roaring::Roaring* filter)
    -> std::expected<std::vector<std::pair<std::uint64_t, float>>, core::error> {

    // Use IndexManager for dense vector search
    index::QueryConfig config;
    config.k = k;

    auto result = index_manager_->search(embedding.data(), config);
    if (!result) {
        return std::vesper_unexpected(result.error());
    }

    // Apply optional filter to dense results if provided
    if (filter != nullptr) {
        auto& vec = *result;
        vec.erase(std::remove_if(vec.begin(), vec.end(), [filter](const auto& p) {
            return !filter->contains(static_cast<uint32_t>(p.first));
        }), vec.end());
    }

    return *result;
}

auto HybridSearcher::Impl::execute_sparse_search(
    const std::string& text,
    std::uint32_t k,
    const roaring::Roaring* filter)
    -> std::expected<std::vector<std::pair<std::uint64_t, float>>, core::error> {

    return bm25_index_->search(text, k, filter);
}

auto HybridSearcher::Impl::fuse_results(
    const std::vector<std::pair<std::uint64_t, float>>& dense_results,
    const std::vector<std::pair<std::uint64_t, float>>& sparse_results,
    const search::HybridSearchConfig& config)
    -> std::vector<search::HybridResult> {

    // Create maps for easy lookup
    std::unordered_map<std::uint64_t, float> dense_scores;
    std::unordered_map<std::uint64_t, float> sparse_scores;
    std::unordered_set<std::uint64_t> all_ids;

    for (const auto& [id, score] : dense_results) {
        dense_scores[id] = score;
        all_ids.insert(id);
    }

    for (const auto& [id, score] : sparse_results) {
        sparse_scores[id] = score;
        all_ids.insert(id);
    }

    // Create hybrid results
    std::vector<search::HybridResult> results;
    results.reserve(all_ids.size());

    for (std::uint64_t id : all_ids) {
        search::HybridResult result;
        result.doc_id = id;
        result.dense_score = dense_scores.count(id) ? dense_scores[id] : 0.0f;
        result.sparse_score = sparse_scores.count(id) ? sparse_scores[id] : 0.0f;
        results.push_back(result);
    }

    // Convert to SearchResult format for fusion algorithms
    std::vector<fusion::SearchResult> dense_search_results;
    std::vector<fusion::SearchResult> sparse_search_results;

    dense_search_results.reserve(dense_results.size());
    for (std::size_t i = 0; i < dense_results.size(); ++i) {
        fusion::SearchResult sr;
        sr.doc_id = dense_results[i].first;
        sr.score = dense_results[i].second;
        sr.rank = static_cast<std::uint32_t>(i + 1);
        dense_search_results.push_back(sr);
    }

    sparse_search_results.reserve(sparse_results.size());
    for (std::size_t i = 0; i < sparse_results.size(); ++i) {
        fusion::SearchResult sr;
        sr.doc_id = sparse_results[i].first;
        sr.score = sparse_results[i].second;
        sr.rank = static_cast<std::uint32_t>(i + 1);
        sparse_search_results.push_back(sr);
    }

    // Apply fusion algorithm
    std::vector<fusion::FusedResult> fused_results;

    switch (config.fusion_algorithm) {
        case search::FusionAlgorithm::RECIPROCAL_RANK: {
            fusion::ReciprocalRankFusion rrf(config.rrf_k);
            fused_results = rrf.fuse(dense_search_results, sparse_search_results, config.k);
            break;
        }

        case search::FusionAlgorithm::WEIGHTED_SUM: {
            fusion::WeightedFusion weighted(config.dense_weight, config.sparse_weight);
            fused_results = weighted.fuse(dense_search_results, sparse_search_results, config.k);
            break;
        }

        case search::FusionAlgorithm::MAX_SCORE: {
            // Create results with max scoring
            for (auto& result : results) {
                result.fused_score = std::max(result.dense_score, result.sparse_score);
            }
            break;
        }

        case search::FusionAlgorithm::LATE_INTERACTION: {
            // Late interaction requires both embeddings to be present
            // For now, fall back to weighted sum
            fusion::WeightedFusion weighted(config.dense_weight, config.sparse_weight);
            fused_results = weighted.fuse(dense_search_results, sparse_search_results, config.k);
            break;
        }
    }

    // Convert fused results to HybridResult if we used fusion algorithms
    if (!fused_results.empty()) {
        results.clear();
        results.reserve(fused_results.size());
        for (const auto& fr : fused_results) {
            search::HybridResult hr;
            hr.doc_id = fr.doc_id;
            hr.dense_score = fr.dense_score;
            hr.sparse_score = fr.sparse_score;
            hr.fused_score = fr.fused_score;
            hr.dense_rank = fr.dense_rank;
            hr.sparse_rank = fr.sparse_rank;
            results.push_back(hr);
        }
    }

    // Sort by fused score
    std::sort(results.begin(), results.end(),
              [](const HybridResult& a, const HybridResult& b) {
                  return a.fused_score > b.fused_score;
              });

    // Limit to top k
    if (results.size() > config.k) {
        results.resize(config.k);
    }

    return results;
}

auto HybridSearcher::Impl::search(const search::HybridQuery& query,
                                  const search::HybridSearchConfig& config)
    -> std::expected<std::vector<search::HybridResult>, core::error> {

    auto start_time = std::chrono::steady_clock::now();

    // Determine strategy
    search::QueryStrategy strategy = config.query_strategy;
    if (strategy == search::QueryStrategy::AUTO) {
        // Auto-detect based on query content
        bool has_text = !query.text.empty();
        bool has_embedding = !query.dense_embedding.empty();

        if (has_text && has_embedding) {
            strategy = search::QueryStrategy::PARALLEL;
        } else if (has_text) {
            strategy = search::QueryStrategy::SPARSE_FIRST;
        } else if (has_embedding) {
            strategy = search::QueryStrategy::DENSE_FIRST;
        } else {
            return std::vesper_unexpected(core::error{
                core::error_code::invalid_argument,
                "Query must have either text or embedding",
                "hybrid_searcher"
            });
        }
    }

    std::vector<std::pair<std::uint64_t, float>> dense_results;
    std::vector<std::pair<std::uint64_t, float>> sparse_results;
    bool did_dense = false;
    bool did_sparse = false;

    // Execute searches based on strategy
    switch (strategy) {
        case search::QueryStrategy::PARALLEL: {
            // Launch both searches in parallel
            did_dense = true;
            did_sparse = true;
            auto dense_future = std::async(std::launch::async,
                [this, &query, &config]() {
                    return execute_dense_search(query.dense_embedding,
                                               config.k * config.rerank_factor,
                                               query.filter.get());
                });

            auto sparse_future = std::async(std::launch::async,
                [this, &query, &config]() {
                    return execute_sparse_search(query.text,
                                                config.k * config.rerank_factor,
                                                query.filter.get());
                });

            auto dense_result = dense_future.get();
            auto sparse_result = sparse_future.get();

            if (!dense_result) {
                return std::vesper_unexpected(dense_result.error());
            }
            if (!sparse_result) {
                return std::vesper_unexpected(sparse_result.error());
            }

            dense_results = std::move(*dense_result);
            sparse_results = std::move(*sparse_result);
            break;
        }

        case search::QueryStrategy::DENSE_FIRST: {
            if (query.dense_embedding.empty()) {
                return std::vesper_unexpected(core::error{
                    core::error_code::invalid_argument,
                    "Dense embedding required for dense-first strategy",
                    "hybrid_searcher"
                });
            }

            did_dense = true;
            auto dense_result = execute_dense_search(query.dense_embedding,
                                                    config.k * config.rerank_factor,
                                                    query.filter.get());
            if (!dense_result) {
                return std::vesper_unexpected(dense_result.error());
            }
            dense_results = std::move(*dense_result);

            // Optionally run sparse search if text is provided
            if (!query.text.empty()) {
                did_sparse = true;
                auto sparse_result = execute_sparse_search(query.text,
                                                          config.k * config.rerank_factor,
                                                          query.filter.get());
                if (sparse_result) {
                    sparse_results = std::move(*sparse_result);
                }
            }
            break;
        }

        case search::QueryStrategy::SPARSE_FIRST: {
            if (query.text.empty()) {
                return std::vesper_unexpected(core::error{
                    core::error_code::invalid_argument,
                    "Text required for sparse-first strategy",
                    "hybrid_searcher"
                });
            }

            did_sparse = true;
            auto sparse_result = execute_sparse_search(query.text,
                                                      config.k * config.rerank_factor,
                                                      query.filter.get());
            if (!sparse_result) {
                return std::vesper_unexpected(sparse_result.error());
            }
            sparse_results = std::move(*sparse_result);

            // Optionally run dense search if embedding is provided
            if (!query.dense_embedding.empty()) {
                did_dense = true;
                auto dense_result = execute_dense_search(query.dense_embedding,
                                                        config.k * config.rerank_factor,
                                                        query.filter.get());
                if (dense_result) {
                    dense_results = std::move(*dense_result);
                }
            }
            break;
        }

        default:
            break;
    }

    // Fuse results
    auto results = fuse_results(dense_results, sparse_results, config);

    // Update stats
    auto end_time = std::chrono::steady_clock::now();
    auto latency = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time);

    stats_.total_queries.fetch_add(1);
    stats_.total_latency_us.fetch_add(latency.count());
    stats_.dense_searches.fetch_add(did_dense ? 1 : 0);
    stats_.sparse_searches.fetch_add(did_sparse ? 1 : 0);

    return results;
}

auto HybridSearcher::Impl::batch_search(
    const std::vector<search::HybridQuery>& queries,
    const search::HybridSearchConfig& config)
    -> std::expected<std::vector<std::vector<search::HybridResult>>, core::error> {

    std::vector<std::vector<search::HybridResult>> results;
    results.reserve(queries.size());

    // Process queries in parallel
    std::vector<std::future<std::expected<std::vector<search::HybridResult>, core::error>>> futures;
    futures.reserve(queries.size());

    for (const auto& query : queries) {
        futures.push_back(std::async(std::launch::async,
            [this, &query, &config]() {
                return search(query, config);
            }));
    }

    // Collect results
    for (auto& future : futures) {
        auto result = future.get();
        if (!result) {
            return std::vesper_unexpected(result.error());
        }
        results.push_back(std::move(*result));
    }

    stats_.batch_queries.fetch_add(1);

    return results;
}

auto HybridSearcher::Impl::get_stats() const noexcept -> search::HybridSearchStats {
    return search::HybridSearchStats{
        stats_.total_queries.load(),
        stats_.dense_searches.load(),
        stats_.sparse_searches.load(),
        stats_.batch_queries.load(),
        stats_.total_latency_us.load()
    };
}

} // namespace vesper::search
