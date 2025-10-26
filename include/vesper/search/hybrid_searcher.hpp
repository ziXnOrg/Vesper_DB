#pragma once

/** \file hybrid_searcher.hpp
 *  \brief Hybrid search combining sparse (BM25) and dense vector search.
 *
 * Provides unified search interface that intelligently combines:
 * - Sparse keyword search (BM25, TF-IDF)
 * - Dense semantic search (vector similarity)
 * - Learned sparse representations (SPLADE, ColBERT)
 *
 * Features multiple fusion algorithms and adaptive query routing
 * for optimal performance across different query types.
 *
 * Thread-safety: All search operations are thread-safe.
 */

#include <cstdint>
#include <expected>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <optional>
#include <functional>
#include <atomic>

#include "vesper/error.hpp"
#include "vesper/index/bm25.hpp"
#include "vesper/index/index_manager.hpp"
#include <roaring/roaring.h>

namespace vesper::search {

/** \brief Query routing strategy. */
enum class QueryStrategy {
    AUTO,           /**< Automatic routing based on query analysis */
    DENSE_FIRST,    /**< Dense search first, optional sparse */
    SPARSE_FIRST,   /**< Sparse search first, optional dense */
    PARALLEL        /**< Both searches in parallel */
};

/** \brief Fusion algorithm for combining results. */
enum class FusionAlgorithm {
    RECIPROCAL_RANK,       /**< RRF with configurable k parameter */
    WEIGHTED_SUM,          /**< Linear combination with weights */
    MAX_SCORE,             /**< Take maximum of normalized scores */
    LATE_INTERACTION       /**< ColBERT-style late interaction */
};

/** \brief Hybrid query combining text and embeddings. */
struct HybridQuery {
    std::string text;                   /**< Text query for sparse search */
    std::vector<float> dense_embedding; /**< Dense embedding for vector search */
    std::shared_ptr<roaring::Roaring> filter; /**< Optional document filter */
};

/** \brief Hybrid search configuration. */
struct HybridSearchConfig {
    // Query parameters
    std::uint32_t k{10};                /**< Number of results to return */
    QueryStrategy query_strategy{QueryStrategy::AUTO}; /**< Query routing strategy */
    
    // Fusion parameters
    FusionAlgorithm fusion_algorithm{FusionAlgorithm::RECIPROCAL_RANK};
    float rrf_k{60.0f};                 /**< RRF k parameter (typically 60) */
    float dense_weight{0.5f};           /**< Weight for dense scores (0-1) */
    float sparse_weight{0.5f};          /**< Weight for sparse scores (0-1) */
    
    // Reranking parameters
    std::uint32_t rerank_factor{10};    /**< Oversample factor for reranking */
    
};

/** \brief Hybrid search result with detailed scoring. */
struct HybridResult {
    std::uint64_t doc_id;               /**< Document identifier */
    float dense_score{0.0f};            /**< Dense vector similarity score */
    float sparse_score{0.0f};           /**< Sparse BM25/TF-IDF score */
    float fused_score{0.0f};            /**< Final fused score */
    std::uint32_t dense_rank{0};        /**< Rank in dense results */
    std::uint32_t sparse_rank{0};       /**< Rank in sparse results */
    
    /** \brief Compare by fused score (for sorting). */
    auto operator<(const HybridResult& other) const noexcept -> bool {
        return fused_score > other.fused_score;  // Higher is better
    }
};

/** \brief Query analysis result. */
struct QueryAnalysis {
    bool has_keywords{false};           /**< Query contains specific keywords */
    bool needs_semantic{false};         /**< Query requires semantic understanding */
    float keyword_score{0.0f};          /**< Keyword importance (0-1) */
    float semantic_score{0.0f};         /**< Semantic importance (0-1) */
    QueryStrategy recommended{QueryStrategy::AUTO}; /**< Recommended strategy */
    std::vector<std::string> keywords;  /**< Extracted keywords */
};

/** \brief Hybrid search statistics. */
struct HybridSearchStats {
    std::atomic<std::size_t> total_queries{0};    /**< Total queries processed */
    std::atomic<std::size_t> dense_searches{0};   /**< Dense searches executed */
    std::atomic<std::size_t> sparse_searches{0};  /**< Sparse searches executed */
    std::atomic<std::size_t> batch_queries{0};    /**< Batch queries processed */
    std::atomic<std::size_t> total_latency_us{0}; /**< Total latency in microseconds */
};

/** \brief Hybrid searcher combining sparse and dense search.
 *
 * Example usage:
 * ```cpp
 * HybridSearcher searcher;
 * 
 * // Initialize with indices
 * searcher.init(sparse_index, dense_index, config);
 * 
 * // Search with automatic routing
 * auto results = searcher.search("machine learning algorithms", 10);
 * 
 * // Search with specific strategy
 * HybridSearchConfig config;
 * config.strategy = QueryStrategy::HybridRRF;
 * auto results = searcher.search("neural networks", config);
 * ```
 */
class HybridSearcher {
public:
    /** \brief Constructor with index managers. */
    HybridSearcher(std::shared_ptr<index::IndexManager> index_manager,
                   std::shared_ptr<index::BM25Index> bm25_index);
    ~HybridSearcher();
    
    /** \brief Search with hybrid query.
     *
     * \param query Hybrid query with text and/or embeddings
     * \param config Search configuration
     * \return Ranked results or error
     */
    auto search(const HybridQuery& query, const HybridSearchConfig& config)
        -> std::expected<std::vector<HybridResult>, core::error>;
    
    /** \brief Batch search with multiple queries.
     *
     * \param queries Vector of hybrid queries
     * \param config Search configuration
     * \return Vector of result sets or error
     */
    auto batch_search(const std::vector<HybridQuery>& queries,
                     const HybridSearchConfig& config)
        -> std::expected<std::vector<std::vector<HybridResult>>, core::error>;
    
    /** \brief Get search statistics. */
    auto get_stats() const noexcept -> HybridSearchStats;
    
    /** \brief Clear query cache. */
    auto clear_cache() -> void;
    
    /** \brief Update search configuration. */
    auto update_config(const HybridSearchConfig& config) -> void;
    
    /** \brief Check if searcher is initialized. */
    auto is_initialized() const noexcept -> bool;
    
    /** \brief Set custom query analyzer.
     *
     * \param analyzer Custom analyzer function
     *
     * Allows replacing the default query analysis logic.
     */
    auto set_query_analyzer(
        std::function<QueryAnalysis(std::string_view)> analyzer) -> void;
    
    /** \brief Set custom fusion function.
     *
     * \param fusion Custom fusion function
     *
     * Allows implementing custom result fusion logic.
     */
    auto set_fusion_function(
        std::function<float(const HybridResult&)> fusion) -> void;
    
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/** \brief Fusion algorithms for combining search results. */
namespace fusion {

/** \brief Reciprocal Rank Fusion (RRF).
 *
 * \param rank_dense Rank in dense results (1-based)
 * \param rank_sparse Rank in sparse results (1-based)
 * \param k RRF k parameter (typically 60)
 * \return Fused score
 *
 * RRF formula: 1/(k + rank_dense) + 1/(k + rank_sparse)
 */
auto reciprocal_rank_fusion(std::uint32_t rank_dense,
                            std::uint32_t rank_sparse,
                            float k = 60.0f) -> float;

/** \brief Weighted linear combination.
 *
 * \param score_dense Normalized dense score (0-1)
 * \param score_sparse Normalized sparse score (0-1)
 * \param weight_dense Weight for dense score
 * \param weight_sparse Weight for sparse score
 * \return Fused score
 */
auto weighted_linear(float score_dense, float score_sparse,
                    float weight_dense = 0.5f,
                    float weight_sparse = 0.5f) -> float;

/** \brief Normalize scores to 0-1 range.
 *
 * \param scores Input scores
 * \return Normalized scores
 */
auto normalize_scores(const std::vector<float>& scores)
    -> std::vector<float>;

/** \brief Late interaction fusion (ColBERT-style).
 *
 * \param dense_scores Token-level dense scores
 * \param sparse_scores Token-level sparse scores
 * \return Fused score
 *
 * Computes maximum similarity across token pairs.
 */
auto late_interaction(const std::vector<float>& dense_scores,
                     const std::vector<float>& sparse_scores) -> float;

} // namespace fusion

} // namespace vesper::search