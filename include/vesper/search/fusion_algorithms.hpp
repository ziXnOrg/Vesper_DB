#pragma once

/** \file fusion_algorithms.hpp
 *  \brief Algorithms for fusing sparse and dense search results.
 *
 * Implements state-of-the-art fusion techniques including:
 * - Reciprocal Rank Fusion (RRF)
 * - Weighted linear combination
 * - Late interaction (ColBERT-style)
 * - Learned fusion models
 */

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
#include <unordered_map>
#include <string>
#include <expected>

namespace vesper::search::fusion {

/** \brief Result from a single search method. */
struct SearchResult {
    std::uint64_t doc_id;
    float score;
    std::uint32_t rank;  // 1-based ranking
};

/** \brief Combined result from multiple search methods. */
struct FusedResult {
    std::uint64_t doc_id;
    float dense_score{0.0f};
    float sparse_score{0.0f};
    float fused_score{0.0f};
    std::uint32_t dense_rank{0};
    std::uint32_t sparse_rank{0};
};

/** \brief Reciprocal Rank Fusion implementation.
 *
 * RRF is parameter-free (except for k) and robust across different
 * score distributions. Used by Milvus, Weaviate, and others.
 */
class ReciprocalRankFusion {
public:
    explicit ReciprocalRankFusion(float k = 60.0f) : k_(k) {}
    
    /** \brief Fuse two ranked lists.
     *
     * \param dense_results Results from dense search
     * \param sparse_results Results from sparse search
     * \param top_k Number of results to return
     * \return Fused and ranked results
     */
    auto fuse(const std::vector<SearchResult>& dense_results,
              const std::vector<SearchResult>& sparse_results,
              std::size_t top_k) const -> std::vector<FusedResult>;
    
    /** \brief Compute RRF score for a single document.
     *
     * \param dense_rank Rank in dense results (0 if not present)
     * \param sparse_rank Rank in sparse results (0 if not present)
     * \return RRF score
     */
    auto score(std::uint32_t dense_rank, std::uint32_t sparse_rank) const -> float {
        float score = 0.0f;
        if (dense_rank > 0) {
            score += 1.0f / (k_ + dense_rank);
        }
        if (sparse_rank > 0) {
            score += 1.0f / (k_ + sparse_rank);
        }
        return score;
    }
    
    /** \brief Update k parameter. */
    auto set_k(float k) -> void { k_ = k; }
    
private:
    float k_;
};

/** \brief Weighted linear combination of scores.
 *
 * Simple but effective when score distributions are calibrated.
 */
class WeightedFusion {
public:
    WeightedFusion(float dense_weight = 0.5f, float sparse_weight = 0.5f)
        : dense_weight_(dense_weight), sparse_weight_(sparse_weight) {
        normalize_weights();
    }
    
    /** \brief Fuse two score lists.
     *
     * \param dense_results Results from dense search
     * \param sparse_results Results from sparse search
     * \param top_k Number of results to return
     * \return Fused and ranked results
     */
    auto fuse(const std::vector<SearchResult>& dense_results,
              const std::vector<SearchResult>& sparse_results,
              std::size_t top_k) const -> std::vector<FusedResult>;
    
    /** \brief Compute weighted score.
     *
     * \param dense_score Normalized dense score
     * \param sparse_score Normalized sparse score
     * \return Weighted combination
     */
    auto score(float dense_score, float sparse_score) const -> float {
        return dense_weight_ * dense_score + sparse_weight_ * sparse_score;
    }
    
    /** \brief Update weights. */
    auto set_weights(float dense_weight, float sparse_weight) -> void {
        dense_weight_ = dense_weight;
        sparse_weight_ = sparse_weight;
        normalize_weights();
    }
    
private:
    float dense_weight_;
    float sparse_weight_;
    
    auto normalize_weights() -> void {
        float sum = dense_weight_ + sparse_weight_;
        if (sum > 0.0f) {
            dense_weight_ /= sum;
            sparse_weight_ /= sum;
        }
    }
};

/** \brief Late interaction fusion (ColBERT-style).
 *
 * Computes token-level interactions between query and document.
 */
class LateInteractionFusion {
public:
    /** \brief Configuration for late interaction. */
    struct Config {
        bool use_max_sim{true};      /**< Use MaxSim aggregation */
        bool normalize_scores{true};  /**< Normalize before fusion */
        float temperature{1.0f};      /**< Temperature for softmax */
    };
    
    explicit LateInteractionFusion(const Config& config = {})
        : config_(config) {}
    
    /** \brief Fuse token-level scores.
     *
     * \param dense_token_scores Dense scores per token [n_tokens x n_docs]
     * \param sparse_token_scores Sparse scores per token [n_tokens x n_docs]
     * \param top_k Number of results to return
     * \return Fused and ranked results
     */
    auto fuse_tokens(const std::vector<std::vector<float>>& dense_token_scores,
                     const std::vector<std::vector<float>>& sparse_token_scores,
                     std::size_t top_k) const -> std::vector<FusedResult>;
    
    /** \brief Compute MaxSim score.
     *
     * \param query_scores Query token scores
     * \param doc_scores Document token scores
     * \return MaxSim score
     */
    static auto max_sim(const std::vector<float>& query_scores,
                       const std::vector<float>& doc_scores) -> float;
    
private:
    Config config_;
};

/** \brief Adaptive fusion that learns optimal weights.
 *
 * Uses click-through data or relevance judgments to learn
 * optimal fusion parameters.
 */
class AdaptiveFusion {
public:
    /** \brief Training example for learning fusion. */
    struct TrainingExample {
        std::vector<SearchResult> dense_results;
        std::vector<SearchResult> sparse_results;
        std::vector<std::uint64_t> relevant_docs;  // Ground truth
    };
    
    AdaptiveFusion() = default;
    
    /** \brief Train fusion parameters.
     *
     * \param examples Training examples with relevance judgments
     * \return Success or error message
     */
    auto train(const std::vector<TrainingExample>& examples)
        -> std::expected<void, std::string>;
    
    /** \brief Fuse with learned parameters.
     *
     * \param dense_results Results from dense search
     * \param sparse_results Results from sparse search
     * \param top_k Number of results to return
     * \return Fused and ranked results
     */
    auto fuse(const std::vector<SearchResult>& dense_results,
              const std::vector<SearchResult>& sparse_results,
              std::size_t top_k) const -> std::vector<FusedResult>;
    
    /** \brief Get learned parameters. */
    auto get_params() const -> std::unordered_map<std::string, float>;
    
private:
    float dense_weight_{0.5f};
    float sparse_weight_{0.5f};
    float rrf_k_{60.0f};
    bool use_rrf_{true};
    
    /** \brief Optimize parameters using coordinate descent. */
    auto optimize_params(const std::vector<TrainingExample>& examples) -> void;
    
    /** \brief Compute NDCG for evaluation. */
    static auto compute_ndcg(const std::vector<FusedResult>& results,
                            const std::vector<std::uint64_t>& relevant,
                            std::size_t k) -> float;
};

/** \brief Utility functions for fusion algorithms. */

/** \brief Normalize scores to [0, 1] range.
 *
 * \param scores Input scores
 * \return Normalized scores
 */
inline auto normalize_scores(std::vector<float> scores) -> std::vector<float> {
    if (scores.empty()) return scores;
    
    float min_score = *std::min_element(scores.begin(), scores.end());
    float max_score = *std::max_element(scores.begin(), scores.end());
    
    if (max_score == min_score) {
        std::fill(scores.begin(), scores.end(), 0.5f);
        return scores;
    }
    
    for (float& score : scores) {
        score = (score - min_score) / (max_score - min_score);
    }
    
    return scores;
}

/** \brief Apply softmax to scores.
 *
 * \param scores Input scores
 * \param temperature Temperature parameter
 * \return Softmax probabilities
 */
inline auto softmax(std::vector<float> scores, float temperature = 1.0f)
    -> std::vector<float> {
    
    if (scores.empty()) return scores;
    
    // Subtract max for numerical stability
    float max_score = *std::max_element(scores.begin(), scores.end());
    
    float sum = 0.0f;
    for (float& score : scores) {
        score = std::exp((score - max_score) / temperature);
        sum += score;
    }
    
    if (sum > 0.0f) {
        for (float& score : scores) {
            score /= sum;
        }
    }
    
    return scores;
}

/** \brief Merge and deduplicate results from multiple sources.
 *
 * \param results_lists Multiple result lists
 * \return Merged and deduplicated results
 */
inline auto merge_results(
    const std::vector<std::vector<SearchResult>>& results_lists)
    -> std::vector<SearchResult> {
    
    std::unordered_map<std::uint64_t, SearchResult> merged;
    
    for (const auto& results : results_lists) {
        for (const auto& result : results) {
            auto it = merged.find(result.doc_id);
            if (it == merged.end() || result.score > it->second.score) {
                merged[result.doc_id] = result;
            }
        }
    }
    
    std::vector<SearchResult> output;
    output.reserve(merged.size());
    for (const auto& [doc_id, result] : merged) {
        output.push_back(result);
    }
    
    // Sort by score
    std::sort(output.begin(), output.end(),
              [](const SearchResult& a, const SearchResult& b) {
                  return a.score > b.score;
              });
    
    // Assign ranks
    for (std::size_t i = 0; i < output.size(); ++i) {
        output[i].rank = static_cast<std::uint32_t>(i + 1);
    }
    
    return output;
}

} // namespace vesper::search::fusion