#include "vesper/search/fusion_algorithms.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <expected>

namespace vesper::search::fusion {

// ReciprocalRankFusion implementation

auto ReciprocalRankFusion::fuse(const std::vector<SearchResult>& dense_results,
                              const std::vector<SearchResult>& sparse_results,
                              std::size_t top_k) const -> std::vector<FusedResult> {
    
    std::unordered_map<std::uint64_t, FusedResult> results;
    
    // Process dense results
    for (std::size_t i = 0; i < dense_results.size(); ++i) {
        const auto& res = dense_results[i];
        auto& fused = results[res.doc_id];
        fused.doc_id = res.doc_id;
        fused.dense_score = res.score;
        fused.dense_rank = static_cast<std::uint32_t>(i + 1);
    }
    
    // Process sparse results
    for (std::size_t i = 0; i < sparse_results.size(); ++i) {
        const auto& res = sparse_results[i];
        auto& fused = results[res.doc_id];
        fused.doc_id = res.doc_id;
        fused.sparse_score = res.score;
        fused.sparse_rank = static_cast<std::uint32_t>(i + 1);
    }
    
    // Calculate RRF scores - avoid structured bindings for MSVC
    std::vector<std::uint64_t> doc_ids;
    doc_ids.reserve(results.size());
    for (const auto& pair : results) {
        doc_ids.push_back(pair.first);
    }
    
    for (std::uint64_t doc_id : doc_ids) {
        auto& result = results[doc_id];
        result.fused_score = score(result.dense_rank, result.sparse_rank);
    }
    
    // Convert to vector and sort
    std::vector<FusedResult> output;
    output.reserve(results.size());
    for (const auto& pair : results) {
        output.push_back(pair.second);
    }
    
    // Sort by fused score descending
    std::sort(output.begin(), output.end(),
              [](const FusedResult& a, const FusedResult& b) {
                  return a.fused_score > b.fused_score;
              });
    
    // Limit to top_k
    if (output.size() > top_k) {
        output.resize(top_k);
    }
    
    return output;
}

// WeightedFusion implementation

auto WeightedFusion::fuse(const std::vector<SearchResult>& dense_results,
                         const std::vector<SearchResult>& sparse_results,
                         std::size_t top_k) const -> std::vector<FusedResult> {
    
    std::unordered_map<std::uint64_t, FusedResult> results;
    
    // Normalize dense scores
    float dense_min = std::numeric_limits<float>::max();
    float dense_max = std::numeric_limits<float>::lowest();
    for (std::size_t i = 0; i < dense_results.size(); ++i) {
        dense_min = std::min(dense_min, dense_results[i].score);
        dense_max = std::max(dense_max, dense_results[i].score);
    }
    
    float dense_range = dense_max - dense_min;
    if (dense_range <= 0) dense_range = 1.0f;
    
    // Normalize sparse scores
    float sparse_min = std::numeric_limits<float>::max();
    float sparse_max = std::numeric_limits<float>::lowest();
    for (std::size_t i = 0; i < sparse_results.size(); ++i) {
        sparse_min = std::min(sparse_min, sparse_results[i].score);
        sparse_max = std::max(sparse_max, sparse_results[i].score);
    }
    
    float sparse_range = sparse_max - sparse_min;
    if (sparse_range <= 0) sparse_range = 1.0f;
    
    // Process dense results
    for (std::size_t i = 0; i < dense_results.size(); ++i) {
        const auto& res = dense_results[i];
        auto& fused = results[res.doc_id];
        fused.doc_id = res.doc_id;
        fused.dense_score = (res.score - dense_min) / dense_range;
        fused.dense_rank = static_cast<std::uint32_t>(i + 1);
    }
    
    // Process sparse results
    for (std::size_t i = 0; i < sparse_results.size(); ++i) {
        const auto& res = sparse_results[i];
        auto& fused = results[res.doc_id];
        fused.doc_id = res.doc_id;
        fused.sparse_score = (res.score - sparse_min) / sparse_range;
        fused.sparse_rank = static_cast<std::uint32_t>(i + 1);
    }
    
    // Calculate weighted scores
    for (auto& pair : results) {
        auto& result = pair.second;
        result.fused_score = score(result.dense_score, result.sparse_score);
    }
    
    // Convert to vector and sort
    std::vector<FusedResult> output;
    output.reserve(results.size());
    for (const auto& pair : results) {
        output.push_back(pair.second);
    }
    
    // Sort by fused score descending
    std::sort(output.begin(), output.end(),
              [](const FusedResult& a, const FusedResult& b) {
                  return a.fused_score > b.fused_score;
              });
    
    // Limit to top_k
    if (output.size() > top_k) {
        output.resize(top_k);
    }
    
    return output;
}

// LateInteractionFusion implementation

auto LateInteractionFusion::fuse_tokens(const std::vector<std::vector<float>>& dense_token_scores,
                                        const std::vector<std::vector<float>>& sparse_token_scores,
                                        std::size_t top_k) const -> std::vector<FusedResult> {
    
    if (dense_token_scores.empty() || sparse_token_scores.empty()) {
        return {};
    }
    
    // Assume first dimension is tokens, second is documents
    std::size_t n_docs = dense_token_scores[0].size();
    std::vector<FusedResult> results;
    results.reserve(n_docs);
    
    for (std::size_t doc_idx = 0; doc_idx < n_docs; ++doc_idx) {
        FusedResult result;
        result.doc_id = static_cast<std::uint64_t>(doc_idx);
        
        // Extract token scores for this document
        std::vector<float> dense_doc_scores;
        std::vector<float> sparse_doc_scores;
        
        for (std::size_t token_idx = 0; token_idx < dense_token_scores.size(); ++token_idx) {
            if (doc_idx < dense_token_scores[token_idx].size()) {
                dense_doc_scores.push_back(dense_token_scores[token_idx][doc_idx]);
            }
        }
        
        for (std::size_t token_idx = 0; token_idx < sparse_token_scores.size(); ++token_idx) {
            if (doc_idx < sparse_token_scores[token_idx].size()) {
                sparse_doc_scores.push_back(sparse_token_scores[token_idx][doc_idx]);
            }
        }
        
        // Compute MaxSim scores
        if (config_.use_max_sim) {
            result.dense_score = max_sim(dense_doc_scores, dense_doc_scores);
            result.sparse_score = max_sim(sparse_doc_scores, sparse_doc_scores);
            result.fused_score = max_sim(dense_doc_scores, sparse_doc_scores);
        } else {
            // Simple average
            float dense_avg = 0.0f;
            for (float score : dense_doc_scores) {
                dense_avg += score;
            }
            if (!dense_doc_scores.empty()) {
                dense_avg /= dense_doc_scores.size();
            }
            
            float sparse_avg = 0.0f;
            for (float score : sparse_doc_scores) {
                sparse_avg += score;
            }
            if (!sparse_doc_scores.empty()) {
                sparse_avg /= sparse_doc_scores.size();
            }
            
            result.dense_score = dense_avg;
            result.sparse_score = sparse_avg;
            result.fused_score = (dense_avg + sparse_avg) / 2.0f;
        }
        
        // Apply temperature if normalizing
        if (config_.normalize_scores) {
            result.fused_score = std::tanh(result.fused_score / config_.temperature);
        }
        
        results.push_back(result);
    }
    
    // Sort by fused score
    std::sort(results.begin(), results.end(),
              [](const FusedResult& a, const FusedResult& b) {
                  return a.fused_score > b.fused_score;
              });
    
    // Limit to top_k
    if (results.size() > top_k) {
        results.resize(top_k);
    }
    
    return results;
}

auto LateInteractionFusion::max_sim(const std::vector<float>& query_scores,
                                   const std::vector<float>& doc_scores) -> float {
    if (query_scores.empty() || doc_scores.empty()) {
        return 0.0f;
    }
    
    float max_score = 0.0f;
    for (std::size_t i = 0; i < query_scores.size(); ++i) {
        for (std::size_t j = 0; j < doc_scores.size(); ++j) {
            float sim = query_scores[i] * doc_scores[j];
            max_score = std::max(max_score, sim);
        }
    }
    
    return max_score;
}

// AdaptiveFusion implementation

auto AdaptiveFusion::train(const std::vector<TrainingExample>& examples)
    -> std::expected<void, std::string> {
    
    if (examples.empty()) {
        return std::vesper_unexpected(std::string("No training examples provided"));
    }
    
    // Optimize parameters using coordinate descent
    optimize_params(examples);
    
    return {};
}

auto AdaptiveFusion::fuse(const std::vector<SearchResult>& dense_results,
                          const std::vector<SearchResult>& sparse_results,
                          std::size_t top_k) const -> std::vector<FusedResult> {
    
    if (use_rrf_) {
        ReciprocalRankFusion rrf(rrf_k_);
        return rrf.fuse(dense_results, sparse_results, top_k);
    } else {
        WeightedFusion weighted(dense_weight_, sparse_weight_);
        return weighted.fuse(dense_results, sparse_results, top_k);
    }
}

auto AdaptiveFusion::get_params() const -> std::unordered_map<std::string, float> {
    return {
        {"dense_weight", dense_weight_},
        {"sparse_weight", sparse_weight_},
        {"rrf_k", rrf_k_},
        {"use_rrf", use_rrf_ ? 1.0f : 0.0f}
    };
}

auto AdaptiveFusion::optimize_params(const std::vector<TrainingExample>& examples) -> void {
    // Simple grid search for demonstration
    float best_score = 0.0f;
    float best_dense_weight = dense_weight_;
    float best_sparse_weight = sparse_weight_;
    float best_rrf_k = rrf_k_;
    bool best_use_rrf = use_rrf_;
    
    // Try different weight combinations
    for (float dw = 0.0f; dw <= 1.0f; dw += 0.1f) {
        float sw = 1.0f - dw;
        
        // Try both RRF and weighted
        for (int use_rrf_int = 0; use_rrf_int <= 1; ++use_rrf_int) {
            bool try_rrf = (use_rrf_int == 1);
            
            // Try different RRF k values
            for (float k = 10.0f; k <= 100.0f; k += 10.0f) {
                float total_score = 0.0f;
                
                // Evaluate on all examples
                for (const auto& example : examples) {
                    std::vector<FusedResult> results;
                    
                    if (try_rrf) {
                        ReciprocalRankFusion rrf(k);
                        results = rrf.fuse(example.dense_results, example.sparse_results, 10);
                    } else {
                        WeightedFusion weighted(dw, sw);
                        results = weighted.fuse(example.dense_results, example.sparse_results, 10);
                    }
                    
                    // Compute NDCG@10
                    float ndcg = compute_ndcg(results, example.relevant_docs, 10);
                    total_score += ndcg;
                }
                
                float avg_score = total_score / examples.size();
                if (avg_score > best_score) {
                    best_score = avg_score;
                    best_dense_weight = dw;
                    best_sparse_weight = sw;
                    best_rrf_k = k;
                    best_use_rrf = try_rrf;
                }
            }
        }
    }
    
    // Update parameters
    dense_weight_ = best_dense_weight;
    sparse_weight_ = best_sparse_weight;
    rrf_k_ = best_rrf_k;
    use_rrf_ = best_use_rrf;
}

auto AdaptiveFusion::compute_ndcg(const std::vector<FusedResult>& results,
                                  const std::vector<std::uint64_t>& relevant,
                                  std::size_t k) -> float {
    
    float dcg = 0.0f;
    float idcg = 0.0f;
    
    // Create relevance map
    std::unordered_set<std::uint64_t> relevant_set(relevant.begin(), relevant.end());
    
    // Compute DCG
    for (std::size_t i = 0; i < std::min(results.size(), k); ++i) {
        if (relevant_set.count(results[i].doc_id)) {
            dcg += 1.0f / std::log2(i + 2);
        }
    }
    
    // Compute IDCG (ideal DCG)
    std::size_t n_relevant_at_k = std::min(relevant.size(), k);
    for (std::size_t i = 0; i < n_relevant_at_k; ++i) {
        idcg += 1.0f / std::log2(i + 2);
    }
    
    if (idcg == 0.0f) {
        return 0.0f;
    }
    
    return dcg / idcg;
}

} // namespace vesper::search::fusion