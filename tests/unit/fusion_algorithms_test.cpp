/** \file fusion_algorithms_test.cpp
 *  \brief Unit tests for search result fusion algorithms.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include "vesper/search/fusion_algorithms.hpp"
#include <algorithm>
#include <random>
#include <unordered_map>
#include <unordered_set>

using namespace vesper::search::fusion;
using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

namespace {

// Helper to create test search results
std::vector<SearchResult> create_dense_results() {
    return {
        {1, 0.95f, 1},
        {3, 0.87f, 2},
        {5, 0.82f, 3},
        {7, 0.76f, 4},
        {9, 0.71f, 5}
    };
}

std::vector<SearchResult> create_sparse_results() {
    return {
        {2, 0.89f, 1},
        {3, 0.85f, 2},  // Overlaps with dense
        {4, 0.78f, 3},
        {7, 0.72f, 4},  // Overlaps with dense
        {8, 0.68f, 5}
    };
}

// Helper to check if results are sorted by score
bool are_results_sorted(const std::vector<FusedResult>& results) {
    for (std::size_t i = 1; i < results.size(); ++i) {
        if (results[i-1].fused_score < results[i].fused_score) {
            return false;
        }
    }
    return true;
}

} // namespace

TEST_CASE("Reciprocal Rank Fusion (RRF)", "[fusion]") {
    ReciprocalRankFusion rrf;
    
    SECTION("Basic RRF with default k=60") {
        auto dense = create_dense_results();
        auto sparse = create_sparse_results();
        
        auto fused = rrf.fuse(dense, sparse, 10);
        
        REQUIRE(!fused.empty());
        REQUIRE(fused.size() <= 10);
        REQUIRE(are_results_sorted(fused));
        
        // Document 3 appears in both lists (rank 2 in both)
        // Should have highest RRF score
        auto doc3_it = std::find_if(fused.begin(), fused.end(),
            [](const FusedResult& r) { return r.doc_id == 3; });
        REQUIRE(doc3_it != fused.end());
        
        // Document 3 should rank high (likely first)
        auto doc3_pos = std::distance(fused.begin(), doc3_it);
        REQUIRE(doc3_pos <= 1);  // Top 2 positions
    }
    
    SECTION("RRF with custom k parameter") {
        ReciprocalRankFusion rrf_k10(10.0f);
        ReciprocalRankFusion rrf_k100(100.0f);
        
        auto dense = create_dense_results();
        auto sparse = create_sparse_results();
        
        auto fused_k10 = rrf_k10.fuse(dense, sparse, 10);
        auto fused_k100 = rrf_k100.fuse(dense, sparse, 10);
        
        // Different k values should produce different scores
        REQUIRE(fused_k10[0].fused_score != fused_k100[0].fused_score);
        
        // Smaller k makes rank differences more pronounced
        // Larger k makes scores more uniform
    }
    
    SECTION("RRF with non-overlapping results") {
        std::vector<SearchResult> dense = {
            {1, 0.9f, 1},
            {2, 0.8f, 2},
            {3, 0.7f, 3}
        };
        
        std::vector<SearchResult> sparse = {
            {4, 0.85f, 1},
            {5, 0.75f, 2},
            {6, 0.65f, 3}
        };
        
        auto fused = rrf.fuse(dense, sparse, 10);
        
        REQUIRE(fused.size() == 6);  // All documents included
        REQUIRE(are_results_sorted(fused));
        
        // Top-ranked documents from each list should rank high
        auto doc1_it = std::find_if(fused.begin(), fused.end(),
            [](const FusedResult& r) { return r.doc_id == 1; });
        auto doc4_it = std::find_if(fused.begin(), fused.end(),
            [](const FusedResult& r) { return r.doc_id == 4; });
        
        REQUIRE(doc1_it != fused.end());
        REQUIRE(doc4_it != fused.end());
        
        // Both should be in top positions
        auto doc1_pos = std::distance(fused.begin(), doc1_it);
        auto doc4_pos = std::distance(fused.begin(), doc4_it);
        REQUIRE(doc1_pos <= 2);
        REQUIRE(doc4_pos <= 2);
    }
    
    SECTION("RRF score calculation") {
        ReciprocalRankFusion rrf(60.0f);
        
        // Document ranked 1st in both lists
        float score_1_1 = rrf.score(1, 1);
        REQUIRE(score_1_1 == Catch::Approx(2.0f / 61.0f));
        
        // Document ranked 1st in one, absent in other
        float score_1_0 = rrf.score(1, 0);
        REQUIRE(score_1_0 == Catch::Approx(1.0f / 61.0f));
        
        // Document ranked 2nd and 3rd
        float score_2_3 = rrf.score(2, 3);
        REQUIRE(score_2_3 == Catch::Approx(1.0f/62.0f + 1.0f/63.0f));
    }
}

TEST_CASE("Weighted Fusion", "[fusion]") {
    SECTION("Equal weights (0.5, 0.5)") {
        WeightedFusion fusion(0.5f, 0.5f);
        
        auto dense = create_dense_results();
        auto sparse = create_sparse_results();
        
        auto fused = fusion.fuse(dense, sparse, 10);
        
        REQUIRE(!fused.empty());
        REQUIRE(are_results_sorted(fused));
        
        // With equal weights, overlapping documents should score well
        auto doc3_it = std::find_if(fused.begin(), fused.end(),
            [](const FusedResult& r) { return r.doc_id == 3; });
        REQUIRE(doc3_it != fused.end());
    }
    
    SECTION("Dense-biased weights (0.8, 0.2)") {
        WeightedFusion fusion(0.8f, 0.2f);
        
        auto dense = create_dense_results();
        auto sparse = create_sparse_results();
        
        auto fused = fusion.fuse(dense, sparse, 10);
        
        // Document 1 (top dense result) should rank very high
        REQUIRE(fused[0].doc_id == 1);
    }
    
    SECTION("Sparse-biased weights (0.2, 0.8)") {
        WeightedFusion fusion(0.2f, 0.8f);
        
        auto dense = create_dense_results();
        auto sparse = create_sparse_results();
        
        auto fused = fusion.fuse(dense, sparse, 10);
        
        // Document 2 (top sparse result) should rank very high
        auto doc2_pos = std::find_if(fused.begin(), fused.end(),
            [](const FusedResult& r) { return r.doc_id == 2; });
        REQUIRE(doc2_pos != fused.end());
        REQUIRE(std::distance(fused.begin(), doc2_pos) == 0);  // First position
    }
    
    SECTION("Weight normalization") {
        // Weights don't sum to 1.0
        WeightedFusion fusion(2.0f, 3.0f);
        
        auto dense = create_dense_results();
        auto sparse = create_sparse_results();
        
        auto fused = fusion.fuse(dense, sparse, 10);
        
        // Should still produce valid results (weights normalized internally)
        REQUIRE(!fused.empty());
        for (const auto& result : fused) {
            REQUIRE(result.fused_score >= 0.0f);
            REQUIRE(result.fused_score <= 1.0f);
        }
    }
    
    SECTION("Single source results") {
        WeightedFusion fusion(0.5f, 0.5f);
        
        auto dense = create_dense_results();
        std::vector<SearchResult> empty_sparse;
        
        auto fused = fusion.fuse(dense, empty_sparse, 10);
        
        REQUIRE(fused.size() == dense.size());
        
        // All results should come from dense
        for (const auto& result : fused) {
            REQUIRE(result.dense_score > 0.0f);
            REQUIRE(result.sparse_score == 0.0f);
        }
    }
}

TEST_CASE("Late Interaction Fusion", "[fusion]") {
    LateInteractionFusion::Config config;
    config.use_max_sim = true;
    config.normalize_scores = true;
    config.temperature = 1.0f;
    
    LateInteractionFusion fusion(config);
    
    SECTION("Token-level fusion with MaxSim") {
        // Mock token scores for 3 documents and 2 query tokens
        std::vector<std::vector<float>> dense_token_scores = {
            {0.8f, 0.6f, 0.4f},  // Token 1 scores for docs 0,1,2
            {0.7f, 0.9f, 0.5f}   // Token 2 scores for docs 0,1,2
        };
        
        std::vector<std::vector<float>> sparse_token_scores = {
            {0.9f, 0.5f, 0.3f},  // Token 1 scores for docs 0,1,2
            {0.6f, 0.8f, 0.7f}   // Token 2 scores for docs 0,1,2
        };
        
        auto fused = fusion.fuse_tokens(dense_token_scores, sparse_token_scores, 3);
        
        REQUIRE(fused.size() == 3);
        REQUIRE(are_results_sorted(fused));
        
        // All documents should have scores
        for (const auto& result : fused) {
            REQUIRE(result.fused_score > 0.0f);
        }
    }
    
    SECTION("MaxSim calculation") {
        std::vector<float> query_scores = {0.8f, 0.6f, 0.4f};
        std::vector<float> doc_scores = {0.7f, 0.9f, 0.5f};
        
        float max_sim = LateInteractionFusion::max_sim(query_scores, doc_scores);
        
        // Should be max of all pairwise products
        float expected = 0.0f;
        for (float q : query_scores) {
            for (float d : doc_scores) {
                expected = std::max(expected, q * d);
            }
        }
        
        REQUIRE(max_sim == Catch::Approx(expected));
    }
    
    SECTION("Temperature scaling") {
        LateInteractionFusion::Config config_low_temp;
        config_low_temp.temperature = 0.5f;
        config_low_temp.normalize_scores = true;
        
        LateInteractionFusion::Config config_high_temp;
        config_high_temp.temperature = 2.0f;
        config_high_temp.normalize_scores = true;
        
        LateInteractionFusion fusion_low(config_low_temp);
        LateInteractionFusion fusion_high(config_high_temp);
        
        std::vector<std::vector<float>> token_scores = {
            {0.8f, 0.6f},
            {0.7f, 0.9f}
        };
        
        auto fused_low = fusion_low.fuse_tokens(token_scores, token_scores, 2);
        auto fused_high = fusion_high.fuse_tokens(token_scores, token_scores, 2);
        
        // Different temperatures should produce different scores
        REQUIRE(fused_low[0].fused_score != fused_high[0].fused_score);
    }
}

TEST_CASE("Adaptive Fusion", "[fusion]") {
    AdaptiveFusion fusion;
    
    SECTION("Training with examples") {
        std::vector<AdaptiveFusion::TrainingExample> examples;
        
        // Create training example
        AdaptiveFusion::TrainingExample ex1;
        ex1.dense_results = {
            {1, 0.9f, 1},
            {2, 0.8f, 2},
            {3, 0.7f, 3}
        };
        ex1.sparse_results = {
            {2, 0.85f, 1},
            {3, 0.75f, 2},
            {4, 0.65f, 3}
        };
        ex1.relevant_docs = {2, 3};  // Documents 2 and 3 are relevant
        
        examples.push_back(ex1);
        
        auto train_result = fusion.train(examples);
        REQUIRE(train_result.has_value());
        
        // After training, fusion should work
        auto fused = fusion.fuse(ex1.dense_results, ex1.sparse_results, 3);
        REQUIRE(!fused.empty());
        
        // Relevant documents should rank high
        bool doc2_found = false;
        bool doc3_found = false;
        for (std::size_t i = 0; i < 2 && i < fused.size(); ++i) {
            if (fused[i].doc_id == 2) doc2_found = true;
            if (fused[i].doc_id == 3) doc3_found = true;
        }
        REQUIRE((doc2_found || doc3_found));  // At least one relevant doc in top 2
    }
    
    SECTION("Get learned parameters") {
        auto params = fusion.get_params();
        
        REQUIRE(params.count("dense_weight") > 0);
        REQUIRE(params.count("sparse_weight") > 0);
        REQUIRE(params.count("rrf_k") > 0);
        REQUIRE(params.count("use_rrf") > 0);
        
        // Weights should be in valid range
        REQUIRE(params["dense_weight"] >= 0.0f);
        REQUIRE(params["dense_weight"] <= 1.0f);
        REQUIRE(params["sparse_weight"] >= 0.0f);
        REQUIRE(params["sparse_weight"] <= 1.0f);
    }
    
    SECTION("Fusion without training uses defaults") {
        auto dense = create_dense_results();
        auto sparse = create_sparse_results();
        
        auto fused = fusion.fuse(dense, sparse, 5);
        
        REQUIRE(!fused.empty());
        REQUIRE(fused.size() <= 5);
        REQUIRE(are_results_sorted(fused));
    }
}

TEST_CASE("Fusion utility functions", "[fusion]") {
    SECTION("Score normalization") {
        std::vector<float> scores = {10.0f, 5.0f, 15.0f, 2.0f, 8.0f};
        
        auto normalized = normalize_scores(scores);
        
        REQUIRE(normalized.size() == scores.size());
        
        // Check all scores are in [0, 1]
        for (float score : normalized) {
            REQUIRE(score >= 0.0f);
            REQUIRE(score <= 1.0f);
        }
        
        // Min should map to 0, max to 1
        auto min_it = std::min_element(scores.begin(), scores.end());
        auto max_it = std::max_element(scores.begin(), scores.end());
        std::size_t min_idx = std::distance(scores.begin(), min_it);
        std::size_t max_idx = std::distance(scores.begin(), max_it);
        
        REQUIRE(normalized[min_idx] == Catch::Approx(0.0f));
        REQUIRE(normalized[max_idx] == Catch::Approx(1.0f));
    }
    
    SECTION("Softmax transformation") {
        std::vector<float> scores = {1.0f, 2.0f, 3.0f};
        
        auto softmax_scores = softmax(scores, 1.0f);
        
        REQUIRE(softmax_scores.size() == scores.size());
        
        // Sum should be approximately 1
        float sum = 0.0f;
        for (float score : softmax_scores) {
            sum += score;
            REQUIRE(score >= 0.0f);
            REQUIRE(score <= 1.0f);
        }
        REQUIRE(sum == Catch::Approx(1.0f));
        
        // Higher original scores should have higher softmax scores
        REQUIRE(softmax_scores[2] > softmax_scores[1]);
        REQUIRE(softmax_scores[1] > softmax_scores[0]);
    }
    
    SECTION("Temperature effects on softmax") {
        std::vector<float> scores = {1.0f, 2.0f, 3.0f};
        
        auto soft_low_temp = softmax(scores, 0.5f);   // Low temperature
        auto soft_high_temp = softmax(scores, 2.0f);  // High temperature
        
        // Low temperature makes distribution more peaked
        // High temperature makes it more uniform
        float range_low = soft_low_temp[2] - soft_low_temp[0];
        float range_high = soft_high_temp[2] - soft_high_temp[0];
        
        REQUIRE(range_low > range_high);
    }
    
    SECTION("Merge and deduplicate results") {
        std::vector<SearchResult> list1 = {
            {1, 0.9f, 1},
            {2, 0.8f, 2},
            {3, 0.7f, 3}
        };
        
        std::vector<SearchResult> list2 = {
            {2, 0.85f, 1},  // Duplicate with higher score
            {4, 0.75f, 2},
            {5, 0.65f, 3}
        };
        
        std::vector<std::vector<SearchResult>> lists = {list1, list2};
        auto merged = merge_results(lists);
        
        // Should have 5 unique documents
        REQUIRE(merged.size() == 5);
        
        // Should be sorted by score
        REQUIRE(are_results_sorted({merged.begin(), merged.end()}));
        
        // Document 2 should have the higher score (0.85)
        auto doc2_it = std::find_if(merged.begin(), merged.end(),
            [](const SearchResult& r) { return r.doc_id == 2; });
        REQUIRE(doc2_it != merged.end());
        REQUIRE(doc2_it->score == Catch::Approx(0.85f));
    }
}