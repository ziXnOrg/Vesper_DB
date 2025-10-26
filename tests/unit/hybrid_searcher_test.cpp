/** \file hybrid_searcher_test.cpp
 *  \brief Unit tests for hybrid sparse-dense search.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/catch_approx.hpp>


#include "vesper/search/hybrid_searcher.hpp"
#include "vesper/index/index_manager.hpp"
#include "vesper/index/bm25.hpp"
#include <memory>
#include <random>
#include <thread>
#include <chrono>
#include <sstream>

using namespace vesper;
using namespace vesper::search;
using namespace vesper::index;
using Catch::Matchers::WithinRel;

namespace {

// Helper to create mock dense embeddings
std::vector<float> create_random_embedding(std::size_t dim, std::uint32_t seed) {
    std::vector<float> embedding(dim);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (auto& val : embedding) {
        val = dist(gen);
    }

    // Normalize for cosine similarity
    float norm = 0.0f;
    for (float val : embedding) {
        norm += val * val;
    }
    norm = std::sqrt(norm);
    for (auto& val : embedding) {
        val /= norm;
    }

    return embedding;
}

// Using real IndexManager for dense side in tests (small corpus)
// No mock subclassing: IndexManager methods are not virtual.

// Create test corpus
void setup_test_corpus(BM25Index& bm25, IndexManager& dense) {
    // Add documents with both text and embeddings
    std::vector<std::pair<std::string, std::uint32_t>> docs = {
        {"machine learning algorithms neural networks", 1},
        {"deep learning transformer models attention", 2},
        {"natural language processing bert gpt", 3},
        {"computer vision image classification cnn", 4},
        {"reinforcement learning policy gradient", 5}
    };

    // Collect embeddings for dense index build
    std::vector<float> all_embeddings;
    all_embeddings.reserve(docs.size() * 128);

    for (const auto& [text, seed] : docs) {
        std::uint64_t id = seed - 1;

        // Add to BM25 (takes string_view directly)
        bm25.add_document(id, text);

        // Generate embedding and collect for batch build
        auto embedding = create_random_embedding(128, seed);
        all_embeddings.insert(all_embeddings.end(), embedding.begin(), embedding.end());
    }

    // Build a small in-memory dense index (HNSW) from collected embeddings
    IndexBuildConfig config;
    config.strategy = SelectionStrategy::Auto; // small n -> HNSW
    auto build_ok = dense.build(all_embeddings.data(), docs.size(), config);
    REQUIRE(build_ok.has_value());
}

} // namespace

TEST_CASE("HybridSearcher construction", "[hybrid]") {
    auto dense_index = std::make_shared<IndexManager>(128);
    auto sparse_index = std::make_shared<BM25Index>();

    SECTION("Basic construction") {
        HybridSearcher searcher(dense_index, sparse_index);
        REQUIRE(searcher.is_initialized());
    }

    SECTION("Get default statistics") {
        HybridSearcher searcher(dense_index, sparse_index);
        auto stats = searcher.get_stats();

        REQUIRE(stats.total_queries == 0);
        REQUIRE(stats.dense_searches == 0);
        REQUIRE(stats.sparse_searches == 0);
        REQUIRE(stats.batch_queries == 0);
    }
}

TEST_CASE("HybridSearcher query routing", "[hybrid]") {
    auto dense_index = std::make_shared<IndexManager>(128);
    auto sparse_index = std::make_shared<BM25Index>();
    setup_test_corpus(*sparse_index, *dense_index);

    HybridSearcher searcher(dense_index, sparse_index);

    SECTION("AUTO mode with text only") {
        HybridQuery query;
        query.text = "machine learning";

        HybridSearchConfig config;
        config.query_strategy = QueryStrategy::AUTO;
        config.k = 3;

        auto results = searcher.search(query, config);
        REQUIRE(results.has_value());
        REQUIRE(!results->empty());

        // Should use sparse search
        auto stats = searcher.get_stats();
        REQUIRE(stats.sparse_searches > 0);
    }

    SECTION("AUTO mode with embedding only") {
        HybridQuery query;
        query.dense_embedding = create_random_embedding(128, 42);

        HybridSearchConfig config;
        config.query_strategy = QueryStrategy::AUTO;
        config.k = 3;

        auto results = searcher.search(query, config);
        REQUIRE(results.has_value());
        REQUIRE(!results->empty());

        // Should use dense search
        auto stats = searcher.get_stats();
        REQUIRE(stats.dense_searches > 0);
    }

    SECTION("AUTO mode with both text and embedding") {
        HybridQuery query;
        query.text = "neural networks";
        query.dense_embedding = create_random_embedding(128, 42);

        HybridSearchConfig config;
        config.query_strategy = QueryStrategy::AUTO;
        config.k = 3;

        auto results = searcher.search(query, config);
        REQUIRE(results.has_value());
        REQUIRE(!results->empty());

        // Should use both searches
        auto stats = searcher.get_stats();
        REQUIRE(stats.dense_searches > 0);
        REQUIRE(stats.sparse_searches > 0);
    }

    SECTION("DENSE_FIRST strategy") {
        HybridQuery query;
        query.text = "learning";
        query.dense_embedding = create_random_embedding(128, 42);

        HybridSearchConfig config;
        config.query_strategy = QueryStrategy::DENSE_FIRST;
        config.k = 3;

        auto results = searcher.search(query, config);
        REQUIRE(results.has_value());
        REQUIRE(!results->empty());

        // Dense search should be executed
        for (const auto& result : *results) {
            const bool any = (result.dense_score > 0.0f) || (result.sparse_score > 0.0f);
            REQUIRE(any);
        }
    }

    SECTION("SPARSE_FIRST strategy") {
        HybridQuery query;
        query.text = "transformer attention";
        query.dense_embedding = create_random_embedding(128, 42);

        HybridSearchConfig config;
        config.query_strategy = QueryStrategy::SPARSE_FIRST;
        config.k = 3;

        auto results = searcher.search(query, config);
        REQUIRE(results.has_value());
        REQUIRE(!results->empty());

        // Sparse search should be executed
        bool has_sparse_results = false;
        for (const auto& result : *results) {
            if (result.sparse_score > 0.0f) {
                has_sparse_results = true;
                break;
            }
        }
        REQUIRE(has_sparse_results);
    }
}

TEST_CASE("HybridSearcher fusion algorithms", "[hybrid]") {
    auto dense_index = std::make_shared<IndexManager>(128);
    auto sparse_index = std::make_shared<BM25Index>();
    setup_test_corpus(*sparse_index, *dense_index);

    HybridSearcher searcher(dense_index, sparse_index);

    HybridQuery query;
    query.text = "machine learning neural";
    query.dense_embedding = create_random_embedding(128, 42);

    SECTION("Reciprocal Rank Fusion") {
        HybridSearchConfig config;
        config.fusion_algorithm = FusionAlgorithm::RECIPROCAL_RANK;
        config.rrf_k = 60.0f;
        config.k = 5;

        auto results = searcher.search(query, config);
        REQUIRE(results.has_value());
        REQUIRE(!results->empty());

        // Check fused scores are computed
        for (const auto& result : *results) {
            REQUIRE(result.fused_score > 0.0f);
        }

        // Results should be sorted by fused score
        for (std::size_t i = 1; i < results->size(); ++i) {
            REQUIRE((*results)[i-1].fused_score >= (*results)[i].fused_score);
        }
    }

    SECTION("Weighted Sum Fusion") {
        HybridSearchConfig config;
        config.fusion_algorithm = FusionAlgorithm::WEIGHTED_SUM;
        config.dense_weight = 0.7f;
        config.sparse_weight = 0.3f;
        config.k = 5;

        auto results = searcher.search(query, config);
        REQUIRE(results.has_value());
        REQUIRE(!results->empty());

        // Fused scores should reflect weighting
        for (const auto& result : *results) {
            if (result.dense_score > 0 && result.sparse_score > 0) {
                // Rough check that fusion happened
                REQUIRE(result.fused_score > 0.0f);
                REQUIRE(result.fused_score <= 1.0f);
            }
        }
    }

    SECTION("MAX_SCORE Fusion") {
        HybridSearchConfig config;
        config.fusion_algorithm = FusionAlgorithm::MAX_SCORE;
        config.k = 5;

        auto results = searcher.search(query, config);
        REQUIRE(results.has_value());
        REQUIRE(!results->empty());

        // Fused score should be max of dense and sparse
        for (const auto& result : *results) {
            float expected_fused = std::max(result.dense_score, result.sparse_score);
            REQUIRE(result.fused_score == Catch::Approx(expected_fused).margin(0.001f));
        }
    }
}

TEST_CASE("HybridSearcher batch operations", "[hybrid]") {
    auto dense_index = std::make_shared<IndexManager>(128);
    auto sparse_index = std::make_shared<BM25Index>();
    setup_test_corpus(*sparse_index, *dense_index);

    HybridSearcher searcher(dense_index, sparse_index);

    SECTION("Batch search") {
        std::vector<HybridQuery> queries;

        // Create multiple queries
        for (int i = 0; i < 3; ++i) {
            HybridQuery query;
            query.text = (i == 0) ? "machine learning" :
                        (i == 1) ? "computer vision" : "neural networks";
            query.dense_embedding = create_random_embedding(128, 100 + i);
            queries.push_back(query);
        }

        HybridSearchConfig config;
        config.k = 3;

        auto results = searcher.batch_search(queries, config);
        REQUIRE(results.has_value());
        REQUIRE(results->size() == 3);

        // Each query should have results
        for (const auto& query_results : *results) {
            REQUIRE(!query_results.empty());
            REQUIRE(query_results.size() <= 3);
        }

        // Stats should reflect batch processing
        auto stats = searcher.get_stats();
        REQUIRE(stats.batch_queries > 0);
        REQUIRE(stats.total_queries >= 3);
    }
}

TEST_CASE("HybridSearcher with filters", "[hybrid]") {
    auto dense_index = std::make_shared<IndexManager>(128);
    auto sparse_index = std::make_shared<BM25Index>();
    setup_test_corpus(*sparse_index, *dense_index);

    HybridSearcher searcher(dense_index, sparse_index);

    SECTION("Search with document filter") {
        HybridQuery query;
        query.text = "learning";
        query.dense_embedding = create_random_embedding(128, 42);

        // Create filter that only includes docs 0, 1, 2
        auto filter = std::make_shared<roaring::Roaring>();
        filter->add(0);
        filter->add(1);
        filter->add(2);
        query.filter = filter;

        HybridSearchConfig config;
        config.k = 5;

        auto results = searcher.search(query, config);
        REQUIRE(results.has_value());

        // All results should be from filtered set
        for (const auto& result : *results) {
            REQUIRE(filter->contains(static_cast<uint32_t>(result.doc_id)));
        }
    }
}

TEST_CASE("HybridSearcher statistics tracking", "[hybrid]") {
    auto dense_index = std::make_shared<IndexManager>(128);
    auto sparse_index = std::make_shared<BM25Index>();
    setup_test_corpus(*sparse_index, *dense_index);

    HybridSearcher searcher(dense_index, sparse_index);

    SECTION("Query counting") {
        auto initial_stats = searcher.get_stats();
        auto initial_count = initial_stats.total_queries.load();

        HybridQuery query;
        query.text = "test";
        HybridSearchConfig config;
        config.k = 1;

        // Execute multiple queries
        for (int i = 0; i < 5; ++i) {
            searcher.search(query, config);
        }

        auto final_stats = searcher.get_stats();
        REQUIRE(final_stats.total_queries.load() == initial_count + 5);
    }

    SECTION("Search type tracking") {
        HybridQuery dense_query;
        dense_query.dense_embedding = create_random_embedding(128, 42);

        HybridQuery sparse_query;
        sparse_query.text = "test";

        HybridSearchConfig config;
        config.k = 1;

        auto initial_stats = searcher.get_stats();
        auto initial_dense = initial_stats.dense_searches.load();
        auto initial_sparse = initial_stats.sparse_searches.load();

        searcher.search(dense_query, config);
        searcher.search(sparse_query, config);

        auto final_stats = searcher.get_stats();
        REQUIRE(final_stats.dense_searches.load() > initial_dense);
        REQUIRE(final_stats.sparse_searches.load() > initial_sparse);
    }

    SECTION("Latency tracking") {
        HybridQuery query;
        query.text = "test";
        HybridSearchConfig config;
        config.k = 1;

        auto initial_stats = searcher.get_stats();
        auto initial_latency = initial_stats.total_latency_us.load();

        searcher.search(query, config);

        auto final_stats = searcher.get_stats();
        REQUIRE(final_stats.total_latency_us.load() > initial_latency);
    }
}

TEST_CASE("HybridSearcher error handling", "[hybrid]") {
    auto dense_index = std::make_shared<IndexManager>(128);
    auto sparse_index = std::make_shared<BM25Index>();

    HybridSearcher searcher(dense_index, sparse_index);

    SECTION("Empty query") {
        HybridQuery query;  // No text or embedding
        HybridSearchConfig config;

        auto results = searcher.search(query, config);
        REQUIRE(!results.has_value());  // Should return error
    }

    SECTION("Invalid strategy for query type") {
        HybridQuery query;
        query.text = "test";  // Only text, no embedding

        HybridSearchConfig config;
        config.query_strategy = QueryStrategy::DENSE_FIRST;  // Requires embedding

        auto results = searcher.search(query, config);
        REQUIRE(!results.has_value());  // Should return error
    }
}

TEST_CASE("HybridSearcher reranking", "[hybrid]") {
    auto dense_index = std::make_shared<IndexManager>(128);
    auto sparse_index = std::make_shared<BM25Index>();
    setup_test_corpus(*sparse_index, *dense_index);

    HybridSearcher searcher(dense_index, sparse_index);

    SECTION("Rerank factor effect") {
        HybridQuery query;
        query.text = "machine learning";
        query.dense_embedding = create_random_embedding(128, 42);

        HybridSearchConfig config;
        config.k = 3;
        config.rerank_factor = 5;  // Fetch 15 candidates, return top 3

        auto results = searcher.search(query, config);
        REQUIRE(results.has_value());
        REQUIRE(results->size() <= 3);

        // With more candidates, should get better results
        // (Can't directly test quality without ground truth)
    }
}
