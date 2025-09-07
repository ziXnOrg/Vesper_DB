/** \file multi_index_test.cpp
 *  \brief Integration tests for multi-index coordination
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "vesper/index/index_manager.hpp"
#include "vesper/index/hnsw.hpp"
#include "vesper/index/ivf_pq.hpp"
#include "vesper/index/disk_graph.hpp"

#include <random>
#include <chrono>
#include <unordered_set>
#include <numeric>

using namespace vesper;
using namespace vesper::index;

namespace {

// Test data generation
std::vector<float> generate_test_vectors(std::size_t n, std::size_t dim, 
                                        std::uint32_t seed = 42) {
    std::vector<float> vectors(n * dim);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (auto& v : vectors) {
        v = dist(gen);
    }
    
    // Normalize for cosine similarity
    for (std::size_t i = 0; i < n; ++i) {
        float norm = 0.0f;
        for (std::size_t d = 0; d < dim; ++d) {
            norm += vectors[i * dim + d] * vectors[i * dim + d];
        }
        norm = std::sqrt(norm);
        if (norm > 0) {
            for (std::size_t d = 0; d < dim; ++d) {
                vectors[i * dim + d] /= norm;
            }
        }
    }
    
    return vectors;
}

// Compute exact k-NN for verification
std::vector<std::pair<std::uint64_t, float>> compute_exact_knn(
    const float* query, const float* vectors, 
    std::size_t n, std::size_t dim, std::uint32_t k) {
    
    std::vector<std::pair<float, std::uint64_t>> distances;
    distances.reserve(n);
    
    for (std::size_t i = 0; i < n; ++i) {
        float dist = 0.0f;
        for (std::size_t d = 0; d < dim; ++d) {
            float diff = query[d] - vectors[i * dim + d];
            dist += diff * diff;
        }
        distances.emplace_back(dist, i);
    }
    
    std::partial_sort(distances.begin(), 
                     distances.begin() + std::min<std::size_t>(k, n),
                     distances.end());
    
    std::vector<std::pair<std::uint64_t, float>> results;
    for (std::size_t i = 0; i < std::min<std::size_t>(k, n); ++i) {
        results.emplace_back(distances[i].second, distances[i].first);
    }
    
    return results;
}

// Compute recall metric
float compute_recall(const std::vector<std::pair<std::uint64_t, float>>& exact,
                    const std::vector<std::pair<std::uint64_t, float>>& approx) {
    if (exact.empty()) return 0.0f;
    
    std::unordered_set<std::uint64_t> exact_set;
    for (const auto& [id, _] : exact) {
        exact_set.insert(id);
    }
    
    std::size_t hits = 0;
    for (const auto& [id, _] : approx) {
        if (exact_set.count(id)) {
            hits++;
        }
    }
    
    return static_cast<float>(hits) / exact.size();
}

} // anonymous namespace

TEST_CASE("Multi-index: HNSW and IVF-PQ coordination", "[integration][multi-index]") {
    const std::size_t dim = 64;
    const std::size_t n = 10000;
    const std::uint32_t k = 10;
    
    auto vectors = generate_test_vectors(n, dim);
    
    SECTION("Build both HNSW and IVF-PQ indexes") {
        IndexManager manager(dim);
        
        // Build with hybrid strategy to get multiple indexes
        IndexBuildConfig config;
        config.strategy = SelectionStrategy::Hybrid;
        config.memory_budget_mb = 512;
        
        auto build_result = manager.build(vectors.data(), n, config);
        REQUIRE(build_result.has_value());
        
        auto active = manager.get_active_indexes();
        REQUIRE(!active.empty());
        
        // Verify search works
        auto query = generate_test_vectors(1, dim);
        QueryConfig qconfig;
        qconfig.k = k;
        
        auto results = manager.search(query.data(), qconfig);
        REQUIRE(results.has_value());
        REQUIRE(results->size() == k);
        
        // Compute recall
        auto exact = compute_exact_knn(query.data(), vectors.data(), n, dim, k);
        float recall = compute_recall(exact, *results);
        REQUIRE(recall >= 0.7f);  // At least 70% recall
    }
}

TEST_CASE("Multi-index: Direct index comparison", "[integration][multi-index]") {
    const std::size_t dim = 32;
    const std::size_t n = 5000;
    const std::uint32_t k = 20;
    
    auto vectors = generate_test_vectors(n, dim);
    auto query = generate_test_vectors(1, dim);
    
    // Compute exact neighbors for comparison
    auto exact = compute_exact_knn(query.data(), vectors.data(), n, dim, k);
    
    SECTION("HNSW index performance") {
        HnswIndex hnsw(dim);
        
        HnswBuildParams params;
        params.M = 16;
        params.efConstruction = 200;
        params.seed = 42;
        
        auto build_result = hnsw.build(vectors.data(), n, params);
        REQUIRE(build_result.has_value());
        
        HnswSearchParams search_params;
        search_params.efSearch = 100;
        
        auto results = hnsw.search(query.data(), k, search_params);
        REQUIRE(results.has_value());
        REQUIRE(results->size() == k);
        
        float recall = compute_recall(exact, *results);
        REQUIRE(recall >= 0.8f);  // HNSW should have high recall
    }
    
    SECTION("IVF-PQ index performance") {
        IvfPqIndex ivf_pq(dim);
        
        IvfPqBuildParams params;
        params.n_lists = 100;
        params.n_subquantizers = 8;
        params.n_bits = 8;
        params.train_size = std::min<std::size_t>(n, 10000);
        
        // Train the index
        auto train_result = ivf_pq.train(vectors.data(), params.train_size, params);
        REQUIRE(train_result.has_value());
        
        // Build the index
        auto build_result = ivf_pq.build(vectors.data(), n);
        REQUIRE(build_result.has_value());
        
        IvfPqSearchParams search_params;
        search_params.n_probes = 10;
        
        auto results = ivf_pq.search(query.data(), k, search_params);
        REQUIRE(results.has_value());
        REQUIRE(results->size() == k);
        
        float recall = compute_recall(exact, *results);
        REQUIRE(recall >= 0.5f);  // IVF-PQ trades recall for compression
    }
}

TEST_CASE("Multi-index: Incremental updates across indexes", "[integration][multi-index]") {
    const std::size_t dim = 32;
    const std::size_t n_initial = 1000;
    const std::size_t n_batch = 100;
    
    auto initial_vectors = generate_test_vectors(n_initial, dim);
    auto batch_vectors = generate_test_vectors(n_batch, dim, 123);
    
    IndexManager manager(dim);
    
    // Initial build
    IndexBuildConfig config;
    config.strategy = SelectionStrategy::Auto;
    
    auto build_result = manager.build(initial_vectors.data(), n_initial, config);
    REQUIRE(build_result.has_value());
    
    SECTION("Single vector addition") {
        auto new_vector = generate_test_vectors(1, dim, 999);
        auto add_result = manager.add(n_initial, new_vector.data());
        REQUIRE(add_result.has_value());
        
        // Search for the newly added vector
        QueryConfig qconfig;
        qconfig.k = 1;
        
        auto results = manager.search(new_vector.data(), qconfig);
        REQUIRE(results.has_value());
        REQUIRE(!results->empty());
        REQUIRE(results->front().first == n_initial);
        REQUIRE(results->front().second < 0.01f);  // Should find exact match
    }
    
    SECTION("Batch addition") {
        std::vector<std::uint64_t> ids(n_batch);
        std::iota(ids.begin(), ids.end(), n_initial);
        
        auto batch_result = manager.add_batch(ids.data(), batch_vectors.data(), n_batch);
        REQUIRE(batch_result.has_value());
        
        // Verify we can find vectors from the batch
        auto query = generate_test_vectors(1, dim, 123);  // Same seed as batch
        QueryConfig qconfig;
        qconfig.k = 10;
        
        auto results = manager.search(query.data(), qconfig);
        REQUIRE(results.has_value());
        REQUIRE(results->size() == 10);
        
        // At least one result should be from the new batch
        bool found_batch = false;
        for (const auto& [id, _] : *results) {
            if (id >= n_initial) {
                found_batch = true;
                break;
            }
        }
        REQUIRE(found_batch);
    }
}

TEST_CASE("Multi-index: Query planner integration", "[integration][multi-index]") {
    const std::size_t dim = 64;
    const std::size_t n = 8000;
    
    auto vectors = generate_test_vectors(n, dim);
    
    IndexManager manager(dim);
    IndexBuildConfig config;
    config.strategy = SelectionStrategy::Auto;
    config.memory_budget_mb = 256;
    
    auto build_result = manager.build(vectors.data(), n, config);
    REQUIRE(build_result.has_value());
    
    QueryPlanner planner(manager);
    
    SECTION("Planner selects appropriate index") {
        auto query = generate_test_vectors(1, dim);
        
        // Small k query
        QueryConfig config_small;
        config_small.k = 5;
        config_small.use_query_planner = true;
        
        auto plan_small = planner.plan(query.data(), config_small);
        REQUIRE(plan_small.estimated_cost_ms >= 0);
        REQUIRE(plan_small.estimated_recall >= 0);
        
        // Large k query  
        QueryConfig config_large;
        config_large.k = 100;
        config_large.use_query_planner = true;
        
        auto plan_large = planner.plan(query.data(), config_large);
        REQUIRE(plan_large.estimated_cost_ms >= plan_small.estimated_cost_ms);
    }
    
    SECTION("Search with query planner") {
        auto query = generate_test_vectors(1, dim);
        
        QueryConfig config;
        config.k = 10;
        config.use_query_planner = true;
        
        // Search should use the planner
        auto results = manager.search(query.data(), config);
        REQUIRE(results.has_value());
        REQUIRE(results->size() == 10);
        
        // Verify results are reasonable
        for (const auto& [id, dist] : *results) {
            REQUIRE(id < n);
            REQUIRE(dist >= 0);
        }
    }
}

TEST_CASE("Multi-index: Memory management", "[integration][multi-index]") {
    const std::size_t dim = 32;
    const std::size_t n = 5000;
    
    auto vectors = generate_test_vectors(n, dim);
    
    SECTION("Memory budget enforcement") {
        IndexManager manager(dim);
        
        // Set a specific memory budget
        const std::size_t budget_mb = 64;
        auto budget_result = manager.set_memory_budget(budget_mb);
        REQUIRE(budget_result.has_value());
        
        // Build with limited memory
        IndexBuildConfig config;
        config.strategy = SelectionStrategy::Auto;
        config.memory_budget_mb = budget_mb;
        
        auto build_result = manager.build(vectors.data(), n, config);
        REQUIRE(build_result.has_value());
        
        // Check memory usage is within reasonable bounds
        auto memory_used = manager.memory_usage();
        REQUIRE(memory_used > 0);
        REQUIRE(memory_used < budget_mb * 2 * 1024 * 1024);  // Allow 2x for overhead
    }
    
    SECTION("Statistics tracking") {
        IndexManager manager(dim);
        
        IndexBuildConfig config;
        config.strategy = SelectionStrategy::Auto;
        
        auto build_result = manager.build(vectors.data(), n, config);
        REQUIRE(build_result.has_value());
        
        auto stats = manager.get_stats();
        REQUIRE(!stats.empty());
        
        for (const auto& stat : stats) {
            REQUIRE(stat.num_vectors == n);
            REQUIRE(stat.memory_usage_bytes > 0);
            REQUIRE(stat.build_time_ms >= 0);
        }
    }
}

TEST_CASE("Multi-index: Error handling", "[integration][multi-index]") {
    const std::size_t dim = 32;
    
    SECTION("Invalid operations on empty manager") {
        IndexManager manager(dim);
        
        // Search on empty index
        auto query = generate_test_vectors(1, dim);
        QueryConfig config;
        config.k = 10;
        
        auto search_result = manager.search(query.data(), config);
        REQUIRE(!search_result.has_value());
        
        // Add to empty index
        auto vector = generate_test_vectors(1, dim);
        auto add_result = manager.add(0, vector.data());
        REQUIRE(!add_result.has_value());
    }
    
    SECTION("Dimension mismatch handling") {
        const std::size_t n = 100;
        auto vectors = generate_test_vectors(n, dim);
        
        IndexManager manager(dim);
        IndexBuildConfig config;
        config.strategy = SelectionStrategy::Auto;
        
        auto build_result = manager.build(vectors.data(), n, config);
        REQUIRE(build_result.has_value());
        
        // Try to search with wrong dimension
        auto wrong_dim_query = generate_test_vectors(1, dim * 2);
        QueryConfig qconfig;
        qconfig.k = 10;
        
        // This should either fail gracefully or handle the dimension mismatch
        auto search_result = manager.search(wrong_dim_query.data(), qconfig);
        // The implementation should handle this case
    }
}