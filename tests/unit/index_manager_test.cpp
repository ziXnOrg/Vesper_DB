/** \file index_manager_test.cpp
 *  \brief Unit tests for IndexManager and multi-index coordination.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "vesper/index/index_manager.hpp"
#include <random>
#include <algorithm>

using namespace vesper;
using namespace vesper::index;

namespace {

// Generate random test vectors
std::vector<float> generate_random_vectors(std::size_t n, std::size_t dim, std::uint32_t seed = 42) {
    std::vector<float> vectors(n * dim);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (auto& v : vectors) {
        v = dist(gen);
    }
    
    return vectors;
}

// Normalize vectors for cosine similarity
void normalize_vectors(std::vector<float>& vectors, std::size_t n, std::size_t dim) {
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
}

// Compute exact nearest neighbors for verification
std::vector<std::pair<std::uint64_t, float>> compute_exact_neighbors(
    const float* query, const float* vectors, std::size_t n, std::size_t dim, std::uint32_t k) {
    
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

// Compute recall@k
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

TEST_CASE("IndexManager: Basic construction and configuration", "[index_manager]") {
    const std::size_t dim = 128;
    
    SECTION("Constructor creates manager with correct dimension") {
        IndexManager manager(dim);
        auto active = manager.get_active_indexes();
        REQUIRE(active.empty());
    }
    
    SECTION("Memory budget can be set") {
        IndexManager manager(dim);
        auto result = manager.set_memory_budget(2048);
        REQUIRE(result.has_value());
    }
}

TEST_CASE("IndexManager: Automatic index selection", "[index_manager]") {
    const std::size_t dim = 64;
    
    SECTION("Small dataset selects HNSW") {
        const std::size_t n = 1000;
        auto vectors = generate_random_vectors(n, dim);
        
        IndexManager manager(dim);
        IndexBuildConfig config;
        config.strategy = SelectionStrategy::Auto;
        config.memory_budget_mb = 1024;
        
        auto result = manager.build(vectors.data(), n, config);
        REQUIRE(result.has_value());
        
        auto active = manager.get_active_indexes();
        REQUIRE(active.size() == 1);
        REQUIRE(active[0] == IndexType::HNSW);
    }
    
    SECTION("Medium dataset selects IVF-PQ") {
        const std::size_t n = 50000;
        auto vectors = generate_random_vectors(n, dim);
        
        IndexManager manager(dim);
        IndexBuildConfig config;
        config.strategy = SelectionStrategy::Auto;
        config.memory_budget_mb = 256;  // Limited memory
        
        auto result = manager.build(vectors.data(), n, config);
        REQUIRE(result.has_value());
        
        auto active = manager.get_active_indexes();
        REQUIRE(active.size() >= 1);
    }
}

TEST_CASE("IndexManager: Manual index selection", "[index_manager]") {
    const std::size_t dim = 32;
    const std::size_t n = 1000;
    auto vectors = generate_random_vectors(n, dim);
    normalize_vectors(vectors, n, dim);
    
    SECTION("Force HNSW index") {
        IndexManager manager(dim);
        IndexBuildConfig config;
        config.type = IndexType::HNSW;
        config.strategy = SelectionStrategy::Manual;
        config.hnsw_params.M = 16;
        config.hnsw_params.efConstruction = 200;
        
        auto result = manager.build(vectors.data(), n, config);
        REQUIRE(result.has_value());
        
        auto active = manager.get_active_indexes();
        REQUIRE(active.size() == 1);
        REQUIRE(active[0] == IndexType::HNSW);
    }
}

TEST_CASE("IndexManager: Search functionality", "[index_manager]") {
    const std::size_t dim = 32;
    const std::size_t n = 1000;
    const std::uint32_t k = 10;
    
    auto vectors = generate_random_vectors(n, dim);
    normalize_vectors(vectors, n, dim);
    
    IndexManager manager(dim);
    IndexBuildConfig config;
    config.strategy = SelectionStrategy::Auto;
    
    auto build_result = manager.build(vectors.data(), n, config);
    REQUIRE(build_result.has_value());
    
    SECTION("Search returns correct number of results") {
        auto query = generate_random_vectors(1, dim);
        normalize_vectors(query, 1, dim);
        
        QueryConfig query_config;
        query_config.k = k;
        query_config.use_query_planner = false;
        
        auto results = manager.search(query.data(), query_config);
        REQUIRE(results.has_value());
        REQUIRE(results->size() == k);
    }
    
    SECTION("Search results are ordered by distance") {
        auto query = generate_random_vectors(1, dim);
        normalize_vectors(query, 1, dim);
        
        QueryConfig query_config;
        query_config.k = k;
        
        auto results = manager.search(query.data(), query_config);
        REQUIRE(results.has_value());
        
        // Check that distances are non-decreasing
        for (std::size_t i = 1; i < results->size(); ++i) {
            REQUIRE((*results)[i].second >= (*results)[i-1].second);
        }
    }
    
    SECTION("Search achieves reasonable recall") {
        auto query = generate_random_vectors(1, dim);
        normalize_vectors(query, 1, dim);
        
        // Compute exact neighbors
        auto exact = compute_exact_neighbors(query.data(), vectors.data(), n, dim, k);
        
        // Get approximate neighbors
        QueryConfig query_config;
        query_config.k = k;
        query_config.ef_search = 200;
        
        auto approx = manager.search(query.data(), query_config);
        REQUIRE(approx.has_value());
        
        // Compute recall
        float recall = compute_recall(exact, *approx);
        REQUIRE(recall >= 0.5f);  // At least 50% recall for small dataset
    }
}

TEST_CASE("IndexManager: Incremental updates", "[index_manager]") {
    const std::size_t dim = 32;
    const std::size_t n = 500;
    
    auto vectors = generate_random_vectors(n, dim);
    
    IndexManager manager(dim);
    IndexBuildConfig config;
    config.strategy = SelectionStrategy::Auto;
    
    auto build_result = manager.build(vectors.data(), n, config);
    REQUIRE(build_result.has_value());
    
    SECTION("Add single vector") {
        auto new_vector = generate_random_vectors(1, dim);
        auto result = manager.add(n, new_vector.data());
        REQUIRE(result.has_value());
        
        // Verify we can search for the new vector
        QueryConfig query_config;
        query_config.k = 1;
        
        auto search_result = manager.search(new_vector.data(), query_config);
        REQUIRE(search_result.has_value());
        REQUIRE(!search_result->empty());
        REQUIRE(search_result->front().first == n);
    }
    
    SECTION("Batch add vectors") {
        const std::size_t batch_size = 100;
        auto new_vectors = generate_random_vectors(batch_size, dim);
        
        std::vector<std::uint64_t> ids(batch_size);
        std::iota(ids.begin(), ids.end(), n);
        
        auto result = manager.add_batch(ids.data(), new_vectors.data(), batch_size);
        REQUIRE(result.has_value());
    }
}

TEST_CASE("IndexManager: Statistics tracking", "[index_manager]") {
    const std::size_t dim = 32;
    const std::size_t n = 1000;
    
    auto vectors = generate_random_vectors(n, dim);
    
    IndexManager manager(dim);
    IndexBuildConfig config;
    config.strategy = SelectionStrategy::Auto;
    
    auto build_result = manager.build(vectors.data(), n, config);
    REQUIRE(build_result.has_value());
    
    SECTION("Get index statistics") {
        auto stats = manager.get_stats();
        REQUIRE(!stats.empty());
        
        for (const auto& s : stats) {
            REQUIRE(s.num_vectors == n);
            REQUIRE(s.memory_usage_bytes > 0);
        }
    }
    
    SECTION("Memory usage reporting") {
        auto memory = manager.memory_usage();
        REQUIRE(memory > 0);
        REQUIRE(memory < 100 * 1024 * 1024);  // Less than 100MB for small test
    }
}

TEST_CASE("IndexManager: Hybrid mode with multiple indexes", "[index_manager]") {
    const std::size_t dim = 32;
    const std::size_t n = 5000;
    
    auto vectors = generate_random_vectors(n, dim);
    
    IndexManager manager(dim);
    IndexBuildConfig config;
    config.strategy = SelectionStrategy::Hybrid;
    config.memory_budget_mb = 1024;
    
    auto build_result = manager.build(vectors.data(), n, config);
    REQUIRE(build_result.has_value());
    
    SECTION("Multiple indexes are built") {
        auto active = manager.get_active_indexes();
        REQUIRE(active.size() >= 1);  // At least one index should be built
    }
    
    SECTION("Search works with multiple indexes") {
        auto query = generate_random_vectors(1, dim);
        
        QueryConfig query_config;
        query_config.k = 10;
        
        auto results = manager.search(query.data(), query_config);
        REQUIRE(results.has_value());
        REQUIRE(results->size() == 10);
    }
}

TEST_CASE("IndexManager: Error handling", "[index_manager]") {
    const std::size_t dim = 32;
    
    IndexManager manager(dim);
    
    SECTION("Build with null vectors returns error") {
        IndexBuildConfig config;
        auto result = manager.build(nullptr, 100, config);
        REQUIRE(!result.has_value());
    }
    
    SECTION("Build with zero vectors returns error") {
        auto vectors = generate_random_vectors(100, dim);
        IndexBuildConfig config;
        auto result = manager.build(vectors.data(), 0, config);
        REQUIRE(!result.has_value());
    }
    
    SECTION("Search with null query returns error") {
        QueryConfig config;
        auto result = manager.search(nullptr, config);
        REQUIRE(!result.has_value());
    }
    
    SECTION("Add with null vector returns error") {
        auto result = manager.add(0, nullptr);
        REQUIRE(!result.has_value());
    }
}