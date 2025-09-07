/** \file query_planner_test.cpp
 *  \brief Unit tests for QueryPlanner cost-based optimization.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "vesper/index/index_manager.hpp"
#include <random>
#include <chrono>
#include <thread>

using namespace vesper;
using namespace vesper::index;

namespace {

std::vector<float> generate_test_vectors(std::size_t n, std::size_t dim, std::uint32_t seed = 42) {
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

} // anonymous namespace

TEST_CASE("QueryPlanner: Basic construction and planning", "[query_planner]") {
    const std::size_t dim = 64;
    const std::size_t n = 1000;
    
    auto vectors = generate_test_vectors(n, dim);
    
    IndexManager manager(dim);
    IndexBuildConfig config;
    config.strategy = SelectionStrategy::Auto;
    
    auto build_result = manager.build(vectors.data(), n, config);
    REQUIRE(build_result.has_value());
    
    SECTION("Planner construction") {
        QueryPlanner planner(manager);
        auto stats = planner.get_stats();
        REQUIRE(stats.plans_generated == 0);
        REQUIRE(stats.plans_executed == 0);
    }
    
    SECTION("Basic query planning") {
        QueryPlanner planner(manager);
        
        auto query = generate_test_vectors(1, dim);
        QueryConfig config;
        config.k = 10;
        
        auto plan = planner.plan(query.data(), config);
        REQUIRE(plan.estimated_cost_ms >= 0);
        REQUIRE(plan.estimated_recall >= 0);
        REQUIRE(plan.estimated_recall <= 1.0f);
        REQUIRE(!plan.explanation.empty());
    }
}

TEST_CASE("QueryPlanner: Cost estimation", "[query_planner]") {
    const std::size_t dim = 64;
    
    SECTION("Small dataset favors HNSW") {
        const std::size_t n = 1000;
        auto vectors = generate_test_vectors(n, dim);
        
        IndexManager manager(dim);
        IndexBuildConfig build_config;
        build_config.strategy = SelectionStrategy::Auto;
        build_config.memory_budget_mb = 1024;
        
        auto build_result = manager.build(vectors.data(), n, build_config);
        REQUIRE(build_result.has_value());
        
        QueryPlanner planner(manager);
        auto query = generate_test_vectors(1, dim);
        
        QueryConfig query_config;
        query_config.k = 10;
        query_config.use_query_planner = true;
        
        auto plan = planner.plan(query.data(), query_config);
        REQUIRE(plan.index == IndexType::HNSW);
        REQUIRE(plan.estimated_cost_ms < 10.0f);  // Should be fast for small dataset
    }
    
    SECTION("Cost increases with k") {
        const std::size_t n = 5000;
        auto vectors = generate_test_vectors(n, dim);
        
        IndexManager manager(dim);
        IndexBuildConfig build_config;
        build_config.strategy = SelectionStrategy::Auto;
        
        auto build_result = manager.build(vectors.data(), n, build_config);
        REQUIRE(build_result.has_value());
        
        QueryPlanner planner(manager);
        auto query = generate_test_vectors(1, dim);
        
        QueryConfig config_k10;
        config_k10.k = 10;
        auto plan_k10 = planner.plan(query.data(), config_k10);
        
        QueryConfig config_k100;
        config_k100.k = 100;
        auto plan_k100 = planner.plan(query.data(), config_k100);
        
        // Higher k should have higher cost
        REQUIRE(plan_k100.estimated_cost_ms >= plan_k10.estimated_cost_ms);
    }
}

TEST_CASE("QueryPlanner: Parameter optimization", "[query_planner]") {
    const std::size_t dim = 32;
    const std::size_t n = 5000;
    
    auto vectors = generate_test_vectors(n, dim);
    
    IndexManager manager(dim);
    IndexBuildConfig config;
    config.strategy = SelectionStrategy::Auto;
    
    auto build_result = manager.build(vectors.data(), n, config);
    REQUIRE(build_result.has_value());
    
    QueryPlanner planner(manager);
    
    SECTION("Adjusts ef_search for HNSW") {
        auto query = generate_test_vectors(1, dim);
        
        QueryConfig config;
        config.k = 50;
        config.ef_search = 64;  // Default value
        
        auto plan = planner.plan(query.data(), config);
        
        // Planner should adjust ef_search based on k and dataset size
        if (plan.index == IndexType::HNSW) {
            REQUIRE(plan.config.ef_search >= config.k);
        }
    }
    
    SECTION("Adjusts parameters based on target recall") {
        auto query = generate_test_vectors(1, dim);
        
        QueryConfig config;
        config.k = 10;
        config.target_recall = 0.99f;  // High recall requirement
        
        auto plan = planner.plan(query.data(), config);
        
        // Should increase search effort for high recall
        REQUIRE(plan.config.ef_search > 64);
    }
}

TEST_CASE("QueryPlanner: Statistics and adaptation", "[query_planner]") {
    const std::size_t dim = 32;
    const std::size_t n = 2000;
    
    auto vectors = generate_test_vectors(n, dim);
    
    IndexManager manager(dim);
    IndexBuildConfig config;
    config.strategy = SelectionStrategy::Auto;
    
    auto build_result = manager.build(vectors.data(), n, config);
    REQUIRE(build_result.has_value());
    
    QueryPlanner planner(manager);
    
    SECTION("Tracks planning statistics") {
        auto query = generate_test_vectors(1, dim);
        QueryConfig config;
        config.k = 10;
        
        // Generate multiple plans
        for (int i = 0; i < 5; ++i) {
            auto plan = planner.plan(query.data(), config);
        }
        
        auto stats = planner.get_stats();
        REQUIRE(stats.plans_generated == 5);
    }
    
    SECTION("Updates statistics from execution") {
        auto query = generate_test_vectors(1, dim);
        QueryConfig config;
        config.k = 10;
        
        auto plan = planner.plan(query.data(), config);
        
        // Simulate execution
        float actual_time_ms = 2.5f;
        float actual_recall = 0.92f;
        
        planner.update_stats(plan, actual_time_ms, actual_recall);
        
        auto stats = planner.get_stats();
        REQUIRE(stats.plans_executed == 1);
        REQUIRE(stats.avg_estimation_error_ms > 0);
    }
    
    SECTION("Adapts estimates based on history") {
        auto query = generate_test_vectors(1, dim);
        QueryConfig config;
        config.k = 10;
        
        // Get initial estimate
        auto plan1 = planner.plan(query.data(), config);
        float initial_estimate = plan1.estimated_cost_ms;
        
        // Provide feedback that actual was slower
        planner.update_stats(plan1, initial_estimate * 2.0f, 0.95f);
        
        // Get new estimate - should be adjusted
        auto plan2 = planner.plan(query.data(), config);
        
        // The planner should learn from the feedback
        REQUIRE(plan2.estimated_cost_ms != initial_estimate);
    }
}

TEST_CASE("QueryPlanner: Multi-index selection", "[query_planner]") {
    const std::size_t dim = 64;
    const std::size_t n = 10000;
    
    auto vectors = generate_test_vectors(n, dim);
    
    IndexManager manager(dim);
    IndexBuildConfig config;
    config.strategy = SelectionStrategy::Hybrid;  // Build multiple indexes
    config.memory_budget_mb = 2048;
    
    auto build_result = manager.build(vectors.data(), n, config);
    REQUIRE(build_result.has_value());
    
    auto active = manager.get_active_indexes();
    
    if (active.size() > 1) {  // Only test if multiple indexes were built
        QueryPlanner planner(manager);
        
        SECTION("Selects appropriate index based on query") {
            auto query = generate_test_vectors(1, dim);
            
            // Small k should favor fast index
            QueryConfig config_small;
            config_small.k = 1;
            auto plan_small = planner.plan(query.data(), config_small);
            
            // Large k might select different index
            QueryConfig config_large;
            config_large.k = 100;
            auto plan_large = planner.plan(query.data(), config_large);
            
            // Plans should be optimized for different scenarios
            REQUIRE(plan_small.estimated_cost_ms > 0);
            REQUIRE(plan_large.estimated_cost_ms > 0);
        }
        
        SECTION("Respects user preference") {
            auto query = generate_test_vectors(1, dim);
            
            QueryConfig config;
            config.k = 10;
            config.preferred_index = IndexType::HNSW;
            
            auto plan = planner.plan(query.data(), config);
            
            // Should respect user preference if HNSW is available
            if (std::find(active.begin(), active.end(), IndexType::HNSW) != active.end()) {
                REQUIRE(plan.index == IndexType::HNSW);
            }
        }
    }
}

TEST_CASE("QueryPlanner: Performance characteristics", "[query_planner]") {
    const std::size_t dim = 32;
    const std::size_t n = 5000;
    
    auto vectors = generate_test_vectors(n, dim);
    
    IndexManager manager(dim);
    IndexBuildConfig config;
    config.strategy = SelectionStrategy::Auto;
    
    auto build_result = manager.build(vectors.data(), n, config);
    REQUIRE(build_result.has_value());
    
    QueryPlanner planner(manager);
    
    SECTION("Planning is fast") {
        auto query = generate_test_vectors(1, dim);
        QueryConfig config;
        config.k = 10;
        
        auto start = std::chrono::high_resolution_clock::now();
        auto plan = planner.plan(query.data(), config);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // Planning should be very fast (< 1ms)
        REQUIRE(duration.count() < 1000);
    }
    
    SECTION("Estimates are reasonable") {
        auto query = generate_test_vectors(1, dim);
        QueryConfig config;
        config.k = 10;
        
        auto plan = planner.plan(query.data(), config);
        
        // Estimates should be in reasonable range
        REQUIRE(plan.estimated_cost_ms > 0.01f);   // At least 0.01ms
        REQUIRE(plan.estimated_cost_ms < 1000.0f); // Less than 1 second
        REQUIRE(plan.estimated_recall >= 0.5f);    // At least 50% recall
        REQUIRE(plan.estimated_recall <= 1.0f);    // At most 100% recall
    }
}