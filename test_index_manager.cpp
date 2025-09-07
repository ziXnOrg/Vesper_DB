// Simple test for IndexManager
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "vesper/index/index_manager.hpp"

using namespace vesper;
using namespace vesper::index;

std::vector<float> generate_vectors(size_t n, size_t dim) {
    std::vector<float> vectors(n * dim);
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (auto& v : vectors) {
        v = dist(gen);
    }
    
    // Normalize
    for (size_t i = 0; i < n; ++i) {
        float norm = 0.0f;
        for (size_t d = 0; d < dim; ++d) {
            norm += vectors[i * dim + d] * vectors[i * dim + d];
        }
        norm = std::sqrt(norm);
        if (norm > 0) {
            for (size_t d = 0; d < dim; ++d) {
                vectors[i * dim + d] /= norm;
            }
        }
    }
    
    return vectors;
}

int main() {
    std::cout << "Testing IndexManager..." << std::endl;
    
    const size_t dim = 32;
    const size_t n = 1000;
    const uint32_t k = 10;
    
    // Generate test data
    auto vectors = generate_vectors(n, dim);
    auto query = generate_vectors(1, dim);
    
    // Create IndexManager
    IndexManager manager(dim);
    
    // Build index
    IndexBuildConfig config;
    config.strategy = SelectionStrategy::Auto;
    config.memory_budget_mb = 256;
    
    std::cout << "Building index with " << n << " vectors..." << std::endl;
    auto build_result = manager.build(vectors.data(), n, config);
    
    if (!build_result.has_value()) {
        std::cerr << "Failed to build index: " << static_cast<int>(build_result.error().code) << std::endl;
        return 1;
    }
    
    std::cout << "Build successful!" << std::endl;
    
    // Check active indexes
    auto active = manager.get_active_indexes();
    std::cout << "Active indexes: ";
    for (auto idx : active) {
        std::cout << static_cast<int>(idx) << " ";
    }
    std::cout << std::endl;
    
    // Perform search
    QueryConfig qconfig;
    qconfig.k = k;
    qconfig.use_query_planner = false;
    
    std::cout << "Searching for " << k << " nearest neighbors..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    auto results = manager.search(query.data(), qconfig);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    if (!results.has_value()) {
        std::cerr << "Search failed: " << static_cast<int>(results.error().code) << std::endl;
        return 1;
    }
    
    std::cout << "Search completed in " << duration.count() << " microseconds" << std::endl;
    std::cout << "Found " << results->size() << " neighbors" << std::endl;
    
    // Print first few results
    std::cout << "Top 5 results:" << std::endl;
    for (size_t i = 0; i < std::min<size_t>(5, results->size()); ++i) {
        std::cout << "  ID: " << (*results)[i].first 
                  << ", Distance: " << (*results)[i].second << std::endl;
    }
    
    // Test memory usage
    auto memory = manager.memory_usage();
    std::cout << "Memory usage: " << (memory / 1024.0 / 1024.0) << " MB" << std::endl;
    
    // Test statistics
    auto stats = manager.get_stats();
    std::cout << "Index statistics:" << std::endl;
    for (const auto& stat : stats) {
        std::cout << "  Index " << static_cast<int>(stat.type) 
                  << ": " << stat.num_vectors << " vectors, "
                  << (stat.memory_usage_bytes / 1024.0 / 1024.0) << " MB" << std::endl;
    }
    
    std::cout << "All tests passed successfully!" << std::endl;
    return 0;
}