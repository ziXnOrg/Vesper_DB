/** Simple HNSW test to debug connectivity issues */

#include <iostream>
#include <vector>
#include <random>
#include "vesper/index/hnsw.hpp"

int main() {
    using namespace vesper::index;
    
    // Very simple test with just 100 vectors
    const std::size_t n = 100;
    const std::size_t dim = 8;
    
    // Generate simple data
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    std::vector<float> data(n * dim);
    std::vector<std::uint64_t> ids(n);
    
    for (std::size_t i = 0; i < n * dim; ++i) {
        data[i] = dist(gen);
    }
    for (std::size_t i = 0; i < n; ++i) {
        ids[i] = i;
    }
    
    // Build index
    HnswIndex index;
    HnswBuildParams params{
        .M = 8,
        .efConstruction = 50,
        .seed = 42,
        .extend_candidates = true,
        .keep_pruned_connections = true,
        .max_M = 8,
        .max_M0 = 16
    };
    
    auto init_result = index.init(dim, params, n);
    if (!init_result.has_value()) {
        std::cerr << "Failed to init" << std::endl;
        return 1;
    }
    
    // Add vectors one by one to debug
    for (std::size_t i = 0; i < n; ++i) {
        auto result = index.add(ids[i], data.data() + i * dim);
        if (!result.has_value()) {
            std::cerr << "Failed to add vector " << i << std::endl;
            return 1;
        }
        
        if (i % 10 == 9) {
            std::cout << "Added " << (i+1) << " vectors" << std::endl;
            
            // Test reachability
            std::vector<float> query(dim, 0.0f);
            HnswSearchParams search_params{.efSearch = 100, .k = static_cast<std::uint32_t>(i+1)};
            auto search_result = index.search(query.data(), search_params);
            
            if (search_result.has_value()) {
                std::cout << "  Reachable: " << search_result->size() << "/" << (i+1) << std::endl;
            }
        }
    }
    
    // Final stats
    auto stats = index.get_stats();
    std::cout << "\nFinal Stats:" << std::endl;
    std::cout << "  Nodes: " << stats.n_nodes << std::endl;
    std::cout << "  Total edges: " << stats.n_edges << std::endl;
    std::cout << "  Levels: " << stats.n_levels << std::endl;
    
    // Test final connectivity
    std::vector<float> query(dim, 0.0f);
    HnswSearchParams search_params{.efSearch = 200, .k = static_cast<std::uint32_t>(n)};
    auto result = index.search(query.data(), search_params);
    
    std::cout << "\nFinal reachability: " << result->size() << "/" << n << std::endl;
    
    if (result->size() < n * 0.95) {
        std::cerr << "ERROR: Poor connectivity!" << std::endl;
        return 1;
    }
    
    std::cout << "SUCCESS: Good connectivity" << std::endl;
    return 0;
}