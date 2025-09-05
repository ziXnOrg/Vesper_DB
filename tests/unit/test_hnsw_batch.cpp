/** Test HNSW with batch addition */

#include <iostream>
#include <vector>
#include <random>
#include "vesper/index/hnsw.hpp"

int main() {
    using namespace vesper::index;
    
    const std::size_t n = 5000;
    const std::size_t dim = 128;
    
    // Generate data
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
        .M = 16,
        .efConstruction = 200,
        .seed = 42,
        .extend_candidates = true,
        .keep_pruned_connections = true,
        .max_M = 16,
        .max_M0 = 32
    };
    
    auto init_result = index.init(dim, params, n);
    if (!init_result.has_value()) {
        std::cerr << "Failed to init" << std::endl;
        return 1;
    }
    
    std::cout << "Adding " << n << " vectors in batch..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    auto result = index.add_batch(ids.data(), data.data(), n);
    if (!result.has_value()) {
        std::cerr << "Failed to add batch" << std::endl;
        return 1;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(end - start).count();
    
    std::cout << "Added in " << duration << " seconds (" << n/duration << " vec/sec)" << std::endl;
    
    // Test connectivity using base-layer BFS (not bounded by efSearch/k)
    auto stats = index.get_stats();
    auto reachable = index.reachable_count_base_layer();

    std::cout << "\nStats:" << std::endl;
    std::cout << "  Nodes: " << stats.n_nodes << std::endl;
    std::cout << "  Reachable: " << reachable << std::endl;
    std::cout << "  Coverage: " << (100.0 * reachable / stats.n_nodes) << "%" << std::endl;

    if (reachable < n * 0.95) {
        std::cerr << "ERROR: Poor connectivity!" << std::endl;
        return 1;
    }

    std::cout << "SUCCESS: Good connectivity" << std::endl;
    return 0;
}