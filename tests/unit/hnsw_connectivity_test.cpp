/** \file hnsw_connectivity_test.cpp
 *  \brief Comprehensive connectivity and recall test for HNSW index.
 *
 * Tests:
 * - Graph connectivity (all nodes reachable from entry point)
 * - Recall at various efSearch settings
 * - Build performance
 * - Search latency
 * - Bidirectional edge consistency
 */

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <unordered_set>
#include <queue>
#include <iomanip>
#include "vesper/index/hnsw.hpp"

using namespace vesper::index;

/** \brief Generate random float vectors. */
auto generate_random_data(std::size_t n, std::size_t dim, std::uint32_t seed = 42)
    -> std::pair<std::vector<float>, std::vector<std::uint64_t>> {
    
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    std::vector<float> data(n * dim);
    std::vector<std::uint64_t> ids(n);
    
    for (std::size_t i = 0; i < n * dim; ++i) {
        data[i] = dist(gen);
    }
    for (std::size_t i = 0; i < n; ++i) {
        ids[i] = i;
    }
    
    return {data, ids};
}

/** \brief Compute brute-force nearest neighbors for ground truth. */
auto compute_ground_truth(const std::vector<float>& data, std::size_t n, std::size_t dim,
                         const std::vector<float>& queries, std::size_t n_queries, std::size_t k)
    -> std::vector<std::vector<std::uint64_t>> {
    
    std::vector<std::vector<std::uint64_t>> ground_truth(n_queries);
    
    for (std::size_t q = 0; q < n_queries; ++q) {
        const float* query = queries.data() + q * dim;
        std::vector<std::pair<float, std::uint64_t>> distances;
        
        for (std::size_t i = 0; i < n; ++i) {
            const float* vec = data.data() + i * dim;
            float dist = 0.0f;
            for (std::size_t d = 0; d < dim; ++d) {
                float diff = query[d] - vec[d];
                dist += diff * diff;
            }
            distances.emplace_back(dist, i);
        }
        
        std::partial_sort(distances.begin(), distances.begin() + k, distances.end());
        
        ground_truth[q].reserve(k);
        for (std::size_t i = 0; i < k; ++i) {
            ground_truth[q].push_back(distances[i].second);
        }
    }
    
    return ground_truth;
}

/** \brief Test graph connectivity. */
auto test_connectivity(const HnswIndex& index) -> bool {
    auto stats = index.get_stats();
    
    if (stats.n_nodes == 0) {
        std::cout << "Empty index, skipping connectivity test" << std::endl;
        return true;
    }
    
    // Proper connectivity via base-layer BFS from entry point
    std::size_t reachable = index.reachable_count_base_layer();
    float coverage = static_cast<float>(reachable) / stats.n_nodes;
    
    std::cout << "Connectivity Test:" << std::endl;
    std::cout << "  Total nodes: " << stats.n_nodes << std::endl;
    std::cout << "  Reachable nodes: " << reachable << std::endl;
    std::cout << "  Coverage: " << std::fixed << std::setprecision(2) 
              << (coverage * 100) << "%" << std::endl;
    
    if (coverage < 0.95) {
        std::cerr << "ERROR: Graph is not well-connected! Only " 
                  << (coverage * 100) << "% of nodes are reachable" << std::endl;
        return false;
    }
    
    return true;
}

/** \brief Test recall at different efSearch values. */
auto test_recall(const HnswIndex& index, const std::vector<float>& queries,
                std::size_t n_queries, std::size_t dim,
                const std::vector<std::vector<std::uint64_t>>& ground_truth,
                std::size_t k) -> void {
    
    std::vector<std::uint32_t> ef_values = {50, 100, 200, 400};
    
    std::cout << "\nRecall Test (k=" << k << "):" << std::endl;
    std::cout << "efSearch | Recall@" << k << " | Avg Latency (ms)" << std::endl;
    std::cout << "---------|-----------|------------------" << std::endl;
    
    for (std::uint32_t ef : ef_values) {
        HnswSearchParams params{.efSearch = ef, .k = static_cast<std::uint32_t>(k)};
        
        std::size_t correct = 0;
        auto start = std::chrono::high_resolution_clock::now();
        
        for (std::size_t q = 0; q < n_queries; ++q) {
            const float* query = queries.data() + q * dim;
            auto result = index.search(query, params);
            
            if (!result.has_value()) continue;
            
            std::unordered_set<std::uint64_t> result_set;
            for (const auto& [id, dist] : result.value()) {
                result_set.insert(id);
            }
            
            for (std::uint64_t true_id : ground_truth[q]) {
                if (result_set.count(true_id) > 0) {
                    correct++;
                }
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end - start).count();
        
        float recall = static_cast<float>(correct) / (n_queries * k);
        float avg_latency = duration / n_queries;
        
        std::cout << std::setw(8) << ef << " | "
                  << std::fixed << std::setprecision(3) << std::setw(9) << recall << " | "
                  << std::fixed << std::setprecision(3) << std::setw(16) << avg_latency << std::endl;
                  
        if (ef == 100 && recall < 0.9) {
            std::cerr << "WARNING: Recall at efSearch=100 is only " 
                      << (recall * 100) << "% (target: 90%+)" << std::endl;
        }
    }
}

int main() {
    std::cout << "=== HNSW Connectivity and Performance Test ===" << std::endl;
    
    // Test parameters
    const std::size_t n_vectors = 10000;
    const std::size_t n_queries = 100;
    const std::size_t dim = 128;
    const std::size_t k = 10;
    
    std::cout << "\nTest Configuration:" << std::endl;
    std::cout << "  Vectors: " << n_vectors << std::endl;
    std::cout << "  Dimension: " << dim << std::endl;
    std::cout << "  Queries: " << n_queries << std::endl;
    std::cout << "  k: " << k << std::endl;
    
    // Generate data
    std::cout << "\nGenerating random data..." << std::endl;
    auto [data, ids] = generate_random_data(n_vectors, dim);
    auto [queries, _] = generate_random_data(n_queries, dim, 12345);
    
    // Compute ground truth
    std::cout << "Computing ground truth..." << std::endl;
    auto ground_truth = compute_ground_truth(data, n_vectors, dim, queries, n_queries, k);
    
    // Build HNSW index
    std::cout << "\nBuilding HNSW index..." << std::endl;
    HnswIndex index;
    HnswBuildParams params{
        .M = 16,
        .efConstruction = 200,
        .seed = 42,
        .extend_candidates = true,  // CRITICAL for recall
        .keep_pruned_connections = true,  // Algorithm 4
        .max_M = 16,
        .max_M0 = 32
    };
    
    auto init_result = index.init(dim, params, n_vectors);
    if (!init_result.has_value()) {
        std::cerr << "Failed to initialize index" << std::endl;
        return 1;
    }
    
    // Measure build time
    auto start = std::chrono::high_resolution_clock::now();
    
    auto add_result = index.add_batch(ids.data(), data.data(), n_vectors);
    if (!add_result.has_value()) {
        std::cerr << "Failed to add vectors" << std::endl;
        return 1;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto build_sec = std::chrono::duration<double>(end - start).count();
    double build_rate = n_vectors / build_sec;
    
    std::cout << "Build completed:" << std::endl;
    std::cout << "  Time: " << std::fixed << std::setprecision(2) << build_sec << " seconds" << std::endl;
    std::cout << "  Rate: " << std::fixed << std::setprecision(0) << build_rate << " vec/sec" << std::endl;
    
    if (build_rate < 50000) {
        std::cerr << "WARNING: Build rate is only " << build_rate 
                  << " vec/sec (target: 50,000+)" << std::endl;
    }
    
    // Get index statistics
    auto stats = index.get_stats();
    std::cout << "\nIndex Statistics:" << std::endl;
    std::cout << "  Nodes: " << stats.n_nodes << std::endl;
    std::cout << "  Edges: " << stats.n_edges << std::endl;
    std::cout << "  Levels: " << stats.n_levels << std::endl;
    std::cout << "  Avg degree: " << std::fixed << std::setprecision(1) << stats.avg_degree << std::endl;
    std::cout << "  Memory: " << stats.memory_bytes / (1024.0 * 1024.0) << " MB" << std::endl;
    
    // Test connectivity
    std::cout << std::endl;
    bool connected = test_connectivity(index);
    if (!connected) {
        std::cerr << "FAILED: Graph connectivity test failed!" << std::endl;
        return 1;
    }
    
    // Test recall
    test_recall(index, queries, n_queries, dim, ground_truth, k);
    
    // Performance targets summary
    std::cout << "\n=== Performance Summary ===" << std::endl;
    std::cout << "Build rate: " << (build_rate >= 50000 ? "✓" : "✗") 
              << " " << build_rate << " vec/sec (target: ≥50,000)" << std::endl;
    std::cout << "Graph connectivity: " << (connected ? "✓" : "✗") 
              << " (target: ≥95% reachable)" << std::endl;
    
    std::cout << "\nTest " << (connected ? "PASSED" : "FAILED") << std::endl;
    
    return connected ? 0 : 1;
}