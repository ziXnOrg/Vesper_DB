#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <numeric>
#include <iomanip>
#include "vesper/index/ivf_pq.hpp"

using namespace vesper::index;
using namespace std::chrono;

int main() {
    std::cout << "=== IVF-PQ Index Test ===" << std::endl;
    
    // Parameters
    const std::size_t dim = 128;
    const std::size_t n_train = 10000;
    const std::size_t n_add = 5000;
    const std::size_t n_queries = 100;
    
    // Generate random data
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    std::vector<float> train_data(n_train * dim);
    std::vector<float> add_data(n_add * dim);
    std::vector<std::uint64_t> ids(n_add);
    std::vector<float> queries(n_queries * dim);
    
    std::cout << "Generating data..." << std::endl;
    for (auto& val : train_data) val = dist(rng);
    for (auto& val : add_data) val = dist(rng);
    for (auto& val : queries) val = dist(rng);
    std::iota(ids.begin(), ids.end(), 0);
    
    // Create and configure index
    IvfPqIndex index;
    
    IvfPqTrainParams train_params;
    train_params.nlist = 256;       // Number of coarse centroids
    train_params.m = 8;              // Number of subquantizers
    train_params.nbits = 8;          // Bits per subquantizer
    train_params.max_iter = 25;
    train_params.epsilon = 1e-4f;
    train_params.verbose = true;
    train_params.use_opq = false;    // OPQ not fully implemented yet
    train_params.seed = 42;
    
    // Train the index
    std::cout << "\nTraining IVF-PQ index..." << std::endl;
    std::cout << "  Coarse centroids: " << train_params.nlist << std::endl;
    std::cout << "  Subquantizers: " << train_params.m << std::endl;
    std::cout << "  Bits per subquantizer: " << train_params.nbits << std::endl;
    
    auto start = high_resolution_clock::now();
    auto train_result = index.train(train_data.data(), dim, n_train, train_params);
    auto end = high_resolution_clock::now();
    
    if (!train_result) {
        std::cerr << "Training failed!" << std::endl;
        return 1;
    }
    
    double train_time = duration<double, std::milli>(end - start).count();
    std::cout << "Training completed in " << std::fixed << std::setprecision(2) 
              << train_time << " ms" << std::endl;
    
    auto train_stats = train_result.value();
    std::cout << "  Training iterations: " << train_stats.iterations << std::endl;
    std::cout << "  Final error: " << train_stats.final_error << std::endl;
    
    // Add vectors to the index
    std::cout << "\nAdding " << n_add << " vectors..." << std::endl;
    start = high_resolution_clock::now();
    auto add_result = index.add(ids.data(), add_data.data(), n_add);
    end = high_resolution_clock::now();
    
    if (!add_result) {
        std::cerr << "Adding vectors failed!" << std::endl;
        return 1;
    }
    
    double add_time = duration<double, std::milli>(end - start).count();
    double add_throughput = (n_add * 1000.0) / add_time;
    std::cout << "Added in " << add_time << " ms (" 
              << std::fixed << std::setprecision(0) << add_throughput << " vec/s)" << std::endl;
    
    // Get index statistics
    auto stats = index.get_stats();
    std::cout << "\nIndex Statistics:" << std::endl;
    std::cout << "  Total vectors: " << stats.n_vectors << std::endl;
    std::cout << "  Inverted lists: " << stats.n_lists << std::endl;
    std::cout << "  Memory usage: " << std::fixed << std::setprecision(2) 
              << stats.memory_bytes / (1024.0 * 1024.0) << " MB" << std::endl;
    
    double bytes_per_vector = static_cast<double>(stats.memory_bytes) / stats.n_vectors;
    std::cout << "  Bytes per vector: " << std::fixed << std::setprecision(1) 
              << bytes_per_vector << std::endl;
    
    double compression_ratio = (dim * sizeof(float)) / bytes_per_vector;
    std::cout << "  Compression ratio: " << std::fixed << std::setprecision(1) 
              << compression_ratio << "x" << std::endl;
    
    // Search test
    std::cout << "\nSearching for " << n_queries << " queries..." << std::endl;
    
    IvfPqSearchParams search_params;
    search_params.nprobe = 8;        // Number of cells to search
    search_params.k = 10;             // Number of neighbors
    search_params.use_exact_rerank = false;
    
    std::cout << "  nprobe: " << search_params.nprobe << std::endl;
    std::cout << "  k: " << search_params.k << std::endl;
    
    start = high_resolution_clock::now();
    auto search_result = index.search_batch(queries.data(), n_queries, search_params);
    end = high_resolution_clock::now();
    
    if (!search_result) {
        std::cerr << "Search failed!" << std::endl;
        return 1;
    }
    
    double search_time = duration<double, std::milli>(end - start).count();
    double avg_latency = search_time / n_queries;
    std::cout << "Search completed in " << search_time << " ms" << std::endl;
    std::cout << "Average latency: " << std::fixed << std::setprecision(3) 
              << avg_latency << " ms/query" << std::endl;
    
    // Verify results
    const auto& results = search_result.value();
    std::size_t total_found = 0;
    for (const auto& result : results) {
        total_found += result.size();
    }
    double avg_found = static_cast<double>(total_found) / n_queries;
    std::cout << "Average results per query: " << std::fixed << std::setprecision(1) 
              << avg_found << std::endl;
    
    // Memory efficiency summary
    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Memory Efficiency:" << std::endl;
    std::cout << "  Original size: " << std::fixed << std::setprecision(2) 
              << (n_add * dim * sizeof(float)) / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "  Compressed size: " << std::fixed << std::setprecision(2) 
              << stats.memory_bytes / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "  Compression: " << std::fixed << std::setprecision(1) 
              << compression_ratio << "x" << std::endl;
    
    std::cout << "\nPerformance:" << std::endl;
    std::cout << "  Training: " << std::fixed << std::setprecision(1) 
              << train_time << " ms" << std::endl;
    std::cout << "  Indexing: " << std::fixed << std::setprecision(0) 
              << add_throughput << " vec/s" << std::endl;
    std::cout << "  Search: " << std::fixed << std::setprecision(3) 
              << avg_latency << " ms/query" << std::endl;
    
    std::cout << "\n✅ IVF-PQ index is working!" << std::endl;
    
    return 0;
}