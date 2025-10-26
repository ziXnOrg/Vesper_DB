#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cstdlib>
#include "vesper/index/ivf_pq.hpp"

using namespace vesper::index;

int main(int argc, char** argv) {
    // Default parameters
    std::size_t dim = 128;
    std::size_t n_train = 10000;
    std::size_t n_test = 1000;
    std::uint32_t nlist = 1024;
    std::uint32_t m = 16;
    bool enable_hierarchical = true;
    
    // Parse command line
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg.find("--dim=") == 0) dim = std::stoull(arg.substr(6));
        else if (arg.find("--n_train=") == 0) n_train = std::stoull(arg.substr(10));
        else if (arg.find("--n_test=") == 0) n_test = std::stoull(arg.substr(9));
        else if (arg.find("--nlist=") == 0) nlist = std::stoul(arg.substr(8));
        else if (arg.find("--no_hierarchical") == 0) enable_hierarchical = false;
    }
    
    // Set environment variable to enable hierarchical KD-tree
    if (enable_hierarchical) {
        #ifdef _WIN32
        _putenv("VESPER_HIERARCHICAL_KD=1");
        _putenv("VESPER_KD_BATCH=1");
        #else
        setenv("VESPER_HIERARCHICAL_KD", "1", 1);
        setenv("VESPER_KD_BATCH", "1", 1);
        #endif
    }
    
    std::cout << "Testing hierarchical KD-tree:\n";
    std::cout << "  dim=" << dim << ", n_train=" << n_train 
              << ", n_test=" << n_test << ", nlist=" << nlist 
              << ", hierarchical=" << (enable_hierarchical ? "enabled" : "disabled") << "\n\n";
    
    // Generate random data
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    std::vector<float> train_data(n_train * dim);
    std::vector<float> test_data(n_test * dim);
    
    for (auto& v : train_data) v = dist(rng);
    for (auto& v : test_data) v = dist(rng);
    
    // Create and train index
    IvfPqIndex index;
    IvfPqTrainParams params;
    params.nlist = nlist;
    params.m = m;
    params.nbits = 8;
    params.coarse_assigner = CoarseAssigner::KDTree;
    params.use_centroid_ann = true;  // Enable KD-tree assignment
    params.verbose = true;
    
    std::cout << "Training index...\n";
    auto train_start = std::chrono::steady_clock::now();
    auto train_result = index.train(train_data.data(), dim, n_train, params);
    auto train_end = std::chrono::steady_clock::now();
    
    if (!train_result.has_value()) {
        std::cerr << "Training failed: " << train_result.error().message << "\n";
        return 1;
    }
    
    auto train_ms = std::chrono::duration_cast<std::chrono::milliseconds>(train_end - train_start).count();
    std::cout << "Training completed in " << train_ms << " ms\n\n";
    
    // Add vectors
    std::vector<std::uint64_t> ids(n_test);
    for (std::size_t i = 0; i < n_test; ++i) {
        ids[i] = i;
    }
    
    std::cout << "Adding " << n_test << " vectors...\n";
    auto add_start = std::chrono::steady_clock::now();
    auto add_result = index.add(ids.data(), test_data.data(), n_test);
    auto add_end = std::chrono::steady_clock::now();
    
    if (!add_result.has_value()) {
        std::cerr << "Add failed: " << add_result.error().message << "\n";
        return 1;
    }
    
    auto add_ms = std::chrono::duration_cast<std::chrono::milliseconds>(add_end - add_start).count();
    std::cout << "Add completed in " << add_ms << " ms\n";
    std::cout << "Throughput: " << (n_test * 1000.0 / add_ms) << " vectors/sec\n\n";
    
    // Get stats
    auto stats = index.get_stats();
    std::cout << "Index stats:\n";
    std::cout << "  n_vectors: " << stats.n_vectors << "\n";
    std::cout << "  n_lists: " << stats.n_lists << "\n";
    std::cout << "  avg_list_size: " << stats.avg_list_size << "\n";
    std::cout << "  memory_bytes: " << stats.memory_bytes << "\n";
    
    if (stats.timings_enabled) {
        std::cout << "\nTiming breakdown:\n";
        std::cout << "  t_assign_ms: " << stats.t_assign_ms << "\n";
        std::cout << "  t_encode_ms: " << stats.t_encode_ms << "\n";
        std::cout << "  t_lists_ms: " << stats.t_lists_ms << "\n";
        
        if (stats.kd_traversal_ms > 0 || stats.kd_leaf_ms > 0) {
            std::cout << "\nKD-tree timing:\n";
            std::cout << "  kd_traversal_ms: " << stats.kd_traversal_ms << "\n";
            std::cout << "  kd_leaf_ms: " << stats.kd_leaf_ms << "\n";
        }
    }
    
    if (stats.kd_nodes_pushed > 0) {
        std::cout << "\nKD-tree instrumentation:\n";
        std::cout << "  kd_nodes_pushed: " << stats.kd_nodes_pushed << "\n";
        std::cout << "  kd_nodes_popped: " << stats.kd_nodes_popped << "\n";
        std::cout << "  kd_leaves_scanned: " << stats.kd_leaves_scanned << "\n";
    }
    
    // Test search
    std::cout << "\nTesting search...\n";
    IvfPqSearchParams search_params;
    search_params.nprobe = 32;
    search_params.k = 10;
    
    const std::size_t n_queries = 10;
    auto search_start = std::chrono::steady_clock::now();
    
    for (std::size_t i = 0; i < n_queries; ++i) {
        const float* query = test_data.data() + i * dim;
        auto results = index.search(query, search_params);
        if (!results.has_value()) {
            std::cerr << "Search failed: " << results.error().message << "\n";
            return 1;
        }
    }
    
    auto search_end = std::chrono::steady_clock::now();
    auto search_ms = std::chrono::duration_cast<std::chrono::milliseconds>(search_end - search_start).count();
    
    std::cout << "Search completed: " << n_queries << " queries in " << search_ms << " ms\n";
    std::cout << "Average latency: " << (search_ms / static_cast<double>(n_queries)) << " ms/query\n";
    
    std::cout << "\nTest completed successfully!\n";
    
    return 0;
}