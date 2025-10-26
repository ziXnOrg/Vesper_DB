/**
 * Real-world integration test for Vesper
 * 
 * Demonstrates:
 * - Creating and configuring an IVF-PQ index
 * - Adding vectors with metadata
 * - Performing searches with different parameters
 * - Save/load functionality
 * - Performance measurement
 */

#include "vesper/index/ivf_pq.hpp"
#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include <iomanip>
#include <filesystem>

namespace fs = std::filesystem;
using namespace vesper::index;

// Generate random vectors for testing
std::vector<float> generate_random_vector(std::size_t dim, std::uint32_t seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    std::vector<float> vec(dim);
    for (auto& v : vec) {
        v = dist(gen);
    }
    
    // Normalize for cosine similarity
    float norm = 0.0f;
    for (auto v : vec) {
        norm += v * v;
    }
    norm = std::sqrt(norm);
    if (norm > 0) {
        for (auto& v : vec) {
            v /= norm;
        }
    }
    
    return vec;
}

// Helper class for timing
class Timer {
    std::chrono::high_resolution_clock::time_point start_;
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}
    
    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }
};

// Test configuration
struct TestConfig {
    std::size_t dimension = 768;  // Like BERT embeddings
    std::size_t num_vectors = 100000;
    std::size_t num_queries = 1000;
    std::uint32_t nlist = 2048;
    std::uint32_t m = 32;  // Higher for better accuracy with 768d
    std::uint32_t nbits = 8;
    std::uint32_t nprobe = 128;
    std::uint32_t k = 10;
    bool use_opq = false;  // Can enable for better accuracy
    std::string save_path = "vesper_test_index.bin";
};

void print_stats(const IvfPqIndex::Stats& stats) {
    std::cout << "\n=== Index Statistics ===" << std::endl;
    std::cout << "  Vectors: " << stats.n_vectors << std::endl;
    std::cout << "  Lists: " << stats.n_lists << std::endl;
    std::cout << "  Subquantizers: " << stats.m << std::endl;
    std::cout << "  Code size: " << stats.code_size << " bytes" << std::endl;
    std::cout << "  Memory usage: " << stats.memory_bytes / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "  Avg list size: " << stats.avg_list_size << std::endl;
    
    if (stats.timings_enabled) {
        std::cout << "\n  Timing breakdown:" << std::endl;
        std::cout << "    Assignment: " << stats.t_assign_ms << " ms" << std::endl;
        std::cout << "    Encoding: " << stats.t_encode_ms << " ms" << std::endl;
        std::cout << "    Lists: " << stats.t_lists_ms << " ms" << std::endl;
    }
}

int main(int argc, char** argv) {
    TestConfig config;
    
    // Parse command-line arguments if provided
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--dim" && i + 1 < argc) {
            config.dimension = std::stoull(argv[++i]);
        } else if (arg == "--nvec" && i + 1 < argc) {
            config.num_vectors = std::stoull(argv[++i]);
        } else if (arg == "--nquery" && i + 1 < argc) {
            config.num_queries = std::stoull(argv[++i]);
        } else if (arg == "--nlist" && i + 1 < argc) {
            config.nlist = std::stoul(argv[++i]);
        } else if (arg == "--nprobe" && i + 1 < argc) {
            config.nprobe = std::stoul(argv[++i]);
        } else if (arg == "--use_opq") {
            config.use_opq = true;
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "  --dim N      Vector dimension (default: 768)" << std::endl;
            std::cout << "  --nvec N     Number of vectors (default: 100000)" << std::endl;
            std::cout << "  --nquery N   Number of queries (default: 1000)" << std::endl;
            std::cout << "  --nlist N    Number of clusters (default: 2048)" << std::endl;
            std::cout << "  --nprobe N   Number of probes (default: 128)" << std::endl;
            std::cout << "  --use_opq    Enable OPQ rotation" << std::endl;
            return 0;
        }
    }
    
    std::cout << "=== Vesper Real-World Test ===" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Dimension: " << config.dimension << std::endl;
    std::cout << "  Vectors: " << config.num_vectors << std::endl;
    std::cout << "  Queries: " << config.num_queries << std::endl;
    std::cout << "  nlist: " << config.nlist << std::endl;
    std::cout << "  m: " << config.m << std::endl;
    std::cout << "  nprobe: " << config.nprobe << std::endl;
    std::cout << "  OPQ: " << (config.use_opq ? "enabled" : "disabled") << std::endl;
    std::cout << std::endl;
    
    try {
        // Generate training data
        std::cout << "Generating training data..." << std::endl;
        std::size_t train_size = std::min(config.num_vectors, std::size_t(50000));
        std::vector<float> train_data;
        train_data.reserve(train_size * config.dimension);
        for (std::size_t i = 0; i < train_size; ++i) {
            auto vec = generate_random_vector(config.dimension, i);
            train_data.insert(train_data.end(), vec.begin(), vec.end());
        }
        
        // Create and train index
        std::cout << "\nTraining IVF-PQ index..." << std::endl;
        IvfPqIndex index;
        
        IvfPqTrainParams train_params;
        train_params.nlist = config.nlist;
        train_params.m = config.m;
        train_params.nbits = config.nbits;
        train_params.use_opq = config.use_opq;
        train_params.verbose = true;
        train_params.timings_enabled = true;
        
        // Use KD-tree for coarse assignment (our optimized default)
        train_params.coarse_assigner = CoarseAssigner::KDTree;
        train_params.kd_leaf_size = 64;  // Our optimized value
        train_params.kd_batch_assign = true;
        
        Timer train_timer;
        auto train_result = index.train(train_data.data(), config.dimension, 
                                        train_size, train_params);
        
        if (!train_result) {
            std::cerr << "Training failed: " << train_result.error().message << std::endl;
            return 1;
        }
        
        double train_time = train_timer.elapsed_ms();
        std::cout << "Training completed in " << train_time << " ms" << std::endl;
        std::cout << "  Iterations: " << train_result->iterations << std::endl;
        std::cout << "  Final error: " << train_result->final_error << std::endl;
        
        // Add vectors
        std::cout << "\nAdding " << config.num_vectors << " vectors..." << std::endl;
        std::vector<std::uint64_t> ids;
        std::vector<float> vectors;
        
        for (std::size_t i = 0; i < config.num_vectors; ++i) {
            ids.push_back(i);
            auto vec = generate_random_vector(config.dimension, i + 1000000);
            vectors.insert(vectors.end(), vec.begin(), vec.end());
        }
        
        Timer add_timer;
        auto add_result = index.add(ids.data(), vectors.data(), config.num_vectors);
        
        if (!add_result) {
            std::cerr << "Add failed: " << add_result.error().message << std::endl;
            return 1;
        }
        
        double add_time = add_timer.elapsed_ms();
        double add_throughput = config.num_vectors / (add_time / 1000.0);
        std::cout << "Add completed in " << add_time << " ms" << std::endl;
        std::cout << "  Throughput: " << std::fixed << std::setprecision(0) 
                  << add_throughput << " vectors/sec" << std::endl;
        
        // Print statistics
        auto stats = index.get_stats();
        print_stats(stats);
        
        // Perform searches
        std::cout << "\n=== Search Performance ===" << std::endl;
        
        std::vector<std::uint32_t> probe_values = {32, 64, 128, 256};
        
        for (auto nprobe : probe_values) {
            IvfPqSearchParams search_params;
            search_params.nprobe = nprobe;
            search_params.k = config.k;
            
            std::vector<double> latencies;
            Timer search_timer;
            
            for (std::size_t i = 0; i < config.num_queries; ++i) {
                auto query = generate_random_vector(config.dimension, i + 2000000);
                
                Timer query_timer;
                auto results = index.search(query.data(), search_params);
                double query_time = query_timer.elapsed_ms();
                latencies.push_back(query_time);
                
                if (!results) {
                    std::cerr << "Search failed: " << results.error().message << std::endl;
                    return 1;
                }
            }
            
            double total_time = search_timer.elapsed_ms();
            double qps = config.num_queries / (total_time / 1000.0);
            
            // Calculate percentiles
            std::sort(latencies.begin(), latencies.end());
            double p50 = latencies[latencies.size() * 0.5];
            double p95 = latencies[latencies.size() * 0.95];
            double p99 = latencies[latencies.size() * 0.99];
            
            std::cout << "\nnprobe=" << nprobe << ":" << std::endl;
            std::cout << "  QPS: " << std::fixed << std::setprecision(0) << qps << std::endl;
            std::cout << "  Latency P50: " << std::setprecision(2) << p50 << " ms" << std::endl;
            std::cout << "  Latency P95: " << p95 << " ms" << std::endl;
            std::cout << "  Latency P99: " << p99 << " ms" << std::endl;
        }
        
        // Save index
        std::cout << "\n=== Persistence Test ===" << std::endl;
        std::cout << "Saving index to " << config.save_path << "..." << std::endl;
        
        Timer save_timer;
        auto save_result = index.save(config.save_path);
        
        if (!save_result) {
            std::cerr << "Save failed: " << save_result.error().message << std::endl;
            return 1;
        }
        
        double save_time = save_timer.elapsed_ms();
        auto file_size = fs::file_size(config.save_path);
        std::cout << "Save completed in " << save_time << " ms" << std::endl;
        std::cout << "  File size: " << (file_size / (1024.0 * 1024.0)) << " MB" << std::endl;
        
        // Load index
        std::cout << "\nLoading index from disk..." << std::endl;
        
        Timer load_timer;
        auto load_result = IvfPqIndex::load(config.save_path);
        
        if (!load_result) {
            std::cerr << "Load failed: " << load_result.error().message << std::endl;
            return 1;
        }
        
        double load_time = load_timer.elapsed_ms();
        std::cout << "Load completed in " << load_time << " ms" << std::endl;
        
        // Verify loaded index
        auto& loaded_index = load_result.value();
        auto loaded_stats = loaded_index.get_stats();
        
        if (loaded_stats.n_vectors != stats.n_vectors) {
            std::cerr << "Error: Loaded index has different number of vectors!" << std::endl;
            return 1;
        }
        
        std::cout << "Loaded index verified: " << loaded_stats.n_vectors << " vectors" << std::endl;
        
        // Test search on loaded index
        std::cout << "\nTesting search on loaded index..." << std::endl;
        auto test_query = generate_random_vector(config.dimension, 3000000);
        
        IvfPqSearchParams test_params;
        test_params.nprobe = 128;
        test_params.k = 10;
        
        auto test_results = loaded_index.search(test_query.data(), test_params);
        
        if (!test_results) {
            std::cerr << "Search on loaded index failed: " << test_results.error().message << std::endl;
            return 1;
        }
        
        std::cout << "Search successful, found " << test_results->size() << " results" << std::endl;
        
        // Clean up
        fs::remove(config.save_path);
        
        std::cout << "\n=== Test Completed Successfully ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}