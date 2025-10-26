// Performance test with actual HNSW and IVF-PQ indices
#include <iostream>
#include <random>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include "vesper/index/hnsw.hpp"
#include "vesper/index/ivf_pq.hpp"

namespace vesper::test {

struct PerformanceMetrics {
    double build_rate_vec_per_sec;
    double search_p50_ms;
    double search_p90_ms;
    double search_p99_ms;
    double memory_mb;
};

auto test_hnsw_performance(std::size_t n_vectors, std::size_t dim, std::size_t n_queries) 
    -> PerformanceMetrics {
    
    std::cout << "\n--- HNSW Performance Test ---" << std::endl;
    std::cout << "Vectors: " << n_vectors << ", Dim: " << dim << std::endl;
    
    // Generate test data
    std::vector<float> data(n_vectors * dim);
    std::vector<std::uint64_t> ids(n_vectors);
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (auto& v : data) v = dist(gen);
    std::iota(ids.begin(), ids.end(), 0);
    
    // Build HNSW index
    index::HnswIndex hnsw;
    index::HnswBuildParams params{
        .M = 16,
        .efConstruction = 200,
        .seed = 42
    };
    
    auto init_result = hnsw.init(dim, params, n_vectors);
    if (!init_result.has_value()) {
        std::cerr << "Failed to initialize HNSW" << std::endl;
        return {};
    }
    
    // Measure build time
    auto build_start = std::chrono::high_resolution_clock::now();
    
    // Batch add for better performance
    auto add_result = hnsw.add_batch(ids.data(), data.data(), n_vectors);
    
    auto build_end = std::chrono::high_resolution_clock::now();
    
    if (!add_result.has_value()) {
        std::cerr << "Failed to add vectors to HNSW" << std::endl;
        return {};
    }
    
    auto build_sec = std::chrono::duration<double>(build_end - build_start).count();
    double build_rate = n_vectors / build_sec;
    
    // Get memory usage
    auto stats = hnsw.get_stats();
    double memory_mb = stats.memory_bytes / (1024.0 * 1024.0);
    
    std::cout << "Build time: " << build_sec << " sec" << std::endl;
    std::cout << "Build rate: " << static_cast<int>(build_rate) << " vec/sec" << std::endl;
    std::cout << "Memory: " << memory_mb << " MB" << std::endl;
    std::cout << "Graph: " << stats.n_edges << " edges, avg degree=" << stats.avg_degree << std::endl;
    
    // Measure search performance
    std::vector<double> latencies;
    latencies.reserve(n_queries);
    
    index::HnswSearchParams search_params{
        .efSearch = 50,
        .k = 10
    };
    
    for (std::size_t q = 0; q < n_queries; ++q) {
        const float* query = data.data() + (q % n_vectors) * dim;
        
        auto start = std::chrono::high_resolution_clock::now();
        auto results = hnsw.search(query, search_params);
        auto end = std::chrono::high_resolution_clock::now();
        
        if (results.has_value()) {
            auto ms = std::chrono::duration<double, std::milli>(end - start).count();
            latencies.push_back(ms);
        }
    }
    
    std::sort(latencies.begin(), latencies.end());
    
    PerformanceMetrics metrics;
    metrics.build_rate_vec_per_sec = build_rate;
    metrics.search_p50_ms = latencies[latencies.size() / 2];
    metrics.search_p90_ms = latencies[latencies.size() * 90 / 100];
    metrics.search_p99_ms = latencies[latencies.size() * 99 / 100];
    metrics.memory_mb = memory_mb;
    
    return metrics;
}

auto test_ivfpq_performance(std::size_t n_vectors, std::size_t dim, std::size_t n_queries)
    -> PerformanceMetrics {
    
    std::cout << "\n--- IVF-PQ Performance Test ---" << std::endl;
    std::cout << "Vectors: " << n_vectors << ", Dim: " << dim << std::endl;
    
    // Generate test data
    std::vector<float> data(n_vectors * dim);
    std::vector<std::uint64_t> ids(n_vectors);
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (auto& v : data) v = dist(gen);
    std::iota(ids.begin(), ids.end(), 0);
    
    // Build IVF-PQ index
    index::IvfPqIndex ivfpq;
    index::IvfPqTrainParams params{
        .nlist = std::min(256u, static_cast<std::uint32_t>(std::sqrt(n_vectors))),
        .m = 8,
        .nbits = 8,
        .max_iter = 10,
        .epsilon = 1e-3f,
        .verbose = false,
        .seed = 42
    };
    
    // Train on subset
    std::size_t n_train = std::min(n_vectors / 2, std::size_t(10000));
    
    auto train_start = std::chrono::high_resolution_clock::now();
    auto train_result = ivfpq.train(data.data(), dim, n_train, params);
    auto train_end = std::chrono::high_resolution_clock::now();
    
    if (!train_result.has_value()) {
        std::cerr << "Failed to train IVF-PQ" << std::endl;
        return {};
    }
    
    auto train_sec = std::chrono::duration<double>(train_end - train_start).count();
    std::cout << "Training time: " << train_sec << " sec" << std::endl;
    
    // Add vectors
    auto add_start = std::chrono::high_resolution_clock::now();
    auto add_result = ivfpq.add(ids.data(), data.data(), n_vectors);
    auto add_end = std::chrono::high_resolution_clock::now();
    
    if (!add_result.has_value()) {
        std::cerr << "Failed to add vectors to IVF-PQ" << std::endl;
        return {};
    }
    
    auto add_sec = std::chrono::duration<double>(add_end - add_start).count();
    double build_rate = n_vectors / (train_sec + add_sec);
    
    // Get memory usage
    auto stats = ivfpq.get_stats();
    double memory_mb = stats.memory_bytes / (1024.0 * 1024.0);
    
    std::cout << "Add time: " << add_sec << " sec" << std::endl;
    std::cout << "Total build rate: " << static_cast<int>(build_rate) << " vec/sec" << std::endl;
    std::cout << "Memory: " << memory_mb << " MB" << std::endl;
    std::cout << "Compression: " << (n_vectors * dim * sizeof(float)) / (1024.0 * 1024.0) / memory_mb << "x" << std::endl;
    
    // Measure search performance
    std::vector<double> latencies;
    latencies.reserve(n_queries);
    
    index::IvfPqSearchParams search_params{
        .nprobe = 8,
        .k = 10
    };
    
    for (std::size_t q = 0; q < n_queries; ++q) {
        const float* query = data.data() + (q % n_vectors) * dim;
        
        auto start = std::chrono::high_resolution_clock::now();
        auto results = ivfpq.search(query, search_params);
        auto end = std::chrono::high_resolution_clock::now();
        
        if (results.has_value()) {
            auto ms = std::chrono::duration<double, std::milli>(end - start).count();
            latencies.push_back(ms);
        }
    }
    
    std::sort(latencies.begin(), latencies.end());
    
    PerformanceMetrics metrics;
    metrics.build_rate_vec_per_sec = build_rate;
    metrics.search_p50_ms = latencies[latencies.size() / 2];
    metrics.search_p90_ms = latencies[latencies.size() * 90 / 100];
    metrics.search_p99_ms = latencies[latencies.size() * 99 / 100];
    metrics.memory_mb = memory_mb;
    
    return metrics;
}

bool validate_performance() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "   Actual Index Performance Validation  " << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Test with realistic dataset sizes
    const std::size_t dim = 128;
    
    // HNSW test
    auto hnsw_metrics = test_hnsw_performance(50'000, dim, 1000);
    
    // IVF-PQ test
    auto ivfpq_metrics = test_ivfpq_performance(50'000, dim, 1000);
    
    // Validate against targets
    std::cout << "\n========================================" << std::endl;
    std::cout << "           Performance Summary          " << std::endl;
    std::cout << "========================================" << std::endl;
    
    std::cout << std::fixed << std::setprecision(3);
    
    std::cout << "\nHNSW:" << std::endl;
    std::cout << "  Build rate: " << static_cast<int>(hnsw_metrics.build_rate_vec_per_sec) << " vec/sec";
    bool hnsw_build_pass = hnsw_metrics.build_rate_vec_per_sec >= 50'000;
    std::cout << (hnsw_build_pass ? " ✓" : " ✗") << " (target: ≥50k)" << std::endl;
    
    std::cout << "  Search P50: " << hnsw_metrics.search_p50_ms << " ms";
    bool hnsw_p50_pass = hnsw_metrics.search_p50_ms <= 3.0;
    std::cout << (hnsw_p50_pass ? " ✓" : " ✗") << " (target: ≤3ms)" << std::endl;
    
    std::cout << "  Search P99: " << hnsw_metrics.search_p99_ms << " ms";
    bool hnsw_p99_pass = hnsw_metrics.search_p99_ms <= 20.0;
    std::cout << (hnsw_p99_pass ? " ✓" : " ✗") << " (target: ≤20ms)" << std::endl;
    
    std::cout << "\nIVF-PQ:" << std::endl;
    std::cout << "  Build rate: " << static_cast<int>(ivfpq_metrics.build_rate_vec_per_sec) << " vec/sec" << std::endl;
    std::cout << "  Search P50: " << ivfpq_metrics.search_p50_ms << " ms";
    bool ivfpq_p50_pass = ivfpq_metrics.search_p50_ms <= 3.0;
    std::cout << (ivfpq_p50_pass ? " ✓" : " ✗") << " (target: ≤3ms)" << std::endl;
    
    std::cout << "  Search P99: " << ivfpq_metrics.search_p99_ms << " ms";
    bool ivfpq_p99_pass = ivfpq_metrics.search_p99_ms <= 20.0;
    std::cout << (ivfpq_p99_pass ? " ✓" : " ✗") << " (target: ≤20ms)" << std::endl;
    
    std::cout << "  Memory: " << ivfpq_metrics.memory_mb << " MB" << std::endl;
    
    bool all_pass = hnsw_build_pass && hnsw_p50_pass && hnsw_p99_pass && 
                   ivfpq_p50_pass && ivfpq_p99_pass;
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Overall: " << (all_pass ? "✓ ALL TARGETS MET" : "✗ SOME TARGETS MISSED") << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    return all_pass;
}

} // namespace vesper::test

int main() {
    bool success = vesper::test::validate_performance();
    return success ? 0 : 1;
}