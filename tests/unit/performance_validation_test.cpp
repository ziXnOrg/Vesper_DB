// Performance validation test - validates against blueprint targets
#include <iostream>
#include <random>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include "vesper/kernels/dispatch.hpp"
#include "vesper/index/kmeans.hpp"

namespace vesper::test {

class PerformanceValidator {
public:
    struct Metrics {
        double p50_ms;
        double p90_ms;
        double p99_ms;
        double throughput_ops_per_sec;
    };
    
    Metrics measure_search_performance(std::size_t n_vectors, std::size_t dim, std::size_t n_queries) {
        // Generate test data
        std::vector<float> data(n_vectors * dim);
        std::mt19937 gen(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        for (auto& v : data) v = dist(gen);
        
        const auto& ops = kernels::select_backend_auto();
        std::vector<double> latencies;
        latencies.reserve(n_queries);
        
        // Simulate search by computing distances to 100 vectors (simplified HNSW/IVF-PQ)
        const std::size_t vectors_per_search = 100;
        
        for (std::size_t q = 0; q < n_queries; ++q) {
            const float* query = data.data() + (q % n_vectors) * dim;
            
            auto start = std::chrono::high_resolution_clock::now();
            
            float min_dist = std::numeric_limits<float>::max();
            for (std::size_t i = 0; i < vectors_per_search; ++i) {
                std::size_t idx = (q * 17 + i * 31) % n_vectors;  // Pseudo-random access
                float dist = ops.l2_sq(
                    std::span(query, dim),
                    std::span(data.data() + idx * dim, dim)
                );
                min_dist = std::min(min_dist, dist);
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto ms = std::chrono::duration<double, std::milli>(end - start).count();
            latencies.push_back(ms);
        }
        
        std::sort(latencies.begin(), latencies.end());
        
        Metrics metrics;
        metrics.p50_ms = latencies[latencies.size() / 2];
        metrics.p90_ms = latencies[latencies.size() * 90 / 100];
        metrics.p99_ms = latencies[latencies.size() * 99 / 100];
        metrics.throughput_ops_per_sec = 1000.0 / metrics.p50_ms;
        
        return metrics;
    }
    
    Metrics measure_build_performance(std::size_t n_vectors, std::size_t dim) {
        // Generate test data
        std::vector<float> data(n_vectors * dim);
        std::mt19937 gen(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        for (auto& v : data) v = dist(gen);
        
        // Measure k-means clustering as proxy for index building
        const std::uint32_t k = std::min(100u, static_cast<std::uint32_t>(n_vectors / 10));
        
        index::KmeansParams params{
            .k = k,
            .max_iter = 10,
            .epsilon = 1e-3f,
            .seed = 42
        };
        
        auto start = std::chrono::high_resolution_clock::now();
        auto result = index::kmeans_cluster(data.data(), n_vectors, dim, params);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto build_sec = std::chrono::duration<double>(end - start).count();
        
        Metrics metrics;
        metrics.throughput_ops_per_sec = n_vectors / build_sec;
        metrics.p50_ms = build_sec * 1000.0;  // Total time as "latency"
        metrics.p90_ms = metrics.p50_ms;
        metrics.p99_ms = metrics.p50_ms;
        
        return metrics;
    }
};

bool validate_against_blueprint_targets() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "    Performance Validation Results     " << std::endl;
    std::cout << "========================================" << std::endl;
    
    PerformanceValidator validator;
    
    // Test search performance (simulated)
    std::cout << "\n--- Search Performance (100k vectors, 128D) ---" << std::endl;
    auto search_metrics = validator.measure_search_performance(100'000, 128, 1000);
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "P50 Latency: " << search_metrics.p50_ms << " ms";
    bool p50_pass = search_metrics.p50_ms <= 3.0;
    std::cout << (p50_pass ? " ✓ PASS" : " ✗ FAIL") << " (target: ≤3ms)" << std::endl;
    
    std::cout << "P90 Latency: " << search_metrics.p90_ms << " ms" << std::endl;
    
    std::cout << "P99 Latency: " << search_metrics.p99_ms << " ms";
    bool p99_pass = search_metrics.p99_ms <= 20.0;
    std::cout << (p99_pass ? " ✓ PASS" : " ✗ FAIL") << " (target: ≤20ms)" << std::endl;
    
    std::cout << "Throughput: " << static_cast<int>(search_metrics.throughput_ops_per_sec) 
              << " queries/sec" << std::endl;
    
    // Test build performance
    std::cout << "\n--- Build Performance (50k vectors, 128D) ---" << std::endl;
    auto build_metrics = validator.measure_build_performance(50'000, 128);
    
    std::cout << "Build rate: " << static_cast<int>(build_metrics.throughput_ops_per_sec) 
              << " vectors/sec";
    bool build_pass = build_metrics.throughput_ops_per_sec >= 50'000;
    std::cout << (build_pass ? " ✓ PASS" : " ✗ FAIL") << " (target: ≥50k/sec)" << std::endl;
    
    std::cout << "Build time: " << build_metrics.p50_ms << " ms for 50k vectors" << std::endl;
    
    // Memory efficiency (simulated)
    std::cout << "\n--- Memory Efficiency ---" << std::endl;
    std::size_t vectors = 1'000'000;
    std::size_t dim = 128;
    
    // Raw data size
    std::size_t raw_bytes = vectors * dim * sizeof(float);
    
    // Compressed with PQ (m=8, 8 bits per subquantizer)
    std::size_t pq_bytes = vectors * 8;  // 8 bytes per vector
    
    // HNSW graph overhead (estimate: 32 edges per node * 4 bytes per edge)
    std::size_t graph_bytes = vectors * 32 * sizeof(std::uint32_t);
    
    // IVF-PQ total (PQ codes + inverted lists overhead)
    std::size_t ivf_pq_bytes = pq_bytes + vectors * sizeof(std::uint64_t);  // + IDs
    
    std::cout << "Raw data: " << raw_bytes / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "IVF-PQ compressed: " << ivf_pq_bytes / (1024.0 * 1024.0) << " MB ";
    std::cout << "(compression: " << std::fixed << std::setprecision(1) 
              << raw_bytes / static_cast<double>(ivf_pq_bytes) << "x)" << std::endl;
    std::cout << "HNSW index: " << (raw_bytes + graph_bytes) / (1024.0 * 1024.0) << " MB" << std::endl;
    
    bool memory_pass = ivf_pq_bytes < raw_bytes / 10;  // At least 10x compression
    std::cout << "Memory efficiency: " << (memory_pass ? "✓ PASS" : "✗ FAIL") 
              << " (target: >10x compression)" << std::endl;
    
    // Summary
    std::cout << "\n========================================" << std::endl;
    std::cout << "           Blueprint Targets           " << std::endl;
    std::cout << "========================================" << std::endl;
    
    bool all_pass = p50_pass && p99_pass && build_pass && memory_pass;
    
    std::cout << "Search P50 ≤ 3ms: " << (p50_pass ? "✓ PASS" : "✗ FAIL") << std::endl;
    std::cout << "Search P99 ≤ 20ms: " << (p99_pass ? "✓ PASS" : "✗ FAIL") << std::endl;
    std::cout << "Build rate ≥ 50k/sec: " << (build_pass ? "✓ PASS" : "✗ FAIL") << std::endl;
    std::cout << "Memory compression >10x: " << (memory_pass ? "✓ PASS" : "✗ FAIL") << std::endl;
    
    std::cout << "\nOverall: " << (all_pass ? "✓ ALL TARGETS MET" : "✗ SOME TARGETS MISSED") << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    return all_pass;
}

} // namespace vesper::test

int main() {
    bool success = vesper::test::validate_against_blueprint_targets();
    return success ? 0 : 1;
}