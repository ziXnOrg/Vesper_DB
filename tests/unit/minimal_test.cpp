// Minimal test - tests only the components without SIMD dependencies
#include "vesper/kernels/dispatch.hpp"
#include "vesper/index/kmeans.hpp"

#include <iostream>
#include <random>
#include <chrono>
#include <cassert>
#include <cmath>
#include <algorithm>

namespace vesper::test {

bool test_simd_kernels() {
    std::cout << "\n=== Testing SIMD Kernels ===" << std::endl;
    
    const std::size_t dim = 128;
    std::vector<float> a(dim), b(dim);
    
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (auto& v : a) v = dist(gen);
    for (auto& v : b) v = dist(gen);
    
    const auto& ops = kernels::select_backend_auto();
    
    // Test L2 distance
    float l2 = ops.l2_sq(a, b);
    std::cout << "L2 distance: " << l2 << std::endl;
    assert(l2 > 0 && std::isfinite(l2));
    
    // Test inner product
    float ip = ops.inner_product(a, b);
    std::cout << "Inner product: " << ip << std::endl;
    assert(std::isfinite(ip));
    
    // Test cosine similarity
    float cos = ops.cosine_similarity(a, b);
    std::cout << "Cosine similarity: " << cos << std::endl;
    assert(cos >= -1.0f && cos <= 1.0f);
    
    std::cout << "✓ SIMD kernels working" << std::endl;
    return true;
}

bool test_kmeans() {
    std::cout << "\n=== Testing K-means ===" << std::endl;
    
    const std::size_t n = 1000;
    const std::size_t dim = 64;
    const std::uint32_t k = 10;
    
    // Generate test data
    std::vector<float> data(n * dim);
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (auto& v : data) v = dist(gen);
    
    // Test Lloyd's k-means
    index::KmeansParams params{
        .k = k,
        .max_iter = 20,
        .epsilon = 1e-4f,
        .seed = 42
    };
    
    auto result = index::kmeans_cluster(data.data(), n, dim, params);
    assert(result.has_value());
    assert(result->centroids.size() == k);
    assert(result->assignments.size() == n);
    
    std::cout << "Lloyd's k-means: " << result->iterations 
              << " iterations, inertia=" << result->inertia << std::endl;
    
    std::cout << "✓ K-means algorithm working" << std::endl;
    return true;
}

bool test_performance_targets() {
    std::cout << "\n=== Testing Performance ===" << std::endl;
    
    const std::size_t n = 100000;
    const std::size_t dim = 128;
    const std::size_t n_queries = 100;
    
    // Generate dataset
    std::vector<float> data(n * dim);
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (auto& v : data) v = dist(gen);
    
    // Test SIMD kernel performance
    const auto& ops = kernels::select_backend_auto();
    
    // Measure search-like operations (computing distances)
    std::vector<double> latencies;
    for (std::size_t q = 0; q < n_queries; ++q) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Simulate searching through 1000 vectors
        float min_dist = std::numeric_limits<float>::max();
        for (std::size_t i = 0; i < 1000; ++i) {
            float dist = ops.l2_sq(
                std::span(data.data() + q * dim, dim),
                std::span(data.data() + i * dim, dim)
            );
            min_dist = std::min(min_dist, dist);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration<double, std::milli>(end - start).count();
        latencies.push_back(ms);
    }
    
    std::sort(latencies.begin(), latencies.end());
    
    double p50 = latencies[latencies.size() / 2];
    double p99 = latencies[latencies.size() * 99 / 100];
    
    std::cout << "Distance computation latency:" << std::endl;
    std::cout << "  P50: " << p50 << " ms" << std::endl;
    std::cout << "  P99: " << p99 << " ms" << std::endl;
    
    // Test k-means build performance
    const std::uint32_t k = 100;
    index::KmeansParams params{
        .k = k,
        .max_iter = 10,
        .epsilon = 1e-3f,
        .seed = 42
    };
    
    auto build_start = std::chrono::high_resolution_clock::now();
    auto result = index::kmeans_cluster(data.data(), 10000, dim, params);
    auto build_end = std::chrono::high_resolution_clock::now();
    
    auto build_duration = std::chrono::duration<double>(build_end - build_start).count();
    double build_rate = 10000 / build_duration;
    
    std::cout << "\nK-means clustering performance:" << std::endl;
    std::cout << "  Build rate: " << build_rate << " vectors/sec" << std::endl;
    std::cout << "  Time: " << build_duration * 1000 << " ms for 10k vectors" << std::endl;
    
    std::cout << "✓ Performance tests completed" << std::endl;
    return true;
}

} // namespace vesper::test

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "        Vesper Minimal Test Suite      " << std::endl;
    std::cout << "========================================" << std::endl;
    
    bool all_passed = true;
    
    try {
        all_passed &= vesper::test::test_simd_kernels();
        all_passed &= vesper::test::test_kmeans();
        all_passed &= vesper::test::test_performance_targets();
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\n========================================" << std::endl;
    if (all_passed) {
        std::cout << "     ALL TESTS PASSED! ✓              " << std::endl;
    } else {
        std::cout << "     SOME TESTS FAILED ✗              " << std::endl;
    }
    std::cout << "========================================" << std::endl;
    
    return all_passed ? 0 : 1;
}