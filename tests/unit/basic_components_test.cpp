// Basic components test - tests only the components that can be compiled
#include "vesper/index/kmeans_elkan.hpp"
#include "vesper/kernels/dispatch.hpp"

#include <iostream>
#include <random>
#include <chrono>
#include <cassert>
#include <cmath>

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
    
    // Test Elkan's k-means
    index::KmeansElkan elkan;
    index::KmeansElkan::Config elkan_config{
        .k = k,
        .max_iter = 20,
        .epsilon = 1e-4f,
        .seed = 42
    };
    
    auto elkan_result = elkan.cluster(data.data(), n, dim, elkan_config);
    assert(elkan_result.has_value());
    
    auto stats = elkan.get_stats();
    std::cout << "Elkan's k-means: " << stats.iterations 
              << " iterations, " << stats.skip_rate * 100 
              << "% distances skipped" << std::endl;
    
    std::cout << "✓ K-means algorithms working" << std::endl;
    return true;
}

bool test_performance_basics() {
    std::cout << "\n=== Testing Basic Performance ===" << std::endl;
    
    const std::size_t n = 10000;
    const std::size_t dim = 128;
    
    // Generate dataset
    std::vector<float> data(n * dim);
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (auto& v : data) v = dist(gen);
    
    // Test SIMD kernel performance
    const auto& ops = kernels::select_backend_auto();
    
    auto start = std::chrono::high_resolution_clock::now();
    float sum = 0.0f;
    for (std::size_t i = 0; i < 1000; ++i) {
        sum += ops.l2_sq(
            std::span(data.data(), dim),
            std::span(data.data() + dim, dim)
        );
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "L2 distance computation: " << microseconds / 1000.0 
              << " μs per operation" << std::endl;
    
    // Test k-means performance
    const std::uint32_t k = 100;
    index::KmeansParams params{
        .k = k,
        .max_iter = 10,
        .epsilon = 1e-3f,
        .seed = 42
    };
    
    start = std::chrono::high_resolution_clock::now();
    auto result = index::kmeans_cluster(data.data(), n, dim, params);
    end = std::chrono::high_resolution_clock::now();
    
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "K-means clustering (n=" << n << ", k=" << k << "): " 
              << milliseconds << " ms" << std::endl;
    
    std::cout << "✓ Basic performance tests passed" << std::endl;
    return true;
}

} // namespace vesper::test

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "     Vesper Basic Components Test      " << std::endl;
    std::cout << "========================================" << std::endl;
    
    bool all_passed = true;
    
    try {
        all_passed &= vesper::test::test_simd_kernels();
        all_passed &= vesper::test::test_kmeans();
        all_passed &= vesper::test::test_performance_basics();
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