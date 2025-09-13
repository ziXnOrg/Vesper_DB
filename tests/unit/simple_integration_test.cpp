// Simple integration test that doesn't require GTest
#include "vesper/index/ivf_pq.hpp"
#include "vesper/index/hnsw.hpp"
#include "vesper/index/kmeans_elkan.hpp"
#include "vesper/kernels/dispatch.hpp"
#include "vesper/core/platform_utils.hpp"


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

bool test_ivf_pq() {
    std::cout << "\n=== Testing IVF-PQ ===" << std::endl;

    const std::size_t n = 10000;
    const std::size_t dim = 128;

    // Generate test data
    std::vector<float> data(n * dim);
    std::vector<std::uint64_t> ids(n);

    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (auto& v : data) v = dist(gen);
    for (std::size_t i = 0; i < n; ++i) ids[i] = i;

    // Build IVF-PQ
    index::IvfPqIndex index;
    index::IvfPqTrainParams params{
        .nlist = 100,
        .m = 8,
        .nbits = 8,
        .verbose = false
    };

    auto train_start = std::chrono::high_resolution_clock::now();
    auto train_result = index.train(data.data(), dim, n/2, params);
    auto train_end = std::chrono::high_resolution_clock::now();

    assert(train_result.has_value());

    auto train_ms = std::chrono::duration<double, std::milli>(train_end - train_start).count();
    std::cout << "Training time: " << train_ms << " ms" << std::endl;

    // Add vectors
    auto add_result = index.add(ids.data(), data.data(), n);
    assert(add_result.has_value());

    auto stats = index.get_stats();
    std::cout << "Index size: " << stats.n_vectors << " vectors, "
              << stats.memory_bytes / (1024.0 * 1024.0) << " MB" << std::endl;

    // Search
    index::IvfPqSearchParams search_params{
        .nprobe = 8,
        .k = 10
    };

    auto search_start = std::chrono::high_resolution_clock::now();
    auto results = index.search(data.data(), search_params);
    auto search_end = std::chrono::high_resolution_clock::now();

    assert(results.has_value());
    assert(!results->empty());

    auto search_ms = std::chrono::duration<double, std::milli>(search_end - search_start).count();
    std::cout << "Search time: " << search_ms << " ms" << std::endl;

    // First result should be the query itself
    assert(results->front().first == 0);

    std::cout << "✓ IVF-PQ working" << std::endl;
    return true;
}

bool test_hnsw() {
    std::cout << "\n=== Testing HNSW ===" << std::endl;

    const std::size_t n = 5000;
    const std::size_t dim = 128;

    // Generate test data
    std::vector<float> data(n * dim);
    std::vector<std::uint64_t> ids(n);

    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (auto& v : data) v = dist(gen);
    for (std::size_t i = 0; i < n; ++i) ids[i] = i;

    // Build HNSW
    index::HnswIndex index;
    index::HnswBuildParams params{
        .M = 16,
        .efConstruction = 200
    };

    auto init_result = index.init(dim, params, n);
    assert(init_result.has_value());

    auto build_start = std::chrono::high_resolution_clock::now();
    auto add_result = index.add_batch(ids.data(), data.data(), n);
    auto build_end = std::chrono::high_resolution_clock::now();

    assert(add_result.has_value());

    auto build_ms = std::chrono::duration<double, std::milli>(build_end - build_start).count();
    std::cout << "Build time: " << build_ms << " ms for " << n << " vectors" << std::endl;

    auto stats = index.get_stats();
    std::cout << "Graph: " << stats.n_nodes << " nodes, "
              << stats.n_edges << " edges, "
              << "avg degree=" << stats.avg_degree << std::endl;

    // Search
    index::HnswSearchParams search_params{
        .efSearch = 50,
        .k = 10
    };

    auto search_start = std::chrono::high_resolution_clock::now();
    auto results = index.search(data.data(), search_params);
    auto search_end = std::chrono::high_resolution_clock::now();

    assert(results.has_value());
    assert(!results->empty());

    auto search_ms = std::chrono::duration<double, std::milli>(search_end - search_start).count();
    std::cout << "Search time: " << search_ms << " ms" << std::endl;

    // First result should be the query itself
    assert(results->front().first == 0);

    std::cout << "✓ HNSW working" << std::endl;
    return true;
}

bool test_performance_targets() {
    std::cout << "\n=== Testing Performance Targets ===" << std::endl;

    const std::size_t n = 100000;
    const std::size_t dim = 128;
    const std::size_t n_queries = 100;

    // Generate dataset
    std::vector<float> data(n * dim);
    std::vector<std::uint64_t> ids(n);

    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (auto& v : data) v = dist(gen);
    for (std::size_t i = 0; i < n; ++i) ids[i] = i;

    // Test HNSW performance
    index::HnswIndex hnsw;
    index::HnswBuildParams hnsw_params{
        .M = 16,
        .efConstruction = 200
    };

    hnsw.init(dim, hnsw_params, n);

    auto build_start = std::chrono::high_resolution_clock::now();
    hnsw.add_batch(ids.data(), data.data(), n);
    auto build_end = std::chrono::high_resolution_clock::now();

    auto build_duration = std::chrono::duration<double>(build_end - build_start).count();
    double build_rate = n / build_duration;

    std::cout << "Build rate: " << build_rate << " vectors/sec" << std::endl;

    // Measure search latency
    index::HnswSearchParams search_params{
        .efSearch = 100,
        .k = 10
    };

    std::vector<double> latencies;
    for (std::size_t q = 0; q < n_queries; ++q) {
        auto start = std::chrono::high_resolution_clock::now();
        auto results = hnsw.search(data.data() + q * dim, search_params);
        auto end = std::chrono::high_resolution_clock::now();

        auto ms = std::chrono::duration<double, std::milli>(end - start).count();
        latencies.push_back(ms);
    }

    std::sort(latencies.begin(), latencies.end());

    double p50 = latencies[latencies.size() / 2];
    double p99 = latencies[latencies.size() * 99 / 100];

    std::cout << "Latency P50: " << p50 << " ms" << std::endl;
    std::cout << "Latency P99: " << p99 << " ms" << std::endl;

    // Check blueprint targets
    bool meets_build_target = build_rate >= 50000;  // ≥ 50k vectors/sec
    bool meets_p50_target = p50 <= 3.0;             // P50 ≤ 3ms
    bool meets_p99_target = p99 <= 20.0;            // P99 ≤ 20ms

    std::cout << "\nBlueprint Targets:" << std::endl;
    std::cout << "Build rate ≥ 50k/sec: " << (meets_build_target ? "✓ PASS" : "✗ FAIL") << std::endl;
    std::cout << "P50 ≤ 3ms: " << (meets_p50_target ? "✓ PASS" : "✗ FAIL") << std::endl;
    std::cout << "P99 ≤ 20ms: " << (meets_p99_target ? "✓ PASS" : "✗ FAIL") << std::endl;

    // Allow CI/CTest to relax perf targets
    if (vesper::core::safe_getenv("VESPER_RELAX_PERF")) {
        std::cout << "\n(VESPER_RELAX_PERF set) Not enforcing performance targets in this run." << std::endl;
        return true;
    }

    return meets_build_target && meets_p50_target && meets_p99_target;
}

} // namespace vesper::test

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "     Vesper Integration Test Suite     " << std::endl;
    std::cout << "========================================" << std::endl;

    bool all_passed = true;

    try {
        all_passed &= vesper::test::test_simd_kernels();
        all_passed &= vesper::test::test_kmeans();
        all_passed &= vesper::test::test_ivf_pq();
        all_passed &= vesper::test::test_hnsw();
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
