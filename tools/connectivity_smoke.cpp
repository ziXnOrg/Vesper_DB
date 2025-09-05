#include <iostream>
#include <vector>
#include <random>
#include <chrono>

#include "vesper/index/hnsw.hpp"

using namespace vesper::index;

int main() {
    const std::size_t n = 5000;
    const std::size_t dim = 128;

    // Generate random data
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> data(n * dim);
    std::vector<std::uint64_t> ids(n);
    for (std::size_t i = 0; i < n * dim; ++i) data[i] = dist(gen);
    for (std::size_t i = 0; i < n; ++i) ids[i] = i;

    HnswBuildParams params{
        .M = 16,
        .efConstruction = 200,
        .seed = 42,
        .extend_candidates = true,
        .keep_pruned_connections = true,
        .max_M = 16,
        .max_M0 = 32,
        .num_threads = 0
    };

    HnswIndex index;
    auto init_res = index.init(dim, params, n);
    if (!init_res.has_value()) {
        std::cerr << "init failed\n";
        return 1;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    auto add_res = index.add_batch(ids.data(), data.data(), n);
    auto t1 = std::chrono::high_resolution_clock::now();
    if (!add_res.has_value()) {
        std::cerr << "add_batch failed\n";
        return 1;
    }

    double secs = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "Built in " << secs << " sec (" << (n / secs) << " vec/s)\n";

    auto stats = index.get_stats();
    auto reachable = index.reachable_count_base_layer();
    double cov = (stats.n_nodes > 0) ? (100.0 * reachable / stats.n_nodes) : 0.0;

    std::cout << "Nodes=" << stats.n_nodes
              << " Reachable=" << reachable
              << " Coverage=" << cov << "%\n";

    if (reachable < n * 0.95) {
        std::cerr << "Connectivity < 95%\n";
        return 2;
    }
    std::cout << "Connectivity >= 95%\n";
    return 0;
}

