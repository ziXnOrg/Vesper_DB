/** Test HNSW with batch addition */

#include <iostream>
#include <vector>
#include <random>
#include <cstdlib>
#include "vesper/index/hnsw.hpp"

static inline std::uint32_t getenv_u32(const char* key, std::uint32_t fallback) {
    if (const char* v = std::getenv(key)) {
        try { return static_cast<std::uint32_t>(std::stoul(v)); } catch (...) { }
    }
    return fallback;
}

static inline bool getenv_bool(const char* key, bool fallback) {
    if (const char* v = std::getenv(key)) {
        std::string s(v); for (auto& c : s) c = std::tolower(c);
        if (s == "1" || s == "true" || s == "yes" || s == "on") return true;
        if (s == "0" || s == "false" || s == "no" || s == "off") return false;
    }
    return fallback;
}

int main() {
    std::cout << "[TEST] BATCH_SMOKE_V2" << std::endl;

    using namespace vesper::index;

    const std::size_t n = 5000;
    const std::size_t dim = 128;

    // Generate data
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> data(n * dim);
    std::vector<std::uint64_t> ids(n);

    for (std::size_t i = 0; i < n * dim; ++i) {
        data[i] = dist(gen);
    }
    for (std::size_t i = 0; i < n; ++i) {
        ids[i] = i;
    }

    // Build index
    HnswIndex index;
    HnswBuildParams params{
        .M = 16,
        .efConstruction = 150,
        .seed = 42,
        .extend_candidates = true,
        .keep_pruned_connections = true,
        .max_M = 16,
        .max_M0 = 32,
        .num_threads = 4,
        .adaptive_ef = false,
        .efConstructionUpper = 0
    };

    // Env overrides
    params.num_threads = getenv_u32("VESPER_NUM_THREADS", params.num_threads);
    params.efConstruction = getenv_u32("VESPER_EFC", params.efConstruction);
    params.efConstructionUpper = getenv_u32("VESPER_EFC_UPPER", params.efConstructionUpper);
    params.adaptive_ef = getenv_bool("VESPER_ADAPTIVE_EF", params.adaptive_ef);


    auto init_result = index.init(dim, params, n);
    if (!init_result.has_value()) {
        std::cerr << "Failed to init" << std::endl;
        return 1;
    }

    std::cout << "Adding " << n << " vectors in batch..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    auto result = index.add_batch(ids.data(), data.data(), n);
    if (!result.has_value()) {
        std::cerr << "Failed to add batch" << std::endl;
        return 1;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(end - start).count();

    std::cout << "Added in " << duration << " seconds (" << n/duration << " vec/sec)" << std::endl;

    // Test connectivity using base-layer BFS (not bounded by efSearch/k)
    auto stats = index.get_stats();
    auto reachable = index.reachable_count_base_layer();

    std::cout << "\nStats:" << std::endl;
    std::cout << "  Nodes: " << stats.n_nodes << std::endl;
    std::cout << "  Reachable: " << reachable << std::endl;
    std::cout << "  Coverage: " << (100.0 * reachable / stats.n_nodes) << "%" << std::endl;

    if (reachable < n * 0.95) {
        std::cerr << "ERROR: Poor connectivity!" << std::endl;
        return 1;
    }

    std::cout << "SUCCESS: Good connectivity" << std::endl;
    return 0;
}