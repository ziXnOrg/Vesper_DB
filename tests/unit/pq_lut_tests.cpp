#include <catch2/catch_test_macros.hpp>
#include <random>
#include <vector>
#include <numeric>
#include <cmath>
#include "vesper/index/pq_fastscan.hpp"

using namespace vesper::index;

static void fill_rand(std::vector<float>& v, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : v) x = dist(rng);
}

TEST_CASE("PQ LUT vs manual accumulation", "[pq][lut]") {
    std::mt19937 rng(7);
    const std::size_t dim = 32;  // small and fast
    const std::uint32_t m = 8;   // dsub=4
    const std::uint32_t nbits = 4; // ksub=16
    const std::size_t n_train = 4000;

    // Generate synthetic residual training data
    std::vector<float> train(n_train * dim);
    fill_rand(train, rng);

    FastScanPqConfig cfg{ .m = m, .nbits = nbits, .block_size = 32, .use_avx512 = false };
    FastScanPq pq(cfg);
    auto r = pq.train(train.data(), n_train, dim);
    REQUIRE(r.has_value());

    // Build some codes from random data
    const std::size_t n_vec = 65; // cross block boundary
    std::vector<float> data(n_vec * dim);
    fill_rand(data, rng);
    auto blocks = pq.encode_blocks(data.data(), n_vec);

    // Random query
    std::vector<float> q(dim); fill_rand(q, rng);

    // Compute via helper
    std::vector<float> dists(blocks.size() * cfg.block_size, 0.0f);
    pq.compute_distances(q.data(), blocks, dists.data());

    // Manual accumulation using LUTs
    auto luts = pq.compute_lookup_tables(q.data());

    std::size_t offset = 0;
    for (const auto& blk : blocks) {
        const std::uint32_t bs = blk.size();
        for (std::uint32_t i = 0; i < bs; ++i) {
            float acc = 0.0f;
            for (std::uint32_t sub = 0; sub < m; ++sub) {
                const std::uint8_t code = blk.get_subquantizer_codes(sub)[i];
                acc += luts[sub][code];
            }
            REQUIRE(std::fabs(acc - dists[offset + i]) <= 1e-5f * (1.0f + std::fabs(acc)));
        }
        offset += cfg.block_size;
    }
}

