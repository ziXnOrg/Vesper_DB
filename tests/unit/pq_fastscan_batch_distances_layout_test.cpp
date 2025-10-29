#include <catch2/catch_test_macros.hpp>
#include <random>
#include <vector>
#include <cmath>

#include "vesper/index/pq_fastscan.hpp"

using namespace vesper::index;

namespace {

static void fill_rand(std::vector<float>& v, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : v) x = dist(rng);
}

static FastScanPq make_trained_pq(std::size_t dim, std::uint32_t m, std::uint32_t nbits, std::uint32_t block_size) {
    FastScanPqConfig cfg{ .m = m, .nbits = nbits, .block_size = block_size, .use_avx512 = false };
    FastScanPq pq(cfg);

    const std::size_t n_train = 2048;
    std::vector<float> train(n_train * dim);
    std::mt19937 rng(123);
    fill_rand(train, rng);

    auto r = pq.train(train.data(), n_train, dim);
    REQUIRE(r.has_value());
    return pq;
}

} // namespace

TEST_CASE("compute_batch_distances handles empty blocks safely", "[pq][fastscan][batch]") {
    const std::size_t dim = 32;
    const std::uint32_t m = 8;
    const std::uint32_t nbits = 6;

    auto pq = make_trained_pq(dim, m, nbits, 32);

    // Two queries
    std::vector<float> queries(2 * dim, 0.0f);
    std::mt19937 rng(7);
    fill_rand(queries, rng);

    // Empty blocks
    std::vector<PqCodeBlock> blocks;

    // No distances storage needed when blocks.empty(); call should be a no-op and not crash
    std::vector<float> distances; // size 0

    // Prior bug: dereferenced blocks[0] (UB). This test should pass without crashes after fix.
    compute_batch_distances(pq, queries.data(), 2, blocks, distances.data());
    SUCCEED("No crash on empty blocks");
}

TEST_CASE("compute_batch_distances: partial single block (n < block_size)", "[pq][fastscan][batch]") {
    const std::size_t dim = 32;
    const std::uint32_t m = 8;
    const std::uint32_t nbits = 6;
    const std::uint32_t block_size = 32;

    auto pq = make_trained_pq(dim, m, nbits, block_size);

    // Build a single partial block (n < block_size)
    const std::size_t n_vec = block_size / 2; // 16
    std::vector<float> data(n_vec * dim);
    std::mt19937 rng(11);
    fill_rand(data, rng);
    auto blocks = pq.encode_blocks(data.data(), n_vec);
    REQUIRE(blocks.size() == 1);
    REQUIRE(blocks[0].size() == n_vec);

    // Two queries to verify per-query stride placement
    const std::size_t n_queries = 2;
    std::vector<float> queries(n_queries * dim);
    fill_rand(queries, rng);

    const std::size_t row_stride = blocks.size() * pq.config().block_size; // must be 32
    std::vector<float> distances(n_queries * row_stride, -123.0f);

    compute_batch_distances(pq, queries.data(), n_queries, blocks, distances.data());

    const float eps = 1e-5f;
    for (std::size_t q = 0; q < n_queries; ++q) {
        auto luts = pq.compute_lookup_tables(queries.data() + q * dim);
        // Validate only first n_vec entries in the block; padding is unspecified
        for (std::uint32_t i = 0; i < blocks[0].size(); ++i) {
            float acc = 0.0f;
            for (std::uint32_t sub = 0; sub < m; ++sub) {
                const std::uint8_t code = blocks[0].get_subquantizer_codes(sub)[i];
                acc += luts[sub][code];
            }
            const float got = distances[q * row_stride + i];
            REQUIRE(std::fabs(acc - got) <= eps * (1.0f + std::fabs(acc)));
        }
    }
}

TEST_CASE("compute_batch_distances: multi-block with partial tail; correct per-query stride", "[pq][fastscan][batch]") {
    const std::size_t dim = 32;
    const std::uint32_t m = 8;
    const std::uint32_t nbits = 6;
    const std::uint32_t block_size = 32;

    auto pq = make_trained_pq(dim, m, nbits, block_size);

    // Build two blocks: one full, one partial
    const std::size_t n_vec = block_size + block_size / 2; // 48
    std::vector<float> data(n_vec * dim);
    std::mt19937 rng(19);
    fill_rand(data, rng);
    auto blocks = pq.encode_blocks(data.data(), n_vec);
    REQUIRE(blocks.size() == 2);
    REQUIRE(blocks[0].size() == block_size);
    REQUIRE(blocks[1].size() == block_size / 2);

    // Two queries to assert row stride placement
    const std::size_t n_queries = 2;
    std::vector<float> queries(n_queries * dim);
    fill_rand(queries, rng);

    const std::size_t row_stride = blocks.size() * pq.config().block_size; // 2 * 32 = 64
    std::vector<float> distances(n_queries * row_stride, 0.0f);

    compute_batch_distances(pq, queries.data(), n_queries, blocks, distances.data());

    const float eps = 1e-5f;
    for (std::size_t q = 0; q < n_queries; ++q) {
        auto luts = pq.compute_lookup_tables(queries.data() + q * dim);

        // Block 0 (full)
        for (std::uint32_t i = 0; i < blocks[0].size(); ++i) {
            float acc = 0.0f;
            for (std::uint32_t sub = 0; sub < m; ++sub) {
                const std::uint8_t code = blocks[0].get_subquantizer_codes(sub)[i];
                acc += luts[sub][code];
            }
            const float got = distances[q * row_stride + 0 * block_size + i];
            REQUIRE(std::fabs(acc - got) <= eps * (1.0f + std::fabs(acc)));
        }
        // Block 1 (partial tail)
        for (std::uint32_t i = 0; i < blocks[1].size(); ++i) {
            float acc = 0.0f;
            for (std::uint32_t sub = 0; sub < m; ++sub) {
                const std::uint8_t code = blocks[1].get_subquantizer_codes(sub)[i];
                acc += luts[sub][code];
            }
            const float got = distances[q * row_stride + 1 * block_size + i];
            REQUIRE(std::fabs(acc - got) <= eps * (1.0f + std::fabs(acc)));
        }
    }
}

