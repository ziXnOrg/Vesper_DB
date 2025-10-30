#include <catch2/catch_test_macros.hpp>
#include <random>
#include <vector>

#include "vesper/index/pq_fastscan.hpp"

using namespace vesper::index;

static void fill_rand(std::vector<float>& v, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : v) x = dist(rng);
}

TEST_CASE("FastScanPq training fails when n < ksub (propagates error and keeps untrained)", "[pq][fastscan][train]") {
    std::mt19937 rng(123);
    const std::size_t dim = 32;
    const std::uint32_t m = 8;
    const std::uint32_t nbits = 8; // ksub = 256
    const std::size_t n_train = 64; // < ksub to force failure

    std::vector<float> train(n_train * dim);
    fill_rand(train, rng);

    FastScanPqConfig cfg{ .m = m, .nbits = nbits, .block_size = 32, .use_avx512 = false };
    FastScanPq pq(cfg);

    auto r = pq.train(train.data(), n_train, dim);
    REQUIRE_FALSE(r.has_value());
    REQUIRE_FALSE(pq.is_trained());
}

TEST_CASE("FastScanPq training fails when dim % m != 0 (precondition)", "[pq][fastscan][train]") {
    std::mt19937 rng(456);
    const std::size_t dim = 30; // not divisible by m
    const std::uint32_t m = 8;
    const std::uint32_t nbits = 6; // ksub = 64
    const std::size_t n_train = 512;

    std::vector<float> train(n_train * dim);
    fill_rand(train, rng);

    FastScanPqConfig cfg{ .m = m, .nbits = nbits, .block_size = 32, .use_avx512 = false };
    FastScanPq pq(cfg);

    auto r = pq.train(train.data(), n_train, dim);
    REQUIRE_FALSE(r.has_value());
    REQUIRE_FALSE(pq.is_trained());
}

TEST_CASE("FastScanPq training succeeds with valid params", "[pq][fastscan][train]") {
    std::mt19937 rng(789);
    const std::size_t dim = 32;
    const std::uint32_t m = 8;
    const std::uint32_t nbits = 6; // ksub = 64
    const std::size_t n_train = 4096;

    std::vector<float> train(n_train * dim);
    fill_rand(train, rng);

    FastScanPqConfig cfg{ .m = m, .nbits = nbits, .block_size = 32, .use_avx512 = false };
    FastScanPq pq(cfg);

    auto r = pq.train(train.data(), n_train, dim);
    REQUIRE(r.has_value());
    REQUIRE(pq.is_trained());
}

