#include <catch2/catch_test_macros.hpp>
#include <random>
#include <vector>
#include <span>

#include "vesper/index/pq_fastscan.hpp"
#include "vesper/error.hpp"

using vesper::core::error_code;
using namespace vesper::index;

namespace {
static void fill_rand(std::vector<float>& v, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : v) x = dist(rng);
}
}

TEST_CASE("FastScanPq checked variants fail with precondition on untrained", "[pq][fastscan][preconditions]") {
    std::mt19937 rng(123);

    const std::size_t dim = 32;
    const std::uint32_t m = 8;       // dsub = 4
    const std::uint32_t nbits = 6;   // ksub = 64

    FastScanPqConfig cfg{ .m = m, .nbits = nbits, .block_size = 32, .use_avx512 = false };
    FastScanPq pq(cfg);

    // Untrained encode_checked
    std::vector<float> data(10 * dim, 0.0f);
    std::vector<std::uint8_t> codes(10 * m, 0);
    auto er = pq.encode_checked(data.data(), 10, codes.data());
    REQUIRE_FALSE(er.has_value());
    REQUIRE(er.error().code == error_code::precondition_failed);

    // Untrained decode_checked
    std::vector<float> decoded(10 * dim, 0.0f);
    auto dr = pq.decode_checked(codes.data(), 10, decoded.data());
    REQUIRE_FALSE(dr.has_value());
    REQUIRE(dr.error().code == error_code::precondition_failed);

    // Untrained compute_lookup_tables_checked
    std::vector<float> q(dim, 0.0f);
    auto lr = pq.compute_lookup_tables_checked(q.data());
    REQUIRE_FALSE(lr.has_value());
    REQUIRE(lr.error().code == error_code::precondition_failed);
}

TEST_CASE("FastScanPq checked variants succeed after train()", "[pq][fastscan][preconditions]") {
    std::mt19937 rng(456);

    const std::size_t dim = 32;
    const std::uint32_t m = 8;      // dsub = 4
    const std::uint32_t nbits = 6;  // ksub = 64

    // Ensure sufficient training samples
    const std::size_t n_train = 4096;

    std::vector<float> train(n_train * dim);
    fill_rand(train, rng);

    FastScanPqConfig cfg{ .m = m, .nbits = nbits, .block_size = 32, .use_avx512 = false };
    FastScanPq pq(cfg);

    auto tr = pq.train(train.data(), n_train, dim);
    REQUIRE(tr.has_value());

    // encode_checked
    const std::size_t n_vec = 33; // cross block boundary
    std::vector<float> data(n_vec * dim);
    fill_rand(data, rng);
    std::vector<std::uint8_t> codes(n_vec * m, 0);
    auto er = pq.encode_checked(data.data(), n_vec, codes.data());
    REQUIRE(er.has_value());

    // decode_checked
    std::vector<float> decoded(n_vec * dim, 0.0f);
    auto dr = pq.decode_checked(codes.data(), n_vec, decoded.data());
    REQUIRE(dr.has_value());

    // LUTs checked
    std::vector<float> q(dim);
    fill_rand(q, rng);
    auto lr = pq.compute_lookup_tables_checked(q.data());
    REQUIRE(lr.has_value());
}

TEST_CASE("FastScanPq checked variants succeed after import_pretrained()", "[pq][fastscan][preconditions]") {
    std::mt19937 rng(789);

    const std::size_t dim = 32;
    const std::uint32_t m = 8;      // dsub = 4
    const std::uint32_t nbits = 6;  // ksub = 64
    const std::uint32_t ksub = 1u << nbits;
    const std::size_t dsub = dim / m;

    FastScanPqConfig cfg{ .m = m, .nbits = nbits, .block_size = 32, .use_avx512 = false };
    FastScanPq pq(cfg);

    // Synthesize codebooks: rows = m * ksub; each row length dsub
    const std::size_t rows = static_cast<std::size_t>(m) * static_cast<std::size_t>(ksub);
    std::vector<float> codebooks(rows * dsub);
    fill_rand(codebooks, rng);

    pq.import_pretrained(dsub, std::span<const float>(codebooks.data(), codebooks.size()));

    // Checked variants should succeed
    const std::size_t n_vec = 16;
    std::vector<float> data(n_vec * dim);
    fill_rand(data, rng);
    std::vector<std::uint8_t> codes(n_vec * m, 0);

    auto er = pq.encode_checked(data.data(), n_vec, codes.data());
    REQUIRE(er.has_value());

    std::vector<float> decoded(n_vec * dim, 0.0f);
    auto dr = pq.decode_checked(codes.data(), n_vec, decoded.data());
    REQUIRE(dr.has_value());

    std::vector<float> q(dim);
    fill_rand(q, rng);
    auto lr = pq.compute_lookup_tables_checked(q.data());
    REQUIRE(lr.has_value());
}

