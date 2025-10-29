#include <catch2/catch_test_macros.hpp>
#include <random>
#include <vector>
#include <numeric>
#include <cmath>

#include "vesper/index/pq_fastscan.hpp"

using namespace vesper::index;

namespace {

inline void fill_rand(std::vector<float>& v, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : v) x = dist(rng);
}

inline float l2_sq_ref(const float* a, const float* b, std::size_t dim) {
    float acc = 0.0f;
    for (std::size_t i = 0; i < dim; ++i) {
        const float d = a[i] - b[i];
        acc += d * d;
    }
    return acc;
}

} // namespace

TEST_CASE("FastScanPq parity: ADC LUT sum matches decode error (self)", "[pq][fastscan][parity]") {
    std::mt19937 rng(123456u);

    const std::size_t dim = 32;              // small and fast (divisible by m)
    const std::uint32_t m = 8;               // dsub = 4
    const std::uint32_t nbits = 6;           // ksub = 64
    const std::size_t n_train = 4096;        // modest training set

    // Generate synthetic training data (represents residual space)
    std::vector<float> train(n_train * dim);
    fill_rand(train, rng);

    FastScanPqConfig cfg{ .m = m, .nbits = nbits, .block_size = 32, .use_avx512 = false };
    FastScanPq pq(cfg);
    auto r = pq.train(train.data(), n_train, dim);
    REQUIRE(r.has_value());

    // Build a set of test vectors and encode them
    const std::size_t n_vec = 65; // cross a block boundary in case blocks are used
    std::vector<float> data(n_vec * dim);
    fill_rand(data, rng);

    std::vector<std::uint8_t> codes(n_vec * m, 0);
    pq.encode(data.data(), n_vec, codes.data());

    // Decode the same codes
    std::vector<float> decoded(n_vec * dim, 0.0f);
    pq.decode(codes.data(), n_vec, decoded.data());

    // For each vector, compute ADC distance from original vector to its own code via LUTs
    // and compare with exact squared L2 distance to the decoded reconstruction.
    const float rel_eps = 1e-5f;
    for (std::size_t i = 0; i < n_vec; ++i) {
        const float* v = data.data() + i * dim;
        const std::uint8_t* code = codes.data() + i * m;
        const float* v_dec = decoded.data() + i * dim;

        // Build LUTs once per vector
        auto luts = pq.compute_lookup_tables(v);

        float adc = 0.0f;
        for (std::uint32_t sub = 0; sub < m; ++sub) {
            const std::uint8_t c = code[sub];
            adc += luts[sub][c];
        }

        const float exact = l2_sq_ref(v, v_dec, dim);
        const float denom = std::max(1.0f, std::fabs(exact));
        REQUIRE(std::fabs(adc - exact) <= rel_eps * denom);
    }
}

