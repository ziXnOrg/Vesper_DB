#include <catch2/catch_test_macros.hpp>
#include <random>
#include <vector>
#include <numeric>
#include <cmath>

#include "vesper/index/product_quantizer.hpp"

using namespace vesper::index;

static void fill_rand(std::vector<float>& v, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : v) x = dist(rng);
}

static float l2_sq_naive(const float* a, const float* b, std::size_t dim) {
    double acc = 0.0;
    for (std::size_t i = 0; i < dim; ++i) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        acc += d * d;
    }
    return static_cast<float>(acc);
}

TEST_CASE("ProductQuantizer OPQ: ADC self-distance matches decode error", "[pq][opq]") {
    std::mt19937 rng(12345);

    const std::size_t dim = 16;
    const std::uint32_t m = 4;      // dsub = 4
    const std::uint32_t nbits = 6;  // ksub = 64
    const std::size_t n_train = 512;

    // Training data
    std::vector<float> train(n_train * dim);
    fill_rand(train, rng);

    // Train OPQ (iter=1 so we only apply initial rotation and train PQ once)
    PqTrainParams pqp{};
    pqp.m = m;
    pqp.nbits = nbits;
    pqp.max_iter = 15;
    pqp.epsilon = 1e-4f;
    pqp.seed = 42;
    pqp.verbose = false;

    OpqParams op{};
    op.iter = 1;             // skip rotation updates; use initial rotation
    op.init_rotation = true; // PCA init path (stubbed to identity if not implemented)
    op.reg = 0.01f;

    ProductQuantizer pq;
    auto r = pq.train_opq(train.data(), n_train, dim, pqp, op);
    REQUIRE(r.has_value());

    // Pick a vector and roundtrip through encode/decode
    const float* v = train.data();
    std::vector<std::uint8_t> code(m);
    REQUIRE(pq.encode_one(v, code.data()).has_value());

    std::vector<float> recon(dim, 0.0f);
    REQUIRE(pq.decode(code.data(), 1, recon.data()).has_value());

    const float decode_err = l2_sq_naive(v, recon.data(), dim);

    // ADC self-distance using the same query should match decode error within tolerance
    std::vector<float> table(m * (1u << nbits));
    REQUIRE(pq.compute_distance_table(v, table.data()).has_value());
    float d = -1.0f;
    REQUIRE(pq.compute_distances_adc(table.data(), code.data(), 1, &d).has_value());

    REQUIRE(std::isfinite(d));
    // Orthogonal rotation preserves squared distances; allow a tiny numerical tolerance
    REQUIRE(std::fabs(d - decode_err) <= 1e-4f * (1.0f + std::fabs(decode_err)));
}

