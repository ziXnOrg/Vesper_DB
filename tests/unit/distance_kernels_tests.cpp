#include <catch2/catch_test_macros.hpp>
#include <random>
#include <vector>
#include <cmath>
#include "vesper/kernels/dispatch.hpp"

using vesper::kernels::select_backend;
using vesper::kernels::select_backend_auto;

static void fill_rand(std::vector<float>& v, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : v) x = dist(rng);
}

TEST_CASE("distance kernels: scalar vs auto l2 parity", "[kernels][simd]") {
    std::mt19937 rng(42);
    const auto& scalar = select_backend("scalar");
    const auto& auto_ops = select_backend_auto();

    for (int dim : {8, 16, 64, 128}) {
        std::vector<float> a(dim), b(dim);
        for (int t = 0; t < 256; ++t) {
            fill_rand(a, rng); fill_rand(b, rng);
            float d_scalar = scalar.l2_sq(std::span<const float>(a.data(), a.size()), std::span<const float>(b.data(), b.size()));
            float d_auto   = auto_ops.l2_sq(std::span<const float>(a.data(), a.size()), std::span<const float>(b.data(), b.size()));
            REQUIRE(std::isfinite(d_scalar));
            REQUIRE(std::isfinite(d_auto));
            // Absolute tolerance; SIMD may reorder ops but should be very close
            REQUIRE(std::fabs(d_scalar - d_auto) <= 1e-4f * (1.0f + d_scalar));
        }
    }
}

TEST_CASE("distance kernels: batch_l2 parity", "[kernels][simd][batch]") {
    std::mt19937 rng(123);
    const auto& scalar = select_backend("scalar");
    const auto& auto_ops = select_backend_auto();

    const int dim = 64;
    const int n = 257; // exercise remainder paths
    std::vector<float> q(dim), xs(n * dim);
    fill_rand(q, rng); fill_rand(xs, rng);

    std::vector<float> ds_scalar(n), ds_auto(n);
    scalar.batch_l2_sq(std::span<const float>(q.data(), dim), xs.data(), n, dim, ds_scalar.data());
    auto_ops.batch_l2_sq(std::span<const float>(q.data(), dim), xs.data(), n, dim, ds_auto.data());

    for (int i = 0; i < n; ++i) {
        REQUIRE(std::isfinite(ds_scalar[i]));
        REQUIRE(std::isfinite(ds_auto[i]));
        REQUIRE(std::fabs(ds_scalar[i] - ds_auto[i]) <= 1e-4f * (1.0f + ds_scalar[i]));
    }
}

