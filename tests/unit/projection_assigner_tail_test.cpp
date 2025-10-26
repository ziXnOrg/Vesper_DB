#include <catch2/catch_test_macros.hpp>
#include "vesper/index/projection_assigner.hpp"

#include <vector>
#include <random>
#include <limits>
#include <cmath>

using namespace vesper::index;

static void fill_random(std::vector<float>& v, float scale = 1.0f, uint32_t seed = 12345) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, scale);
    for (auto& x : v) x = nd(rng);
}

TEST_CASE("projection_assigner handles AVX2 tail rows (n % 16 != 0) without crash and sane outputs", "[projection][avx2][tail]") {
    const std::size_t n = 17; // triggers remainder handling
    const std::size_t p = 16; // AVX2 microkernel path
    const std::size_t C = 32; // 4 blocks of 8
    const std::uint32_t L = 8; // shortlist per row

    std::vector<float> qproj(n * p);
    std::vector<float> qnorm(n);
    std::vector<float> centroids_rm(C * p);
    std::vector<float> centroid_norms(C);

    fill_random(qproj);
    fill_random(centroids_rm, 1.0f, 6789);

    for (std::size_t i = 0; i < n; ++i) {
        float s = 0.0f; const float* qp = qproj.data() + i * p;
        for (std::size_t k = 0; k < p; ++k) s += qp[k] * qp[k];
        qnorm[i] = s;
    }
    for (std::size_t j = 0; j < C; ++j) {
        float s = 0.0f; const float* yc = centroids_rm.data() + j * p;
        for (std::size_t k = 0; k < p; ++k) s += yc[k] * yc[k];
        centroid_norms[j] = s;
    }

    const std::size_t blocks = (C + 7) / 8;
    std::vector<float> centroids_pack8(blocks * 16 * 8, 0.0f);
    for (std::size_t blk = 0; blk < blocks; ++blk) {
        for (int k = 0; k < 16; ++k) {
            for (int lane = 0; lane < 8; ++lane) {
                const std::size_t cj = blk * 8 + static_cast<std::size_t>(lane);
                const float v = (cj < C) ? centroids_rm[cj * p + static_cast<std::size_t>(k)] : 0.0f;
                centroids_pack8[blk * (16 * 8) + k * 8 + lane] = v;
            }
        }
    }

    std::vector<std::uint32_t> cand_idx(n * L, 0);
    std::vector<float> cand_dist(n * L, std::numeric_limits<float>::infinity());

    ProjScreenInputs in{qproj.data(), qnorm.data(), n, p, centroids_rm.data(), centroid_norms.data(), centroids_pack8.data(), C, L};
    ProjScreenOutputs out{cand_idx.data(), cand_dist.data()};

    REQUIRE_NOTHROW(projection_screen_select(in, out));

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t t = 0; t < L; ++t) {
            const std::uint32_t idx = cand_idx[i * L + t];
            REQUIRE(idx < C);
            const float d = cand_dist[i * L + t];
            REQUIRE(std::isfinite(d));
        }
    }
}

