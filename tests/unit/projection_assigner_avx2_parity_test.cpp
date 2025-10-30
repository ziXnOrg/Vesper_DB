#include <catch2/catch_test_macros.hpp>
#include "vesper/index/projection_assigner.hpp"

#include <vector>
#include <random>
#include <algorithm>
#include <limits>
#include <cmath>

using namespace vesper::index;

static void fill_normal(std::vector<float>& v, float scale = 1.0f, uint32_t seed = 1337) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, scale);
    for (auto& x : v) x = nd(rng);
}

static void compute_norms(const std::vector<float>& a, std::size_t rows, std::size_t cols, std::vector<float>& norms) {
    norms.resize(rows);
    for (std::size_t i = 0; i < rows; ++i) {
        const float* r = a.data() + i * cols;
        float s = 0.0f;
        for (std::size_t k = 0; k < cols; ++k) s += r[k] * r[k];
        norms[i] = s;
    }
}

static void pack_centroids_8x16(const std::vector<float>& centroids_rm, std::size_t C, std::size_t p, std::vector<float>& pack8) {
    // Layout expected by AVX2 microkernel: for each block (8 centroids), and each k in [0,15],
    // store 8 lanes [lane=0..7] consecutively: pack[blk*(16*8) + k*8 + lane] = centroids_rm[(blk*8+lane)*p + k]
    const std::size_t blocks = (C + 7) / 8;
    pack8.assign(blocks * 16 * 8, 0.0f);
    for (std::size_t blk = 0; blk < blocks; ++blk) {
        for (int k = 0; k < 16; ++k) {
            for (int lane = 0; lane < 8; ++lane) {
                const std::size_t cj = blk * 8 + static_cast<std::size_t>(lane);
                const float v = (cj < C) ? centroids_rm[cj * p + static_cast<std::size_t>(k)] : 0.0f;
                pack8[blk * (16 * 8) + static_cast<std::size_t>(k) * 8 + static_cast<std::size_t>(lane)] = v;
            }
        }
    }
}

static void sort_pairs_per_row(std::vector<std::uint32_t>& idx, std::vector<float>& dist, std::size_t n, std::size_t L) {
    for (std::size_t i = 0; i < n; ++i) {
        auto ib = idx.begin() + i * L;
        auto db = dist.begin() + i * L;
        std::vector<std::pair<float, std::uint32_t>> pairs(L);
        for (std::size_t t = 0; t < L; ++t) pairs[t] = {db[t], ib[t]};
        std::sort(pairs.begin(), pairs.end(), [](auto& a, auto& b){ return a.first < b.first; });
        for (std::size_t t = 0; t < L; ++t) { db[t] = pairs[t].first; ib[t] = pairs[t].second; }
    }
}

TEST_CASE("projection_assigner AVX2 parity vs scalar for p=16", "[projection][avx2][parity]") {
    const std::size_t n = 32;   // two tiles
    const std::size_t p = 16;   // AVX2 microkernel constraint
    const std::size_t C = 64;   // 8-lane blocks
    const std::uint32_t L = 8;  // shortlist

    std::vector<float> qproj(n * p);
    std::vector<float> qnorm(n);
    std::vector<float> centroids_rm(C * p);
    std::vector<float> centroid_norms(C);

    fill_normal(qproj, 1.0f, 123);
    fill_normal(centroids_rm, 1.0f, 9876);

    compute_norms(qproj, n, p, qnorm);
    compute_norms(centroids_rm, C, p, centroid_norms);

    // Baseline: scalar backend (no pack)
    std::vector<std::uint32_t> idx_scalar(n * L, 0);
    std::vector<float> dist_scalar(n * L, std::numeric_limits<float>::infinity());
    {
        ProjScreenInputs in{qproj.data(), qnorm.data(), n, p, centroids_rm.data(), centroid_norms.data(), /*centroids_pack8*/ nullptr, C, L};
        ProjScreenOutputs out{idx_scalar.data(), dist_scalar.data()};
        projection_screen_select(in, out);
        sort_pairs_per_row(idx_scalar, dist_scalar, n, L);
    }

    // AVX2 path
    std::vector<float> centroids_pack8;
    pack_centroids_8x16(centroids_rm, C, p, centroids_pack8);

    std::vector<std::uint32_t> idx_avx(n * L, 0);
    std::vector<float> dist_avx(n * L, std::numeric_limits<float>::infinity());
    {
        ProjScreenInputs in{qproj.data(), qnorm.data(), n, p, centroids_rm.data(), centroid_norms.data(), centroids_pack8.data(), C, L};
        ProjScreenOutputs out{idx_avx.data(), dist_avx.data()};
        projection_screen_select(in, out);
        sort_pairs_per_row(idx_avx, dist_avx, n, L);
    }

    // Compare per row within tolerance; allow identical or near-identical ordering
    const float eps = 1e-4f;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t t = 0; t < L; ++t) {
            REQUIRE(idx_avx[i * L + t] == idx_scalar[i * L + t]);
            REQUIRE(std::fabs(dist_avx[i * L + t] - dist_scalar[i * L + t]) <= eps * (1.0f + std::fabs(dist_scalar[i * L + t])));
        }
    }
}

