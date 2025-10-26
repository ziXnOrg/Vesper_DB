#include <catch2/catch_test_macros.hpp>
#include <vector>
#include <random>
#include <catch2/catch_approx.hpp>

#include "vesper/index/kmeans_elkan.hpp"

using vesper::index::kmeans_parallel_init;
using vesper::index::KmeansElkan;

namespace {
std::vector<float> make_data(std::size_t n, std::size_t dim, std::uint32_t seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> d(0.0f, 1.0f);
    std::vector<float> data(n * dim);
    for (std::size_t i = 0; i < n * dim; ++i) data[i] = d(gen);
    return data;
}
}

TEST_CASE("k-means|| initialization produces k centers and is deterministic", "[kmeans_parallel]") {
    const std::size_t n = 500;
    const std::size_t dim = 16;
    const std::uint32_t k = 8;
    const std::uint32_t seed = 123;

    auto data = make_data(n, dim, seed + 1);

    auto c1 = kmeans_parallel_init(data.data(), n, dim, k, /*rounds=*/5, /*l=*/0, seed);
    auto c2 = kmeans_parallel_init(data.data(), n, dim, k, /*rounds=*/5, /*l=*/0, seed);

    REQUIRE(c1.size() == k);
    REQUIRE(c2.size() == k);
    for (const auto& cent : c1) REQUIRE(cent.size() == dim);

    // Determinism: same seed -> same centers
    REQUIRE(c1 == c2);

    // Different seed -> likely different
    auto c3 = kmeans_parallel_init(data.data(), n, dim, k, /*rounds=*/5, /*l=*/0, seed + 7);
    // Allow possibility of equality but expect difference in practice
    if (c1 == c3) {
        WARN("k-means|| produced same centers with different seed (rare but possible)");
    } else {
        SUCCEED();
    }
}



TEST_CASE("k-means|| initialization has comparable quality to k-means++", "[kmeans_parallel][quality]") {
    using vesper::index::kmeans_plusplus_init; // only for reference if needed

    const std::size_t dim = 16;
    const std::uint32_t seed = 777;
    std::mt19937 gen(seed);

    // Generate 4 well-separated Gaussian clusters, 800 points total
    const std::size_t clusters = 4;
    const std::size_t points_per = 200;
    const std::size_t n = clusters * points_per;

    // Create cluster means separated by 10.0 in random orthogonal-ish directions
    std::vector<std::vector<float>> means(clusters, std::vector<float>(dim, 0.0f));
    for (std::size_t c = 0; c < clusters; ++c) {
        means[c][c % dim] = static_cast<float>(10.0 * (c + 1));
    }

    std::normal_distribution<float> noise(0.0f, 0.5f);
    std::vector<float> data(n * dim, 0.0f);
    for (std::size_t c = 0; c < clusters; ++c) {
        for (std::size_t i = 0; i < points_per; ++i) {
            const std::size_t idx = c * points_per + i;
            float* dst = data.data() + idx * dim;
            for (std::size_t d = 0; d < dim; ++d) {
                dst[d] = means[c][d] + noise(gen);
            }
        }
    }

    const std::uint32_t k = static_cast<std::uint32_t>(clusters);

    // Configure Elkan with identical params except init method
    KmeansElkan elkan_pp;
    KmeansElkan::Config cfg_pp{ .k = k, .max_iter = 50, .epsilon = 1e-5f, .seed = seed, .use_parallel = true, .verbose = false, .init_method = KmeansElkan::Config::InitMethod::KMeansPlusPlus };

    KmeansElkan elkan_par;
    KmeansElkan::Config cfg_par = cfg_pp;
    cfg_par.init_method = KmeansElkan::Config::InitMethod::KMeansParallel;
    cfg_par.kmeans_parallel_rounds = 5;
    cfg_par.kmeans_parallel_oversampling = 0; // default 2k

    auto r_pp = elkan_pp.cluster(data.data(), n, dim, cfg_pp);
    auto r_par = elkan_par.cluster(data.data(), n, dim, cfg_par);

    REQUIRE(r_pp.has_value());
    REQUIRE(r_par.has_value());

    const double inertia_pp = r_pp->inertia;
    const double inertia_par = r_par->inertia;

    // Quality: within 10% relative difference either way
    const double max_ratio = std::max(inertia_pp, inertia_par) / std::max(1e-12, std::min(inertia_pp, inertia_par));
    REQUIRE(max_ratio <= 1.10);

    // Iteration counts are within 3
    const auto it_pp = r_pp->iterations;
    const auto it_par = r_par->iterations;
    REQUIRE(static_cast<int>(std::abs(static_cast<long long>(it_pp) - static_cast<long long>(it_par))) <= 3);
}
