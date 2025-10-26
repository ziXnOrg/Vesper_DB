#include <catch2/catch_test_macros.hpp>
#include "vesper/index/ivf_pq.hpp"

#include <vector>
#include <random>
#include <numeric>

using namespace vesper::index;

static void fill_random(std::vector<float>& v, float scale = 1.0f, uint32_t seed = 4242) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, scale);
    for (auto& x : v) x = nd(rng);
}

TEST_CASE("Projection coarse assigner stability smoke: train+add small batch", "[projection][ivfpq][smoke]") {
    const std::size_t dim = 128;
    const std::size_t n_train = 2048;
    const std::size_t n_add = 512;

    std::vector<float> data((n_train + n_add) * dim);
    fill_random(data);

    std::vector<std::uint64_t> ids(n_add);
    std::iota(ids.begin(), ids.end(), 1000);

    IvfPqIndex index;

    IvfPqTrainParams tp{};
    tp.nlist = 128;
    tp.m = 8;
    tp.nbits = 8;
    tp.coarse_assigner = CoarseAssigner::Projection;
    tp.projection_dim = 16; // AVX2 optimized path
    tp.validate_ann_assignment = true; // enable sampled validation (just telemetry)
    tp.validate_ann_sample_rate = 0.05f;

    // Train
    auto tr = index.train(data.data(), dim, n_train, tp);
    REQUIRE(tr.has_value());
    REQUIRE(index.is_trained());
    REQUIRE(index.dimension() == dim);

    // Add (should not throw). We only assert basic counters; projection accuracy is not expected.
    auto add_res = index.add(ids.data(), data.data() + n_train * dim, n_add);
    REQUIRE(add_res.has_value());

    auto s = index.get_stats();
    REQUIRE(s.n_vectors == n_add);
    REQUIRE(s.n_lists == tp.nlist);
    // Projection path enables ANN telemetry in coarse assignment
    REQUIRE(s.ann_enabled == true);
    REQUIRE(s.ann_assignments == n_add);
    // Validated count is sample-based and environment-dependent; ensure within [0, n_add]
    REQUIRE(s.ann_validated <= n_add);
}

