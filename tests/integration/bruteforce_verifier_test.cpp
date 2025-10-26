#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_all.hpp>

#include "tests/integration/dataset_loader.hpp"
#include "vesper/index/index_manager.hpp"

#include <algorithm>
#include <limits>
#include <numeric>

using namespace vesper;
using namespace vesper::index;
using vesper::test::DatasetLoader;
using vesper::test::SearchMetrics;

namespace {
static float l2_sq(const float* a, const float* b, std::size_t dim) {
    float acc = 0.0f;
    for (std::size_t i = 0; i < dim; ++i) {
        float d = a[i] - b[i];
        acc += d * d;
    }
    return acc;
}
}

TEST_CASE("Brute-force verifier on subsample (ID/mapping/metric check)", "[integration][bfv]") {
    auto ds = DatasetLoader::load_benchmark("sift-128-euclidean", "data");
    if (!ds) {
        WARN("Dataset not found; skipping brute-force verifier. Run scripts/download_datasets.py");
        SUCCEED();
        return;
    }

    const std::size_t dim = ds->info.dimension;
    const std::size_t n_full = ds->info.num_vectors;
    REQUIRE(dim > 0);
    REQUIRE(n_full > 0);

    // Subsample base and queries for a fast, exact verification
    const std::size_t n_sub = std::min<std::size_t>(100000, n_full);
    const std::size_t nq = std::min<std::size_t>(200, ds->info.num_queries);
    const std::uint32_t k = 10;

    std::vector<float> base_sub(n_sub * dim);
    std::copy(ds->base_vectors.begin(), ds->base_vectors.begin() + n_sub * dim, base_sub.begin());

    std::vector<float> queries_sub(nq * dim);
    std::copy(ds->query_vectors.begin(), ds->query_vectors.begin() + nq * dim, queries_sub.begin());

    // Compute brute-force top-k over the subsampled base
    std::vector<std::uint32_t> bf_gt; bf_gt.reserve(nq * k);
    for (std::size_t qi = 0; qi < nq; ++qi) {
        const float* q = &queries_sub[qi * dim];
        std::vector<std::pair<float, std::uint32_t>> dists; dists.reserve(n_sub);
        for (std::uint32_t id = 0; id < n_sub; ++id) {
            const float* v = &base_sub[id * dim];
            dists.emplace_back(l2_sq(q, v, dim), id);
        }
        std::partial_sort(dists.begin(), dists.begin() + k, dists.end(),
                          [](const auto& a, const auto& b){ return a.first < b.first; });
        for (std::size_t j = 0; j < k; ++j) bf_gt.push_back(dists[j].second);
    }

    // Build index over the same subsample and search
    IndexManager mgr(dim);
    IndexBuildConfig cfg;
    cfg.strategy = SelectionStrategy::Auto;
    cfg.ivf_params.nlist = 4096; // deeper coarse quantization even for 100k
    cfg.ivf_params.m = 16;
    cfg.ivf_params.nbits = 8;

    auto br = mgr.build(base_sub.data(), n_sub, cfg);
    REQUIRE(br.has_value());

    QueryConfig qcfg; qcfg.k = k; qcfg.ef_search = 200; qcfg.nprobe = 512;

    std::vector<std::uint32_t> results_flat; results_flat.reserve(nq * k);
    for (std::size_t i = 0; i < nq; ++i) {
        auto r = mgr.search(&queries_sub[i * dim], qcfg);
        REQUIRE(r.has_value());
        REQUIRE(r->size() == k);
        for (std::size_t j = 0; j < k; ++j) results_flat.push_back(static_cast<std::uint32_t>((*r)[j].first));
    }

    float recall = SearchMetrics::compute_recall(results_flat, bf_gt, nq, k, k);
    INFO("Brute-force verifier recall@" << k << " = " << recall);
    // This test validates dataset/ID/metric plumbing. We only require reasonable recall.
    REQUIRE(recall >= 0.60f);
}

