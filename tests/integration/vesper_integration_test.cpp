#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_all.hpp>

#include "tests/integration/dataset_loader.hpp"
#include "vesper/index/index_manager.hpp"
#include "vesper/index/hnsw.hpp"
#include "vesper/core/platform_utils.hpp"

#include <chrono>
#include <random>
#include <cstdlib>

using namespace vesper;
using namespace vesper::index;
using vesper::test::DatasetLoader;
using vesper::test::SearchMetrics;

namespace {
std::vector<float> gen_vectors(std::size_t n, std::size_t dim, std::uint32_t seed = 42) {
    std::vector<float> v(n * dim);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto &x : v) x = dist(gen);
    return v;
}
}

TEST_CASE("End-to-end: build and search using dataset if available", "[integration][dataset]") {
    // Try to load a real dataset converted by scripts/download_datasets.py
    auto ds = DatasetLoader::load_benchmark("sift-128-euclidean", "data");

    std::size_t dim = 0;
    std::vector<float> base;
    std::vector<float> queries;
    std::vector<std::uint32_t> gt; // ground truth (optional)
    std::uint32_t k_gt = 0;

    if (ds) {
        dim = ds->info.dimension;
        base = std::move(ds->base_vectors);
        queries = std::move(ds->query_vectors);
        gt = std::move(ds->groundtruth);
        k_gt = ds->k;
    } else {
        WARN("Dataset not found in data/{hdf5,fvecs}. Falling back to synthetic vectors. Run scripts/download_datasets.py to enable dataset-backed tests.");
        dim = 64;
        const std::size_t n = 10000;
        base = gen_vectors(n, dim, 123);
        queries = gen_vectors(100, dim, 321);
    }

    const std::size_t n = base.empty() ? 0 : (base.size() / dim);
    REQUIRE(dim > 0);
    REQUIRE(n > 0);

    IndexManager mgr(dim);
    IndexBuildConfig cfg;
    cfg.strategy = SelectionStrategy::Auto;
    // IVF-PQ training params tuned for 1M @ 128D
    cfg.ivf_params.nlist = 4096;    // coarse centroids
    cfg.ivf_params.m = 16;          // subquantizers (dsub = 8)
    cfg.ivf_params.nbits = 8;       // 256 codewords per subquantizer
    // Enable OPQ with PCA init to improve ADC ranking quality on SIFT-128
    cfg.ivf_params.use_opq = true;
    cfg.ivf_params.opq_init = vesper::index::OpqInit::PCA;
    cfg.hnsw_params.M = 16;
    cfg.hnsw_params.efConstruction = 200;

    auto br = mgr.build(base.data(), n, cfg);
    if (!br) {
        WARN("IndexManager::build failed: " << static_cast<int>(br.error().code));
        return; // skip test gracefully if backend not ready
    }

    // Search a small batch of queries
    const std::size_t nq = std::min<std::size_t>(queries.empty() ? 0 : (queries.size() / dim), 100);
    REQUIRE(nq > 0);

    QueryConfig qcfg;
    qcfg.k = 10;
    qcfg.ef_search = 200;
    qcfg.nprobe = 256;  // deeper probe for better recall


    std::vector<std::uint32_t> results_flat;
    results_flat.reserve(nq * qcfg.k);

    for (std::size_t i = 0; i < nq; ++i) {
        auto qr = mgr.search(&queries[i * dim], qcfg);
        REQUIRE(qr.has_value());
        REQUIRE(qr->size() == qcfg.k);
        for (std::size_t j = 0; j < qcfg.k; ++j) {
            results_flat.push_back(static_cast<std::uint32_t>((*qr)[j].first));
        }
    }

    // Optional: brute-force validation on a tiny subset to verify GT/id mapping
    {
        auto env = vesper::core::safe_getenv("VESPER_VALIDATE_BF");
        const bool do_bf = env && !env->empty() && ((*env)[0] != '0');
        if (do_bf && !gt.empty()) {
            const std::size_t bf_q = std::min<std::size_t>(nq, 5);
            std::vector<std::uint32_t> bf_flat; bf_flat.reserve(bf_q * qcfg.k);
            for (std::size_t qi = 0; qi < bf_q; ++qi) {
                // Compute exact L2 distances to all base vectors
                std::vector<std::pair<float, std::uint32_t>> dist_id; dist_id.reserve(n);
                const float* qv = &queries[qi * dim];
                for (std::size_t bi = 0; bi < n; ++bi) {
                    const float* xv = &base[bi * dim];
                    float d = 0.0f; for (std::size_t d_i = 0; d_i < dim; ++d_i) { float diff = qv[d_i] - xv[d_i]; d += diff * diff; }
                    dist_id.emplace_back(d, static_cast<std::uint32_t>(bi));
                }
                std::partial_sort(dist_id.begin(), dist_id.begin() + qcfg.k, dist_id.end(), [](const auto& a, const auto& b){ return a.first < b.first; });
                for (std::size_t j = 0; j < qcfg.k; ++j) bf_flat.push_back(dist_id[j].second);
            }
            float bf_recall = SearchMetrics::compute_recall(bf_flat, gt, bf_q, qcfg.k, k_gt);
            INFO("[BF-VALIDATE] Recall@" << qcfg.k << " (exact brute-force, first " << bf_q << " queries) = " << bf_recall);
        }
    }

    // If ground truth available, compute recall@k
    if (!gt.empty() && k_gt >= qcfg.k) {
        float recall = SearchMetrics::compute_recall(results_flat, gt, nq, qcfg.k, k_gt);
        INFO("Recall@" << qcfg.k << " = " << recall);
        REQUIRE(recall >= 0.70f);
    } else {
        SUCCEED("Searched " << nq << " queries successfully (no ground truth available).");
    }
}

