#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_all.hpp>

#include "tests/integration/dataset_loader.hpp"
#include "vesper/index/index_manager.hpp"

#include <vector>
#include <algorithm>
#include <cstdlib>
#include "vesper/core/platform_utils.hpp"
#include <iostream>
#include <string>


using namespace vesper;
using namespace vesper::index;
using vesper::test::DatasetLoader;
using vesper::test::SearchMetrics;

TEST_CASE("IVF-PQ nprobe sweep recall measurements", "[integration][sweep]") {
    auto ds = DatasetLoader::load_benchmark("sift-128-euclidean", "data");
    if (!ds) {
        WARN("Dataset not found; skipping ivfpq_sweep_test. Run scripts/download_datasets.py");
        SUCCEED();
        return;
    }

    const std::size_t dim = ds->info.dimension;
    const std::size_t n = ds->info.num_vectors;
    REQUIRE(dim > 0);
    REQUIRE(n > 0);

    // Optionally cap the base size via env var to speed local runs
    std::size_t max_n = n;
    if (auto env = vesper::core::safe_getenv("VESPER_SWEEP_MAX_N")) {
        try { max_n = std::min<std::size_t>(n, static_cast<std::size_t>(std::stoull(*env))); } catch (...) {}
    }
    // Ensure Hybrid builds include HNSW (threshold is n < 1,000,000)
    max_n = std::min<std::size_t>(max_n, static_cast<std::size_t>(999999));

    IndexManager mgr(dim);
    IndexBuildConfig cfg;
    cfg.strategy = SelectionStrategy::Hybrid; // build HNSW+IVF-PQ to enable exact rerank
    cfg.type = IndexType::IVF_PQ;
    cfg.memory_budget_mb = 256; // avoid building DiskANN in Hybrid to keep runtime reasonable
    cfg.ivf_params.nlist = (max_n >= 1000000 ? 4096 : 2048);
    cfg.ivf_params.m = 16;
    cfg.ivf_params.nbits = 8;
    cfg.ivf_params.use_opq = true;

    // Defaults for OPQ training
    cfg.ivf_params.opq_iter = 5;
    cfg.ivf_params.opq_sample_n = std::min<std::size_t>(max_n, static_cast<std::size_t>(100000));
    cfg.ivf_params.opq_init = OpqInit::PCA;

    // Optional env overrides for OPQ grid tuning
    if (auto v = vesper::core::safe_getenv("VESPER_OPQ")) {
        std::string s(*v); for (auto& c : s) c = static_cast<char>(::tolower(c));
        if (s == "1" || s == "true") cfg.ivf_params.use_opq = true;
        if (s == "0" || s == "false") cfg.ivf_params.use_opq = false;
    }
    if (auto v = vesper::core::safe_getenv("VESPER_OPQ_ITER")) {
        try { cfg.ivf_params.opq_iter = static_cast<std::uint32_t>(std::stoul(*v)); } catch (...) {}
        if (cfg.ivf_params.opq_iter == 0) cfg.ivf_params.opq_iter = 1; // clamp to >=1
    }
    if (auto v = vesper::core::safe_getenv("VESPER_OPQ_SAMPLE")) {
        try {
            std::size_t s = static_cast<std::size_t>(std::stoull(*v));
            cfg.ivf_params.opq_sample_n = std::min<std::size_t>(max_n, s);
        } catch (...) {}
    }
    if (auto v = vesper::core::safe_getenv("VESPER_OPQ_INIT")) {
        std::string s(*v); for (auto& c : s) c = static_cast<char>(::tolower(c));
        if (s == "pca") cfg.ivf_params.opq_init = OpqInit::PCA;
        if (s == "identity" || s == "id") cfg.ivf_params.opq_init = OpqInit::Identity;
    }

    std::cout << "[opq] use=" << (cfg.ivf_params.use_opq?1:0)
              << " iter=" << cfg.ivf_params.opq_iter
              << " sample=" << cfg.ivf_params.opq_sample_n
              << " init=" << (cfg.ivf_params.opq_init==OpqInit::PCA?"PCA":"Identity")
              << std::endl;

    auto br = mgr.build(ds->base_vectors.data(), max_n, cfg);
    REQUIRE(br.has_value());

    const std::size_t nq = ds->info.num_queries; // use all available queries for realistic benchmark
    REQUIRE(nq > 0);

    // Optionally override number of queries via env var to speed local sweeps
    std::size_t nq_use = nq;
    if (auto env_nq = vesper::core::safe_getenv("VESPER_SWEEP_NQ")) {
        try { nq_use = std::min<std::size_t>(nq, static_cast<std::size_t>(std::stoull(*env_nq))); } catch (...) {}
        if (nq_use == 0) nq_use = nq; // ignore zero
    }

    const std::uint32_t k = 10;

    for (std::uint32_t nprobe : {32u, 128u, 512u, 1024u}) {
        QueryConfig qcfg; qcfg.k = k; qcfg.ef_search = 200; qcfg.nprobe = nprobe; qcfg.preferred_index = IndexType::IVF_PQ;
        qcfg.use_exact_rerank = true;
        qcfg.rerank_k = 200; // rerank shortlist; IVF-PQ now returns cand_k >= rerank_k


        // Additional env overrides for rerank tuning
        if (auto v = vesper::core::safe_getenv("VESPER_USE_EXACT_RERANK")) {
            std::string s(*v); for (auto& c : s) c = static_cast<char>(::tolower(c));
            if (s == "1" || s == "true") qcfg.use_exact_rerank = true;
            if (s == "0" || s == "false") qcfg.use_exact_rerank = false;
        }
        if (auto v = vesper::core::safe_getenv("VESPER_RERANK_K")) {
            try { qcfg.rerank_k = static_cast<std::uint32_t>(std::stoul(*v)); } catch (...) {}
        }
        if (auto v = vesper::core::safe_getenv("VESPER_RERANK_ALPHA")) {
            try { qcfg.rerank_alpha = std::stof(*v); } catch (...) {}
            if (!(qcfg.rerank_alpha > 0.0f) || !std::isfinite(qcfg.rerank_alpha)) qcfg.rerank_alpha = 2.0f;
        }
        if (auto v = vesper::core::safe_getenv("VESPER_RERANK_CEIL")) {
            try { qcfg.rerank_cand_ceiling = static_cast<std::uint32_t>(std::stoul(*v)); } catch (...) {}
        }

        if (auto env = vesper::core::safe_getenv("VESPER_RERANK_AUTO")) {
            std::string s(*env);
            if (s == "1" || s == "true") {
                qcfg.rerank_k = 0; // enable adaptive shortlist heuristic
            }
        }

        std::vector<std::uint32_t> results_flat; results_flat.reserve(nq_use * k);
        for (std::size_t i = 0; i < nq_use; ++i) {
            auto r = mgr.search(&ds->query_vectors[i * dim], qcfg);
            REQUIRE(r.has_value());
            REQUIRE(r->size() == k);
            for (std::size_t j = 0; j < k; ++j)
                results_flat.push_back(static_cast<std::uint32_t>((*r)[j].first));
        }

        float recall = SearchMetrics::compute_recall(results_flat, ds->groundtruth, nq_use, k, ds->k);
        std::cout << "nprobe=" << nprobe << ", recall@" << k << " = " << recall << std::endl;
    }

    SUCCEED();
}

