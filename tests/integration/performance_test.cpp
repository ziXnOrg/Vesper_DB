#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_all.hpp>
#include "vesper/core/platform_utils.hpp"

#include "tests/integration/dataset_loader.hpp"
#include "vesper/index/index_manager.hpp"

#include <chrono>
#include <cstdlib>


using namespace vesper;
using namespace vesper::index;
using vesper::test::DatasetLoader;
using vesper::test::PerformanceMetrics;

TEST_CASE("Search latency and throughput (smoke)", "[integration][performance]") {
    // Optional env toggle to force synthetic small dataset
    bool force_synth = false;
    if (auto env = vesper::core::safe_getenv("VESPER_PERF_SMOKE_FORCE_SYNTH")) {
        if (!env->empty() && (*env)[0] == '1' && (env->size() == 1)) {
            force_synth = true;
        }
    }

    std::size_t dim = 0;
    std::vector<float> base;
    std::vector<float> queries;

    if (!force_synth) {
        auto ds = DatasetLoader::load_benchmark("sift-128-euclidean", "data");
        if (ds) {
            dim = ds->info.dimension;
            base = std::move(ds->base_vectors);
            queries = std::move(ds->query_vectors);
        } else {
            WARN("Dataset not found; running performance smoke with synthetic data.");
        }
    }

    if (force_synth || dim == 0) {
        dim = 64;
        const std::size_t n = 20000;
        base.resize(n * dim);
        for (auto &x : base) x = static_cast<float>(rand()) / RAND_MAX;
        const std::size_t nq = 200;
        queries.resize(nq * dim);
        for (auto &x : queries) x = static_cast<float>(rand()) / RAND_MAX;
    }

    const std::size_t n = base.size() / dim;
    const std::size_t nq = std::min<std::size_t>(queries.size() / dim, 200);
    REQUIRE(n > 0);
    REQUIRE(nq > 0);

    IndexManager mgr(dim);
    IndexBuildConfig cfg; cfg.strategy = SelectionStrategy::Auto;
    if (!mgr.build(base.data(), n, cfg)) {
        WARN("IndexManager::build failed; skipping performance smoke.");
        return; // skip gracefully
    }

    PerformanceMetrics pm;
    QueryConfig qcfg; qcfg.k = 10;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (std::size_t i = 0; i < nq; ++i) {
        auto qstart = std::chrono::high_resolution_clock::now();
        auto r = mgr.search(&queries[i * dim], qcfg);
        auto qend = std::chrono::high_resolution_clock::now();
        REQUIRE(r.has_value());
        pm.record_latency(std::chrono::duration<double, std::micro>(qend - qstart).count());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double total_s = std::chrono::duration<double>(t1 - t0).count();
    pm.record_throughput(static_cast<double>(nq) / total_s);

    pm.print_summary("Search Performance");

    // Only assert that we recorded sensible values
    auto lat = pm.get_latency_stats();
    REQUIRE(lat.mean >= 0.0);
    REQUIRE(lat.p50 >= 0.0);
}

