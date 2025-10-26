/**
 * index_manager_planner_tests.cpp
 * Verify that IndexManager applies QueryPlanner's tuned plan.config when use_query_planner=true.
 * RED phase: These tests are expected to fail before H1 implementation because
 * IndexManager currently ignores plan.config and uses the caller's config.
 */

#include <catch2/catch_test_macros.hpp>

#include "vesper/index/index_manager.hpp"

#include <random>
#include <numeric>
#include <algorithm>
#include <thread>
#include <future>
#include <atomic>
#include <vector>

#include <cstdlib>

using namespace vesper;
using namespace vesper::index;

namespace {

std::vector<float> gen_vectors(std::size_t n, std::size_t dim, std::uint32_t seed = 42) {
    std::vector<float> v(n * dim);
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& x : v) x = dist(rng);
    return v;
}

void l2_normalize(std::vector<float>& v, std::size_t n, std::size_t dim) {
    for (std::size_t i = 0; i < n; ++i) {
        double norm2 = 0.0;
        for (std::size_t d = 0; d < dim; ++d) norm2 += static_cast<double>(v[i*dim + d]) * v[i*dim + d];
        float inv = norm2 > 0 ? static_cast<float>(1.0 / std::sqrt(norm2)) : 1.0f;
        for (std::size_t d = 0; d < dim; ++d) v[i*dim + d] *= inv;
    }
}

} // namespace

TEST_CASE("Planner config applied for HNSW", "[index_manager][planner][hnsw]") {
    const std::size_t dim = 32;
    const std::size_t n = 2000;
    auto data = gen_vectors(n, dim);
    l2_normalize(data, n, dim);

    IndexManager manager(dim);
    IndexBuildConfig build;
    build.strategy = SelectionStrategy::Manual;
    build.type = IndexType::HNSW;
    build.hnsw_params.M = 8;
    build.hnsw_params.efConstruction = 100;

    REQUIRE(manager.build(data.data(), n, build));

    auto query = gen_vectors(1, dim);
    l2_normalize(query, 1, dim);

    QueryConfig base;
    base.k = 25;                 // large k encourages planner to raise ef_search
    base.ef_search = 8;          // intentionally low compared to k
    base.use_query_planner = true;

    QueryPlanner planner(manager);
    auto plan = planner.plan(query.data(), base);
    REQUIRE(plan.index == IndexType::HNSW);

    // Execute via manager with planner enabled
    auto res = manager.search(query.data(), base);
    REQUIRE(res.has_value());

#ifdef VESPER_ENABLE_TESTS
    auto applied = manager.last_applied_query_config_debug();
    // Expectation: applied ef_search should come from plan.config (not caller base)
    REQUIRE(applied.ef_search == plan.config.ef_search);
#endif
}

TEST_CASE("Planner config applied for IVF-PQ", "[index_manager][planner][ivfpq]") {
    const std::size_t dim = 32;
    const std::size_t n = 15000; // moderate size to make IVF-PQ a plausible choice
    auto data = gen_vectors(n, dim);

    IndexManager manager(dim);
    IndexBuildConfig build;
    build.strategy = SelectionStrategy::Manual;
    build.type = IndexType::IVF_PQ;
    build.ivf_params.nlist = 256;
    build.ivf_params.m = 8;
    build.ivf_params.nbits = 8;

    REQUIRE(manager.build(data.data(), n, build));

    auto query = gen_vectors(1, dim);

    QueryConfig base;
    base.k = 20;
    base.nprobe = 1;            // intentionally low so planner should raise it
    base.use_query_planner = true;

    QueryPlanner planner(manager);
    auto plan = planner.plan(query.data(), base);
    REQUIRE(plan.index == IndexType::IVF_PQ);

    auto res = manager.search(query.data(), base);
    REQUIRE(res.has_value());

#ifdef VESPER_ENABLE_TESTS
    auto applied = manager.last_applied_query_config_debug();
    REQUIRE(applied.nprobe == plan.config.nprobe);
#endif
}

TEST_CASE("Planner config applied for DiskANN", "[index_manager][planner][diskann]") {
    const std::size_t dim = 32;
    const std::size_t n = 5000;
    auto data = gen_vectors(n, dim);

    IndexManager manager(dim);
    IndexBuildConfig build;
    build.strategy = SelectionStrategy::Manual;
    build.type = IndexType::DiskANN;
    build.vamana_params.degree = 32;
    build.vamana_params.L = 64;

    REQUIRE(manager.build(data.data(), n, build));

    auto query = gen_vectors(1, dim);

    QueryConfig base;
    base.k = 15;
    base.l_search = 8;          // intentionally low so planner should raise it
    base.use_query_planner = true;

    QueryPlanner planner(manager);
    auto plan = planner.plan(query.data(), base);
    REQUIRE(plan.index == IndexType::DiskANN);

    auto res = manager.search(query.data(), base);
    REQUIRE(res.has_value());

#ifdef VESPER_ENABLE_TESTS
    auto applied = manager.last_applied_query_config_debug();
    REQUIRE(applied.l_search == plan.config.l_search);
#endif
}

TEST_CASE("Planner rerank params are applied (IVF-PQ)", "[index_manager][planner][rerank]") {
    const std::size_t dim = 32;
    const std::size_t n = 12000;
    auto data = gen_vectors(n, dim);

    IndexManager manager(dim);
    IndexBuildConfig build;
    build.strategy = SelectionStrategy::Manual;
    build.type = IndexType::IVF_PQ;
    build.ivf_params.nlist = 256;
    build.ivf_params.m = 8;
    build.ivf_params.nbits = 8;

    REQUIRE(manager.build(data.data(), n, build));

    auto query = gen_vectors(1, dim);

    QueryConfig base;
    base.k = 10;
    base.nprobe = 4;
    base.use_query_planner = true;
    // Explicit rerank controls so plan.config reflects them deterministically
    base.use_exact_rerank = true;
    base.rerank_k = 30;
    base.rerank_alpha = 1.5f;
    base.rerank_cand_ceiling = 64;

    QueryPlanner planner(manager);
    auto plan = planner.plan(query.data(), base);
    REQUIRE(plan.index == IndexType::IVF_PQ);

    auto res = manager.search(query.data(), base);
    REQUIRE(res.has_value());

#ifdef VESPER_ENABLE_TESTS
    auto applied = manager.last_applied_query_config_debug();
    REQUIRE(applied.use_exact_rerank == plan.config.use_exact_rerank);
    REQUIRE(applied.rerank_k == plan.config.rerank_k);
    REQUIRE(applied.rerank_alpha == plan.config.rerank_alpha);
    REQUIRE(applied.rerank_cand_ceiling == plan.config.rerank_cand_ceiling);
#endif
}



// Concurrency tests (H2)
TEST_CASE("planner concurrent plan calls", "[planner][concurrency][plan]") {
    const std::size_t dim = 32;
    const std::size_t n = 8000;
    auto data = gen_vectors(n, dim);
    l2_normalize(data, n, dim);

    IndexManager manager(dim);
    IndexBuildConfig build;
    build.strategy = SelectionStrategy::Manual;
    build.type = IndexType::HNSW;
    build.hnsw_params.M = 8;
    build.hnsw_params.efConstruction = 80;
    REQUIRE(manager.build(data.data(), n, build));

    QueryPlanner planner(manager);

    std::atomic<bool> start{false};
    const int num_threads = 8;
    const int iters_per_thread = 200;

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            auto q = gen_vectors(1, dim, 1234u + t);
            QueryConfig base;
            base.k = 10;
            base.use_query_planner = true;
            for (int i = 0; i < iters_per_thread; ++i) {
                auto plan = planner.plan(q.data(), base);
                (void)plan;
            }
        });
    }

    start.store(true, std::memory_order_release);
    for (auto &th : threads) th.join();

    REQUIRE(true);
}

TEST_CASE("planner concurrent plan and update_stats", "[planner][concurrency][plan][update]") {
    const std::size_t dim = 32;
    const std::size_t n = 10000;
    auto data = gen_vectors(n, dim);

    IndexManager manager(dim);
    IndexBuildConfig build;
    build.strategy = SelectionStrategy::Manual;
    build.type = IndexType::IVF_PQ;
    build.ivf_params.nlist = 128;
    build.ivf_params.m = 8;
    build.ivf_params.nbits = 8;
    REQUIRE(manager.build(data.data(), n, build));

    QueryPlanner planner(manager);

    std::atomic<bool> start{false};
    const int readers = 6;
    const int writer_iters = 400;
    const int reader_iters = 400;

    std::vector<std::thread> threads;

    // reader threads: plan() only
    for (int t = 0; t < readers; ++t) {
        threads.emplace_back([&, t]() {
            while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
            auto q = gen_vectors(1, dim, 99u + t);
            QueryConfig base; base.k = 15; base.use_query_planner = true;
            for (int i = 0; i < reader_iters; ++i) {
                auto plan = planner.plan(q.data(), base);
                (void)plan;
            }
        });
    }

    // writer thread: plan() + update_stats() concurrently with readers
    threads.emplace_back([&]() {
        while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
        auto q = gen_vectors(1, dim, 777);
        QueryConfig base; base.k = 20; base.use_query_planner = true;
        for (int i = 0; i < writer_iters; ++i) {
            auto plan = planner.plan(q.data(), base);
            float simulated_ms = 0.2f + static_cast<float>(i % 7) * 0.05f;
            planner.update_stats(plan, simulated_ms, std::optional<float>{});
        }
    });

    start.store(true, std::memory_order_release);
    for (auto &th : threads) th.join();

    REQUIRE(true);
}

TEST_CASE("planner property-based randomized concurrent ops (TSan)", "[planner][concurrency][random]") {
    const std::size_t dim = 32;
    const std::size_t n = 9000;
    auto data = gen_vectors(n, dim);

    IndexManager manager(dim);
    IndexBuildConfig build;
    build.strategy = SelectionStrategy::Manual;
    build.type = IndexType::HNSW;
    build.hnsw_params.M = 8;
    build.hnsw_params.efConstruction = 80;
    REQUIRE(manager.build(data.data(), n, build));

    QueryPlanner planner(manager);

    std::atomic<bool> start{false};
    const int num_threads = 8;
    const int iters = 500;

    std::vector<std::thread> ts;
    for (int t = 0; t < num_threads; ++t) {
        ts.emplace_back([&, t]() {
            while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
            std::mt19937 rng(2024u + t);
            auto q = gen_vectors(1, dim, 2024u + t);
            QueryConfig base; base.k = 10; base.use_query_planner = true;
            for (int i = 0; i < iters; ++i) {
                int op = static_cast<int>(rng() % 3);
                if (op == 0) {
                    auto plan = planner.plan(q.data(), base);
                    (void)plan;
                } else if (op == 1) {
                    auto plan = planner.plan(q.data(), base);
                    float ms = 0.1f + static_cast<float>(rng() % 10) * 0.01f;
                    std::optional<float> rec;
                    if ((rng() % 5) == 0) rec = 0.8f + static_cast<float>((rng() % 20)) / 100.0f;
                    planner.update_stats(plan, ms, rec);
                } else {
                    auto stats = planner.get_stats();
                    (void)stats;
                }
            }
        });
    }

    start.store(true, std::memory_order_release);
    for (auto &th : ts) th.join();
    REQUIRE(true);
}




// Determinism tests (H3 - RED)
namespace {
inline void set_env_var(const char* key, const char* val) {
#if defined(_WIN32)
    _putenv_s(key, val ? val : "");
#else
    if (val) { setenv(key, val, 1); } else { unsetenv(key); }
#endif
}
}

TEST_CASE("planner deterministic plan with fixed seed", "[planner][determinism][plan]") {
    const std::size_t dim = 16;
    const std::size_t n = 50000; // reduced for faster Debug runs; still triggers multiple indexes under Hybrid
    auto data = gen_vectors(n, dim, 7);

    IndexManager manager(dim);
    IndexBuildConfig build;
    build.strategy = SelectionStrategy::Hybrid;
    build.memory_budget_mb = 1024;
    REQUIRE(manager.build(data.data(), n, build));

    QueryPlanner p1(manager);
    QueryPlanner p2(manager);

    auto q = gen_vectors(1, dim, 123);
    QueryConfig base; base.k = 10; base.use_query_planner = true;

    auto first1 = p1.plan(q.data(), base);
    auto first2 = p2.plan(q.data(), base);

    for (int i = 0; i < 10; ++i) {
        auto a = p1.plan(q.data(), base);
        auto b = p2.plan(q.data(), base);
        REQUIRE(a.index == first1.index);
        REQUIRE(b.index == first2.index);
        REQUIRE(a.index == b.index);
        REQUIRE(a.explanation == b.explanation);
        REQUIRE(a.config.k == b.config.k);
        REQUIRE(a.config.ef_search == b.config.ef_search);
        REQUIRE(a.config.nprobe == b.config.nprobe);
        REQUIRE(a.config.l_search == b.config.l_search);
        REQUIRE(a.config.use_exact_rerank == b.config.use_exact_rerank);
        REQUIRE(a.config.rerank_k == b.config.rerank_k);
        REQUIRE(a.config.rerank_alpha == b.config.rerank_alpha);
        REQUIRE(a.config.rerank_cand_ceiling == b.config.rerank_cand_ceiling);
        REQUIRE(a.estimated_cost_ms == b.estimated_cost_ms);
        REQUIRE(a.estimated_recall == b.estimated_recall);
    }
}

TEST_CASE("planner frozen mode no adaptive updates", "[planner][determinism][frozen]") {
    set_env_var("VESPER_PLANNER_FROZEN", "1");

    const std::size_t dim = 16;
    const std::size_t n = 5000;
    auto data = gen_vectors(n, dim, 5);
    l2_normalize(data, n, dim);

    IndexManager manager(dim);
    IndexBuildConfig build;
    build.strategy = SelectionStrategy::Manual;
    build.type = IndexType::HNSW;
    build.hnsw_params.M = 8;
    build.hnsw_params.efConstruction = 80;
    REQUIRE(manager.build(data.data(), n, build));

    QueryPlanner planner(manager);
    auto q = gen_vectors(1, dim, 42);
    QueryConfig base; base.k = 10; base.use_query_planner = true;

    auto plan0 = planner.plan(q.data(), base);
    for (int i = 0; i < 200; ++i) {
        float ms = 0.1f + static_cast<float>(i % 10) * 0.01f;
        std::optional<float> rec;
        if ((i % 3) == 0) rec = 0.9f;
        planner.update_stats(plan0, ms, rec);
        auto plan_i = planner.plan(q.data(), base);
        REQUIRE(plan_i.index == plan0.index);
        REQUIRE(plan_i.explanation == plan0.explanation);
        REQUIRE(plan_i.config.k == plan0.config.k);
    }
    auto stats = planner.get_stats();
    // In frozen mode, adaptive aggregates should not change
    REQUIRE(stats.avg_estimation_error_ms == 0.0f);
    REQUIRE(stats.avg_recall_error == 0.0f);

    set_env_var("VESPER_PLANNER_FROZEN", "0");
}

TEST_CASE("planner deterministic index selection order", "[planner][determinism][order]") {
    const std::size_t dim = 16;
    const std::size_t n = 50000; // reduced for faster Debug runs; expect HNSW+IVF_PQ active (DiskANN may also appear)
    auto data = gen_vectors(n, dim, 11);

    IndexManager manager(dim);
    IndexBuildConfig build;
    build.strategy = SelectionStrategy::Hybrid;
    build.memory_budget_mb = 1024;
    REQUIRE(manager.build(data.data(), n, build));

    // Expected tie-breaker: HNSW > IVF_PQ > DiskANN (stable order)
    auto active = manager.get_active_indexes();
    bool has_hnsw = std::find(active.begin(), active.end(), IndexType::HNSW) != active.end();
    bool has_ivf  = std::find(active.begin(), active.end(), IndexType::IVF_PQ) != active.end();
    bool has_disk = std::find(active.begin(), active.end(), IndexType::DiskANN) != active.end();
    (void)has_disk;

    IndexType expected = has_hnsw ? IndexType::HNSW : (has_ivf ? IndexType::IVF_PQ : IndexType::DiskANN);

    QueryPlanner planner(manager);
    auto q = gen_vectors(1, dim, 77);
    QueryConfig base; base.k = 10; base.use_query_planner = true;

    auto first = planner.plan(q.data(), base);
    REQUIRE(first.index == expected);

    for (int i = 0; i < 50; ++i) {
        auto p = planner.plan(q.data(), base);
        REQUIRE(p.index == expected);
    }
}
