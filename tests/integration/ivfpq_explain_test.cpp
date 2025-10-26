#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_all.hpp>

#include "tests/integration/dataset_loader.hpp"
#include "vesper/index/ivf_pq.hpp"
#include "vesper/core/platform_utils.hpp"

#include <vector>
#include <iostream>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace vesper;
using namespace vesper::index;
using vesper::test::DatasetLoader;

TEST_CASE("IVF-PQ Explain: ADC rank of GT@1 on SIFT-128", "[ivfpq][explain][dataset]") {
    auto ds = DatasetLoader::load_benchmark("sift-128-euclidean", "data");
    REQUIRE(ds.has_value());

    const std::size_t dim = ds->info.dimension;
    auto base = std::move(ds->base_vectors);
    auto queries = std::move(ds->query_vectors);
    auto gt = std::move(ds->groundtruth); // uint32_t ids
    const std::uint32_t k_gt = ds->k;

    // Optional environment overrides to speed local runs
    std::size_t max_n = base.size() / dim;
    if (auto env = vesper::core::safe_getenv("VESPER_EXPLAIN_MAX_N"); env && !env->empty()) {
        try { max_n = std::min<std::size_t>(max_n, static_cast<std::size_t>(std::stoull(*env))); } catch (...) {}
    }
    if (max_n < base.size() / dim) {
        base.resize(max_n * dim);
    }

    const std::size_t n = base.size() / dim;
    REQUIRE(n > 0);
    REQUIRE(!gt.empty());
    REQUIRE(k_gt >= 1);

    // Prepare training subset (match integration: up to 200k) with optional override
    std::size_t train_n = std::min<std::size_t>(n, 200000);
    if (auto env = vesper::core::safe_getenv("VESPER_EXPLAIN_TRAIN_N"); env && !env->empty()) {
        try { train_n = std::min<std::size_t>(n, static_cast<std::size_t>(std::stoull(*env))); } catch (...) {}
    }

    IvfPqTrainParams tp;
    tp.nlist = 4096;
    tp.m = 16;
    tp.nbits = 8;
    tp.use_opq = false;              // Temporarily disable OPQ for SIGSEGV bisect
    tp.opq_init = OpqInit::PCA;
    tp.seed = 42;

    IvfPqIndex index;
    // Debug gate available early for markers
    const bool dbg = [](){ auto v = vesper::core::safe_getenv("VESPER_IVFPQ_DEBUG"); return v && !v->empty() && ((*v)[0] != '0'); }();


    // Train on prefix sample
    if (dbg) std::cerr << "[EXPLAIN][marker] before train" << std::endl;
    auto tr = index.train(base.data(), dim, train_n, tp);
    if (dbg) std::cerr << "[EXPLAIN][marker] after train call; has_value=" << (tr.has_value() ? 1 : 0) << std::endl;
    if (!tr.has_value()) {
        std::cerr << "[EXPLAIN][train][error] code=" << static_cast<int>(tr.error().code) << std::endl;
    }
    REQUIRE(tr.has_value());

    // Add all vectors with IDs 0..n-1
    if (dbg) std::cerr << "[EXPLAIN][marker] before add n=" << n << std::endl;
    std::vector<std::uint64_t> ids(n);
    for (std::size_t i = 0; i < n; ++i) ids[i] = static_cast<std::uint64_t>(i);
    auto ar = index.add(ids.data(), base.data(), n);
    if (dbg) std::cerr << "[EXPLAIN][marker] after add; has_value=" << (ar.has_value() ? 1 : 0) << std::endl;
    REQUIRE(ar.has_value());

    // Search config to mirror integration
    IvfPqSearchParams sp;
    sp.nprobe = 256;
    sp.k = 10;

    // Only run when debug is enabled (the hook is gated)
    REQUIRE(dbg);

    // Audit first few queries whose GT@1 lies within [0, n)
    std::size_t examined = 0;
    std::size_t max_q = std::min<std::size_t>(queries.size() / dim, static_cast<std::size_t>(200));
    if (auto env = vesper::core::safe_getenv("VESPER_EXPLAIN_SCAN_Q"); env && !env->empty()) {
        try { max_q = std::min<std::size_t>(queries.size() / dim, static_cast<std::size_t>(std::stoull(*env))); } catch (...) {}
    }
    if (dbg) std::cerr << "[EXPLAIN][marker] entering per-query explain loop (max_q=" << max_q << ")" << std::endl;
    for (std::size_t qi = 0; qi < max_q && examined < 3; ++qi) {
        const std::uint32_t gt_id = gt[qi * k_gt + 0];
        if (gt_id >= n) continue; // skip queries whose GT lies outside capped base
        const float* qv = &queries[qi * dim];
        ++examined;

        // Explain ADC rank for GT@1
        auto exp = index.debug_explain_adc_rank(qv, gt_id);
        if (exp) {
            auto [rank, adc_dist] = exp.value();
            std::cout << "[IVFPQ][EXPLAIN] query=" << qi
                      << " gt_id=" << gt_id
                      << " adc_rank_gt1=" << rank
                      << " adc_dist_gt1=" << adc_dist << std::endl;
        } else {
            std::cout << "[IVFPQ][EXPLAIN] query=" << qi << " gt_id=" << gt_id
                      << " error: " << static_cast<int>(exp.error().code) << "\n";
        }

        // Centroid rank for GT@1 (coarse coverage diagnostic)
        auto cent = index.debug_explain_centroid_rank(qv, gt_id);
        if (cent) {
            auto [crank, cid] = cent.value();
            std::cout << "[IVFPQ][EXPLAIN-CENTROID] query=" << qi
                      << " gt_id=" << gt_id
                      << " centroid_rank=" << crank
                      << " centroid_id=" << cid
                      << " nprobe=" << sp.nprobe << std::endl;
        }


        // Also compute the search top-1 and report its ID/rank for comparison
        auto sr = index.search(qv, sp);
        REQUIRE(sr.has_value());
        REQUIRE(sr->size() >= 1);
        const auto top1_id = static_cast<std::uint32_t>((*sr)[0].first);
        auto exp2 = index.debug_explain_adc_rank(qv, top1_id);
        if (exp2) {
            auto [rank2, adc_dist2] = exp2.value();
            std::cout << "[IVFPQ][EXPLAIN] query=" << qi
                      << " top1_id=" << top1_id
                      << " adc_rank_top1=" << rank2
                      << " adc_dist_top1=" << adc_dist2 << std::endl;
        }

        // Batch/parallel correctness validation (debug-gated)
        const bool validate_batch = [](){ auto v = vesper::core::safe_getenv("VESPER_VALIDATE_BATCH"); return v && !v->empty() && ((*v)[0] != '0'); }();
        if (validate_batch) {
            const std::size_t qcount = queries.size() / dim;
            int sample_max = 10;
            if (auto env = vesper::core::safe_getenv("VESPER_VALIDATE_BATCH_COUNT"); env && !env->empty()) {
                try { sample_max = std::max(1, static_cast<int>(std::stol(*env))); } catch (...) {}
            }
            std::vector<std::size_t> sample;
            sample.reserve(static_cast<std::size_t>(sample_max));
            for (std::size_t qj = 0; qj < qcount && sample.size() < static_cast<std::size_t>(sample_max); qj += 100) {
                const std::uint32_t gt_id2 = gt[qj * k_gt + 0];
                if (gt_id2 < n) sample.push_back(qj); // only queries whose GT@1 lies within [0, n)
            }
            REQUIRE(!sample.empty());

            auto run_once = [&](int threads) {
                std::vector<std::vector<std::pair<std::uint64_t, float>>> out;
                out.reserve(sample.size());
                (void)threads;
                for (std::size_t idx = 0; idx < sample.size(); ++idx) {
                    const std::size_t qs = sample[idx];
                    const float* qv = &queries[qs * dim];
                    auto sr = index.search(qv, sp);
                    REQUIRE(sr.has_value());
                    std::vector<std::pair<std::uint64_t, float>> v;
                    v.reserve(sr->size());
                    for (const auto& p : *sr) v.emplace_back(p.first, p.second);
                    out.emplace_back(std::move(v));
                }
                return out;
            };

            int desired_parallel = 8;
            if (auto env = vesper::core::safe_getenv("OMP_NUM_THREADS"); env && !env->empty()) {
                try { desired_parallel = std::max(1, static_cast<int>(std::stol(*env))); } catch (...) {}
            }

#ifdef _OPENMP
            const int prev_threads = omp_get_max_threads();
            omp_set_num_threads(1);
#endif
            auto serial = run_once(1);

#ifdef _OPENMP
            omp_set_num_threads(desired_parallel);
#endif
            auto parallel = run_once(desired_parallel);

#ifdef _OPENMP
            omp_set_num_threads(prev_threads);
#endif

            // Log results and compare
            auto print_run = [&](const char* tag, const std::vector<std::vector<std::pair<std::uint64_t, float>>>& run){
                for (std::size_t i = 0; i < sample.size(); ++i) {
                    std::cerr << "[BATCH][" << tag << "] qi=" << sample[i] << " ids=";
                    std::cerr << "[";
                    for (std::size_t j = 0; j < run[i].size(); ++j) {
                        if (j) std::cerr << ",";
                        std::cerr << run[i][j].first;
                    }
                    std::cerr << "] dists=[";
                    for (std::size_t j = 0; j < run[i].size(); ++j) {
                        if (j) std::cerr << ",";
                        std::cerr << run[i][j].second;
                    }
                    std::cerr << "]\n";
                }
            };
            print_run("serial", serial);
            print_run("parallel", parallel);

            std::size_t total_mismatches = 0;
            for (std::size_t si = 0; si < sample.size(); ++si) {
                const auto& a = serial[si];
                const auto& b = parallel[si];
                const std::size_t topN = std::min(a.size(), b.size());
                for (std::size_t j = 0; j < topN; ++j) {
                    const auto [aid, ad] = a[j];
                    const auto [bid, bd] = b[j];
                    const float eps = 1e-5f;
                    const float denom = std::max(1.0f, std::max(std::fabs(ad), std::fabs(bd)));
                    const bool id_ok = (aid == bid);
                    const bool dist_ok = (std::fabs(ad - bd) / denom) <= eps;
                    if (!(id_ok && dist_ok)) {
                        if (total_mismatches == 0) {
                            std::cerr << "[BATCH][diverge] showing mismatches (first 20)\n";
                        }
                        if (total_mismatches < 20) {
                            std::cerr << "[BATCH][diverge] qi=" << sample[si]
                                      << " pos=" << j
                                      << " serial=(" << aid << "," << ad << ")"
                                      << " parallel=(" << bid << "," << bd << ")\n";
                        }
                        ++total_mismatches;
                    }
                }
            }
            if (total_mismatches == 0) {
                std::cerr << "[BATCH][ok] serial (threads=1) matches parallel (threads=" << desired_parallel << ") for sampled queries\n";
            } else {
                std::cerr << "[BATCH][summary] total_mismatches=" << total_mismatches << " across " << sample.size() << " queries\n";
            }
        }
    }
}

