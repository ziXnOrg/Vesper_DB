#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_all.hpp>

#include "tests/integration/dataset_loader.hpp"
#include "vesper/index/index_manager.hpp"

using namespace vesper;
using namespace vesper::index;
using vesper::test::DatasetLoader;
using vesper::test::SearchMetrics;
#include "vesper/core/platform_utils.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>


TEST_CASE("Search accuracy across k values", "[integration][accuracy]") {
    auto ds = DatasetLoader::load_benchmark("sift-128-euclidean", "data");
    if (!ds) {
        WARN("Dataset not found; skipping accuracy_test. Run scripts/download_datasets.py");
        SUCCEED();
        return;
    }

    const std::size_t dim = ds->info.dimension;
    const std::size_t n = ds->info.num_vectors;
    REQUIRE(dim > 0);
    REQUIRE(n > 0);

    IndexManager mgr(dim);
    IndexBuildConfig cfg;
    cfg.strategy = SelectionStrategy::Auto;
    // IVF-PQ training params tuned for 1M @ 128D
    cfg.ivf_params.nlist = 4096;
    cfg.ivf_params.m = 16;
    cfg.ivf_params.nbits = 8;

    auto br = mgr.build(ds->base_vectors.data(), n, cfg);
    REQUIRE(br.has_value());

    const std::size_t nq = std::min<std::size_t>(ds->info.num_queries, 100);
    REQUIRE(nq > 0);

    for (std::uint32_t k : {1u, 10u, 100u}) {
        if (ds->k < k) continue; // ground truth smaller than k
        QueryConfig qcfg; qcfg.k = k; qcfg.ef_search = 200; qcfg.nprobe = 256;

        std::vector<std::uint32_t> results_flat; results_flat.reserve(nq * k);
        for (std::size_t i = 0; i < nq; ++i) {
            auto r = mgr.search(&ds->query_vectors[i * dim], qcfg);
            REQUIRE(r.has_value());
            REQUIRE(r->size() == k);
            for (std::size_t j = 0; j < k; ++j) results_flat.push_back(static_cast<std::uint32_t>((*r)[j].first));
        }

        // Optional diagnostic: print centroid rank and ADC global rank for GT@1
        #ifdef VESPER_ENABLE_TESTS
        const bool diag = [](){ auto v = vesper::core::safe_getenv("VESPER_IVFPQ_DIAG_RANK"); return v && !v->empty() && ((*v)[0] == '1'); }();
        if (k == 1 && diag) {
            const IvfPqIndex* ivf = mgr.ivf_pq_index_debug();
            if (ivf) {
                const std::size_t sample = std::min<std::size_t>(nq, 10);
                for (std::size_t qi = 0; qi < sample; ++qi) {
                    const float* q = &ds->query_vectors[qi * dim];
                    const std::uint32_t gt = ds->groundtruth[qi * ds->k + 0];
                    auto cr = ivf->debug_explain_centroid_rank(q, gt);
                    auto ar = ivf->debug_explain_adc_rank(q, gt);
                    if (cr && ar) {
                        std::cerr << "[DIAG][k=1] q=" << qi
                                  << " gt=" << gt
                                  << " centroid_rank=" << cr->first
                                  << " centroid_id=" << cr->second
                                  << " adc_rank=" << ar->first
                                  << " adc_dist=" << ar->second << "\n";
                    }
                }
            }
        }

        // Optional diagnostic: compare ADC ranking vs true L2 ranking
        const bool rank_cmp = [](){ auto v = vesper::core::safe_getenv("VESPER_IVFPQ_RANK_CMP"); return v && !v->empty() && ((*v)[0] == '1'); }();
        if (k == 1 && rank_cmp) {
            // Find failing queries (where GT is not in top-1)
            std::vector<std::size_t> failing_queries;
            for (std::size_t qi = 0; qi < nq; ++qi) {
                const std::uint32_t gt = ds->groundtruth[qi * ds->k + 0];
                const std::uint32_t result = results_flat[qi];
                if (result != gt) {
                    failing_queries.push_back(qi);
                }
            }

            const std::size_t sample_limit = std::min<std::size_t>(failing_queries.size(), 10);
            std::cerr << "[RANK_CMP] Found " << failing_queries.size() << " failing queries (GT not in top-1); analyzing first " << sample_limit << "\n";

            for (std::size_t si = 0; si < sample_limit; ++si) {
                const std::size_t qi = failing_queries[si];
                const float* query = &ds->query_vectors[qi * dim];
                const std::uint32_t gt_id = ds->groundtruth[qi * ds->k + 0];

                // Get search results with larger k to capture more candidates
                QueryConfig diag_qcfg = qcfg;
                diag_qcfg.k = 1000;  // Get top-1000 to analyze
                auto search_result = mgr.search(query, diag_qcfg);
                if (!search_result.has_value()) continue;

                // Compute true L2 distances for all returned candidates
                std::vector<std::pair<std::uint64_t, float>> adc_ranking;  // (id, adc_dist)
                std::vector<std::pair<std::uint64_t, float>> true_ranking; // (id, true_l2_dist)

                for (const auto& [id, adc_dist] : *search_result) {
                    adc_ranking.emplace_back(id, adc_dist);

                    // Compute true L2 distance from query to original vector
                    const float* vec = &ds->base_vectors[id * dim];
                    double dist_sq = 0.0;
                    for (std::size_t d = 0; d < dim; ++d) {
                        const double diff = static_cast<double>(query[d]) - static_cast<double>(vec[d]);
                        dist_sq += diff * diff;
                    }
                    true_ranking.emplace_back(id, static_cast<float>(dist_sq));
                }

                // Sort by true L2 distance
                std::sort(true_ranking.begin(), true_ranking.end(),
                         [](const auto& a, const auto& b) { return a.second < b.second; });

                // Find GT rank in both rankings
                std::size_t gt_adc_rank = 0;
                std::size_t gt_true_rank = 0;
                float gt_adc_dist = 0.0f;
                float gt_true_dist = 0.0f;

                for (std::size_t i = 0; i < adc_ranking.size(); ++i) {
                    if (adc_ranking[i].first == gt_id) {
                        gt_adc_rank = i + 1;
                        gt_adc_dist = adc_ranking[i].second;
                    }
                    if (true_ranking[i].first == gt_id) {
                        gt_true_rank = i + 1;
                        gt_true_dist = true_ranking[i].second;
                    }
                }

                std::cerr << "[RANK_CMP][q=" << qi << "] gt=" << gt_id
                          << " adc_rank=" << gt_adc_rank << " true_rank=" << gt_true_rank
                          << " adc_dist=" << gt_adc_dist << " true_dist=" << gt_true_dist
                          << " in_results=" << (gt_adc_rank > 0 ? "YES" : "NO") << "\n";

                // Show top-5 by ADC vs top-5 by true L2
                std::cerr << "[RANK_CMP][top5_adc]";
                for (std::size_t i = 0; i < std::min<std::size_t>(5, adc_ranking.size()); ++i) {
                    std::cerr << " " << adc_ranking[i].first << ":" << adc_ranking[i].second;
                }
                std::cerr << "\n";

                std::cerr << "[RANK_CMP][top5_true]";
                for (std::size_t i = 0; i < std::min<std::size_t>(5, true_ranking.size()); ++i) {
                    std::cerr << " " << true_ranking[i].first << ":" << true_ranking[i].second;
                }
                std::cerr << "\n";

                // If GT is not in results, compute its true distance and compare to returned candidates
                if (gt_adc_rank == 0) {
                    const float* gt_vec = &ds->base_vectors[gt_id * dim];
                    double gt_dist_sq = 0.0;
                    for (std::size_t d = 0; d < dim; ++d) {
                        const double diff = static_cast<double>(query[d]) - static_cast<double>(gt_vec[d]);
                        gt_dist_sq += diff * diff;
                    }
                    std::cerr << "[RANK_CMP][gt_missing] gt_true_dist=" << gt_dist_sq
                              << " top1_adc_dist=" << adc_ranking[0].second
                              << " top1_true_dist=" << true_ranking[0].second
                              << " gt_should_rank=" << (gt_dist_sq < true_ranking[0].second ? "FIRST" : "LOWER") << "\n";
                }

                // For first failing query, show detailed comparison of GT vs top-1 ADC
                if (si == 0 && gt_adc_rank > 1) {
                    const std::uint64_t top1_adc_id = adc_ranking[0].first;
                    const std::uint64_t gt_true_id = true_ranking[0].first;

                    std::cerr << "[RANK_CMP][detail] query=" << qi << "\n";
                    std::cerr << "[RANK_CMP][detail] GT_id=" << gt_id
                              << " adc_rank=" << gt_adc_rank << " adc_dist=" << gt_adc_dist
                              << " true_rank=" << gt_true_rank << " true_dist=" << gt_true_dist << "\n";
                    std::cerr << "[RANK_CMP][detail] Top1_ADC_id=" << top1_adc_id
                              << " adc_dist=" << adc_ranking[0].second << "\n";

                    // Find top1_adc in true ranking
                    for (std::size_t i = 0; i < true_ranking.size(); ++i) {
                        if (true_ranking[i].first == top1_adc_id) {
                            std::cerr << "[RANK_CMP][detail] Top1_ADC true_rank=" << (i+1)
                                      << " true_dist=" << true_ranking[i].second << "\n";
                            break;
                        }
                    }

                    std::cerr << "[RANK_CMP][detail] Top1_True_id=" << gt_true_id
                              << " true_dist=" << true_ranking[0].second << "\n";

                    // Find top1_true in ADC ranking
                    for (std::size_t i = 0; i < adc_ranking.size(); ++i) {
                        if (adc_ranking[i].first == gt_true_id) {
                            std::cerr << "[RANK_CMP][detail] Top1_True adc_rank=" << (i+1)
                                      << " adc_dist=" << adc_ranking[i].second << "\n";
                            break;
                        }
                    }
                }

                // Compute correlation between ADC and true L2 distances
                if (adc_ranking.size() >= 2) {
                    // Simple Pearson correlation
                    double mean_adc = 0.0, mean_true = 0.0;
                    for (std::size_t i = 0; i < adc_ranking.size(); ++i) {
                        mean_adc += adc_ranking[i].second;
                        mean_true += true_ranking[i].second;
                    }
                    mean_adc /= adc_ranking.size();
                    mean_true /= true_ranking.size();

                    double cov = 0.0, var_adc = 0.0, var_true = 0.0;
                    for (std::size_t i = 0; i < adc_ranking.size(); ++i) {
                        const double adc_dev = adc_ranking[i].second - mean_adc;
                        const double true_dev = true_ranking[i].second - mean_true;
                        cov += adc_dev * true_dev;
                        var_adc += adc_dev * adc_dev;
                        var_true += true_dev * true_dev;
                    }

                    const double pearson = (var_adc > 0 && var_true > 0) ? (cov / std::sqrt(var_adc * var_true)) : 0.0;
                    std::cerr << "[RANK_CMP][correlation] pearson=" << pearson << "\n";
                }
            }
        }
        #endif


        float recall = SearchMetrics::compute_recall(results_flat, ds->groundtruth, nq, k, ds->k);
        INFO("k=" << k << ", recall=" << recall);
        REQUIRE(recall >= (k == 1 ? 0.6f : k == 10 ? 0.7f : 0.7f));
    }
}

