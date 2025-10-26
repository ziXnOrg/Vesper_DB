#include "tests/integration/dataset_loader.hpp"
#include "vesper/index/index_manager.hpp"
#include "vesper/index/ivf_pq.hpp"

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>
#include <sstream>
#include <algorithm>

using vesper::index::IndexBuildConfig;
using vesper::index::IndexManager;
using vesper::index::QueryConfig;
using vesper::index::SelectionStrategy;
using vesper::test::DatasetLoader;
using vesper::test::PerformanceMetrics;
using vesper::test::SearchMetrics;

namespace fs = std::filesystem;

namespace {
struct Args {
    std::string dataset{"sift-128-euclidean"};
    std::string data_dir{"data"};
    std::uint32_t nlist{4096};
    std::uint32_t m{16};
    std::uint32_t nbits{8};
    std::uint32_t k{10};
    std::uint32_t nprobe{256};
    std::uint32_t ef_search{200};
    std::size_t max_queries{200};
    std::string out_dir; // empty => temp
    bool use_ivfpq{false};
    // Mixed workload options
    bool mixed{false};
    std::size_t mixed_add_batch{50000};
    std::size_t mixed_q_per_round{50};
    // Parameter sweep options
    bool run_sweep{false};
    std::vector<std::uint32_t> sweep_nlists;
    std::vector<std::uint32_t> sweep_nprobes;
    // Quick-mode controls
    std::size_t limit_n{0};      // 0 = all base vectors
    std::size_t limit_nq{0};     // 0 = all queries (still capped by max_queries)
    bool skip_persist{false};    // skip save/load round-trip
};

static std::optional<std::string> eat(std::string_view a, std::string_view key) {
    if (a.rfind(key, 0) == 0) return std::string(a.substr(key.size()));
    return std::nullopt;
}

static void print_usage() {
    std::cout << "Vesper E2E runner\n"
              << "Usage: vesper_e2e_runner [--dataset=name] [--data_dir=path]\n"
              << "  [--nlist=4096] [--m=16] [--nbits=8] [--k=10] [--nprobe=256] [--ef_search=200]\n"
              << "  [--max_queries=200] [--out_dir=path] [--use_ivfpq]\n"
              << "  [--mixed] [--mixed_add_batch=N] [--mixed_q_per_round=Q]\n"
              << "  [--sweep_nlist=1024,2048,4096] [--sweep_nprobe=64,128,256]\n"
              << "  [--limit_n=N] [--limit_nq=Q] [--skip_persist]\n"
              << "Datasets present: sift-128-euclidean, glove-100-angular, fashion-mnist-784-euclidean, mnist-784-euclidean\n";
}
}
static std::vector<std::uint32_t> parse_csv_u32(const std::string& s){
    std::vector<std::uint32_t> out; std::stringstream ss(s); std::string tok;
    while (std::getline(ss, tok, ',')) { if (!tok.empty()) out.push_back(static_cast<std::uint32_t>(std::stoul(tok))); }
    return out;
}


int main(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if (a == "--help" || a == "-h") { print_usage(); return 0; }
        else if (auto v = eat(a, "--dataset=")) args.dataset = *v;
        else if (auto v = eat(a, "--data_dir=")) args.data_dir = *v;
        else if (auto v = eat(a, "--nlist=")) args.nlist = static_cast<std::uint32_t>(std::stoul(*v));
        else if (auto v = eat(a, "--m=")) args.m = static_cast<std::uint32_t>(std::stoul(*v));
        else if (auto v = eat(a, "--nbits=")) args.nbits = static_cast<std::uint32_t>(std::stoul(*v));
        else if (auto v = eat(a, "--k=")) args.k = static_cast<std::uint32_t>(std::stoul(*v));
        else if (auto v = eat(a, "--nprobe=")) args.nprobe = static_cast<std::uint32_t>(std::stoul(*v));
        else if (auto v = eat(a, "--ef_search=")) args.ef_search = static_cast<std::uint32_t>(std::stoul(*v));
        else if (auto v = eat(a, "--max_queries=")) args.max_queries = static_cast<std::size_t>(std::stoull(*v));
        else if (auto v = eat(a, "--out_dir=")) args.out_dir = *v;
        else if (a == "--use_ivfpq") args.use_ivfpq = true;
        else if (a == "--mixed") args.mixed = true;
        else if (auto v = eat(a, "--mixed_add_batch=")) args.mixed_add_batch = static_cast<std::size_t>(std::stoull(*v));

        else if (auto v = eat(a, "--mixed_q_per_round=")) args.mixed_q_per_round = static_cast<std::size_t>(std::stoull(*v));
        else if (auto v = eat(a, "--sweep_nlist=")) { args.sweep_nlists = parse_csv_u32(*v); args.run_sweep = true; }
        else if (auto v = eat(a, "--sweep_nprobe=")) { args.sweep_nprobes = parse_csv_u32(*v); args.run_sweep = true; }
        else if (auto v = eat(a, "--limit_n=")) args.limit_n = static_cast<std::size_t>(std::stoull(*v));
        else if (auto v = eat(a, "--limit_nq=")) args.limit_nq = static_cast<std::size_t>(std::stoull(*v));
        else if (a == "--skip_persist") args.skip_persist = true;
    }

    // Load dataset
    auto ds = DatasetLoader::load_benchmark(args.dataset, args.data_dir);
    if (!ds) {
        std::cerr << "Dataset not found or failed to load: " << args.dataset << " (under '" << args.data_dir << "')\n";
        return 2;
    }

    const std::size_t dim = ds->info.dimension;
    const std::size_t n_all = ds->info.num_vectors;
    const std::size_t nq_all = std::min<std::size_t>(ds->info.num_queries, args.max_queries);
    const std::size_t n = args.limit_n ? std::min<std::size_t>(n_all, args.limit_n) : n_all;
    const std::size_t nq = args.limit_nq ? std::min<std::size_t>(nq_all, args.limit_nq) : nq_all;
    if (dim == 0 || n == 0 || nq == 0) {
        std::cerr << "Invalid dataset sizes (dim=" << dim << ", n=" << n << ", nq=" << nq << ")\n";
        return 3;
    }

    std::cout << "=== E2E: dataset='" << ds->info.name << "' dim=" << dim << " n=" << n << " nq=" << nq
              << " metric=" << (int)ds->info.metric << " ===\n";

    // Optional direct IVF-PQ path (bypass IndexManager)
    if (args.use_ivfpq) {
        using vesper::index::IvfPqIndex; using vesper::index::IvfPqTrainParams; using vesper::index::IvfPqSearchParams;
        // Quick parameter sweep: handle before any pre-save/persist path
        if (args.run_sweep) {
            using vesper::index::IvfPqIndex; using vesper::index::IvfPqTrainParams; using vesper::index::IvfPqSearchParams;
            auto choose_m = [&](std::size_t dim_in, std::uint32_t pref){ if (pref && dim_in%pref==0) return pref; for (std::uint32_t cand: {16u,25u,20u,10u,8u,4u,2u}) if (dim_in%cand==0) return cand; return static_cast<std::uint32_t>(dim_in);} ;
            if (args.sweep_nlists.empty()) args.sweep_nlists.push_back(args.nlist);
            if (args.sweep_nprobes.empty()) args.sweep_nprobes.push_back(args.nprobe);
            std::cout << "dataset,nlist,nprobe,recall@"<<args.k<<",mean_us,p95,p99\n";
            for (auto nlist : args.sweep_nlists) {
                IvfPqIndex idx;
                IvfPqTrainParams tp2; tp2.nlist = nlist; tp2.m = choose_m(dim, args.m); tp2.nbits = args.nbits;
                auto tr2 = idx.train(ds->base_vectors.data(), dim, n, tp2);
                if (!tr2) { std::cerr << "IVFPQ train failed: code=" << (int)tr2.error().code << "\n"; return 21; }
                std::vector<std::uint64_t> ids2(n); for (std::size_t i=0;i<n;++i) ids2[i]=static_cast<std::uint64_t>(i);
                auto ar2 = idx.add(ids2.data(), ds->base_vectors.data(), n);
                if (!ar2) { std::cerr << "IVFPQ add failed: code=" << (int)ar2.error().code << "\n"; return 22; }
                for (auto nprobe : args.sweep_nprobes) {
                    IvfPqSearchParams sp2; sp2.k = args.k; sp2.nprobe = nprobe;
                    PerformanceMetrics pm2; std::vector<std::uint32_t> flat2; flat2.reserve(nq*args.k);
                    for (std::size_t i=0;i<nq;++i) {
                        auto t0 = std::chrono::high_resolution_clock::now();
                        auto r2 = idx.search(&ds->query_vectors[i*dim], sp2);
                        auto t1 = std::chrono::high_resolution_clock::now();
                        if (!r2) { std::cerr << "IVFPQ search failed at q="<<i<<", code="<<(int)r2.error().code<<"\n"; return 23; }
                        std::size_t take2 = std::min<std::size_t>(args.k, r2->size());
                        for (std::size_t j=0;j<take2;++j) flat2.push_back(static_cast<std::uint32_t>((*r2)[j].first));
                        for (std::size_t j=take2;j<args.k;++j) flat2.push_back(take2>0? static_cast<std::uint32_t>((*r2)[take2-1].first):0u);
                        pm2.record_latency(std::chrono::duration<double, std::micro>(t1-t0).count());
                    }
                    float rec2 = 0.0f; if (ds->info.has_groundtruth) rec2 = SearchMetrics::compute_recall(flat2, ds->groundtruth, nq, args.k, ds->k);
                    auto st2 = pm2.get_latency_stats();
                    std::cout << ds->info.name << "," << nlist << "," << nprobe << "," << std::fixed << std::setprecision(3) << rec2
                              << "," << st2.mean << "," << st2.p95 << "," << st2.p99 << "\n";
                }
            }
            return 0;
        }

        IvfPqIndex index;
        IvfPqTrainParams tp; tp.nlist = args.nlist; tp.m = args.m; tp.nbits = args.nbits; tp.timings_enabled = true;
        auto tr = index.train(ds->base_vectors.data(), dim, n, tp);
        if (!tr) { std::cerr << "IVFPQ train failed: code=" << (int)tr.error().code << "\n"; return 11; }
        // ids 0..n-1
        std::vector<std::uint64_t> ids(n); for (std::size_t i=0;i<n;++i) ids[i]=static_cast<std::uint64_t>(i);

        IvfPqSearchParams sp; sp.k = args.k; sp.nprobe = args.nprobe;

        // Mixed ingest mode: interleave add() in batches with queries to measure tail latency during ingest
        if (args.mixed) {
            PerformanceMetrics pm_mix;
            std::size_t added = 0;
            while (added < n) {
                std::size_t batch = std::min<std::size_t>(args.mixed_add_batch, n - added);
                auto ar_part = index.add(ids.data() + added, ds->base_vectors.data() + added * dim, batch);
                if (!ar_part) { std::cerr << "IVFPQ add(batch) failed at added="<<added<<" code="<<(int)ar_part.error().code<<"\n"; return 12; }
                std::size_t qround = std::min<std::size_t>(args.mixed_q_per_round, nq);
                for (std::size_t r=0; r<qround; ++r) {
                    std::size_t qi = r % nq;
                    auto t0 = std::chrono::high_resolution_clock::now();
                    auto rr = index.search(&ds->query_vectors[qi*dim], sp);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    if (!rr) { std::cerr << "IVFPQ search(during-ingest) failed at q="<<qi<<" code="<<(int)rr.error().code<<"\n"; return 13; }
                    pm_mix.record_latency(std::chrono::duration<double, std::micro>(t1-t0).count());
                }
                added += batch;
            }
            auto mix = pm_mix.get_latency_stats();
            std::cout << std::fixed << std::setprecision(3);
            std::cout << "During-ingest latency_us{mean="<<mix.mean<<", p95="<<mix.p95<<", p99="<<mix.p99<<"}\n";
        } else {
            // Non-mixed: add all at once
            auto ar = index.add(ids.data(), ds->base_vectors.data(), n);
            if (!ar) { std::cerr << "IVFPQ add failed: code=" << (int)ar.error().code << "\n"; return 12; }
        }

        // Pre-save evaluation
        std::vector<std::uint32_t> pre_flat; pre_flat.reserve(nq*args.k);
        PerformanceMetrics pm_pre;
        for (std::size_t i=0;i<nq;++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto r = index.search(&ds->query_vectors[i*dim], sp);
            auto t1 = std::chrono::high_resolution_clock::now();
            if (!r) { std::cerr << "IVFPQ search failed at q="<<i<<", code="<<(int)r.error().code<<"\n"; return 13; }
            std::size_t take = std::min<std::size_t>(args.k, r->size());
            for (std::size_t j=0;j<take;++j) pre_flat.push_back(static_cast<std::uint32_t>((*r)[j].first));
            for (std::size_t j=take;j<args.k;++j) pre_flat.push_back(take>0? static_cast<std::uint32_t>((*r)[take-1].first):0u);
            pm_pre.record_latency(std::chrono::duration<double, std::micro>(t1-t0).count());
        }
        float pre_recall = 0.0f; if (ds->info.has_groundtruth) pre_recall = SearchMetrics::compute_recall(pre_flat, ds->groundtruth, nq, args.k, ds->k);
        auto ls_pre = pm_pre.get_latency_stats();
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "IVFPQ Pre-save: recall@"<<args.k<<"="<<pre_recall<<", latency_us{mean="<<ls_pre.mean<<", p95="<<ls_pre.p95<<", p99="<<ls_pre.p99<<"}\n";

        fs::path out = args.out_dir.empty() ? (fs::temp_directory_path() / ("vesper_e2e_ivfpq_" + ds->info.name)) : fs::path(args.out_dir);
        std::error_code ec; fs::remove_all(out, ec); fs::create_directories(out, ec);
        if (!args.skip_persist) {
            auto sv = index.save(out.string());
            if (!sv) { std::cerr << "IVFPQ save failed: code="<<(int)sv.error().code<<"\n"; return 14; }
            auto loaded = IvfPqIndex::load(out.string());
            if (!loaded) { std::cerr << "IVFPQ load failed: code="<<(int)loaded.error().code<<"\n"; return 15; }

            std::vector<std::uint32_t> post_flat; post_flat.reserve(nq*args.k);
            PerformanceMetrics pm_post;
            for (std::size_t i=0;i<nq;++i) {
                auto t0 = std::chrono::high_resolution_clock::now();
                auto r = loaded->search(&ds->query_vectors[i*dim], sp);
                auto t1 = std::chrono::high_resolution_clock::now();
                if (!r) { std::cerr << "IVFPQ post-load search failed at q="<<i<<", code="<<(int)r.error().code<<"\n"; return 16; }
                std::size_t take = std::min<std::size_t>(args.k, r->size());
                for (std::size_t j=0;j<take;++j) post_flat.push_back(static_cast<std::uint32_t>((*r)[j].first));
                for (std::size_t j=take;j<args.k;++j) post_flat.push_back(take>0? static_cast<std::uint32_t>((*r)[take-1].first):0u);
                pm_post.record_latency(std::chrono::duration<double, std::micro>(t1-t0).count());
            }
            float post_recall = 0.0f; if (ds->info.has_groundtruth) post_recall = SearchMetrics::compute_recall(post_flat, ds->groundtruth, nq, args.k, ds->k);
            auto ls_post = pm_post.get_latency_stats();
            std::cout << "IVFPQ Post-load: recall@"<<args.k<<"="<<post_recall<<", latency_us{mean="<<ls_post.mean<<", p95="<<ls_post.p95<<", p99="<<ls_post.p99<<"}\n";

            double delta = std::abs(pre_recall - post_recall);
            std::cout << "Recall delta (abs): " << delta << "\n";
            if (delta > 0.001) { std::cerr << "Recall changed after reload beyond tolerance (0.001)." << std::endl; return 17; }
            std::cout << "Output directory: " << out.string() << "\n";
            std::cout << "E2E OK" << std::endl;
            return 0;
        } else {
            std::cout << "Skipping persist round-trip (--skip_persist). E2E OK (pre-save only).\n";
            return 0;
        }
    }

    // Build index
    IndexManager mgr(dim);
    IndexBuildConfig cfg; cfg.strategy = SelectionStrategy::Manual; cfg.type = vesper::index::IndexType::IVF_PQ;
    cfg.ivf_params.nlist = args.nlist; cfg.ivf_params.m = args.m; cfg.ivf_params.nbits = args.nbits;
    auto br = mgr.build(ds->base_vectors.data(), n, cfg);
    if (!br) { std::cerr << "Build failed: code=" << (int)br.error().code << "\n"; return 4; }

    // Query pass 1 (pre-save)
    QueryConfig qcfg; qcfg.k = args.k; qcfg.ef_search = args.ef_search; qcfg.nprobe = args.nprobe;
    qcfg.use_query_planner = false; // ensure stable selection
    qcfg.preferred_index = vesper::index::IndexType::IVF_PQ;
    std::vector<std::uint32_t> results_flat; results_flat.reserve(nq * args.k);
    PerformanceMetrics pm1;

    for (std::size_t i = 0; i < nq; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto r = mgr.search(&ds->query_vectors[i * dim], qcfg);
        auto t1 = std::chrono::high_resolution_clock::now();
        if (!r) { std::cerr << "Search failed at q=" << i << ", code=" << (int)r.error().code << "\n"; return 5; }
        std::size_t got = r->size();
        std::size_t take = std::min<std::size_t>(args.k, got);
        for (std::size_t j = 0; j < take; ++j) results_flat.push_back(static_cast<std::uint32_t>((*r)[j].first));
        // pad to k if needed (duplicates don't increase recall)
        for (std::size_t j = take; j < args.k; ++j) results_flat.push_back(take > 0 ? static_cast<std::uint32_t>((*r)[take-1].first) : 0u);
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        pm1.record_latency(us);
    }

    float recall1 = 0.0f;
    if (ds->info.has_groundtruth && !ds->groundtruth.empty()) {
        recall1 = SearchMetrics::compute_recall(results_flat, ds->groundtruth, nq, args.k, ds->k);
    }

    auto lat1 = pm1.get_latency_stats();
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Pre-save: recall@" << args.k << "=" << recall1
              << ", latency_us{mean=" << lat1.mean << ", p95=" << lat1.p95 << ", p99=" << lat1.p99 << "}\n";

    // Persist (v1.1 format) and reload into a fresh manager
    fs::path out = args.out_dir.empty() ? (fs::temp_directory_path() / ("vesper_e2e_" + ds->info.name)) : fs::path(args.out_dir);
    std::error_code ec; fs::remove_all(out, ec); fs::create_directories(out, ec);

    auto sr = mgr.save(out.string());
    if (!sr) { std::cerr << "Save failed: code=" << (int)sr.error().code << "\n"; return 7; }

    IndexManager mgr2(dim);
    auto lr = mgr2.load(out.string());
    if (!lr) { std::cerr << "Load failed: code=" << (int)lr.error().code << "\n"; return 8; }

    // Query pass 2 (post-load)
    std::vector<std::uint32_t> results_flat2; results_flat2.reserve(nq * args.k);
    PerformanceMetrics pm2;

    for (std::size_t i = 0; i < nq; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto r = mgr2.search(&ds->query_vectors[i * dim], qcfg);
        auto t1 = std::chrono::high_resolution_clock::now();
        if (!r) { std::cerr << "Search (post-load) failed at q=" << i << ", code=" << (int)r.error().code << "\n"; return 9; }
        std::size_t got = r->size();
        std::size_t take = std::min<std::size_t>(args.k, got);
        for (std::size_t j = 0; j < take; ++j) results_flat2.push_back(static_cast<std::uint32_t>((*r)[j].first));
        for (std::size_t j = take; j < args.k; ++j) results_flat2.push_back(take > 0 ? static_cast<std::uint32_t>((*r)[take-1].first) : 0u);
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        pm2.record_latency(us);
    }

    float recall2 = 0.0f;
    if (ds->info.has_groundtruth && !ds->groundtruth.empty()) {
        recall2 = SearchMetrics::compute_recall(results_flat2, ds->groundtruth, nq, args.k, ds->k);
    }

    auto lat2 = pm2.get_latency_stats();
    std::cout << "Post-load: recall@" << args.k << "=" << recall2
              << ", latency_us{mean=" << lat2.mean << ", p95=" << lat2.p95 << ", p99=" << lat2.p99 << "}\n";

    // Simple gates and delta check
    if (ds->info.has_groundtruth) {
        double delta = std::abs(recall1 - recall2);
        std::cout << "Recall delta (abs): " << delta << "\n";
        if (delta > 0.001) {
            std::cerr << "Recall changed after reload beyond tolerance (0.001)." << std::endl;
            return 10;
        }
    }

    std::cout << "Output directory: " << out.string() << "\n";
    std::cout << "E2E OK" << std::endl;
    return 0;
}

