#include <vesper/index/ivf_pq.hpp>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
#include <stdexcept>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <eh.h>
#endif

#ifdef _WIN32
static void seh_translate(unsigned int code, EXCEPTION_POINTERS*) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "0x%08X", code);
    throw std::runtime_error(std::string("SEH ") + buf);
}
#endif

using vesper::index::IvfPqIndex;

// Forward decls for terminate handler
struct Scenario;
static void print_json_error(const Scenario& sc, std::string_view msg);

// Global pointer used by terminate handler to emit JSON with Scenario fields
static Scenario* g_current_sc = nullptr;
static void bench_terminate() noexcept {
    if (g_current_sc) {
        print_json_error(*g_current_sc, "terminate");
        std::fflush(stdout);
    } else {
        std::fprintf(stderr, "terminate without scenario\n");
        std::fflush(stderr);
    }
    std::abort();
}
#ifdef _WIN32
static LONG WINAPI bench_unhandled(EXCEPTION_POINTERS* ep) noexcept {
    if (g_current_sc) {
        char buf[32];
        unsigned int code = ep && ep->ExceptionRecord ? ep->ExceptionRecord->ExceptionCode : 0u;
        std::snprintf(buf, sizeof(buf), "seh_unhandled:0x%08X", code);
        print_json_error(*g_current_sc, buf);
        std::fflush(stdout);
    }
    return EXCEPTION_EXECUTE_HANDLER;
}
static void bench_invalid_param(const wchar_t*, const wchar_t*, const wchar_t*, unsigned int, uintptr_t) noexcept {
    if (g_current_sc) {
        print_json_error(*g_current_sc, "crt_invalid_parameter");
        std::fflush(stdout);
    }
}
#endif

using vesper::index::IvfPqTrainParams;

struct Scenario {
    std::size_t dim{128};
    std::size_t nvec{200000};
    std::uint32_t nlist{2048};
    std::uint32_t m{16};
    std::uint32_t nbits{8};
    std::size_t chunk{20000};
    std::uint64_t seed{123456789};
    // ANN toggles
    std::uint32_t ef_search{96};
    std::uint32_t ef_construction{200};
    std::uint32_t hnsw_M{16};
    bool validate{true};
    float validate_rate{0.01f};
    std::uint32_t refine_k{96}; // top-L refine for ANN refine
    std::uint32_t projection_dim{16}; // projection dimensionality for ann_assigner=proj
    std::string ann_assigner{"hnsw"}; // hnsw|kd|proj|brute (default hnsw for ANN)
    // KD-tree tuning flags (CLI)
    std::uint32_t kd_leaf_size{256};
    std::string kd_split{"variance"}; // variance|bbox
    int kd_batch{1}; // 1=batch (default), 0=per-query parallel
};

struct Result {
    double train_ms{0};
    double add_ms{0};
    IvfPqIndex::Stats stats{}; // includes ANN telemetry
};

static std::optional<std::string> eat_arg(std::string_view a, std::string_view key) {
    if (a.rfind(key, 0) == 0) return std::string(a.substr(key.size()));
    return std::nullopt;
}

static std::vector<float> make_vectors(std::size_t n, std::size_t dim, std::mt19937_64& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v; v.resize(n * dim);
    for (std::size_t i = 0; i < n * dim; ++i) v[i] = dist(rng);
    return v;
}

static void fill_ids(std::vector<std::uint64_t>& ids, std::uint64_t start) {
    for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = start + static_cast<std::uint64_t>(i);
}

static Result run_once(const Scenario& sc, bool use_ann) {
    Result r{};
    std::mt19937_64 rng(sc.seed);

    // Train on a subset
    const std::size_t train_n = std::max<std::size_t>(sc.nlist, std::min<std::size_t>(sc.nvec, 50000));
    auto train_data = make_vectors(train_n, sc.dim, rng);

    IvfPqIndex idx;
    IvfPqTrainParams tp;
    tp.nlist = sc.nlist;
    tp.m = sc.m;
    tp.nbits = sc.nbits;
    tp.max_iter = 25;
    tp.use_opq = false;
    tp.opq_init = vesper::index::OpqInit::Identity;
    // Enable timings in Stats
    tp.timings_enabled = true;
    // Coarse assigner + ANN toggles
    if (!use_ann) {
        tp.coarse_assigner = vesper::index::CoarseAssigner::Brute;
        tp.use_centroid_ann = false;
    } else {
        if (sc.ann_assigner == "hnsw") {
            tp.coarse_assigner = vesper::index::CoarseAssigner::HNSW;
            tp.use_centroid_ann = true;
        } else if (sc.ann_assigner == "kd") {
            tp.coarse_assigner = vesper::index::CoarseAssigner::KDTree;
            tp.use_centroid_ann = true;
        } else if (sc.ann_assigner == "proj") {
            tp.coarse_assigner = vesper::index::CoarseAssigner::Projection;
            tp.use_centroid_ann = true;
        } else {
            tp.coarse_assigner = vesper::index::CoarseAssigner::Brute;
            tp.use_centroid_ann = false;
        }
    }
    tp.centroid_ann_ef_search = sc.ef_search;
    tp.centroid_ann_ef_construction = sc.ef_construction;
    tp.centroid_ann_M = sc.hnsw_M;
    tp.centroid_ann_refine_k = sc.refine_k;
    tp.projection_dim = sc.projection_dim;
    // KD-tree tuning from CLI
    tp.kd_leaf_size = sc.kd_leaf_size;
    tp.kd_batch_assign = (sc.kd_batch != 0);
    tp.kd_split = (sc.kd_split == "bbox" ? vesper::index::KdSplitHeuristic::BBoxExtent
                                         : vesper::index::KdSplitHeuristic::Variance);

    tp.validate_ann_assignment = use_ann && sc.validate && (sc.validate_rate > 0.0f);
    tp.validate_ann_sample_rate = sc.validate_rate;

    auto t0 = std::chrono::high_resolution_clock::now();
    auto tr = idx.train(train_data.data(), sc.dim, train_n, tp);
    auto t1 = std::chrono::high_resolution_clock::now();
    r.train_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (!tr.has_value()) {
        std::cerr << "Train failed: code=" << static_cast<int>(tr.error().code) << "\n";
        return r;
    }

    // Prepare add workload
    std::vector<std::uint64_t> ids(std::min(sc.chunk, sc.nvec));
    std::size_t added = 0;

    auto a0 = std::chrono::high_resolution_clock::now();
    while (added < sc.nvec) {
        const std::size_t take = std::min(sc.chunk, sc.nvec - added);
        auto vecs = make_vectors(take, sc.dim, rng);
        ids.resize(take);
        fill_ids(ids, static_cast<std::uint64_t>(added));
        auto ar = idx.add(ids.data(), vecs.data(), take);
        if (!ar.has_value()) {
            std::cerr << "Add failed at " << added << ", code=" << static_cast<int>(ar.error().code) << "\n";
            break;
        }
        added += take;
    }
    auto a1 = std::chrono::high_resolution_clock::now();
    r.add_ms = std::chrono::duration<double, std::milli>(a1 - a0).count();

    r.stats = idx.get_stats();
    return r;
}

static void print_json_line(const Scenario& sc, bool use_ann, const Result& r) {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "{\n";
    std::cout << "  \"mode\": \"" << (use_ann?"ann":"bruteforce") << "\", \"dim\": " << sc.dim
              << ", \"nvec\": " << sc.nvec << ", \"nlist\": " << sc.nlist
              << ", \"m\": " << sc.m << ", \"nbits\": " << sc.nbits << ", \"chunk\": " << sc.chunk
              << ", \"seed\": " << sc.seed << ", \"ef_search\": " << sc.ef_search
              << ", \"ef_construction\": " << sc.ef_construction
              << ", \"hnsw_M\": " << sc.hnsw_M
              << ", \"refine_k\": " << sc.refine_k
              << ", \"ann_assigner\": \"" << sc.ann_assigner << "\""
              << ", \"projection_dim\": " << sc.projection_dim
              << ", \"validate_rate\": " << sc.validate_rate << ",\n";
    std::cout << "  \"train_ms\": " << r.train_ms << ", \"add_ms\": " << r.add_ms << ", \"add_Mvec_per_s\": "
              << (r.add_ms>0? ( (sc.nvec/1e6) / (r.add_ms/1000.0) ) : 0.0) << ",\n";
    std::cout << "  \"stats\": { \"ann_enabled\": " << (r.stats.ann_enabled?1:0)
              << ", \"ann_assignments\": " << r.stats.ann_assignments
              << ", \"ann_validated\": " << r.stats.ann_validated
              << ", \"ann_mismatches\": " << r.stats.ann_mismatches;
    if (r.stats.timings_enabled) {
        std::cout << ", \"t_assign_ms\": " << r.stats.t_assign_ms
                  << ", \"t_encode_ms\": " << r.stats.t_encode_ms
                  << ", \"t_lists_ms\": " << r.stats.t_lists_ms;
    }
    std::cout << " }\n";
    std::cout << "}" << std::endl;
}

static void print_json_error(const Scenario& sc, std::string_view msg) {
    std::cout << std::fixed << std::setprecision(4)
              << "{ \"error\": 1, \"message\": \"" << msg << "\""
              << ", \"nlist\": " << sc.nlist
              << ", \"ef_search\": " << sc.ef_search
              << ", \"refine_k\": " << sc.refine_k
              << ", \"hnsw_M\": " << sc.hnsw_M
              << " }" << std::endl;
}




int main(int argc, char** argv) {
    Scenario sc;
    g_current_sc = &sc;

    bool json = true; // default JSONL
    bool text = false;
    bool acceptance = false;
    double speedup_threshold = 3.0; // ≥3x add() throughput
#ifdef _WIN32
    // Suppress Windows error dialogs; prefer JSON error output
    SetErrorMode(SetErrorMode(0) | SEM_NOGPFAULTERRORBOX | SEM_FAILCRITICALERRORS);
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    SetUnhandledExceptionFilter(bench_unhandled);
    _set_invalid_parameter_handler(bench_invalid_param);

#endif

    double max_mismatch_rate = 0.01; // ≤1% among validated samples

#ifdef _WIN32
    _set_se_translator(seh_translate);
    #ifdef _OPENMP
    #pragma omp parallel
    {
        _set_se_translator(seh_translate);
    }
    #endif
#endif

    // Ensure we emit JSON even if std::terminate is invoked
    std::set_terminate(bench_terminate);

    for (int i=1; i<argc; ++i) {
        std::string a(argv[i]);
        if (a=="--text") { text=true; json=false; }
        else if (a=="--both") { text=true; json=true; }
        else if (a=="--json") { json=true; text=false; }
        else if (a=="--acceptance") { acceptance=true; }
        else if (auto v=eat_arg(a, "--dim=")) sc.dim = static_cast<std::size_t>(std::stoull(*v));
        else if (auto v=eat_arg(a, "--nvec=")) sc.nvec = static_cast<std::size_t>(std::stoull(*v));
        else if (auto v=eat_arg(a, "--nlist=")) sc.nlist = static_cast<std::uint32_t>(std::stoul(*v));
        else if (auto v=eat_arg(a, "--m=")) sc.m = static_cast<std::uint32_t>(std::stoul(*v));
        else if (auto v=eat_arg(a, "--nbits=")) sc.nbits = static_cast<std::uint32_t>(std::stoul(*v));
        else if (auto v=eat_arg(a, "--chunk=")) sc.chunk = static_cast<std::size_t>(std::stoull(*v));
        else if (auto v=eat_arg(a, "--seed=")) sc.seed = static_cast<std::uint64_t>(std::stoull(*v));
        else if (auto v=eat_arg(a, "--ef_search=")) sc.ef_search = static_cast<std::uint32_t>(std::stoul(*v));
        else if (auto v=eat_arg(a, "--ef_construction=")) sc.ef_construction = static_cast<std::uint32_t>(std::stoul(*v));
        else if (auto v=eat_arg(a, "--hnsw_M=")) sc.hnsw_M = static_cast<std::uint32_t>(std::stoul(*v));
        else if (auto v=eat_arg(a, "--validate_rate=")) sc.validate_rate = std::stof(*v), sc.validate = (sc.validate_rate>0.0f);
        else if (auto v=eat_arg(a, "--refine_k=")) sc.refine_k = static_cast<std::uint32_t>(std::stoul(*v));
        else if (auto v=eat_arg(a, "--projection_dim=")) sc.projection_dim = static_cast<std::uint32_t>(std::stoul(*v));
        else if (auto v=eat_arg(a, "--ann_assigner=")) sc.ann_assigner = *v; // hnsw|kd|proj|brute
        else if (auto v=eat_arg(a, "--kd_leaf_size=")) sc.kd_leaf_size = static_cast<std::uint32_t>(std::stoul(*v));
        else if (auto v=eat_arg(a, "--kd_split=")) sc.kd_split = *v; // variance|bbox
        else if (auto v=eat_arg(a, "--kd_batch=")) sc.kd_batch = static_cast<int>(std::stoi(*v));
        else if (auto v=eat_arg(a, "--speedup_threshold=")) speedup_threshold = std::stod(*v);
        else if (auto v=eat_arg(a, "--max_mismatch_rate=")) max_mismatch_rate = std::stod(*v);
        else if (a=="--help"||a=="-h") {
            std::cout << "Usage: ivfpq_add_bench [--json|--text|--both] [--acceptance]\n"
                         "  --dim=128 --nvec=200000 --nlist=2048 --m=16 --nbits=8 --chunk=20000 --seed=123456789\n"
                         "  --ef_search=96 --ef_construction=200 --refine_k=96 --projection_dim=16 --ann_assigner=hnsw --validate_rate=0.01\n"
                         "  --kd_leaf_size=256 --kd_split=variance|bbox --kd_batch=0|1\n"
                         "  --speedup_threshold=3.0 --max_mismatch_rate=0.01\n";
            return 0;
        }
    }

    try {
        if ((sc.dim % sc.m) != 0) {
            if (json) print_json_error(sc, "invalid_params: dim % m != 0");
            else std::cerr << "Invalid params: dim % m != 0\n";
            return 1;
        }
        if (sc.nvec < sc.nlist) {
            if (json) print_json_error(sc, "invalid_params: nvec < nlist");
            else std::cerr << "Invalid params: nvec < nlist\n";
            return 1;
        }

        auto base = run_once(sc, /*use_ann=*/false);
        auto ann  = run_once(sc, /*use_ann=*/true);

        double base_throughput = (base.add_ms>0 ? ( (sc.nvec/1e6) / (base.add_ms/1000.0) ) : 0.0);
        double ann_throughput  = (ann.add_ms>0  ? ( (sc.nvec/1e6) / (ann.add_ms/1000.0) )  : 0.0);
        double speedup = (base_throughput>0 ? ann_throughput / base_throughput : 0.0);
        double mismatch_rate = (ann.stats.ann_validated>0 ? static_cast<double>(ann.stats.ann_mismatches) / static_cast<double>(ann.stats.ann_validated) : 0.0);

        if (json) {
            print_json_line(sc, false, base);
            print_json_line(sc, true, ann);
            // Summary line
            std::cout << std::fixed << std::setprecision(4)
                      << "{ \"summary\": 1, \"nlist\": " << sc.nlist
                      << ", \"base_add_ms\": " << base.add_ms
                      << ", \"ann_add_ms\": " << ann.add_ms
                      << ", \"speedup\": " << speedup
                      << ", \"ann_assignments\": " << ann.stats.ann_assignments
                      << ", \"ann_validated\": " << ann.stats.ann_validated
                      << ", \"ann_mismatches\": " << ann.stats.ann_mismatches
                      << ", \"mismatch_rate\": " << mismatch_rate
                      << " }" << std::endl;
        }

        if (text) {
            std::cout << std::fixed << std::setprecision(2);
            std::cout << "add() throughput: baseline=" << base_throughput << " Mvec/s, ann=" << ann_throughput
                      << " Mvec/s, speedup=" << (base_throughput>0? ann_throughput/base_throughput : 0.0) << "x\n";
            std::cout << "ANN stats: assignments=" << ann.stats.ann_assignments
                      << ", validated=" << ann.stats.ann_validated
                      << ", mismatches=" << ann.stats.ann_mismatches
                      << ", mismatch_rate=" << (mismatch_rate*100.0) << "%\n";
            // Single-line KD summary (V1)
            std::cout << "KD stats: pushed=" << ann.stats.kd_nodes_pushed
                      << ", popped=" << ann.stats.kd_nodes_popped
                      << ", leaves_scanned=" << ann.stats.kd_leaves_scanned;
            if (ann.stats.timings_enabled) {
                std::cout << ", trav_ms=" << ann.stats.kd_traversal_ms
                          << ", leaf_ms=" << ann.stats.kd_leaf_ms;
            }
            std::cout << "\n";
        }

        if (acceptance && sc.nlist >= 1024) {
            bool ok = true;
            if (speedup < speedup_threshold) {
                std::cerr << "[ACCEPT] FAIL: speedup=" << speedup << " < " << speedup_threshold << " (nlist=" << sc.nlist << ")\n";
                ok = false;
            }
            if (ann.stats.ann_validated > 0 && mismatch_rate > max_mismatch_rate) {
                std::cerr << "[ACCEPT] FAIL: mismatch_rate=" << mismatch_rate << " > " << max_mismatch_rate << "\n";
                ok = false;
            }
            return ok ? 0 : 2;
        }

        return 0;
    } catch (const std::exception& e) {
        if (json) print_json_error(sc, std::string("exception: ") + e.what());
        else std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    } catch (...) {
        if (json) print_json_error(sc, "unknown_exception");
        else std::cerr << "Unknown exception\n";
        return 1;
    }
}
