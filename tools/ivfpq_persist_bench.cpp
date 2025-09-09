#include <vesper/index/ivf_pq.hpp>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#ifdef _WIN32
  #define NOMINMAX
  #include <windows.h>
  #include <psapi.h>
#endif

namespace fs = std::filesystem;
using vesper::index::IvfPqIndex;
using vesper::index::IvfPqTrainParams;

struct MemStats { size_t cur_bytes{0}; size_t peak_bytes{0}; };

static MemStats sample_process_memory() {
    MemStats s{};
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        s.cur_bytes = static_cast<size_t>(pmc.WorkingSetSize);
        s.peak_bytes = static_cast<size_t>(pmc.PeakWorkingSetSize);
    }
#endif
    return s;
}

struct Sampler {
    std::atomic<bool> running{false};
    size_t peak_ws{0};
    std::thread thr;
    void start() {
        running = true;
        thr = std::thread([this]{
            while (running.load(std::memory_order_relaxed)) {
                auto m = sample_process_memory();
                if (m.cur_bytes > peak_ws) peak_ws = m.cur_bytes;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        });
    }
    size_t stop() {
        running = false;
        if (thr.joinable()) thr.join();
        return peak_ws;
    }
};

static void set_env(const char* k, const char* v) {
#ifdef _WIN32
    _putenv_s(k, v);
#else
    setenv(k, v, 1);
#endif
}

static std::vector<float> make_vectors(size_t n, size_t dim, std::mt19937_64& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v; v.resize(n * dim);
    for (size_t i = 0; i < n * dim; ++i) v[i] = dist(rng);
    return v;
}

static void fill_ids(std::vector<std::uint64_t>& ids, std::uint64_t start) {
    for (size_t i = 0; i < ids.size(); ++i) ids[i] = start + static_cast<std::uint64_t>(i);
}

struct SectionInfo { uint32_t type; uint64_t unc; uint64_t comp; };

static std::vector<SectionInfo> parse_v11_sections(const fs::path& file) {
    std::vector<SectionInfo> secs;
    std::ifstream f(file, std::ios::binary);
    if (!f) return secs;
    char magic[8]; f.read(magic, 8);
    if (!f || std::string_view(magic, 8) != std::string_view("IVFPQv11", 8)) return secs;
    auto rd16=[&](uint16_t& v){ f.read(reinterpret_cast<char*>(&v),2); };
    auto rd32=[&](uint32_t& v){ f.read(reinterpret_cast<char*>(&v),4); };
    auto rd64=[&](uint64_t& v){ f.read(reinterpret_cast<char*>(&v),8); };
    uint16_t maj=0,min=0; rd16(maj); rd16(min);
    uint32_t flags=0; rd32(flags);
    uint32_t dim=0,nlist=0,m=0,nbits=0,dsub=0; uint64_t nvec=0, build_ts=0; uint32_t code_size=0, meta_len=0;
    rd32(dim); rd32(nlist); rd32(m); rd32(nbits); rd32(dsub); rd64(nvec); rd32(code_size); rd64(build_ts); rd32(meta_len);
    // Now a stream of section headers and payloads until trailer
    while (f) {
        struct H { uint32_t type; uint64_t unc; uint64_t comp; uint64_t shash; } hdr{};
        f.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
        if (!f) break;
        secs.push_back(SectionInfo{hdr.type, hdr.unc, hdr.comp});
        if (hdr.comp > 0) f.seekg(static_cast<std::streamoff>(hdr.comp), std::ios::cur);
        else if (hdr.unc == 0) {
            // Likely end when an empty header is encountered erroneously; guard
            break;
        }
        // Heuristic: trailer starts with 'CHKS' 4 bytes; peek ahead
        std::streampos p = f.tellg();
        char sig[4]; f.read(sig, 4);
        if (f && std::string_view(sig,4)==std::string_view("CHKS",4)) { break; }
        f.clear(); f.seekg(p);
    }
    return secs;
}

struct Scenario {
    size_t dim{128};
    size_t nvec{100000};
    uint32_t nlist{1000};
    uint32_t m{16};
    uint32_t nbits{8};
    size_t chunk{10000};
    size_t meta_bytes{0};
    int zstd_level{0}; // 0=no compression; 1..3 compressed
    bool use_v11{true};
    bool load_mmap{false};
};

struct Result {
    double train_ms{0}, add_ms{0}, save_ms{0}, load_ms{0};
    size_t out_bytes{0};
    double save_throughput_MBps{0}, load_throughput_MBps{0};
    size_t peak_ws_save{0}, peak_ws_load{0};
    std::vector<SectionInfo> sections;
};

static Result run_benchmark(const Scenario& sc, const fs::path& outdir) {
    Result r{};
    fs::create_directories(outdir);
    std::mt19937_64 rng(12345);

    IvfPqIndex idx;

    // Train on a subset
    const size_t train_n = std::max<size_t>(sc.nlist, std::min<size_t>(sc.nvec, 50000));
    auto train_data = make_vectors(train_n, sc.dim, rng);
    IvfPqTrainParams tp; tp.nlist = sc.nlist; tp.m = sc.m; tp.nbits = sc.nbits; tp.use_opq = false; tp.opq_init = vesper::index::OpqInit::Identity;

    auto t0 = std::chrono::high_resolution_clock::now();
    auto tr = idx.train(train_data.data(), sc.dim, train_n, tp);
    auto t1 = std::chrono::high_resolution_clock::now();
    r.train_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (!tr.has_value()) {
        std::cerr << "Train failed: " << (int)tr.error().code << "\n";
    }

    // Optional metadata
    if (sc.meta_bytes) {
        std::string meta; meta.resize(sc.meta_bytes, 'x'); idx.set_metadata_json(meta);
    }

    // Add in chunks to bound memory
    std::vector<std::uint64_t> ids(std::min(sc.chunk, sc.nvec));
    size_t added = 0; auto add_start = std::chrono::high_resolution_clock::now();
    while (added < sc.nvec) {
        size_t take = std::min(sc.chunk, sc.nvec - added);
        auto vecs = make_vectors(take, sc.dim, rng);
        ids.resize(take); fill_ids(ids, static_cast<std::uint64_t>(added));
        auto ar = idx.add(ids.data(), vecs.data(), take);
        if (!ar.has_value()) { std::cerr << "Add failed at " << added << "\n"; break; }
        added += take;
    }
    auto add_end = std::chrono::high_resolution_clock::now();
    r.add_ms = std::chrono::duration<double, std::milli>(add_end - add_start).count();

    // Set env for save
    set_env("VESPER_IVFPQ_SAVE_V11", sc.use_v11 ? "1" : "0");
#ifdef VESPER_HAS_ZSTD
    if (sc.zstd_level > 0) set_env("VESPER_IVFPQ_ZSTD_LEVEL", (sc.zstd_level==1?"1":(sc.zstd_level==2?"2":"3")));
    else set_env("VESPER_IVFPQ_ZSTD_LEVEL", "0");
#endif

    // Save timing + memory sampling
    Sampler smp_save; smp_save.start();
    auto s0 = std::chrono::high_resolution_clock::now();
    auto sr = idx.save(outdir.string());
    auto s1 = std::chrono::high_resolution_clock::now();
    r.peak_ws_save = smp_save.stop();
    r.save_ms = std::chrono::duration<double, std::milli>(s1 - s0).count();
    if (!sr.has_value()) {
        std::cerr << "Save failed: " << (int)sr.error().code << "\n";
    }

    auto bin_path = outdir / "ivfpq.bin";
    if (fs::exists(bin_path)) {
        r.out_bytes = static_cast<size_t>(fs::file_size(bin_path));
        r.sections = parse_v11_sections(bin_path);
        r.save_throughput_MBps = r.out_bytes ? ( (r.out_bytes / (1024.0*1024.0)) / (r.save_ms / 1000.0) ) : 0.0;
    }

    // Release large in-memory structures before measuring load
    idx = IvfPqIndex();

    // Load timing (mmap/stream controlled by env)
    set_env("VESPER_IVFPQ_LOAD_MMAP", sc.load_mmap ? "1" : "0");
    Sampler smp_load; smp_load.start();
    auto l0 = std::chrono::high_resolution_clock::now();
    auto lr = IvfPqIndex::load(outdir.string());
    auto l1 = std::chrono::high_resolution_clock::now();
    r.peak_ws_load = smp_load.stop();
    r.load_ms = std::chrono::duration<double, std::milli>(l1 - l0).count();
    r.load_throughput_MBps = r.out_bytes ? ( (r.out_bytes / (1024.0*1024.0)) / (r.load_ms / 1000.0) ) : 0.0;
    if (!lr.has_value()) {
        std::cerr << "Load failed: " << (int)lr.error().code << "\n";
    }

    return r;
}

static void print_text(const Scenario& sc, const Result& r) {
    std::cout << "== Scenario ==\n"
              << "dim="<<sc.dim<<", nvec="<<sc.nvec<<", nlist="<<sc.nlist
              << ", m="<<sc.m<<", nbits="<<sc.nbits
              << ", meta="<<sc.meta_bytes<<"B, v11="<<(sc.use_v11?"1":"0")
              << ", zstd="<<sc.zstd_level<<", mmap="<<(sc.load_mmap?"1":"0")
              << "\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "train: "<< r.train_ms << " ms, add: "<< r.add_ms << " ms\n";
    std::cout << "save:  "<< r.save_ms  << " ms ("<< r.save_throughput_MBps << " MB/s), peakWS="<< r.peak_ws_save/ (1024*1024) << " MiB\n";
    std::cout << "load:  "<< r.load_ms  << " ms ("<< r.load_throughput_MBps << " MB/s), peakWS="<< r.peak_ws_load/ (1024*1024) << " MiB\n";
    std::cout << "file:  "<< r.out_bytes/(1024.0*1024.0) << " MiB\n";
    if (!r.sections.empty()) {
        std::cout << "sections (type: unc/comp MiB):\n";
        for (auto& s : r.sections) {
            std::cout << "  "<< s.type << ": "<< (s.unc/(1024.0*1024.0)) << "/"<< (s.comp/(1024.0*1024.0)) << "\n";
        }
    }
}

static void print_json_line(const Scenario& sc, const Result& r) {
    std::cout << "{\n";
    std::cout << "  \"dim\": "<<sc.dim<<", \"nvec\": "<<sc.nvec<<", \"nlist\": "<<sc.nlist
              << ", \"m\": "<<sc.m<<", \"nbits\": "<<sc.nbits
              << ", \"meta\": "<<sc.meta_bytes<<", \"v11\": "<<(sc.use_v11?1:0)
              << ", \"zstd\": "<<sc.zstd_level<<", \"mmap\": "<<(sc.load_mmap?1:0) << ",\n";
    std::cout << "  \"train_ms\": "<< r.train_ms << ", \"add_ms\": "<< r.add_ms << ", \"save_ms\": "<< r.save_ms << ", \"load_ms\": "<< r.load_ms << ",\n";
    std::cout << "  \"file_bytes\": "<< r.out_bytes << ", \"save_MBps\": "<< r.save_throughput_MBps << ", \"load_MBps\": "<< r.load_throughput_MBps << ",\n";
    std::cout << "  \"peak_ws_save\": "<< r.peak_ws_save << ", \"peak_ws_load\": "<< r.peak_ws_load << ",\n";
    std::cout << "  \"sections\": [";
    for (size_t i=0;i<r.sections.size();++i){ auto&s=r.sections[i];
        std::cout << "{\"type\": "<<s.type<<", \"unc\": "<<s.unc<<", \"comp\": "<<s.comp<<"}";
        if (i+1<r.sections.size()) std::cout << ", ";
    }
    std::cout << "]\n}" << std::endl;
}

static std::vector<size_t> parse_list(const std::string& s) {
    std::vector<size_t> v; size_t i=0; while (i < s.size()) {
        size_t j=i; while (j<s.size() && s[j]!=',') ++j; v.push_back(std::stoull(s.substr(i,j-i))); i = (j==s.size()?j:j+1);
    } return v;
}

int main(int argc, char** argv) {
    // Defaults: modest sizes to run quickly; user can expand via CLI
    std::vector<size_t> dims{64,128};
    std::vector<size_t> nvecs{10000,100000};
    std::vector<size_t> nlists{100,1000};
    std::vector<size_t> ms{8,16};
    std::vector<size_t> metas{0, 1024*64};
    std::vector<int> zstd{0, 1};
    std::vector<int> mmap{0, 1};
    fs::path out_root = fs::current_path() / "bench_out";
    bool json=false, text=true;

    for (int i=1;i<argc;++i) {
        std::string a(argv[i]);
        auto eat=[&](std::string k){ return a.rfind(k,0)==0 ? std::optional<std::string>(a.substr(k.size())) : std::nullopt; };
        if (a=="--json") { json=true; text=false; }
        else if (a=="--both") { json=true; text=true; }
        else if (auto v=eat("--dims=")) dims = parse_list(*v);
        else if (auto v=eat("--nvecs=")) nvecs = parse_list(*v);
        else if (auto v=eat("--nlists=")) nlists = parse_list(*v);
        else if (auto v=eat("--ms=")) ms = parse_list(*v);
        else if (auto v=eat("--metas=")) metas = parse_list(*v);
        else if (auto v=eat("--zstd=")) { zstd.clear(); for (auto x: parse_list(*v)) zstd.push_back(static_cast<int>(x)); }
        else if (auto v=eat("--mmap=")) { mmap.clear(); for (auto x: parse_list(*v)) mmap.push_back(static_cast<int>(x)); }
        else if (auto v=eat("--out=")) out_root = *v;
        else if (a=="--help"||a=="-h") {
            std::cout << "Usage: ivfpq_persist_bench [--dims=64,128,256,512] [--nvecs=10000,100000,1000000]\n"
                         "                         [--nlists=100,1000,10000] [--ms=8,16,32] [--metas=0,1024,65536,1048570]\n"
                         "                         [--zstd=0,1,3] [--mmap=0,1] [--out=bench_out] [--json|--both]\n";
            return 0;
        }
    }

    fs::create_directories(out_root);

    for (auto d: dims) for (auto n: nvecs) for (auto nl: nlists) for (auto m: ms) for (auto meta: metas) for (auto z: zstd) for (auto mm: mmap) {
        Scenario sc; sc.dim=d; sc.nvec=n; sc.nlist=static_cast<uint32_t>(nl); sc.m=static_cast<uint32_t>(m); sc.meta_bytes=meta; sc.zstd_level=z; sc.use_v11=true; sc.load_mmap=(mm!=0);
        if ((sc.dim % sc.m)!=0) continue; // invalid
        if (sc.nvec < sc.nlist) continue; // invalid
        fs::path od = out_root / ("d"+std::to_string(d)+"_n"+std::to_string(n)+"_nl"+std::to_string(nl)+"_m"+std::to_string(m)+"_meta"+std::to_string(meta)+"_z"+std::to_string(z)+"_mm"+std::to_string(mm));
        auto res = run_benchmark(sc, od);
        if (text) print_text(sc, res);
        if (json) print_json_line(sc, res);
    }

    return 0;
}

