#include <benchmark/benchmark.h>
#include <vesper/index/ivf_pq.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "vesper/core/platform_utils.hpp"
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#ifdef VESPER_HAS_ZSTD
#include <zstd.h>
#endif

namespace fs = std::filesystem;
using vesper::index::IvfPqIndex;

static void fnv_update(uint64_t& h, const void* ptr, size_t nbytes) {
    const auto* p = static_cast<const uint8_t*>(ptr);
    constexpr uint64_t FNV_PRIME = 1099511628211ull;
    for (size_t i = 0; i < nbytes; ++i) { h ^= p[i]; h *= FNV_PRIME; }
}

static bool write_v10_synthetic(const fs::path& dir,
                                uint32_t dim,
                                uint32_t nlist,
                                uint32_t m,
                                uint32_t nbits,
                                uint32_t avg_list_size,
                                uint32_t seed) {
    fs::create_directories(dir);
    const fs::path file_path = dir / "ivfpq.bin";
    std::ofstream file(file_path, std::ios::binary);
    if (!file) return false;

    auto write_bytes = [&](const void* ptr, size_t nbytes){ file.write(reinterpret_cast<const char*>(ptr), static_cast<std::streamsize>(nbytes)); };
    auto write_and_hash = [&](uint64_t& h, const void* ptr, size_t nbytes){ write_bytes(ptr, nbytes); fnv_update(h, ptr, nbytes); };

    uint64_t checksum = 1469598103934665603ull;

    // Header
    const char magic[8] = {'I','V','F','P','Q','v','1','0'};
    write_and_hash(checksum, magic, sizeof(magic));
    uint16_t ver_major = 1, ver_minor = 0;
    write_and_hash(checksum, &ver_major, sizeof(ver_major));
    write_and_hash(checksum, &ver_minor, sizeof(ver_minor));

    // Flags (no OPQ, no RaBitQ)
    uint32_t flags = 0;
    write_and_hash(checksum, &flags, sizeof(flags));

    // Core dims/params
    uint32_t dsub = dim / m;
    uint64_t nvec = static_cast<uint64_t>(nlist) * avg_list_size;
    uint32_t code_size = m;
    uint64_t build_ts = 0;

    write_and_hash(checksum, &dim, sizeof(dim));
    write_and_hash(checksum, &nlist, sizeof(nlist));
    write_and_hash(checksum, &m, sizeof(m));
    write_and_hash(checksum, &nbits, sizeof(nbits));
    write_and_hash(checksum, &dsub, sizeof(dsub));
    write_and_hash(checksum, &nvec, sizeof(nvec));
    write_and_hash(checksum, &code_size, sizeof(code_size));
    write_and_hash(checksum, &build_ts, sizeof(build_ts));

    // Metadata (empty)
    uint32_t meta_len = 0; write_and_hash(checksum, &meta_len, sizeof(meta_len));

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uf(-1.0f, 1.0f);
    std::uniform_int_distribution<int> ui(0, (1u << nbits) - 1);

    // Coarse centroids [nlist x dim]
    std::vector<float> row(dim);
    for (uint32_t c = 0; c < nlist; ++c) {
        for (uint32_t d = 0; d < dim; ++d) row[d] = uf(rng);
        write_and_hash(checksum, row.data(), sizeof(float) * dim);
    }

    // PQ codebooks [m*ksub x dsub]
    const size_t ksub = 1u << nbits;
    std::vector<float> codebooks(static_cast<size_t>(m) * ksub * dsub);
    for (auto& v : codebooks) v = uf(rng);
    write_and_hash(checksum, codebooks.data(), codebooks.size() * sizeof(float));

    // No OPQ rotation matrix (flags=0)

    // Inverted lists
    uint32_t lists_count = nlist; write_and_hash(checksum, &lists_count, sizeof(lists_count));
    uint64_t cur_id = 0;
    std::vector<uint8_t> code(m);
    for (uint32_t li = 0; li < nlist; ++li) {
        uint64_t sz = avg_list_size;
        write_and_hash(checksum, &sz, sizeof(sz));
        for (uint64_t i = 0; i < sz; ++i) {
            write_and_hash(checksum, &cur_id, sizeof(cur_id));
            for (uint32_t j = 0; j < m; ++j) code[j] = static_cast<uint8_t>(ui(rng));
            write_and_hash(checksum, code.data(), code.size());
            ++cur_id;
        }
    }

    // Trailer checksum (not hashed)
    const char tail[4] = {'C','H','K','S'};
    write_bytes(tail, sizeof(tail));
    write_bytes(&checksum, sizeof(checksum));

    return true;
}

struct GenParams {
    uint32_t dim = 128;
    uint32_t nlist = 8192;
    uint32_t m = 16;
    uint32_t nbits = 8;
    uint32_t avg_list_size = 100;
    uint32_t seed = 1234;
};

static bool write_v11_synthetic(const fs::path& dir,
                                 uint32_t dim,
                                 uint32_t nlist,
                                 uint32_t m,
                                 uint32_t nbits,
                                 uint32_t avg_list_size,
                                 uint32_t seed) {
    fs::create_directories(dir);
    const fs::path file_path = dir / "ivfpq.bin";
    std::ofstream file(file_path, std::ios::binary);
    if (!file) return false;

    auto write_bytes = [&](const void* ptr, size_t nbytes){ file.write(reinterpret_cast<const char*>(ptr), static_cast<std::streamsize>(nbytes)); };
    auto fnv64 = [&](const void* ptr, size_t nbytes){ const auto* p = static_cast<const uint8_t*>(ptr); uint64_t h=1469598103934665603ull; constexpr uint64_t F=1099511628211ull; for(size_t i=0;i<nbytes;++i){ h^=p[i]; h*=F; } return h; };
    uint64_t checksum = 1469598103934665603ull;

    // Header
    const char magic[8] = {'I','V','F','P','Q','v','1','1'};
    write_bytes(magic, sizeof(magic));
    uint16_t ver_major = 1, ver_minor = 1; write_bytes(&ver_major, sizeof(ver_major)); write_bytes(&ver_minor, sizeof(ver_minor));
    uint32_t flags = 0; write_bytes(&flags, sizeof(flags));
    uint32_t dsub = dim / m; uint64_t nvec = static_cast<uint64_t>(nlist) * avg_list_size; uint32_t code_size = m; uint64_t build_ts = 0;
    write_bytes(&dim, sizeof(dim)); write_bytes(&nlist, sizeof(nlist)); write_bytes(&m, sizeof(m)); write_bytes(&nbits, sizeof(nbits));
    write_bytes(&dsub, sizeof(dsub)); write_bytes(&nvec, sizeof(nvec)); write_bytes(&code_size, sizeof(code_size)); write_bytes(&build_ts, sizeof(build_ts));
    uint32_t meta_len = 0; write_bytes(&meta_len, sizeof(meta_len));

    struct SectionHdr { uint32_t type; uint64_t unc; uint64_t comp; uint64_t shash; };
    constexpr uint32_t SEC_CENTROIDS=1, SEC_CODEBOOKS=2, SEC_INVERTED=3, SEC_OPQ=4;

    auto write_section = [&](uint32_t type, const void* data, size_t bytes){
        SectionHdr hdr{type, static_cast<uint64_t>(bytes), static_cast<uint64_t>(bytes), fnv64(data, bytes)};
#ifdef VESPER_HAS_ZSTD
        int level = 0; if (auto e = vesper::core::safe_getenv("VESPER_IVFPQ_ZSTD_LEVEL")) level = std::atoi(e->c_str());
        std::vector<char> comp;
        if (level > 0 && bytes > 0) {
            size_t bound = ZSTD_compressBound(bytes); comp.resize(bound);
            size_t got = ZSTD_compress(comp.data(), comp.size(), data, bytes, level);
            if (!ZSTD_isError(got) && got < bytes) { hdr.comp = got; }
        }
        write_bytes(&hdr, sizeof(hdr));
        if (hdr.comp == hdr.unc) {
            write_bytes(data, bytes);
        } else {
            write_bytes(comp.data(), static_cast<size_t>(hdr.comp));
        }
#else
        write_bytes(&hdr, sizeof(hdr)); write_bytes(data, bytes);
#endif
        // Update global checksum over header + payload
        fnv_update(checksum, &hdr, sizeof(hdr));
        if (hdr.comp == hdr.unc) fnv_update(checksum, data, bytes);
#ifdef VESPER_HAS_ZSTD
        else fnv_update(checksum, comp.data(), static_cast<size_t>(hdr.comp));
#endif
    };

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uf(-1.0f, 1.0f);
    std::uniform_int_distribution<int> ui(0, (1u << nbits) - 1);

    // Centroids
    std::vector<float> centroids(static_cast<size_t>(nlist) * dim);
    for (auto& v : centroids) v = uf(rng);
    write_section(SEC_CENTROIDS, centroids.data(), centroids.size()*sizeof(float));

    // Codebooks
    const size_t ksub = 1u << nbits;
    std::vector<float> codebooks(static_cast<size_t>(m) * ksub * dsub);
    for (auto& v : codebooks) v = uf(rng);
    write_section(SEC_CODEBOOKS, codebooks.data(), codebooks.size()*sizeof(float));

    // No OPQ

    // Inverted lists
    std::vector<uint8_t> buf;
    {
        // Build inverted section into buffer: [lists_count][for each list: sz][(id, code)*]
        std::vector<uint8_t> tmp;
        auto append = [&](const void* ptr, size_t n){ size_t off = tmp.size(); tmp.resize(off+n); std::memcpy(tmp.data()+off, ptr, n); };
        uint32_t lists_count = nlist; append(&lists_count, sizeof(lists_count));
        uint64_t cur_id = 0; std::vector<uint8_t> code(m);
        for (uint32_t li=0; li<nlist; ++li) {
            uint64_t szl = avg_list_size; append(&szl, sizeof(szl));
            for (uint64_t i=0;i<szl;++i) {
                append(&cur_id, sizeof(cur_id));
                for (uint32_t j=0;j<m;++j) code[j] = static_cast<uint8_t>(ui(rng));
                append(code.data(), code.size());
                ++cur_id;
            }
        }
        buf.swap(tmp);
    }
    write_section(SEC_INVERTED, buf.data(), buf.size());

    // Trailer
    const char tail[4] = {'C','H','K','S'}; write_bytes(tail, sizeof(tail)); write_bytes(&checksum, sizeof(checksum));
    return true;
}

static fs::path ensure_synthetic_index(const GenParams& p) {
    if (auto ov = vesper::core::safe_getenv("VESPER_BENCH_INDEX_DIR")) {
        fs::path dir = fs::path(*ov);
        fs::create_directories(dir);
        if (!fs::exists(dir / "ivfpq.bin")) {
            bool v11 = false; if (auto v = vesper::core::safe_getenv("VESPER_IVFPQ_SAVE_V11")) v11 = (!v->empty() && (*v)[0]=='1');
            if (v11) write_v11_synthetic(dir, p.dim, p.nlist, p.m, p.nbits, p.avg_list_size, p.seed);
            else write_v10_synthetic(dir, p.dim, p.nlist, p.m, p.nbits, p.avg_list_size, p.seed);
        }
        return dir;
    }
    fs::path base = fs::temp_directory_path() / "vesper_ivfpq_load_bench";
    fs::create_directories(base);
    // Use a folder name derived from params to reuse across runs
    char buf[128];
    std::snprintf(buf, sizeof(buf), "d%u_nl%u_m%u_b%u_L%u", p.dim, p.nlist, p.m, p.nbits, p.avg_list_size);
    fs::path dir = base / buf;
    if (!fs::exists(dir / "ivfpq.bin")) {
        write_v10_synthetic(dir, p.dim, p.nlist, p.m, p.nbits, p.avg_list_size, p.seed);
    }
    return dir;
}

static void SetEnvStreamCentroids(bool stream) {
#if defined(_WIN32)
    _putenv_s("VESPER_IVFPQ_LOAD_STREAM_CENTROIDS", stream ? "1" : "0");
#else
    if (stream) setenv("VESPER_IVFPQ_LOAD_STREAM_CENTROIDS", "1", 1);
    else setenv("VESPER_IVFPQ_LOAD_STREAM_CENTROIDS", "0", 1);
#endif
}

static void BM_IvfpqLoad_Baseline(benchmark::State& state) {
    GenParams p; // defaults
    auto dir = ensure_synthetic_index(p);
    SetEnvStreamCentroids(false);
    for (auto _ : state) {
        auto result = IvfPqIndex::load(dir.string());
        benchmark::DoNotOptimize(result);
        if (!result.has_value()) state.SkipWithError("Load failed (baseline)");
    }
    state.SetLabel(dir.string());
}
BENCHMARK(BM_IvfpqLoad_Baseline)->UseRealTime()->Unit(benchmark::kMillisecond);

static void BM_IvfpqLoad_Optimized(benchmark::State& state) {
    GenParams p; // defaults
    auto dir = ensure_synthetic_index(p);
    SetEnvStreamCentroids(true);
    for (auto _ : state) {
        auto result = IvfPqIndex::load(dir.string());
        benchmark::DoNotOptimize(result);
        if (!result.has_value()) state.SkipWithError("Load failed (optimized)");
    }
    state.SetLabel(dir.string());
}
BENCHMARK(BM_IvfpqLoad_Optimized)->UseRealTime()->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();

