#include <catch2/catch_test_macros.hpp>
#include <vesper/index/ivf_pq.hpp>
#include "vesper/core/platform_utils.hpp"

#include <filesystem>
#include <fstream>
#include <random>

using namespace vesper::index;
namespace fs = std::filesystem;

static auto tmp_path(const char* name) -> fs::path {
    auto base = fs::temp_directory_path();
    std::mt19937_64 rng{7654321};
    for (int i = 0; i < 1000; ++i) {
        auto p = base / (std::string(name) + "." + std::to_string(rng()));
        if (!fs::exists(p)) return p;
    }
    return base / (std::string(name) + ".fallback");
}

TEST_CASE("IVFPQ v1.1 metadata JSON roundtrip and size limits", "[ivfpq][serialize][metadata]") {
    // Force v1.1 format
#ifdef _WIN32
    _putenv_s("VESPER_IVFPQ_SAVE_V11", "1");
#else
    setenv("VESPER_IVFPQ_SAVE_V11", "1", 1);
#endif

    // Build a tiny trained index in-memory
    IvfPqIndex idx;
    IvfPqTrainParams t{};
    t.nlist = 8; t.m = 4; t.nbits = 8; t.use_opq = false;

    const int dim = 16; const int n = 200;
    std::vector<float> train(n * dim);
    for (int i = 0; i < n * dim; ++i) train[i] = float(i % 13) * 0.1f;
    auto tr = idx.train(train.data(), dim, n, t);
    REQUIRE(tr.has_value());

    // Set small metadata and save
    const std::string meta = R"({"dataset":"toy","note":"hello","k":10})";
    idx.set_metadata_json(meta);

    auto path = tmp_path("ivfpq_meta_v11.vx");
    auto sr = idx.save(path.string().c_str());
    REQUIRE(sr.has_value());

    // Streaming load
    auto lr1 = IvfPqIndex::load(path.string().c_str());
    REQUIRE(lr1.has_value());
    CHECK(lr1->get_metadata_json() == meta);

    // Mmap load
#ifdef _WIN32
    _putenv_s("VESPER_IVFPQ_LOAD_MMAP", "1");
#else
    setenv("VESPER_IVFPQ_LOAD_MMAP", "1", 1);
#endif
    auto lr2 = IvfPqIndex::load(path.string().c_str());
    REQUIRE(lr2.has_value());
    CHECK(lr2->get_metadata_json() == meta);

#ifdef _WIN32
    _putenv_s("VESPER_IVFPQ_LOAD_MMAP", "0");
#else
    setenv("VESPER_IVFPQ_LOAD_MMAP", "0", 1);
#endif

    // Size limit enforcement: >64 KiB should be rejected at save
    std::string big(64 * 1024 + 1, 'x');
    IvfPqIndex idx_big;
    IvfPqTrainParams t2{}; t2.nlist = 4; t2.m = 2; t2.nbits = 8; t2.use_opq = false;
    std::vector<float> small_train(64 * 8, 0.0f);
    auto trb = idx_big.train(small_train.data(), 8, 64, t2);
    REQUIRE(trb.has_value());
    idx_big.set_metadata_json(big);
    auto sr2 = idx_big.save(tmp_path("should_fail_save.vx").string().c_str());
    REQUIRE_FALSE(sr2.has_value());
}

TEST_CASE("IVFPQ v1.1 metadata fallback from fixed header when no section present (opt-in)", "[ivfpq][serialize][metadata][optional]") {
    if (!vesper::core::safe_getenv("VESPER_TEST_HEADER_META")) {
        SUCCEED("skipped: set VESPER_TEST_HEADER_META=1 to run");
        return;
    }
    // This test simulates legacy behavior: write a file with meta in the fixed header
    // but without a SEC_METADATA section; loader should populate metadata_json from header

    // Manually craft a minimal v1.1 file with empty sections and header meta
    fs::path p = tmp_path("ivfpq_v11_header_meta_only.vx");
    std::ofstream f(p, std::ios::binary);
    REQUIRE(f.good());

    auto fnv_update = [](uint64_t& h, const void* ptr, size_t n){
        const auto* q = static_cast<const unsigned char*>(ptr);
        constexpr uint64_t P = 1099511628211ull;
        for (size_t i = 0; i < n; ++i) { h ^= q[i]; h *= P; }
    };
    uint64_t checksum = 1469598103934665603ull;
    auto write_h = [&](const void* buf, size_t n){ f.write(reinterpret_cast<const char*>(buf), n); REQUIRE(f.good()); fnv_update(checksum, buf, n); };

    const char magic[8] = {'I','V','F','P','Q','v','1','1'}; write_h(magic, 8);
    uint16_t ver_major=1, ver_minor=1; write_h(&ver_major,2); write_h(&ver_minor,2);
    uint32_t flags=0; write_h(&flags,4);
    uint32_t dim=16, nlist=1, m=1, nbits=8, dsub=16; uint64_t nvec=0; uint32_t code_size=m; uint64_t build_ts=0;
    write_h(&dim,4); write_h(&nlist,4); write_h(&m,4); write_h(&nbits,4); write_h(&dsub,4);
    write_h(&nvec,8); write_h(&code_size,4); write_h(&build_ts,8);

    // meta in header
    std::string hdr_meta = "{\"from\":\"header\"}";
    uint32_t meta_len = static_cast<uint32_t>(hdr_meta.size());
    write_h(&meta_len,4); if (meta_len) write_h(hdr_meta.data(), hdr_meta.size());

    // Helper FNV64 for section payloads
    auto fnv64 = [](const void* ptr, size_t n){ uint64_t h=1469598103934665603ull; const auto* p=(const unsigned char*)ptr; constexpr uint64_t P=1099511628211ull; for(size_t i=0;i<n;++i){h^=p[i]; h*=P;} return h; };

    struct Sec { uint32_t t; uint64_t u; uint64_t c; uint64_t h; };

    // SEC_CENTROIDS: nlist*dim floats
    std::vector<float> centroids(nlist * dim, 0.0f);
    Sec sc{1, static_cast<uint64_t>(centroids.size()*sizeof(float)), static_cast<uint64_t>(centroids.size()*sizeof(float)), fnv64(centroids.data(), centroids.size()*sizeof(float))};
    write_h(&sc, sizeof(sc)); if (sc.c) write_h(centroids.data(), centroids.size()*sizeof(float));

    // SEC_CODEBOOKS: m * (1<<nbits) * dsub floats
    const size_t cb_elems = size_t(m) * (1u<<nbits) * size_t(dsub);
    std::vector<float> codebooks(cb_elems, 0.0f);
    Sec sb{2, static_cast<uint64_t>(codebooks.size()*sizeof(float)), static_cast<uint64_t>(codebooks.size()*sizeof(float)), fnv64(codebooks.data(), codebooks.size()*sizeof(float))};
    write_h(&sb, sizeof(sb)); if (sb.c) write_h(codebooks.data(), codebooks.size()*sizeof(float));

    // SEC_INVERTED: lists_count (uint32), then per-list entries (we write empty list)
    std::vector<unsigned char> inv;
    {
        uint32_t lists_count = nlist; inv.insert(inv.end(), reinterpret_cast<unsigned char*>(&lists_count), reinterpret_cast<unsigned char*>(&lists_count)+4);
        // For each list: count(uint32)=0; then nothing
        uint32_t zero = 0; inv.insert(inv.end(), reinterpret_cast<unsigned char*>(&zero), reinterpret_cast<unsigned char*>(&zero)+4);
    }
    Sec si{3, static_cast<uint64_t>(inv.size()), static_cast<uint64_t>(inv.size()), fnv64(inv.data(), inv.size())};
    write_h(&si, sizeof(si)); if (si.c) write_h(inv.data(), inv.size());

    // Trailer
    const char tail[4] = {'C','H','K','S'}; f.write(tail,4); f.write(reinterpret_cast<const char*>(&checksum),8);
    f.close();

    // Load: should succeed and pick up header meta
    auto lr = IvfPqIndex::load(p.string().c_str());
    if (!lr.has_value()) {
        INFO("load error: " << static_cast<int>(lr.error().code) << " - " << lr.error().message);
    }
    REQUIRE(lr.has_value());
    CHECK(lr->get_metadata_json() == hdr_meta);
}

