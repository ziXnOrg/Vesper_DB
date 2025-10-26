#include <catch2/catch_test_macros.hpp>
#include <vesper/index/ivf_pq.hpp>

#include <filesystem>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cstdlib>

namespace fs = std::filesystem;
using vesper::index::IvfPqIndex;
using vesper::core::error;
using vesper::core::error_code;

static void write_bytes(std::ofstream& f, const void* ptr, size_t n){ f.write(reinterpret_cast<const char*>(ptr), static_cast<std::streamsize>(n)); }
static void fnv_update(uint64_t& h, const void* ptr, size_t n){ const auto* p = static_cast<const uint8_t*>(ptr); constexpr uint64_t F=1099511628211ull; for(size_t i=0;i<n;++i){ h^=p[i]; h*=F; } }

static void write_minimal_v11_ok(const fs::path& dir) {
    fs::create_directories(dir);
    std::ofstream f(dir / "ivfpq.bin", std::ios::binary);
    const char magic[8] = {'I','V','F','P','Q','v','1','1'}; write_bytes(f, magic, 8);
    uint16_t maj=1, min=1; write_bytes(f, &maj, 2); write_bytes(f, &min, 2);
    uint32_t flags=0; write_bytes(f, &flags, 4);
    uint32_t dim=4, nlist=1, m=1, nbits=8, dsub=4; uint64_t nvec=0; uint32_t code_size=1; uint64_t ts=0;
    write_bytes(f, &dim, 4); write_bytes(f, &nlist,4); write_bytes(f, &m,4); write_bytes(f,&nbits,4);
    write_bytes(f,&dsub,4); write_bytes(f,&nvec,8); write_bytes(f,&code_size,4); write_bytes(f,&ts,8);
    uint32_t meta_len=0; write_bytes(f,&meta_len,4);
    struct SH { uint32_t t; uint64_t u; uint64_t c; uint64_t sh; };
    auto fnv64 = [&](const void* ptr, size_t n){ const auto* p = static_cast<const uint8_t*>(ptr); uint64_t h=1469598103934665603ull; constexpr uint64_t F=1099511628211ull; for(size_t i=0;i<n;++i){ h^=p[i]; h*=F;} return h; };
    // centroids: 1x4 floats
    float cent[4] = {0,1,2,3}; SH h1{1, sizeof(cent), sizeof(cent), fnv64(cent, sizeof(cent))};
    write_bytes(f,&h1,sizeof(h1)); write_bytes(f,cent,sizeof(cent));
    // codebooks: m*(2^nbits)*dsub = 1*256*4 floats
    std::vector<float> cbs(256*4, 0.f); SH h2{2, cbs.size()*sizeof(float), cbs.size()*sizeof(float), fnv64(cbs.data(), cbs.size()*sizeof(float))};
    write_bytes(f,&h2,sizeof(h2)); write_bytes(f,cbs.data(),cbs.size()*sizeof(float));
    // inverted lists: count=1, size=0
    std::vector<uint8_t> inv; inv.resize(4+8); uint32_t cnt=1; std::memcpy(inv.data(), &cnt, 4); uint64_t szl=0; std::memcpy(inv.data()+4, &szl, 8);
    SH h3{3, inv.size(), inv.size(), fnv64(inv.data(), inv.size())}; write_bytes(f,&h3,sizeof(h3)); write_bytes(f,inv.data(),inv.size());
    // trailer: compute checksum over all previous bytes
    f.flush(); f.close();
    std::ifstream in(dir / "ivfpq.bin", std::ios::binary); std::vector<char> buf((std::istreambuf_iterator<char>(in)), {});
    uint64_t ch = 1469598103934665603ull; fnv_update(ch, buf.data(), buf.size());
    std::ofstream f2(dir / "ivfpq.bin", std::ios::binary | std::ios::app);
    const char tail[4] = {'C','H','K','S'}; write_bytes(f2, tail, 4); write_bytes(f2, &ch, 8);
}

TEST_CASE("v1.1 streaming loader reads first section header after early checksum", "[ivfpq][v11][streampos]") {
    fs::path dir = fs::temp_directory_path() / "vesper_v11_streampos";
    write_minimal_v11_ok(dir);

    // Ensure streaming path (not mmap)
#ifdef _WIN32
    _putenv_s("VESPER_IVFPQ_LOAD_MMAP", "0");
#else
    setenv("VESPER_IVFPQ_LOAD_MMAP", "0", 1);
#endif

    auto r = IvfPqIndex::load(dir.string());
    REQUIRE(r.has_value());

    const auto& index = r.value();
    // If stream positioning was wrong after early checksum, parsing would fail.
    // Assert key properties parsed from first section (centroids) and header.
    auto nlists = index.get_num_clusters();
    REQUIRE(nlists.has_value());
    REQUIRE(nlists.value() == 1);

    auto dim = index.get_dimension();
    REQUIRE(dim.has_value());
    REQUIRE(dim.value() == 4);

    auto cent0 = index.get_cluster_centroid(0);
    REQUIRE(cent0.has_value());
    REQUIRE(cent0->size() == 4);
    CHECK((*cent0)[0] == 0.0f);
    CHECK((*cent0)[1] == 1.0f);
    CHECK((*cent0)[2] == 2.0f);
    CHECK((*cent0)[3] == 3.0f);
}

