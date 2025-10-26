#include <catch2/catch_test_macros.hpp>
#include <vesper/index/ivf_pq.hpp>

#include <filesystem>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstring>

namespace fs = std::filesystem;
using vesper::index::IvfPqIndex;
using vesper::core::error;
using vesper::core::error_code;

static void write_bytes(std::ofstream& f, const void* ptr, size_t n){ f.write(reinterpret_cast<const char*>(ptr), static_cast<std::streamsize>(n)); }
static void fnv_update(uint64_t& h, const void* ptr, size_t n){ const auto* p = static_cast<const uint8_t*>(ptr); constexpr uint64_t F=1099511628211ull; for(size_t i=0;i<n;++i){ h^=p[i]; h*=F; } }

static void write_minimal_v11_ok(const fs::path& dir) {
    fs::create_directories(dir);
    std::ofstream f(dir / "ivfpq.bin", std::ios::binary);
    uint64_t H=1469598103934665603ull;
    const char magic[8] = {'I','V','F','P','Q','v','1','1'}; write_bytes(f, magic, 8);
    uint16_t maj=1, min=1; write_bytes(f, &maj, 2); write_bytes(f, &min, 2);
    uint32_t flags=0; write_bytes(f, &flags, 4);
    uint32_t dim=4, nlist=1, m=1, nbits=8, dsub=4; uint64_t nvec=0; uint32_t code_size=1; uint64_t ts=0;
    write_bytes(f, &dim, 4); write_bytes(f, &nlist,4); write_bytes(f, &m,4); write_bytes(f,&nbits,4);
    write_bytes(f,&dsub,4); write_bytes(f,&nvec,8); write_bytes(f,&code_size,4); write_bytes(f,&ts,8);
    uint32_t meta_len=0; write_bytes(f,&meta_len,4);
    // sections
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
    // trailer
    // compute checksum over all previous bytes: re-open and compute
    f.flush(); f.close();
    std::ifstream in(dir / "ivfpq.bin", std::ios::binary); std::vector<char> buf((std::istreambuf_iterator<char>(in)), {});
    uint64_t ch = 1469598103934665603ull; fnv_update(ch, buf.data(), buf.size());
    std::ofstream f2(dir / "ivfpq.bin", std::ios::binary | std::ios::app);
    const char tail[4] = {'C','H','K','S'}; write_bytes(f2, tail, 4); write_bytes(f2, &ch, 8);
}

TEST_CASE("v1.1 invalid magic header is rejected", "[ivfpq][v11][error]") {
    fs::path dir = fs::temp_directory_path() / "vesper_v11_badmagic";
    fs::create_directories(dir);
    // Write wrong magic
    {
        std::ofstream f(dir / "ivfpq.bin", std::ios::binary);
        const char magic[8] = {'B','A','D','M','A','G','I','C'}; write_bytes(f, magic, 8);
        uint64_t zero=0; write_bytes(f,&zero,8); // just filler
        const char tail[4] = {'C','H','K','S'}; write_bytes(f, tail, 4); write_bytes(f, &zero, 8);
    }
    auto r = IvfPqIndex::load(dir.string());
    REQUIRE_FALSE(r.has_value());
    REQUIRE(r.error().code == error_code::config_invalid);
}

TEST_CASE("v1.1 truncated file is rejected", "[ivfpq][v11][error]") {
    fs::path dir = fs::temp_directory_path() / "vesper_v11_trunc";
    write_minimal_v11_ok(dir);
    // Truncate last 4 bytes
    fs::path f = dir / "ivfpq.bin";
    auto sz = fs::file_size(f);
    REQUIRE(sz > 12);
    std::vector<char> data(sz - 4);
    {
        std::ifstream in(f, std::ios::binary); in.read(data.data(), static_cast<std::streamsize>(data.size()));
    }
    {
        std::ofstream out(f, std::ios::binary | std::ios::trunc); write_bytes(out, data.data(), data.size());
    }
    auto r = IvfPqIndex::load(dir.string());
    REQUIRE_FALSE(r.has_value());
    REQUIRE(r.error().code == error_code::config_invalid);
}

TEST_CASE("v1.1 section checksum mismatch is rejected", "[ivfpq][v11][error]") {
    fs::path dir = fs::temp_directory_path() / "vesper_v11_badsec";
    write_minimal_v11_ok(dir);
    // Corrupt one byte inside the file (in the codebooks payload)
    fs::path f = dir / "ivfpq.bin";
    // Flip a byte inside the codebooks payload directly on disk to avoid any in-memory copying issues
    struct SH { uint32_t t; uint64_t u; uint64_t c; uint64_t sh; };
    const size_t payload_offset = 60 + sizeof(SH) + 16 + sizeof(SH) + 10; // leave trailer (CHKS + 8B) intact
    {
        std::fstream io(f, std::ios::in | std::ios::out | std::ios::binary);
        io.seekg(0, std::ios::end);
        auto sz = static_cast<size_t>(io.tellg());
        REQUIRE(sz > payload_offset + 12);
        io.seekg(payload_offset, std::ios::beg);
        char c{}; io.read(&c, 1);
        io.seekp(payload_offset, std::ios::beg);
        c ^= static_cast<char>(0xFF);
        io.write(&c, 1);
        io.flush();
    }
    auto r = IvfPqIndex::load(dir.string());
    REQUIRE_FALSE(r.has_value());
    // Either section checksum mismatch (mmap path) or checksum mismatch (streaming)
    REQUIRE((r.error().code == error_code::config_invalid || r.error().code == error_code::io_failed));
}

TEST_CASE("v1.1 invalid section header type is tolerated (skipped)", "[ivfpq][v11][error]") {
    fs::path dir = fs::temp_directory_path() / "vesper_v11_badtype";
    write_minimal_v11_ok(dir);
    // Overwrite first section header type with invalid value
    fs::path f = dir / "ivfpq.bin";
    std::fstream io(f, std::ios::in | std::ios::out | std::ios::binary);
    // Seek to first section header after fixed header
    io.seekp(8 + 2 + 2 + 4 + 4 + 4 + 4 + 4 + 8 + 4 + 4, std::ios::beg);
    uint32_t bad = 9999; io.write(reinterpret_cast<const char*>(&bad), sizeof(bad)); io.flush(); io.close();
    // Should still detect invalid state later due to missing required sections
    auto r = IvfPqIndex::load(dir.string());
    REQUIRE_FALSE(r.has_value());
}

TEST_CASE("v1.1 decompression failure is graceful", "[ivfpq][v11][error]") {
    // Only meaningful if zstd is present; we simulate by writing compressed size != uncompressed size without valid payload
    fs::path dir = fs::temp_directory_path() / "vesper_v11_badcomp";
    fs::create_directories(dir);
    std::ofstream f(dir / "ivfpq.bin", std::ios::binary);
    const char magic[8] = {'I','V','F','P','Q','v','1','1'}; write_bytes(f, magic, 8);
    uint16_t maj=1,min=1; write_bytes(f,&maj,2); write_bytes(f,&min,2);
    uint32_t flags=0; write_bytes(f,&flags,4);
    uint32_t dim=4,nlist=1,m=1,nbits=8,dsub=4; uint64_t nvec=0; uint32_t code_size=1; uint64_t ts=0;
    write_bytes(f,&dim,4); write_bytes(f,&nlist,4); write_bytes(f,&m,4); write_bytes(f,&nbits,4);
    write_bytes(f,&dsub,4); write_bytes(f,&nvec,8); write_bytes(f,&code_size,4); write_bytes(f,&ts,8);
    uint32_t meta_len=0; write_bytes(f,&meta_len,4);
    struct SH { uint32_t t; uint64_t u; uint64_t c; uint64_t sh; };
    // Provide a section with comp < unc and garbage payload
    SH hdr{1, 16, 8, 0}; write_bytes(f,&hdr,sizeof(hdr));
    uint64_t garbage = 0xDEADBEEFDEADBEEFULL; write_bytes(f,&garbage,8);
    // trailer with dummy checksum over everything written so far
    f.flush(); f.close();
    std::ifstream in(dir / "ivfpq.bin", std::ios::binary); std::vector<char> buf((std::istreambuf_iterator<char>(in)), {});
    uint64_t ch = 1469598103934665603ull; fnv_update(ch, buf.data(), buf.size());
    std::ofstream f2(dir / "ivfpq.bin", std::ios::binary | std::ios::app);
    const char tail[4] = {'C','H','K','S'}; write_bytes(f2, tail, 4); write_bytes(f2, &ch, 8);

    auto r = IvfPqIndex::load(dir.string());
    REQUIRE_FALSE(r.has_value());
    REQUIRE(r.error().code == error_code::config_invalid);
}

