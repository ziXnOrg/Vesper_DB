#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_all.hpp>

#include "vesper/index/ivf_pq.hpp"

#include <vector>
#include <random>
#include <filesystem>
#include <cstdlib>
#include <fstream>
#include <chrono>


using namespace vesper::index;
namespace fs = std::filesystem;

static void set_env(const char* k, const char* v) {
#if defined(_WIN32)
    _putenv_s(k, v);
#else
    setenv(k, v, 1);
#endif
}

static void make_dataset(std::size_t n, std::size_t dim, std::vector<float>& out, unsigned seed=123) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    out.resize(n * dim);
    for (std::size_t i = 0; i < n * dim; ++i) out[i] = dist(rng);
}

static auto build_small_index(std::size_t dim,
                              const std::vector<float>& train,
                              const std::vector<float>& base)
    -> IvfPqIndex {
    IvfPqIndex index;
    IvfPqTrainParams params;
    params.nlist = 64;
    params.m = 8;
    params.nbits = 8;
    params.max_iter = 5;
    params.epsilon = 1e-3f;
    params.use_opq = false; // keep it simple/fast for compat checks
    REQUIRE(index.train(train.data(), dim, train.size()/dim, params).has_value());

    std::vector<std::uint64_t> ids(base.size()/dim);
    for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = static_cast<std::uint64_t>(i);
    REQUIRE(index.add(ids.data(), base.data(), ids.size()).has_value());
    return index;
}

static void assert_search_works(IvfPqIndex& idx, const std::vector<float>& queries, std::size_t dim) {
    IvfPqSearchParams sp; sp.nprobe = 8; sp.k = 10; sp.use_exact_rerank = false;
    for (std::size_t i = 0; i < std::min<std::size_t>(queries.size()/dim, 4); ++i) {
        auto r = idx.search(&queries[i*dim], sp);
        REQUIRE(r.has_value());
        REQUIRE(r->size() == 10);
    }
}

TEST_CASE("compat: v1.0 streaming load works", "[ivfpq][compat]") {
    const std::size_t dim = 32;
    std::vector<float> train, base, queries;
    make_dataset(256, dim, train);
    make_dataset(2000, dim, base);
    make_dataset(16, dim, queries);

    set_env("VESPER_IVFPQ_SAVE_V11", "0");
    set_env("VESPER_IVFPQ_LOAD_MMAP", "0");
    set_env("VESPER_IVFPQ_ZSTD_LEVEL", "0");

    auto idx = build_small_index(dim, train, base);

    std::random_device rdu; auto uniq = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "_" + std::to_string(static_cast<unsigned long long>(rdu()));
    fs::path dir = fs::temp_directory_path() / ("vesper_compat_v10_stream_" + uniq);
    std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);
    REQUIRE(idx.save(dir.string()).has_value());

    auto loaded = IvfPqIndex::load(dir.string());
    REQUIRE(loaded.has_value());
    assert_search_works(*loaded, queries, dim);
}

TEST_CASE("compat: v1.1 streaming load works", "[ivfpq][compat][v11]") {
    const std::size_t dim = 32;
    std::vector<float> train, base, queries;
    make_dataset(256, dim, train);
    make_dataset(2000, dim, base);
    make_dataset(16, dim, queries);

    set_env("VESPER_IVFPQ_SAVE_V11", "1");
    set_env("VESPER_IVFPQ_LOAD_MMAP", "0");
    set_env("VESPER_IVFPQ_ZSTD_LEVEL", "0");

    auto idx = build_small_index(dim, train, base);
    std::random_device rdu2; auto uniq2 = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "_" + std::to_string(static_cast<unsigned long long>(rdu2()));
    fs::path dir = fs::temp_directory_path() / ("vesper_compat_v11_stream_" + uniq2);
    std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);
    REQUIRE(idx.save(dir.string()).has_value());

    auto loaded = IvfPqIndex::load(dir.string());
    REQUIRE(loaded.has_value());
    assert_search_works(*loaded, queries, dim);
}

TEST_CASE("compat: v1.1 mmap load works", "[ivfpq][compat][v11][mmap]") {
    const std::size_t dim = 32;
    std::vector<float> train, base, queries;
    make_dataset(256, dim, train);
    make_dataset(2000, dim, base);
    make_dataset(16, dim, queries);

    set_env("VESPER_IVFPQ_SAVE_V11", "1");
    set_env("VESPER_IVFPQ_LOAD_MMAP", "1");
    set_env("VESPER_IVFPQ_ZSTD_LEVEL", "0");

    auto idx = build_small_index(dim, train, base);
    std::random_device rdu3; auto uniq3 = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "_" + std::to_string(static_cast<unsigned long long>(rdu3()));
    fs::path dir = fs::temp_directory_path() / ("vesper_compat_v11_mmap_" + uniq3);
    std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);
    REQUIRE(idx.save(dir.string()).has_value());

    auto loaded = IvfPqIndex::load(dir.string());
    REQUIRE(loaded.has_value());
    assert_search_works(*loaded, queries, dim);
}

#ifdef VESPER_HAS_ZSTD
TEST_CASE("compat: v1.1 zstd streaming load works", "[ivfpq][compat][v11][zstd]") {
    const std::size_t dim = 32;
    std::vector<float> train, base, queries;
    make_dataset(256, dim, train);
    make_dataset(2000, dim, base);
    make_dataset(16, dim, queries);

    set_env("VESPER_IVFPQ_SAVE_V11", "1");
    set_env("VESPER_IVFPQ_LOAD_MMAP", "0");
    set_env("VESPER_IVFPQ_ZSTD_LEVEL", "1");

    auto idx = build_small_index(dim, train, base);
    std::random_device rdu4; auto uniq4 = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "_" + std::to_string(static_cast<unsigned long long>(rdu4()));
    fs::path dir = fs::temp_directory_path() / ("vesper_compat_v11_zstd_stream_" + uniq4);
    std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);
    REQUIRE(idx.save(dir.string()).has_value());

    auto loaded = IvfPqIndex::load(dir.string());
    REQUIRE(loaded.has_value());
    assert_search_works(*loaded, queries, dim);
}

TEST_CASE("compat: v1.1 zstd mmap load works", "[ivfpq][compat][v11][zstd][mmap]") {
    const std::size_t dim = 32;
    std::vector<float> train, base, queries;
    make_dataset(256, dim, train);
    make_dataset(2000, dim, base);
    make_dataset(16, dim, queries);

    set_env("VESPER_IVFPQ_SAVE_V11", "1");
    set_env("VESPER_IVFPQ_LOAD_MMAP", "1");
    set_env("VESPER_IVFPQ_ZSTD_LEVEL", "1");

    auto idx = build_small_index(dim, train, base);
    std::random_device rdu5; auto uniq5 = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "_" + std::to_string(static_cast<unsigned long long>(rdu5()));
    fs::path dir = fs::temp_directory_path() / ("vesper_compat_v11_zstd_mmap_" + uniq5);
    std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);
    REQUIRE(idx.save(dir.string()).has_value());

    auto loaded = IvfPqIndex::load(dir.string());
    REQUIRE(loaded.has_value());
    assert_search_works(*loaded, queries, dim);
}
#endif

TEST_CASE("compat: future (too-new) version is rejected", "[ivfpq][compat][forward]") {
    // Manually write a minimal v1.1-like file but with a too-new major version to simulate future format
    fs::path dir = fs::temp_directory_path() / "vesper_compat_future_version";
    std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);
    fs::path fpath = dir / "ivfpq.bin";

    auto write_bytes = [](std::ofstream& f, const void* ptr, size_t n){ f.write(reinterpret_cast<const char*>(ptr), static_cast<std::streamsize>(n)); };
    auto fnv_update = [](uint64_t& h, const void* ptr, size_t n){ const auto* p = static_cast<const uint8_t*>(ptr); constexpr uint64_t F=1099511628211ull; for(size_t i=0;i<n;++i){ h^=p[i]; h*=F; } };

    {
        std::ofstream f(fpath, std::ios::binary);
        uint64_t H=1469598103934665603ull;
        const char magic[8] = {'I','V','F','P','Q','v','1','1'}; write_bytes(f, magic, 8);
        uint16_t maj=9, min=0; write_bytes(f, &maj, 2); write_bytes(f, &min, 2); // too-new major
        uint32_t flags=0; write_bytes(f, &flags, 4);
        uint32_t dim=4, nlist=1, m=1, nbits=8, dsub=4; uint64_t nvec=0; uint32_t code_size=1; uint64_t ts=0;
        write_bytes(f, &dim, 4); write_bytes(f, &nlist,4); write_bytes(f, &m,4); write_bytes(f,&nbits,4);
        write_bytes(f,&dsub,4); write_bytes(f,&nvec,8); write_bytes(f,&code_size,4); write_bytes(f,&ts,8);
        uint32_t meta_len=0; write_bytes(f,&meta_len,4);
        // No sections, just write trailer
        f.flush(); f.close();
        // trailer checksum over what we wrote
        std::ifstream in(fpath, std::ios::binary); std::vector<char> buf((std::istreambuf_iterator<char>(in)), {});
        uint64_t ch = 1469598103934665603ull; fnv_update(ch, buf.data(), buf.size());
        std::ofstream f2(fpath, std::ios::binary | std::ios::app);
        const char tail[4] = {'C','H','K','S'}; write_bytes(f2, tail, 4); write_bytes(f2, &ch, 8);
    }

    auto r = IvfPqIndex::load(dir.string());
    REQUIRE_FALSE(r.has_value());
    // Accept config_invalid for unsupported/too-new version
    REQUIRE(r.error().code == vesper::core::error_code::config_invalid);
}

