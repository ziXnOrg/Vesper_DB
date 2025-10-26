#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_all.hpp>

#include "vesper/index/ivf_pq.hpp"

#include <random>
#include <vector>
#include <filesystem>
#include <numeric>
#include <cstdlib>
#include <chrono>


using namespace vesper::index;

static void make_dataset(std::size_t n, std::size_t dim, std::vector<float>& out) {
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    out.resize(n * dim);
    for (std::size_t i = 0; i < n * dim; ++i) out[i] = dist(rng);
}

static void set_env(const char* k, const char* v) {
#if defined(_WIN32)
    _putenv_s(k, v);
#else
    setenv(k, v, 1);
#endif
}

TEST_CASE("IVF-PQ v1.1 sectioned save/load roundtrip", "[integration][ivfpq][serialize][v11]") {
    const std::size_t dim = 32;
    const std::size_t n_train = 2500;
    const std::size_t n_add = 1500;
    const std::size_t n_query = 6;
    const std::uint32_t k = 10;

    std::vector<float> train, base, queries;
    make_dataset(n_train, dim, train);
    make_dataset(n_add, dim, base);
    make_dataset(n_query, dim, queries);

    IvfPqTrainParams params;
    params.nlist = 128;
    params.m = 8;
    params.nbits = 8;
    params.max_iter = 10;
    params.epsilon = 1e-3f;
    params.use_opq = true;
    params.opq_iter = 1;
    params.opq_sample_n = 512;
    params.opq_init = OpqInit::Identity;

    IvfPqIndex index;
    auto tr = index.train(train.data(), dim, n_train, params);
    REQUIRE(tr.has_value());

    std::vector<std::uint64_t> ids(n_add); std::iota(ids.begin(), ids.end(), 0ull);
    auto add_ok = index.add(ids.data(), base.data(), n_add);
    REQUIRE(add_ok.has_value());

    IvfPqSearchParams sp; sp.nprobe = 16; sp.k = k; sp.use_exact_rerank = false;

    // Baseline results from in-memory index
    std::vector<std::pair<std::uint64_t,float>> baseline;
    baseline.reserve(n_query * k);
    for (std::size_t i = 0; i < n_query; ++i) {
        auto r = index.search(&queries[i * dim], sp);
        REQUIRE(r.has_value()); REQUIRE(r->size() == k);
        baseline.insert(baseline.end(), r->begin(), r->end());
    }

    // Save v1.1 (sectioned). Ask for compression level 1 if available.
    set_env("VESPER_IVFPQ_SAVE_V11", "1");
#if defined(VESPER_HAS_ZSTD)
    set_env("VESPER_IVFPQ_ZSTD_LEVEL", "1");
#else
    set_env("VESPER_IVFPQ_ZSTD_LEVEL", "0");
#endif

    std::random_device rd; auto uniq = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "_" + std::to_string(static_cast<unsigned long long>(rd()));
    const auto tmpdir = std::filesystem::temp_directory_path() / ("vesper_ivfpq_v11_rt_" + uniq);
    std::error_code ec; std::filesystem::remove_all(tmpdir, ec); std::filesystem::create_directories(tmpdir, ec);
    REQUIRE(!ec);

    auto save_ok = index.save(tmpdir.string());
    REQUIRE(save_ok.has_value());

    // Load
    auto loaded = IvfPqIndex::load(tmpdir.string());
    REQUIRE(loaded.has_value());

    for (std::size_t i = 0; i < n_query; ++i) {
        auto r = loaded->search(&queries[i * dim], sp);
        REQUIRE(r.has_value()); REQUIRE(r->size() == k);
        for (std::size_t j = 0; j < k; ++j) {
            const auto& base_pair = baseline[i * k + j];
            const auto& new_pair = (*r)[j];
            REQUIRE(base_pair.first == new_pair.first);
            REQUIRE(new_pair.second == Catch::Approx(base_pair.second).margin(1e-6));
        }
    }
}

