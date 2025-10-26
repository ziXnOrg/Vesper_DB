#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_all.hpp>

#include "vesper/index/ivf_pq.hpp"

#include <random>
#include <vector>
#include <filesystem>
#include <numeric>
#include <fstream>
#include <array>
#include <chrono>

using namespace vesper::index;

static void make_dataset(std::size_t n, std::size_t dim, std::vector<float>& out) {
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    out.resize(n * dim);
    for (std::size_t i = 0; i < n * dim; ++i) out[i] = dist(rng);
}

TEST_CASE("IVF-PQ save/load roundtrip preserves results", "[integration][ivfpq][serialize]") {
    const std::size_t dim = 32;
    const std::size_t n_train = 3000;  // modest size for speed
    const std::size_t n_add = 2000;
    const std::size_t n_query = 8;
    const std::uint32_t k = 10;

    std::vector<float> train;
    std::vector<float> base;
    std::vector<float> queries;
    make_dataset(n_train, dim, train);
    make_dataset(n_add, dim, base);
    make_dataset(n_query, dim, queries);

    IvfPqTrainParams params;
    params.nlist = 128;
    params.m = 8;
    params.nbits = 8;
    params.max_iter = 10;
    params.epsilon = 1e-3f;
    params.use_opq = true;           // cover OPQ path
    params.opq_iter = 1;             // keep very small for speed
    params.opq_sample_n = 512;       // tiny sample
    params.opq_init = OpqInit::Identity; // avoid PCA cost

    IvfPqIndex index;
    auto tr = index.train(train.data(), dim, n_train, params);
    REQUIRE(tr.has_value());

    std::vector<std::uint64_t> ids(n_add);
    std::iota(ids.begin(), ids.end(), 0ull);
    auto add_ok = index.add(ids.data(), base.data(), n_add);
    REQUIRE(add_ok.has_value());

    IvfPqSearchParams sp;
    sp.nprobe = 16;
    sp.k = k;
    sp.use_exact_rerank = false; // rely on IVF-PQ-only path for determinism

    // Collect baseline results
    std::vector<std::pair<std::uint64_t,float>> baseline_concat; baseline_concat.reserve(n_query * k);
    for (std::size_t i = 0; i < n_query; ++i) {
        auto r = index.search(&queries[i * dim], sp);
        REQUIRE(r.has_value());
        REQUIRE(r->size() == k);
        baseline_concat.insert(baseline_concat.end(), r->begin(), r->end());
    }

    // Save to a unique temp directory to avoid cross-process collisions
    std::random_device rd; auto unique_suffix = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "_" + std::to_string(static_cast<unsigned long long>(rd()));
    const auto tmpdir = std::filesystem::temp_directory_path() / ("vesper_ivfpq_rt_" + unique_suffix);
    std::error_code ec;
    std::filesystem::remove_all(tmpdir, ec);
    auto mk = std::filesystem::create_directories(tmpdir, ec);
    REQUIRE(!ec);

    auto save_ok = index.save(tmpdir.string());
    REQUIRE(save_ok.has_value());

    // Load
    auto loaded = IvfPqIndex::load(tmpdir.string());
    REQUIRE(loaded.has_value());

    // Compare results
    for (std::size_t i = 0; i < n_query; ++i) {
        auto r = loaded->search(&queries[i * dim], sp);
        REQUIRE(r.has_value());
        REQUIRE(r->size() == k);
        for (std::size_t j = 0; j < k; ++j) {
            const auto& base_pair = baseline_concat[i * k + j];
            const auto& new_pair = (*r)[j];
            REQUIRE(base_pair.first == new_pair.first);
            REQUIRE(new_pair.second == Catch::Approx(base_pair.second).margin(1e-6));
        }
    }

    SUCCEED();
}



TEST_CASE("IVF-PQ roundtrip with OPQ PCA and exact rerank", "[integration][ivfpq][serialize][opq][rerank]") {
    const std::size_t dim = 64;
    const std::size_t n_train = 8000;
    const std::size_t n_add = 6000;
    const std::size_t n_query = 10;
    const std::uint32_t k = 10;

    std::vector<float> train, base, queries;
    make_dataset(n_train, dim, train);
    make_dataset(n_add, dim, base);
    make_dataset(n_query, dim, queries);

    IvfPqTrainParams params;
    params.nlist = 256;
    params.m = 8;
    params.nbits = 8;
    params.max_iter = 15;
    params.epsilon = 1e-3f;
    params.use_opq = true;
    params.opq_iter = 2;
    params.opq_sample_n = 2048;
    params.opq_init = OpqInit::PCA;

    IvfPqIndex index;
    auto tr = index.train(train.data(), dim, n_train, params);
    REQUIRE(tr.has_value());

    std::vector<std::uint64_t> ids(n_add); std::iota(ids.begin(), ids.end(), 0ull);
    auto add_ok = index.add(ids.data(), base.data(), n_add);
    REQUIRE(add_ok.has_value());

    IvfPqSearchParams sp;
    sp.nprobe = 32;
    sp.k = k;
    sp.use_exact_rerank = true;
    sp.rerank_k = 64;

    std::vector<std::pair<std::uint64_t,float>> baseline_concat; baseline_concat.reserve(n_query * k);
    for (std::size_t i = 0; i < n_query; ++i) {
        auto r = index.search(&queries[i * dim], sp);
        REQUIRE(r.has_value());
        REQUIRE(r->size() == k);
        baseline_concat.insert(baseline_concat.end(), r->begin(), r->end());
    }

    std::random_device rd2; auto unique_suffix2 = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "_" + std::to_string(static_cast<unsigned long long>(rd2()));
    const auto tmpdir = std::filesystem::temp_directory_path() / ("vesper_ivfpq_rt_opq_pca_" + unique_suffix2);
    std::error_code ec;
    std::filesystem::remove_all(tmpdir, ec);
    std::filesystem::create_directories(tmpdir, ec);
    REQUIRE(!ec);

    auto save_ok = index.save(tmpdir.string());
    REQUIRE(save_ok.has_value());

    auto loaded = IvfPqIndex::load(tmpdir.string());
    REQUIRE(loaded.has_value());

    // Note: current design doesn't persist raw vectors; exact-rerank must operate on
    // available state. This test asserts roundtrip preserves behavior under the same
    // available data model.
    for (std::size_t i = 0; i < n_query; ++i) {
        auto r = loaded->search(&queries[i * dim], sp);
        REQUIRE(r.has_value());
        REQUIRE(r->size() == k);
        for (std::size_t j = 0; j < k; ++j) {
            const auto& base_pair = baseline_concat[i * k + j];
            const auto& new_pair  = (*r)[j];
            REQUIRE(base_pair.first == new_pair.first);
            REQUIRE(new_pair.second == Catch::Approx(base_pair.second).margin(1e-6));
        }
    }
}

#include "vesper/error.hpp"
TEST_CASE("IVF-PQ load error handling and corruption detection", "[integration][ivfpq][serialize][errors]") {
    using vesper::core::error_code;

    // 1) Non-existent path
    {
        auto res = IvfPqIndex::load("__this_dir_should_not_exist__/nope");
        REQUIRE_FALSE(res.has_value());
        auto err = res.error();
        REQUIRE(err.code == error_code::not_found);
    }

    auto mk_dir = [](const std::string& name) {
        auto dir = std::filesystem::temp_directory_path() / name;
        std::error_code ec; std::filesystem::remove_all(dir, ec); std::filesystem::create_directories(dir, ec);
        REQUIRE(!ec); return dir;
    };

    // 2) Corrupted magic header
    {
        auto dir = mk_dir("vesper_ivfpq_corrupt_magic");
        std::ofstream f((dir / "ivfpq.bin").string(), std::ios::binary);
        const char bad_magic[8] = {'B','A','D','M','A','G','I','C'};
        f.write(bad_magic, 8);
        std::array<char, 64-8> pad{}; f.write(pad.data(), pad.size());
        f.close();
        auto res = IvfPqIndex::load(dir.string());
        REQUIRE_FALSE(res.has_value());
        auto err = res.error();
        REQUIRE(err.code == error_code::config_invalid);
    }

    // 3) Unsupported version
    {
        auto dir = mk_dir("vesper_ivfpq_unsupported_ver");
        std::ofstream f((dir / "ivfpq.bin").string(), std::ios::binary);
        const char magic[8] = {'I','V','F','P','Q','v','1','0'}; f.write(magic, 8);
        std::uint16_t ver_major = 2, ver_minor = 0; f.write((char*)&ver_major, 2); f.write((char*)&ver_minor, 2);
        std::uint32_t flags = 0; f.write((char*)&flags, 4);
        std::uint32_t zeros32[5] = {0,0,0,0,0}; f.write((char*)zeros32, sizeof(zeros32));
        std::uint64_t nvec = 0; f.write((char*)&nvec, 8);
        std::uint32_t code_size = 0; f.write((char*)&code_size, 4);
        std::uint64_t ts = 0; f.write((char*)&ts, 8);
        std::uint32_t meta_len = 0; f.write((char*)&meta_len, 4);
        const char tail[4] = {'C','H','K','S'}; f.write(tail, 4);
        std::uint64_t fake = 0; f.write((char*)&fake, 8);
        f.close();
        auto res = IvfPqIndex::load(dir.string());
        REQUIRE_FALSE(res.has_value());
        auto err = res.error();
        REQUIRE(err.code == error_code::config_invalid);
    }

    // 4) Checksum mismatch
    {
        auto dir = mk_dir("vesper_ivfpq_checksum_mismatch");
        std::ofstream f((dir / "ivfpq.bin").string(), std::ios::binary);
        const char magic[8] = {'I','V','F','P','Q','v','1','0'}; f.write(magic, 8);
        std::uint16_t ver_major = 1, ver_minor = 0; f.write((char*)&ver_major, 2); f.write((char*)&ver_minor, 2);
        std::uint32_t flags = 0; f.write((char*)&flags, 4);
        std::uint32_t zeros32[5] = {0,0,0,0,0}; f.write((char*)zeros32, sizeof(zeros32));
        std::uint64_t nvec = 0; f.write((char*)&nvec, 8);
        std::uint32_t code_size = 0; f.write((char*)&code_size, 4);
        std::uint64_t ts = 0; f.write((char*)&ts, 8);
        std::uint32_t meta_len = 0; f.write((char*)&meta_len, 4);
        const char tail[4] = {'C','H','K','S'}; f.write(tail, 4);
        std::uint64_t fake_checksum = 0xDEADBEEFCAFEBABEull; f.write((char*)&fake_checksum, 8);
        f.close();
        auto res = IvfPqIndex::load(dir.string());
        REQUIRE_FALSE(res.has_value());
        auto err = res.error();
        REQUIRE(err.code == error_code::config_invalid);
        REQUIRE(!err.message.empty());
    }

    // 5) Truncated file -> io_failed
    {
        auto dir = mk_dir("vesper_ivfpq_truncated");
        std::ofstream f((dir / "ivfpq.bin").string(), std::ios::binary);
        const char magic[8] = {'I','V','F','P','Q','v','1','0'}; f.write(magic, 8);
        std::uint16_t ver_major = 1, ver_minor = 0; f.write((char*)&ver_major, 2); f.write((char*)&ver_minor, 2);
        std::uint32_t flags = 0; f.write((char*)&flags, 4);
        std::uint32_t dim=8, nlist=2, m=2, nbits=8, dsub=4;
        f.write((char*)&dim, 4); f.write((char*)&nlist, 4); f.write((char*)&m, 4); f.write((char*)&nbits, 4); f.write((char*)&dsub, 4);
        std::uint64_t nvec = 0; f.write((char*)&nvec, 8);
        std::uint32_t code_size = m; f.write((char*)&code_size, 4);
        std::uint64_t ts = 0; f.write((char*)&ts, 8);
        std::uint32_t meta_len = 0; f.write((char*)&meta_len, 4);
        // truncate before writing centroids to force read failure
        f.close();
        auto res = IvfPqIndex::load(dir.string());
        REQUIRE_FALSE(res.has_value());
        auto err = res.error();
        REQUIRE((err.code == error_code::io_failed || err.code == error_code::config_invalid));
    }

}

TEST_CASE("IVF-PQ quick OPQ PCA roundtrip (CI fast)", "[integration][ivfpq][serialize][opq][quick][ci-quick]") {
    const std::size_t dim = 32;
    const std::size_t n_train = 512;
    const std::size_t n_add = 400;
    const std::size_t n_query = 3;
    const std::uint32_t k = 5;

    std::vector<float> train, base, queries;
    make_dataset(n_train, dim, train);
    make_dataset(n_add, dim, base);
    make_dataset(n_query, dim, queries);

    IvfPqTrainParams params;
    params.nlist = 64;
    params.m = 8;
    params.nbits = 8;
    params.max_iter = 8;
    params.epsilon = 1e-3f;
    params.use_opq = true;
    params.opq_iter = 1;
    params.opq_sample_n = 256;
    params.opq_init = OpqInit::PCA;

    IvfPqIndex index;
    auto tr = index.train(train.data(), dim, n_train, params);
    REQUIRE(tr.has_value());

    std::vector<std::uint64_t> ids(n_add); std::iota(ids.begin(), ids.end(), 0ull);
    auto add_ok = index.add(ids.data(), base.data(), n_add);
    REQUIRE(add_ok.has_value());

    IvfPqSearchParams sp;
    sp.nprobe = 8;
    sp.k = k;
    sp.use_exact_rerank = false; // quickest deterministic path

    std::vector<std::pair<std::uint64_t,float>> baseline_concat; baseline_concat.reserve(n_query * k);
    for (std::size_t i = 0; i < n_query; ++i) {
        auto r = index.search(&queries[i * dim], sp);
        REQUIRE(r.has_value());
        REQUIRE(r->size() == k);
        baseline_concat.insert(baseline_concat.end(), r->begin(), r->end());
    }

    std::random_device rd3; auto unique_suffix3 = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "_" + std::to_string(static_cast<unsigned long long>(rd3()));
    const auto tmpdir = std::filesystem::temp_directory_path() / ("vesper_ivfpq_rt_quick_opq_pca_" + unique_suffix3);
    std::error_code ec; std::filesystem::remove_all(tmpdir, ec); std::filesystem::create_directories(tmpdir, ec); REQUIRE(!ec);

    auto save_ok = index.save(tmpdir.string());
    REQUIRE(save_ok.has_value());

    auto loaded = IvfPqIndex::load(tmpdir.string());
    REQUIRE(loaded.has_value());

    for (std::size_t i = 0; i < n_query; ++i) {
        auto r = loaded->search(&queries[i * dim], sp);
        REQUIRE(r.has_value());
        REQUIRE(r->size() == k);
        for (std::size_t j = 0; j < k; ++j) {
            const auto& base_pair = baseline_concat[i * k + j];
            const auto& new_pair  = (*r)[j];
            REQUIRE(base_pair.first == new_pair.first);
            REQUIRE(new_pair.second == Catch::Approx(base_pair.second).margin(1e-6));
        }
    }
}

TEST_CASE("IVF-PQ metadata roundtrip validation", "[integration][ivfpq][serialize][meta]") {
    const std::size_t dim = 32;
    const std::size_t n_train = 1024;
    const std::size_t n_add = 500;

    std::vector<float> train, base;
    make_dataset(n_train, dim, train);
    make_dataset(n_add, dim, base);

    IvfPqTrainParams params;
    params.nlist = 32;
    params.m = 8;
    params.nbits = 8;
    params.use_opq = true;
    params.opq_iter = 1;
    params.opq_sample_n = 256;
    params.opq_init = OpqInit::PCA;

    IvfPqIndex index;
    auto tr = index.train(train.data(), dim, n_train, params);
    REQUIRE(tr.has_value());

    std::vector<std::uint64_t> ids(n_add); std::iota(ids.begin(), ids.end(), 0ull);
    REQUIRE(index.add(ids.data(), base.data(), n_add).has_value());

    // Save/Load
    const auto tmpdir = std::filesystem::temp_directory_path() / "vesper_ivfpq_rt_meta";
    std::error_code ec; std::filesystem::remove_all(tmpdir, ec); std::filesystem::create_directories(tmpdir, ec); REQUIRE(!ec);
    REQUIRE(index.save(tmpdir.string()).has_value());
    auto loaded = IvfPqIndex::load(tmpdir.string());
    REQUIRE(loaded.has_value());

    // Validate metadata
    REQUIRE(loaded->dimension() == dim);
    auto stats = loaded->get_stats();
    REQUIRE(stats.n_lists == params.nlist);
    REQUIRE(stats.m == params.m);
    const std::size_t expected_code_size = (params.m * params.nbits + 7) / 8; // bytes
    REQUIRE(stats.code_size == expected_code_size);
    REQUIRE(stats.n_vectors == n_add);

    // Also cross-check via expected-returning helpers
    auto nlists_e = loaded->get_num_clusters();
    REQUIRE(nlists_e.has_value());
    REQUIRE(nlists_e.value() == params.nlist);
    auto dim_e = loaded->get_dimension();
    REQUIRE(dim_e.has_value());
    REQUIRE(dim_e.value() == dim);
}

// Add a targeted error case for the precise "File too small" branch
TEST_CASE("IVF-PQ load: file too small exact error", "[integration][ivfpq][serialize][errors]") {
    using vesper::core::error_code;
    auto dir = std::filesystem::temp_directory_path() / "vesper_ivfpq_file_too_small";
    std::error_code ec; std::filesystem::remove_all(dir, ec); std::filesystem::create_directories(dir, ec); REQUIRE(!ec);
    // Create ivfpq.bin with < 64 bytes to trigger the explicit check
    {
        std::ofstream f((dir / "ivfpq.bin").string(), std::ios::binary);
        const char few[16] = {0};
        f.write(few, sizeof(few));
    }
    auto res = IvfPqIndex::load(dir.string());
    REQUIRE_FALSE(res.has_value());
    auto err = res.error();
    REQUIRE(err.code == error_code::config_invalid);
    REQUIRE(std::string(err.message) == "File too small");
}




TEST_CASE("IVF-PQ OPQ rotation presence vs absence roundtrip", "[integration][ivfpq][serialize][opq]") {
    const std::size_t dim = 32;
    const std::size_t n_train = 3000;
    const std::size_t n_add = 2000;
    const std::size_t n_query = 6;
    const std::uint32_t k = 10;

    std::vector<float> train, base, queries;
    make_dataset(n_train, dim, train);
    make_dataset(n_add, dim, base);
    make_dataset(n_query, dim, queries);

    // Train A: no OPQ rotation
    IvfPqTrainParams params_id;
    params_id.nlist = 128; params_id.m = 8; params_id.nbits = 8;
    params_id.use_opq = false; // no rotation stored
    params_id.max_iter = 10; params_id.epsilon = 1e-3f;

    IvfPqIndex idx_id;
    REQUIRE(idx_id.train(train.data(), dim, n_train, params_id).has_value());
    std::vector<std::uint64_t> ids(n_add); std::iota(ids.begin(), ids.end(), 0ull);
    REQUIRE(idx_id.add(ids.data(), base.data(), n_add).has_value());

    // Train B: OPQ with PCA init
    IvfPqTrainParams params_pca = params_id;
    params_pca.use_opq = true;
    params_pca.opq_init = OpqInit::PCA;
    params_pca.opq_iter = 2; params_pca.opq_sample_n = 1024;

    IvfPqIndex idx_pca;
    REQUIRE(idx_pca.train(train.data(), dim, n_train, params_pca).has_value());
    REQUIRE(idx_pca.add(ids.data(), base.data(), n_add).has_value());

    IvfPqSearchParams sp; sp.nprobe = 16; sp.k = k; sp.use_exact_rerank = false;

    // Baselines
    std::vector<std::pair<std::uint64_t,float>> base_id; base_id.reserve(n_query*k);
    std::vector<std::pair<std::uint64_t,float>> base_pca; base_pca.reserve(n_query*k);
    for (std::size_t i=0;i<n_query;++i){
        auto r1 = idx_id.search(&queries[i*dim], sp); REQUIRE(r1.has_value()); REQUIRE(r1->size()==k);
        auto r2 = idx_pca.search(&queries[i*dim], sp); REQUIRE(r2.has_value()); REQUIRE(r2->size()==k);
        base_id.insert(base_id.end(), r1->begin(), r1->end());
        base_pca.insert(base_pca.end(), r2->begin(), r2->end());
    }

    // Expect behavioral difference between OPQ off vs on (IDs or distances differ for at least one query)
    bool any_diff=false;
    for (std::size_t i=0;i<n_query && !any_diff;++i){
        for (std::size_t j=0;j<k;++j){
            const auto& a = base_id[i*k+j];
            const auto& b = base_pca[i*k+j];
            if (a.first != b.first || std::fabs(a.second - b.second) > 1e-7f){ any_diff=true; break; }
        }
    }
    REQUIRE(any_diff);

    // Save/Load each and verify roundtrip preserves their own behavior
    std::random_device rd4; auto unique_suffix4 = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "_" + std::to_string(static_cast<unsigned long long>(rd4()));
    auto d_id = std::filesystem::temp_directory_path() / ("vesper_ivfpq_rt_opq_none_" + unique_suffix4);
    auto d_pca = std::filesystem::temp_directory_path() / ("vesper_ivfpq_rt_opq_pca2_" + unique_suffix4);
    std::error_code ec;
    std::filesystem::remove_all(d_id, ec); std::filesystem::create_directories(d_id, ec); REQUIRE(!ec);
    std::filesystem::remove_all(d_pca, ec); std::filesystem::create_directories(d_pca, ec); REQUIRE(!ec);

    REQUIRE(idx_id.save(d_id.string()).has_value());
    REQUIRE(idx_pca.save(d_pca.string()).has_value());

    auto l_id = IvfPqIndex::load(d_id.string()); REQUIRE(l_id.has_value());
    auto l_pca = IvfPqIndex::load(d_pca.string()); REQUIRE(l_pca.has_value());

    for (std::size_t i=0;i<n_query;++i){
        auto r1 = l_id->search(&queries[i*dim], sp); REQUIRE(r1.has_value()); REQUIRE(r1->size()==k);
        auto r2 = l_pca->search(&queries[i*dim], sp); REQUIRE(r2.has_value()); REQUIRE(r2->size()==k);
        for (std::size_t j=0;j<k;++j){
            const auto& a0 = base_id[i*k+j]; const auto& a1 = (*r1)[j];
            REQUIRE(a0.first == a1.first);
            REQUIRE(a1.second == Catch::Approx(a0.second).margin(1e-6));
            const auto& b0 = base_pca[i*k+j]; const auto& b1 = (*r2)[j];
            REQUIRE(b0.first == b1.first);
            REQUIRE(b1.second == Catch::Approx(b0.second).margin(1e-6));
        }
    }
}
