#include <catch2/catch_test_macros.hpp>
#include <random>
#include <vector>
#include <filesystem>

#include "vesper/index/product_quantizer.hpp"

using namespace vesper::index;

static void fill_rand(std::vector<float>& v, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : v) x = dist(rng);
}

TEST_CASE("ProductQuantizer save/load via std::filesystem::path", "[pq][io]") {
    std::mt19937 rng(1234);

    const std::size_t dim = 4;
    PqTrainParams p{};
    p.m = 2;          // dsub = 2
    p.nbits = 2;      // ksub = 4
    p.max_iter = 10;
    p.epsilon = 1e-4f;
    p.seed = 42;
    p.verbose = false;

    const std::size_t n_train = 32; // >= ksub * m = 8
    std::vector<float> train(n_train * dim);
    fill_rand(train, rng);

    ProductQuantizer pq;
    REQUIRE(pq.train(train.data(), n_train, dim, p));
    REQUIRE(pq.is_trained());

    // Prepare a temp directory and path
    std::filesystem::path dir = std::filesystem::temp_directory_path() / "vesper_pq_tests";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    std::filesystem::path path = dir / "pq_save_load_test.vspq";
    std::filesystem::remove(path, ec);

    // Save using the path overload
    REQUIRE(pq.save(path));

    // Load using the path overload
    auto loaded = ProductQuantizer::load(path);
    REQUIRE(loaded.has_value());

    // Basic sanity checks on loaded quantizer
    REQUIRE(loaded->is_trained());
    auto info = loaded->get_info();
    REQUIRE(info.m == p.m);
    REQUIRE(info.dim == dim);
    REQUIRE(info.ksub == (1u << p.nbits));

    // Cleanup (best-effort)
    std::filesystem::remove(path, ec);
    std::filesystem::remove(dir, ec);
}

