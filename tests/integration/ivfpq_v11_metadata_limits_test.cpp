#include <catch2/catch_test_macros.hpp>
#include <vesper/index/ivf_pq.hpp>
#include <filesystem>
#include <vector>

using namespace vesper::index;
namespace fs = std::filesystem;

static auto tmp_path(const char* name) -> fs::path {
    auto base = fs::temp_directory_path();
    return base / (std::string(name) + "." + std::to_string(std::rand()));
}

TEST_CASE("IVFPQ metadata checked setter enforces size, UTF-8, and structural caps", "[ivfpq][metadata]") {
#ifdef _WIN32
    _putenv_s("VESPER_IVFPQ_SAVE_V11", "1");
#else
    setenv("VESPER_IVFPQ_SAVE_V11", "1", 1);
#endif

    // Train a tiny index
    IvfPqIndex idx;
    IvfPqTrainParams p{}; p.nlist = 4; p.m = 2; p.nbits = 8; p.use_opq = false;
    std::vector<float> train(64 * 8, 0.0f);
    auto tr = idx.train(train.data(), 8, 64, p);
    REQUIRE(tr.has_value());

    // Valid small JSON
    auto ok = idx.set_metadata_json_checked("{\"a\":1,\"b\":[true,false,null]}");
    REQUIRE(ok.has_value());

    // Too large (>64 KiB)
    std::string big(64 * 1024 + 1, 'x');
    auto too_big = idx.set_metadata_json_checked(std::string("\"") + big + "\""); // wrap as JSON string
    REQUIRE_FALSE(too_big.has_value());

    // Invalid UTF-8 (overlong encoding bytes in a string)
    std::string bad_utf8;
    bad_utf8.push_back('"'); bad_utf8.push_back(static_cast<char>(0xC0)); bad_utf8.push_back(static_cast<char>(0xAF)); bad_utf8.push_back('"');
    auto utf8_res = idx.set_metadata_json_checked(bad_utf8);
    REQUIRE_FALSE(utf8_res.has_value());

    // Too deep nesting (>64)
    std::string deep;
    for (int i = 0; i < 65; ++i) deep.push_back('[');
    deep += "0";
    for (int i = 0; i < 65; ++i) deep.push_back(']');
    auto deep_res = idx.set_metadata_json_checked(deep);
    REQUIRE_FALSE(deep_res.has_value());

    // Too many object keys (>4096)
    std::string many = "{";
    for (int i = 0; i < 4097; ++i) {
        many += "\"k" + std::to_string(i) + "\":0";
        if (i != 4096) many += ",";
    }
    many += "}";
    auto many_res = idx.set_metadata_json_checked(many);
    REQUIRE_FALSE(many_res.has_value());
}

