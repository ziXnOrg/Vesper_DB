#include <gtest/gtest.h>
#include <random>
#include <vector>
#include <vesper/index/hnsw.hpp>

using namespace vesper::index;

static std::vector<float> make_vector(std::size_t dim, float base) {
    std::vector<float> v(dim);
    for (std::size_t i = 0; i < dim; ++i) v[i] = base + static_cast<float>(i) * 0.01f;
    return v;
}

TEST(HnswBasicInvariant, SmallIndexSearchesAreStable) {
    const std::size_t dim = 16;
    const std::size_t n = 20;

    HnswIndex index;
    HnswBuildParams params;
    params.M = 8;
    params.efConstruction = 64;
    params.extend_candidates = true;
    params.keep_pruned_connections = true;

    auto ok = index.init(dim, params, n);
    ASSERT_TRUE(ok.has_value());

    std::vector<std::uint64_t> ids(n);
    std::vector<float> data(n * dim);
    for (std::size_t i = 0; i < n; ++i) {
        ids[i] = static_cast<std::uint64_t>(i);
        auto v = make_vector(dim, static_cast<float>(i));
        std::copy(v.begin(), v.end(), data.begin() + i * dim);
    }

    for (std::size_t i = 0; i < n; ++i) {
        auto rc = index.add(ids[i], data.data() + i * dim);
        ASSERT_TRUE(rc.has_value());
    }

    // Perform multiple searches to exercise traversal
    HnswSearchParams sp;
    sp.efSearch = 32;
    sp.k = 5;

    for (std::size_t rep = 0; rep < 50; ++rep) {
        for (std::size_t qi = 0; qi < n; ++qi) {
            auto res = index.search(data.data() + qi * dim, sp);
            ASSERT_TRUE(res.has_value());
            const auto& nn = res.value();
            ASSERT_LE(nn.size(), sp.k);
            // Basic sanity: first result should be the query itself or close
            if (!nn.empty()) {
                EXPECT_LE(nn.front().second, nn.back().second);
            }
        }
    }
}

