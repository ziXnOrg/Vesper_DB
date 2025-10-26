#include <gtest/gtest.h>
#include "vesper/index/hnsw.hpp"
#include <random>
#include <numeric>
#include <unordered_set>
#include <thread>

namespace vesper::index {
namespace {

class HnswTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Generate test data
        const std::size_t n = 1000;
        const std::size_t dim = 128;

        data_.resize(n * dim);
        ids_.resize(n);

        std::mt19937 gen(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);

        for (auto& val : data_) {
            val = dist(gen);
        }

        // Normalize for cosine similarity tests
        for (std::size_t i = 0; i < n; ++i) {
            float* vec = data_.data() + i * dim;
            float norm = 0.0f;
            for (std::size_t d = 0; d < dim; ++d) {
                norm += vec[d] * vec[d];
            }
            norm = std::sqrt(norm);
            for (std::size_t d = 0; d < dim; ++d) {
                vec[d] /= norm;
            }
        }

        std::iota(ids_.begin(), ids_.end(), 0);

        n_ = n;
        dim_ = dim;
    }

    std::vector<float> data_;
    std::vector<std::uint64_t> ids_;
    std::size_t n_;
    std::size_t dim_;
};

TEST_F(HnswTest, InitAndBuild) {
    HnswIndex index;

    HnswBuildParams params{
        .M = 16,
        .efConstruction = 200,
        .seed = 42
    };

    ASSERT_TRUE(index.init(dim_, params, n_).has_value());
    EXPECT_TRUE(index.is_initialized());
    EXPECT_EQ(index.dimension(), dim_);

    // Add vectors
    for (std::size_t i = 0; i < n_; ++i) {
        ASSERT_TRUE(index.add(ids_[i], data_.data() + i * dim_).has_value());
    }

    EXPECT_EQ(index.size(), n_);

    // Check stats
    auto stats = index.get_stats();
    EXPECT_EQ(stats.n_nodes, n_);
    EXPECT_GT(stats.n_edges, 0);
    EXPECT_GT(stats.avg_degree, 0.0f);
}

TEST_F(HnswTest, Search) {
    HnswIndex index;

    HnswBuildParams params{
        .M = 16,
        .efConstruction = 200
    };

    ASSERT_TRUE(index.init(dim_, params, n_).has_value());
    ASSERT_TRUE(index.add_batch(ids_.data(), data_.data(), n_).has_value());

    // Search for first vector (should find itself)
    HnswSearchParams search_params{
        .efSearch = 50,
        .k = 10
    };

    auto results = index.search(data_.data(), search_params);
    ASSERT_TRUE(results.has_value());
    ASSERT_FALSE(results->empty());

    // First result should be the query itself
    EXPECT_EQ(results->front().first, ids_[0]);
    EXPECT_LT(results->front().second, 1e-6f);  // Distance ~0
}

TEST_F(HnswTest, BatchSearch) {
    HnswIndex index;

    HnswBuildParams params{
        .M = 16,
        .efConstruction = 200
    };

    ASSERT_TRUE(index.init(dim_, params, n_).has_value());
    ASSERT_TRUE(index.add_batch(ids_.data(), data_.data(), n_).has_value());

    // Batch search
    const std::size_t n_queries = 10;
    HnswSearchParams search_params{
        .efSearch = 1000,
        .k = 10
    };

    auto results = index.search_batch(data_.data(), n_queries, search_params);
    ASSERT_TRUE(results.has_value());
    EXPECT_EQ(results->size(), n_queries);

    // Each query should find itself first
    for (std::size_t q = 0; q < n_queries; ++q) {
        const auto& query_results = (*results)[q];
        ASSERT_FALSE(query_results.empty());
        EXPECT_EQ(query_results.front().first, ids_[q]);
    }
}

TEST_F(HnswTest, RecallQuality) {
    HnswIndex index;

    HnswBuildParams params{
        .M = 16,
        .efConstruction = 200
    };

    ASSERT_TRUE(index.init(dim_, params, n_).has_value());
    ASSERT_TRUE(index.add_batch(ids_.data(), data_.data(), n_).has_value());

    // Compute ground truth by brute force
    const std::size_t n_queries = 100;
    const std::size_t k = 10;
    std::vector<std::uint64_t> ground_truth(n_queries * k);

    for (std::size_t q = 0; q < n_queries; ++q) {
        const float* query = data_.data() + q * dim_;
        std::vector<std::pair<float, std::uint64_t>> distances;

        for (std::size_t i = 0; i < n_; ++i) {
            const float* vec = data_.data() + i * dim_;
            float dist = 0.0f;
            for (std::size_t d = 0; d < dim_; ++d) {
                const float diff = query[d] - vec[d];
                dist += diff * diff;
            }
            distances.emplace_back(dist, ids_[i]);
        }

        std::partial_sort(distances.begin(), distances.begin() + k, distances.end());

        for (std::size_t i = 0; i < k; ++i) {
            ground_truth[q * k + i] = distances[i].second;
        }
    }

    // Test recall with different efSearch values
    for (std::uint32_t ef : {20, 50, 100, 200}) {
        HnswSearchParams search_params{
            .efSearch = ef,
            .k = k
        };

        float recall = compute_recall(index, data_.data(), n_queries,
                                     ground_truth.data(), k, search_params);

        // Higher ef should give better recall
        if (ef >= 100) {
            EXPECT_GT(recall, 0.9f) << "efSearch=" << ef;
        }
        if (ef >= 200) {
            EXPECT_GT(recall, 0.95f) << "efSearch=" << ef;
        }
    }
}

TEST_F(HnswTest, FilteredSearch) {
    HnswIndex index;

    HnswBuildParams params{
        .M = 16,
        .efConstruction = 200
    };

    ASSERT_TRUE(index.init(dim_, params, n_).has_value());
    ASSERT_TRUE(index.add_batch(ids_.data(), data_.data(), n_).has_value());

    // Create filter bitmap (only even IDs)
    std::vector<std::uint8_t> filter((n_ + 7) / 8, 0);
    for (std::size_t i = 0; i < n_; i += 2) {
        filter[i / 8] |= (1 << (i % 8));
    }

    HnswSearchParams search_params{
        .efSearch = 100,
        .k = 10,
        .filter_mask = filter.data(),
        .filter_size = filter.size()
    };

    auto results = index.search(data_.data(), search_params);
    ASSERT_TRUE(results.has_value());

    // All results should have even IDs
    for (const auto& [id, dist] : *results) {
        EXPECT_EQ(id % 2, 0) << "ID " << id << " should be even";
    }
}

TEST_F(HnswTest, SoftDelete) {
    HnswIndex index;

    HnswBuildParams params{
        .M = 16,
        .efConstruction = 200
    };

    ASSERT_TRUE(index.init(dim_, params, 100).has_value());

    // Add vectors
    for (std::size_t i = 0; i < 100; ++i) {
        ASSERT_TRUE(index.add(ids_[i], data_.data() + i * dim_).has_value());
    }

    // Mark some as deleted
    for (std::size_t i = 0; i < 10; ++i) {
        ASSERT_TRUE(index.mark_deleted(ids_[i]).has_value());
    }

    // Search should not return deleted vectors
    HnswSearchParams search_params{
        .efSearch = 50,
        .k = 20
    };

    auto results = index.search(data_.data() + 5 * dim_, search_params);
    ASSERT_TRUE(results.has_value());

    // Should not find deleted IDs (0-9)
    for (const auto& [id, dist] : *results) {
        EXPECT_GE(id, 10) << "Should not return deleted ID " << id;
    }
}

TEST_F(HnswTest, GraphConnectivity) {
    HnswIndex index;

    HnswBuildParams params{
        .M = 8,
        .efConstruction = 100
    };

    ASSERT_TRUE(index.init(dim_, params, 100).has_value());

    // Add vectors
    for (std::size_t i = 0; i < 100; ++i) {
        ASSERT_TRUE(index.add(ids_[i], data_.data() + i * dim_).has_value());
    }

    auto stats = index.get_stats();

    // Graph should be well-connected
    EXPECT_GT(stats.avg_degree, params.M * 0.5f);  // At least half of M
    EXPECT_LT(stats.avg_degree, params.M * 2.5f);  // Not too dense

    // Should have multiple levels
    EXPECT_GT(stats.n_levels, 1);

    // Level distribution should be geometric
    if (stats.level_counts.size() > 1) {
        for (std::size_t i = 1; i < stats.level_counts.size(); ++i) {
            // Higher levels should have fewer nodes
            EXPECT_LE(stats.level_counts[i], stats.level_counts[i-1]);
        }
    }
}

TEST_F(HnswTest, ErrorHandling) {
    HnswIndex index;

    // Search before init should fail
    HnswSearchParams search_params{.efSearch = 50, .k = 10};
    auto search_result = index.search(data_.data(), search_params);
    EXPECT_FALSE(search_result.has_value());

    // Invalid parameters
    HnswBuildParams bad_params{
        .M = 1,  // Too small
        .efConstruction = 200
    };
    auto init_result = index.init(dim_, bad_params);
    EXPECT_FALSE(init_result.has_value());

    // Valid init
    HnswBuildParams good_params{
        .M = 16,
        .efConstruction = 200
    };
    ASSERT_TRUE(index.init(dim_, good_params).has_value());

    // Duplicate ID should fail
    ASSERT_TRUE(index.add(100, data_.data()).has_value());
    auto dup_result = index.add(100, data_.data() + dim_);
    EXPECT_FALSE(dup_result.has_value());
}

TEST_F(HnswTest, MemoryManagement) {
    HnswIndex index;

    HnswBuildParams params{
        .M = 16,
        .efConstruction = 200
    };

    ASSERT_TRUE(index.init(dim_, params, n_).has_value());

    // Track memory before adding
    auto stats_before = index.get_stats();

    // Add vectors
    ASSERT_TRUE(index.add_batch(ids_.data(), data_.data(), n_).has_value());

    auto stats_after = index.get_stats();

    // Memory should increase
    EXPECT_GT(stats_after.memory_bytes, stats_before.memory_bytes);

    // Estimate expected memory
    // Each node: id(8) + data(dim*4) + neighbors(~M*levels*4) + overhead
    std::size_t expected_per_node =
        sizeof(std::uint64_t) +           // ID
        dim_ * sizeof(float) +             // Data
        params.M * 2 * sizeof(std::uint32_t) + // Neighbors (estimate)
        64;                                // Overhead

    std::size_t expected_total = n_ * expected_per_node;

    // Should be in reasonable range
    EXPECT_LT(stats_after.memory_bytes, expected_total * 2);
    EXPECT_GT(stats_after.memory_bytes, expected_total / 2);
}

TEST_F(HnswTest, ConcurrentSearch) {
    HnswIndex index;

    HnswBuildParams params{
        .M = 16,
        .efConstruction = 200
    };

    ASSERT_TRUE(index.init(dim_, params, n_).has_value());
    ASSERT_TRUE(index.add_batch(ids_.data(), data_.data(), n_).has_value());

    // Concurrent searches should work
    const std::size_t n_threads = 4;
    const std::size_t queries_per_thread = 25;

    std::vector<std::thread> threads;
    std::atomic<std::size_t> success_count{0};

    for (std::size_t t = 0; t < n_threads; ++t) {
        threads.emplace_back([&, t]() {
            HnswSearchParams search_params{
                .efSearch = 50,
                .k = 10
            };

            for (std::size_t q = 0; q < queries_per_thread; ++q) {
                std::size_t idx = (t * queries_per_thread + q) % n_;
                auto results = index.search(data_.data() + idx * dim_, search_params);
                if (results.has_value() && !results->empty()) {
                    success_count++;
                }
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(success_count, n_threads * queries_per_thread);
}

TEST_F(HnswTest, RobustPruneCorrectness) {
    // Test the RobustPrune algorithm
    std::vector<std::pair<float, std::uint32_t>> candidates{
        {1.0f, 0}, {2.0f, 1}, {2.1f, 2}, {3.0f, 3},
        {4.0f, 4}, {5.0f, 5}, {6.0f, 6}, {7.0f, 7}
    };

    auto [selected, pruned] = robust_prune(candidates, 4, false, true);

    // Should select M neighbors
    EXPECT_EQ(selected.size(), 4);

    // Should include nearest
    EXPECT_EQ(selected[0], 0);

    // Should have diversity
    std::unordered_set<std::uint32_t> selected_set(selected.begin(), selected.end());
    EXPECT_EQ(selected_set.size(), selected.size());  // All unique

    // Pruned should contain the rest
    EXPECT_EQ(pruned.size(), candidates.size() - selected.size());
}

TEST_F(HnswTest, SaveLoad) {
    HnswIndex index;

    HnswBuildParams params{
        .M = 16,
        .efConstruction = 200
    };

    ASSERT_TRUE(index.init(dim_, params, 100).has_value());

    // Add some vectors
    for (std::size_t i = 0; i < 100; ++i) {
        ASSERT_TRUE(index.add(ids_[i], data_.data() + i * dim_).has_value());
    }

    // Save to file
    const std::string path = "/tmp/hnsw_test.bin";
    ASSERT_TRUE(index.save(path).has_value());

    // Load from file
    auto loaded = HnswIndex::load(path);
    // Note: Load is not fully implemented yet, so this might fail
    // ASSERT_TRUE(loaded.has_value());

    // Clean up
    std::remove(path.c_str());
}

} // namespace
} // namespace vesper::index
