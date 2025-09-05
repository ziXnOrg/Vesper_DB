#include <gtest/gtest.h>
#include "vesper/index/ivf_pq.hpp"
#include <random>
#include <numeric>

namespace vesper::index {
namespace {

class IvfPqTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Generate synthetic test data
        const std::size_t n = 10000;
        const std::size_t dim = 128;
        
        data_.resize(n * dim);
        ids_.resize(n);
        
        std::mt19937 gen(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        for (auto& val : data_) {
            val = dist(gen);
        }
        
        std::iota(ids_.begin(), ids_.end(), 0);
    }
    
    std::vector<float> data_;
    std::vector<std::uint64_t> ids_;
};

TEST_F(IvfPqTest, TrainAndAdd) {
    IvfPqIndex index;
    
    IvfPqTrainParams params{
        .nlist = 100,
        .m = 8,
        .nbits = 8,
        .max_iter = 25,
        .epsilon = 1e-4f,
        .verbose = false,
        .use_opq = false,
        .seed = 42
    };
    
    // Train on subset
    const std::size_t n_train = 5000;
    const std::size_t dim = 128;
    
    auto train_result = index.train(data_.data(), dim, n_train, params);
    ASSERT_TRUE(train_result.has_value());
    EXPECT_TRUE(index.is_trained());
    EXPECT_EQ(index.dimension(), dim);
    
    // Add vectors
    const std::size_t n_add = 1000;
    auto add_result = index.add(ids_.data(), data_.data(), n_add);
    ASSERT_TRUE(add_result.has_value());
    
    auto stats = index.get_stats();
    EXPECT_EQ(stats.n_vectors, n_add);
    EXPECT_EQ(stats.n_lists, params.nlist);
    EXPECT_EQ(stats.m, params.m);
}

TEST_F(IvfPqTest, Search) {
    IvfPqIndex index;
    
    IvfPqTrainParams train_params{
        .nlist = 100,
        .m = 8,
        .nbits = 8
    };
    
    const std::size_t dim = 128;
    const std::size_t n_train = 5000;
    const std::size_t n_add = 1000;
    
    // Train and add
    ASSERT_TRUE(index.train(data_.data(), dim, n_train, train_params).has_value());
    ASSERT_TRUE(index.add(ids_.data(), data_.data(), n_add).has_value());
    
    // Search for first vector (should find itself)
    IvfPqSearchParams search_params{
        .nprobe = 8,
        .k = 10
    };
    
    auto results = index.search(data_.data(), search_params);
    ASSERT_TRUE(results.has_value());
    ASSERT_FALSE(results->empty());
    
    // First result should be the query itself with distance ~0
    EXPECT_EQ(results->front().first, ids_[0]);
    EXPECT_LT(results->front().second, 0.1f);
}

TEST_F(IvfPqTest, BatchSearch) {
    IvfPqIndex index;
    
    IvfPqTrainParams train_params{
        .nlist = 100,
        .m = 8,
        .nbits = 8
    };
    
    const std::size_t dim = 128;
    const std::size_t n_train = 5000;
    const std::size_t n_add = 1000;
    const std::size_t n_queries = 10;
    
    // Train and add
    ASSERT_TRUE(index.train(data_.data(), dim, n_train, train_params).has_value());
    ASSERT_TRUE(index.add(ids_.data(), data_.data(), n_add).has_value());
    
    // Batch search
    IvfPqSearchParams search_params{
        .nprobe = 8,
        .k = 10
    };
    
    auto results = index.search_batch(data_.data(), n_queries, search_params);
    ASSERT_TRUE(results.has_value());
    EXPECT_EQ(results->size(), n_queries);
    
    // Each query should find results
    for (const auto& query_results : *results) {
        EXPECT_FALSE(query_results.empty());
        EXPECT_LE(query_results.size(), search_params.k);
    }
}

TEST_F(IvfPqTest, ClearAndReset) {
    IvfPqIndex index;
    
    IvfPqTrainParams params{
        .nlist = 100,
        .m = 8,
        .nbits = 8
    };
    
    const std::size_t dim = 128;
    const std::size_t n_train = 5000;
    const std::size_t n_add = 1000;
    
    // Train and add
    ASSERT_TRUE(index.train(data_.data(), dim, n_train, params).has_value());
    ASSERT_TRUE(index.add(ids_.data(), data_.data(), n_add).has_value());
    
    // Clear should remove vectors but keep training
    index.clear();
    EXPECT_TRUE(index.is_trained());
    EXPECT_EQ(index.get_stats().n_vectors, 0);
    
    // Reset should clear everything
    index.reset();
    EXPECT_FALSE(index.is_trained());
    EXPECT_EQ(index.dimension(), 0);
}

TEST_F(IvfPqTest, RecallValidation) {
    IvfPqIndex index;
    
    IvfPqTrainParams train_params{
        .nlist = 100,
        .m = 8,
        .nbits = 8
    };
    
    const std::size_t dim = 128;
    const std::size_t n_train = 5000;
    const std::size_t n_add = 1000;
    const std::size_t n_queries = 100;
    const std::size_t k = 10;
    
    // Train and add
    ASSERT_TRUE(index.train(data_.data(), dim, n_train, train_params).has_value());
    ASSERT_TRUE(index.add(ids_.data(), data_.data(), n_add).has_value());
    
    // Create ground truth (brute force search)
    std::vector<std::uint64_t> ground_truth(n_queries * k);
    
    for (std::size_t q = 0; q < n_queries; ++q) {
        const float* query = data_.data() + q * dim;
        std::vector<std::pair<float, std::uint64_t>> distances;
        
        for (std::size_t i = 0; i < n_add; ++i) {
            const float* vec = data_.data() + i * dim;
            float dist = 0.0f;
            for (std::size_t d = 0; d < dim; ++d) {
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
    
    // Test recall with different nprobe values
    for (std::uint32_t nprobe : {1, 4, 8, 16}) {
        IvfPqSearchParams search_params{
            .nprobe = nprobe,
            .k = k
        };
        
        const float recall = compute_recall(index, data_.data(), n_queries,
                                           ground_truth.data(), k, search_params);
        
        // Higher nprobe should give better recall
        if (nprobe >= 8) {
            EXPECT_GT(recall, 0.7f) << "nprobe=" << nprobe;
        }
    }
}

TEST_F(IvfPqTest, ErrorHandling) {
    IvfPqIndex index;
    
    // Search before training should fail
    IvfPqSearchParams search_params{.nprobe = 8, .k = 10};
    auto search_result = index.search(data_.data(), search_params);
    EXPECT_FALSE(search_result.has_value());
    
    // Add before training should fail
    auto add_result = index.add(ids_.data(), data_.data(), 100);
    EXPECT_FALSE(add_result.has_value());
    
    // Train with invalid parameters
    IvfPqTrainParams bad_params{
        .nlist = 1000,  // More than training samples
        .m = 8,
        .nbits = 8
    };
    
    auto train_result = index.train(data_.data(), 128, 100, bad_params);
    EXPECT_FALSE(train_result.has_value());
}

TEST_F(IvfPqTest, WithOPQ) {
    IvfPqIndex index;
    
    IvfPqTrainParams params{
        .nlist = 100,
        .m = 8,
        .nbits = 8,
        .use_opq = true,
        .opq_iter = 10
    };
    
    const std::size_t dim = 128;
    const std::size_t n_train = 5000;
    const std::size_t n_add = 1000;
    
    // Train with OPQ
    auto train_result = index.train(data_.data(), dim, n_train, params);
    ASSERT_TRUE(train_result.has_value());
    
    // Add and search should work with OPQ
    ASSERT_TRUE(index.add(ids_.data(), data_.data(), n_add).has_value());
    
    IvfPqSearchParams search_params{.nprobe = 8, .k = 10};
    auto results = index.search(data_.data(), search_params);
    ASSERT_TRUE(results.has_value());
    EXPECT_FALSE(results->empty());
}

} // namespace
} // namespace vesper::index