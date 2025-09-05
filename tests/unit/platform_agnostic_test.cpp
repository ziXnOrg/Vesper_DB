// Platform-agnostic test that works on both x86 and ARM
#include <gtest/gtest.h>
#include "vesper/kernels/dispatch.hpp"
#include "vesper/index/kmeans.hpp"
#include "vesper/index/kmeans_elkan.hpp"

#include <random>
#include <numeric>
#include <cmath>

namespace vesper::test {

class KernelTest : public ::testing::Test {
protected:
    void SetUp() override {
        gen_.seed(42);
        
        // Generate test vectors
        const std::size_t dim = 128;
        a_.resize(dim);
        b_.resize(dim);
        
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (std::size_t i = 0; i < dim; ++i) {
            a_[i] = dist(gen_);
            b_[i] = dist(gen_);
        }
    }
    
    std::mt19937 gen_;
    std::vector<float> a_, b_;
};

TEST_F(KernelTest, L2Distance) {
    const auto& ops = kernels::select_backend_auto();
    
    float result = ops.l2_sq(a_, b_);
    
    EXPECT_GT(result, 0.0f);
    EXPECT_TRUE(std::isfinite(result));
    
    // Self-distance should be 0
    float self_dist = ops.l2_sq(a_, a_);
    EXPECT_NEAR(self_dist, 0.0f, 1e-6f);
}

TEST_F(KernelTest, InnerProduct) {
    const auto& ops = kernels::select_backend_auto();
    
    float result = ops.inner_product(a_, b_);
    EXPECT_TRUE(std::isfinite(result));
    
    // Commutative property
    float ba = ops.inner_product(b_, a_);
    EXPECT_FLOAT_EQ(result, ba);
}

TEST_F(KernelTest, CosineSimilarity) {
    const auto& ops = kernels::select_backend_auto();
    
    // Normalize vectors
    float norm_a = 0.0f, norm_b = 0.0f;
    for (std::size_t i = 0; i < a_.size(); ++i) {
        norm_a += a_[i] * a_[i];
        norm_b += b_[i] * b_[i];
    }
    norm_a = std::sqrt(norm_a);
    norm_b = std::sqrt(norm_b);
    
    for (auto& val : a_) val /= norm_a;
    for (auto& val : b_) val /= norm_b;
    
    float result = ops.cosine_similarity(a_, b_);
    
    EXPECT_GE(result, -1.0f);
    EXPECT_LE(result, 1.0f);
    
    // Self-similarity should be 1
    float self_sim = ops.cosine_similarity(a_, a_);
    EXPECT_NEAR(self_sim, 1.0f, 1e-5f);
}

class KmeansTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Generate clustered test data
        const std::size_t n_per_cluster = 100;
        const std::size_t k = 3;
        const std::size_t dim = 16;
        
        data_.resize(n_per_cluster * k * dim);
        
        std::mt19937 gen(42);
        std::normal_distribution<float> noise(0.0f, 0.1f);
        
        // Create 3 well-separated clusters
        std::vector<std::vector<float>> centers = {
            std::vector<float>(dim, 0.0f),
            std::vector<float>(dim, 5.0f),
            std::vector<float>(dim, -5.0f)
        };
        
        for (std::size_t c = 0; c < k; ++c) {
            for (std::size_t i = 0; i < n_per_cluster; ++i) {
                for (std::size_t d = 0; d < dim; ++d) {
                    data_[(c * n_per_cluster + i) * dim + d] = 
                        centers[c][d] + noise(gen);
                }
            }
        }
        
        n_ = n_per_cluster * k;
        dim_ = dim;
        k_ = k;
    }
    
    std::vector<float> data_;
    std::size_t n_;
    std::size_t dim_;
    std::uint32_t k_;
};

TEST_F(KmeansTest, LloydsAlgorithm) {
    index::KmeansParams params{
        .k = k_,
        .max_iter = 50,
        .epsilon = 1e-5f,
        .seed = 42
    };
    
    auto result = index::kmeans_cluster(data_.data(), n_, dim_, params);
    
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->centroids.size(), k_);
    EXPECT_EQ(result->assignments.size(), n_);
    EXPECT_GT(result->iterations, 0);
    EXPECT_LT(result->iterations, params.max_iter);
    
    // Check that all clusters have some points
    std::vector<int> cluster_counts(k_, 0);
    for (auto assignment : result->assignments) {
        EXPECT_LT(assignment, k_);
        cluster_counts[assignment]++;
    }
    
    for (auto count : cluster_counts) {
        EXPECT_GT(count, 0);
    }
}

TEST_F(KmeansTest, ElkansAlgorithm) {
    index::KmeansElkan elkan;
    index::KmeansElkan::Config config{
        .k = k_,
        .max_iter = 50,
        .epsilon = 1e-5f,
        .seed = 42
    };
    
    auto result = elkan.cluster(data_.data(), n_, dim_, config);
    
    ASSERT_TRUE(result.has_value());
    
    auto stats = elkan.get_stats();
    EXPECT_GT(stats.iterations, 0);
    EXPECT_LT(stats.iterations, config.max_iter);
    EXPECT_GT(stats.distance_skipped, 0);  // Should skip some computations
    EXPECT_GT(stats.skip_rate, 0.0f);
}

TEST_F(KmeansTest, ConvergenceOnSimpleData) {
    // Create perfectly separable data
    const std::size_t dim = 2;
    std::vector<float> simple_data = {
        0.0f, 0.0f,  0.1f, 0.1f,  // Cluster 1
        5.0f, 5.0f,  5.1f, 5.1f,  // Cluster 2
    };
    
    index::KmeansParams params{
        .k = 2,
        .max_iter = 100,
        .epsilon = 1e-6f,
        .seed = 42
    };
    
    auto result = index::kmeans_cluster(simple_data.data(), 4, dim, params);
    
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->centroids.size(), 2);
    
    // Should converge to the obvious clusters
    EXPECT_EQ(result->assignments[0], result->assignments[1]);
    EXPECT_EQ(result->assignments[2], result->assignments[3]);
    EXPECT_NE(result->assignments[0], result->assignments[2]);
}

} // namespace vesper::test