#include <gtest/gtest.h>
#include "vesper/index/kmeans.hpp"
#include "vesper/index/kmeans_elkan.hpp"
#include <random>
#include <numeric>
#include <algorithm>
#include <unordered_set>
#include <chrono>

namespace vesper::index {
namespace {

class KmeansTest : public ::testing::Test {
protected:
    void SetUp() override {
        gen_.seed(42);
        
        // Generate clustered data
        GenerateClusteredData(10000, 128, 10);
        
        // Generate edge case datasets
        GenerateEdgeCases();
    }
    
    void GenerateClusteredData(std::size_t n, std::size_t dim, std::size_t k) {
        data_.resize(n * dim);
        true_labels_.resize(n);
        
        // Generate k cluster centers
        std::normal_distribution<float> center_dist(0.0f, 10.0f);
        std::vector<std::vector<float>> centers(k, std::vector<float>(dim));
        
        for (auto& center : centers) {
            for (auto& val : center) {
                val = center_dist(gen_);
            }
        }
        
        // Generate points around centers
        std::normal_distribution<float> noise_dist(0.0f, 1.0f);
        std::uniform_int_distribution<std::size_t> cluster_dist(0, k - 1);
        
        for (std::size_t i = 0; i < n; ++i) {
            std::size_t cluster = cluster_dist(gen_);
            true_labels_[i] = cluster;
            
            for (std::size_t d = 0; d < dim; ++d) {
                data_[i * dim + d] = centers[cluster][d] + noise_dist(gen_);
            }
        }
        
        n_ = n;
        dim_ = dim;
        k_ = k;
    }
    
    void GenerateEdgeCases() {
        // Single point
        single_point_ = {1.0f, 2.0f, 3.0f};
        
        // Identical points
        identical_points_.resize(300);  // 100 3D points
        std::fill(identical_points_.begin(), identical_points_.end(), 5.0f);
        
        // Linear data (all on a line)
        linear_data_.resize(100 * 2);  // 100 2D points
        for (std::size_t i = 0; i < 100; ++i) {
            linear_data_[i * 2] = static_cast<float>(i);
            linear_data_[i * 2 + 1] = static_cast<float>(i) * 2.0f;
        }
    }
    
    float ComputeInertia(const float* data, std::size_t n, std::size_t dim,
                        const std::vector<std::vector<float>>& centroids,
                        const std::vector<std::uint32_t>& assignments) {
        float inertia = 0.0f;
        
        for (std::size_t i = 0; i < n; ++i) {
            const float* point = data + i * dim;
            const auto& centroid = centroids[assignments[i]];
            
            for (std::size_t d = 0; d < dim; ++d) {
                float diff = point[d] - centroid[d];
                inertia += diff * diff;
            }
        }
        
        return inertia;
    }
    
    std::mt19937 gen_;
    std::vector<float> data_;
    std::vector<std::size_t> true_labels_;
    std::size_t n_, dim_, k_;
    
    std::vector<float> single_point_;
    std::vector<float> identical_points_;
    std::vector<float> linear_data_;
};

// Lloyd's K-means Tests

TEST_F(KmeansTest, Lloyd_BasicClustering) {
    KmeansParams params{
        .k = 10,
        .max_iter = 100,
        .epsilon = 1e-4f,
        .seed = 42,
        .balanced = false,
        .verbose = false,
        .n_redo = 1
    };
    
    auto result = kmeans_cluster(data_.data(), n_, dim_, params);
    
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->centroids.size(), params.k);
    EXPECT_EQ(result->assignments.size(), n_);
    EXPECT_EQ(result->cluster_sizes.size(), params.k);
    
    // Check that all points are assigned
    for (auto assignment : result->assignments) {
        EXPECT_LT(assignment, params.k);
    }
    
    // Check cluster sizes sum to n
    std::uint32_t total = 0;
    for (auto size : result->cluster_sizes) {
        total += size;
    }
    EXPECT_EQ(total, n_);
    
    // Inertia should be positive
    EXPECT_GT(result->inertia, 0.0f);
}

TEST_F(KmeansTest, Lloyd_Convergence) {
    KmeansParams params{
        .k = 5,
        .max_iter = 100,
        .epsilon = 1e-6f,
        .seed = 42
    };
    
    auto result = kmeans_cluster(data_.data(), 1000, dim_, params);
    
    ASSERT_TRUE(result.has_value());
    
    // Should converge before max iterations
    EXPECT_LT(result->iterations, params.max_iter);
    
    // Verify convergence by recomputing inertia
    float computed_inertia = ComputeInertia(data_.data(), 1000, dim_,
                                           result->centroids, result->assignments);
    EXPECT_NEAR(result->inertia, computed_inertia, 1e-3f);
}

TEST_F(KmeansTest, Lloyd_MultipleRuns) {
    KmeansParams params{
        .k = 10,
        .max_iter = 50,
        .epsilon = 1e-4f,
        .seed = 42,
        .balanced = false,
        .verbose = false,
        .n_redo = 5  // Multiple runs
    };
    
    auto result = kmeans_cluster(data_.data(), n_, dim_, params);
    
    ASSERT_TRUE(result.has_value());
    
    // Multiple runs should find good solution
    // (Lower inertia than single run on average)
    KmeansParams single_run_params = params;
    single_run_params.n_redo = 1;
    
    auto single_result = kmeans_cluster(data_.data(), n_, dim_, single_run_params);
    ASSERT_TRUE(single_result.has_value());
    
    // Multi-run should be at least as good
    EXPECT_LE(result->inertia, single_result->inertia * 1.1f);
}

// K-means++ Initialization Tests

TEST_F(KmeansTest, KmeansPlusPlus_Deterministic) {
    const std::uint32_t seed = 123;
    
    auto centers1 = kmeans_plusplus_init(data_.data(), n_, dim_, k_, seed);
    auto centers2 = kmeans_plusplus_init(data_.data(), n_, dim_, k_, seed);
    
    ASSERT_EQ(centers1.size(), k_);
    ASSERT_EQ(centers2.size(), k_);
    
    // Same seed should give same centers
    for (std::size_t i = 0; i < k_; ++i) {
        for (std::size_t d = 0; d < dim_; ++d) {
            EXPECT_FLOAT_EQ(centers1[i][d], centers2[i][d]);
        }
    }
}

TEST_F(KmeansTest, KmeansPlusPlus_Diversity) {
    auto centers = kmeans_plusplus_init(data_.data(), n_, dim_, k_, 42);
    
    ASSERT_EQ(centers.size(), k_);
    
    // Centers should be diverse (not too close)
    for (std::size_t i = 0; i < k_; ++i) {
        for (std::size_t j = i + 1; j < k_; ++j) {
            float dist = 0.0f;
            for (std::size_t d = 0; d < dim_; ++d) {
                float diff = centers[i][d] - centers[j][d];
                dist += diff * diff;
            }
            
            // Centers shouldn't be identical
            EXPECT_GT(dist, 1e-6f);
        }
    }
}

// Elkan's Algorithm Tests

TEST_F(KmeansTest, Elkan_Correctness) {
    KmeansElkan elkan;
    KmeansElkan::Config config{
        .k = 10,
        .max_iter = 50,
        .epsilon = 1e-4f,
        .seed = 42
    };
    
    auto result = elkan.cluster(data_.data(), n_, dim_, config);
    
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->centroids.size(), config.k);
    EXPECT_EQ(result->assignments.size(), n_);
    
    // Get statistics
    auto stats = elkan.get_stats();
    
    // Should skip many distance computations
    EXPECT_GT(stats.distance_skipped, 0);
    EXPECT_GT(stats.skip_rate, 0.3f);  // Typically 30-90% skipped
    
    // Should use fewer computations than naive
    const std::uint64_t naive_computations = n_ * config.k * result->iterations;
    EXPECT_LT(stats.distance_computations, naive_computations);
}

TEST_F(KmeansTest, Elkan_VsLloyd_Equivalence) {
    // Both algorithms should find similar solutions
    KmeansParams lloyd_params{
        .k = 5,
        .max_iter = 50,
        .epsilon = 1e-4f,
        .seed = 42
    };
    
    KmeansElkan::Config elkan_config{
        .k = 5,
        .max_iter = 50,
        .epsilon = 1e-4f,
        .seed = 42
    };
    
    auto lloyd_result = kmeans_cluster(data_.data(), 1000, dim_, lloyd_params);
    
    KmeansElkan elkan;
    auto elkan_result = elkan.cluster(data_.data(), 1000, dim_, elkan_config);
    
    ASSERT_TRUE(lloyd_result.has_value());
    ASSERT_TRUE(elkan_result.has_value());
    
    // Inertias should be similar (within 10%)
    EXPECT_NEAR(lloyd_result->inertia, elkan_result->inertia,
                lloyd_result->inertia * 0.1f);
}

// Edge Cases Tests

TEST_F(KmeansTest, EdgeCase_MoreClustersThanPoints) {
    KmeansParams params{
        .k = 10,
        .max_iter = 50,
        .seed = 42
    };
    
    // Only 3 points but k=10
    auto result = kmeans_cluster(single_point_.data(), 1, 3, params);
    
    // Should fail gracefully
    EXPECT_FALSE(result.has_value());
}

TEST_F(KmeansTest, EdgeCase_IdenticalPoints) {
    KmeansParams params{
        .k = 3,
        .max_iter = 50,
        .seed = 42
    };
    
    // All points are identical
    auto result = kmeans_cluster(identical_points_.data(), 100, 3, params);
    
    ASSERT_TRUE(result.has_value());
    
    // Inertia should be 0 (all points at centroids)
    EXPECT_NEAR(result->inertia, 0.0f, 1e-6f);
    
    // All centroids should be the same
    for (const auto& centroid : result->centroids) {
        EXPECT_NEAR(centroid[0], 5.0f, 1e-6f);
        EXPECT_NEAR(centroid[1], 5.0f, 1e-6f);
        EXPECT_NEAR(centroid[2], 5.0f, 1e-6f);
    }
}

TEST_F(KmeansTest, EdgeCase_LinearData) {
    KmeansParams params{
        .k = 5,
        .max_iter = 50,
        .seed = 42
    };
    
    // Points on a line
    auto result = kmeans_cluster(linear_data_.data(), 100, 2, params);
    
    ASSERT_TRUE(result.has_value());
    
    // Centroids should be roughly evenly spaced on the line
    std::vector<float> centroid_positions;
    for (const auto& centroid : result->centroids) {
        centroid_positions.push_back(centroid[0]);
    }
    
    std::sort(centroid_positions.begin(), centroid_positions.end());
    
    // Check spacing
    for (std::size_t i = 1; i < centroid_positions.size(); ++i) {
        float spacing = centroid_positions[i] - centroid_positions[i-1];
        EXPECT_GT(spacing, 5.0f);  // Reasonable spacing
    }
}

TEST_F(KmeansTest, EdgeCase_ZeroK) {
    KmeansParams params{
        .k = 0,
        .max_iter = 50,
        .seed = 42
    };
    
    auto result = kmeans_cluster(data_.data(), n_, dim_, params);
    
    // Should fail gracefully
    EXPECT_FALSE(result.has_value());
}

// Parallel K-means|| Tests

TEST_F(KmeansTest, ParallelInit_Correctness) {
    auto centers = kmeans_parallel_init(data_.data(), n_, dim_, k_, 5, 42);
    
    EXPECT_EQ(centers.size(), k_);
    
    // Each center should be valid
    for (const auto& center : centers) {
        EXPECT_EQ(center.size(), dim_);
        
        // Should be finite
        for (float val : center) {
            EXPECT_TRUE(std::isfinite(val));
        }
    }
}

TEST_F(KmeansTest, ParallelInit_Oversampling) {
    // With oversampling, should handle more initial centers
    const std::uint32_t k_target = 10;
    auto centers = kmeans_parallel_init(data_.data(), n_, dim_, k_target, 5, 42);
    
    // Should reduce to exactly k centers
    EXPECT_EQ(centers.size(), k_target);
}

// Performance Characteristics Tests

TEST_F(KmeansTest, Performance_ScalabilityWithK) {
    // Test that time scales reasonably with k
    std::vector<std::uint32_t> k_values{5, 10, 20, 50};
    std::vector<float> times;
    
    for (std::uint32_t k : k_values) {
        KmeansParams params{
            .k = k,
            .max_iter = 20,
            .epsilon = 1e-3f,
            .seed = 42
        };
        
        auto start = std::chrono::steady_clock::now();
        auto result = kmeans_cluster(data_.data(), 1000, dim_, params);
        auto end = std::chrono::steady_clock::now();
        
        ASSERT_TRUE(result.has_value());
        
        auto duration = std::chrono::duration<float>(end - start).count();
        times.push_back(duration);
    }
    
    // Time should increase with k, but not explosively
    for (std::size_t i = 1; i < times.size(); ++i) {
        // Each doubling of k should take < 3x time
        if (k_values[i] == k_values[i-1] * 2) {
            EXPECT_LT(times[i], times[i-1] * 3.0f);
        }
    }
}

// Cluster Quality Tests

TEST_F(KmeansTest, ClusterQuality_Metrics) {
    KmeansParams params{
        .k = static_cast<std::uint32_t>(k_),
        .max_iter = 100,
        .epsilon = 1e-6f,
        .seed = 42
    };
    
    auto result = kmeans_cluster(data_.data(), n_, dim_, params);
    ASSERT_TRUE(result.has_value());
    
    // Compute silhouette coefficient (simplified)
    float avg_silhouette = 0.0f;
    std::size_t sample_size = std::min(std::size_t(100), n_);
    
    for (std::size_t i = 0; i < sample_size; ++i) {
        const float* point = data_.data() + i * dim_;
        std::uint32_t cluster = result->assignments[i];
        
        // Compute a(i) - avg distance to same cluster
        float a_i = 0.0f;
        std::size_t same_cluster_count = 0;
        
        for (std::size_t j = 0; j < n_; ++j) {
            if (i != j && result->assignments[j] == cluster) {
                const float* other = data_.data() + j * dim_;
                float dist = 0.0f;
                for (std::size_t d = 0; d < dim_; ++d) {
                    float diff = point[d] - other[d];
                    dist += diff * diff;
                }
                a_i += std::sqrt(dist);
                same_cluster_count++;
            }
        }
        
        if (same_cluster_count > 0) {
            a_i /= same_cluster_count;
        }
        
        // Compute b(i) - min avg distance to other clusters
        float b_i = std::numeric_limits<float>::max();
        
        for (std::uint32_t c = 0; c < k_; ++c) {
            if (c == cluster) continue;
            
            float avg_dist = 0.0f;
            std::size_t other_cluster_count = 0;
            
            for (std::size_t j = 0; j < n_; ++j) {
                if (result->assignments[j] == c) {
                    const float* other = data_.data() + j * dim_;
                    float dist = 0.0f;
                    for (std::size_t d = 0; d < dim_; ++d) {
                        float diff = point[d] - other[d];
                        dist += diff * diff;
                    }
                    avg_dist += std::sqrt(dist);
                    other_cluster_count++;
                }
            }
            
            if (other_cluster_count > 0) {
                avg_dist /= other_cluster_count;
                b_i = std::min(b_i, avg_dist);
            }
        }
        
        // Silhouette coefficient for point i
        float s_i = (b_i - a_i) / std::max(a_i, b_i);
        avg_silhouette += s_i;
    }
    
    avg_silhouette /= sample_size;
    
    // Good clustering should have positive silhouette
    EXPECT_GT(avg_silhouette, 0.0f);
}

} // namespace
} // namespace vesper::index