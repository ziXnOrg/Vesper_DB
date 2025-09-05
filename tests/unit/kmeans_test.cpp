#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <vector>
#include <random>
#include <numeric>

#include "vesper/index/kmeans.hpp"

using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

namespace {

/** \brief Generate synthetic clustered data for testing. */
auto generate_clustered_data(std::size_t n_clusters, std::size_t points_per_cluster,
                             std::size_t dim, std::uint32_t seed) 
    -> std::vector<float> {
    std::mt19937 gen(seed);
    std::normal_distribution<float> center_dist(0.0f, 10.0f);
    std::normal_distribution<float> noise_dist(0.0f, 0.5f);
    
    const std::size_t n = n_clusters * points_per_cluster;
    std::vector<float> data(n * dim);
    
    // Generate cluster centers
    std::vector<std::vector<float>> centers(n_clusters);
    for (auto& center : centers) {
        center.resize(dim);
        for (std::size_t d = 0; d < dim; ++d) {
            center[d] = center_dist(gen);
        }
    }
    
    // Generate points around centers
    for (std::size_t c = 0; c < n_clusters; ++c) {
        for (std::size_t p = 0; p < points_per_cluster; ++p) {
            const std::size_t idx = c * points_per_cluster + p;
            for (std::size_t d = 0; d < dim; ++d) {
                data[idx * dim + d] = centers[c][d] + noise_dist(gen);
            }
        }
    }
    
    return data;
}

/** \brief Compute purity of clustering against ground truth. */
auto compute_purity(const std::vector<std::uint32_t>& assignments,
                   const std::vector<std::uint32_t>& ground_truth,
                   std::uint32_t k) -> float {
    std::size_t n = assignments.size();
    std::size_t correct = 0;
    
    // For each cluster, count most frequent ground truth label
    for (std::uint32_t c = 0; c < k; ++c) {
        std::vector<std::uint32_t> label_counts(k, 0);
        std::size_t cluster_size = 0;
        
        for (std::size_t i = 0; i < n; ++i) {
            if (assignments[i] == c) {
                label_counts[ground_truth[i]]++;
                cluster_size++;
            }
        }
        
        if (cluster_size > 0) {
            correct += *std::max_element(label_counts.begin(), label_counts.end());
        }
    }
    
    return static_cast<float>(correct) / static_cast<float>(n);
}

} // anonymous namespace

TEST_CASE("K-means clustering", "[kmeans]") {
    
    SECTION("K-means++ initialization produces k centroids") {
        const std::size_t n = 100;
        const std::size_t dim = 8;
        const std::uint32_t k = 5;
        
        std::vector<float> data = generate_clustered_data(k, n/k, dim, 42);
        
        auto centroids = vesper::index::kmeans_plusplus_init(
            data.data(), n, dim, k, 42);
        
        REQUIRE(centroids.size() == k);
        for (const auto& centroid : centroids) {
            REQUIRE(centroid.size() == dim);
        }
        
        // Verify centroids are distinct
        for (std::size_t i = 0; i < k; ++i) {
            for (std::size_t j = i + 1; j < k; ++j) {
                float dist = 0.0f;
                for (std::size_t d = 0; d < dim; ++d) {
                    float diff = centroids[i][d] - centroids[j][d];
                    dist += diff * diff;
                }
                REQUIRE(dist > 0.0f);
            }
        }
    }
    
    SECTION("K-means assigns all points to clusters") {
        const std::size_t n = 100;
        const std::size_t dim = 8;
        const std::uint32_t k = 5;
        
        std::vector<float> data = generate_clustered_data(k, n/k, dim, 42);
        
        auto centroids = vesper::index::kmeans_plusplus_init(
            data.data(), n, dim, k, 42);
        
        std::vector<std::uint32_t> assignments(n);
        float inertia = vesper::index::kmeans_assign(
            data.data(), n, centroids, assignments);
        
        REQUIRE(inertia > 0.0f);
        REQUIRE(inertia < std::numeric_limits<float>::max());
        
        // Verify all assignments are valid
        for (std::uint32_t a : assignments) {
            REQUIRE(a < k);
        }
        
        // Verify each cluster has at least one point (with high probability)
        std::vector<bool> cluster_used(k, false);
        for (std::uint32_t a : assignments) {
            cluster_used[a] = true;
        }
        std::size_t used_count = std::count(cluster_used.begin(), cluster_used.end(), true);
        REQUIRE(used_count >= k - 1); // Allow one empty cluster
    }
    
    SECTION("K-means converges on well-separated clusters") {
        const std::size_t n_clusters = 4;
        const std::size_t points_per_cluster = 25;
        const std::size_t n = n_clusters * points_per_cluster;
        const std::size_t dim = 8;
        
        std::vector<float> data = generate_clustered_data(
            n_clusters, points_per_cluster, dim, 42);
        
        // Create ground truth labels
        std::vector<std::uint32_t> ground_truth(n);
        for (std::size_t i = 0; i < n; ++i) {
            ground_truth[i] = i / points_per_cluster;
        }
        
        vesper::index::KmeansParams params{
            .k = n_clusters,
            .max_iter = 100,
            .epsilon = 1e-4f,
            .seed = 42,
            .balanced = false,
            .verbose = false,
            .n_redo = 3
        };
        
        auto result = vesper::index::kmeans_cluster(
            data.data(), n, dim, params);
        
        REQUIRE(result.has_value());
        
        const auto& res = result.value();
        REQUIRE(res.centroids.size() == n_clusters);
        REQUIRE(res.assignments.size() == n);
        REQUIRE(res.cluster_sizes.size() == n_clusters);
        REQUIRE(res.iterations > 0);
        REQUIRE(res.iterations <= params.max_iter);
        REQUIRE(res.inertia > 0.0f);
        
        // Check clustering quality (purity)
        float purity = compute_purity(res.assignments, ground_truth, n_clusters);
        REQUIRE(purity > 0.8f); // Should achieve high purity on well-separated clusters
    }
    
    SECTION("K-means handles edge cases") {
        const std::size_t dim = 4;
        
        SECTION("Single cluster") {
            const std::size_t n = 10;
            std::vector<float> data(n * dim, 1.0f);
            
            vesper::index::KmeansParams params{.k = 1};
            auto result = vesper::index::kmeans_cluster(
                data.data(), n, dim, params);
            
            REQUIRE(result.has_value());
            REQUIRE(result->centroids.size() == 1);
            REQUIRE_THAT(result->inertia, WithinAbs(0.0f, 1e-6f));
        }
        
        SECTION("K equals N") {
            const std::size_t n = 5;
            std::vector<float> data = generate_clustered_data(n, 1, dim, 42);
            
            vesper::index::KmeansParams params{.k = n};
            auto result = vesper::index::kmeans_cluster(
                data.data(), n, dim, params);
            
            REQUIRE(result.has_value());
            REQUIRE(result->centroids.size() == n);
            REQUIRE_THAT(result->inertia, WithinAbs(0.0f, 1e-6f));
        }
        
        SECTION("Invalid parameters") {
            const std::size_t n = 10;
            std::vector<float> data(n * dim);
            
            // K > N
            vesper::index::KmeansParams params1{.k = n + 1};
            auto result1 = vesper::index::kmeans_cluster(
                data.data(), n, dim, params1);
            REQUIRE(!result1.has_value());
            
            // K = 0
            vesper::index::KmeansParams params2{.k = 0};
            auto result2 = vesper::index::kmeans_cluster(
                data.data(), n, dim, params2);
            REQUIRE(!result2.has_value());
        }
    }
    
    SECTION("K-means is deterministic with fixed seed") {
        const std::size_t n = 50;
        const std::size_t dim = 8;
        const std::uint32_t k = 3;
        
        std::vector<float> data = generate_clustered_data(k, n/k, dim, 137);
        
        vesper::index::KmeansParams params{
            .k = k,
            .seed = 42,
            .n_redo = 1
        };
        
        auto result1 = vesper::index::kmeans_cluster(data.data(), n, dim, params);
        auto result2 = vesper::index::kmeans_cluster(data.data(), n, dim, params);
        
        REQUIRE(result1.has_value());
        REQUIRE(result2.has_value());
        
        // Results should be identical
        REQUIRE(result1->assignments == result2->assignments);
        REQUIRE_THAT(result1->inertia, WithinRel(result2->inertia, 1e-6f));
    }
}