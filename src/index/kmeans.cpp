#include "vesper/index/kmeans.hpp"
#include "vesper/kernels/dispatch.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <numeric>
#include <random>
#include <execution>
#include <cmath>

namespace vesper::index {

namespace {

/** \brief Compute squared L2 distance between vectors. */
[[gnu::hot]] inline auto compute_distance(const float* a, const float* b, std::size_t dim) -> float {
    const auto& ops = kernels::select_backend_auto();
    return ops.l2_sq(std::span(a, dim), std::span(b, dim));
}

/** \brief Find nearest centroid for a point. */
auto find_nearest_centroid(const float* point, 
                           const std::vector<std::vector<float>>& centroids,
                           std::size_t dim) -> std::pair<std::uint32_t, float> {
    std::uint32_t best_idx = 0;
    float best_dist = std::numeric_limits<float>::max();
    
    for (std::uint32_t i = 0; i < centroids.size(); ++i) {
        const float dist = compute_distance(point, centroids[i].data(), dim);
        if (dist < best_dist) {
            best_dist = dist;
            best_idx = i;
        }
    }
    
    return {best_idx, best_dist};
}

} // anonymous namespace

auto kmeans_plusplus_init(const float* data, std::size_t n, std::size_t dim,
                          std::uint32_t k, std::uint32_t seed)
    -> std::vector<std::vector<float>> {
    std::vector<std::vector<float>> centroids;
    centroids.reserve(k);
    
    std::mt19937 gen(seed);
    
    // Choose first centroid randomly
    std::uniform_int_distribution<std::size_t> first_dist(0, n - 1);
    const std::size_t first_idx = first_dist(gen);
    centroids.emplace_back(data + first_idx * dim, data + (first_idx + 1) * dim);
    
    // For remaining centroids, use D² weighting
    std::vector<float> min_distances(n, std::numeric_limits<float>::max());
    
    for (std::uint32_t c = 1; c < k; ++c) {
        // Update minimum distances to nearest centroid
        const auto& last_centroid = centroids.back();
        
        #pragma omp parallel for
        for (std::size_t i = 0; i < n; ++i) {
            const float dist = compute_distance(data + i * dim, last_centroid.data(), dim);
            min_distances[i] = std::min(min_distances[i], dist);
        }
        
        // Compute cumulative distribution
        std::vector<double> cumsum(n);
        cumsum[0] = min_distances[0];
        for (std::size_t i = 1; i < n; ++i) {
            cumsum[i] = cumsum[i - 1] + min_distances[i];
        }
        
        // Sample next centroid proportional to squared distance
        std::uniform_real_distribution<double> sample_dist(0.0, cumsum.back());
        const double target = sample_dist(gen);
        
        const auto it = std::lower_bound(cumsum.begin(), cumsum.end(), target);
        const std::size_t idx = std::distance(cumsum.begin(), it);
        
        centroids.emplace_back(data + idx * dim, data + (idx + 1) * dim);
    }
    
    return centroids;
}

auto kmeans_assign(const float* data, std::size_t n,
                  const std::vector<std::vector<float>>& centroids,
                  std::span<std::uint32_t> assignments) -> float {
    const std::size_t dim = centroids[0].size();
    double total_inertia = 0.0;
    
    #pragma omp parallel for reduction(+:total_inertia)
    for (std::size_t i = 0; i < n; ++i) {
        const auto [idx, dist] = find_nearest_centroid(data + i * dim, centroids, dim);
        assignments[i] = idx;
        total_inertia += dist;
    }
    
    return static_cast<float>(total_inertia);
}

auto kmeans_update_centroids(const float* data, std::size_t n, std::size_t dim,
                             std::span<const std::uint32_t> assignments,
                             std::uint32_t k,
                             std::vector<std::vector<float>>& centroids) -> void {
    // Initialize accumulators
    std::vector<std::vector<double>> sums(k, std::vector<double>(dim, 0.0));
    std::vector<std::uint32_t> counts(k, 0);
    
    // Accumulate points
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint32_t cluster = assignments[i];
        counts[cluster]++;
        
        const float* point = data + i * dim;
        for (std::size_t d = 0; d < dim; ++d) {
            sums[cluster][d] += point[d];
        }
    }
    
    // Update centroids
    for (std::uint32_t c = 0; c < k; ++c) {
        if (counts[c] > 0) {
            for (std::size_t d = 0; d < dim; ++d) {
                centroids[c][d] = static_cast<float>(sums[c][d] / counts[c]);
            }
        }
        // If cluster is empty, keep previous centroid
    }
}

auto kmeans_cluster(const float* data, std::size_t n, std::size_t dim,
                   const KmeansParams& params)
    -> std::expected<KmeansResult, core::error> {
    using core::error;
    using core::error_code;
    
    if (n < params.k) {
        return std::vesper_unexpected(error{error_code::precondition_failed, 
                                    "Not enough data points for k clusters",
                                    "kmeans"});
    }
    
    if (params.k == 0) {
        return std::vesper_unexpected(error{error_code::precondition_failed, 
                                    "k must be > 0",
                                    "kmeans"});
    }
    
    const auto start_time = std::chrono::steady_clock::now();
    
    KmeansResult best_result;
    float best_inertia = std::numeric_limits<float>::max();
    
    // Multiple runs with different initializations
    for (std::uint32_t redo = 0; redo < params.n_redo; ++redo) {
        const std::uint32_t seed = params.seed + redo;
        
        // Initialize centroids using k-means++
        auto centroids = kmeans_plusplus_init(data, n, dim, params.k, seed);
        
        std::vector<std::uint32_t> assignments(n);
        float prev_inertia = std::numeric_limits<float>::max();
        std::uint32_t iter = 0;
        
        // Lloyd's algorithm iterations
        for (; iter < params.max_iter; ++iter) {
            // Assign points to nearest centroids
            const float inertia = kmeans_assign(data, n, centroids, assignments);
            
            // Check convergence
            const float change = std::abs(prev_inertia - inertia) / (prev_inertia + 1e-10f);
            if (change < params.epsilon) {
                if (params.verbose) {
                    // Log convergence
                }
                break;
            }
            
            prev_inertia = inertia;
            
            // Update centroids
            kmeans_update_centroids(data, n, dim, assignments, params.k, centroids);
        }
        
        // Keep best result
        if (prev_inertia < best_inertia) {
            best_inertia = prev_inertia;
            
            // Compute cluster sizes
            std::vector<std::uint32_t> cluster_sizes(params.k, 0);
            for (std::uint32_t a : assignments) {
                cluster_sizes[a]++;
            }
            
            best_result = KmeansResult{
                .centroids = std::move(centroids),
                .assignments = std::move(assignments),
                .cluster_sizes = std::move(cluster_sizes),
                .inertia = prev_inertia,
                .iterations = iter
            };
        }
    }
    
    const auto end_time = std::chrono::steady_clock::now();
    const auto duration = std::chrono::duration<float>(end_time - start_time);
    best_result.time_sec = duration.count();
    
    return best_result;
}

auto hierarchical_kmeans(const float* data, std::size_t n, std::size_t dim,
                         std::uint32_t /* k */, std::uint32_t branching_factor,
                         const KmeansParams& params)
    -> std::expected<KmeansResult, core::error> {
    using core::error;
    using core::error_code;
    
    if (branching_factor < 2) {
        return std::vesper_unexpected(error{error_code::precondition_failed, 
                                    "Branching factor must be >= 2",
                                    "kmeans"});
    }
    
    // For now, implement simple non-hierarchical k-means
    // Full hierarchical implementation would build a tree structure
    return kmeans_cluster(data, n, dim, params);
}

auto evaluate_clustering(const float* /* data */, std::size_t /* n */, std::size_t /* dim */,
                         const KmeansResult& /* result */) -> ClusterMetrics {
    ClusterMetrics metrics;
    
    // Simplified metrics computation
    // Full implementation would compute:
    // - Silhouette coefficient
    // - Davies-Bouldin index
    // - Calinski-Harabasz index
    
    // For now, return placeholder values
    metrics.silhouette = 0.5f;
    metrics.davies_bouldin = 1.0f;
    metrics.calinski_harabasz = 100.0f;
    
    return metrics;
}

} // namespace vesper::index