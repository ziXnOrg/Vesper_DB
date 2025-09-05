#pragma once

/** \file kmeans.hpp
 *  \brief K-means clustering for coarse quantization in IVF indices.
 *
 * Implements k-means++ initialization and Lloyd's algorithm with optimizations.
 * Features:
 * - K-means++ for better initial centroids
 * - Early stopping on convergence
 * - SIMD-accelerated distance computations
 * - Balanced cluster assignment options
 *
 * Thread-safety: Training is internally parallelized but not thread-safe.
 * Determinism: Fixed seed produces reproducible results.
 */

#include <cstdint>
#include <expected>
#include <vesper/span_polyfill.hpp>
#include <vector>
#include <random>

#include "vesper/error.hpp"

namespace vesper::index {

/** \brief K-means clustering parameters. */
struct KmeansParams {
    std::uint32_t k{256};                /**< Number of clusters */
    std::uint32_t max_iter{25};          /**< Maximum iterations */
    float epsilon{1e-4f};                 /**< Convergence threshold */
    std::uint32_t seed{42};              /**< Random seed */
    bool balanced{false};                /**< Force balanced clusters */
    bool verbose{false};                  /**< Progress output */
    std::uint32_t n_redo{1};             /**< Number of runs (best selected) */
};

/** \brief K-means clustering result. */
struct KmeansResult {
    std::vector<std::vector<float>> centroids;  /**< Cluster centers [k x dim] */
    std::vector<std::uint32_t> assignments;     /**< Point assignments [n] */
    std::vector<std::uint32_t> cluster_sizes;   /**< Points per cluster [k] */
    float inertia{0.0f};                        /**< Sum of squared distances */
    std::uint32_t iterations{0};                /**< Iterations performed */
    float time_sec{0.0f};                       /**< Wall time */
};

/** \brief K-means clustering algorithm.
 *
 * Partitions data into k clusters minimizing within-cluster variance.
 *
 * \param data Input vectors [n x dim]
 * \param n Number of vectors
 * \param dim Vector dimensionality
 * \param params Clustering parameters
 * \return Clustering result or error
 *
 * Preconditions: n >= k; data contains finite values
 * Complexity: O(n * k * dim * iterations)
 */
auto kmeans_cluster(const float* data, std::size_t n, std::size_t dim,
                   const KmeansParams& params)
    -> std::expected<KmeansResult, core::error>;

/** \brief K-means++ initialization.
 *
 * Selects initial centroids with probability proportional to squared distance.
 *
 * \param data Input vectors [n x dim]
 * \param n Number of vectors
 * \param dim Vector dimensionality  
 * \param k Number of centroids
 * \param seed Random seed
 * \return Initial centroids [k x dim]
 *
 * Complexity: O(n * k * dim)
 */
auto kmeans_plusplus_init(const float* data, std::size_t n, std::size_t dim,
                          std::uint32_t k, std::uint32_t seed)
    -> std::vector<std::vector<float>>;

/** \brief Assign points to nearest centroids.
 *
 * \param data Points [n x dim]
 * \param n Number of points
 * \param centroids Cluster centers [k x dim]
 * \param assignments Output assignments [n]
 * \return Total inertia (sum of squared distances)
 *
 * Thread-safety: Internally parallelized
 * Complexity: O(n * k * dim)
 */
auto kmeans_assign(const float* data, std::size_t n,
                  const std::vector<std::vector<float>>& centroids,
                  std::span<std::uint32_t> assignments) -> float;

/** \brief Update centroids from assignments.
 *
 * \param data Points [n x dim]
 * \param n Number of points  
 * \param dim Dimensionality
 * \param assignments Point assignments [n]
 * \param k Number of clusters
 * \param centroids Updated centroids [k x dim]
 *
 * Complexity: O(n * dim)
 */
auto kmeans_update_centroids(const float* data, std::size_t n, std::size_t dim,
                             std::span<const std::uint32_t> assignments,
                             std::uint32_t k,
                             std::vector<std::vector<float>>& centroids) -> void;

/** \brief Hierarchical balanced k-means.
 *
 * Builds balanced tree of clusters for large k values.
 * Useful for IVF with many lists (nlist > 1024).
 *
 * \param data Input vectors [n x dim]
 * \param n Number of vectors
 * \param dim Vector dimensionality
 * \param k Target number of leaf clusters
 * \param branching_factor Tree branching (typically 8-16)
 * \param params Base k-means parameters
 * \return Hierarchical clustering result
 */
auto hierarchical_kmeans(const float* data, std::size_t n, std::size_t dim,
                         std::uint32_t k, std::uint32_t branching_factor,
                         const KmeansParams& params)
    -> std::expected<KmeansResult, core::error>;

/** \brief Compute clustering quality metrics.
 *
 * \param data Points [n x dim]
 * \param n Number of points
 * \param dim Dimensionality
 * \param result Clustering result
 * \return Metrics including silhouette coefficient, Davies-Bouldin index
 */
struct ClusterMetrics {
    float silhouette{0.0f};              /**< Silhouette coefficient [-1, 1] */
    float davies_bouldin{0.0f};          /**< Davies-Bouldin index (lower is better) */
    float calinski_harabasz{0.0f};       /**< Calinski-Harabasz index (higher is better) */
};

auto evaluate_clustering(const float* data, std::size_t n, std::size_t dim,
                         const KmeansResult& result) -> ClusterMetrics;

} // namespace vesper::index