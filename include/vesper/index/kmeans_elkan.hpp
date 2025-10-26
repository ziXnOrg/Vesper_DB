#pragma once

/** \file kmeans_elkan.hpp
 *  \brief Elkan's accelerated k-means using triangle inequality.
 *
 * Elkan's algorithm reduces distance computations by 50-90% compared to Lloyd's
 * by using the triangle inequality to maintain bounds on point-to-centroid distances.
 *
 * Key optimizations:
 * - Lower bounds for distances to all non-assigned centroids
 * - Upper bound for distance to assigned centroid
 * - Inter-centroid distance matrix for bound updates
 * - SIMD-accelerated distance computations
 * - Cache-friendly memory access patterns
 *
 * Thread-safety: No shared mutable state; concurrent calls on distinct instances are thread-safe.
 *             Not thread-safe for concurrent calls on the same instance without external synchronization.
 * Memory: O(n*k) for bounds, O(k*k) for inter-centroid distances.
 * Performance: 2-10x faster than Lloyd's algorithm on typical datasets.
 */

#include <cstdint>
#include <expected>
#include <vesper/span_polyfill.hpp>
#include <vector>
#include <memory>
#include <atomic>

#include "vesper/error.hpp"
#include "vesper/index/kmeans.hpp"
#include "vesper/index/aligned_buffer.hpp"

namespace vesper::index {

/** \brief Elkan's k-means algorithm with triangle inequality optimization.
 *
 * Maintains bounds to avoid unnecessary distance calculations:
 * - upper[i]: Upper bound on distance from point i to its assigned centroid
 * - lower[i][j]: Lower bound on distance from point i to centroid j
 * - s_quarter[j]: One quarter of the squared minimum inter-centroid distance for j (i.e., 0.25 * min_{m!=j} ||μ_j-μ_m||^2) used with squared-distance Elkan bounds
 */
class KmeansElkan {
public:
    /** \brief Configuration for Elkan's algorithm. */
    struct Config {
        std::uint32_t k{256};                /**< Number of clusters */
        std::uint32_t max_iter{25};          /**< Maximum iterations */
        float epsilon{1e-4f};                 /**< Convergence threshold */
        std::uint32_t seed{42};              /**< Random seed */
        bool use_parallel{true};             /**< Enable parallelization */
        std::uint32_t n_threads{0};          /**< Threads (0=auto) */
        bool verbose{false};                  /**< Progress output */

        // Initialization method
        enum class InitMethod { KMeansPlusPlus, KMeansParallel };
        InitMethod init_method{InitMethod::KMeansPlusPlus};
        // k-means|| parameters
        std::uint32_t kmeans_parallel_rounds{5};
        std::uint32_t kmeans_parallel_oversampling{0}; // 0 => default 2*k
    };

    /** \brief Performance statistics. */
    struct Stats {
        std::uint64_t distance_computations{0};   /**< Total distance calcs */
        std::uint64_t distance_skipped{0};        /**< Skipped via bounds */
        float skip_rate{0.0f};                    /**< Fraction skipped */
        std::uint32_t iterations{0};              /**< Iterations performed */
        float final_inertia{0.0f};                /**< Final SSE */
        float time_sec{0.0f};                     /**< Wall time */
        std::uint64_t reassignments{0};           /**< Total reassignments */
        std::uint32_t empty_clusters{0};          /**< Empty clusters encountered */
    };

    KmeansElkan() = default;
    ~KmeansElkan() = default;

    /** \brief Run Elkan's k-means clustering.
     *
     * \param data Input vectors [n x dim]
     * \param n Number of vectors
     * \param dim Vector dimensionality
     * \param config Algorithm configuration
     * \return Clustering result with statistics or error
     *
     * Preconditions: n >= k; data contains finite values
     * Complexity: O(n*k) per iteration typical, O(n*k²) worst case
     */
    auto cluster(const float* data, std::size_t n, std::size_t dim,
                const Config& config)
        -> std::expected<KmeansResult, core::error>;

    /** \brief Get performance statistics from last run. */
    auto get_stats() const noexcept -> Stats { return stats_; }

private:
    /** \brief Point bounds for Elkan's algorithm. */
    struct PointBounds {
        float upper;                          /**< Upper bound to assigned */
        std::vector<float> lower;             /**< Lower bounds to all */
        std::uint32_t assignment;             /**< Current assignment */
        bool upper_tight{false};              /**< Is upper bound exact? */
    };

    /** \brief Initialize bounds for all points. */
    auto initialize_bounds(const float* data, std::size_t n, std::size_t dim,
                          const std::vector<std::vector<float>>& centroids)
        -> std::vector<PointBounds>;

    /** \brief Update inter-centroid distances and s values. */
    auto update_centroid_distances(const std::vector<std::vector<float>>& centroids)
        -> std::pair<std::vector<std::vector<float>>, std::vector<float>>;

    /** \brief Update inter-centroid distances with aligned buffers. */
    auto update_centroid_distances_aligned(const AlignedCentroidBuffer& centroids)
        -> std::pair<AlignedDistanceMatrix, std::vector<float>>;

    /** \brief Assign points using triangle inequality bounds. */
    auto assign_with_bounds(const float* data, std::size_t n, std::size_t dim,
                           const std::vector<std::vector<float>>& centroids,
                           std::vector<PointBounds>& bounds,
                           const std::vector<float>& s_quarter)
        -> float;

    /** \brief Update bounds after centroid movement. */
    auto update_bounds(std::vector<PointBounds>& bounds,
                       const std::vector<float>& centroid_shift,
                       const std::vector<std::vector<float>>& inter_centroid_dist);

    /** \brief Compute distance with instrumentation. */
    inline auto compute_distance_instrumented(const float* a, const float* b,
                                             std::size_t dim) -> float;

    Stats stats_;
    std::atomic<std::uint64_t> distance_counter_{0};
    std::atomic<std::uint64_t> skip_counter_{0};
    std::atomic<std::uint64_t> reassignment_counter_{0};

    std::vector<std::vector<float>> inter_centroid_dist_cache_;
};

/** \brief Parallel k-means|| initialization.
 *
 * Scalable variant of k-means++ that reduces passes over data.
 * Instead of k sequential passes, uses O(log n) rounds.
 *
 * \param data Input vectors [n x dim]
 * \param n Number of vectors
 * \param dim Vector dimensionality
 * \param k Number of clusters
 * \param rounds Number of sampling rounds (default: 5)
 * \param seed Random seed
 * \return Initial centroids
 *
 * Reference: Bahmani et al. "Scalable K-Means++" (2012)
 */
auto kmeans_parallel_init(const float* data, std::size_t n, std::size_t dim,
                          std::uint32_t k, std::uint32_t rounds = 5,
                          std::uint32_t oversampling_l = 0,
                          std::uint32_t seed = 42)
    -> std::vector<std::vector<float>>;

/** \brief Mini-batch k-means for large datasets.
 *
 * Updates centroids using random mini-batches instead of full data.
 * Trades accuracy for speed on very large datasets.
 *
 * \param data Input vectors [n x dim]
 * \param n Number of vectors
 * \param dim Vector dimensionality
 * \param batch_size Mini-batch size (typical: 100-1000)
 * \param config Algorithm configuration
 * \return Clustering result or error
 */
auto kmeans_minibatch(const float* data, std::size_t n, std::size_t dim,
                     std::size_t batch_size, const KmeansElkan::Config& config)
    -> std::expected<KmeansResult, core::error>;

} // namespace vesper::index
