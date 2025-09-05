#pragma once

/** \file hnsw.hpp
 *  \brief Hierarchical Navigable Small World (HNSW) index for high-recall ANN search.
 *
 * HNSW provides excellent recall/latency tradeoffs for RAM-resident indices.
 * Features:
 * - Multi-layer proximity graph with hierarchical structure
 * - Greedy search with beam expansion (efSearch)
 * - Filtered search with bitmap masks
 * - Lock-free concurrent search operations
 *
 * Thread-safety: Build is single-threaded; search operations are thread-safe.
 * Memory: O(M * N) edges where M is max connections per node.
 */

#include <cstdint>
#include <expected>
#include <memory>
#include <vesper/span_polyfill.hpp>
#include <vector>
#include <atomic>
#include <optional>

#include "vesper/error.hpp"

namespace vesper::index {

/** \brief HNSW build parameters. */
struct HnswBuildParams {
    std::uint32_t M{16};                    /**< Max connections per node */
    std::uint32_t efConstruction{200};      /**< Beam width during construction */
    std::uint32_t seed{42};                 /**< Random seed for level assignment */
    bool extend_candidates{true};           /**< Use extended candidate set (CRITICAL for recall) */
    bool keep_pruned_connections{true};     /**< Keep pruned connections (Algorithm 4) */
    std::uint32_t max_M{16};                /**< Max connections for level > 0 */
    std::uint32_t max_M0{32};               /**< Max connections for level 0 (2x M) */
    std::uint32_t num_threads{0};           /**< Number of threads (0 = auto) */
    // Adaptive efConstruction schedule for upper layers
    bool adaptive_ef{false};                /**< If true, use smaller ef on upper layers */
    std::uint32_t efConstructionUpper{0};   /**< Optional explicit ef for layers > 0 (0 = auto: max(50, efConstruction/2)) */
};

/** \brief HNSW search parameters. */
struct HnswSearchParams {
    std::uint32_t efSearch{100};            /**< Beam width during search (min 100 for good recall) */
    std::uint32_t k{10};                    /**< Number of neighbors to return */
    const std::uint8_t* filter_mask{nullptr}; /**< Optional bitmap filter */
    std::size_t filter_size{0};             /**< Size of filter in bytes */
};

/** \brief HNSW index statistics. */
struct HnswStats {
    std::size_t n_nodes{0};                 /**< Total nodes in graph */
    std::size_t n_edges{0};                 /**< Total edges in graph */
    std::size_t n_levels{0};                /**< Number of hierarchy levels */
    std::size_t memory_bytes{0};            /**< Total memory usage */
    float avg_degree{0.0f};                 /**< Average node degree */
    std::vector<std::size_t> level_counts;  /**< Nodes per level */
};

/** \brief Hierarchical Navigable Small World index.
 *
 * HNSW algorithm for approximate nearest neighbor search
 * with support for filtered queries and incremental construction.
 */
class HnswIndex {
public:
    HnswIndex();
    ~HnswIndex();
    HnswIndex(HnswIndex&&) noexcept;
    HnswIndex& operator=(HnswIndex&&) noexcept;
    HnswIndex(const HnswIndex&) = delete;
    HnswIndex& operator=(const HnswIndex&) = delete;
    
    /** \brief Initialize index with parameters.
     *
     * \param dim Vector dimensionality
     * \param params Build parameters
     * \param max_elements Expected maximum elements (for pre-allocation)
     * \return Success or error
     *
     * Preconditions: dim > 0; M >= 2; efConstruction >= M
     * Complexity: O(1) - just initialization
     */
    auto init(std::size_t dim, const HnswBuildParams& params,
              std::size_t max_elements = 0)
        -> std::expected<void, core::error>;
    
    /** \brief Add a vector to the index.
     *
     * \param id Vector identifier
     * \param data Vector data [dim]
     * \return Success or error
     *
     * Preconditions: Index is initialized; vector has correct dim
     * Complexity: O(M * log(N) * efConstruction)
     * Thread-safety: NOT thread-safe; use external synchronization
     */
    auto add(std::uint64_t id, const float* data)
        -> std::expected<void, core::error>;
    
    /** \brief Batch add vectors.
     *
     * \param ids Vector identifiers [n]
     * \param data Vectors [n x dim]
     * \param n Number of vectors
     * \return Success or error
     *
     * Thread-safety: Internally parallelized where safe
     */
    auto add_batch(const std::uint64_t* ids, const float* data, std::size_t n)
        -> std::expected<void, core::error>;
    
    /** \brief Search for k nearest neighbors.
     *
     * \param query Query vector [dim]
     * \param params Search parameters
     * \return Vector IDs and distances of k nearest neighbors
     *
     * Preconditions: Index is non-empty
     * Complexity: O(efSearch * log(N))
     * Thread-safety: Safe for concurrent calls
     */
    auto search(const float* query, const HnswSearchParams& params) const
        -> std::expected<std::vector<std::pair<std::uint64_t, float>>, core::error>;
    
    /** \brief Batch search for multiple queries.
     *
     * \param queries Query vectors [n_queries x dim]
     * \param n_queries Number of queries
     * \param params Search parameters
     * \return Results for each query
     *
     * Thread-safety: Internally parallelized
     */
    auto search_batch(const float* queries, std::size_t n_queries,
                      const HnswSearchParams& params) const
        -> std::expected<std::vector<std::vector<std::pair<std::uint64_t, float>>>, core::error>;
    
    /** \brief Get index statistics. */
    auto get_stats() const noexcept -> HnswStats;

    /** \brief Compute reachability on base layer via BFS (testing/diagnostics). */
    auto reachable_count_base_layer() const noexcept -> std::size_t;

    /** \brief Check if index is initialized. */
    auto is_initialized() const noexcept -> bool;

    /** \brief Get vector dimensionality. */
    auto dimension() const noexcept -> std::size_t;

    /** \brief Get number of indexed vectors. */
    auto size() const noexcept -> std::size_t;
    
    /** \brief Mark vector as deleted (soft delete). */
    auto mark_deleted(std::uint64_t id) -> std::expected<void, core::error>;
    
    /** \brief Resize index capacity.
     *
     * \param new_max_elements New maximum capacity
     * \return Success or error
     */
    auto resize(std::size_t new_max_elements) -> std::expected<void, core::error>;
    
    /** \brief Optimize graph connectivity.
     *
     * Runs additional optimization passes to improve graph quality.
     * This can improve search performance at the cost of build time.
     */
    auto optimize() -> std::expected<void, core::error>;
    
    /** \brief Save index to file.
     *
     * \param path Output file path
     * \return Success or error
     */
    auto save(const std::string& path) const -> std::expected<void, core::error>;
    
    /** \brief Load index from file.
     *
     * \param path Input file path
     * \return Loaded index or error
     */
    static auto load(const std::string& path) -> std::expected<HnswIndex, core::error>;
    
    /** \brief Get build parameters. */
    auto get_build_params() const noexcept -> HnswBuildParams;
    
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/** \brief Compute recall of HNSW index against ground truth.
 *
 * \param index Built HNSW index
 * \param queries Test queries [n_queries x dim]
 * \param n_queries Number of queries
 * \param ground_truth True nearest neighbors [n_queries x k]
 * \param k Number of neighbors to evaluate
 * \param params Search parameters
 * \return Recall@k in [0, 1]
 */
auto compute_recall(const HnswIndex& index,
                   const float* queries, std::size_t n_queries,
                   const std::uint64_t* ground_truth, std::size_t k,
                   const HnswSearchParams& params) -> float;

/** \brief RobustPrune algorithm for edge selection.
 *
 * Selects diverse neighbors to maintain connectivity and search quality.
 *
 * \param candidates Candidate neighbors with distances
 * \param M Maximum number of neighbors to select
 * \param extend_candidates Include pruned candidates if space allows
 * \param keep_pruned Return pruned candidates separately
 * \return Selected neighbors and optionally pruned ones
 */
auto robust_prune(
    std::vector<std::pair<float, std::uint32_t>>& candidates,
    std::uint32_t M,
    bool extend_candidates = false,
    bool keep_pruned = false)
    -> std::pair<std::vector<std::uint32_t>, std::vector<std::uint32_t>>;

} // namespace vesper::index