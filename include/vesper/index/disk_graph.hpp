#pragma once

/** \file disk_graph.hpp
 *  \brief DiskANN-style graph index (Vamana) for billion-scale vector search.
 *
 * Implements a flat proximity graph with SSD-resident adjacency lists,
 * enabling billion-scale search with limited RAM. Key features:
 * - RobustPrune algorithm for graph construction
 * - SSD-aware beam search with PQ assistance
 * - LRU cache for hot nodes and entry points
 * - Filtered search with bitmap masks
 *
 * Thread-safety: Build is single-threaded; search operations are thread-safe.
 * Memory: O(cache_size * (d + degree)) RAM; rest on SSD.
 * 
 * References:
 * - Subramanya et al., "DiskANN: Fast Accurate Billion-point Nearest 
 *   Neighbor Search on a Single Node", NeurIPS 2019
 * - Gollapudi et al., "Filtered-DiskANN", SIGMOD 2023
 */

#include <cstdint>
#include <expected>
#include <memory>
#include <vesper/span_polyfill.hpp>
#include <vector>
#include <string>
#include <atomic>
#include <optional>
#include <unordered_map>
#include <deque>
#include <mutex>

#include "vesper/error.hpp"
#include "vesper/index/product_quantizer.hpp"
#include "vesper/cache/lru_cache.hpp"

namespace vesper::index {

template<typename T>
using span = std::span<T>;

/** \brief Build parameters for Vamana graph construction. */
struct VamanaBuildParams {
    std::uint32_t degree{64};           /**< Max out-degree R (typically 32-128) */
    float alpha{1.2f};                   /**< Pruning parameter α > 1 for longer edges */
    std::uint32_t L{128};                /**< Search list size during construction */
    std::uint32_t max_iters{2};          /**< Number of construction iterations */
    bool use_pq{true};                   /**< Enable PQ compression for vectors */
    std::uint32_t pq_m{8};               /**< Number of PQ subquantizers */
    std::uint32_t pq_bits{8};            /**< Bits per PQ subquantizer */
    std::uint32_t num_threads{0};        /**< Build threads (0 = auto) */
    std::uint32_t shard_size{1000000};   /**< Points per shard for large datasets */
    std::uint32_t seed{1337};            /**< Random seed for reproducibility */
    bool two_pass{true};                 /**< Two-pass construction for better quality */
    std::string cache_path{""};          /**< Optional path for build cache */
};

/** \brief Search parameters for Vamana graph. */
struct VamanaSearchParams {
    std::uint32_t beam_width{64};        /**< Beam size W for search (32-256) */
    std::uint32_t k{10};                 /**< Number of neighbors to return */
    std::uint32_t cache_nodes{10000};    /**< Max nodes to keep in RAM cache */
    bool use_pq_distance{true};          /**< Use PQ for distance estimation */
    const std::uint8_t* filter_mask{nullptr}; /**< Optional bitmap filter */
    std::size_t filter_size{0};          /**< Size of filter in bytes */
    std::uint32_t init_list_size{100};   /**< Initial candidate list size */
};

/** \brief Statistics from graph construction. */
struct VamanaBuildStats {
    std::uint32_t num_nodes{0};          /**< Total nodes in graph */
    std::uint32_t max_degree{0};         /**< Maximum out-degree observed */
    float avg_degree{0.0f};              /**< Average out-degree */
    std::uint64_t total_edges{0};        /**< Total edges in graph */
    std::uint64_t build_time_ms{0};      /**< Total build time */
    std::uint64_t io_time_ms{0};         /**< Time spent in I/O */
    std::size_t graph_size_bytes{0};     /**< Total size on disk */
    std::size_t cache_size_bytes{0};     /**< RAM cache size */
};

/** \brief Node metadata for caching and I/O. */
struct GraphNode {
    std::uint32_t id{0};                 /**< Node ID */
    std::vector<std::uint32_t> neighbors; /**< Adjacency list */
    std::vector<std::uint8_t> pq_code;   /**< PQ-compressed vector */
    std::uint64_t disk_offset{0};        /**< Offset in adjacency file */
    std::uint32_t access_count{0};       /**< For LRU caching */
    std::uint64_t last_access{0};        /**< Timestamp for LRU */
};

/** \brief Cache entry for frequently accessed nodes. */
struct CacheEntry {
    std::uint32_t node_id{0};
    std::vector<float> vector;           /**< Full vector (if cached) */
    std::vector<std::uint32_t> neighbors; /**< Adjacency list */
    std::vector<std::uint8_t> pq_code;   /**< PQ code */
    std::uint64_t last_access{0};
    std::uint32_t access_count{0};
};

/** \brief I/O statistics for monitoring. */
struct IOStats {
    std::atomic<std::uint64_t> reads{0};
    std::atomic<std::uint64_t> read_bytes{0};
    std::atomic<std::uint64_t> cache_hits{0};
    std::atomic<std::uint64_t> cache_misses{0};
    std::atomic<std::uint64_t> prefetch_hits{0};
    
    // Default constructor
    IOStats() = default;
    
    // Copy constructor (needed because atomics are not copyable)
    IOStats(const IOStats& other) 
        : reads(other.reads.load())
        , read_bytes(other.read_bytes.load())
        , cache_hits(other.cache_hits.load())
        , cache_misses(other.cache_misses.load())
        , prefetch_hits(other.prefetch_hits.load()) {}
    
    // Copy assignment
    IOStats& operator=(const IOStats& other) {
        if (this != &other) {
            reads.store(other.reads.load());
            read_bytes.store(other.read_bytes.load());
            cache_hits.store(other.cache_hits.load());
            cache_misses.store(other.cache_misses.load());
            prefetch_hits.store(other.prefetch_hits.load());
        }
        return *this;
    }
};

/** \brief DiskANN graph index for billion-scale vector search.
 *
 * Example usage:
 * \code
 * VamanaBuildParams params;
 * params.degree = 64;
 * params.alpha = 1.2f;
 * 
 * DiskGraphIndex index(128); // 128-dimensional vectors
 * auto stats = index.build(vectors, params);
 * 
 * VamanaSearchParams search_params;
 * search_params.beam_width = 64;
 * auto results = index.search(query, search_params);
 * \endcode
 */
class DiskGraphIndex {
public:
    /** \brief Construct index for given vector dimension.
     * \param dim Vector dimension
     */
    explicit DiskGraphIndex(std::size_t dim);
    
    /** \brief Destructor ensures files are properly closed. */
    ~DiskGraphIndex();

    /** \brief Build graph index from vectors.
     * 
     * Constructs a Vamana graph using RobustPrune algorithm.
     * For datasets larger than RAM, uses sharding.
     * 
     * \param vectors Training vectors [n x dim]
     * \param params Build parameters
     * \return Build statistics or error
     */
    [[nodiscard]] auto build(
        span<const float> vectors,
        const VamanaBuildParams& params = {})
        -> std::expected<VamanaBuildStats, core::error>;

    /** \brief Add vectors to existing index.
     * 
     * Incrementally adds vectors using RobustPrune to maintain graph quality.
     * 
     * \param vectors Vectors to add [n x dim]
     * \param ids Optional IDs (auto-generated if empty)
     * \return Number of vectors added or error
     */
    [[nodiscard]] auto add_vectors(
        span<const float> vectors,
        span<const std::uint32_t> ids = {})
        -> std::expected<std::uint32_t, core::error>;

    /** \brief Search for k-nearest neighbors.
     * 
     * Performs beam search with optional filtering.
     * 
     * \param query Query vector [dim]
     * \param params Search parameters
     * \return Vector of (distance, id) pairs sorted by distance
     */
    [[nodiscard]] auto search(
        span<const float> query,
        const VamanaSearchParams& params = {}) const
        -> std::expected<std::vector<std::pair<float, std::uint32_t>>, core::error>;

    /** \brief Batch search for multiple queries.
     * 
     * \param queries Query vectors [n x dim]
     * \param params Search parameters
     * \return Vector of result vectors, one per query
     */
    [[nodiscard]] auto batch_search(
        span<const float> queries,
        std::size_t n_queries,
        const VamanaSearchParams& params = {}) const
        -> std::expected<std::vector<std::vector<std::pair<float, std::uint32_t>>>, core::error>;

    /** \brief Save index to disk.
     * 
     * Writes graph structure, PQ codes, and metadata to disk.
     * 
     * \param path Directory path for index files
     * \return Success or error
     */
    [[nodiscard]] auto save(const std::string& path) const
        -> std::expected<void, core::error>;

    /** \brief Load index from disk.
     * 
     * Loads graph structure and initializes cache.
     * 
     * \param path Directory path containing index files
     * \return Success or error
     */
    [[nodiscard]] auto load(const std::string& path)
        -> std::expected<void, core::error>;

    /** \brief Warm up cache with frequently accessed nodes.
     * 
     * Pre-loads entry points and high-degree nodes into RAM.
     * 
     * \param max_nodes Maximum nodes to cache
     * \return Number of nodes cached
     */
    [[nodiscard]] auto warmup_cache(std::uint32_t max_nodes = 10000)
        -> std::uint32_t;

    /** \brief Get current I/O statistics. */
    [[nodiscard]] auto io_stats() const -> IOStats;

    /** \brief Get build statistics. */
    [[nodiscard]] auto build_stats() const -> VamanaBuildStats;

    /** \brief Get number of vectors in index. */
    [[nodiscard]] auto size() const -> std::uint32_t;

    /** \brief Get vector dimension. */
    [[nodiscard]] auto dimension() const -> std::size_t;

    /** \brief Clear cache (useful for benchmarking). */
    auto clear_cache() -> void;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/** \brief Standalone RobustPrune algorithm for graph construction.
 * 
 * Given a set of candidates and a node, prunes edges to maintain
 * graph quality while limiting degree.
 * 
 * \param node_id Current node being processed
 * \param candidates Candidate neighbors with distances
 * \param degree Maximum out-degree R
 * \param alpha Pruning parameter (α > 1 for longer edges)
 * \param vectors All vectors for distance computation
 * \param dim Vector dimension
 * \return Pruned neighbor list
 */
[[nodiscard]] auto robust_prune(
    std::uint32_t node_id,
    std::vector<std::pair<float, std::uint32_t>>& candidates,
    std::uint32_t degree,
    float alpha,
    const float* vectors,
    std::size_t dim)
    -> std::vector<std::uint32_t>;

/** \brief Greedy search to find nearest neighbors in graph.
 * 
 * Used during both construction and query processing.
 * 
 * \param query Query vector
 * \param entry_points Starting nodes for search
 * \param graph Graph structure (adjacency lists)
 * \param L Search list size
 * \param vectors All vectors (for distance computation)
 * \param dim Vector dimension
 * \return Nearest neighbors with distances
 */
[[nodiscard]] auto greedy_search(
    span<const float> query,
    const std::vector<std::uint32_t>& entry_points,
    const std::vector<std::vector<std::uint32_t>>& graph,
    std::uint32_t L,
    const float* vectors,
    std::size_t dim)
    -> std::vector<std::pair<float, std::uint32_t>>;

} // namespace vesper::index