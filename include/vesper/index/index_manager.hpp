#pragma once

/** \file index_manager.hpp
 *  \brief Unified index management for multiple index types.
 *
 * Central component that manages the three-index architecture:
 * - HNSW for hot in-memory segments
 * - IVF-PQ for compact retrieval
 * - DiskANN for billion-scale SSD-resident search
 *
 * Provides unified API for index selection, query planning, and lifecycle management.
 */

#include <memory>
#include <variant>
#include <vector>
#include <expected>
#include <string>
#include <optional>
#include <atomic>
#include <shared_mutex>

#include "vesper/error.hpp"
#include "vesper/filter_expr.hpp"
#include "vesper/metadata/metadata_store.hpp"
#include "vesper/index/hnsw.hpp"
#include "vesper/index/ivf_pq.hpp"
#include "vesper/index/disk_graph.hpp"
#include "vesper/index/product_quantizer.hpp"

namespace vesper::index {

/** \brief Index type enumeration. */
enum class IndexType {
    HNSW,        /**< Hierarchical Navigable Small World */
    IVF_PQ,      /**< Inverted File with Product Quantization */
    DiskANN      /**< Disk-based graph index */
};

/** \brief Index selection strategy. */
enum class SelectionStrategy {
    Auto,        /**< Automatic selection based on data characteristics */
    Manual,      /**< User-specified index type */
    Hybrid       /**< Use multiple indexes with query planning */
};

/** \brief Index build configuration. */
struct IndexBuildConfig {
    IndexType type{IndexType::HNSW};
    SelectionStrategy strategy{SelectionStrategy::Auto};
    
    // HNSW parameters
    HnswBuildParams hnsw_params;
    
    // IVF-PQ parameters
    IvfPqTrainParams ivf_params;
    PqTrainParams pq_params;
    
    // DiskANN parameters
    VamanaBuildParams vamana_params;
    
    // Resource limits
    std::size_t memory_budget_mb{1024};  /**< Memory budget in MB */
    std::size_t cache_size_mb{256};      /**< Cache size for disk indexes */
    
    // Optimization hints
    bool optimize_for_recall{true};      /**< Prioritize recall over speed */
    float target_recall{0.95f};          /**< Target recall@k */
};

/** \brief Query configuration for index manager. */
struct QueryConfig {
    std::uint32_t k{10};                 /**< Number of results */
    float epsilon{0.0f};                 /**< Range query epsilon */
    
    // Index-specific parameters
    std::uint32_t ef_search{64};         /**< HNSW search parameter */
    std::uint32_t nprobe{8};             /**< IVF probing depth */
    std::uint32_t l_search{128};         /**< DiskANN search list size */
    
    // Query planning hints
    bool use_query_planner{true};        /**< Enable cost-based planning */
    std::optional<IndexType> preferred_index; /**< Force specific index */
    
    // Metadata filtering
    std::optional<filter_expr> filter;   /**< Optional metadata filter */
};

/** \brief Statistics for an index. */
struct IndexStats {
    IndexType type;
    std::size_t num_vectors{0};
    std::size_t memory_usage_bytes{0};
    std::size_t disk_usage_bytes{0};
    float build_time_seconds{0.0f};
    float avg_query_time_ms{0.0f};
    float measured_recall{0.0f};
    std::uint64_t query_count{0};  // Not atomic for copyability
};

/** \brief Unified index manager for multiple index types.
 *
 * Example usage:
 * ```cpp
 * IndexManager manager(dimension);
 * 
 * // Configure and build indexes
 * IndexBuildConfig config;
 * config.strategy = SelectionStrategy::Hybrid;
 * config.memory_budget_mb = 2048;
 * 
 * auto result = manager.build(vectors, n, config);
 * 
 * // Query with automatic index selection
 * QueryConfig query_config;
 * query_config.k = 100;
 * 
 * auto results = manager.search(query, query_config);
 * ```
 */
class IndexManager {
public:
    /** \brief Construct manager for given dimension. */
    explicit IndexManager(std::size_t dimension);
    
    ~IndexManager();
    IndexManager(IndexManager&&) noexcept;
    IndexManager& operator=(IndexManager&&) noexcept;
    IndexManager(const IndexManager&) = delete;
    IndexManager& operator=(const IndexManager&) = delete;
    
    /** \brief Build indexes from data.
     *
     * \param vectors Training/indexing vectors [n x dim]
     * \param n Number of vectors
     * \param config Build configuration
     * \return Success or error
     *
     * Thread-safety: Not thread-safe with concurrent queries
     */
    auto build(const float* vectors, std::size_t n, const IndexBuildConfig& config)
        -> std::expected<void, core::error>;
    
    /** \brief Add vectors incrementally.
     *
     * \param id Vector ID
     * \param vector Vector data [dim]
     * \return Success or error
     *
     * Thread-safety: Thread-safe with concurrent reads
     */
    auto add(std::uint64_t id, const float* vector)
        -> std::expected<void, core::error>;
    
    /** \brief Batch add vectors.
     *
     * \param ids Vector IDs [n]
     * \param vectors Vector data [n x dim]
     * \param n Number of vectors
     * \return Success or error
     */
    auto add_batch(const std::uint64_t* ids, const float* vectors, std::size_t n)
        -> std::expected<void, core::error>;
    
    /** \brief Search for nearest neighbors.
     *
     * \param query Query vector [dim]
     * \param config Query configuration
     * \return Top-k results or error
     *
     * Thread-safety: Thread-safe for concurrent calls
     */
    auto search(const float* query, const QueryConfig& config)
        -> std::expected<std::vector<std::pair<std::uint64_t, float>>, core::error>;
    
    /** \brief Batch search for multiple queries.
     *
     * \param queries Query vectors [nq x dim]
     * \param nq Number of queries
     * \param config Query configuration
     * \return Results [nq x k] or error
     */
    auto search_batch(const float* queries, std::size_t nq, const QueryConfig& config)
        -> std::expected<std::vector<std::vector<std::pair<std::uint64_t, float>>>, core::error>;
    
    /** \brief Update/replace an existing vector.
     *
     * \param id Vector ID to update
     * \param vector New vector data [dim]
     * \return Success or error
     *
     * Thread-safety: Thread-safe with concurrent reads
     */
    auto update(std::uint64_t id, const float* vector)
        -> std::expected<void, core::error>;
    
    /** \brief Batch update vectors.
     *
     * \param ids Vector IDs to update [n]
     * \param vectors New vector data [n x dim]
     * \param n Number of vectors
     * \return Success or error
     */
    auto update_batch(const std::uint64_t* ids, const float* vectors, std::size_t n)
        -> std::expected<void, core::error>;
    
    /** \brief Remove vector by ID.
     *
     * \param id Vector ID to remove
     * \return Success or error
     */
    auto remove(std::uint64_t id) -> std::expected<void, core::error>;
    
    /** \brief Get statistics for all indexes. */
    auto get_stats() const -> std::vector<IndexStats>;
    
    /** \brief Get active index types. */
    auto get_active_indexes() const -> std::vector<IndexType>;
    
    /** \brief Optimize indexes for query performance.
     *
     * Rebuilds or reorganizes indexes based on access patterns.
     *
     * \param force Force optimization even if not needed
     * \return Success or error
     */
    auto optimize(bool force = false) -> std::expected<void, core::error>;
    
    /** \brief Save indexes to disk.
     *
     * \param path Directory path for index files
     * \return Success or error
     */
    auto save(const std::string& path) const -> std::expected<void, core::error>;
    
    /** \brief Load indexes from disk.
     *
     * \param path Directory path with index files
     * \return Success or error
     */
    auto load(const std::string& path) -> std::expected<void, core::error>;
    
    /** \brief Get memory usage in bytes. */
    auto memory_usage() const -> std::size_t;
    
    /** \brief Get disk usage in bytes. */
    auto disk_usage() const -> std::size_t;
    
    /** \brief Set memory budget for indexes.
     *
     * May trigger index eviction or conversion.
     *
     * \param budget_mb Memory budget in megabytes
     * \return Success or error
     */
    auto set_memory_budget(std::size_t budget_mb) -> std::expected<void, core::error>;
    
    /** \brief Add or update metadata for a vector.
     *
     * \param id Vector ID
     * \param metadata Key-value metadata pairs
     * \return Success or error
     */
    auto set_metadata(std::uint64_t id, 
                     const std::unordered_map<std::string, metadata::MetadataValue>& metadata)
        -> std::expected<void, core::error>;
    
    /** \brief Get metadata for a vector.
     *
     * \param id Vector ID
     * \return Metadata or error
     */
    auto get_metadata(std::uint64_t id) const
        -> std::expected<std::unordered_map<std::string, metadata::MetadataValue>, core::error>;
    
    /** \brief Remove metadata for a vector.
     *
     * \param id Vector ID
     * \return Success or error
     */
    auto remove_metadata(std::uint64_t id) -> std::expected<void, core::error>;
    
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/** \brief Query planner for cost-based index selection.
 *
 * Estimates query costs and selects optimal index based on:
 * - Data distribution and size
 * - Query characteristics
 * - Resource constraints
 * - Historical performance
 */
class QueryPlanner {
public:
    /** \brief Query plan with selected index and parameters. */
    struct QueryPlan {
        IndexType index;
        QueryConfig config;
        float estimated_cost_ms;
        float estimated_recall;
        std::string explanation;
    };
    
    /** \brief Create planner for index manager. */
    explicit QueryPlanner(const IndexManager& manager);
    
    ~QueryPlanner();
    
    /** \brief Plan query execution.
     *
     * \param query Query vector [dim]
     * \param config Base query configuration
     * \return Optimized query plan
     */
    auto plan(const float* query, const QueryConfig& config) -> QueryPlan;
    
    /** \brief Update statistics from query execution.
     *
     * \param plan Executed plan
     * \param actual_time_ms Actual execution time
     * \param actual_recall Measured recall (if known)
     */
    auto update_stats(const QueryPlan& plan, float actual_time_ms, 
                     std::optional<float> actual_recall = {}) -> void;
    
    /** \brief Get planner statistics. */
    struct PlannerStats {
        std::uint64_t plans_generated{0};
        std::uint64_t plans_executed{0};
        float avg_estimation_error_ms{0.0f};
        float avg_recall_error{0.0f};
    };
    
    auto get_stats() const -> PlannerStats;
    
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vesper::index