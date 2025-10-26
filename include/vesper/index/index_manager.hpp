#pragma once

/** \file index_manager.hpp
 *  \brief Unified index management for multiple index types.
 *
 * Serves as the central component that manages the three-index architecture:
 * - HNSW for hot in-memory segments
 * - IVF-PQ for compact retrieval
 * - DiskANN for billion-scale SSD-resident search
 *
 * Serves as a unified API for index selection, query planning, and lifecycle management.
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

    // Quantization options
    bool enable_rabitq{false};           /**< Use RaBitQ quantization */
    std::uint8_t quantization_bits{1};   /**< Bits for quantization (1, 4, or 8) */
    bool enable_matryoshka{false};       /**< Use Matryoshka embeddings */
    std::vector<std::uint32_t> matryoshka_dims; /**< Matryoshka dimension levels */

    // Resource limits
    std::size_t memory_budget_mb{1024};  /**< Memory budget in MB */
    std::size_t cache_size_mb{256};      /**< Cache size for disk indexes */

    // Optimization hints
    bool optimize_for_recall{true};      /**< Prioritize recall over speed */
    float target_recall{0.95f};          /**< Target recall@k */
    bool use_parallel_construction{true}; /**< Use parallel index building */
};

/** \brief Query configuration for index manager. */
struct QueryConfig {
    std::uint32_t k{10};                 /**< Number of results */
    float epsilon{0.0f};                 /**< Range query epsilon */

    // Index-specific parameters
    std::uint32_t ef_search{100};        /**< HNSW search parameter (default raised for better recall) */
    std::uint32_t nprobe{8};             /**< IVF probing depth */
    std::uint32_t l_search{128};         /**< DiskANN search list size */

    // Rerank controls (IVF-PQ)
    bool use_exact_rerank{false};        /**< Recompute exact distances on shortlist if raw vectors available */
    std::uint32_t rerank_k{0};           /**< Shortlist size for rerank (0=auto -> cand heuristic) */
    // Adaptive shortlist sizing (when rerank_k==0)
    float rerank_alpha{2.0f};            /**< Heuristic multiplier for cand_k: alpha * k * log2(1+nprobe) */
    std::uint32_t rerank_cand_ceiling{2000}; /**< Hard ceiling for cand_k (0 = no cap) */

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
    /** \brief Get current memory budget in MB. */
    auto get_memory_budget() const -> std::size_t;

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

#ifdef VESPER_ENABLE_TESTS
    // Debug-only accessor for tests: expose IVF-PQ index (read-only)
    auto ivf_pq_index_debug() const -> const IvfPqIndex*;
    // Returns the effective query config applied by the most recent search() call
    auto last_applied_query_config_debug() const -> QueryConfig;

#endif

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
 *
 * Determinism and tie-breaking:
 * - Index selection order is stable and deterministic. When multiple indexes are active,
 *   the planner uses the first active index returned by IndexManager in stable enum order:
 *   HNSW > IVF_PQ > DiskANN. This guarantees reproducible plan() results given the same
 *   IndexManager state and QueryConfig.
 *
 * Determinism “frozen mode” (H3):
 * - Environment variable: VESPER_PLANNER_FROZEN
 *   - Set to "1" to enable frozen mode; any other value (or unset) leaves it disabled.
 *   - The flag is read once at planner construction (via vesper::core::safe_getenv) and
 *     cached as a const member to avoid data races and per-call overhead.
 * - Behavior when frozen:
 *   - plan(): uses only deterministic, fixed logic; no adaptive model/history is read.
 *   - update_stats(): increments counters but skips all adaptive aggregate updates.
 *   - get_stats(): counters continue to reflect usage; adaptive aggregates remain unchanged.
 * - Intended use: testing, debugging, and reproducible benchmarking where adaptive tuning
 *   must be disabled while preserving deterministic plan generation.
 *
 * Thread-safety:
 * - plan() is thread-safe for concurrent calls (read-only access under shared lock; counters via atomics).
 * - update_stats() is thread-safe and may run concurrently with plan() (exclusive lock for writes when not frozen).
 *   In frozen mode, update_stats() performs an early return after incrementing counters and does not acquire the
 *   state mutex since it makes no shared-state modifications.
 * - get_stats() is thread-safe (atomics for counters; shared lock for aggregates).
 * - Synchronization strategy: shared_mutex protects cost model, performance history, and aggregate totals; simple
 *   counters use std::atomic with acquire/release semantics.
 *
 * Example (enable frozen mode for reproducibility):
 * \code{.cpp}
 * // On Windows PowerShell:
 * //   $env:VESPER_PLANNER_FROZEN = "1"
 * // On POSIX shells:
 * //   export VESPER_PLANNER_FROZEN=1
 * vesper::index::QueryPlanner planner(manager);
 * auto plan = planner.plan(query, cfg);
 * // plan/config are deterministic; update_stats() will not change adaptive aggregates
 * \endcode
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
     * Thread-safe: yes. May be called concurrently from multiple threads.
     * Acquires shared lock for reading planner state; increments counters atomically.
     *
     * \param query Query vector [dim]
     * \param config Base query configuration
     * \return Optimized query plan
     */
    auto plan(const float* query, const QueryConfig& config) -> QueryPlan;

    /** \brief Update statistics from query execution.
     *
     * Thread-safe: yes. May run concurrently with plan(); acquires exclusive lock
     * to update history, cost model, and aggregate totals; increments counters atomically.
     *
     * \param plan Executed plan
     * \param actual_time_ms Actual execution time
     * \param actual_recall Measured recall (if known)
     */
    auto update_stats(const QueryPlan& plan, float actual_time_ms,
                     std::optional<float> actual_recall = {}) -> void;

    /** \brief Get planner statistics.
     *
     * Thread-safe: yes. Reads counters atomically and aggregates under shared lock.
     */
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
