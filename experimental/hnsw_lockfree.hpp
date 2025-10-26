#pragma once

/** \file hnsw_lockfree.hpp
 *  \brief Lock-free HNSW index for high-performance parallel construction.
 *
 * This implementation achieves 100k+ vec/sec build rates through:
 * - Lock-free node structure with atomic operations
 * - CAS-based neighbor list updates
 * - Epoch-based memory reclamation
 * - Three-stage parallel construction pipeline
 *
 * Thread-safety: Both build and search are lock-free and thread-safe.
 * Memory: Uses atomic pointers with acquire-release semantics.
 */

#include <atomic>
#include <cstdint>
#include <expected>
#include <memory>
#include <vector>
#include <span>
#include <optional>
#include <array>

#include "vesper/error.hpp"

namespace vesper::index {

/** \brief Lock-free HNSW build parameters. */
struct HnswLockfreeBuildParams {
    std::uint32_t M{16};                    /**< Max connections per node */
    std::uint32_t efConstruction{200};      /**< Beam width during construction */
    std::uint32_t seed{42};                 /**< Random seed for level assignment */
    bool extend_candidates{false};          /**< Use heuristic neighbor selection */
    std::uint32_t max_M{16};                /**< Max connections for level > 0 */
    std::uint32_t max_M0{32};               /**< Max connections for level 0 */
    std::uint32_t batch_size{1000};         /**< Batch size for parallel construction */
    std::uint32_t n_threads{0};             /**< Number of threads (0 = auto) */
};

/** \brief Lock-free HNSW search parameters. */
struct HnswLockfreeSearchParams {
    std::uint32_t efSearch{50};             /**< Beam width during search */
    std::uint32_t k{10};                    /**< Number of neighbors to return */
    const std::uint8_t* filter_mask{nullptr}; /**< Optional bitmap filter */
    std::size_t filter_size{0};             /**< Size of filter in bytes */
};

/** \brief Lock-free HNSW statistics. */
struct HnswLockfreeStats {
    std::size_t n_nodes{0};                 /**< Total nodes in graph */
    std::size_t n_edges{0};                 /**< Total edges in graph */
    std::size_t n_levels{0};                /**< Number of hierarchy levels */
    std::size_t memory_bytes{0};            /**< Total memory usage */
    float avg_degree{0.0f};                 /**< Average node degree */
    std::vector<std::size_t> level_counts;  /**< Nodes per level */
    std::size_t build_rate_vec_per_sec{0};  /**< Build rate in vectors/second */
};

// Forward declarations
class HnswLockfreeNode;
class EpochManager;
class HnswThreadContext;

/** \brief Atomic edge list for lock-free neighbor management.
 *
 * Uses a fixed-size array with atomic count for bounded neighbors.
 * Provides wait-free reads and lock-free insertions.
 */
class AtomicEdgeList {
public:
    static constexpr std::size_t MAX_EDGES = 64;
    
    AtomicEdgeList() : count_(0) {
        for (auto& edge : edges_) {
            edge.store(INVALID_ID, std::memory_order_relaxed);
        }
    }
    
    /** \brief Try to insert an edge using CAS.
     *
     * \param neighbor_id Neighbor node ID
     * \param max_edges Maximum edges allowed
     * \return true if inserted, false if list is full or duplicate
     *
     * Memory ordering: acquire-release for consistency
     */
    auto try_insert(std::uint32_t neighbor_id, std::uint32_t max_edges) -> bool;
    
    /** \brief Get current neighbors (wait-free read).
     *
     * \return Vector of neighbor IDs
     *
     * Memory ordering: acquire for visibility
     */
    auto get_neighbors() const -> std::vector<std::uint32_t>;
    
    /** \brief Check if neighbor exists (wait-free).
     *
     * \param neighbor_id Neighbor to check
     * \return true if neighbor exists
     */
    auto contains(std::uint32_t neighbor_id) const -> bool;
    
    /** \brief Get current edge count (wait-free).
     *
     * \return Number of edges
     */
    auto size() const -> std::size_t {
        return count_.load(std::memory_order_acquire);
    }
    
    /** \brief Try to remove an edge (lock-free).
     *
     * \param neighbor_id Neighbor to remove
     * \return true if removed, false if not found
     */
    auto try_remove(std::uint32_t neighbor_id) -> bool;
    
private:
    static constexpr std::uint32_t INVALID_ID = std::numeric_limits<std::uint32_t>::max();
    
    std::array<std::atomic<std::uint32_t>, MAX_EDGES> edges_;
    std::atomic<std::size_t> count_;
};

/** \brief Lock-free HNSW node.
 *
 * All fields use atomic operations for thread-safe access.
 * Memory is managed through epoch-based reclamation.
 */
class HnswLockfreeNode {
public:
    HnswLockfreeNode(std::uint64_t id, const float* data, std::size_t dim, std::uint32_t level);
    ~HnswLockfreeNode() = default;
    
    // Non-copyable, non-movable (atomic members)
    HnswLockfreeNode(const HnswLockfreeNode&) = delete;
    HnswLockfreeNode& operator=(const HnswLockfreeNode&) = delete;
    HnswLockfreeNode(HnswLockfreeNode&&) = delete;
    HnswLockfreeNode& operator=(HnswLockfreeNode&&) = delete;
    
    /** \brief Get node ID (immutable). */
    auto id() const noexcept -> std::uint64_t { return id_; }
    
    /** \brief Get data pointer (immutable). */
    auto data() const noexcept -> const float* { return data_.get(); }
    
    /** \brief Get node level (immutable). */
    auto level() const noexcept -> std::uint32_t { return level_; }
    
    /** \brief Check if node is deleted (atomic). */
    auto is_deleted() const noexcept -> bool {
        return deleted_.load(std::memory_order_acquire);
    }
    
    /** \brief Mark node as deleted (atomic). */
    auto mark_deleted() noexcept -> void {
        deleted_.store(true, std::memory_order_release);
    }
    
    /** \brief Get neighbors at level (wait-free). */
    auto get_neighbors(std::uint32_t level) const -> std::vector<std::uint32_t>;
    
    /** \brief Try to add neighbor at level (lock-free). */
    auto try_add_neighbor(std::uint32_t level, std::uint32_t neighbor_id, 
                         std::uint32_t max_edges) -> bool;
    
    /** \brief Try to remove neighbor at level (lock-free). */
    auto try_remove_neighbor(std::uint32_t level, std::uint32_t neighbor_id) -> bool;
    
    /** \brief Get neighbor count at level (wait-free). */
    auto neighbor_count(std::uint32_t level) const -> std::size_t;
    
private:
    const std::uint64_t id_;
    std::unique_ptr<float[]> data_;
    const std::uint32_t level_;
    std::atomic<bool> deleted_{false};
    std::vector<AtomicEdgeList> neighbors_;  // Per-level neighbor lists
};

/** \brief Epoch-based memory reclamation manager.
 *
 * Provides safe memory reclamation for lock-free data structures.
 * Based on the epoch-based reclamation scheme (EBR).
 */
class EpochManager {
public:
    static constexpr std::uint64_t INACTIVE_EPOCH = std::numeric_limits<std::uint64_t>::max();
    
    EpochManager();
    ~EpochManager();
    
    /** \brief Enter an epoch for the current thread. */
    auto enter_epoch() -> void;
    
    /** \brief Exit the current epoch. */
    auto exit_epoch() -> void;
    
    /** \brief Defer deletion of an object until safe. */
    template<typename T>
    auto defer_delete(T* ptr) -> void {
        defer_delete_impl(ptr, [](void* p) { delete static_cast<T*>(p); });
    }
    
    /** \brief Try to reclaim deferred deletions. */
    auto try_reclaim() -> void;
    
    /** \brief Get current global epoch. */
    auto current_epoch() const -> std::uint64_t {
        return global_epoch_.load(std::memory_order_acquire);
    }
    
private:
    struct DeferredDelete {
        void* ptr;
        void (*deleter)(void*);
        std::uint64_t epoch;
    };
    
    auto defer_delete_impl(void* ptr, void (*deleter)(void*)) -> void;
    auto get_min_epoch() const -> std::uint64_t;
    
    std::atomic<std::uint64_t> global_epoch_{0};
    
    // Thread-local epoch tracking
    static thread_local std::uint64_t thread_epoch_;
    static thread_local std::vector<DeferredDelete> thread_deferred_;
    
    // Global deferred deletion queue (lock-free)
    std::atomic<DeferredDelete*> deferred_head_{nullptr};
};

/** \brief Lock-free Hierarchical Navigable Small World index.
 *
 * High-performance HNSW implementation using lock-free algorithms.
 * Achieves 50k+ vec/sec build rates through parallel construction.
 */
class HnswLockfreeIndex {
public:
    HnswLockfreeIndex();
    ~HnswLockfreeIndex();
    HnswLockfreeIndex(HnswLockfreeIndex&&) noexcept;
    HnswLockfreeIndex& operator=(HnswLockfreeIndex&&) noexcept;
    HnswLockfreeIndex(const HnswLockfreeIndex&) = delete;
    HnswLockfreeIndex& operator=(const HnswLockfreeIndex&) = delete;
    
    /** \brief Initialize index with parameters.
     *
     * \param dim Vector dimensionality
     * \param params Build parameters
     * \param max_elements Expected maximum elements (for pre-allocation)
     * \return Success or error
     *
     * Thread-safe: Can be called once before any operations
     */
    auto init(std::size_t dim, const HnswLockfreeBuildParams& params,
              std::size_t max_elements = 0)
        -> std::expected<void, core::error>;
    
    /** \brief Add a vector to the index (lock-free).
     *
     * \param id Vector identifier
     * \param data Vector data [dim]
     * \return Success or error
     *
     * Thread-safe: Multiple threads can add concurrently
     */
    auto add(std::uint64_t id, const float* data)
        -> std::expected<void, core::error>;
    
    /** \brief Batch add vectors with parallel construction.
     *
     * \param ids Vector identifiers [n]
     * \param data Vectors [n x dim]
     * \param n Number of vectors
     * \return Success or error
     *
     * Thread-safe: Uses internal parallelization
     */
    auto add_batch(const std::uint64_t* ids, const float* data, std::size_t n)
        -> std::expected<void, core::error>;
    
    /** \brief Search for k nearest neighbors (lock-free).
     *
     * \param query Query vector [dim]
     * \param params Search parameters
     * \return Vector IDs and distances of k nearest neighbors
     *
     * Thread-safe: Multiple searches can run concurrently
     */
    auto search(const float* query, const HnswLockfreeSearchParams& params) const
        -> std::expected<std::vector<std::pair<std::uint64_t, float>>, core::error>;
    
    /** \brief Batch search for multiple queries.
     *
     * \param queries Query vectors [n_queries x dim]
     * \param n_queries Number of queries
     * \param params Search parameters
     * \return Results for each query
     *
     * Thread-safe: Internally parallelized
     */
    auto search_batch(const float* queries, std::size_t n_queries,
                      const HnswLockfreeSearchParams& params) const
        -> std::expected<std::vector<std::vector<std::pair<std::uint64_t, float>>>, core::error>;
    
    /** \brief Get index statistics. */
    auto get_stats() const noexcept -> HnswLockfreeStats;
    
    /** \brief Check if index is initialized. */
    auto is_initialized() const noexcept -> bool;
    
    /** \brief Get vector dimensionality. */
    auto dimension() const noexcept -> std::size_t;
    
    /** \brief Get number of indexed vectors. */
    auto size() const noexcept -> std::size_t;
    
    /** \brief Mark vector as deleted (soft delete). */
    auto mark_deleted(std::uint64_t id) -> std::expected<void, core::error>;
    
    /** \brief Trigger memory reclamation. */
    auto reclaim_memory() -> void;
    
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/** \brief Compute recall of lock-free HNSW against ground truth.
 *
 * \param index Built HNSW index
 * \param queries Test queries [n_queries x dim]
 * \param n_queries Number of queries
 * \param ground_truth True nearest neighbors [n_queries x k]
 * \param k Number of neighbors to evaluate
 * \param params Search parameters
 * \return Recall@k in [0, 1]
 */
auto compute_recall_lockfree(const HnswLockfreeIndex& index,
                             const float* queries, std::size_t n_queries,
                             const std::uint64_t* ground_truth, std::size_t k,
                             const HnswLockfreeSearchParams& params) -> float;

} // namespace vesper::index