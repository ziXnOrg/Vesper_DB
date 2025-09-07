#pragma once

/** \file numa_allocator.hpp
 *  \brief NUMA-aware memory allocation for optimized memory access patterns.
 *
 * Provides NUMA node detection, affinity management, and local allocation.
 * Features:
 * - Automatic NUMA topology discovery
 * - Thread-to-node affinity binding
 * - Local/interleaved allocation policies
 * - Memory migration between nodes
 * - Huge page support (2MB/1GB)
 *
 * Thread-safety: Thread-safe allocations with per-node arenas
 */

#include <vesper/core/error.hpp>
#include <vesper/span_polyfill.hpp>
#include <memory>
#include <vector>
#include <expected>
#include <cstddef>
#include <atomic>

#ifdef __linux__
#include <numa.h>
#include <numaif.h>
#endif

namespace vesper::memory {

/** \brief NUMA node information. */
struct NumaNode {
    std::uint32_t id;                    ///< Node ID
    std::size_t memory_size;              ///< Total memory in bytes
    std::size_t memory_free;              ///< Free memory in bytes
    std::vector<std::uint32_t> cpus;     ///< CPUs on this node
    std::vector<std::uint32_t> distances; ///< Distance to other nodes
};

/** \brief NUMA allocation policy. */
enum class NumaPolicy {
    LOCAL,       ///< Allocate from local node
    INTERLEAVE,  ///< Interleave across nodes
    PREFERRED,   ///< Prefer specific node, fallback allowed
    BIND         ///< Bind to specific nodes only
};

/** \brief NUMA-aware allocator configuration. */
struct NumaConfig {
    NumaPolicy policy = NumaPolicy::LOCAL;
    std::vector<std::uint32_t> nodes;     ///< Nodes for BIND policy
    bool use_huge_pages = false;          ///< Use 2MB huge pages
    bool use_1gb_pages = false;           ///< Use 1GB huge pages
    std::size_t alignment = 64;           ///< Memory alignment
};

/** \brief NUMA topology information. */
class NumaTopology {
public:
    /** \brief Detect and initialize NUMA topology. */
    [[nodiscard]] static auto detect() 
        -> std::expected<std::unique_ptr<NumaTopology>, core::error>;

    /** \brief Get number of NUMA nodes. */
    [[nodiscard]] auto num_nodes() const noexcept -> std::uint32_t {
        return static_cast<std::uint32_t>(nodes_.size());
    }

    /** \brief Get node information. */
    [[nodiscard]] auto get_node(std::uint32_t id) const noexcept -> const NumaNode* {
        if (id >= nodes_.size()) return nullptr;
        return &nodes_[id];
    }

    /** \brief Get current thread's NUMA node. */
    [[nodiscard]] auto current_node() const noexcept -> std::uint32_t;

    /** \brief Get distance between nodes. */
    [[nodiscard]] auto distance(std::uint32_t from, std::uint32_t to) const noexcept
        -> std::uint32_t;

    /** \brief Check if system has NUMA support. */
    [[nodiscard]] auto is_numa_available() const noexcept -> bool {
        return numa_available_;
    }

    /** \brief Get total system memory. */
    [[nodiscard]] auto total_memory() const noexcept -> std::size_t {
        return total_memory_;
    }

private:
    NumaTopology() = default;
    [[nodiscard]] auto init() -> std::expected<void, core::error>;

    std::vector<NumaNode> nodes_;
    std::size_t total_memory_ = 0;
    bool numa_available_ = false;
};

/** \brief NUMA-aware memory allocator. */
class NumaAllocator {
public:
    /** \brief Create allocator with configuration. */
    [[nodiscard]] static auto create(const NumaConfig& config = {})
        -> std::expected<std::unique_ptr<NumaAllocator>, core::error>;

    ~NumaAllocator();

    // Non-copyable, movable
    NumaAllocator(const NumaAllocator&) = delete;
    NumaAllocator& operator=(const NumaAllocator&) = delete;
    NumaAllocator(NumaAllocator&&) noexcept;
    NumaAllocator& operator=(NumaAllocator&&) noexcept;

    /** \brief Allocate memory according to policy.
     *
     * \param size Size in bytes
     * \return Allocated memory or error
     */
    [[nodiscard]] auto allocate(std::size_t size)
        -> std::expected<void*, core::error>;

    /** \brief Allocate aligned memory.
     *
     * \param size Size in bytes
     * \param alignment Alignment requirement
     * \return Allocated memory or error
     */
    [[nodiscard]] auto allocate_aligned(std::size_t size, std::size_t alignment)
        -> std::expected<void*, core::error>;

    /** \brief Allocate on specific NUMA node.
     *
     * \param size Size in bytes
     * \param node Node ID
     * \return Allocated memory or error
     */
    [[nodiscard]] auto allocate_on_node(std::size_t size, std::uint32_t node)
        -> std::expected<void*, core::error>;

    /** \brief Deallocate memory.
     *
     * \param ptr Memory to free
     * \param size Original allocation size
     */
    auto deallocate(void* ptr, std::size_t size) noexcept -> void;

    /** \brief Migrate memory to different node.
     *
     * \param ptr Memory to migrate
     * \param size Size in bytes
     * \param node Target node
     * \return Error if migration fails
     */
    [[nodiscard]] auto migrate(void* ptr, std::size_t size, std::uint32_t node)
        -> std::expected<void, core::error>;

    /** \brief Touch memory to fault pages on current node.
     *
     * \param ptr Memory to touch
     * \param size Size in bytes
     */
    auto touch_pages(void* ptr, std::size_t size) noexcept -> void;

    /** \brief Get allocation statistics. */
    struct Stats {
        std::size_t total_allocated = 0;
        std::size_t total_deallocated = 0;
        std::size_t current_usage = 0;
        std::size_t peak_usage = 0;
        std::vector<std::size_t> per_node_usage;
    };

    [[nodiscard]] auto get_stats() const noexcept -> Stats;

private:
    explicit NumaAllocator(const NumaConfig& config);
    [[nodiscard]] auto init() -> std::expected<void, core::error>;

    NumaConfig config_;
    std::unique_ptr<NumaTopology> topology_;
    
    // Statistics
    std::atomic<std::size_t> total_allocated_{0};
    std::atomic<std::size_t> total_deallocated_{0};
    std::atomic<std::size_t> current_usage_{0};
    std::atomic<std::size_t> peak_usage_{0};
    std::vector<std::atomic<std::size_t>> per_node_usage_;
};

/** \brief STL-compatible NUMA allocator. */
template <typename T>
class StlNumaAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template <typename U>
    struct rebind {
        using other = StlNumaAllocator<U>;
    };

    StlNumaAllocator() = default;
    
    explicit StlNumaAllocator(NumaAllocator* allocator)
        : allocator_(allocator) {}

    template <typename U>
    StlNumaAllocator(const StlNumaAllocator<U>& other) noexcept
        : allocator_(other.allocator_) {}

    [[nodiscard]] T* allocate(size_type n) {
        if (!allocator_) {
            return static_cast<T*>(::operator new(n * sizeof(T)));
        }
        
        auto result = allocator_->allocate(n * sizeof(T));
        if (!result) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(result.value());
    }

    void deallocate(T* ptr, size_type n) noexcept {
        if (!allocator_) {
            ::operator delete(ptr);
        } else {
            allocator_->deallocate(ptr, n * sizeof(T));
        }
    }

    template <typename U>
    bool operator==(const StlNumaAllocator<U>& other) const noexcept {
        return allocator_ == other.allocator_;
    }

    template <typename U>
    bool operator!=(const StlNumaAllocator<U>& other) const noexcept {
        return !(*this == other);
    }

private:
    template <typename U>
    friend class StlNumaAllocator;
    
    NumaAllocator* allocator_ = nullptr;
};

/** \brief Thread affinity management. */
class ThreadAffinity {
public:
    /** \brief Bind current thread to NUMA node. */
    [[nodiscard]] static auto bind_to_node(std::uint32_t node)
        -> std::expected<void, core::error>;

    /** \brief Bind current thread to specific CPUs. */
    [[nodiscard]] static auto bind_to_cpus(std::span<const std::uint32_t> cpus)
        -> std::expected<void, core::error>;

    /** \brief Get current thread's CPU. */
    [[nodiscard]] static auto current_cpu() noexcept -> std::uint32_t;

    /** \brief Reset thread affinity to default. */
    static auto reset() noexcept -> void;
};

/** \brief Global NUMA allocator instance. */
class NumaAllocatorPool {
public:
    /** \brief Get or create per-node allocator. */
    [[nodiscard]] static auto get_for_node(std::uint32_t node,
                                           const NumaConfig& config = {})
        -> std::expected<NumaAllocator*, core::error>;

    /** \brief Get allocator for current thread's node. */
    [[nodiscard]] static auto get_local(const NumaConfig& config = {})
        -> std::expected<NumaAllocator*, core::error>;

    /** \brief Reset all allocators. */
    static auto reset() noexcept -> void;

private:
    static std::vector<std::unique_ptr<NumaAllocator>> allocators_;
    static std::mutex mutex_;
};

} // namespace vesper::memory