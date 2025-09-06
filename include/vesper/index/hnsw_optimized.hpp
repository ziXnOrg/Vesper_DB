#pragma once

/** \file hnsw_optimized.hpp
 *  \brief Cache-optimized HNSW data structures.
 *
 * Optimizations:
 * - Separate hot/cold data paths
 * - Cache-line aligned structures
 * - Contiguous memory for vectors
 * - Compressed neighbor storage
 */

#include <cstdint>
#include <vector>
#include <atomic>
#include <memory>

namespace vesper::index {

/** \brief Cache-line size (64 bytes on x86). */
static constexpr std::size_t kCacheLineSize = 64;

/** \brief Align to cache line boundary. */
template<typename T>
using CacheAligned = alignas(kCacheLineSize) T;

/** \brief Compressed neighbor list with fixed capacity. 
 *
 * Uses array instead of vector to avoid indirection.
 * Stores count separately to avoid checking all slots.
 */
struct CompactNeighborList {
    static constexpr std::uint32_t kMaxNeighbors = 32;
    
    std::uint32_t count{0};
    std::uint32_t neighbors[kMaxNeighbors];
    
    void add(std::uint32_t neighbor) {
        if (count < kMaxNeighbors) {
            neighbors[count++] = neighbor;
        }
    }
    
    void remove(std::uint32_t neighbor) {
        for (std::uint32_t i = 0; i < count; ++i) {
            if (neighbors[i] == neighbor) {
                // Swap with last and decrement count
                neighbors[i] = neighbors[--count];
                break;
            }
        }
    }
    
    bool contains(std::uint32_t neighbor) const {
        for (std::uint32_t i = 0; i < count; ++i) {
            if (neighbors[i] == neighbor) return true;
        }
        return false;
    }
};

/** \brief Hot data for HNSW node (frequently accessed during search).
 *
 * Packed into minimal cache lines for search operations.
 * Size: 16 bytes (fits 4 nodes per cache line).
 */
struct alignas(16) HnswNodeHot {
    std::uint32_t level;           // 4 bytes
    std::uint32_t base_neighbors;  // 4 bytes - offset into neighbor storage
    std::uint32_t upper_neighbors; // 4 bytes - offset for level > 0
    std::uint32_t data_offset;     // 4 bytes - offset into vector storage
};

/** \brief Cold data for HNSW node (rarely accessed).
 *
 * Separated from hot path to improve cache utilization.
 */
struct HnswNodeCold {
    std::uint64_t id;
    std::atomic<bool> deleted{false};
};

/** \brief Optimized HNSW graph storage.
 *
 * Uses structure-of-arrays (SoA) layout for better cache performance.
 */
class OptimizedHnswGraph {
public:
    /** \brief Initialize graph with capacity. */
    void init(std::size_t max_nodes, std::size_t dim) {
        hot_nodes_.reserve(max_nodes);
        cold_nodes_.reserve(max_nodes);
        
        // Pre-allocate contiguous vector storage
        vector_data_.reserve(max_nodes * dim);
        
        // Pre-allocate neighbor storage
        neighbor_lists_.reserve(max_nodes * 2);  // Assume avg 2 levels per node
        
        dimension_ = dim;
    }
    
    /** \brief Add node to graph. */
    std::uint32_t add_node(std::uint64_t id, const float* data, std::uint32_t level) {
        std::uint32_t idx = static_cast<std::uint32_t>(hot_nodes_.size());
        
        // Store vector data contiguously
        std::uint32_t data_offset = static_cast<std::uint32_t>(vector_data_.size());
        vector_data_.insert(vector_data_.end(), data, data + dimension_);
        
        // Allocate neighbor lists
        std::uint32_t base_neighbors = static_cast<std::uint32_t>(neighbor_lists_.size());
        neighbor_lists_.emplace_back();  // Level 0
        
        std::uint32_t upper_neighbors = base_neighbors;
        if (level > 0) {
            upper_neighbors = static_cast<std::uint32_t>(neighbor_lists_.size());
            for (std::uint32_t l = 1; l <= level; ++l) {
                neighbor_lists_.emplace_back();
            }
        }
        
        // Add hot node
        hot_nodes_.push_back({level, base_neighbors, upper_neighbors, data_offset});
        
        // Add cold node
        cold_nodes_.push_back({id, false});
        
        return idx;
    }
    
    /** \brief Get vector data for node. */
    const float* get_vector(std::uint32_t idx) const {
        return &vector_data_[hot_nodes_[idx].data_offset];
    }
    
    /** \brief Get neighbors at level. */
    CompactNeighborList& get_neighbors(std::uint32_t idx, std::uint32_t level) {
        const auto& node = hot_nodes_[idx];
        if (level == 0) {
            return neighbor_lists_[node.base_neighbors];
        } else {
            return neighbor_lists_[node.upper_neighbors + level - 1];
        }
    }
    
    const CompactNeighborList& get_neighbors(std::uint32_t idx, std::uint32_t level) const {
        const auto& node = hot_nodes_[idx];
        if (level == 0) {
            return neighbor_lists_[node.base_neighbors];
        } else {
            return neighbor_lists_[node.upper_neighbors + level - 1];
        }
    }
    
    /** \brief Get node level. */
    std::uint32_t get_level(std::uint32_t idx) const {
        return hot_nodes_[idx].level;
    }
    
    /** \brief Get node ID. */
    std::uint64_t get_id(std::uint32_t idx) const {
        return cold_nodes_[idx].id;
    }
    
    /** \brief Check if node is deleted. */
    bool is_deleted(std::uint32_t idx) const {
        return cold_nodes_[idx].deleted.load();
    }
    
    /** \brief Mark node as deleted. */
    void mark_deleted(std::uint32_t idx) {
        cold_nodes_[idx].deleted.store(true);
    }
    
    /** \brief Get number of nodes. */
    std::size_t size() const { return hot_nodes_.size(); }
    
    /** \brief Get dimension. */
    std::size_t dimension() const { return dimension_; }

private:
    // Hot data - accessed frequently during search
    CacheAligned<std::vector<HnswNodeHot>> hot_nodes_;
    
    // Cold data - rarely accessed
    std::vector<HnswNodeCold> cold_nodes_;
    
    // Contiguous vector storage for better cache locality
    std::vector<float> vector_data_;
    
    // Neighbor lists storage
    std::vector<CompactNeighborList> neighbor_lists_;
    
    std::size_t dimension_{0};
};

/** \brief Prefetch hints for optimized traversal. */
inline void prefetch_node(const OptimizedHnswGraph& graph, std::uint32_t idx) {
#ifdef _MSC_VER
    // Prefetch vector data
    _mm_prefetch(reinterpret_cast<const char*>(graph.get_vector(idx)), _MM_HINT_T0);
    // Prefetch neighbor list
    _mm_prefetch(reinterpret_cast<const char*>(&graph.get_neighbors(idx, 0)), _MM_HINT_T1);
#elif defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(graph.get_vector(idx), 0, 3);
    __builtin_prefetch(&graph.get_neighbors(idx, 0), 0, 2);
#endif
}

} // namespace vesper::index