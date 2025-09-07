/** \file tombstone_manager.hpp
 *  \brief Manages deleted vectors with tombstone pattern for soft deletion
 */

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <expected>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_set>
#include <vector>

#include "vesper/error.hpp"
#include "roaring.hh"

namespace vesper::tombstone {

/**
 * \brief Tombstone statistics
 */
struct TombstoneStats {
    std::atomic<std::uint64_t> total_deleted{0};
    std::atomic<std::uint64_t> total_restored{0};
    std::atomic<std::uint64_t> compaction_count{0};
    std::atomic<std::uint64_t> memory_bytes{0};
    
    TombstoneStats() = default;
    
    // Copy constructor for atomic members
    TombstoneStats(const TombstoneStats& other) 
        : total_deleted(other.total_deleted.load())
        , total_restored(other.total_restored.load())
        , compaction_count(other.compaction_count.load())
        , memory_bytes(other.memory_bytes.load()) {}
    
    // Copy assignment
    TombstoneStats& operator=(const TombstoneStats& other) {
        if (this != &other) {
            total_deleted.store(other.total_deleted.load());
            total_restored.store(other.total_restored.load());
            compaction_count.store(other.compaction_count.load());
            memory_bytes.store(other.memory_bytes.load());
        }
        return *this;
    }
    
    [[nodiscard]] double deletion_ratio(std::uint64_t total_vectors) const {
        return total_vectors > 0 ? 
            static_cast<double>(total_deleted.load()) / total_vectors : 0.0;
    }
};

/**
 * \brief Configuration for tombstone management
 */
struct TombstoneConfig {
    std::size_t compaction_threshold{10000};  // Compact after this many deletions
    double compaction_ratio{0.2};  // Compact when 20% of vectors are deleted
    bool use_compression{true};  // Use Roaring bitmap compression
    std::chrono::seconds ttl{0};  // Time-to-live for tombstones (0 = infinite)
};

/**
 * \brief Manages tombstones for deleted vectors
 */
class TombstoneManager {
public:
    explicit TombstoneManager(TombstoneConfig config = {});
    ~TombstoneManager();
    
    /**
     * \brief Mark a vector as deleted
     */
    auto mark_deleted(std::uint32_t vector_id) -> std::expected<void, core::error>;
    
    /**
     * \brief Mark multiple vectors as deleted
     */
    auto mark_deleted_batch(const std::vector<std::uint32_t>& vector_ids) 
        -> std::expected<void, core::error>;
    
    /**
     * \brief Restore a deleted vector
     */
    auto restore(std::uint32_t vector_id) -> std::expected<void, core::error>;
    
    /**
     * \brief Check if a vector is deleted
     */
    [[nodiscard]] auto is_deleted(std::uint32_t vector_id) const -> bool;
    
    /**
     * \brief Get all deleted vector IDs
     */
    [[nodiscard]] auto get_deleted_ids() const -> std::vector<std::uint32_t>;
    
    /**
     * \brief Count of deleted vectors
     */
    [[nodiscard]] auto deleted_count() const -> std::uint64_t;
    
    /**
     * \brief Clear all tombstones
     */
    auto clear() -> void;
    
    /**
     * \brief Compact tombstones (optimize storage)
     */
    auto compact() -> std::expected<void, core::error>;
    
    /**
     * \brief Check if compaction is needed
     */
    [[nodiscard]] auto needs_compaction(std::uint64_t total_vectors) const -> bool;
    
    /**
     * \brief Save tombstones to disk
     */
    auto save(const std::string& path) const -> std::expected<void, core::error>;
    
    /**
     * \brief Load tombstones from disk
     */
    auto load(const std::string& path) -> std::expected<void, core::error>;
    
    /**
     * \brief Get memory usage in bytes
     */
    [[nodiscard]] auto memory_usage() const -> std::size_t;
    
    /**
     * \brief Get statistics
     */
    [[nodiscard]] auto get_stats() const -> TombstoneStats;
    
    /**
     * \brief Get oversampling factor to compensate for filtered results
     */
    [[nodiscard]] auto get_oversampling_factor() const -> float;
    
    /**
     * \brief Filter out deleted IDs from a result set
     */
    template<typename Container>
    auto filter_results(const Container& results) const -> Container {
        Container filtered;
        filtered.reserve(results.size());
        
        std::shared_lock lock(mutex_);
        for (const auto& item : results) {
            std::uint32_t id;
            if constexpr (std::is_integral_v<std::decay_t<decltype(item)>>) {
                id = static_cast<std::uint32_t>(item);
            } else {
                // Assume it's a pair<score, id>
                id = item.second;
            }
            
            if (!deleted_bitmap_->contains(id)) {
                filtered.push_back(item);
            }
        }
        
        return filtered;
    }
    
private:
    TombstoneConfig config_;
    mutable std::shared_mutex mutex_;
    
    // Roaring bitmap for efficient storage of deleted IDs
    std::unique_ptr<roaring::Roaring> deleted_bitmap_;
    
    // Timestamp tracking for TTL support
    struct DeletionRecord {
        std::uint32_t id;
        std::chrono::steady_clock::time_point timestamp;
    };
    std::vector<DeletionRecord> deletion_timeline_;
    
    // Statistics
    mutable TombstoneStats stats_;
    
    // Helper methods
    void update_memory_usage();
    void expire_old_tombstones();
};

/**
 * \brief Thread-safe tombstone filter for query results
 */
class TombstoneFilter {
public:
    explicit TombstoneFilter(std::shared_ptr<TombstoneManager> manager)
        : manager_(std::move(manager)) {}
    
    /**
     * \brief Filter search results to exclude deleted vectors
     */
    template<typename T>
    auto filter(std::vector<std::pair<T, std::uint32_t>>& results) const -> void {
        if (!manager_) return;
        
        auto end = std::remove_if(results.begin(), results.end(),
            [this](const auto& result) {
                return manager_->is_deleted(result.second);
            });
        
        results.erase(end, results.end());
    }
    
    /**
     * \brief Get extra results to compensate for filtered vectors
     */
    [[nodiscard]] auto get_oversampling_factor() const -> float {
        if (!manager_) return 1.0f;
        
        auto stats = manager_->get_stats();
        auto deletion_ratio = stats.deletion_ratio(100000); // Approximate
        
        // If 20% deleted, oversample by 1.25x
        return 1.0f + deletion_ratio * 1.25f;
    }
    
private:
    std::shared_ptr<TombstoneManager> manager_;
};

} // namespace vesper::tombstone