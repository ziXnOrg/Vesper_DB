/** \file tombstone_manager.cpp
 *  \brief Implementation of tombstone management for soft deletion
 */

#include "vesper/tombstone/tombstone_manager.hpp"
#include <fstream>
#include <algorithm>

// For std::unexpected workaround - MSVC C++20 doesn't have std::unexpect
#include <expected>

// Use the project's vesper_unexpected workaround

namespace vesper::tombstone {

TombstoneManager::TombstoneManager(TombstoneConfig config)
    : config_(config)
    , deleted_bitmap_(std::make_unique<roaring::Roaring>()) {
    
    if (config_.use_compression) {
        deleted_bitmap_->runOptimize();
    }
}

TombstoneManager::~TombstoneManager() = default;

auto TombstoneManager::mark_deleted(std::uint32_t vector_id) 
    -> std::expected<void, core::error> {
    
    std::unique_lock lock(mutex_);
    
    if (deleted_bitmap_->contains(vector_id)) {
        return std::vesper_unexpected(core::error{
            core::error_code::precondition_failed,
            "Vector already marked as deleted",
            "tombstone.mark_deleted"
        });
    }
    
    deleted_bitmap_->add(vector_id);
    
    // Track deletion time if TTL is enabled
    if (config_.ttl.count() > 0) {
        deletion_timeline_.push_back({
            vector_id,
            std::chrono::steady_clock::now()
        });
    }
    
    stats_.total_deleted.fetch_add(1);
    update_memory_usage();
    
    // Check if compaction is needed
    if (config_.use_compression && 
        deleted_bitmap_->cardinality() % config_.compaction_threshold == 0) {
        deleted_bitmap_->runOptimize();
        update_memory_usage();
    }
    
    return {};
}

auto TombstoneManager::mark_deleted_batch(const std::vector<std::uint32_t>& vector_ids)
    -> std::expected<void, core::error> {
    
    if (vector_ids.empty()) {
        return {};
    }
    
    std::unique_lock lock(mutex_);
    
    auto now = std::chrono::steady_clock::now();
    std::uint64_t added_count = 0;
    
    for (std::uint32_t id : vector_ids) {
        if (!deleted_bitmap_->contains(id)) {
            deleted_bitmap_->add(id);
            added_count++;
            
            if (config_.ttl.count() > 0) {
                deletion_timeline_.push_back({id, now});
            }
        }
    }
    
    stats_.total_deleted.fetch_add(added_count);
    
    // Optimize after batch operation
    if (config_.use_compression) {
        deleted_bitmap_->runOptimize();
    }
    
    update_memory_usage();
    
    return {};
}

auto TombstoneManager::restore(std::uint32_t vector_id) 
    -> std::expected<void, core::error> {
    
    std::unique_lock lock(mutex_);
    
    if (!deleted_bitmap_->contains(vector_id)) {
        return std::vesper_unexpected(core::error{
            core::error_code::not_found,
            "Vector not marked as deleted",
            "tombstone.restore"
        });
    }
    
    deleted_bitmap_->remove(vector_id);
    
    // Remove from timeline if TTL is enabled
    if (config_.ttl.count() > 0) {
        auto it = std::remove_if(deletion_timeline_.begin(), deletion_timeline_.end(),
            [vector_id](const auto& record) {
                return record.id == vector_id;
            });
        deletion_timeline_.erase(it, deletion_timeline_.end());
    }
    
    stats_.total_restored.fetch_add(1);
    update_memory_usage();
    
    return {};
}

auto TombstoneManager::is_deleted(std::uint32_t vector_id) const -> bool {
    std::shared_lock lock(mutex_);
    return deleted_bitmap_->contains(vector_id);
}

auto TombstoneManager::get_deleted_ids() const -> std::vector<std::uint32_t> {
    std::shared_lock lock(mutex_);
    
    std::vector<std::uint32_t> ids;
    ids.reserve(deleted_bitmap_->cardinality());
    
    for (auto it = deleted_bitmap_->begin(); it != deleted_bitmap_->end(); ++it) {
        ids.push_back(*it);
    }
    
    return ids;
}

auto TombstoneManager::deleted_count() const -> std::uint64_t {
    std::shared_lock lock(mutex_);
    return deleted_bitmap_->cardinality();
}

auto TombstoneManager::clear() -> void {
    std::unique_lock lock(mutex_);
    deleted_bitmap_ = std::make_unique<roaring::Roaring>();  // Reset to empty bitmap
    deletion_timeline_.clear();
    update_memory_usage();
}

auto TombstoneManager::compact() -> std::expected<void, core::error> {
    std::unique_lock lock(mutex_);
    
    // Expire old tombstones if TTL is enabled
    if (config_.ttl.count() > 0) {
        expire_old_tombstones();
    }
    
    // Optimize bitmap storage
    deleted_bitmap_->runOptimize();
    deleted_bitmap_->shrinkToFit();
    
    // Shrink timeline vector
    deletion_timeline_.shrink_to_fit();
    
    stats_.compaction_count.fetch_add(1);
    update_memory_usage();
    
    return {};
}

auto TombstoneManager::needs_compaction(std::uint64_t total_vectors) const -> bool {
    if (total_vectors == 0) return false;
    
    std::shared_lock lock(mutex_);
    
    auto deleted = deleted_bitmap_->cardinality();
    
    // Check absolute threshold
    if (deleted >= config_.compaction_threshold) {
        return true;
    }
    
    // Check ratio threshold
    double ratio = static_cast<double>(deleted) / total_vectors;
    return ratio >= config_.compaction_ratio;
}

auto TombstoneManager::save(const std::string& path) const 
    -> std::expected<void, core::error> {
    
    std::shared_lock lock(mutex_);
    
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        return std::vesper_unexpected(core::error{
            core::error_code::io_failed,
            "Failed to open file for writing",
            "tombstone.save"
        });
    }
    
    // Write header
    std::uint32_t version = 1;
    file.write(reinterpret_cast<const char*>(&version), sizeof(version));
    
    // Serialize bitmap
    std::size_t bitmap_size = deleted_bitmap_->getSizeInBytes();
    file.write(reinterpret_cast<const char*>(&bitmap_size), sizeof(bitmap_size));
    
    std::vector<char> buffer(bitmap_size);
    deleted_bitmap_->write(buffer.data());
    file.write(buffer.data(), bitmap_size);
    
    // Write timeline if TTL is enabled
    if (config_.ttl.count() > 0) {
        std::uint64_t timeline_size = deletion_timeline_.size();
        file.write(reinterpret_cast<const char*>(&timeline_size), sizeof(timeline_size));
        
        for (const auto& record : deletion_timeline_) {
            file.write(reinterpret_cast<const char*>(&record.id), sizeof(record.id));
            auto time_since_epoch = record.timestamp.time_since_epoch().count();
            file.write(reinterpret_cast<const char*>(&time_since_epoch), 
                      sizeof(time_since_epoch));
        }
    }
    
    return {};
}

auto TombstoneManager::load(const std::string& path) 
    -> std::expected<void, core::error> {
    
    std::unique_lock lock(mutex_);
    
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return std::vesper_unexpected(core::error{
            core::error_code::io_failed,
            "Failed to open file for reading",
            "tombstone.load"
        });
    }
    
    // Read header
    std::uint32_t version;
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    
    if (version != 1) {
        return std::vesper_unexpected(core::error{
            core::error_code::data_integrity,
            "Unsupported tombstone file version",
            "tombstone.load"
        });
    }
    
    // Read bitmap
    std::size_t bitmap_size;
    file.read(reinterpret_cast<char*>(&bitmap_size), sizeof(bitmap_size));
    
    std::vector<char> buffer(bitmap_size);
    file.read(buffer.data(), bitmap_size);
    
    deleted_bitmap_ = std::make_unique<roaring::Roaring>(
        roaring::Roaring::read(buffer.data()));
    
    // Read timeline if present
    if (config_.ttl.count() > 0 && file.peek() != EOF) {
        std::uint64_t timeline_size;
        file.read(reinterpret_cast<char*>(&timeline_size), sizeof(timeline_size));
        
        deletion_timeline_.clear();
        deletion_timeline_.reserve(timeline_size);
        
        for (std::uint64_t i = 0; i < timeline_size; ++i) {
            DeletionRecord record;
            file.read(reinterpret_cast<char*>(&record.id), sizeof(record.id));
            
            std::chrono::steady_clock::duration::rep time_since_epoch;
            file.read(reinterpret_cast<char*>(&time_since_epoch), 
                     sizeof(time_since_epoch));
            
            record.timestamp = std::chrono::steady_clock::time_point(
                std::chrono::steady_clock::duration(time_since_epoch));
            
            deletion_timeline_.push_back(record);
        }
    }
    
    update_memory_usage();
    
    return {};
}

auto TombstoneManager::memory_usage() const -> std::size_t {
    std::shared_lock lock(mutex_);
    return stats_.memory_bytes.load();
}

auto TombstoneManager::get_stats() const -> TombstoneStats {
    return stats_;
}

void TombstoneManager::update_memory_usage() {
    std::size_t bytes = deleted_bitmap_->getSizeInBytes();
    bytes += deletion_timeline_.size() * sizeof(DeletionRecord);
    bytes += sizeof(*this);
    
    stats_.memory_bytes.store(bytes);
}

void TombstoneManager::expire_old_tombstones() {
    if (config_.ttl.count() <= 0) return;
    
    auto now = std::chrono::steady_clock::now();
    auto expiry_time = now - config_.ttl;
    
    // Find expired entries
    std::vector<std::uint32_t> expired_ids;
    auto it = std::remove_if(deletion_timeline_.begin(), deletion_timeline_.end(),
        [&expired_ids, expiry_time](const auto& record) {
            if (record.timestamp < expiry_time) {
                expired_ids.push_back(record.id);
                return true;
            }
            return false;
        });
    
    // Remove expired entries from timeline
    deletion_timeline_.erase(it, deletion_timeline_.end());
    
    // Remove from bitmap
    for (std::uint32_t id : expired_ids) {
        deleted_bitmap_->remove(id);
    }
}

auto TombstoneManager::get_oversampling_factor() const -> float {
    auto stats = get_stats();
    auto deletion_ratio = stats.deletion_ratio(100000); // Approximate
    
    // If 20% deleted, oversample by 1.25x
    return 1.0f + deletion_ratio * 1.25f;
}

} // namespace vesper::tombstone