/** \file lru_cache.hpp
 *  \brief Thread-safe LRU cache with sharding for reduced contention
 */

#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

namespace vesper::cache {

/**
 * \brief Statistics for cache performance monitoring
 */
struct CacheStats {
    std::atomic<std::uint64_t> hits{0};
    std::atomic<std::uint64_t> misses{0};
    std::atomic<std::uint64_t> evictions{0};
    std::atomic<std::uint64_t> inserts{0};
    std::atomic<std::uint64_t> updates{0};
    std::atomic<std::uint64_t> bytes_used{0};
    
    // Custom copy constructor to handle atomics
    CacheStats() = default;
    
    CacheStats(const CacheStats& other) 
        : hits(other.hits.load())
        , misses(other.misses.load())
        , evictions(other.evictions.load())
        , inserts(other.inserts.load())
        , updates(other.updates.load())
        , bytes_used(other.bytes_used.load()) {}
    
    CacheStats& operator=(const CacheStats& other) {
        if (this != &other) {
            hits.store(other.hits.load());
            misses.store(other.misses.load());
            evictions.store(other.evictions.load());
            inserts.store(other.inserts.load());
            updates.store(other.updates.load());
            bytes_used.store(other.bytes_used.load());
        }
        return *this;
    }
    
    // Move operations
    CacheStats(CacheStats&& other) noexcept
        : hits(other.hits.load())
        , misses(other.misses.load())
        , evictions(other.evictions.load())
        , inserts(other.inserts.load())
        , updates(other.updates.load())
        , bytes_used(other.bytes_used.load()) {
        // Reset the moved-from object
        other.reset();
    }
    
    CacheStats& operator=(CacheStats&& other) noexcept {
        if (this != &other) {
            hits.store(other.hits.load());
            misses.store(other.misses.load());
            evictions.store(other.evictions.load());
            inserts.store(other.inserts.load());
            updates.store(other.updates.load());
            bytes_used.store(other.bytes_used.load());
            other.reset();
        }
        return *this;
    }
    
    [[nodiscard]] auto hit_rate() const -> double {
        auto total = hits.load() + misses.load();
        return total > 0 ? static_cast<double>(hits.load()) / total : 0.0;
    }
    
    void reset() {
        hits = 0;
        misses = 0;
        evictions = 0;
        inserts = 0;
        updates = 0;
        bytes_used = 0;
    }
};

/**
 * \brief Eviction callback for cleanup operations
 */
template<typename K, typename V>
using EvictionCallback = std::function<void(const K&, const V&)>;

/**
 * \brief LRU cache shard for reduced lock contention
 */
template<typename K, typename V, typename Hash = std::hash<K>>
class LruCacheShard {
public:
    using KeyType = K;
    using ValueType = V;
    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;
    
    struct Entry {
        K key;
        V value;
        std::size_t size_bytes;
        TimePoint last_access;
        std::uint64_t access_count{0};
    };
    
    using ListIterator = typename std::list<Entry>::iterator;
    
    explicit LruCacheShard(std::size_t max_bytes,
                           std::optional<std::chrono::seconds> ttl = std::nullopt,
                           EvictionCallback<K, V> on_evict = nullptr)
        : max_bytes_(max_bytes)
        , ttl_(ttl)
        , on_evict_(on_evict) {}
    
    // Non-copyable and non-movable due to mutex
    LruCacheShard(const LruCacheShard&) = delete;
    LruCacheShard& operator=(const LruCacheShard&) = delete;
    LruCacheShard(LruCacheShard&&) = delete;
    LruCacheShard& operator=(LruCacheShard&&) = delete;
    
    /**
     * \brief Get value from cache
     * \return Optional containing value if found and not expired
     */
    [[nodiscard]] auto get(const K& key) -> std::optional<V> {
        std::unique_lock lock(mutex_);
        
        auto it = index_.find(key);
        if (it == index_.end()) {
            stats_.misses.fetch_add(1);
            return std::nullopt;
        }
        
        auto list_it = it->second;
        
        // Check TTL if configured
        if (ttl_ && is_expired(list_it->last_access)) {
            evict_entry(it);
            stats_.misses.fetch_add(1);
            return std::nullopt;
        }
        
        // Move to front (most recently used)
        if (list_it != lru_list_.begin()) {
            lru_list_.splice(lru_list_.begin(), lru_list_, list_it);
        }
        
        // Update access metadata
        list_it->last_access = Clock::now();
        list_it->access_count++;
        
        stats_.hits.fetch_add(1);
        return list_it->value;
    }
    
    /**
     * \brief Insert or update value in cache
     */
    auto put(const K& key, V value, std::size_t size_bytes) -> void {
        std::unique_lock lock(mutex_);
        
        auto it = index_.find(key);
        if (it != index_.end()) {
            // Update existing entry
            auto list_it = it->second;
            bytes_used_ -= list_it->size_bytes;
            bytes_used_ += size_bytes;
            
            list_it->value = std::move(value);
            list_it->size_bytes = size_bytes;
            list_it->last_access = Clock::now();
            list_it->access_count++;
            
            // Move to front
            if (list_it != lru_list_.begin()) {
                lru_list_.splice(lru_list_.begin(), lru_list_, list_it);
            }
            
            stats_.updates.fetch_add(1);
        } else {
            // Insert new entry
            make_space(size_bytes);
            
            lru_list_.emplace_front(Entry{
                key,
                std::move(value),
                size_bytes,
                Clock::now(),
                1
            });
            
            index_[key] = lru_list_.begin();
            bytes_used_ += size_bytes;
            
            stats_.inserts.fetch_add(1);
        }
        
        stats_.bytes_used.store(bytes_used_);
    }
    
    /**
     * \brief Remove entry from cache
     */
    auto remove(const K& key) -> bool {
        std::unique_lock lock(mutex_);
        
        auto it = index_.find(key);
        if (it == index_.end()) {
            return false;
        }
        
        evict_entry(it);
        return true;
    }
    
    /**
     * \brief Clear all entries
     */
    auto clear() -> void {
        std::unique_lock lock(mutex_);
        
        if (on_evict_) {
            for (const auto& entry : lru_list_) {
                on_evict_(entry.key, entry.value);
            }
        }
        
        lru_list_.clear();
        index_.clear();
        bytes_used_ = 0;
        stats_.bytes_used.store(0);
    }
    
    [[nodiscard]] auto size() const -> std::size_t {
        std::shared_lock lock(mutex_);
        return index_.size();
    }
    
    [[nodiscard]] auto bytes_used() const -> std::size_t {
        return bytes_used_;
    }
    
    [[nodiscard]] auto stats() const -> const CacheStats& {
        return stats_;
    }
    
private:
    [[nodiscard]] auto is_expired(TimePoint last_access) const -> bool {
        if (!ttl_) return false;
        return (Clock::now() - last_access) > *ttl_;
    }
    
    auto make_space(std::size_t required_bytes) -> void {
        // Evict entries until we have enough space
        while (bytes_used_ + required_bytes > max_bytes_ && !lru_list_.empty()) {
            // Evict from back (least recently used)
            auto it = index_.find(lru_list_.back().key);
            evict_entry(it);
        }
        
        // Also evict expired entries
        if (ttl_) {
            auto now = Clock::now();
            while (!lru_list_.empty() && is_expired(lru_list_.back().last_access)) {
                auto it = index_.find(lru_list_.back().key);
                evict_entry(it);
            }
        }
    }
    
    auto evict_entry(typename std::unordered_map<K, ListIterator, Hash>::iterator it) -> void {
        auto list_it = it->second;
        
        if (on_evict_) {
            on_evict_(list_it->key, list_it->value);
        }
        
        bytes_used_ -= list_it->size_bytes;
        lru_list_.erase(list_it);
        index_.erase(it);
        
        stats_.evictions.fetch_add(1);
        stats_.bytes_used.store(bytes_used_);
    }
    
    mutable std::shared_mutex mutex_;
    std::list<Entry> lru_list_;
    std::unordered_map<K, ListIterator, Hash> index_;
    
    std::size_t max_bytes_;
    std::size_t bytes_used_{0};
    std::optional<std::chrono::seconds> ttl_;
    EvictionCallback<K, V> on_evict_;
    
    mutable CacheStats stats_;
};

/**
 * \brief Sharded LRU cache for high concurrency
 */
template<typename K, typename V, typename Hash = std::hash<K>>
class ShardedLruCache {
public:
    static constexpr std::size_t DEFAULT_NUM_SHARDS = 16;
    
    explicit ShardedLruCache(std::size_t max_bytes,
                            std::size_t num_shards = DEFAULT_NUM_SHARDS,
                            std::optional<std::chrono::seconds> ttl = std::nullopt,
                            EvictionCallback<K, V> on_evict = nullptr)
        : num_shards_(num_shards)
        , hasher_() {
        
        auto bytes_per_shard = max_bytes / num_shards_;
        shards_.reserve(num_shards_);
        
        for (std::size_t i = 0; i < num_shards_; ++i) {
            shards_.emplace_back(std::make_unique<LruCacheShard<K, V, Hash>>(
                bytes_per_shard, ttl, on_evict));
        }
    }
    
    [[nodiscard]] auto get(const K& key) -> std::optional<V> {
        auto& shard = get_shard(key);
        return shard->get(key);
    }
    
    auto put(const K& key, V value, std::size_t size_bytes) -> void {
        auto& shard = get_shard(key);
        shard->put(key, std::move(value), size_bytes);
    }
    
    auto remove(const K& key) -> bool {
        auto& shard = get_shard(key);
        return shard->remove(key);
    }
    
    auto clear() -> void {
        for (auto& shard : shards_) {
            shard->clear();
        }
    }
    
    [[nodiscard]] auto size() const -> std::size_t {
        std::size_t total = 0;
        for (const auto& shard : shards_) {
            total += shard->size();
        }
        return total;
    }
    
    [[nodiscard]] auto bytes_used() const -> std::size_t {
        std::size_t total = 0;
        for (const auto& shard : shards_) {
            total += shard->bytes_used();
        }
        return total;
    }
    
    [[nodiscard]] auto stats() const -> CacheStats {
        CacheStats total;
        for (const auto& shard : shards_) {
            const auto& s = shard->stats();
            total.hits += s.hits.load();
            total.misses += s.misses.load();
            total.evictions += s.evictions.load();
            total.inserts += s.inserts.load();
            total.updates += s.updates.load();
        }
        total.bytes_used = bytes_used();
        return total;
    }
    
    /**
     * \brief Pin an entry to prevent eviction
     * Note: This is a no-op in basic LRU, included for API compatibility
     */
    auto pin(const K& key) -> bool {
        // Could be implemented with a separate pinned set
        return get(key).has_value();
    }
    
    auto unpin(const K& key) -> bool {
        // Could be implemented with a separate pinned set
        return get(key).has_value();
    }
    
private:
    [[nodiscard]] auto get_shard(const K& key) -> std::unique_ptr<LruCacheShard<K, V, Hash>>& {
        auto hash = hasher_(key);
        auto shard_idx = hash % num_shards_;
        return shards_[shard_idx];
    }
    
    [[nodiscard]] auto get_shard(const K& key) const -> const std::unique_ptr<LruCacheShard<K, V, Hash>>& {
        auto hash = hasher_(key);
        auto shard_idx = hash % num_shards_;
        return shards_[shard_idx];
    }
    
    std::size_t num_shards_;
    std::vector<std::unique_ptr<LruCacheShard<K, V, Hash>>> shards_;
    Hash hasher_;
};

/**
 * \brief Type aliases for common use cases
 */
template<typename T>
using NodeCache = ShardedLruCache<std::uint32_t, T>;

using VectorCache = ShardedLruCache<std::uint32_t, std::vector<float>>;
using GraphNodeCache = ShardedLruCache<std::uint32_t, std::vector<std::uint32_t>>;

} // namespace vesper::cache