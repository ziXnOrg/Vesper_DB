/** \file prefetch_manager.hpp
 *  \brief Predictive prefetching manager for DiskANN graph traversals
 */

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <queue>
#include <shared_mutex>
#include <span>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// #include "vesper/cache/lru_cache.hpp"  // Temporarily disabled

namespace vesper::io {

/**
 * \brief Priority levels for prefetch requests
 */
enum class PrefetchPriority : std::uint8_t {
    LOW = 0,
    NORMAL = 1,
    HIGH = 2,
    CRITICAL = 3
};

/**
 * \brief Statistics for prefetch performance monitoring
 */
struct PrefetchStats {
    std::atomic<std::uint64_t> requests_submitted{0};
    std::atomic<std::uint64_t> requests_completed{0};
    std::atomic<std::uint64_t> requests_cancelled{0};
    std::atomic<std::uint64_t> cache_hits{0};
    std::atomic<std::uint64_t> cache_misses{0};
    std::atomic<std::uint64_t> accurate_predictions{0};
    std::atomic<std::uint64_t> wasted_prefetches{0};
    
    // Custom copy/move constructors to handle atomics
    PrefetchStats() = default;
    
    PrefetchStats(const PrefetchStats& other)
        : requests_submitted(other.requests_submitted.load())
        , requests_completed(other.requests_completed.load())
        , requests_cancelled(other.requests_cancelled.load())
        , cache_hits(other.cache_hits.load())
        , cache_misses(other.cache_misses.load())
        , accurate_predictions(other.accurate_predictions.load())
        , wasted_prefetches(other.wasted_prefetches.load()) {}
    
    PrefetchStats& operator=(const PrefetchStats& other) {
        if (this != &other) {
            requests_submitted.store(other.requests_submitted.load());
            requests_completed.store(other.requests_completed.load());
            requests_cancelled.store(other.requests_cancelled.load());
            cache_hits.store(other.cache_hits.load());
            cache_misses.store(other.cache_misses.load());
            accurate_predictions.store(other.accurate_predictions.load());
            wasted_prefetches.store(other.wasted_prefetches.load());
        }
        return *this;
    }
    
    PrefetchStats(PrefetchStats&& other) noexcept
        : requests_submitted(other.requests_submitted.load())
        , requests_completed(other.requests_completed.load())
        , requests_cancelled(other.requests_cancelled.load())
        , cache_hits(other.cache_hits.load())
        , cache_misses(other.cache_misses.load())
        , accurate_predictions(other.accurate_predictions.load())
        , wasted_prefetches(other.wasted_prefetches.load()) {
        other.reset();
    }
    
    PrefetchStats& operator=(PrefetchStats&& other) noexcept {
        if (this != &other) {
            requests_submitted.store(other.requests_submitted.load());
            requests_completed.store(other.requests_completed.load());
            requests_cancelled.store(other.requests_cancelled.load());
            cache_hits.store(other.cache_hits.load());
            cache_misses.store(other.cache_misses.load());
            accurate_predictions.store(other.accurate_predictions.load());
            wasted_prefetches.store(other.wasted_prefetches.load());
            other.reset();
        }
        return *this;
    }
    
    [[nodiscard]] double hit_rate() const {
        auto total = cache_hits.load() + cache_misses.load();
        return total > 0 ? static_cast<double>(cache_hits.load()) / total : 0.0;
    }
    
    [[nodiscard]] double accuracy_rate() const {
        auto total = accurate_predictions.load() + wasted_prefetches.load();
        return total > 0 ? static_cast<double>(accurate_predictions.load()) / total : 0.0;
    }
    
    void reset() {
        requests_submitted = 0;
        requests_completed = 0;
        requests_cancelled = 0;
        cache_hits = 0;
        cache_misses = 0;
        accurate_predictions = 0;
        wasted_prefetches = 0;
    }
};

/**
 * \brief Access pattern analyzer for prediction
 */
class AccessPatternAnalyzer {
public:
    /**
     * \brief Record an access to a node
     */
    void record_access(std::uint32_t node_id, 
                      std::chrono::steady_clock::time_point timestamp);
    
    /**
     * \brief Record a transition between nodes
     */
    void record_transition(std::uint32_t from_node, std::uint32_t to_node);
    
    /**
     * \brief Predict likely next nodes based on current node
     */
    [[nodiscard]] std::vector<std::uint32_t> predict_next(
        std::uint32_t current_node, std::size_t max_predictions = 8) const;
    
    /**
     * \brief Get access frequency for a node
     */
    [[nodiscard]] double get_frequency(std::uint32_t node_id) const;
    
    /**
     * \brief Clear old patterns (decay)
     */
    void decay_patterns(double decay_factor = 0.9);
    
private:
    struct NodeStats {
        std::uint64_t access_count{0};
        std::chrono::steady_clock::time_point last_access{};
        std::unordered_map<std::uint32_t, std::uint64_t> transitions;  // to_node -> count
    };
    
    mutable std::shared_mutex mutex_;
    std::unordered_map<std::uint32_t, NodeStats> node_stats_;
    std::chrono::steady_clock::time_point last_decay_{};
};

/**
 * \brief Prefetch request
 */
struct PrefetchRequest {
    std::uint32_t node_id;
    PrefetchPriority priority;
    std::chrono::steady_clock::time_point created_at;
    std::promise<bool> completion_promise;
    
    PrefetchRequest(std::uint32_t id, PrefetchPriority prio)
        : node_id(id)
        , priority(prio)
        , created_at(std::chrono::steady_clock::now()) {}
};

/**
 * \brief Comparison function for priority queue
 */
struct PrefetchRequestComparator {
    bool operator()(const std::unique_ptr<PrefetchRequest>& a,
                   const std::unique_ptr<PrefetchRequest>& b) const {
        // Higher priority values come first
        if (a->priority != b->priority) {
            return a->priority < b->priority;
        }
        // For same priority, earlier requests come first
        return a->created_at > b->created_at;
    }
};

/**
 * \brief Prefetch manager for predictive loading
 */
class PrefetchManager {
public:
    struct Config {
        std::size_t max_queue_size{512};
        std::size_t max_concurrent_requests{32};
        std::size_t prefetch_window_size{64};
        std::chrono::milliseconds max_request_age{1000};
        double pattern_decay_interval_s{60.0};
        bool enable_pattern_learning{true};
        bool enable_sequential_prefetch{true};
        bool enable_beam_prefetch{true};
    };
    
    explicit PrefetchManager(Config config = {});
    ~PrefetchManager();
    
    /**
     * \brief Start the prefetch manager
     */
    void start();
    
    /**
     * \brief Stop the prefetch manager
     */
    void stop();
    
    /**
     * \brief Hint that a node will be accessed soon
     */
    auto hint_next(std::uint32_t node_id, 
                   PrefetchPriority priority = PrefetchPriority::NORMAL)
        -> std::future<bool>;
    
    /**
     * \brief Hint multiple nodes (e.g., search beam)
     */
    auto hint_batch(std::span<const std::uint32_t> nodes,
                    PrefetchPriority priority = PrefetchPriority::NORMAL)
        -> std::vector<std::future<bool>>;
    
    /**
     * \brief Check if a node is ready (cached or prefetched)
     */
    [[nodiscard]] bool is_ready(std::uint32_t node_id) const;
    
    /**
     * \brief Wait for a node to be available
     */
    auto wait_for(std::uint32_t node_id, 
                  std::chrono::milliseconds timeout = std::chrono::milliseconds{100})
        -> std::future<bool>;
    
    /**
     * \brief Record that a node was actually accessed (for accuracy tracking)
     */
    void record_access(std::uint32_t node_id);
    
    /**
     * \brief Set the data loader function
     */
    void set_loader(std::function<bool(std::uint32_t)> loader);
    
    /**
     * \brief Set the cache for storing prefetched data (temporarily disabled)
     */
    // void set_cache(std::shared_ptr<cache::GraphNodeCache> cache);
    
    /**
     * \brief Get statistics
     */
    [[nodiscard]] PrefetchStats get_stats() const;
    
    /**
     * \brief Clear statistics
     */
    void clear_stats();
    
private:
    void worker_thread();
    void pattern_decay_thread();
    void process_request(std::unique_ptr<PrefetchRequest> request);
    void cleanup_old_requests();
    
    Config config_;
    std::atomic<bool> running_{false};
    
    // Threading
    std::thread worker_;
    std::thread pattern_decay_worker_;
    std::condition_variable cv_;
    mutable std::mutex queue_mutex_;
    
    // Request queue (priority queue)
    std::priority_queue<
        std::unique_ptr<PrefetchRequest>,
        std::vector<std::unique_ptr<PrefetchRequest>>,
        PrefetchRequestComparator> request_queue_;
    
    // Active requests tracking
    std::unordered_set<std::uint32_t> active_requests_;
    std::unordered_set<std::uint32_t> pending_nodes_;
    
    // Pattern analysis
    std::unique_ptr<AccessPatternAnalyzer> pattern_analyzer_;
    
    // External dependencies
    std::function<bool(std::uint32_t)> loader_;
    // std::shared_ptr<cache::GraphNodeCache> cache_;  // Temporarily disabled
    
    // Statistics
    mutable PrefetchStats stats_;
};

/**
 * \brief RAII context for managing prefetch hints during search
 */
class PrefetchContext {
public:
    explicit PrefetchContext(std::shared_ptr<PrefetchManager> manager)
        : manager_(std::move(manager)) {}
    
    /**
     * \brief Hint next node in traversal
     */
    void hint_next(std::uint32_t node_id, 
                   PrefetchPriority priority = PrefetchPriority::NORMAL) {
        if (manager_) {
            futures_.push_back(manager_->hint_next(node_id, priority));
        }
    }
    
    /**
     * \brief Hint batch of nodes (search beam)
     */
    void hint_batch(std::span<const std::uint32_t> nodes,
                    PrefetchPriority priority = PrefetchPriority::NORMAL) {
        if (manager_) {
            auto batch_futures = manager_->hint_batch(nodes, priority);
            futures_.insert(futures_.end(), 
                           std::make_move_iterator(batch_futures.begin()),
                           std::make_move_iterator(batch_futures.end()));
        }
    }
    
    /**
     * \brief Check if node is ready
     */
    [[nodiscard]] bool is_ready(std::uint32_t node_id) const {
        return manager_ ? manager_->is_ready(node_id) : false;
    }
    
    /**
     * \brief Wait for node with timeout
     */
    auto wait_for(std::uint32_t node_id, 
                  std::chrono::milliseconds timeout = std::chrono::milliseconds{10})
        -> std::future<bool> {
        if (manager_) {
            return manager_->wait_for(node_id, timeout);
        }
        
        std::promise<bool> promise;
        promise.set_value(false);
        return promise.get_future();
    }
    
    /**
     * \brief Record actual access
     */
    void record_access(std::uint32_t node_id) {
        if (manager_) {
            manager_->record_access(node_id);
        }
    }
    
    /**
     * \brief Wait for all pending prefetches to complete or timeout
     */
    void wait_all(std::chrono::milliseconds timeout = std::chrono::milliseconds{100}) {
        auto deadline = std::chrono::steady_clock::now() + timeout;
        
        for (auto& future : futures_) {
            auto remaining = deadline - std::chrono::steady_clock::now();
            if (remaining > std::chrono::milliseconds{0}) {
                if (future.wait_for(remaining) == std::future_status::timeout) {
                    break;  // Don't wait for remaining futures
                }
            }
        }
        
        futures_.clear();
    }
    
    ~PrefetchContext() {
        // Best effort wait on destruction
        wait_all(std::chrono::milliseconds{10});
    }
    
private:
    std::shared_ptr<PrefetchManager> manager_;
    std::vector<std::future<bool>> futures_;
};

} // namespace vesper::io