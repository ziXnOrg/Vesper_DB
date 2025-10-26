/** \file prefetch_manager.cpp
 *  \brief Implementation of predictive prefetch manager
 */

#include "vesper/io/prefetch_manager.hpp"
#include <algorithm>
#include <random>

namespace vesper::io {

// AccessPatternAnalyzer implementation

void AccessPatternAnalyzer::record_access(std::uint32_t node_id, 
                                         std::chrono::steady_clock::time_point timestamp) {
    std::unique_lock lock(mutex_);
    
    auto& stats = node_stats_[node_id];
    stats.access_count++;
    stats.last_access = timestamp;
}

void AccessPatternAnalyzer::record_transition(std::uint32_t from_node, std::uint32_t to_node) {
    std::unique_lock lock(mutex_);
    
    auto& stats = node_stats_[from_node];
    stats.transitions[to_node]++;
}

std::vector<std::uint32_t> AccessPatternAnalyzer::predict_next(
    std::uint32_t current_node, std::size_t max_predictions) const {
    
    std::shared_lock lock(mutex_);
    
    auto it = node_stats_.find(current_node);
    if (it == node_stats_.end()) {
        return {};
    }
    
    const auto& transitions = it->second.transitions;
    if (transitions.empty()) {
        return {};
    }
    
    // Sort transitions by frequency
    std::vector<std::pair<std::uint64_t, std::uint32_t>> sorted_transitions;
    sorted_transitions.reserve(transitions.size());
    
    for (const auto& [to_node, count] : transitions) {
        sorted_transitions.emplace_back(count, to_node);
    }
    
    std::sort(sorted_transitions.rbegin(), sorted_transitions.rend());
    
    // Return top predictions
    std::vector<std::uint32_t> predictions;
    predictions.reserve(std::min(max_predictions, sorted_transitions.size()));
    
    for (std::size_t i = 0; i < std::min(max_predictions, sorted_transitions.size()); ++i) {
        predictions.push_back(sorted_transitions[i].second);
    }
    
    return predictions;
}

double AccessPatternAnalyzer::get_frequency(std::uint32_t node_id) const {
    std::shared_lock lock(mutex_);
    
    auto it = node_stats_.find(node_id);
    if (it == node_stats_.end()) {
        return 0.0;
    }
    
    // Simple frequency based on access count and recency
    auto now = std::chrono::steady_clock::now();
    auto age = std::chrono::duration_cast<std::chrono::seconds>(now - it->second.last_access);
    
    // Decay based on age (half-life of 60 seconds)
    double decay = std::exp(-age.count() / 60.0);
    return static_cast<double>(it->second.access_count) * decay;
}

void AccessPatternAnalyzer::decay_patterns(double decay_factor) {
    std::unique_lock lock(mutex_);
    
    auto now = std::chrono::steady_clock::now();
    
    // Only decay if enough time has passed
    if (now - last_decay_ < std::chrono::seconds(30)) {
        return;
    }
    
    for (auto it = node_stats_.begin(); it != node_stats_.end();) {
        auto& stats = it->second;
        
        // Decay access count
        stats.access_count = static_cast<std::uint64_t>(stats.access_count * decay_factor);
        
        // Decay transitions
        for (auto trans_it = stats.transitions.begin(); trans_it != stats.transitions.end();) {
            trans_it->second = static_cast<std::uint64_t>(trans_it->second * decay_factor);
            
            // Remove very low counts
            if (trans_it->second == 0) {
                trans_it = stats.transitions.erase(trans_it);
            } else {
                ++trans_it;
            }
        }
        
        // Remove nodes with no activity
        if (stats.access_count == 0 && stats.transitions.empty()) {
            it = node_stats_.erase(it);
        } else {
            ++it;
        }
    }
    
    last_decay_ = now;
}

// PrefetchManager implementation

PrefetchManager::PrefetchManager(Config config)
    : config_(config)
    , pattern_analyzer_(std::make_unique<AccessPatternAnalyzer>()) {
}

PrefetchManager::~PrefetchManager() {
    stop();
}

void PrefetchManager::start() {
    if (running_.exchange(true)) {
        return;  // Already running
    }
    
    worker_ = std::thread(&PrefetchManager::worker_thread, this);
    
    if (config_.enable_pattern_learning) {
        pattern_decay_worker_ = std::thread(&PrefetchManager::pattern_decay_thread, this);
    }
}

void PrefetchManager::stop() {
    if (!running_.exchange(false)) {
        return;  // Already stopped
    }
    
    cv_.notify_all();
    
    if (worker_.joinable()) {
        worker_.join();
    }
    
    if (pattern_decay_worker_.joinable()) {
        pattern_decay_worker_.join();
    }
    
    // Clear pending requests
    std::lock_guard lock(queue_mutex_);
    while (!request_queue_.empty()) {
        auto request = std::move(const_cast<std::unique_ptr<PrefetchRequest>&>(request_queue_.top()));
        request_queue_.pop();
        request->completion_promise.set_value(false);
    }
    
    active_requests_.clear();
    pending_nodes_.clear();
}

std::future<bool> PrefetchManager::hint_next(std::uint32_t node_id, PrefetchPriority priority) {
    if (!running_) {
        std::promise<bool> promise;
        promise.set_value(false);
        return promise.get_future();
    }
    
    // Cache temporarily disabled
    /*
    if (cache_ && cache_->get(node_id).has_value()) {
        stats_.cache_hits.fetch_add(1);
        std::promise<bool> promise;
        promise.set_value(true);
        return promise.get_future();
    }
    */
    
    stats_.cache_misses.fetch_add(1);
    
    std::lock_guard lock(queue_mutex_);
    
    // Check if already pending
    if (pending_nodes_.count(node_id)) {
        std::promise<bool> promise;
        promise.set_value(true);  // Will be loaded by existing request
        return promise.get_future();
    }
    
    // Check queue size limit
    if (request_queue_.size() >= config_.max_queue_size) {
        std::promise<bool> promise;
        promise.set_value(false);
        return promise.get_future();
    }
    
    auto request = std::make_unique<PrefetchRequest>(node_id, priority);
    auto future = request->completion_promise.get_future();
    
    pending_nodes_.insert(node_id);
    request_queue_.push(std::move(request));
    stats_.requests_submitted.fetch_add(1);
    
    cv_.notify_one();
    
    return future;
}

std::vector<std::future<bool>> PrefetchManager::hint_batch(
    std::span<const std::uint32_t> nodes, PrefetchPriority priority) {
    
    std::vector<std::future<bool>> futures;
    futures.reserve(nodes.size());
    
    for (auto node_id : nodes) {
        futures.push_back(hint_next(node_id, priority));
    }
    
    return futures;
}

bool PrefetchManager::is_ready(std::uint32_t node_id) const {
    // Cache temporarily disabled
    return false; // cache_ && cache_->get(node_id).has_value();
}

std::future<bool> PrefetchManager::wait_for(std::uint32_t node_id, 
                                           std::chrono::milliseconds timeout) {
    // First check if already available
    if (is_ready(node_id)) {
        std::promise<bool> promise;
        promise.set_value(true);
        return promise.get_future();
    }
    
    // Submit prefetch request if not pending
    return hint_next(node_id, PrefetchPriority::CRITICAL);
}

void PrefetchManager::record_access(std::uint32_t node_id) {
    auto now = std::chrono::steady_clock::now();
    
    if (config_.enable_pattern_learning) {
        pattern_analyzer_->record_access(node_id, now);
        
        // Record transition from last accessed node
        static thread_local std::uint32_t last_node = UINT32_MAX;
        if (last_node != UINT32_MAX && last_node != node_id) {
            pattern_analyzer_->record_transition(last_node, node_id);
        }
        last_node = node_id;
    }
    
    // Check if this was accurately predicted
    {
        std::lock_guard lock(queue_mutex_);
        if (pending_nodes_.count(node_id)) {
            stats_.accurate_predictions.fetch_add(1);
            pending_nodes_.erase(node_id);
        }
    }
}

void PrefetchManager::set_loader(std::function<bool(std::uint32_t)> loader) {
    loader_ = std::move(loader);
}

/*
void PrefetchManager::set_cache(std::shared_ptr<cache::GraphNodeCache> cache) {
    cache_ = std::move(cache);
}
*/

PrefetchStats PrefetchManager::get_stats() const {
    // Can't copy atomic struct, create and populate manually
    PrefetchStats result;
    result.requests_submitted.store(stats_.requests_submitted.load());
    result.requests_completed.store(stats_.requests_completed.load());
    result.requests_cancelled.store(stats_.requests_cancelled.load());
    result.cache_hits.store(stats_.cache_hits.load());
    result.cache_misses.store(stats_.cache_misses.load());
    result.accurate_predictions.store(stats_.accurate_predictions.load());
    result.wasted_prefetches.store(stats_.wasted_prefetches.load());
    return std::move(result);
}

void PrefetchManager::clear_stats() {
    stats_.reset();
}

void PrefetchManager::worker_thread() {
    while (running_) {
        std::unique_lock lock(queue_mutex_);
        
        // Wait for requests or shutdown
        cv_.wait(lock, [this] { 
            return !request_queue_.empty() || !running_; 
        });
        
        if (!running_) break;
        
        // Process requests while under concurrent limit
        while (!request_queue_.empty() && 
               active_requests_.size() < config_.max_concurrent_requests) {
            
            auto request = std::move(const_cast<std::unique_ptr<PrefetchRequest>&>(request_queue_.top()));
            request_queue_.pop();
            
            // Check age - cancel old requests
            auto age = std::chrono::steady_clock::now() - request->created_at;
            if (age > config_.max_request_age) {
                request->completion_promise.set_value(false);
                stats_.requests_cancelled.fetch_add(1);
                continue;
            }
            
            active_requests_.insert(request->node_id);
            lock.unlock();
            
            // Process request in background
            std::thread([this, req = std::move(request)]() mutable {
                process_request(std::move(req));
            }).detach();
            
            lock.lock();
        }
        
        // Cleanup periodically
        cleanup_old_requests();
    }
}

void PrefetchManager::pattern_decay_thread() {
    while (running_) {
        std::this_thread::sleep_for(
            std::chrono::seconds(static_cast<long>(config_.pattern_decay_interval_s)));
        
        if (!running_) break;
        
        pattern_analyzer_->decay_patterns(0.9);
    }
}

void PrefetchManager::process_request(std::unique_ptr<PrefetchRequest> request) {
    bool success = false;
    
    try {
        // Load data if loader is available
        if (loader_) {
            success = loader_(request->node_id);
        }
        
        stats_.requests_completed.fetch_add(1);
    } catch (...) {
        // Swallow exceptions in prefetch thread
        success = false;
    }
    
    // Remove from active set
    {
        std::lock_guard lock(queue_mutex_);
        active_requests_.erase(request->node_id);
        
        // If not accessed yet, mark as wasted
        if (!success && pending_nodes_.count(request->node_id)) {
            stats_.wasted_prefetches.fetch_add(1);
            pending_nodes_.erase(request->node_id);
        }
    }
    
    request->completion_promise.set_value(success);
}

void PrefetchManager::cleanup_old_requests() {
    auto now = std::chrono::steady_clock::now();
    
    // Remove stale pending nodes
    for (auto it = pending_nodes_.begin(); it != pending_nodes_.end();) {
        // This is a simple cleanup - in practice we'd track timestamps
        ++it;  // Skip for now
    }
}

} // namespace vesper::io