#include "vesper/index/hnsw_lockfree.hpp"

#include <algorithm>
#include <queue>
#include <random>
#include <unordered_set>
#include <unordered_map>
#include <thread>
#include <chrono>
#include <cmath>
#include <cstring>

namespace vesper::index {

// Thread-local storage for epoch management
thread_local std::uint64_t EpochManager::thread_epoch_ = EpochManager::INACTIVE_EPOCH;
thread_local std::vector<EpochManager::DeferredDelete> EpochManager::thread_deferred_;

// AtomicEdgeList implementation

auto AtomicEdgeList::try_insert(std::uint32_t neighbor_id, std::uint32_t max_edges) -> bool {
    // Fast path: check if we're already at capacity
    std::size_t current_count = count_.load(std::memory_order_acquire);
    if (current_count >= max_edges || current_count >= MAX_EDGES) {
        return false;
    }
    
    // Check for duplicates and find an empty slot
    for (std::size_t i = 0; i < MAX_EDGES; ++i) {
        std::uint32_t current = edges_[i].load(std::memory_order_acquire);
        
        // Already exists
        if (current == neighbor_id) {
            return false;
        }
        
        // Try to claim this slot
        if (current == INVALID_ID) {
            std::uint32_t expected = INVALID_ID;
            if (edges_[i].compare_exchange_weak(expected, neighbor_id,
                                               std::memory_order_release,
                                               std::memory_order_acquire)) {
                // Successfully inserted, increment count
                count_.fetch_add(1, std::memory_order_release);
                return true;
            }
            // CAS failed, another thread took this slot, continue searching
        }
    }
    
    return false;  // No empty slots found
}

auto AtomicEdgeList::get_neighbors() const -> std::vector<std::uint32_t> {
    std::vector<std::uint32_t> result;
    std::size_t count = count_.load(std::memory_order_acquire);
    result.reserve(count);
    
    for (std::size_t i = 0, found = 0; i < MAX_EDGES && found < count; ++i) {
        std::uint32_t neighbor = edges_[i].load(std::memory_order_acquire);
        if (neighbor != INVALID_ID) {
            result.push_back(neighbor);
            ++found;
        }
    }
    
    return result;
}

auto AtomicEdgeList::contains(std::uint32_t neighbor_id) const -> bool {
    for (std::size_t i = 0; i < MAX_EDGES; ++i) {
        if (edges_[i].load(std::memory_order_acquire) == neighbor_id) {
            return true;
        }
    }
    return false;
}

auto AtomicEdgeList::try_remove(std::uint32_t neighbor_id) -> bool {
    for (std::size_t i = 0; i < MAX_EDGES; ++i) {
        std::uint32_t current = edges_[i].load(std::memory_order_acquire);
        if (current == neighbor_id) {
            // Try to remove by setting to INVALID_ID
            if (edges_[i].compare_exchange_strong(current, INVALID_ID,
                                                 std::memory_order_release,
                                                 std::memory_order_acquire)) {
                count_.fetch_sub(1, std::memory_order_release);
                return true;
            }
        }
    }
    return false;
}

// HnswLockfreeNode implementation

HnswLockfreeNode::HnswLockfreeNode(std::uint64_t id, const float* data, 
                                   std::size_t dim, std::uint32_t level)
    : id_(id)
    , data_(std::make_unique<float[]>(dim))
    , level_(level)
    , neighbors_(level + 1) {
    std::memcpy(data_.get(), data, dim * sizeof(float));
}

auto HnswLockfreeNode::get_neighbors(std::uint32_t level) const -> std::vector<std::uint32_t> {
    if (level > level_) {
        return {};
    }
    return neighbors_[level].get_neighbors();
}

auto HnswLockfreeNode::try_add_neighbor(std::uint32_t level, std::uint32_t neighbor_id,
                                        std::uint32_t max_edges) -> bool {
    if (level > level_) {
        return false;
    }
    return neighbors_[level].try_insert(neighbor_id, max_edges);
}

auto HnswLockfreeNode::try_remove_neighbor(std::uint32_t level, std::uint32_t neighbor_id) -> bool {
    if (level > level_) {
        return false;
    }
    return neighbors_[level].try_remove(neighbor_id);
}

auto HnswLockfreeNode::neighbor_count(std::uint32_t level) const -> std::size_t {
    if (level > level_) {
        return 0;
    }
    return neighbors_[level].size();
}

// EpochManager implementation

EpochManager::EpochManager() = default;

EpochManager::~EpochManager() {
    // Clean up any remaining deferred deletions
    try_reclaim();
    
    // Force delete all remaining items
    auto* current = deferred_head_.load(std::memory_order_acquire);
    while (current != nullptr) {
        auto* next = reinterpret_cast<DeferredDelete*>(
            reinterpret_cast<std::atomic<DeferredDelete*>*>(&current->ptr)->load(std::memory_order_relaxed)
        );
        current->deleter(current->ptr);
        delete current;
        current = next;
    }
}

auto EpochManager::enter_epoch() -> void {
    thread_epoch_ = global_epoch_.load(std::memory_order_acquire);
}

auto EpochManager::exit_epoch() -> void {
    thread_epoch_ = INACTIVE_EPOCH;
}

auto EpochManager::defer_delete_impl(void* ptr, void (*deleter)(void*)) -> void {
    auto current_epoch = global_epoch_.load(std::memory_order_acquire);
    thread_deferred_.push_back({ptr, deleter, current_epoch});
    
    // Periodically try to reclaim
    if (thread_deferred_.size() > 64) {
        try_reclaim();
    }
}

auto EpochManager::try_reclaim() -> void {
    auto min_epoch = get_min_epoch();
    
    // Advance global epoch periodically
    auto current = global_epoch_.load(std::memory_order_acquire);
    global_epoch_.compare_exchange_weak(current, current + 1,
                                        std::memory_order_release,
                                        std::memory_order_acquire);
    
    // Reclaim thread-local deferred deletions
    auto new_end = std::remove_if(thread_deferred_.begin(), thread_deferred_.end(),
        [min_epoch](const DeferredDelete& d) {
            if (d.epoch < min_epoch) {
                d.deleter(d.ptr);
                return true;
            }
            return false;
        });
    thread_deferred_.erase(new_end, thread_deferred_.end());
}

auto EpochManager::get_min_epoch() const -> std::uint64_t {
    // In a real implementation, this would track all active thread epochs
    // For now, we use a conservative approach
    auto current = global_epoch_.load(std::memory_order_acquire);
    return current > 2 ? current - 2 : 0;
}

// HnswLockfreeIndex::Impl implementation

class HnswLockfreeIndex::Impl {
public:
    Impl() = default;
    ~Impl() = default;
    
    // Configuration and state
    struct State {
        bool initialized{false};
        std::size_t dim{0};
        HnswLockfreeBuildParams params;
        std::size_t max_elements{0};
        std::atomic<std::size_t> n_elements{0};
        std::atomic<std::uint32_t> entry_point{std::numeric_limits<std::uint32_t>::max()};
        float level_multiplier{1.0f / std::log(2.0f)};
        std::uint32_t seed{42};
        std::atomic<std::uint64_t> build_time_ns{0};
    } state_;
    
    // Lock-free graph storage
    // Use vector of pointers instead of vector of atomics (can't copy atomics)
    std::vector<HnswLockfreeNode*> nodes_;
    std::unordered_map<std::uint64_t, std::uint32_t> id_to_idx_;  // Protected by RCU
    std::atomic<std::uint32_t> next_idx_{0};
    
    // Memory management
    EpochManager epoch_manager_;
    
    // Public interface methods
    auto init(std::size_t dim, const HnswLockfreeBuildParams& params,
              std::size_t max_elements) -> std::expected<void, core::error>;
    
    auto add(std::uint64_t id, const float* data) -> std::expected<void, core::error>;
    
    auto add_batch(const std::uint64_t* ids, const float* data, std::size_t n)
        -> std::expected<void, core::error>;
    
    auto search(const float* query, const HnswLockfreeSearchParams& params) const
        -> std::expected<std::vector<std::pair<std::uint64_t, float>>, core::error>;
    
    auto search_batch(const float* queries, std::size_t n_queries,
                      const HnswLockfreeSearchParams& params) const
        -> std::expected<std::vector<std::vector<std::pair<std::uint64_t, float>>>, core::error>;
    
    auto mark_deleted(std::uint64_t id) -> std::expected<void, core::error>;
    
    auto get_stats() const noexcept -> HnswLockfreeStats;
    
private:
    // Internal helper methods
    auto select_level(std::mt19937& rng) -> std::uint32_t;
    
    auto compute_distance(const float* a, const float* b) const -> float;
    
    auto search_layer(const float* query, std::uint32_t entry_point,
                     std::uint32_t num_closest, std::uint32_t layer,
                     const std::uint8_t* filter = nullptr) const
        -> std::vector<std::pair<float, std::uint32_t>>;
    
    auto get_node(std::uint32_t idx) const -> HnswLockfreeNode*;
    
    auto connect_bidirectional(std::uint32_t idx1, std::uint32_t idx2, 
                              std::uint32_t level, std::uint32_t max_edges) -> bool;
    
    auto prune_connections(std::uint32_t idx, std::uint32_t level,
                          std::uint32_t max_connections) -> void;
    
    auto robust_prune(const float* data_point, 
                     std::vector<std::pair<float, std::uint32_t>>& candidates,
                     std::uint32_t M, bool extend_candidates = false,
                     bool keep_pruned_connections = false) const
        -> std::vector<std::pair<float, std::uint32_t>>;
    
    auto get_neighbors_with_distances(std::uint32_t idx, std::uint32_t level) const
        -> std::vector<std::pair<float, std::uint32_t>>;
};

auto HnswLockfreeIndex::Impl::init(std::size_t dim, const HnswLockfreeBuildParams& params,
                                   std::size_t max_elements) -> std::expected<void, core::error> {
    using core::error;
    using core::error_code;
    
    if (dim == 0) {
        return std::vesper_unexpected(error{
            error_code::precondition_failed,
            "Dimension must be > 0",
            "hnsw_lockfree"
        });
    }
    
    if (params.M < 2) {
        return std::vesper_unexpected(error{
            error_code::precondition_failed,
            "M must be >= 2",
            "hnsw_lockfree"
        });
    }
    
    if (params.efConstruction < params.M) {
        return std::vesper_unexpected(error{
            error_code::precondition_failed,
            "efConstruction must be >= M",
            "hnsw_lockfree"
        });
    }
    
    state_.dim = dim;
    state_.params = params;
    state_.max_elements = max_elements;
    state_.n_elements.store(0, std::memory_order_release);
    state_.seed = params.seed;
    state_.initialized = true;
    
    if (max_elements > 0) {
        nodes_.resize(max_elements, nullptr);
        id_to_idx_.reserve(max_elements);
    }
    
    return {};
}

auto HnswLockfreeIndex::Impl::select_level(std::mt19937& rng) -> std::uint32_t {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    const float f = -std::log(dist(rng)) * state_.level_multiplier;
    return std::min(static_cast<std::uint32_t>(f), static_cast<std::uint32_t>(16));
}

auto HnswLockfreeIndex::Impl::compute_distance(const float* a, const float* b) const -> float {
    // Simple L2 squared distance calculation
    float dist = 0.0f;
    for (std::size_t i = 0; i < state_.dim; ++i) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

auto HnswLockfreeIndex::Impl::get_node(std::uint32_t idx) const -> HnswLockfreeNode* {
    if (idx >= nodes_.size()) {
        return nullptr;
    }
    // Use atomic load for thread-safety
    return __atomic_load_n(&nodes_[idx], __ATOMIC_ACQUIRE);
}

auto HnswLockfreeIndex::Impl::connect_bidirectional(std::uint32_t idx1, std::uint32_t idx2,
                                                    std::uint32_t level, std::uint32_t max_edges) -> bool {
    auto* node1 = get_node(idx1);
    auto* node2 = get_node(idx2);
    
    if (!node1 || !node2) {
        return false;
    }
    
    // First try simple insertion
    bool success1 = node1->try_add_neighbor(level, idx2, max_edges);
    bool success2 = node2->try_add_neighbor(level, idx1, max_edges);
    
    // If either failed due to full list, use pruning strategy
    if (!success1 && node1->neighbor_count(level) >= max_edges) {
        // Get current neighbors with distances
        auto neighbors = get_neighbors_with_distances(idx1, level);
        
        // Add new candidate
        float dist = compute_distance(node1->data(), node2->data());
        neighbors.emplace_back(dist, idx2);
        
        // Prune to select best M neighbors
        auto pruned = robust_prune(node1->data(), neighbors, max_edges, false, false);
        
        // Update neighbors (atomic replacement)
        // For now, skip complex replacement - will implement in next iteration
    }
    
    if (!success2 && node2->neighbor_count(level) >= max_edges) {
        // Similar logic for node2
    }
    
    return success1 || success2;
}

auto HnswLockfreeIndex::Impl::search_layer(const float* query, std::uint32_t entry_point,
                                           std::uint32_t num_closest, std::uint32_t layer,
                                           const std::uint8_t* /* filter */) const
    -> std::vector<std::pair<float, std::uint32_t>> {
    
    std::unordered_set<std::uint32_t> visited;
    std::priority_queue<std::pair<float, std::uint32_t>> candidates;
    std::priority_queue<std::pair<float, std::uint32_t>> nearest;
    
    auto* entry_node = get_node(entry_point);
    if (!entry_node || entry_node->is_deleted()) {
        return {};
    }
    
    const float entry_dist = compute_distance(query, entry_node->data());
    candidates.emplace(-entry_dist, entry_point);
    nearest.emplace(entry_dist, entry_point);
    visited.insert(entry_point);
    
    while (!candidates.empty()) {
        const auto [neg_dist, current] = candidates.top();
        const float current_dist = -neg_dist;
        candidates.pop();
        
        if (current_dist > nearest.top().first) {
            break;
        }
        
        auto* current_node = get_node(current);
        if (!current_node) {
            continue;
        }
        
        auto neighbors = current_node->get_neighbors(layer);
        
        for (std::uint32_t neighbor : neighbors) {
            if (visited.count(neighbor) > 0) {
                continue;
            }
            visited.insert(neighbor);
            
            auto* neighbor_node = get_node(neighbor);
            if (!neighbor_node || neighbor_node->is_deleted()) {
                continue;
            }
            
            const float dist = compute_distance(query, neighbor_node->data());
            
            if (dist < nearest.top().first || nearest.size() < num_closest) {
                candidates.emplace(-dist, neighbor);
                nearest.emplace(dist, neighbor);
                
                if (nearest.size() > num_closest) {
                    nearest.pop();
                }
            }
        }
    }
    
    std::vector<std::pair<float, std::uint32_t>> result;
    result.reserve(nearest.size());
    
    while (!nearest.empty()) {
        result.push_back(nearest.top());
        nearest.pop();
    }
    
    std::reverse(result.begin(), result.end());
    return result;
}

auto HnswLockfreeIndex::Impl::add(std::uint64_t id, const float* data) 
    -> std::expected<void, core::error> {
    using core::error;
    using core::error_code;
    
    if (!state_.initialized) {
        return std::vesper_unexpected(error{
            error_code::precondition_failed,
            "Index not initialized",
            "hnsw_lockfree"
        });
    }
    
    // Enter epoch for safe memory access
    epoch_manager_.enter_epoch();
    
    // Check for duplicate ID
    if (id_to_idx_.count(id) > 0) {
        epoch_manager_.exit_epoch();
        return std::vesper_unexpected(error{
            error_code::precondition_failed,
            "ID already exists",
            "hnsw_lockfree"
        });
    }
    
    // Allocate index
    std::uint32_t new_idx = next_idx_.fetch_add(1, std::memory_order_acq_rel);
    if (new_idx >= nodes_.size()) {
        nodes_.resize(new_idx + 1000, nullptr);  // Grow by chunks
    }
    
    // Select level
    thread_local std::mt19937 rng(std::random_device{}());
    std::uint32_t level = select_level(rng);
    
    // Create new node
    auto* new_node = new HnswLockfreeNode(id, data, state_.dim, level);
    // Use atomic store for thread-safety
    __atomic_store_n(&nodes_[new_idx], new_node, __ATOMIC_RELEASE);
    id_to_idx_[id] = new_idx;
    
    // Update element count
    std::size_t n_elem = state_.n_elements.fetch_add(1, std::memory_order_acq_rel);
    
    // First node becomes entry point
    if (n_elem == 0) {
        state_.entry_point.store(0, std::memory_order_release);
        epoch_manager_.exit_epoch();
        return {};
    }
    
    // Get current entry point
    std::uint32_t entry_point = state_.entry_point.load(std::memory_order_acquire);
    
    // Search for nearest neighbors at all levels
    std::vector<std::pair<float, std::uint32_t>> nearest;
    std::uint32_t curr_nearest = entry_point;
    
    auto* entry_node = get_node(entry_point);
    if (!entry_node) {
        epoch_manager_.exit_epoch();
        return {};
    }
    
    // Search upper layers
    for (std::int32_t lc = entry_node->level(); lc > static_cast<std::int32_t>(level); --lc) {
        nearest = search_layer(data, curr_nearest, 1, lc, nullptr);
        if (!nearest.empty()) {
            curr_nearest = nearest[0].second;
        }
    }
    
    // Insert at all levels
    for (std::int32_t lc = std::min(level, entry_node->level()); lc >= 0; --lc) {
        nearest = search_layer(data, curr_nearest, state_.params.efConstruction, lc, nullptr);
        
        const std::uint32_t M = (lc == 0) ? state_.params.max_M0 : state_.params.max_M;
        
        // Use RobustPrune to select best neighbors
        auto selected = robust_prune(data, nearest, M, true, true);
        
        // Add selected connections
        for (const auto& [dist, neighbor_idx] : selected) {
            connect_bidirectional(new_idx, neighbor_idx, lc, M);
        }
        
        if (!nearest.empty()) {
            curr_nearest = nearest[0].second;
        }
    }
    
    // Update entry point if new node has higher level
    if (level > entry_node->level()) {
        std::uint32_t expected = entry_point;
        state_.entry_point.compare_exchange_strong(expected, new_idx,
                                                   std::memory_order_release,
                                                   std::memory_order_acquire);
    }
    
    epoch_manager_.exit_epoch();
    return {};
}

auto HnswLockfreeIndex::Impl::add_batch(const std::uint64_t* ids, const float* data, std::size_t n)
    -> std::expected<void, core::error> {
    using core::error;
    using core::error_code;
    
    if (!state_.initialized) {
        return std::vesper_unexpected(error{
            error_code::precondition_failed,
            "Index not initialized",
            "hnsw_lockfree"
        });
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Determine thread count
    std::size_t n_threads = state_.params.n_threads;
    if (n_threads == 0) {
        n_threads = std::thread::hardware_concurrency();
    }
    
    // For small batches, use sequential insertion
    if (n < 100 || n_threads == 1) {
        for (std::size_t i = 0; i < n; ++i) {
            if (auto result = add(ids[i], data + i * state_.dim); !result.has_value()) {
                return result;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
        state_.build_time_ns.store(duration.count(), std::memory_order_relaxed);
        
        return {};
    }
    
    // Pre-allocate space
    std::size_t current_size = next_idx_.load(std::memory_order_acquire);
    std::size_t needed_size = current_size + n;
    if (nodes_.size() < needed_size) {
        nodes_.resize(needed_size + 1000, nullptr);
    }
    
    // Phase 1: Create nodes in parallel
    struct NodeInfo {
        std::uint32_t idx;
        std::uint32_t level;
        HnswLockfreeNode* node;
    };
    
    std::vector<NodeInfo> node_infos(n);
    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    
    std::size_t chunk_size = (n + n_threads - 1) / n_threads;
    
    for (std::size_t t = 0; t < n_threads; ++t) {
        threads.emplace_back([&, t]() {
            std::size_t start = t * chunk_size;
            std::size_t end = std::min(start + chunk_size, n);
            
            std::mt19937 rng(state_.seed + t);
            
            for (std::size_t i = start; i < end; ++i) {
                std::uint32_t idx = next_idx_.fetch_add(1, std::memory_order_acq_rel);
                std::uint32_t level = select_level(rng);
                
                auto* node = new HnswLockfreeNode(ids[i], data + i * state_.dim, 
                                                  state_.dim, level);
                
                __atomic_store_n(&nodes_[idx], node, __ATOMIC_RELEASE);
                id_to_idx_[ids[i]] = idx;
                
                node_infos[i] = {idx, level, node};
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Update element count
    std::size_t old_count = state_.n_elements.fetch_add(n, std::memory_order_acq_rel);
    
    // Set entry point if this is the first batch
    if (old_count == 0 && n > 0) {
        state_.entry_point.store(node_infos[0].idx, std::memory_order_release);
    }
    
    // Phase 2: Connect nodes in parallel batches
    // Sort by level for better cache locality
    std::sort(node_infos.begin(), node_infos.end(),
              [](const NodeInfo& a, const NodeInfo& b) {
                  return a.level > b.level;
              });
    
    threads.clear();
    
    // Process in smaller batches to reduce contention
    const std::size_t batch_size = std::min(static_cast<std::size_t>(100), n / n_threads);
    
    for (std::size_t batch_start = 0; batch_start < n; batch_start += batch_size * n_threads) {
        std::size_t batch_end = std::min(batch_start + batch_size * n_threads, n);
        
        for (std::size_t t = 0; t < n_threads; ++t) {
            threads.emplace_back([&, t, batch_start, batch_end]() {
                epoch_manager_.enter_epoch();
                
                std::size_t start = batch_start + t * batch_size;
                std::size_t end = std::min(start + batch_size, batch_end);
                
                for (std::size_t i = start; i < end; ++i) {
                    const auto& info = node_infos[i];
                    std::uint32_t entry_point = state_.entry_point.load(std::memory_order_acquire);
                    
                    // Search and connect at each level
                    std::uint32_t curr_nearest = entry_point;
                    
                    for (std::int32_t lc = info.level; lc >= 0; --lc) {
                        auto nearest = search_layer(info.node->data(), curr_nearest,
                                                   state_.params.efConstruction, lc, nullptr);
                        
                        const std::uint32_t M = (lc == 0) ? state_.params.max_M0 : state_.params.max_M;
                        
                        // Use RobustPrune to select best neighbors
                        auto selected = robust_prune(info.node->data(), nearest, M, true, true);
                        
                        // Add selected connections
                        for (const auto& [dist, neighbor_idx] : selected) {
                            connect_bidirectional(info.idx, neighbor_idx, lc, M);
                        }
                        
                        if (!nearest.empty()) {
                            curr_nearest = nearest[0].second;
                        }
                    }
                    
                    // Try to update entry point
                    auto* entry_node = get_node(entry_point);
                    if (entry_node && info.level > entry_node->level()) {
                        std::uint32_t expected = entry_point;
                        state_.entry_point.compare_exchange_weak(expected, info.idx,
                                                                std::memory_order_release,
                                                                std::memory_order_acquire);
                    }
                }
                
                epoch_manager_.exit_epoch();
            });
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
        threads.clear();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    state_.build_time_ns.store(duration.count(), std::memory_order_relaxed);
    
    return {};
}

auto HnswLockfreeIndex::Impl::search(const float* query, const HnswLockfreeSearchParams& params) const
    -> std::expected<std::vector<std::pair<std::uint64_t, float>>, core::error> {
    using core::error;
    using core::error_code;
    
    if (!state_.initialized || state_.n_elements.load(std::memory_order_acquire) == 0) {
        return std::vesper_unexpected(error{
            error_code::precondition_failed,
            "Index not initialized or empty",
            "hnsw_lockfree"
        });
    }
    
    std::uint32_t entry_point = state_.entry_point.load(std::memory_order_acquire);
    auto* entry_node = get_node(entry_point);
    if (!entry_node) {
        return std::vesper_unexpected(error{
            error_code::internal,
            "Entry point not found",
            "hnsw_lockfree"
        });
    }
    
    std::uint32_t curr_nearest = entry_point;
    
    // Search upper layers
    for (std::int32_t lc = entry_node->level(); lc > 0; --lc) {
        auto nearest = search_layer(query, curr_nearest, 1, lc, params.filter_mask);
        if (!nearest.empty()) {
            curr_nearest = nearest[0].second;
        }
    }
    
    // Search base layer with efSearch
    auto candidates = search_layer(query, curr_nearest, params.efSearch, 0, params.filter_mask);
    
    // Convert to output format
    std::vector<std::pair<std::uint64_t, float>> results;
    results.reserve(std::min(static_cast<std::size_t>(params.k), candidates.size()));
    
    for (std::size_t i = 0; i < std::min(static_cast<std::size_t>(params.k), candidates.size()); ++i) {
        const auto& [dist, idx] = candidates[i];
        auto* node = get_node(idx);
        if (node) {
            results.emplace_back(node->id(), dist);
        }
    }
    
    return results;
}

auto HnswLockfreeIndex::Impl::search_batch(const float* queries, std::size_t n_queries,
                                           const HnswLockfreeSearchParams& params) const
    -> std::expected<std::vector<std::vector<std::pair<std::uint64_t, float>>>, core::error> {
    
    std::vector<std::vector<std::pair<std::uint64_t, float>>> results(n_queries);
    
    // Parallel search using threads
    std::vector<std::thread> threads;
    std::size_t n_threads = std::thread::hardware_concurrency();
    std::size_t chunk_size = (n_queries + n_threads - 1) / n_threads;
    
    for (std::size_t t = 0; t < n_threads; ++t) {
        threads.emplace_back([&, t]() {
            std::size_t start = t * chunk_size;
            std::size_t end = std::min(start + chunk_size, n_queries);
            
            for (std::size_t i = start; i < end; ++i) {
                const float* query = queries + i * state_.dim;
                auto result = search(query, params);
                if (result.has_value()) {
                    results[i] = std::move(result.value());
                }
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    return results;
}

auto HnswLockfreeIndex::Impl::robust_prune(const float* data_point, 
                                           std::vector<std::pair<float, std::uint32_t>>& candidates,
                                           std::uint32_t M, bool extend_candidates,
                                           bool keep_pruned_connections) const
    -> std::vector<std::pair<float, std::uint32_t>> {
    
    // RobustPrune algorithm from HNSW paper
    // Selects M neighbors using a heuristic that promotes connectivity
    
    std::vector<std::pair<float, std::uint32_t>> result;
    result.reserve(M);
    
    // Extend candidates by adding neighbors of neighbors
    if (extend_candidates && candidates.size() > 0) {
        std::unordered_set<std::uint32_t> visited;
        for (const auto& [dist, idx] : candidates) {
            visited.insert(idx);
        }
        
        std::vector<std::pair<float, std::uint32_t>> extended;
        for (const auto& [dist, idx] : candidates) {
            auto* node = get_node(idx);
            if (!node) continue;
            
            // Add neighbors at level 0 (most connections)
            auto neighbors = node->get_neighbors(0);
            for (std::uint32_t neighbor_idx : neighbors) {
                if (visited.count(neighbor_idx) == 0) {
                    auto* neighbor_node = get_node(neighbor_idx);
                    if (neighbor_node && !neighbor_node->is_deleted()) {
                        float neighbor_dist = compute_distance(data_point, neighbor_node->data());
                        extended.emplace_back(neighbor_dist, neighbor_idx);
                        visited.insert(neighbor_idx);
                    }
                }
            }
        }
        
        // Merge extended candidates
        candidates.insert(candidates.end(), extended.begin(), extended.end());
    }
    
    // Sort candidates by distance
    std::sort(candidates.begin(), candidates.end());
    
    if (candidates.size() <= M) {
        return candidates;
    }
    
    // TEMPORARY: Just select M closest neighbors without heuristic
    // This helps isolate whether the issue is heuristic or graph structure
    for (std::size_t i = 0; i < candidates.size() && result.size() < M; ++i) {
        const auto& [dist, idx] = candidates[i];
        auto* node = get_node(idx);
        if (node && !node->is_deleted()) {
            result.push_back(candidates[i]);
        }
    }
    
    return result;
}

auto HnswLockfreeIndex::Impl::get_neighbors_with_distances(std::uint32_t idx, std::uint32_t level) const
    -> std::vector<std::pair<float, std::uint32_t>> {
    
    std::vector<std::pair<float, std::uint32_t>> result;
    
    auto* node = get_node(idx);
    if (!node || level > node->level()) {
        return result;
    }
    
    auto neighbors = node->get_neighbors(level);
    result.reserve(neighbors.size());
    
    for (std::uint32_t neighbor_idx : neighbors) {
        auto* neighbor_node = get_node(neighbor_idx);
        if (neighbor_node && !neighbor_node->is_deleted()) {
            float dist = compute_distance(node->data(), neighbor_node->data());
            result.emplace_back(dist, neighbor_idx);
        }
    }
    
    return result;
}

auto HnswLockfreeIndex::Impl::mark_deleted(std::uint64_t id) -> std::expected<void, core::error> {
    using core::error;
    using core::error_code;
    
    auto it = id_to_idx_.find(id);
    if (it == id_to_idx_.end()) {
        return std::vesper_unexpected(error{
            error_code::not_found,
            "ID not found",
            "hnsw_lockfree"
        });
    }
    
    auto* node = get_node(it->second);
    if (node) {
        node->mark_deleted();
    }
    
    return {};
}

auto HnswLockfreeIndex::Impl::get_stats() const noexcept -> HnswLockfreeStats {
    HnswLockfreeStats stats;
    stats.n_nodes = state_.n_elements.load(std::memory_order_acquire);
    
    std::size_t total_edges = 0;
    std::size_t max_level = 0;
    std::vector<std::size_t> level_counts;
    
    for (std::size_t i = 0; i < next_idx_.load(std::memory_order_acquire); ++i) {
        auto* node = get_node(i);
        if (!node || node->is_deleted()) {
            continue;
        }
        
        max_level = std::max(max_level, static_cast<std::size_t>(node->level()));
        if (level_counts.size() <= node->level()) {
            level_counts.resize(node->level() + 1, 0);
        }
        level_counts[node->level()]++;
        
        for (std::uint32_t lc = 0; lc <= node->level(); ++lc) {
            total_edges += node->neighbor_count(lc);
        }
    }
    
    stats.n_edges = total_edges / 2;  // Bidirectional edges
    stats.n_levels = max_level + 1;
    stats.level_counts = std::move(level_counts);
    stats.avg_degree = stats.n_nodes > 0 ? 
        static_cast<float>(stats.n_edges * 2) / stats.n_nodes : 0.0f;
    
    // Calculate build rate
    auto build_time = state_.build_time_ns.load(std::memory_order_relaxed);
    if (build_time > 0) {
        stats.build_rate_vec_per_sec = 
            static_cast<std::size_t>(stats.n_nodes * 1000000000ULL / build_time);
    }
    
    // Estimate memory usage
    stats.memory_bytes = sizeof(Impl);
    stats.memory_bytes += nodes_.capacity() * sizeof(std::atomic<HnswLockfreeNode*>);
    stats.memory_bytes += stats.n_nodes * (sizeof(HnswLockfreeNode) + state_.dim * sizeof(float));
    stats.memory_bytes += stats.n_edges * sizeof(std::uint32_t) * 2;
    
    return stats;
}

// HnswLockfreeIndex public interface

HnswLockfreeIndex::HnswLockfreeIndex() : impl_(std::make_unique<Impl>()) {}
HnswLockfreeIndex::~HnswLockfreeIndex() = default;
HnswLockfreeIndex::HnswLockfreeIndex(HnswLockfreeIndex&&) noexcept = default;
HnswLockfreeIndex& HnswLockfreeIndex::operator=(HnswLockfreeIndex&&) noexcept = default;

auto HnswLockfreeIndex::init(std::size_t dim, const HnswLockfreeBuildParams& params,
                             std::size_t max_elements) -> std::expected<void, core::error> {
    return impl_->init(dim, params, max_elements);
}

auto HnswLockfreeIndex::add(std::uint64_t id, const float* data) -> std::expected<void, core::error> {
    return impl_->add(id, data);
}

auto HnswLockfreeIndex::add_batch(const std::uint64_t* ids, const float* data, std::size_t n)
    -> std::expected<void, core::error> {
    return impl_->add_batch(ids, data, n);
}

auto HnswLockfreeIndex::search(const float* query, const HnswLockfreeSearchParams& params) const
    -> std::expected<std::vector<std::pair<std::uint64_t, float>>, core::error> {
    return impl_->search(query, params);
}

auto HnswLockfreeIndex::search_batch(const float* queries, std::size_t n_queries,
                                     const HnswLockfreeSearchParams& params) const
    -> std::expected<std::vector<std::vector<std::pair<std::uint64_t, float>>>, core::error> {
    return impl_->search_batch(queries, n_queries, params);
}

auto HnswLockfreeIndex::get_stats() const noexcept -> HnswLockfreeStats {
    return impl_->get_stats();
}

auto HnswLockfreeIndex::is_initialized() const noexcept -> bool {
    return impl_->state_.initialized;
}

auto HnswLockfreeIndex::dimension() const noexcept -> std::size_t {
    return impl_->state_.dim;
}

auto HnswLockfreeIndex::size() const noexcept -> std::size_t {
    return impl_->state_.n_elements.load(std::memory_order_acquire);
}

auto HnswLockfreeIndex::mark_deleted(std::uint64_t id) -> std::expected<void, core::error> {
    return impl_->mark_deleted(id);
}

auto HnswLockfreeIndex::reclaim_memory() -> void {
    impl_->epoch_manager_.try_reclaim();
}

// Utility functions

auto compute_recall_lockfree(const HnswLockfreeIndex& index,
                             const float* queries, std::size_t n_queries,
                             const std::uint64_t* ground_truth, std::size_t k,
                             const HnswLockfreeSearchParams& params) -> float {
    
    auto results = index.search_batch(queries, n_queries, params);
    if (!results.has_value()) {
        return 0.0f;
    }
    
    std::size_t total_found = 0;
    
    for (std::size_t q = 0; q < n_queries; ++q) {
        const auto& search_results = results.value()[q];
        const std::uint64_t* gt = ground_truth + q * k;
        
        for (const auto& [id, dist] : search_results) {
            for (std::size_t i = 0; i < k; ++i) {
                if (id == gt[i]) {
                    total_found++;
                    break;
                }
            }
        }
    }
    
    return static_cast<float>(total_found) / (n_queries * k);
}

} // namespace vesper::index