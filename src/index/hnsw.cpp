#include "vesper/index/hnsw.hpp"
#include "vesper/index/hnsw_thread_pool.hpp"
#include "vesper/kernels/dispatch.hpp"
#include "vesper/kernels/batch_distances.hpp"
#include "vesper/core/memory_pool.hpp"

#include <algorithm>
#include <array>

#include <queue>
#include <random>
#include <unordered_set>
#include <unordered_map>
#include <mutex>
#include <fstream>
#include <cmath>
#include <thread>
#include <iostream>

namespace vesper::index {

/** \brief Node in HNSW graph. */
struct HnswNode {
    std::uint64_t id;
    std::vector<float> data;
    std::vector<std::vector<std::uint32_t>> neighbors;  // Per level
    std::uint32_t level;
    std::atomic<bool> deleted{false};
};

/** \brief Internal implementation of HNSW index. */
class HnswIndex::Impl {
public:
    Impl() = default;
    ~Impl() = default;

    /** \brief Index configuration and state. */
    struct State {
        bool initialized{false};
        std::size_t dim{0};
        HnswBuildParams params;
        std::size_t max_elements{0};
        std::size_t n_elements{0};
        std::uint32_t entry_point{std::numeric_limits<std::uint32_t>::max()};
        float level_multiplier{1.0f / std::log(2.0f)};
        float ml{1.0f / std::log(2.0f)};  // Level assignment probability
        std::uint32_t seed{42};
        std::uint32_t last_base_chain{std::numeric_limits<std::uint32_t>::max()};

        // Cached kernel ops for hot paths (selected once at init)
        const kernels::KernelOps* ops{nullptr};

        std::mt19937 rng;
    } state_;

    /** \brief Graph nodes. */
    std::vector<std::unique_ptr<HnswNode>> nodes_;
    std::unordered_map<std::uint64_t, std::uint32_t> id_to_idx_;
    mutable std::mutex graph_mutex_;  // For structural changes only
    mutable std::mutex label_lookup_mutex_;  // For id_to_idx_ access
#if defined(VESPER_PARTITIONED_BASE_LOCKS) && VESPER_PARTITIONED_BASE_LOCKS
    static constexpr std::size_t kBaseShardCount = 64;
    mutable std::array<std::mutex, kBaseShardCount> base_layer_shards_{};
    inline std::size_t base_shard_for(std::uint32_t idx) const noexcept { return static_cast<std::size_t>(idx) % kBaseShardCount; }
#endif

    std::vector<std::unique_ptr<std::mutex>> node_mutexes_;  // Per-node fine-grained locking


    /** \brief Initialize index. */
    auto init(std::size_t dim, const HnswBuildParams& params,
              std::size_t max_elements)
        -> std::expected<void, core::error>;

    /** \brief Add vector to index. */
    auto add(std::uint64_t id, const float* data)
        -> std::expected<void, core::error>;

    /** \brief Search for nearest neighbors. */
    auto search(const float* query, const HnswSearchParams& params) const
        -> std::expected<std::vector<std::pair<std::uint64_t, float>>, core::error>;

    /** \brief Batch search. */
    auto search_batch(const float* queries, std::size_t n_queries,
                      const HnswSearchParams& params) const
        -> std::expected<std::vector<std::vector<std::pair<std::uint64_t, float>>>, core::error>;

    /** \brief Select random level for new node. */
    auto select_level() -> std::uint32_t;

    /** \brief Search layer for nearest neighbors. */
    auto search_layer(const float* query, std::uint32_t entry_point,
                     std::uint32_t num_closest, std::uint32_t layer,
                     const std::uint8_t* filter = nullptr) const
        -> std::vector<std::pair<float, std::uint32_t>>;

    /** \brief Get neighbors at level. */
    auto get_neighbors(std::uint32_t idx, std::uint32_t level) const
        -> std::vector<std::uint32_t>;

    /** \brief Connect new node to graph. */
    auto connect_node(std::uint32_t new_idx,
                     const std::vector<std::pair<float, std::uint32_t>>& candidates,
                     std::uint32_t M, std::uint32_t level) -> void;

    /** \brief Prune connections of a node. */
    auto prune_connections(std::uint32_t idx, std::uint32_t level,
                           std::uint32_t max_connections) -> void;

    /** \brief Compute distance between vectors. */
    auto compute_distance(const float* a, const float* b) const -> float;

    /** \brief Check if node passes filter. */
    auto passes_filter(std::uint32_t idx, const std::uint8_t* filter,
                       std::size_t filter_size) const -> bool;

    /** \brief Parallel batch addition with proper connectivity. */
    auto add_batch_parallel(const std::uint64_t* ids, const float* data, std::size_t n)
        -> std::expected<void, core::error>;

    /** \brief Compute base-layer reachability using BFS (diagnostics). */
    auto reachable_count_base_layer() const -> std::size_t;

private:
};

auto HnswIndex::Impl::init(std::size_t dim, const HnswBuildParams& params,
                           std::size_t max_elements)
    -> std::expected<void, core::error> {
    using core::error;
    using core::error_code;

    if (dim == 0) {
        return std::vesper_unexpected(error{
            error_code::precondition_failed,
            "Dimension must be > 0",
            "hnsw"
        });
    }

    if (params.M < 2) {
        return std::vesper_unexpected(error{
            error_code::precondition_failed,
            "M must be >= 2",
            "hnsw"
        });
    }

    if (params.efConstruction < params.M) {
        return std::vesper_unexpected(error{
            error_code::precondition_failed,
            "efConstruction must be >= M",
            "hnsw"
        });
    }

    state_.dim = dim;
    state_.params = params;
    state_.max_elements = max_elements;
    state_.n_elements = 0;
    state_.seed = params.seed;
    state_.rng.seed(params.seed);
    // Use standard HNSW level multiplier based on M
    const float ml = 1.0f / std::log(static_cast<float>(state_.params.M));
    state_.level_multiplier = ml;
    state_.ml = ml;

    // Cache kernel ops once for hot paths
    state_.ops = &kernels::select_backend_auto();

    state_.initialized = true;

    if (max_elements > 0) {
        nodes_.reserve(max_elements);
        id_to_idx_.reserve(max_elements);
        node_mutexes_.reserve(max_elements);
        for (std::size_t i = 0; i < max_elements; ++i) {
            node_mutexes_.push_back(std::make_unique<std::mutex>());
        }
    } else {
        // Start with reasonable default
        for (std::size_t i = 0; i < 1000; ++i) {
            node_mutexes_.push_back(std::make_unique<std::mutex>());
        }
    }

    return {};
}

auto HnswIndex::Impl::select_level() -> std::uint32_t {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    const float f = -std::log(dist(state_.rng)) * state_.level_multiplier;
    return static_cast<std::uint32_t>(f);
}

auto HnswIndex::Impl::compute_distance(const float* a, const float* b) const -> float {
    // Use cached kernel ops selected at init
    return state_.ops->l2_sq(std::span(a, state_.dim), std::span(b, state_.dim));
}

auto HnswIndex::Impl::passes_filter(std::uint32_t idx, const std::uint8_t* filter,
                                    std::size_t filter_size) const -> bool {
    if (!filter || filter_size == 0) return true;

    const std::size_t byte_idx = idx / 8;
    const std::uint8_t bit_idx = idx % 8;

    if (byte_idx >= filter_size) return false;

    return (filter[byte_idx] & (1 << bit_idx)) != 0;
}

auto HnswIndex::Impl::search_layer(const float* query, std::uint32_t entry_point,
                                   std::uint32_t num_closest, std::uint32_t layer,
                                   const std::uint8_t* filter) const
    -> std::vector<std::pair<float, std::uint32_t>> {

    std::unordered_set<std::uint32_t> visited;
    std::priority_queue<std::pair<float, std::uint32_t>> candidates;
    std::priority_queue<std::pair<float, std::uint32_t>> nearest;

    const float entry_dist = compute_distance(query, nodes_[entry_point]->data.data());
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

        // Take a consistent snapshot of neighbors under lock to avoid races with concurrent writers
        std::vector<std::uint32_t> neighbors_copy;
        {
            std::lock_guard<std::mutex> g(*node_mutexes_[current]);
            neighbors_copy = nodes_[current]->neighbors[layer];
        }

        // Collect unvisited neighbors for batch distance computation
        std::vector<std::uint32_t> unvisited_neighbors;
        std::vector<const float*> neighbor_ptrs;

        for (std::uint32_t neighbor : neighbors_copy) {
            if (visited.count(neighbor) > 0) continue;
            if (nodes_[neighbor]->deleted.load()) continue;
            if (filter && !passes_filter(neighbor, filter, 0)) continue;

            visited.insert(neighbor);
            unvisited_neighbors.push_back(neighbor);
            neighbor_ptrs.push_back(nodes_[neighbor]->data.data());
        }

        // Compute distances in batch for better SIMD utilization
        if (!unvisited_neighbors.empty()) {
            std::vector<float> distances(unvisited_neighbors.size());

            // Always use optimized distance computation
            const auto& ops = kernels::select_backend_auto();
            for (std::size_t i = 0; i < unvisited_neighbors.size(); ++i) {
                distances[i] = ops.l2_sq(
                    std::span(query, state_.dim),
                    std::span(neighbor_ptrs[i], state_.dim)
                );
            }

            // Process results
            for (std::size_t i = 0; i < unvisited_neighbors.size(); ++i) {
                const float dist = distances[i];
                const std::uint32_t neighbor = unvisited_neighbors[i];

                if (dist < nearest.top().first || nearest.size() < num_closest) {
                    candidates.emplace(-dist, neighbor);
                    nearest.emplace(dist, neighbor);

                    if (nearest.size() > num_closest) {
                        nearest.pop();
                    }
                }
            }
        }
    }

    std::vector<std::pair<float, std::uint32_t>> result;
    while (!nearest.empty()) {
        result.push_back(nearest.top());
        nearest.pop();
    }

    std::reverse(result.begin(), result.end());
    return result;
}

auto HnswIndex::Impl::connect_node(std::uint32_t new_idx,
                                   const std::vector<std::pair<float, std::uint32_t>>& candidates,
                                   std::uint32_t M, std::uint32_t level) -> void {

    auto& new_neighbors = nodes_[new_idx]->neighbors[level];
    new_neighbors.clear();

    // Select M neighbors using heuristic
    std::vector<std::uint32_t> selected;
    std::vector<std::uint32_t> pruned_candidates;

    if (state_.params.extend_candidates || state_.params.keep_pruned_connections) {
        // Use RobustPrune for better connectivity
        auto candidates_copy = candidates;
        auto [selected_nodes, pruned_nodes] = robust_prune(
            candidates_copy, M,
            state_.params.extend_candidates,
            state_.params.keep_pruned_connections
        );
        selected = std::move(selected_nodes);
        pruned_candidates = std::move(pruned_nodes);
    } else {
        // Simple selection of M nearest
        for (std::size_t i = 0; i < std::min(static_cast<std::size_t>(M), candidates.size()); ++i) {
            selected.push_back(candidates[i].second);
        }
    }

    // Add selected neighbors to new node (lock current node)
    {
        std::lock_guard<std::mutex> lock_new(*node_mutexes_[new_idx]);
        auto& new_neighbors_ref = nodes_[new_idx]->neighbors[level];
        new_neighbors_ref.clear();
        for (std::uint32_t neighbor : selected) {
            if (std::find(new_neighbors_ref.begin(), new_neighbors_ref.end(), neighbor) == new_neighbors_ref.end()) {
                new_neighbors_ref.push_back(neighbor);
            }
        }
    }

    bool accepted_any = false;


    // CRITICAL: Properly update reverse edges with back-pruning
    // Lock both nodes consistently to avoid concurrent modification races
    for (std::uint32_t neighbor : selected) {
        std::unique_lock<std::mutex> l1(*node_mutexes_[new_idx], std::defer_lock);
        std::unique_lock<std::mutex> l2(*node_mutexes_[neighbor], std::defer_lock);
        std::lock(l1, l2);

        auto& neighbor_neighbors = nodes_[neighbor]->neighbors[level];

        // Collect all candidates for this neighbor (including new_idx)
        std::vector<std::pair<float, std::uint32_t>> neighbor_candidates;
        const float* neighbor_data = nodes_[neighbor]->data.data();

        // Add existing connections
        for (std::uint32_t nn : neighbor_neighbors) {
            if (nn != new_idx) {  // Avoid duplicates
                const float dist = compute_distance(neighbor_data, nodes_[nn]->data.data());
                neighbor_candidates.emplace_back(dist, nn);
            }
        }

        // Prune limit and add the new node (with slight bias at base layer if saturated)
        const std::uint32_t max_conn = (level == 0) ? state_.params.max_M0 : state_.params.max_M;
        bool saturated_base = (level == 0 && neighbor_candidates.size() >= max_conn);
        float dist_to_new = compute_distance(neighbor_data, nodes_[new_idx]->data.data());
        if (saturated_base) {
            dist_to_new *= 0.95f; // bias keep-new when neighbor saturated at base layer
        }
        neighbor_candidates.emplace_back(dist_to_new, new_idx);

        if (neighbor_candidates.size() > max_conn) {
            auto [new_selected_nv, _] = robust_prune(
                neighbor_candidates, max_conn,
                state_.params.extend_candidates,
                false  // Don't need to keep pruned here
            );
            neighbor_neighbors = std::move(new_selected_nv);
        } else {

            // Add new_idx if under limit and not already present
            if (std::find(neighbor_neighbors.begin(), neighbor_neighbors.end(), new_idx) == neighbor_neighbors.end()) {
                neighbor_neighbors.push_back(new_idx);
            }
        }
        // Track acceptance of reverse edge for this neighbor
        if (std::find(neighbor_neighbors.begin(), neighbor_neighbors.end(), new_idx) != neighbor_neighbors.end()) {
            accepted_any = true;
        }

    }

    // Guarantee reciprocal connectivity at base layer: ensure at least one reverse edge
    if (level == 0 && !accepted_any && !selected.empty()) {
        std::uint32_t forced_neighbor = selected[0];
        std::unique_lock<std::mutex> l1(*node_mutexes_[new_idx], std::defer_lock);
        std::unique_lock<std::mutex> l2(*node_mutexes_[forced_neighbor], std::defer_lock);
        std::lock(l1, l2);

        auto& nn = nodes_[forced_neighbor]->neighbors[level];
        const std::uint32_t max_conn = state_.params.max_M0;
        if (std::find(nn.begin(), nn.end(), new_idx) == nn.end()) {
            if (nn.size() < max_conn) {
                nn.push_back(new_idx);
            } else {
                // Replace the farthest
                const float* fd = nodes_[forced_neighbor]->data.data();
                float worst_dist = -1.0f; std::size_t worst_pos = 0;
                for (std::size_t i = 0; i < nn.size(); ++i) {
                    float d = compute_distance(fd, nodes_[nn[i]]->data.data());
                    if (d > worst_dist) { worst_dist = d; worst_pos = i; }
                }
                nn[worst_pos] = new_idx;
            }
        }
    }


    // Handle pruned connections if keep_pruned_connections is enabled
    if (state_.params.keep_pruned_connections && !pruned_candidates.empty()) {
        // Add pruned candidates with lower priority
        // This helps maintain graph connectivity
        for (std::uint32_t pruned_neighbor : pruned_candidates) {

            std::unique_lock<std::mutex> l1(*node_mutexes_[new_idx], std::defer_lock);
            std::unique_lock<std::mutex> l2(*node_mutexes_[pruned_neighbor], std::defer_lock);
            std::lock(l1, l2);

            auto& neighbor_neighbors = nodes_[pruned_neighbor]->neighbors[level];
            const std::uint32_t max_conn = (level == 0) ? state_.params.max_M0 : state_.params.max_M;

            // Only add if neighbor has room; do not grow new node beyond its cap here
            if (neighbor_neighbors.size() < max_conn) {
                if (std::find(neighbor_neighbors.begin(), neighbor_neighbors.end(), new_idx) == neighbor_neighbors.end()) {
                    neighbor_neighbors.push_back(new_idx);
                }
            }
        }
    }

    // Enforce degree cap on the new node after all updates
    {
        std::lock_guard<std::mutex> lock_new(*node_mutexes_[new_idx]);
        auto& nv = nodes_[new_idx]->neighbors[level];
        const std::uint32_t max_conn = (level == 0) ? state_.params.max_M0 : state_.params.max_M;
        if (nv.size() > max_conn) {
            std::vector<std::pair<float, std::uint32_t>> cand;
            cand.reserve(nv.size());
            const float* nd = nodes_[new_idx]->data.data();
            for (auto nb : nv) {
                cand.emplace_back(compute_distance(nd, nodes_[nb]->data.data()), nb);
            }
            auto [sel, _] = robust_prune(cand, max_conn, state_.params.extend_candidates, false);
            nv = std::move(sel);
        }
    }

}

auto HnswIndex::Impl::prune_connections(std::uint32_t idx, std::uint32_t level,
                                        std::uint32_t max_connections) -> void {
    auto& neighbors = nodes_[idx]->neighbors[level];
    if (neighbors.size() <= max_connections) return;

    // Collect all neighbors with distances
    std::vector<std::pair<float, std::uint32_t>> candidates;
    const float* node_data = nodes_[idx]->data.data();


    for (std::uint32_t neighbor : neighbors) {
        const float dist = compute_distance(node_data, nodes_[neighbor]->data.data());
        candidates.emplace_back(dist, neighbor);
    }

    // Prune to max_connections using heuristic
    if (state_.params.extend_candidates) {
        auto [pruned, extended] = robust_prune(candidates, max_connections, false, false);
        neighbors = std::move(pruned);
    } else {
        std::sort(candidates.begin(), candidates.end());
        neighbors.clear();
        for (std::size_t i = 0; i < max_connections; ++i) {
            neighbors.push_back(candidates[i].second);
        }
    }
}

auto HnswIndex::Impl::add(std::uint64_t id, const float* data)
    -> std::expected<void, core::error> {
    using core::error;
    using core::error_code;

    if (!state_.initialized) {
        return std::vesper_unexpected(error{
            error_code::precondition_failed,
            "Index not initialized",
            "hnsw"
        });
    }

    // Check for duplicate ID with minimal locking
    {
        std::lock_guard<std::mutex> lock(label_lookup_mutex_);
        if (id_to_idx_.count(id) > 0) {
            return std::vesper_unexpected(error{
                error_code::precondition_failed,
                "ID already exists",
                "hnsw"
            });
        }
    }

    // Allocate new index and create node
    std::uint32_t new_idx;
    std::uint32_t level = select_level();

    // Only lock for structural changes (adding node to graph)
    {
        std::lock_guard<std::mutex> lock(graph_mutex_);
        new_idx = state_.n_elements++;

        // Create new node
        auto node = std::make_unique<HnswNode>();
        node->id = id;
        node->data.assign(data, data + state_.dim);
        node->level = level;
        node->neighbors.resize(level + 1);

        // Ensure we have enough mutexes
        if (node_mutexes_.size() <= new_idx) {
            std::size_t old_size = node_mutexes_.size();
            std::size_t new_size = new_idx + 1000;
            node_mutexes_.reserve(new_size);
            for (std::size_t i = old_size; i < new_size; ++i) {
                node_mutexes_.push_back(std::make_unique<std::mutex>());
            }
        }

        nodes_.push_back(std::move(node));

        // Update lookup table
        {
            std::lock_guard<std::mutex> lock_label(label_lookup_mutex_);
            id_to_idx_[id] = new_idx;
        }
    }

    // First node becomes entry point
    if (new_idx == 0) {
        std::lock_guard<std::mutex> lock(graph_mutex_);
        state_.entry_point = 0;
        return {};
    }

    // Search for nearest neighbors at all levels
    std::vector<std::pair<float, std::uint32_t>> nearest;

    // Snapshot entry point and its top level under lock to avoid races during traversal
    std::uint32_t ep_idx_snapshot;
    std::uint32_t ep_top_level_snapshot;
    {
        std::lock_guard<std::mutex> lock(graph_mutex_);
        ep_idx_snapshot = state_.entry_point;
        ep_top_level_snapshot = nodes_[ep_idx_snapshot]->level;
    }
    std::uint32_t curr_nearest = ep_idx_snapshot;

    // Search upper layers using the snapshot
    for (std::int32_t lc = static_cast<std::int32_t>(ep_top_level_snapshot); lc > static_cast<std::int32_t>(level); --lc) {
        nearest = search_layer(data, curr_nearest, 1, lc, nullptr);
        if (!nearest.empty()) {
            curr_nearest = nearest[0].second;
        }
    }

    // Insert at upper levels (l > 0) under global lock to avoid concurrent structural changes
    {
        std::lock_guard<std::mutex> lock(graph_mutex_);
        for (std::int32_t lc = static_cast<std::int32_t>(std::min(level, ep_top_level_snapshot)); lc > 0; --lc) {
            std::uint32_t efL = state_.params.efConstruction;
            if (state_.params.adaptive_ef) {
                efL = state_.params.efConstructionUpper != 0
                    ? state_.params.efConstructionUpper
                    : std::max<std::uint32_t>(50u, state_.params.efConstruction / 2);
            }
            nearest = search_layer(data, curr_nearest, efL, lc, nullptr);
            const std::uint32_t M = state_.params.max_M;
            connect_node(new_idx, nearest, M, lc);
            if (!nearest.empty()) {
                curr_nearest = nearest[0].second;
            }
        }
    }

    // Insert at base layer (l == 0)
#if defined(VESPER_SERIALIZE_BASE_LAYER) && VESPER_SERIALIZE_BASE_LAYER
    {
        std::lock_guard<std::mutex> lock(graph_mutex_); // serialize entire base-layer connect
        nearest = search_layer(data, curr_nearest, state_.params.efConstruction, 0, nullptr);
        const std::uint32_t M0 = state_.params.max_M0;
        connect_node(new_idx, nearest, M0, 0);
        if (!nearest.empty()) {
            curr_nearest = nearest[0].second;
        }
    }
#else
    // Serialize only the search, perform connect in parallel
    #if defined(VESPER_PARTITIONED_BASE_LOCKS) && VESPER_PARTITIONED_BASE_LOCKS
    {
        // Partitioned base-layer lock: only serialize within shard of new_idx
        std::lock_guard<std::mutex> shard_lock(base_layer_shards_[base_shard_for(new_idx)]);
        nearest = search_layer(data, curr_nearest, state_.params.efConstruction, 0, nullptr);
        const std::uint32_t M0 = state_.params.max_M0;
        connect_node(new_idx, nearest, M0, 0);
        if (!nearest.empty()) {
            curr_nearest = nearest[0].second;
        }
    }
    #else
    {
        std::lock_guard<std::mutex> lock(graph_mutex_);
        nearest = search_layer(data, curr_nearest, state_.params.efConstruction, 0, nullptr);
    }
    {
        const std::uint32_t M0 = state_.params.max_M0;
        connect_node(new_idx, nearest, M0, 0);
        if (!nearest.empty()) {
            curr_nearest = nearest[0].second;
        }
    }
    #endif
#endif

    // Update entry point if new node has higher level
    {
        std::lock_guard<std::mutex> lock(graph_mutex_);
        if (level > nodes_[state_.entry_point]->level) {
            state_.entry_point = new_idx;
        }
    }

    return {};
}

auto HnswIndex::Impl::add_batch_parallel(const std::uint64_t* ids, const float* data, std::size_t n)
    -> std::expected<void, core::error> {
    using core::error;
    using core::error_code;

    if (!state_.initialized) {
        return std::vesper_unexpected(error{
            error_code::precondition_failed,
            "Index not initialized",
            "hnsw"
        });
    }

    // Following hnswlib's approach: use ParallelFor but call regular add with fine-grained locking
    // This maintains proper graph connectivity while allowing parallelism

    HnswParallelContext context(state_.params.num_threads);  // Use configured thread count (0=auto)
    auto& pool = context.pool();

    // Pre-allocate space and mutexes to avoid reallocation during parallel phase
    {
        std::lock_guard<std::mutex> lock(graph_mutex_);
        nodes_.reserve(state_.n_elements + n);
        id_to_idx_.reserve(id_to_idx_.size() + n);

        // Ensure we have enough node mutexes
        if (node_mutexes_.size() < state_.n_elements + n) {
            // Initialize new mutexes
            std::size_t old_size = node_mutexes_.size();
            std::size_t new_size = state_.n_elements + n + 1000;  // Some extra buffer
            node_mutexes_.reserve(new_size);
            for (std::size_t i = old_size; i < new_size; ++i) {
                node_mutexes_.push_back(std::make_unique<std::mutex>());

            }
        }
    }

    // Track errors from parallel execution
    std::atomic<bool> has_error{false};
    std::mutex error_mutex;
    std::optional<error> first_error;

    // Process in parallel, but each thread calls the regular add() which uses fine-grained locking
    // This is exactly how hnswlib does it - ParallelFor with addPoint inside
    const std::size_t batch_size = 64;  // Process in small batches for better load balancing
    std::vector<std::future<void>> futures;

    for (std::size_t start = 0; start < n; start += batch_size) {
        std::size_t end = std::min(start + batch_size, n);

        futures.push_back(pool.submit([this, ids, data, start, end, &has_error, &error_mutex, &first_error]() {
            for (std::size_t i = start; i < end; ++i) {
                if (has_error.load()) {
                    break;  // Early exit if another thread encountered an error
                }

                auto result = add(ids[i], data + i * state_.dim);

                if (!result.has_value()) {
                    std::lock_guard<std::mutex> lock(error_mutex);
                    if (!has_error.exchange(true)) {
                        first_error = result.error();
                    }
                }
            }
        }));
    }

    // Wait for all threads to complete
    for (auto& future : futures) {
        future.wait();
    }

    // Check if any thread encountered an error
    if (has_error.load() && first_error.has_value()) {
        return std::vesper_unexpected(first_error.value());
    }

    // (removed temporary diagnostics)

    return {};
}

auto HnswIndex::Impl::search(const float* query, const HnswSearchParams& params) const
    -> std::expected<std::vector<std::pair<std::uint64_t, float>>, core::error> {
    using core::error;
    using core::error_code;

    if (!state_.initialized || state_.n_elements == 0) {
        return std::vesper_unexpected(error{
            error_code::precondition_failed,
            "Index not initialized or empty",
            "hnsw"
        });
    }

    // Snapshot entry point and top level under lock to avoid race with concurrent insertions
    std::uint32_t ep_idx_snapshot;
    std::uint32_t ep_top_level_snapshot;
    {
        std::lock_guard<std::mutex> lock(graph_mutex_);
        ep_idx_snapshot = state_.entry_point;
        ep_top_level_snapshot = nodes_[ep_idx_snapshot]->level;
    }
    std::uint32_t curr_nearest = ep_idx_snapshot;

    // Search upper layers using the snapshot
    for (std::int32_t lc = static_cast<std::int32_t>(ep_top_level_snapshot); lc > 0; --lc) {
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
        results.emplace_back(nodes_[idx]->id, dist);
    }

    return results;
}

auto HnswIndex::Impl::search_batch(const float* queries, std::size_t n_queries,
                                   const HnswSearchParams& params) const
    -> std::expected<std::vector<std::vector<std::pair<std::uint64_t, float>>>, core::error> {

    std::vector<std::vector<std::pair<std::uint64_t, float>>> results(n_queries);

    #pragma omp parallel for
    for (std::size_t i = 0; i < n_queries; ++i) {
        const float* query = queries + i * state_.dim;
        auto result = search(query, params);

        if (result.has_value()) {
            results[i] = std::move(result.value());
        }
    }

    return results;
}

// HnswIndex public interface implementation

HnswIndex::HnswIndex() : impl_(std::make_unique<Impl>()) {}
HnswIndex::~HnswIndex() = default;
HnswIndex::HnswIndex(HnswIndex&&) noexcept = default;
HnswIndex& HnswIndex::operator=(HnswIndex&&) noexcept = default;

auto HnswIndex::init(std::size_t dim, const HnswBuildParams& params,
                    std::size_t max_elements)
    -> std::expected<void, core::error> {
    return impl_->init(dim, params, max_elements);
}

auto HnswIndex::add(std::uint64_t id, const float* data)
    -> std::expected<void, core::error> {
    return impl_->add(id, data);
}

auto HnswIndex::add_batch(const std::uint64_t* ids, const float* data, std::size_t n)
    -> std::expected<void, core::error> {
    if (n == 0) return {};
    return impl_->add_batch_parallel(ids, data, n);
}

auto HnswIndex::search(const float* query, const HnswSearchParams& params) const
    -> std::expected<std::vector<std::pair<std::uint64_t, float>>, core::error> {
    return impl_->search(query, params);
}

auto HnswIndex::search_batch(const float* queries, std::size_t n_queries,
                             const HnswSearchParams& params) const
    -> std::expected<std::vector<std::vector<std::pair<std::uint64_t, float>>>, core::error> {
    return impl_->search_batch(queries, n_queries, params);
}

auto HnswIndex::get_stats() const noexcept -> HnswStats {
    HnswStats stats;
    stats.n_nodes = impl_->state_.n_elements;

    std::size_t total_edges = 0;
    std::size_t max_level = 0;
    std::vector<std::size_t> level_counts;

    for (const auto& node : impl_->nodes_) {
        if (!node) continue;

        max_level = std::max(max_level, static_cast<std::size_t>(node->level));
        if (level_counts.size() <= node->level) {
            level_counts.resize(node->level + 1, 0);
        }
        level_counts[node->level]++;

        for (const auto& neighbors : node->neighbors) {
            total_edges += neighbors.size();
        }
    }

    stats.n_edges = total_edges / 2;  // Bidirectional edges
    stats.n_levels = max_level + 1;
    stats.level_counts = std::move(level_counts);
    stats.avg_degree = stats.n_nodes > 0 ?
        static_cast<float>(stats.n_edges * 2) / stats.n_nodes : 0.0f;

    // Estimate memory usage
    stats.memory_bytes = sizeof(Impl);
    stats.memory_bytes += impl_->nodes_.capacity() * sizeof(std::unique_ptr<HnswNode>);
    for (const auto& node : impl_->nodes_) {
        if (!node) continue;
        stats.memory_bytes += sizeof(HnswNode);
        stats.memory_bytes += node->data.capacity() * sizeof(float);
        for (const auto& neighbors : node->neighbors) {
            stats.memory_bytes += neighbors.capacity() * sizeof(std::uint32_t);
        }
    }

    return stats;
}

auto HnswIndex::is_initialized() const noexcept -> bool {
    return impl_->state_.initialized;
}

auto HnswIndex::dimension() const noexcept -> std::size_t {
    return impl_->state_.dim;
}

auto HnswIndex::size() const noexcept -> std::size_t {
    return impl_->state_.n_elements;
}

auto HnswIndex::mark_deleted(std::uint64_t id) -> std::expected<void, core::error> {
    using core::error;
    using core::error_code;

    auto it = impl_->id_to_idx_.find(id);
    if (it == impl_->id_to_idx_.end()) {
        return std::vesper_unexpected(error{
            error_code::not_found,
            "ID not found",
            "hnsw"
        });
    }

    impl_->nodes_[it->second]->deleted.store(true);
    return {};
}

auto HnswIndex::resize(std::size_t new_max_elements) -> std::expected<void, core::error> {
    impl_->nodes_.reserve(new_max_elements);
    impl_->state_.max_elements = new_max_elements;
    return {};
}

auto HnswIndex::optimize() -> std::expected<void, core::error> {
    // Optimization passes can be added here
    // For now, just return success
    return {};
}

auto HnswIndex::save(const std::string& path) const -> std::expected<void, core::error> {


    using core::error;
    using core::error_code;

    std::ofstream file(path, std::ios::binary);
    if (!file) {
        return std::vesper_unexpected(error{
            error_code::io_failed,
            "Failed to open file for writing",
            "hnsw"
        });
    }

    // Write header
    const char magic[] = "HNSW0001";
    file.write(magic, 8);

    // Write parameters and state
    file.write(reinterpret_cast<const char*>(&impl_->state_.dim), sizeof(std::size_t));
    file.write(reinterpret_cast<const char*>(&impl_->state_.params), sizeof(HnswBuildParams));
    file.write(reinterpret_cast<const char*>(&impl_->state_.n_elements), sizeof(std::size_t));
    file.write(reinterpret_cast<const char*>(&impl_->state_.entry_point), sizeof(std::uint32_t));

    // Write nodes
    for (std::size_t i = 0; i < impl_->state_.n_elements; ++i) {
        const auto& node = impl_->nodes_[i];
        file.write(reinterpret_cast<const char*>(&node->id), sizeof(std::uint64_t));
        file.write(reinterpret_cast<const char*>(&node->level), sizeof(std::uint32_t));
        file.write(reinterpret_cast<const char*>(node->data.data()),
                  impl_->state_.dim * sizeof(float));

        // Write neighbors
        for (const auto& neighbors : node->neighbors) {
            const std::size_t n_neighbors = neighbors.size();
            file.write(reinterpret_cast<const char*>(&n_neighbors), sizeof(std::size_t));
            file.write(reinterpret_cast<const char*>(neighbors.data()),
                      n_neighbors * sizeof(std::uint32_t));
        }
    }

    return {};
}

auto HnswIndex::load(const std::string& /* path */) -> std::expected<HnswIndex, core::error> {
    // Full serialization will be implemented in next phase
    return HnswIndex();
}

auto HnswIndex::get_build_params() const noexcept -> HnswBuildParams {
    return impl_->state_.params;
}

auto compute_recall(const HnswIndex& index,
                   const float* queries, std::size_t n_queries,
                   const std::uint64_t* ground_truth, std::size_t k,
                   const HnswSearchParams& params) -> float {

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

auto robust_prune(
    std::vector<std::pair<float, std::uint32_t>>& candidates,
    std::uint32_t M,
    bool extend_candidates,
    bool keep_pruned)
    -> std::pair<std::vector<std::uint32_t>, std::vector<std::uint32_t>> {

    // Implementation of Algorithm 4 (Select-Neighbors-Heuristic) from HNSW paper


    // This ensures both connectivity and diversity in the graph

    if (candidates.empty()) {
        return {{}, {}};
    }

    // Sort candidates by distance (ascending)
    std::sort(candidates.begin(), candidates.end());

    std::vector<std::uint32_t> R;      // Result set
    std::vector<std::uint32_t> W_d;    // Discarded candidates

    // Always include closest neighbor for connectivity
    R.push_back(candidates[0].second);

    // Process remaining candidates
    for (std::size_t i = 1; i < candidates.size(); ++i) {
        const auto& [dist_c, c_idx] = candidates[i];

        if (R.size() >= M) {
            // We have enough neighbors
            if (keep_pruned || extend_candidates) {
                W_d.push_back(c_idx);
            }
            continue;
        }

        // Heuristic: check if candidate adds diversity
        // A candidate is diverse if it's not too similar to already selected neighbors
        bool is_diverse = true;

        // Simple diversity check:
        // - Always accept if we have less than M/2 neighbors (build connectivity)
        // - Check distance ratio for remaining slots (ensure diversity)
        if (R.size() >= M / 2) {
            // Prefer adding some longer-range edges for connectivity; reject only if too close to nearest
            if (dist_c <= candidates[0].first * 1.5f) {
                is_diverse = false;
            }
        }

        if (is_diverse) {
            R.push_back(c_idx);
        } else if (keep_pruned || extend_candidates) {
            W_d.push_back(c_idx);
        }
    }

    // If extend_candidates is true and we have room, add some discarded back
    // This implements the "extended candidate set" from the paper
    if (extend_candidates && R.size() < M && !W_d.empty()) {
        std::size_t to_add = std::min(M - R.size(), W_d.size());
        for (std::size_t i = 0; i < to_add; ++i) {
            R.push_back(W_d[i]);
        }
        // Remove added elements from W_d if not keeping pruned
        if (!keep_pruned) {
            W_d.erase(W_d.begin(), W_d.begin() + to_add);
        }
    }

    return {R, W_d};
}


auto HnswIndex::Impl::reachable_count_base_layer() const -> std::size_t {
    if (state_.n_elements == 0) return 0;

    std::uint32_t ep_idx_snapshot;
    {
        std::lock_guard<std::mutex> lock(graph_mutex_);
        ep_idx_snapshot = state_.entry_point;
    }

    std::vector<char> visited(state_.n_elements, 0);
    std::queue<std::uint32_t> q;
    if (ep_idx_snapshot < state_.n_elements && !nodes_[ep_idx_snapshot]->deleted.load()) {
        visited[ep_idx_snapshot] = 1;
        q.push(ep_idx_snapshot);
    }

    std::size_t count = 0;
    while (!q.empty()) {
        auto current = q.front(); q.pop();
        ++count;

        std::vector<std::uint32_t> neighbors_copy;
        {
            std::lock_guard<std::mutex> g(*node_mutexes_[current]);
            if (!nodes_[current]->neighbors.empty()) {
                neighbors_copy = nodes_[current]->neighbors[0];
            }
        }

        for (std::uint32_t nb : neighbors_copy) {
            if (nb >= state_.n_elements) continue;
            if (nodes_[nb]->deleted.load()) continue;
            if (!visited[nb]) {
                visited[nb] = 1;
                q.push(nb);
            }
        }
    }

    return count;
}

auto HnswIndex::reachable_count_base_layer() const noexcept -> std::size_t {
    return impl_->reachable_count_base_layer();
}

} // namespace vesper::index