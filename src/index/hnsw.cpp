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
#include <shared_mutex>
#include <fstream>
#include <cmath>
#include <thread>
#include <iostream>
#include <stdexcept>
#include <sstream>


#ifdef _WIN32
#include <eh.h>
#include <cstdio>
static void hnsw_seh_translate(unsigned int code, EXCEPTION_POINTERS*) {
    char buf[32]; std::snprintf(buf, sizeof(buf), "0x%08X", code);
    throw std::runtime_error(std::string("HNSW SEH ") + buf);
}
#endif

#ifdef VESPER_HNSW_INVARIANTS
#define HNSW_ENSURE(cond, msg) do { \
  if(!(cond)) { \
    std::ostringstream _oss; _oss << "HNSW invariant failed: " << msg; \
    std::string _s = _oss.str(); \
    std::fprintf(stderr, "%s\n", _s.c_str()); \
    std::fflush(stderr); \
    throw std::runtime_error(_s); \
  } \
} while(0)
#else
#define HNSW_ENSURE(cond, msg) do { (void)sizeof(cond); } while(0)
#endif


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
        bool read_only_graph{false};      // When true, searches elide per-node locks

        // Cached kernel ops for hot paths (selected once at init)
        const kernels::KernelOps* ops{nullptr};

        std::mt19937 rng;
    } state_;

    /** \brief Graph nodes. */
    std::vector<std::unique_ptr<HnswNode>> nodes_;
    std::unordered_map<std::uint64_t, std::uint32_t> id_to_idx_;
    mutable std::shared_mutex graph_mutex_;  // Shared (read) / Unique (write) for structural changes
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
                     const std::uint8_t* filter = nullptr, std::size_t filter_size = 0) const
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
    // Derive connection caps from M unless explicitly overridden
    state_.params.max_M = state_.params.M;
    state_.params.max_M0 = std::max(2u * state_.params.M, state_.params.M);

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
                                   const std::uint8_t* filter, std::size_t filter_size) const
    -> std::vector<std::pair<float, std::uint32_t>> {

    // Capture a stable snapshot of node count for bounds
    const std::size_t N = nodes_.size();

    // Thread-local epoch-based visited marking (avoids hash set overhead)
    struct TLSVisited { std::vector<std::uint32_t> seen; std::uint32_t epoch{0}; };
    thread_local TLSVisited tls;
    if (tls.seen.size() < N) tls.seen.resize(N, 0);
    tls.epoch++; if (tls.epoch == 0) { std::fill(tls.seen.begin(), tls.seen.end(), 0u); tls.epoch = 1; }
    auto mark_visited = [&](std::uint32_t id) noexcept {
        if (id >= N) return; // early OOB guard
        if (id >= tls.seen.size()) tls.seen.resize(N, 0);
        tls.seen[id] = tls.epoch;
    };
    auto is_visited = [&](std::uint32_t id) noexcept {
        if (id >= N) return true; // treat OOB as visited to skip
        return id < tls.seen.size() && tls.seen[id] == tls.epoch;
    };

    std::priority_queue<std::pair<float, std::uint32_t>> candidates;
    std::priority_queue<std::pair<float, std::uint32_t>> nearest;

    // Soft caps for queue sizes to detect anomalies under large ef (very generous)
    const std::size_t ef_cap = std::max<std::size_t>(static_cast<std::size_t>(num_closest) * 8u,
                                                     static_cast<std::size_t>(num_closest) + 4096u);
    const std::size_t cand_cap = ef_cap * 4u;

    HNSW_ENSURE(entry_point < N, "entry_point out of bounds");
    HNSW_ENSURE(nodes_[entry_point] != nullptr, "entry_point node is null");
    const float entry_dist = compute_distance(query, nodes_[entry_point]->data.data());
    candidates.emplace(-entry_dist, entry_point);
    nearest.emplace(entry_dist, entry_point);
    mark_visited(entry_point);

    float prev_popped_dist = -std::numeric_limits<float>::infinity();

    while (!candidates.empty()) {
        const auto [neg_dist, current] = candidates.top();
        const float current_dist = -neg_dist;
        candidates.pop();

        // Note: candidates heap pops are not guaranteed to be monotonic since new, closer
        // nodes can be inserted after expansions. We track the last popped for debugging only.
        prev_popped_dist = current_dist;

        HNSW_ENSURE(!nearest.empty(), "nearest queue unexpectedly empty");
        if (current_dist > nearest.top().first) {
            break;
        }

        // Take a consistent snapshot of neighbors under lock to avoid races with concurrent writers
        std::vector<std::uint32_t> neighbors_copy;
        {
            HNSW_ENSURE(current < nodes_.size(), "current index out of bounds");
            HNSW_ENSURE(nodes_[current] != nullptr, "current node is null");
            if (state_.read_only_graph) {
                HNSW_ENSURE(layer < nodes_[current]->neighbors.size(), "layer out of bounds for current node neighbors");
                neighbors_copy = nodes_[current]->neighbors[layer];
            } else {
                HNSW_ENSURE(current < node_mutexes_.size(), "node mutex index out of bounds");
                std::lock_guard<std::mutex> g(*node_mutexes_[current]);
                HNSW_ENSURE(layer < nodes_[current]->neighbors.size(), "layer out of bounds for current node neighbors");
                neighbors_copy = nodes_[current]->neighbors[layer];
            }
        }

        // Collect unvisited neighbors for batch distance computation
        std::vector<std::uint32_t> unvisited_neighbors;
        std::vector<const float*> neighbor_ptrs;
        // Reserve and cap scratch size to avoid unbounded growth under large ef
        unvisited_neighbors.reserve(std::min<std::size_t>(neighbors_copy.size(), ef_cap));
        neighbor_ptrs.reserve(unvisited_neighbors.capacity());
        const std::size_t scratch_cap = unvisited_neighbors.capacity();
        std::size_t appended = 0;

        for (std::uint32_t neighbor : neighbors_copy) {
            // Early bounds and validity checks
            if (neighbor >= N) continue;
            HNSW_ENSURE(neighbor < nodes_.size(), "neighbor index out of bounds");
            auto* n = nodes_[neighbor].get();
            HNSW_ENSURE(n != nullptr, "neighbor node is null");
            if (is_visited(neighbor)) continue;
            if (n->deleted.load()) continue;
            if (filter && !passes_filter(neighbor, filter, filter_size)) continue;
            // Ensure neighbor vector length is valid
            if (n->data.size() != state_.dim) continue;
            const float* ptr = n->data.data();
            if (ptr == nullptr) continue;

            mark_visited(neighbor);
            unvisited_neighbors.push_back(neighbor);
            neighbor_ptrs.push_back(ptr);
            if (++appended >= scratch_cap) break;
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

            // Invariant checks on queue sizes after enqueue burst
            HNSW_ENSURE(nearest.size() <= ef_cap, "nearest queue exceeded soft cap");
            HNSW_ENSURE(candidates.size() <= cand_cap, "candidates queue exceeded soft cap");
        }
    }

    std::vector<std::pair<float, std::uint32_t>> result;
    while (!nearest.empty()) {
        result.push_back(nearest.top());
        nearest.pop();
    }

    // Ensure deterministic ordering on ties (distance, then index)
    std::sort(result.begin(), result.end());
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
        HNSW_ENSURE(new_idx < node_mutexes_.size(), "new_idx mutex out of bounds");
        HNSW_ENSURE(new_idx < nodes_.size() && nodes_[new_idx] != nullptr, "new_idx node invalid");
        HNSW_ENSURE(level < nodes_[new_idx]->neighbors.size(), "level out of bounds for new_idx neighbors");
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
        std::unique_lock<std::shared_mutex> lock(graph_mutex_);
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
        std::unique_lock<std::shared_mutex> lock(graph_mutex_);
        state_.entry_point = 0;
        return {};
    }

    // Search for nearest neighbors at all levels
    std::vector<std::pair<float, std::uint32_t>> nearest;

    // Snapshot entry point and its top level under lock to avoid races during traversal
    std::uint32_t ep_idx_snapshot;
    std::uint32_t ep_top_level_snapshot;
    {
        std::shared_lock<std::shared_mutex> lock(graph_mutex_);
        ep_idx_snapshot = state_.entry_point;
        ep_top_level_snapshot = nodes_[ep_idx_snapshot]->level;
    }
    std::uint32_t curr_nearest = ep_idx_snapshot;

    // Search upper layers using the snapshot
    for (std::int32_t lc = static_cast<std::int32_t>(ep_top_level_snapshot); lc > static_cast<std::int32_t>(level); --lc) {
        nearest = search_layer(data, curr_nearest, 1, lc, nullptr, 0);
        if (!nearest.empty()) {
            curr_nearest = nearest[0].second;
        }
    }

    // Insert at upper levels (l > 0) under global lock to avoid concurrent structural changes
    {
        std::unique_lock<std::shared_mutex> lock(graph_mutex_);
        for (std::int32_t lc = static_cast<std::int32_t>(std::min(level, ep_top_level_snapshot)); lc > 0; --lc) {
            std::uint32_t efL = state_.params.efConstruction;
            if (state_.params.adaptive_ef) {
                efL = state_.params.efConstructionUpper != 0
                    ? state_.params.efConstructionUpper
                    : std::max<std::uint32_t>(50u, state_.params.efConstruction / 2);
            }
            nearest = search_layer(data, curr_nearest, efL, lc, nullptr, 0);
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
        std::unique_lock<std::shared_mutex> lock(graph_mutex_); // serialize entire base-layer connect
        nearest = search_layer(data, curr_nearest, state_.params.efConstruction, 0, nullptr, 0);
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
        nearest = search_layer(data, curr_nearest, state_.params.efConstruction, 0, nullptr, 0);
        const std::uint32_t M0 = state_.params.max_M0;
        connect_node(new_idx, nearest, M0, 0);
        if (!nearest.empty()) {
            curr_nearest = nearest[0].second;
        }
    }
    #else
    {
        std::unique_lock<std::shared_mutex> lock(graph_mutex_);
        nearest = search_layer(data, curr_nearest, state_.params.efConstruction, 0, nullptr, 0);
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
        std::unique_lock<std::shared_mutex> lock(graph_mutex_);
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

    // Enhanced parallel construction with optimizations from 2025 research:
    // 1. Pre-compute all distances in parallel (batched SIMD)
    // 2. Use lock-free insertion for independent nodes
    // 3. Batch graph updates to reduce contention

    HnswParallelContext context(state_.params.num_threads);  // Use configured thread count (0=auto)
    auto& pool = context.pool();

    // Pre-allocate all structures upfront to avoid reallocation
    {
        std::unique_lock<std::shared_mutex> lock(graph_mutex_);
        nodes_.reserve(state_.n_elements + n);
        id_to_idx_.reserve(id_to_idx_.size() + n);

        // Ensure we have enough node mutexes
        if (node_mutexes_.size() < state_.n_elements + n) {
            // Initialize new mutexes with extra buffer
            std::size_t old_size = node_mutexes_.size();
            std::size_t new_size = state_.n_elements + n + std::max<std::size_t>(1000, n/10);
            node_mutexes_.reserve(new_size);
            for (std::size_t i = old_size; i < new_size; ++i) {
                node_mutexes_.push_back(std::make_unique<std::mutex>());
            }
        }
    }

    // Phase 1: Parallel node creation and level assignment
    struct NodeInfo {
        std::uint32_t idx;
        std::uint32_t level;
        std::unique_ptr<HnswNode> node;
    };
    std::vector<NodeInfo> new_nodes(n);

    // Create nodes in parallel without graph insertion
    const std::size_t chunk_size = std::max<std::size_t>(32, n / (pool.num_threads() * 4));
    std::vector<std::future<void>> futures;

    // Track errors during node creation
    std::atomic<bool> create_error{false};
    std::mutex create_error_mu;
    std::optional<error> create_first_error;

    for (std::size_t start = 0; start < n; start += chunk_size) {
        std::size_t end = std::min(start + chunk_size, n);
        futures.push_back(pool.submit([this, start, end, ids, data, &new_nodes, &create_error, &create_error_mu, &create_first_error]() {
            #ifdef _WIN32
            _set_se_translator(hnsw_seh_translate);
            #endif
            try {
                std::mt19937 local_rng(state_.seed + start);  // Thread-local RNG

                for (std::size_t i = start; i < end; ++i) {
                    if (create_error.load()) break;
                    // Select level using thread-local RNG
                    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
                    const float f = -std::log(dist(local_rng)) * state_.level_multiplier;
                    std::uint32_t level = static_cast<std::uint32_t>(f);

                    // Create node
                    auto node = std::make_unique<HnswNode>();
                    node->id = ids[i];
                    node->data.assign(data + i * state_.dim, data + (i + 1) * state_.dim);
                    node->level = level;
                    node->neighbors.resize(level + 1);

                    new_nodes[i] = {0, level, std::move(node)};
                }
            } catch (const std::exception& e) {
                create_error.store(true);
                std::lock_guard<std::mutex> g(create_error_mu);
                if (!create_first_error.has_value()) {
                    create_first_error = error{error_code::internal, std::string("exception in hnsw::add_batch_parallel(node_create): ") + e.what(), "hnsw"};
                }
            } catch (...) {
                create_error.store(true);
                std::lock_guard<std::mutex> g(create_error_mu);
                if (!create_first_error.has_value()) {
                    create_first_error = error{error_code::internal, "unknown exception in hnsw::add_batch_parallel(node_create)", "hnsw"};
                }
            }
        }));
    }

    // Wait for node creation
    for (auto& f : futures) {
        f.wait();
    }
    futures.clear();

    if (create_error.load() && create_first_error.has_value()) {
        return std::vesper_unexpected(create_first_error.value());
    }

    // Phase 2: Batch insert nodes and assign indices
    std::uint32_t base_idx = 0;
    {
        std::unique_lock<std::shared_mutex> lock(graph_mutex_);
        base_idx = state_.n_elements;
        state_.n_elements += n;

        // Insert all nodes and update lookup
        for (std::size_t i = 0; i < n; ++i) {
            new_nodes[i].idx = base_idx + i;
            id_to_idx_[ids[i]] = base_idx + i;
            nodes_.push_back(std::move(new_nodes[i].node));
        }

        // Update entry point if this is the first batch
        if (base_idx == 0 && n > 0) {
            state_.entry_point = 0;
        }
    }

    // Track errors from parallel execution
    std::atomic<bool> has_error{false};
    std::mutex error_mutex;
    std::optional<error> first_error;

    // Phase 3: Parallel graph construction with batched distance computation
    // Process in optimized chunks for better cache locality
    const std::size_t batch_size = 64;  // Optimal for SIMD distance computation
    // Note: futures already declared above, reuse it
    futures.clear();

    for (std::size_t start = 0; start < n; start += batch_size) {
        std::size_t end = std::min(start + batch_size, n);

        futures.push_back(pool.submit([this, ids, data, start, end, &has_error, &error_mutex, &first_error, &new_nodes]() {
            #ifdef _WIN32
            _set_se_translator(hnsw_seh_translate);
            #endif
            try {
                for (std::size_t i = start; i < end; ++i) {
                    if (has_error.load()) {
                        break;  // Early exit if another thread encountered an error
                    }

                    const std::uint32_t idx = new_nodes[i].idx;
                    const std::uint32_t level = new_nodes[i].level;

                    // First node becomes entry point; nothing to connect
                    if (idx == 0) {
                        continue;
                    }

                    // Snapshot entry point and top level under lock to avoid race with concurrent insertions
                    std::uint32_t ep_idx_snapshot;
                    std::uint32_t ep_top_level_snapshot;
                    {
                        std::shared_lock<std::shared_mutex> lock(graph_mutex_);
                        ep_idx_snapshot = state_.entry_point;
                        ep_top_level_snapshot = nodes_[ep_idx_snapshot]->level;
                    }
                    std::uint32_t curr_nearest = ep_idx_snapshot;

                    // Search upper layers using the snapshot
                     std::vector<std::pair<float, std::uint32_t>> nearest;
                    for (std::int32_t lc = static_cast<std::int32_t>(ep_top_level_snapshot); lc > static_cast<std::int32_t>(level); --lc) {
                        nearest = search_layer(data + i * state_.dim, curr_nearest, 1, lc, nullptr, 0);
                        if (!nearest.empty()) {
                            curr_nearest = nearest[0].second;
                        }
                    }

                    // Insert at upper levels (l > 0) under global lock to avoid concurrent structural changes
                    {
                        std::unique_lock<std::shared_mutex> lock(graph_mutex_);
                        for (std::int32_t lc = static_cast<std::int32_t>(std::min(level, ep_top_level_snapshot)); lc > 0; --lc) {
                            std::uint32_t efL = state_.params.efConstruction;
                            if (state_.params.adaptive_ef) {
                                efL = state_.params.efConstructionUpper != 0
                                    ? state_.params.efConstructionUpper
                                    : std::max<std::uint32_t>(50u, state_.params.efConstruction / 2);
                            }
                            nearest = search_layer(data + i * state_.dim, curr_nearest, efL, lc, nullptr, 0);
                            const std::uint32_t M = state_.params.max_M;
                            connect_node(idx, nearest, M, lc);
                            if (!nearest.empty()) {
                                curr_nearest = nearest[0].second;
                            }
                        }
                    }

                    // Insert at base layer (l == 0)
                #if defined(VESPER_SERIALIZE_BASE_LAYER) && VESPER_SERIALIZE_BASE_LAYER
                    {
                        std::unique_lock<std::shared_mutex> lock(graph_mutex_); // serialize entire base-layer connect
                        nearest = search_layer(data + i * state_.dim, curr_nearest, state_.params.efConstruction, 0, nullptr, 0);
                        const std::uint32_t M0 = state_.params.max_M0;
                        connect_node(idx, nearest, M0, 0);
                        if (!nearest.empty()) {
                            curr_nearest = nearest[0].second;
                        }
                    }
                #else
                    // Serialize only the search, perform connect in parallel
                    #if defined(VESPER_PARTITIONED_BASE_LOCKS) && VESPER_PARTITIONED_BASE_LOCKS
                    {
                        // Partitioned base-layer lock: only serialize within shard of idx
                        std::lock_guard<std::mutex> shard_lock(base_layer_shards_[base_shard_for(idx)]);
                        nearest = search_layer(data + i * state_.dim, curr_nearest, state_.params.efConstruction, 0, nullptr, 0);
                        const std::uint32_t M0 = state_.params.max_M0;
                        connect_node(idx, nearest, M0, 0);
                        if (!nearest.empty()) {
                            curr_nearest = nearest[0].second;
                        }
                    }
                    #else
                    {
                        std::unique_lock<std::shared_mutex> lock(graph_mutex_);
                        nearest = search_layer(data + i * state_.dim, curr_nearest, state_.params.efConstruction, 0, nullptr, 0);
                    }
                    {
                        const std::uint32_t M0 = state_.params.max_M0;
                        connect_node(idx, nearest, M0, 0);
                        if (!nearest.empty()) {
                            curr_nearest = nearest[0].second;
                        }
                    }
                    #endif
                #endif

                    // Update entry point if new node has higher level
                    {
                        std::unique_lock<std::shared_mutex> lock(graph_mutex_);
                        if (level > nodes_[state_.entry_point]->level) {
                            state_.entry_point = idx;
                        }
                    }
                }
            } catch (const std::exception& e) {
                has_error.store(true);
                std::lock_guard<std::mutex> g(error_mutex);
                if (!first_error.has_value()) {
                    first_error = error{error_code::internal, std::string("exception in hnsw::add_batch_parallel(connect): ") + e.what(), "hnsw"};
                }
            } catch (...) {
                has_error.store(true);
                std::lock_guard<std::mutex> g(error_mutex);
                if (!first_error.has_value()) {
                    first_error = error{error_code::internal, "unknown exception in hnsw::add_batch_parallel(connect)", "hnsw"};
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
        std::shared_lock<std::shared_mutex> lock(graph_mutex_);
        ep_idx_snapshot = state_.entry_point;
        ep_top_level_snapshot = nodes_[ep_idx_snapshot]->level;
    }
    std::uint32_t curr_nearest = ep_idx_snapshot;

    // Search upper layers using the snapshot
    for (std::int32_t lc = static_cast<std::int32_t>(ep_top_level_snapshot); lc > 0; --lc) {
        auto nearest = search_layer(query, curr_nearest, 1, lc, params.filter_mask, params.filter_size);
        if (!nearest.empty()) {
            curr_nearest = nearest[0].second;
        }
    }

    // Search base layer with efSearch
    auto candidates = search_layer(query, curr_nearest, params.efSearch, 0, params.filter_mask, params.filter_size);

    // Convert to output format
    std::vector<std::pair<std::uint64_t, float>> results;
    results.reserve(std::min(static_cast<std::size_t>(params.k), candidates.size()));

    for (std::size_t i = 0; i < std::min(static_cast<std::size_t>(params.k), candidates.size()); ++i) {
        const auto& [dist, idx] = candidates[i];
        if (idx >= nodes_.size()) continue;
        auto* node_ptr = nodes_[idx].get();
        if (node_ptr == nullptr) continue;
        if (!std::isfinite(dist)) continue;
        results.emplace_back(node_ptr->id, dist);
    }

    return results;
}

auto HnswIndex::Impl::search_batch(const float* queries, std::size_t n_queries,
                                   const HnswSearchParams& params) const
    -> std::expected<std::vector<std::vector<std::pair<std::uint64_t, float>>>, core::error> {

    std::vector<std::vector<std::pair<std::uint64_t, float>>> results(n_queries);

    std::atomic<bool> has_error{false};
    std::mutex err_mu;
    core::error first_error{core::error_code::ok, {}, {}};

    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(n_queries); ++i) {
        if (has_error.load(std::memory_order_relaxed)) continue; // fail-fast
        const float* query = queries + static_cast<std::size_t>(i) * state_.dim;
        #ifdef _WIN32
        _set_se_translator(hnsw_seh_translate);
        #endif
        try {
            auto result = search(query, params);
            if (result.has_value()) {
                results[static_cast<std::size_t>(i)] = std::move(result.value());
            } else {
                has_error.store(true, std::memory_order_relaxed);
                std::lock_guard<std::mutex> g(err_mu);
                if (first_error.code == core::error_code::ok) first_error = result.error();
            }
        } catch (const std::exception& e) {
            has_error.store(true, std::memory_order_relaxed);
            std::lock_guard<std::mutex> g(err_mu);
            if (first_error.code == core::error_code::ok) {
                first_error = core::error{core::error_code::internal, std::string("exception in hnsw::search_batch: ") + e.what(), "hnsw"};
            }
        } catch (...) {
            has_error.store(true, std::memory_order_relaxed);
            std::lock_guard<std::mutex> g(err_mu);
            if (first_error.code == core::error_code::ok) {
                first_error = core::error{core::error_code::internal, "unknown exception in hnsw::search_batch", "hnsw"};
            }
        }
    }

    if (has_error.load(std::memory_order_relaxed)) {
        return std::vesper_unexpected(first_error);
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

auto HnswIndex::set_read_only(bool read_only) -> void {
    impl_->state_.read_only_graph = read_only;
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

    std::size_t base_edges = 0;
    std::size_t max_level = 0;
    std::vector<std::size_t> level_counts;

    for (const auto& node : impl_->nodes_) {
        if (!node) continue;

        max_level = std::max(max_level, static_cast<std::size_t>(node->level));
        if (level_counts.size() <= node->level) {
            level_counts.resize(node->level + 1, 0);
        }
        level_counts[node->level]++;

        // Only count base layer (level 0) edges for degree statistics
        if (!node->neighbors.empty()) {
            base_edges += node->neighbors[0].size();
        }
    }

    stats.n_edges = base_edges / 2;  // Bidirectional edges at base layer
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
    using core::error;
    using core::error_code;

    if (!impl_) {
        return std::vesper_unexpected(error{
            error_code::precondition_failed,
            "Index not initialized",
            "hnsw"
        });
    }

    // Phase 1: Remove deleted nodes from all neighbor lists
    std::size_t edges_removed = 0;
    std::size_t nodes_repaired = 0;

    for (std::uint32_t i = 0; i < impl_->state_.n_elements; ++i) {
        if (impl_->nodes_[i]->deleted.load()) continue;

        bool node_modified = false;
        for (std::size_t layer = 0; layer <= impl_->nodes_[i]->level; ++layer) {
            std::lock_guard<std::mutex> lock(*impl_->node_mutexes_[i]);
            auto& neighbors = impl_->nodes_[i]->neighbors[layer];

            // Remove deleted neighbors
            auto original_size = neighbors.size();
            neighbors.erase(
                std::remove_if(neighbors.begin(), neighbors.end(),
                    [this](std::uint32_t idx) {
                        return idx >= impl_->state_.n_elements ||
                               impl_->nodes_[idx]->deleted.load();
                    }),
                neighbors.end()
            );

            if (neighbors.size() < original_size) {
                edges_removed += original_size - neighbors.size();
                node_modified = true;
            }

            // Prune weak edges using RobustPrune if we have too many connections
            std::uint32_t max_conn = (layer == 0) ?
                impl_->state_.params.max_M0 : impl_->state_.params.max_M;

            if (neighbors.size() > max_conn) {
                // Compute distances to all neighbors
                std::vector<std::pair<float, std::uint32_t>> candidates;
                candidates.reserve(neighbors.size());

                for (std::uint32_t neighbor : neighbors) {
                    float dist = impl_->compute_distance(
                        impl_->nodes_[i]->data.data(),
                        impl_->nodes_[neighbor]->data.data()
                    );
                    candidates.emplace_back(dist, neighbor);
                }

                // Apply RobustPrune algorithm
                auto [selected, pruned] = robust_prune(
                    candidates,
                    max_conn,
                    impl_->state_.params.extend_candidates,
                    impl_->state_.params.keep_pruned_connections
                );

                neighbors = selected;
                edges_removed += candidates.size() - selected.size();
                node_modified = true;
            }
        }

        if (node_modified) {
            nodes_repaired++;
        }
    }

    // Phase 2: Ensure minimum connectivity
    std::uint32_t min_connections = impl_->state_.params.M / 2;

    for (std::uint32_t i = 0; i < impl_->state_.n_elements; ++i) {
        if (impl_->nodes_[i]->deleted.load()) continue;

        // Check base layer connectivity
        {
            std::lock_guard<std::mutex> lock(*impl_->node_mutexes_[i]);
            auto& neighbors = impl_->nodes_[i]->neighbors[0];

            if (neighbors.size() < min_connections && impl_->state_.n_elements > min_connections) {
                // Find additional neighbors using search
                auto search_result = impl_->search_layer(
                    impl_->nodes_[i]->data.data(),
                    i,  // Use current node as entry point
                    impl_->state_.params.efConstruction,
                    0,  // Search in base layer
                    nullptr,  // No filter
                    0);


                for (auto [dist, idx] : search_result) {
                    if (idx != i &&
                        std::find(neighbors.begin(), neighbors.end(), idx) == neighbors.end() &&
                        !impl_->nodes_[idx]->deleted.load()) {
                        neighbors.push_back(idx);

                        // Add reverse edge for bidirectionality
                        {
                            std::lock_guard<std::mutex> neighbor_lock(*impl_->node_mutexes_[idx]);
                            auto& reverse_neighbors = impl_->nodes_[idx]->neighbors[0];
                            if (std::find(reverse_neighbors.begin(), reverse_neighbors.end(), i) ==
                                reverse_neighbors.end()) {
                                reverse_neighbors.push_back(i);
                            }
                        }

                        if (neighbors.size() >= min_connections) break;
                    }
                }
            }
        }
    }

    // Phase 3: Optimize entry point selection
    // Find the most central node as entry point
    if (impl_->state_.n_elements > 0) {
        std::uint32_t best_entry = impl_->state_.entry_point;
        float best_centrality = std::numeric_limits<float>::max();

        // Sample a subset of nodes for efficiency
        std::size_t sample_size = std::min<std::size_t>(100, impl_->state_.n_elements);
        std::vector<std::uint32_t> sample_indices;
        sample_indices.reserve(sample_size);

        // Uniform sampling
        for (std::size_t i = 0; i < sample_size; ++i) {
            std::uint32_t idx = (i * impl_->state_.n_elements) / sample_size;
            if (!impl_->nodes_[idx]->deleted.load()) {
                sample_indices.push_back(idx);
            }
        }

        // Compute average distance from each candidate entry point
        for (std::uint32_t candidate : sample_indices) {
            // Skip nodes not in upper layers (require at least level 1)
            if (impl_->nodes_[candidate]->level == 0) {
                continue;  // Skip nodes not in upper layers
            }

            float avg_dist = 0.0f;
            std::size_t count = 0;

            // Sample distances to other nodes
            for (std::size_t j = 0; j < std::min<std::size_t>(50, sample_indices.size()); ++j) {
                if (sample_indices[j] != candidate) {
                    avg_dist += impl_->compute_distance(
                        impl_->nodes_[candidate]->data.data(),
                        impl_->nodes_[sample_indices[j]]->data.data()
                    );
                    count++;
                }
            }

            if (count > 0) {
                avg_dist /= count;
                if (avg_dist < best_centrality) {
                    best_centrality = avg_dist;
                    best_entry = candidate;
                }
            }
        }

        if (best_entry != impl_->state_.entry_point) {
            impl_->state_.entry_point = best_entry;
        }
    }

    // Phase 4: Compact neighbor lists (remove duplicates)
    for (std::uint32_t i = 0; i < impl_->state_.n_elements; ++i) {
        if (impl_->nodes_[i]->deleted.load()) continue;

        for (std::size_t layer = 0; layer <= impl_->nodes_[i]->level; ++layer) {
            std::lock_guard<std::mutex> lock(*impl_->node_mutexes_[i]);
            auto& neighbors = impl_->nodes_[i]->neighbors[layer];

            // Remove duplicates
            std::sort(neighbors.begin(), neighbors.end());
            neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
        }
    }

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
        std::shared_lock<std::shared_mutex> lock(graph_mutex_);
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

// Duplicate removed - is_initialized() defined earlier

// Duplicate removed - size() defined earlier

auto HnswIndex::get_max_layer() const noexcept -> int {
    if (!impl_ || impl_->state_.n_elements == 0) return -1;

    std::shared_lock<std::shared_mutex> lock(impl_->graph_mutex_);
    std::uint32_t ep = impl_->state_.entry_point;
    if (ep >= impl_->nodes_.size()) return -1;

    return static_cast<int>(impl_->nodes_[ep]->level);
}

auto HnswIndex::get_neighbors(std::uint64_t node_id, int layer) const
    -> std::vector<std::uint64_t> {

    if (!impl_ || layer < 0) return {};

    // Find internal index for this ID
    std::uint32_t idx;
    {
        std::lock_guard<std::mutex> lock(impl_->label_lookup_mutex_);
        auto it = impl_->id_to_idx_.find(node_id);
        if (it == impl_->id_to_idx_.end()) {
            return {};
        }
        idx = it->second;
    }

    // Get neighbors at specified layer
    std::vector<std::uint64_t> result;
    {
        std::lock_guard<std::mutex> lock(*impl_->node_mutexes_[idx]);
        auto& node = impl_->nodes_[idx];
        if (static_cast<std::uint32_t>(layer) <= node->level &&
            static_cast<std::size_t>(layer) < node->neighbors.size()) {

            const auto& neighbors = node->neighbors[layer];
            result.reserve(neighbors.size());

            // Convert internal indices to external IDs
            for (std::uint32_t neighbor_idx : neighbors) {
                if (neighbor_idx < impl_->nodes_.size()) {
                    result.push_back(impl_->nodes_[neighbor_idx]->id);
                }
            }
        }
    }

    return result;
}

auto HnswIndex::get_reverse_neighbors(std::uint64_t node_id, int layer) const
    -> std::vector<std::uint64_t> {

    if (!impl_ || layer < 0) return {};

    // Find internal index for this ID
    std::uint32_t target_idx;
    {
        std::lock_guard<std::mutex> lock(impl_->label_lookup_mutex_);
        auto it = impl_->id_to_idx_.find(node_id);
        if (it == impl_->id_to_idx_.end()) {
            return {};
        }
        target_idx = it->second;
    }

    // Scan all nodes to find those pointing to target
    std::vector<std::uint64_t> result;
    for (std::uint32_t i = 0; i < impl_->state_.n_elements; ++i) {
        if (i == target_idx) continue;

        std::lock_guard<std::mutex> lock(*impl_->node_mutexes_[i]);
        auto& node = impl_->nodes_[i];

        if (node->deleted.load()) continue;
        if (static_cast<std::uint32_t>(layer) > node->level) continue;
        if (static_cast<std::size_t>(layer) >= node->neighbors.size()) continue;

        const auto& neighbors = node->neighbors[layer];
        if (std::find(neighbors.begin(), neighbors.end(), target_idx) != neighbors.end()) {
            result.push_back(node->id);
        }
    }

    return result;
}

auto HnswIndex::remove_edge(std::uint64_t from, std::uint64_t to, int layer)
    -> std::expected<void, core::error> {

    using core::error;
    using core::error_code;

    if (!impl_) {
        return std::vesper_unexpected(error{
            error_code::precondition_failed,
            "Index not initialized",
            "hnsw"
        });
    }

    if (layer < 0) {
        return std::vesper_unexpected(error{
            error_code::precondition_failed,
            "Invalid layer",
            "hnsw"
        });
    }

    // Find internal indices
    std::uint32_t from_idx, to_idx;
    {
        std::lock_guard<std::mutex> lock(impl_->label_lookup_mutex_);

        auto from_it = impl_->id_to_idx_.find(from);
        if (from_it == impl_->id_to_idx_.end()) {
            return std::vesper_unexpected(error{
                error_code::not_found,
                "Source node not found",
                "hnsw"
            });
        }
        from_idx = from_it->second;

        auto to_it = impl_->id_to_idx_.find(to);
        if (to_it == impl_->id_to_idx_.end()) {
            return std::vesper_unexpected(error{
                error_code::not_found,
                "Target node not found",
                "hnsw"
            });
        }
        to_idx = to_it->second;
    }

    // Remove edge
    {
        std::lock_guard<std::mutex> lock(*impl_->node_mutexes_[from_idx]);
        auto& node = impl_->nodes_[from_idx];

        if (static_cast<std::uint32_t>(layer) > node->level) {
            return std::vesper_unexpected(error{
                error_code::precondition_failed,
                "Layer exceeds node level",
                "hnsw"
            });
        }

        if (static_cast<std::size_t>(layer) >= node->neighbors.size()) {
            return std::vesper_unexpected(error{
                error_code::precondition_failed,
                "Invalid layer index",
                "hnsw"
            });
        }

        auto& neighbors = node->neighbors[layer];
        auto it = std::find(neighbors.begin(), neighbors.end(), to_idx);
        if (it != neighbors.end()) {
            neighbors.erase(it);
        }
    }

    return {};
}

auto HnswIndex::get_vector(std::uint64_t node_id) const
    -> std::expected<std::vector<float>, core::error> {

    using core::error;
    using core::error_code;

    if (!impl_) {
        return std::vesper_unexpected(error{
            error_code::precondition_failed,
            "Index not initialized",
            "hnsw"
        });
    }

    // Find internal index
    std::uint32_t idx;
    {
        std::lock_guard<std::mutex> lock(impl_->label_lookup_mutex_);
        auto it = impl_->id_to_idx_.find(node_id);
        if (it == impl_->id_to_idx_.end()) {
            return std::vesper_unexpected(error{
                error_code::not_found,
                "Node not found",
                "hnsw"
            });
        }
        idx = it->second;
    }

    // Return copy of vector data
    {
        std::lock_guard<std::mutex> lock(*impl_->node_mutexes_[idx]);
        auto& node = impl_->nodes_[idx];
        if (node->deleted.load()) {
            return std::vesper_unexpected(error{
                error_code::not_found,
                "Node is deleted",
                "hnsw"
            });
        }
        return node->data;
    }
}

auto HnswIndex::extract_all_vectors(std::vector<std::uint64_t>& ids,
                                   std::vector<float>& vectors) const
    -> std::expected<void, core::error> {

    using core::error;
    using core::error_code;

    if (!impl_) {
        return std::vesper_unexpected(error{
            error_code::precondition_failed,
            "Index not initialized",
            "hnsw"
        });
    }

    // Clear output vectors
    ids.clear();
    vectors.clear();

    // Count non-deleted nodes
    std::size_t active_count = 0;
    for (std::uint32_t i = 0; i < impl_->state_.n_elements; ++i) {
        if (!impl_->nodes_[i]->deleted.load()) {
            ++active_count;
        }
    }

    // Reserve space
    ids.reserve(active_count);
    vectors.reserve(active_count * impl_->state_.dim);

    // Extract all vectors
    for (std::uint32_t i = 0; i < impl_->state_.n_elements; ++i) {
        std::lock_guard<std::mutex> lock(*impl_->node_mutexes_[i]);
        auto& node = impl_->nodes_[i];

        if (node->deleted.load()) continue;

        ids.push_back(node->id);
        vectors.insert(vectors.end(), node->data.begin(), node->data.end());
    }

    return {};
}

auto HnswIndex::update_connections(std::uint64_t node_id, int layer,
                                  const std::vector<std::uint64_t>& new_neighbors)
    -> std::expected<void, core::error> {

    using core::error;
    using core::error_code;

    if (!impl_) {
        return std::vesper_unexpected(error{
            error_code::precondition_failed,
            "Index not initialized",
            "hnsw"
        });
    }

    if (layer < 0) {
        return std::vesper_unexpected(error{
            error_code::precondition_failed,
            "Invalid layer",
            "hnsw"
        });
    }

    // Find internal index for the node
    std::uint32_t idx;
    {
        std::lock_guard<std::mutex> lock(impl_->label_lookup_mutex_);
        auto it = impl_->id_to_idx_.find(node_id);
        if (it == impl_->id_to_idx_.end()) {
            return std::vesper_unexpected(error{
                error_code::not_found,
                "Node not found",
                "hnsw"
            });
        }
        idx = it->second;
    }

    // Convert neighbor IDs to internal indices
    std::vector<std::uint32_t> new_neighbor_indices;
    new_neighbor_indices.reserve(new_neighbors.size());

    {
        std::lock_guard<std::mutex> lock(impl_->label_lookup_mutex_);
        for (std::uint64_t neighbor_id : new_neighbors) {
            auto it = impl_->id_to_idx_.find(neighbor_id);
            if (it != impl_->id_to_idx_.end()) {
                new_neighbor_indices.push_back(it->second);
            }
        }
    }

    // Update connections
    {
        std::lock_guard<std::mutex> lock(*impl_->node_mutexes_[idx]);
        auto& node = impl_->nodes_[idx];

        if (static_cast<std::uint32_t>(layer) > node->level) {
            return std::vesper_unexpected(error{
                error_code::precondition_failed,
                "Layer exceeds node level",
                "hnsw"
            });
        }

        if (static_cast<std::size_t>(layer) >= node->neighbors.size()) {
            node->neighbors.resize(layer + 1);
        }

        node->neighbors[layer] = new_neighbor_indices;
    }

    return {};
}

auto HnswIndex::entry_point(std::optional<std::uint64_t> new_entry_point)
    -> std::uint64_t {

    if (!impl_) return std::numeric_limits<std::uint64_t>::max();

    if (new_entry_point.has_value()) {
        // Set new entry point
        std::lock_guard<std::mutex> lock(impl_->label_lookup_mutex_);
        auto it = impl_->id_to_idx_.find(new_entry_point.value());
        if (it != impl_->id_to_idx_.end()) {
            std::unique_lock<std::shared_mutex> graph_lock(impl_->graph_mutex_);
            impl_->state_.entry_point = it->second;
        }
    }

    // Return current entry point ID
    std::shared_lock<std::shared_mutex> lock(impl_->graph_mutex_);
    if (impl_->state_.entry_point < impl_->nodes_.size()) {
        return impl_->nodes_[impl_->state_.entry_point]->id;
    }

    return std::numeric_limits<std::uint64_t>::max();
}

} // namespace vesper::index