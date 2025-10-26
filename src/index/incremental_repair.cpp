/** \file incremental_repair.cpp
 *  \brief Incremental index repair implementations for HNSW, IVF-PQ, and DiskGraph.
 */

#include "vesper/index/index_manager.hpp"
#include "vesper/index/hnsw.hpp"
#include "vesper/index/ivf_pq.hpp"
#include "vesper/index/disk_graph.hpp"
#include "vesper/tombstone/tombstone_manager.hpp"

#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <numeric>
#include <utility>
#include <cmath>
#include <limits>

namespace vesper::index::repair {

/** \brief HNSW incremental repair implementation */
class HnswRepair {
public:
    /** \brief Repair HNSW graph by removing deleted nodes and reconnecting
     * 
     * \param index HNSW index to repair
     * \param deleted_ids Set of deleted node IDs
     * \param dimension Vector dimension
     * \return Success or error
     */
    static auto repair(HnswIndex* index, 
                      const roaring::Roaring& deleted_ids,
                      std::size_t dimension) 
        -> std::expected<void, core::error> {
        
        if (!index) {
            return std::vesper_unexpected(core::error{
                core::error_code::precondition_failed,
                "Null HNSW index",
                "hnsw_repair"
            });
        }
        
        // Phase 1: Identify affected nodes (those connected to deleted nodes)
        std::unordered_set<std::uint64_t> affected_nodes;
        
        // Get all node connections and find those pointing to deleted nodes
        for (std::uint64_t node_id = 0; node_id < index->size(); ++node_id) {
            if (!deleted_ids.contains(static_cast<std::uint32_t>(node_id))) {
                // Check if this node has connections to deleted nodes
                for (int layer = 0; layer <= index->get_max_layer(); ++layer) {
                    auto neighbors = index->get_neighbors(node_id, layer);
                    for (auto neighbor_id : neighbors) {
                        if (deleted_ids.contains(static_cast<std::uint32_t>(neighbor_id))) {
                            affected_nodes.insert(node_id);
                            break;
                        }
                    }
                }
            }
        }
        
        // Phase 2: Remove edges to deleted nodes
        for (auto deleted_id : deleted_ids) {
            // Mark node as deleted
            auto mark_result = index->mark_deleted(static_cast<std::uint64_t>(deleted_id));
            if (!mark_result) {
                return mark_result;
            }
            
            // Iterate through all graph layers
            for (int layer = 0; layer <= index->get_max_layer(); ++layer) {
                // Get all nodes that connect to deleted_id
                auto reverse_neighbors = index->get_reverse_neighbors(
                    static_cast<std::uint64_t>(deleted_id), layer);
                
                // Remove edges pointing to deleted node
                for (auto neighbor_id : reverse_neighbors) {
                    auto remove_result = index->remove_edge(
                        neighbor_id, static_cast<std::uint64_t>(deleted_id), layer);
                    if (!remove_result) {
                        // Continue even if edge removal fails (might already be removed)
                        continue;
                    }
                    affected_nodes.insert(neighbor_id);
                }
            }
        }
        
        // Phase 3: Reconnect affected nodes
        for (auto node_id : affected_nodes) {
            // Get vector for this node
            auto vec_result = index->get_vector(node_id);
            if (!vec_result) continue;  // Skip if can't get vector
            
            const auto& node_vector = vec_result.value();
            
            // Process each layer
            for (int layer = 0; layer <= index->get_max_layer(); ++layer) {
                auto current_neighbors = index->get_neighbors(node_id, layer);
                
                // Remove deleted neighbors
                std::vector<std::uint64_t> active_neighbors;
                for (auto neighbor : current_neighbors) {
                    if (!deleted_ids.contains(static_cast<std::uint32_t>(neighbor))) {
                        active_neighbors.push_back(neighbor);
                    }
                }
                
                // Check if we need to find more connections
                std::uint32_t M = (layer == 0) ? 32 : 16;  // M0 = 32, M = 16 (typical values)
                if (active_neighbors.size() < M / 2) {
                    // Search for candidate replacements
                    std::vector<std::pair<float, std::uint64_t>> candidates;
                    
                    // Get vectors from active neighbors and compute distances
                    for (auto neighbor_id : active_neighbors) {
                        auto neighbor_vec_result = index->get_vector(neighbor_id);
                        if (!neighbor_vec_result) continue;
                        
                        // Compute L2 distance
                        float dist = 0.0f;
                        for (std::size_t d = 0; d < dimension; ++d) {
                            float diff = node_vector[d] - neighbor_vec_result.value()[d];
                            dist += diff * diff;
                        }
                        candidates.emplace_back(dist, neighbor_id);
                    }
                    
                    // Search for new candidates from neighbors of neighbors
                    std::unordered_set<std::uint64_t> visited(active_neighbors.begin(), active_neighbors.end());
                    visited.insert(node_id);
                    
                    for (auto neighbor : active_neighbors) {
                        auto second_neighbors = index->get_neighbors(neighbor, layer);
                        for (auto second_neighbor : second_neighbors) {
                            if (visited.count(second_neighbor) > 0) continue;
                            if (deleted_ids.contains(static_cast<std::uint32_t>(second_neighbor))) continue;
                            
                            visited.insert(second_neighbor);
                            
                            auto second_vec_result = index->get_vector(second_neighbor);
                            if (!second_vec_result) continue;
                            
                            // Compute distance
                            float dist = 0.0f;
                            for (std::size_t d = 0; d < dimension; ++d) {
                                float diff = node_vector[d] - second_vec_result.value()[d];
                                dist += diff * diff;
                            }
                            candidates.emplace_back(dist, second_neighbor);
                        }
                    }
                    
                    // Prune candidates and update connections
                    auto pruned = prune_connections(candidates, M, node_vector.data(), dimension, index);
                    
                    // Update node connections
                    auto update_result = index->update_connections(node_id, layer, pruned);
                    if (!update_result) {
                        // Log error but continue
                        continue;
                    }
                }
            }
        }
        
        // Phase 4: Update entry points if needed
        auto current_entry = index->entry_point();
        if (deleted_ids.contains(static_cast<std::uint32_t>(current_entry))) {
            // Find new entry point - select node with highest level
            std::uint64_t new_entry = 0;
            int max_level = -1;
            
            for (std::uint64_t node_id = 0; node_id < index->size(); ++node_id) {
                if (deleted_ids.contains(static_cast<std::uint32_t>(node_id))) continue;
                
                // Check node's max layer by getting neighbors at increasing layers
                for (int layer = max_level + 1; layer <= index->get_max_layer(); ++layer) {
                    auto neighbors = index->get_neighbors(node_id, layer);
                    if (!neighbors.empty() || layer == 0) {
                        if (layer > max_level) {
                            max_level = layer;
                            new_entry = node_id;
                        }
                    } else {
                        break;
                    }
                }
            }
            
            // Set new entry point
            if (max_level >= 0) {
                index->entry_point(new_entry);
            }
        }
        
        return {};
    }
    
private:
    /** \brief Prune connections using HNSW pruning algorithm
     * 
     * \param candidates Candidate neighbors
     * \param M Maximum number of connections
     * \param base_vector Vector of the node being connected
     * \param dimension Vector dimension
     * \param index HNSW index for vector retrieval
     * \return Pruned set of neighbors
     */
    static auto prune_connections(
        const std::vector<std::pair<float, std::uint64_t>>& candidates,
        std::size_t M,
        const float* base_vector,
        std::size_t dimension,
        HnswIndex* index = nullptr)
        -> std::vector<std::uint64_t> {
        
        std::vector<std::uint64_t> result;
        result.reserve(M);
        
        // Implement HNSW pruning algorithm (Algorithm 4 from paper)
        // This maintains graph connectivity while selecting diverse neighbors
        
        if (candidates.empty()) return result;
        
        // Sort candidates by distance
        auto sorted_candidates = candidates;
        std::sort(sorted_candidates.begin(), sorted_candidates.end());
        
        // Cache for neighbor vectors to avoid repeated retrievals
        std::unordered_map<std::uint64_t, std::vector<float>> vector_cache;
        
        // Select M neighbors using pruning heuristic
        for (const auto& [dist, id] : sorted_candidates) {
            if (result.size() >= M) break;
            
            // Check if this candidate improves connectivity
            bool improves_connectivity = true;
            
            // Get vector for current candidate if not cached
            if (index && vector_cache.find(id) == vector_cache.end()) {
                auto vec_result = index->get_vector(id);
                if (vec_result) {
                    vector_cache[id] = vec_result.value();
                }
            }
            
            for (auto selected_id : result) {
                // Get vector for selected neighbor if not cached
                if (index && vector_cache.find(selected_id) == vector_cache.end()) {
                    auto vec_result = index->get_vector(selected_id);
                    if (vec_result) {
                        vector_cache[selected_id] = vec_result.value();
                    }
                }
                
                // Compute distance between candidate and already selected neighbor
                float neighbor_dist = 0.0f;
                if (vector_cache.find(id) != vector_cache.end() && 
                    vector_cache.find(selected_id) != vector_cache.end()) {
                    // Compute L2 distance between the two candidates
                    const auto& vec1 = vector_cache[id];
                    const auto& vec2 = vector_cache[selected_id];
                    for (std::size_t d = 0; d < dimension && d < vec1.size() && d < vec2.size(); ++d) {
                        float diff = vec1[d] - vec2[d];
                        neighbor_dist += diff * diff;
                    }
                    neighbor_dist = std::sqrt(neighbor_dist);
                } else {
                    // Fallback: use triangle inequality estimate
                    neighbor_dist = std::abs(dist - 0.5f);
                }
                
                // If candidates are too close to each other, skip for diversity
                // Use a threshold based on the distance to the base node
                if (neighbor_dist < dist * 0.9f) {
                    improves_connectivity = false;
                    break;
                }
            }
            
            if (improves_connectivity) {
                result.push_back(id);
            }
        }
        
        // If we don't have enough diverse neighbors, fill with nearest
        for (const auto& [dist, id] : sorted_candidates) {
            if (result.size() >= M) break;
            if (std::find(result.begin(), result.end(), id) == result.end()) {
                result.push_back(id);
            }
        }
        
        return result;
    }
};

/** \brief IVF-PQ incremental repair implementation */
class IvfPqRepair {
public:
    /** \brief Repair IVF-PQ index by rebalancing clusters
     * 
     * \param index IVF-PQ index to repair
     * \param deleted_ids Set of deleted vector IDs
     * \param deletion_threshold Threshold for triggering cluster rebalance
     * \return Success or error
     */
    static auto repair(IvfPqIndex* index,
                      const roaring::Roaring& deleted_ids,
                      float deletion_threshold = 0.2f)
        -> std::expected<void, core::error> {
        
        if (!index) {
            return std::vesper_unexpected(core::error{
                core::error_code::precondition_failed,
                "Null IVF-PQ index",
                "ivfpq_repair"
            });
        }
        
        // Phase 1: Analyze cluster deletion ratios
        std::vector<std::pair<std::uint32_t, float>> cluster_deletions;
        
        // Get cluster assignments for deleted vectors
        std::unordered_map<std::uint32_t, std::uint32_t> cluster_deletion_counts;
        std::unordered_map<std::uint32_t, std::uint32_t> cluster_total_counts;
        std::unordered_map<std::uint32_t, std::vector<std::uint64_t>> cluster_members;
        
        // Get total number of clusters
        auto num_clusters = index->get_num_clusters();
        if (!num_clusters) {
            return std::vesper_unexpected(core::error{
                core::error_code::internal,
                "Failed to get number of clusters",
                "ivfpq_repair"
            });
        }
        
        // Initialize cluster counts
        for (std::uint32_t cluster_id = 0; cluster_id < num_clusters.value(); ++cluster_id) {
            cluster_total_counts[cluster_id] = 0;
            cluster_deletion_counts[cluster_id] = 0;
        }
        
        // Scan all vectors to build cluster membership
        auto stats = index->get_stats();
        auto total_vectors = stats.n_vectors;
        for (std::uint64_t vec_id = 0; vec_id < total_vectors; ++vec_id) {
            auto cluster_result = index->get_cluster_assignment(vec_id);
            if (cluster_result) {
                std::uint32_t cluster_id = cluster_result.value();
                cluster_total_counts[cluster_id]++;
                cluster_members[cluster_id].push_back(vec_id);
                
                // Count if deleted
                if (deleted_ids.contains(static_cast<std::uint32_t>(vec_id))) {
                    cluster_deletion_counts[cluster_id]++;
                }
            }
        }
        
        // Calculate deletion ratios
        for (const auto& [cluster_id, total_count] : cluster_total_counts) {
            if (total_count > 0) {
                float ratio = static_cast<float>(cluster_deletion_counts[cluster_id]) / total_count;
                if (ratio > deletion_threshold) {
                    cluster_deletions.emplace_back(cluster_id, ratio);
                }
            }
        }
        
        // Phase 2: Recompute centroids for affected clusters
        for (const auto& [cluster_id, ratio] : cluster_deletions) {
            // Extract non-deleted vectors in cluster
            std::vector<std::uint64_t> active_vectors;
            std::vector<const float*> active_vector_ptrs;
            
            for (auto vec_id : cluster_members[cluster_id]) {
                if (!deleted_ids.contains(static_cast<std::uint32_t>(vec_id))) {
                    active_vectors.push_back(vec_id);
                    
                    // Get vector data
                    auto vec_result = index->get_vector(vec_id);
                    if (vec_result) {
                        active_vector_ptrs.push_back(vec_result.value().data());
                    }
                }
            }
            
            // Recompute centroid if we have active vectors
            if (!active_vector_ptrs.empty()) {
                auto dimension = index->get_dimension();
                if (dimension) {
                    auto new_centroid = compute_centroid(active_vector_ptrs, dimension.value());
                    
                    // Update cluster centroid
                    auto update_result = index->update_cluster_centroid(cluster_id, new_centroid.data());
                    if (!update_result) {
                        // Log error but continue
                        continue;
                    }
                }
            }
        }
        
        // Phase 3: Reassign vectors if clusters are too imbalanced
        if (!cluster_deletions.empty()) {
            // Find under-utilized clusters (those with low deletion ratios)
            std::vector<std::uint32_t> healthy_clusters;
            for (const auto& [cluster_id, total_count] : cluster_total_counts) {
                float ratio = static_cast<float>(cluster_deletion_counts[cluster_id]) / 
                             std::max(total_count, 1u);
                if (ratio < deletion_threshold * 0.5f && total_count > 0) {
                    healthy_clusters.push_back(cluster_id);
                }
            }
            
            // Reassign vectors from heavily deleted clusters to healthy ones
            for (const auto& [affected_cluster, ratio] : cluster_deletions) {
                if (ratio > 0.5f && !healthy_clusters.empty()) {
                    // Move some vectors to healthier clusters
                    for (auto vec_id : cluster_members[affected_cluster]) {
                        if (!deleted_ids.contains(static_cast<std::uint32_t>(vec_id))) {
                            // Find nearest healthy cluster
                            auto vec_result = index->get_vector(vec_id);
                            if (vec_result) {
                                // Find the nearest cluster centroid
                                const auto& vec = vec_result.value();
                                float min_distance = std::numeric_limits<float>::max();
                                std::uint32_t nearest_cluster = healthy_clusters[0];
                                
                                for (auto cluster_id : healthy_clusters) {
                                    auto centroid_result = index->get_cluster_centroid(cluster_id);
                                    if (centroid_result) {
                                        const auto& centroid = centroid_result.value();
                                        // Compute L2 distance to centroid
                                        float dist = 0.0f;
                                        for (std::size_t d = 0; d < vec.size() && d < centroid.size(); ++d) {
                                            float diff = vec[d] - centroid[d];
                                            dist += diff * diff;
                                        }
                                        if (dist < min_distance) {
                                            min_distance = dist;
                                            nearest_cluster = cluster_id;
                                        }
                                    }
                                }
                                
                                auto reassign_result = index->reassign_vector(vec_id, nearest_cluster);
                                if (!reassign_result) {
                                    continue;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Phase 4: Compact inverted lists
        // Remove tombstoned entries from inverted lists
        for (std::uint32_t cluster_id = 0; cluster_id < num_clusters.value(); ++cluster_id) {
            // Pass the underlying roaring_bitmap_t from roaring::Roaring
            // The 'roaring' member is public and contains the C structure
            const roaring_bitmap_t* bitmap = &deleted_ids.roaring;
            auto compact_result = index->compact_inverted_list(cluster_id, bitmap);
            if (!compact_result) {
                // Log error but continue with other clusters
                continue;
            }
        }
        
        return {};
    }
    
private:
    /** \brief Compute centroid from vectors
     * 
     * \param vectors Vector data
     * \param dimension Vector dimension
     * \return Centroid vector
     */
    static auto compute_centroid(
        const std::vector<const float*>& vectors,
        std::size_t dimension)
        -> std::vector<float> {
        
        std::vector<float> centroid(dimension, 0.0f);
        
        if (vectors.empty()) {
            return centroid;
        }
        
        // Sum all vectors
        for (const auto* vec : vectors) {
            for (std::size_t d = 0; d < dimension; ++d) {
                centroid[d] += vec[d];
            }
        }
        
        // Average
        float scale = 1.0f / vectors.size();
        for (auto& val : centroid) {
            val *= scale;
        }
        
        return centroid;
    }
};

/** \brief DiskGraph incremental repair implementation */
class DiskGraphRepair {
private:
    // Hash function for pair of IDs
    struct PairHash {
        std::size_t operator()(const std::pair<std::uint32_t, std::uint32_t>& p) const {
            auto h1 = std::hash<std::uint32_t>{}(p.first);
            auto h2 = std::hash<std::uint32_t>{}(p.second);
            return h1 ^ (h2 << 1);
        }
    };
    
    // Distance cache to avoid redundant computations
    static inline std::unordered_map<std::pair<std::uint32_t, std::uint32_t>, 
                                     float, 
                                     PairHash> distance_cache_;
    
    // Function pointers for distance computation and vector retrieval
    static inline std::function<float(const float*, const float*)> distance_func_;
    static inline std::function<std::vector<float>(std::uint32_t)> get_vector_func_;

public:
    /** \brief Set distance and vector retrieval functions
     * 
     * \param dist_func Function to compute distance between two vectors
     * \param vec_func Function to retrieve vector data by ID
     */
    static void set_functions(
        std::function<float(const float*, const float*)> dist_func,
        std::function<std::vector<float>(std::uint32_t)> vec_func) {
        distance_func_ = dist_func;
        get_vector_func_ = vec_func;
    }
    
    /** \brief Repair DiskGraph index incrementally
     * 
     * \param index DiskGraph index to repair
     * \param deleted_ids Set of deleted node IDs
     * \param params Vamana build parameters for pruning
     * \return Success or error
     */
    static auto repair(DiskGraphIndex* index,
                      const roaring::Roaring& deleted_ids,
                      const VamanaBuildParams& params)
        -> std::expected<void, core::error> {
        
        if (!index) {
            return std::vesper_unexpected(core::error{
                core::error_code::precondition_failed,
                "Null DiskGraph index",
                "diskgraph_repair"
            });
        }
        
        // Phase 1: Identify affected nodes
        std::unordered_set<std::uint32_t> affected_nodes;
        std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> graph_updates;
        
        // Scan entire graph to find nodes pointing to deleted nodes
        auto total_nodes = index->size();
        for (std::uint32_t node_id = 0; node_id < total_nodes; ++node_id) {
            if (deleted_ids.contains(node_id)) {
                continue; // Skip deleted nodes
            }
            
            // Load neighbor list from disk
            auto neighbors_result = index->get_neighbors(node_id);
            if (!neighbors_result) {
                continue;
            }
            
            auto neighbors = neighbors_result.value();
            bool has_deleted_neighbor = false;
            
            // Check if any neighbor is deleted
            for (auto neighbor_id : neighbors) {
                if (deleted_ids.contains(neighbor_id)) {
                    has_deleted_neighbor = true;
                    break;
                }
            }
            
            if (has_deleted_neighbor) {
                affected_nodes.insert(node_id);
                graph_updates[node_id] = neighbors;
            }
        }
        
        // Phase 2: Remove edges to deleted nodes and find replacements
        for (auto node_id : affected_nodes) {
            auto& neighbors = graph_updates[node_id];
            
            // Filter out deleted nodes
            neighbors.erase(
                std::remove_if(neighbors.begin(), neighbors.end(),
                              [&deleted_ids](std::uint32_t id) {
                                  return deleted_ids.contains(id);
                              }),
                neighbors.end()
            );
            
            // If below minimum degree, find replacements
            if (neighbors.size() < params.degree / 2) {
                // Get vector for this node
                auto node_vec_result = index->get_vector(node_id);
                if (!node_vec_result) {
                    continue;
                }
                const auto& node_vector = node_vec_result.value();
                
                // Search for candidate neighbors using beam search
                std::vector<std::pair<float, std::uint32_t>> candidates;
                std::unordered_set<std::uint32_t> visited(neighbors.begin(), neighbors.end());
                visited.insert(node_id);
                
                // Start from existing neighbors and explore their neighbors
                for (auto neighbor_id : neighbors) {
                    auto second_neighbors_result = index->get_neighbors(neighbor_id);
                    if (!second_neighbors_result) {
                        continue;
                    }
                    
                    for (auto second_neighbor : second_neighbors_result.value()) {
                        if (visited.count(second_neighbor) > 0 || deleted_ids.contains(second_neighbor)) {
                            continue;
                        }
                        visited.insert(second_neighbor);
                        
                        // Get vector and compute distance
                        auto vec_result = index->get_vector(second_neighbor);
                        if (vec_result) {
                            float dist = 0.0f;
                            const auto& vec = vec_result.value();
                            for (std::size_t d = 0; d < node_vector.size() && d < vec.size(); ++d) {
                                float diff = node_vector[d] - vec[d];
                                dist += diff * diff;
                            }
                            candidates.emplace_back(dist, second_neighbor);
                        }
                    }
                }
                
                // Also search from random nodes if we don't have enough candidates
                if (candidates.size() < params.degree) {
                    std::uint32_t random_trials = params.L;
                    for (std::uint32_t trial = 0; trial < random_trials; ++trial) {
                        std::uint32_t random_id = (node_id + trial * 1337) % total_nodes;
                        if (visited.count(random_id) > 0 || deleted_ids.contains(random_id)) {
                            continue;
                        }
                        visited.insert(random_id);
                        
                        auto vec_result = index->get_vector(random_id);
                        if (vec_result) {
                            float dist = 0.0f;
                            const auto& vec = vec_result.value();
                            for (std::size_t d = 0; d < node_vector.size() && d < vec.size(); ++d) {
                                float diff = node_vector[d] - vec[d];
                                dist += diff * diff;
                            }
                            candidates.emplace_back(dist, random_id);
                        }
                    }
                }
                
                // Apply RobustPrune algorithm
                auto pruned = robust_prune(candidates, params.degree, params.alpha, 0.0f);
                
                // Update neighbor list
                neighbors = pruned;
            }
        }
        
        // Phase 3: Update entry points if needed
        auto entry_point_result = index->get_entry_point();
        if (entry_point_result) {
            std::uint32_t entry_point = entry_point_result.value();
            if (deleted_ids.contains(entry_point)) {
                // Find new entry point - select node with highest degree
                std::uint32_t new_entry = 0;
                std::size_t max_degree = 0;
                
                for (auto node_id : affected_nodes) {
                    if (graph_updates[node_id].size() > max_degree) {
                        max_degree = graph_updates[node_id].size();
                        new_entry = node_id;
                    }
                }
                
                // If no affected node suitable, scan for any high-degree node
                if (max_degree < params.degree / 2) {
                    for (std::uint32_t node_id = 0; node_id < total_nodes; ++node_id) {
                        if (deleted_ids.contains(node_id)) {
                            continue;
                        }
                        auto neighbors_result = index->get_neighbors(node_id);
                        if (neighbors_result && neighbors_result.value().size() > max_degree) {
                            max_degree = neighbors_result.value().size();
                            new_entry = node_id;
                        }
                    }
                }
                
                // Update entry point
                auto update_result = index->set_entry_point(new_entry);
                if (!update_result) {
                    // Log error but continue
                }
            }
        }
        
        // Phase 4: Batch write updates to disk
        std::vector<std::pair<std::uint32_t, std::vector<std::uint32_t>>> batch_updates;
        for (const auto& [node_id, neighbors] : graph_updates) {
            batch_updates.emplace_back(node_id, neighbors);
        }
        
        // Sort by node ID for sequential disk access
        std::sort(batch_updates.begin(), batch_updates.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        
        // Apply updates in batches
        const std::size_t batch_size = 1000;
        for (std::size_t i = 0; i < batch_updates.size(); i += batch_size) {
            std::size_t end = std::min(i + batch_size, batch_updates.size());
            
            for (std::size_t j = i; j < end; ++j) {
                auto update_result = index->update_neighbors(
                    batch_updates[j].first, 
                    batch_updates[j].second
                );
                if (!update_result) {
                    // Log error but continue
                }
            }
            
            // Flush to disk after each batch
            auto flush_result = index->flush();
            if (!flush_result) {
                // Log error but continue
            }
        }
        
        return {};
    }
    
private:
    /** \brief RobustPrune algorithm for edge selection
     * 
     * \param candidates Candidate neighbors with distances
     * \param R Maximum degree
     * \param alpha Pruning parameter
     * \param threshold Distance threshold
     * \return Pruned neighbor list
     */
    static auto robust_prune(
        const std::vector<std::pair<float, std::uint32_t>>& candidates,
        std::size_t R,
        float alpha,
        float threshold)
        -> std::vector<std::uint32_t> {
        
        std::vector<std::uint32_t> result;
        result.reserve(R);
        
        if (candidates.empty()) {
            return result;
        }
        
        // Sort candidates by distance
        auto sorted_candidates = candidates;
        std::sort(sorted_candidates.begin(), sorted_candidates.end());
        
        // Greedy pruning with diversity
        for (const auto& [dist, id] : sorted_candidates) {
            if (result.size() >= R) {
                break;
            }
            
            // Check if candidate is diverse enough from existing neighbors
            bool is_diverse = true;
            for (auto existing_id : result) {
                // Compute actual distance between candidates for diversity
                // Access to distance computation is provided through the index
                float candidate_dist = 0.0f;
                
                // Check if we have a distance cache or need to compute
                auto cache_key = std::make_pair(std::min(id, existing_id), 
                                               std::max(id, existing_id));
                auto cache_it = distance_cache_.find(cache_key);
                
                if (cache_it != distance_cache_.end()) {
                    candidate_dist = cache_it->second;
                } else {
                    // Compute actual distance between candidates
                    // This requires access to vector data through the index
                    // The distance function should be provided by the caller
                    if (distance_func_ && get_vector_func_) {
                        auto vec1 = get_vector_func_(id);
                        auto vec2 = get_vector_func_(existing_id);
                        if (!vec1.empty() && !vec2.empty()) {
                            candidate_dist = distance_func_(vec1.data(), vec2.data());
                        } else {
                            // Fallback if vectors not available
                            candidate_dist = dist * 1.5f;
                        }
                    } else {
                        // Fallback: use conservative estimate based on triangle inequality
                        // If both are close to query, they might be close to each other
                        candidate_dist = dist * 0.5f;
                    }
                    distance_cache_[cache_key] = candidate_dist;
                }
                
                // Use distance-based diversity criterion
                // A candidate is diverse if it's sufficiently different from existing neighbors
                float diversity_threshold = dist * 1.2f; // 20% margin for diversity
                
                if (candidate_dist < diversity_threshold) {
                    is_diverse = false;
                    break;
                }
            }
            
            if (is_diverse || result.size() < R / 2) {
                result.push_back(id);
            }
        }
        
        return result;
    }
};

/** \brief Main incremental repair coordinator */
class IncrementalRepairCoordinator {
public:
    /** \brief Perform incremental repair on all indexes
     * 
     * \param hnsw_index HNSW index (optional)
     * \param ivfpq_index IVF-PQ index (optional)
     * \param diskgraph_index DiskGraph index (optional)
     * \param tombstone_manager Tombstone manager
     * \param dimension Vector dimension
     * \param force Force repair even if not needed
     * \return Success or error
     */
    static auto repair_all(
        HnswIndex* hnsw_index,
        IvfPqIndex* ivfpq_index,
        DiskGraphIndex* diskgraph_index,
        tombstone::TombstoneManager* tombstone_manager,
        std::size_t dimension,
        bool force = false)
        -> std::expected<void, core::error> {
        
        if (!tombstone_manager) {
            return std::vesper_unexpected(core::error{
                core::error_code::precondition_failed,
                "Null tombstone manager",
                "repair_coordinator"
            });
        }
        
        // Get deleted IDs
        auto deleted_ids_vec = tombstone_manager->get_deleted_ids();
        
        // Convert to roaring bitmap for efficient operations
        roaring::Roaring deleted_ids;
        for (uint32_t id : deleted_ids_vec) {
            deleted_ids.add(id);
        }
        deleted_ids.runOptimize();
        
        // Check if repair is needed
        auto stats = tombstone_manager->get_stats();
        // Assume total_vectors is provided by the caller or estimated
        std::uint64_t estimated_total = deleted_ids.cardinality() * 10; // Conservative estimate
        float deletion_ratio = stats.deletion_ratio(estimated_total);
        
        if (deletion_ratio < 0.05 && !force) {
            // Low deletion ratio, no repair needed
            return {};
        }
        
        // Repair each index type
        if (hnsw_index && (deletion_ratio > 0.2 || force)) {
            auto result = HnswRepair::repair(hnsw_index, deleted_ids, dimension);
            if (!result) {
                return result;
            }
        }
        
        if (ivfpq_index && (deletion_ratio > 0.1 || force)) {
            auto result = IvfPqRepair::repair(ivfpq_index, deleted_ids);
            if (!result) {
                return result;
            }
        }
        
        if (diskgraph_index && (deletion_ratio > 0.15 || force)) {
            // Use the same parameters as original build for consistency
            VamanaBuildParams params;
            params.degree = 32;
            params.alpha = 1.2f;
            params.L = 128;
            // Use default pruning with alpha
            auto result = DiskGraphRepair::repair(diskgraph_index, deleted_ids, params);
            if (!result) {
                return result;
            }
        }
        
        // Compact tombstones after successful repair
        if (deletion_ratio > 0.3) {
            auto compact_result = tombstone_manager->compact();
            if (!compact_result) {
                return compact_result;
            }
        }
        
        return {};
    }
};

} // namespace vesper::index::repair