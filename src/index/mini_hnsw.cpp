/** \file mini_hnsw.cpp
 *  \brief Small HNSW graphs per cluster for local refinement.
 *
 * Final phase of CGF pipeline - builds small navigable graphs
 * within each cluster to efficiently find exact nearest neighbors
 * among the filtered candidates.
 */

#include <algorithm>
#include <random>
#include <queue>
#include <unordered_set>
#include <mutex>
#include <shared_mutex>

#include "vesper/index/cgf.hpp"
#include "vesper/index/hnsw.hpp"
#include "vesper/kernels/distance.hpp"

namespace vesper::index {

class MiniHNSW {
public:
    /** Mini graph node for local navigation. */
    struct Node {
        std::uint64_t id;
        std::uint32_t local_idx;  // Index within cluster
        std::vector<std::uint32_t> neighbors;  // Local indices
        std::uint8_t level;
    };
    
    /** Configuration for mini graphs. */
    struct Config {
        std::uint32_t M = 8;          // Max neighbors per node
        std::uint32_t efConstruction = 50;  // Build-time search width
        std::uint32_t efSearch = 100;       // Query-time search width
        std::uint32_t min_cluster_size = 50;  // Min size to build graph
        float ml = 1.0f / std::log(2.0f);    // Level assignment factor
    };
    
    /** Individual mini graph for a cluster. */
    class ClusterGraph {
    public:
        ClusterGraph(std::uint32_t cluster_id, const Config& config)
            : cluster_id_(cluster_id), config_(config), gen_(cluster_id) {}
        
        /** Build graph from cluster members. */
        auto build(const std::vector<std::uint64_t>& ids,
                  const std::vector<std::vector<float>>& vectors) -> void {
            
            if (ids.size() < config_.min_cluster_size) {
                // Too small for graph, use flat search
                use_flat_search_ = true;
                flat_ids_ = ids;
                flat_vectors_ = vectors;
                return;
            }
            
            nodes_.resize(ids.size());
            id_to_local_.clear();
            
            // Initialize nodes
            for (std::size_t i = 0; i < ids.size(); ++i) {
                nodes_[i].id = ids[i];
                nodes_[i].local_idx = static_cast<std::uint32_t>(i);
                nodes_[i].level = assign_level();
                nodes_[i].neighbors.reserve(config_.M);
                
                id_to_local_[ids[i]] = static_cast<std::uint32_t>(i);
            }
            
            // Find entry point (highest level node)
            entry_point_ = 0;
            for (std::size_t i = 1; i < nodes_.size(); ++i) {
                if (nodes_[i].level > nodes_[entry_point_].level) {
                    entry_point_ = static_cast<std::uint32_t>(i);
                }
            }
            
            // Build graph incrementally
            for (std::size_t i = 0; i < nodes_.size(); ++i) {
                insert_node(static_cast<std::uint32_t>(i), vectors[i]);
            }
            
            // Prune edges for better quality
            prune_graph(vectors);
        }
        
        /** Search within mini graph. */
        auto search(const float* query,
                   std::uint32_t k,
                   const std::function<float(std::uint32_t)>& distance_func) const
            -> std::vector<std::pair<std::uint64_t, float>> {
            
            if (use_flat_search_) {
                return flat_search(query, k, distance_func);
            }
            
            // HNSW search starting from entry point
            std::priority_queue<std::pair<float, std::uint32_t>> candidates;
            std::priority_queue<std::pair<float, std::uint32_t>,
                              std::vector<std::pair<float, std::uint32_t>>,
                              std::greater<>> w;
            
            std::unordered_set<std::uint32_t> visited;
            
            // Start from entry point
            float dist = distance_func(entry_point_);
            candidates.emplace(-dist, entry_point_);
            w.emplace(dist, entry_point_);
            visited.insert(entry_point_);
            
            // Search with ef parameter
            while (!candidates.empty()) {
                auto [neg_dist, current] = candidates.top();
                candidates.pop();
                
                if (-neg_dist > w.top().first) {
                    break;  // All remaining candidates are worse
                }
                
                // Check neighbors
                for (std::uint32_t neighbor : nodes_[current].neighbors) {
                    if (visited.count(neighbor)) continue;
                    visited.insert(neighbor);
                    
                    float d = distance_func(neighbor);
                    
                    if (d < w.top().first || w.size() < config_.efSearch) {
                        candidates.emplace(-d, neighbor);
                        w.emplace(d, neighbor);
                        
                        if (w.size() > config_.efSearch) {
                            w.pop();
                        }
                    }
                }
            }
            
            // Extract top-k results
            std::vector<std::pair<std::uint64_t, float>> results;
            while (!w.empty() && results.size() < k) {
                auto [dist, idx] = w.top();
                w.pop();
                results.emplace_back(nodes_[idx].id, dist);
            }
            
            std::reverse(results.begin(), results.end());
            return results;
        }
        
        /** Add new vector to existing graph. */
        auto add(std::uint64_t id, const std::vector<float>& vec) -> void {
            if (use_flat_search_) {
                flat_ids_.push_back(id);
                flat_vectors_.push_back(vec);
                return;
            }
            
            std::uint32_t idx = static_cast<std::uint32_t>(nodes_.size());
            Node node;
            node.id = id;
            node.local_idx = idx;
            node.level = assign_level();
            
            nodes_.push_back(node);
            id_to_local_[id] = idx;
            
            insert_node(idx, vec);
        }
        
        /** Get statistics. */
        struct Stats {
            std::size_t n_nodes;
            std::size_t n_edges;
            float avg_degree;
            std::uint8_t max_level;
            bool uses_flat_search;
        };
        
        auto get_stats() const -> Stats {
            Stats stats{};
            stats.n_nodes = nodes_.size();
            stats.uses_flat_search = use_flat_search_;
            
            if (!use_flat_search_) {
                for (const auto& node : nodes_) {
                    stats.n_edges += node.neighbors.size();
                    stats.max_level = std::max(stats.max_level, node.level);
                }
                if (!nodes_.empty()) {
                    stats.avg_degree = static_cast<float>(stats.n_edges) / nodes_.size();
                }
            }
            
            return stats;
        }
        
    private:
        std::uint32_t cluster_id_;
        Config config_;
        std::minstd_rand gen_;
        
        // Graph structure
        std::vector<Node> nodes_;
        std::unordered_map<std::uint64_t, std::uint32_t> id_to_local_;
        std::uint32_t entry_point_ = 0;
        
        // Flat search fallback for small clusters
        bool use_flat_search_ = false;
        std::vector<std::uint64_t> flat_ids_;
        std::vector<std::vector<float>> flat_vectors_;
        
        /** Assign level to new node. */
        std::uint8_t assign_level() {
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            float level_f = -std::log(dist(gen_)) * config_.ml;
            return static_cast<std::uint8_t>(std::min(level_f, 16.0f));
        }
        
        /** Insert node into graph. */
        void insert_node(std::uint32_t idx, const std::vector<float>& vec) {
            // Find neighbors using beam search
            auto neighbors = search_neighbors(vec.data(), config_.efConstruction);
            
            // Connect to M nearest neighbors
            std::uint32_t M = config_.M;
            if (nodes_[idx].level == 0) {
                M = config_.M * 2;  // Layer 0 has more connections
            }
            
            for (std::size_t i = 0; i < std::min<std::size_t>(M, neighbors.size()); ++i) {
                std::uint32_t neighbor = neighbors[i].second;
                if (neighbor != idx) {
                    nodes_[idx].neighbors.push_back(neighbor);
                    nodes_[neighbor].neighbors.push_back(idx);
                }
            }
        }
        
        /** Search for nearest neighbors during construction. */
        std::vector<std::pair<float, std::uint32_t>> 
        search_neighbors(const float* query, std::uint32_t ef) const {
            
            std::vector<std::pair<float, std::uint32_t>> results;
            
            if (nodes_.empty()) return results;
            
            // Simple linear search for construction
            // (In production, use proper HNSW search)
            const auto& ops = kernels::select_backend_auto();
            
            for (std::uint32_t i = 0; i < nodes_.size(); ++i) {
                // Note: This assumes we have access to vectors
                // In real implementation, pass distance function
                float dist = 0.0f;  // Placeholder
                results.emplace_back(dist, i);
            }
            
            std::sort(results.begin(), results.end());
            if (results.size() > ef) {
                results.resize(ef);
            }
            
            return results;
        }
        
        /** Prune graph edges for quality. */
        void prune_graph(const std::vector<std::vector<float>>& vectors) {
            // Implement edge pruning using relative neighborhood graph
            // or other heuristics to improve search quality
            
            for (auto& node : nodes_) {
                if (node.neighbors.size() > config_.M * 2) {
                    // Sort by distance and keep closest
                    // ... (implementation details)
                    node.neighbors.resize(config_.M * 2);
                }
            }
        }
        
        /** Flat search for small clusters. */
        std::vector<std::pair<std::uint64_t, float>>
        flat_search(const float* query, std::uint32_t k,
                   const std::function<float(std::uint32_t)>& distance_func) const {
            
            std::vector<std::pair<float, std::uint64_t>> distances;
            const auto& ops = kernels::select_backend_auto();
            
            for (std::size_t i = 0; i < flat_ids_.size(); ++i) {
                float dist = ops.l2_sq(
                    std::span(query, flat_vectors_[i].size()),
                    std::span(flat_vectors_[i].data(), flat_vectors_[i].size())
                );
                distances.emplace_back(dist, flat_ids_[i]);
            }
            
            std::partial_sort(distances.begin(),
                            distances.begin() + std::min<std::size_t>(k, distances.size()),
                            distances.end());
            
            std::vector<std::pair<std::uint64_t, float>> results;
            for (std::size_t i = 0; i < std::min<std::size_t>(k, distances.size()); ++i) {
                results.emplace_back(distances[i].second, distances[i].first);
            }
            
            return results;
        }
    };
    
    MiniHNSW(std::size_t dim, const Config& config = {})
        : dim_(dim), config_(config) {}
    
    /** Build mini graphs for all clusters. */
    auto build_graphs(const std::unordered_map<std::uint32_t, 
                                               std::vector<std::uint64_t>>& cluster_members,
                     const std::function<std::vector<float>(std::uint64_t)>& get_vector)
        -> void {
        
        graphs_.clear();
        
        for (const auto& [cluster_id, members] : cluster_members) {
            auto graph = std::make_unique<ClusterGraph>(cluster_id, config_);
            
            // Collect vectors for this cluster
            std::vector<std::vector<float>> vectors;
            vectors.reserve(members.size());
            
            for (std::uint64_t id : members) {
                vectors.push_back(get_vector(id));
            }
            
            // Build mini graph
            graph->build(members, vectors);
            
            std::lock_guard<std::shared_mutex> lock(graphs_mutex_);
            graphs_[cluster_id] = std::move(graph);
        }
    }
    
    /** Search within specific clusters. */
    auto search_clusters(const float* query,
                        const std::vector<std::uint32_t>& cluster_ids,
                        std::uint32_t k,
                        const std::function<float(std::uint64_t)>& distance_func) const
        -> std::vector<std::pair<std::uint64_t, float>> {
        
        std::vector<std::pair<std::uint64_t, float>> all_results;
        
        for (std::uint32_t cluster_id : cluster_ids) {
            std::shared_lock<std::shared_mutex> lock(graphs_mutex_);
            
            auto it = graphs_.find(cluster_id);
            if (it == graphs_.end()) continue;
            
            // Create local distance function for this cluster
            auto local_dist = [&](std::uint32_t local_idx) -> float {
                std::uint64_t global_id = it->second->nodes_[local_idx].id;
                return distance_func(global_id);
            };
            
            auto cluster_results = it->second->search(query, k, local_dist);
            
            all_results.insert(all_results.end(),
                             cluster_results.begin(),
                             cluster_results.end());
        }
        
        // Merge and sort results from all clusters
        std::sort(all_results.begin(), all_results.end(),
                 [](const auto& a, const auto& b) {
                     return a.second < b.second;
                 });
        
        if (all_results.size() > k) {
            all_results.resize(k);
        }
        
        return all_results;
    }
    
    /** Add vector to appropriate cluster graph. */
    auto add_to_cluster(std::uint64_t id,
                       const std::vector<float>& vec,
                       std::uint32_t cluster_id) -> void {
        
        std::lock_guard<std::shared_mutex> lock(graphs_mutex_);
        
        auto it = graphs_.find(cluster_id);
        if (it == graphs_.end()) {
            // Create new graph for this cluster
            auto graph = std::make_unique<ClusterGraph>(cluster_id, config_);
            graph->add(id, vec);
            graphs_[cluster_id] = std::move(graph);
        } else {
            it->second->add(id, vec);
        }
    }
    
    /** Get graph statistics. */
    struct GlobalStats {
        std::size_t n_graphs;
        std::size_t total_nodes;
        std::size_t total_edges;
        float avg_graph_size;
        float avg_degree;
        std::size_t n_flat_search;
    };
    
    auto get_global_stats() const -> GlobalStats {
        GlobalStats stats{};
        
        std::shared_lock<std::shared_mutex> lock(graphs_mutex_);
        
        stats.n_graphs = graphs_.size();
        
        for (const auto& [cluster_id, graph] : graphs_) {
            auto local_stats = graph->get_stats();
            stats.total_nodes += local_stats.n_nodes;
            stats.total_edges += local_stats.n_edges;
            if (local_stats.uses_flat_search) {
                stats.n_flat_search++;
            }
        }
        
        if (stats.n_graphs > 0) {
            stats.avg_graph_size = static_cast<float>(stats.total_nodes) / stats.n_graphs;
        }
        
        if (stats.total_nodes > 0) {
            stats.avg_degree = static_cast<float>(stats.total_edges) / stats.total_nodes;
        }
        
        return stats;
    }
    
    /** Optimize graphs by rebalancing. */
    auto optimize_graphs(const std::function<std::vector<float>(std::uint64_t)>& get_vector)
        -> void {
        
        std::lock_guard<std::shared_mutex> lock(graphs_mutex_);
        
        for (auto& [cluster_id, graph] : graphs_) {
            auto stats = graph->get_stats();
            
            // Rebuild graph if it's become too unbalanced
            if (stats.avg_degree > config_.M * 3 || 
                stats.avg_degree < config_.M / 2) {
                
                // Collect current members
                std::vector<std::uint64_t> ids;
                std::vector<std::vector<float>> vectors;
                
                for (const auto& node : graph->nodes_) {
                    ids.push_back(node.id);
                    vectors.push_back(get_vector(node.id));
                }
                
                // Rebuild
                graph->build(ids, vectors);
            }
        }
    }
    
private:
    std::size_t dim_;
    Config config_;
    
    // Cluster graphs
    mutable std::shared_mutex graphs_mutex_;
    std::unordered_map<std::uint32_t, std::unique_ptr<ClusterGraph>> graphs_;
};

} // namespace vesper::index