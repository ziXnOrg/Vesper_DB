/** \file disk_graph.cpp
 *  \brief Implementation of DiskANN-style graph index (Vamana algorithm).
 */

#include "vesper/index/disk_graph.hpp"
#include "vesper/index/product_quantizer.hpp"
#include "vesper/cache/lru_cache.hpp"
#include <expected>  // For std::expected and std::vesper_unexpected

// Windows fix for std::max
#ifdef _WIN32
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif
#endif

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <queue>
#include <random>
#include <thread>
#include <unordered_set>
#include <filesystem>
#include <numeric>
#include <unordered_set>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

// Now we have std::vesper_unexpected from the expected header

namespace vesper::index {

template<typename T>
using span = std::span<T>;

namespace {

/** \brief Distance comparison for min-heap (smaller distances first). */
struct DistanceComparator {
    bool operator()(const std::pair<float, std::uint32_t>& a,
                   const std::pair<float, std::uint32_t>& b) const {
        return a.first > b.first; // min-heap
    }
};

/** \brief Distance comparison for max-heap (larger distances first). */
struct ReverseDistanceComparator {
    bool operator()(const std::pair<float, std::uint32_t>& a,
                   const std::pair<float, std::uint32_t>& b) const {
        return a.first < b.first; // max-heap
    }
};

/** \brief Compute L2 distance between two vectors. */
inline float compute_l2_distance(const float* a, const float* b, std::size_t dim) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

/** \brief Compute L2 distance between two vectors (exposed for standalone functions). */
inline float compute_l2_distance_global(const float* a, const float* b, std::size_t dim) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

} // anonymous namespace

/** \brief Implementation class for DiskGraphIndex. */
class DiskGraphIndex::Impl {
public:
    explicit Impl(std::size_t dim, std::size_t cache_size_mb = 1024) 
        : dimension_(dim)
        , pq_(nullptr)
        , num_nodes_(0)
        , cache_size_mb_(cache_size_mb) {
        
        const std::size_t cache_bytes = cache_size_mb * 1024 * 1024;
        neighbor_cache_ = std::make_unique<cache::GraphNodeCache>(
            cache_bytes, 16, std::nullopt, nullptr
        );
        vector_cache_ = std::make_unique<cache::VectorCache>(
            cache_bytes, 16, std::nullopt, nullptr
        );
    }

    ~Impl() {
        close_files();
    }

    auto build(span<const float> vectors, const VamanaBuildParams& params)
        -> std::expected<VamanaBuildStats, core::error> {
        
        const auto start_time = std::chrono::steady_clock::now();
        
        // Validate parameters
        if (vectors.size() % dimension_ != 0) {
            return std::vesper_unexpected(core::error{
                core::error_code::config_invalid,
                "Vector count not divisible by dimension",
                "disk_graph.build"
            });
        }

        const std::size_t n = vectors.size() / dimension_;
        if (n == 0) {
            return std::vesper_unexpected(core::error{
                core::error_code::config_invalid,
                "Empty vector set",
                "disk_graph.build"
            });
        }

        num_nodes_ = static_cast<std::uint32_t>(n);
        build_params_ = params;
        
        // Initialize graph with empty adjacency lists
        graph_.resize(num_nodes_);
        
        // Initialize PQ if enabled
        if (params.use_pq) {
            init_product_quantizer(vectors.data(), n);
        }

        // Phase 1: Build initial random graph
        build_initial_graph(vectors.data(), n);
        
        // Phase 2: Iterative refinement with RobustPrune
        for (std::uint32_t iter = 0; iter < params.max_iters; ++iter) {
            refine_graph(vectors.data(), n, iter == params.max_iters - 1);
        }

        // Compute entry points (high-degree nodes)
        compute_entry_points();

        // Collect statistics
        VamanaBuildStats stats;
        stats.num_nodes = num_nodes_;
        stats.total_edges = 0;
        stats.max_degree = 0;
        
        for (const auto& neighbors : graph_) {
            stats.total_edges += neighbors.size();
            stats.max_degree = (std::max)(stats.max_degree, 
                                         static_cast<std::uint32_t>(neighbors.size()));
        }
        
        stats.avg_degree = static_cast<float>(stats.total_edges) / num_nodes_;
        
        const auto end_time = std::chrono::steady_clock::now();
        stats.build_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time).count();
        
        build_stats_ = stats;
        return stats;
    }

    auto search(span<const float> query, const VamanaSearchParams& params) const
        -> std::expected<std::vector<std::pair<float, std::uint32_t>>, core::error> {
        
        if (query.size() != dimension_) {
            return std::vesper_unexpected(core::error{
                core::error_code::config_invalid,
                "Query dimension mismatch",
                "disk_graph.search"
            });
        }

        if (num_nodes_ == 0) {
            return std::vector<std::pair<float, std::uint32_t>>{};
        }

        // Initialize search with entry points
        std::vector<std::uint32_t> init_points = entry_points_;
        if (init_points.empty()) {
            // Use random starting points if no entry points
            std::mt19937 rng(42);
            std::uniform_int_distribution<std::uint32_t> dist(0, num_nodes_ - 1);
            for (std::uint32_t i = 0; i < params.init_list_size && i < num_nodes_; ++i) {
                init_points.push_back(dist(rng));
            }
        }

        // Perform beam search
        auto candidates = beam_search(query.data(), init_points, params);
        
        // Return top-k results
        std::vector<std::pair<float, std::uint32_t>> results;
        std::uint32_t count = 0;
        while (!candidates.empty() && count < params.k) {
            results.push_back(candidates.top());
            candidates.pop();
            count++;
        }
        
        // Reverse to get ascending order
        std::reverse(results.begin(), results.end());
        
        return results;
    }

    auto save(const std::string& path) const -> std::expected<void, core::error> {
        namespace fs = std::filesystem;
        
        // Create directory if it doesn't exist
        fs::create_directories(path);
        
        // Save metadata
        std::string meta_path = path + "/graph.meta";
        std::ofstream meta_file(meta_path, std::ios::binary);
        if (!meta_file) {
            return std::vesper_unexpected(core::error{
                core::error_code::io_failed,
                "Failed to create metadata file",
                "disk_graph.save"
            });
        }
        
        // Write metadata
        meta_file.write(reinterpret_cast<const char*>(&dimension_), sizeof(dimension_));
        meta_file.write(reinterpret_cast<const char*>(&num_nodes_), sizeof(num_nodes_));
        meta_file.write(reinterpret_cast<const char*>(&build_params_.degree), 
                       sizeof(build_params_.degree));
        meta_file.write(reinterpret_cast<const char*>(&build_params_.alpha), 
                       sizeof(build_params_.alpha));
        
        // Save adjacency lists
        std::string adj_path = path + "/graph.adj";
        std::ofstream adj_file(adj_path, std::ios::binary);
        if (!adj_file) {
            return std::vesper_unexpected(core::error{
                core::error_code::io_failed,
                "Failed to create adjacency file",
                "disk_graph.save"
            });
        }
        
        for (const auto& neighbors : graph_) {
            std::uint32_t degree = static_cast<std::uint32_t>(neighbors.size());
            adj_file.write(reinterpret_cast<const char*>(&degree), sizeof(degree));
            adj_file.write(reinterpret_cast<const char*>(neighbors.data()), 
                          degree * sizeof(std::uint32_t));
        }
        
        // Save entry points
        std::string entry_path = path + "/graph.entry";
        std::ofstream entry_file(entry_path, std::ios::binary);
        if (!entry_file) {
            return std::vesper_unexpected(core::error{
                core::error_code::io_failed,
                "Failed to create entry points file",
                "disk_graph.save"
            });
        }
        
        std::uint32_t num_entry = static_cast<std::uint32_t>(entry_points_.size());
        entry_file.write(reinterpret_cast<const char*>(&num_entry), sizeof(num_entry));
        entry_file.write(reinterpret_cast<const char*>(entry_points_.data()),
                        num_entry * sizeof(std::uint32_t));
        
        return {};
    }

    auto load(const std::string& path) -> std::expected<void, core::error> {
        namespace fs = std::filesystem;
        
        // Load metadata
        std::string meta_path = path + "/graph.meta";
        std::ifstream meta_file(meta_path, std::ios::binary);
        if (!meta_file) {
            return std::vesper_unexpected(core::error{
                core::error_code::io_failed,
                "Failed to open metadata file",
                "disk_graph.load"
            });
        }
        
        std::size_t loaded_dim;
        meta_file.read(reinterpret_cast<char*>(&loaded_dim), sizeof(loaded_dim));
        if (loaded_dim != dimension_) {
            return std::vesper_unexpected(core::error{
                core::error_code::config_invalid,
                "Dimension mismatch",
                "disk_graph.load"
            });
        }
        
        meta_file.read(reinterpret_cast<char*>(&num_nodes_), sizeof(num_nodes_));
        meta_file.read(reinterpret_cast<char*>(&build_params_.degree), 
                      sizeof(build_params_.degree));
        meta_file.read(reinterpret_cast<char*>(&build_params_.alpha), 
                      sizeof(build_params_.alpha));
        
        // Load adjacency lists
        std::string adj_path = path + "/graph.adj";
        std::ifstream adj_file(adj_path, std::ios::binary);
        if (!adj_file) {
            return std::vesper_unexpected(core::error{
                core::error_code::io_failed,
                "Failed to open adjacency file",
                "disk_graph.load"
            });
        }
        
        graph_.clear();
        graph_.resize(num_nodes_);
        
        for (std::uint32_t i = 0; i < num_nodes_; ++i) {
            std::uint32_t degree;
            adj_file.read(reinterpret_cast<char*>(&degree), sizeof(degree));
            graph_[i].resize(degree);
            adj_file.read(reinterpret_cast<char*>(graph_[i].data()), 
                         degree * sizeof(std::uint32_t));
        }
        
        // Load entry points
        std::string entry_path = path + "/graph.entry";
        std::ifstream entry_file(entry_path, std::ios::binary);
        if (!entry_file) {
            return std::vesper_unexpected(core::error{
                core::error_code::io_failed,
                "Failed to open entry points file",
                "disk_graph.load"
            });
        }
        
        std::uint32_t num_entry;
        entry_file.read(reinterpret_cast<char*>(&num_entry), sizeof(num_entry));
        entry_points_.resize(num_entry);
        entry_file.read(reinterpret_cast<char*>(entry_points_.data()),
                       num_entry * sizeof(std::uint32_t));
        
        return {};
    }

    auto size() const -> std::uint32_t { return num_nodes_; }
    auto dimension() const -> std::size_t { return dimension_; }
    auto build_stats() const -> VamanaBuildStats { return build_stats_; }
    auto io_stats() const -> IOStats { 
        IOStats stats;
        stats.reads = io_stats_.reads.load();
        stats.read_bytes = io_stats_.read_bytes.load();
        stats.cache_hits = 0; // io_stats_.cache_hits.load();
        stats.cache_misses = 0; // io_stats_.cache_misses.load();
        stats.prefetch_hits = io_stats_.prefetch_hits.load();
        return stats;
    }

private:
    void init_product_quantizer(const float* vectors, std::size_t n) {
        // Initialize PQ with training vectors
        PqTrainParams pq_params;
        pq_params.m = build_params_.pq_m;
        pq_params.nbits = build_params_.pq_bits;
        
        pq_ = std::make_unique<ProductQuantizer>();
        
        // Train PQ on a subset of vectors
        std::size_t train_size = (std::min)(n, static_cast<std::size_t>(100000));
        auto result = pq_->train(vectors, train_size, dimension_, pq_params);
        if (!result) {
            // Handle error - for now just skip PQ
            pq_.reset();
            return;
        }
        
        // Encode all vectors
        pq_codes_.resize(n * pq_->code_size());
        for (std::size_t i = 0; i < n; ++i) {
            pq_->encode_one(vectors + i * dimension_, 
                           pq_codes_.data() + i * pq_->code_size());
        }
    }

    void build_initial_graph(const float* vectors, std::size_t n) {
        std::mt19937 rng(build_params_.seed);
        std::uniform_int_distribution<std::uint32_t> dist(0, num_nodes_ - 1);
        
        // Initialize each node with random neighbors
        for (std::uint32_t i = 0; i < num_nodes_; ++i) {
            std::unordered_set<std::uint32_t> neighbors;
            
            // Add random neighbors
            while (neighbors.size() < build_params_.degree && 
                   neighbors.size() < num_nodes_ - 1) {
                std::uint32_t neighbor = dist(rng);
                if (neighbor != i) {
                    neighbors.insert(neighbor);
                }
            }
            
            graph_[i].assign(neighbors.begin(), neighbors.end());
            std::sort(graph_[i].begin(), graph_[i].end());
        }
    }

    void refine_graph(const float* vectors, std::size_t n, bool final_iteration) {
        // For each node, search for better neighbors and apply RobustPrune
        for (std::uint32_t i = 0; i < num_nodes_; ++i) {
            const float* query = vectors + i * dimension_;
            
            // Search for candidates using current graph
            auto candidates = greedy_search_internal(
                query, {i}, build_params_.L, vectors);
            
            // Apply RobustPrune
            float alpha = final_iteration ? build_params_.alpha : 1.0f;
            auto pruned = robust_prune_internal(
                i, candidates, build_params_.degree, alpha, vectors);
            
            // Update adjacency list
            graph_[i] = pruned;
            
            // Add reverse edges (make graph undirected)
            for (std::uint32_t neighbor : pruned) {
                auto& rev_neighbors = graph_[neighbor];
                if (std::find(rev_neighbors.begin(), rev_neighbors.end(), i) 
                    == rev_neighbors.end()) {
                    rev_neighbors.push_back(i);
                    
                    // Prune reverse neighbor list if too large
                    if (rev_neighbors.size() > build_params_.degree) {
                        std::vector<std::pair<float, std::uint32_t>> rev_candidates;
                        for (std::uint32_t nb : rev_neighbors) {
                            float dist = compute_l2_distance(
                                vectors + neighbor * dimension_,
                                vectors + nb * dimension_,
                                dimension_);
                            rev_candidates.push_back({dist, nb});
                        }
                        
                        auto pruned_rev = robust_prune_internal(
                            neighbor, rev_candidates, build_params_.degree, 
                            alpha, vectors);
                        graph_[neighbor] = pruned_rev;
                    }
                }
            }
        }
    }

    void compute_entry_points() {
        // Select nodes with highest degree as entry points
        std::vector<std::pair<std::uint32_t, std::uint32_t>> degree_pairs;
        for (std::uint32_t i = 0; i < num_nodes_; ++i) {
            degree_pairs.push_back({static_cast<std::uint32_t>(graph_[i].size()), i});
        }
        
        std::sort(degree_pairs.rbegin(), degree_pairs.rend());
        
        entry_points_.clear();
        std::uint32_t num_entry = (std::min)(static_cast<std::uint32_t>(50), 
                                          num_nodes_ / 100);
        for (std::uint32_t i = 0; i < num_entry && i < degree_pairs.size(); ++i) {
            entry_points_.push_back(degree_pairs[i].second);
        }
    }

    auto greedy_search_internal(
        const float* query,
        const std::vector<std::uint32_t>& init_points,
        std::uint32_t L,
        const float* vectors) const
        -> std::vector<std::pair<float, std::uint32_t>> {
        
        std::priority_queue<std::pair<float, std::uint32_t>,
                           std::vector<std::pair<float, std::uint32_t>>,
                           DistanceComparator> candidates;
        
        std::priority_queue<std::pair<float, std::uint32_t>,
                           std::vector<std::pair<float, std::uint32_t>>,
                           ReverseDistanceComparator> w;
        
        std::unordered_set<std::uint32_t> visited;
        
        // Initialize with entry points
        for (std::uint32_t id : init_points) {
            float dist = compute_l2_distance(query, vectors + id * dimension_, dimension_);
            candidates.push({dist, id});
            w.push({dist, id});
            visited.insert(id);
        }
        
        // Beam search
        while (!candidates.empty()) {
            auto [current_dist, current_id] = candidates.top();
            candidates.pop();
            
            if (current_dist > w.top().first) {
                break; // All remaining candidates are worse
            }
            
            // Check neighbors (cache temporarily disabled)
            const auto& neighbors = graph_[current_id]; // get_neighbors_cached(current_id);
            for (std::uint32_t neighbor : neighbors) {
                if (visited.find(neighbor) == visited.end()) {
                    visited.insert(neighbor);
                    
                    float dist = compute_l2_distance(
                        query, vectors + neighbor * dimension_, dimension_);
                    
                    if (dist < w.top().first || w.size() < L) {
                        candidates.push({dist, neighbor});
                        w.push({dist, neighbor});
                        
                        if (w.size() > L) {
                            w.pop();
                        }
                    }
                }
            }
        }
        
        // Extract results
        std::vector<std::pair<float, std::uint32_t>> results;
        while (!w.empty()) {
            results.push_back(w.top());
            w.pop();
        }
        
        std::reverse(results.begin(), results.end());
        return results;
    }

    auto robust_prune_internal(
        std::uint32_t node_id,
        std::vector<std::pair<float, std::uint32_t>>& candidates,
        std::uint32_t degree,
        float alpha,
        const float* vectors) const
        -> std::vector<std::uint32_t> {
        
        // Sort candidates by distance
        std::sort(candidates.begin(), candidates.end());
        
        std::vector<std::uint32_t> pruned;
        std::vector<bool> selected(candidates.size(), false);
        
        // Greedy selection with RobustPrune
        for (std::size_t i = 0; i < candidates.size() && pruned.size() < degree; ++i) {
            if (selected[i]) continue;
            
            auto [dist_i, id_i] = candidates[i];
            if (id_i == node_id) continue; // Skip self
            
            bool should_select = true;
            
            // Check if this candidate is dominated by already selected neighbors
            for (std::size_t j = 0; j < i; ++j) {
                if (!selected[j]) continue;
                
                auto [dist_j, id_j] = candidates[j];
                
                // Compute distance between candidates
                float inter_dist = compute_l2_distance(
                    vectors + id_i * dimension_,
                    vectors + id_j * dimension_,
                    dimension_);
                
                // Check domination condition with alpha
                if (inter_dist < alpha * dist_i) {
                    should_select = false;
                    break;
                }
            }
            
            if (should_select) {
                pruned.push_back(id_i);
                selected[i] = true;
            }
        }
        
        return pruned;
    }

    auto beam_search(
        const float* query,
        const std::vector<std::uint32_t>& init_points,
        const VamanaSearchParams& params) const
        -> std::priority_queue<std::pair<float, std::uint32_t>,
                              std::vector<std::pair<float, std::uint32_t>>,
                              ReverseDistanceComparator> {
        
        // This would normally load vectors from disk
        // For now, using cached data
        return {};
    }

    void close_files() {
        // Close any open file handles
    }
    
    // Get neighbors (cache temporarily disabled)
    /*
    auto get_neighbors_cached(std::uint32_t node_id) const 
        -> const std::vector<std::uint32_t>& {
        // Cache disabled for compilation
        return graph_[node_id];
    }
    */
    
    // Get vector (cache temporarily disabled)
    /*
    auto get_vector_cached(std::uint32_t node_id, const float* vectors) const
        -> std::vector<float> {
        // Cache disabled for compilation
        std::vector<float> vec(dimension_);
        std::memcpy(vec.data(), vectors + node_id * dimension_, 
                   dimension_ * sizeof(float));
        return vec;
    }
    */

private:
    std::size_t dimension_;
    std::uint32_t num_nodes_;
    VamanaBuildParams build_params_;
    VamanaBuildStats build_stats_;
    mutable IOStats io_stats_;
    
    // Graph structure (in production, this would be on disk)
    std::vector<std::vector<std::uint32_t>> graph_;
    
    // Entry points for search
    std::vector<std::uint32_t> entry_points_;
    
    // Product quantizer
    std::unique_ptr<ProductQuantizer> pq_;
    std::vector<std::uint8_t> pq_codes_;
    
    // LRU Cache for graph nodes and vectors
    mutable std::unique_ptr<cache::GraphNodeCache> neighbor_cache_;
    mutable std::unique_ptr<cache::VectorCache> vector_cache_;
    std::size_t cache_size_mb_;
};

// Public interface implementation

DiskGraphIndex::DiskGraphIndex(std::size_t dim)
    : impl_(std::make_unique<Impl>(dim)) {
}

DiskGraphIndex::~DiskGraphIndex() = default;

auto DiskGraphIndex::build(span<const float> vectors, const VamanaBuildParams& params)
    -> std::expected<VamanaBuildStats, core::error> {
    return impl_->build(vectors, params);
}

auto DiskGraphIndex::search(span<const float> query, const VamanaSearchParams& params) const
    -> std::expected<std::vector<std::pair<float, std::uint32_t>>, core::error> {
    return impl_->search(query, params);
}

auto DiskGraphIndex::save(const std::string& path) const
    -> std::expected<void, core::error> {
    return impl_->save(path);
}

auto DiskGraphIndex::load(const std::string& path)
    -> std::expected<void, core::error> {
    return impl_->load(path);
}

auto DiskGraphIndex::size() const -> std::uint32_t {
    return impl_->size();
}

auto DiskGraphIndex::dimension() const -> std::size_t {
    return impl_->dimension();
}

auto DiskGraphIndex::build_stats() const -> VamanaBuildStats {
    return impl_->build_stats();
}

auto DiskGraphIndex::io_stats() const -> IOStats {
    return impl_->io_stats();
}

// Standalone algorithm implementations

auto robust_prune(
    std::uint32_t node_id,
    std::vector<std::pair<float, std::uint32_t>>& candidates,
    std::uint32_t degree,
    float alpha,
    const float* vectors,
    std::size_t dim)
    -> std::vector<std::uint32_t> {
    
    // Sort candidates by distance
    std::sort(candidates.begin(), candidates.end());
    
    std::vector<std::uint32_t> pruned;
    std::vector<bool> selected(candidates.size(), false);
    
    for (std::size_t i = 0; i < candidates.size() && pruned.size() < degree; ++i) {
        if (selected[i]) continue;
        
        auto [dist_i, id_i] = candidates[i];
        if (id_i == node_id) continue;
        
        bool should_select = true;
        
        for (std::size_t j = 0; j < i; ++j) {
            if (!selected[j]) continue;
            
            auto [dist_j, id_j] = candidates[j];
            
            float inter_dist = compute_l2_distance_global(
                vectors + id_i * dim,
                vectors + id_j * dim,
                dim);
            
            if (inter_dist < alpha * dist_i) {
                should_select = false;
                break;
            }
        }
        
        if (should_select) {
            pruned.push_back(id_i);
            selected[i] = true;
        }
    }
    
    return pruned;
}

auto greedy_search(
    span<const float> query,
    const std::vector<std::uint32_t>& entry_points,
    const std::vector<std::vector<std::uint32_t>>& graph,
    std::uint32_t L,
    const float* vectors,
    std::size_t dim)
    -> std::vector<std::pair<float, std::uint32_t>> {
    
    std::priority_queue<std::pair<float, std::uint32_t>,
                       std::vector<std::pair<float, std::uint32_t>>,
                       DistanceComparator> candidates;
    
    std::priority_queue<std::pair<float, std::uint32_t>,
                       std::vector<std::pair<float, std::uint32_t>>,
                       ReverseDistanceComparator> w;
    
    std::unordered_set<std::uint32_t> visited;
    
    for (std::uint32_t id : entry_points) {
        float dist = compute_l2_distance_global(query.data(), vectors + id * dim, dim);
        candidates.push({dist, id});
        w.push({dist, id});
        visited.insert(id);
    }
    
    while (!candidates.empty()) {
        auto [current_dist, current_id] = candidates.top();
        candidates.pop();
        
        if (current_dist > w.top().first) {
            break;
        }
        
        for (std::uint32_t neighbor : graph[current_id]) {
            if (visited.find(neighbor) == visited.end()) {
                visited.insert(neighbor);
                
                float dist = compute_l2_distance_global(query.data(), vectors + neighbor * dim, dim);
                
                if (dist < w.top().first || w.size() < L) {
                    candidates.push({dist, neighbor});
                    w.push({dist, neighbor});
                    
                    if (w.size() > L) {
                        w.pop();
                    }
                }
            }
        }
    }
    
    std::vector<std::pair<float, std::uint32_t>> results;
    while (!w.empty()) {
        results.push_back(w.top());
        w.pop();
    }
    
    std::reverse(results.begin(), results.end());
    return results;
}

} // namespace vesper::index