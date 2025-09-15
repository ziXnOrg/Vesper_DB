#include "vesper/index/kmeans.hpp"
#include "vesper/kernels/dispatch.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <numeric>
#include <random>
#include <execution>
#include <cmath>
#include <queue>
#include <memory>
#include <cstring>

namespace vesper::index {

namespace {

/** \brief Compute squared L2 distance between vectors. */
[[gnu::hot]] inline auto compute_distance(const float* a, const float* b, std::size_t dim) -> float {
    const auto& ops = kernels::select_backend_auto();
    return ops.l2_sq(std::span(a, dim), std::span(b, dim));
}

/** \brief Find nearest centroid for a point. */
auto find_nearest_centroid(const float* point,
                           const std::vector<std::vector<float>>& centroids,
                           std::size_t dim) -> std::pair<std::uint32_t, float> {
    std::uint32_t best_idx = 0;
    float best_dist = std::numeric_limits<float>::max();

    for (std::uint32_t i = 0; i < centroids.size(); ++i) {
        const float dist = compute_distance(point, centroids[i].data(), dim);
        if (dist < best_dist) {
            best_dist = dist;
            best_idx = i;
        }
    }

    return {best_idx, best_dist};
}

} // anonymous namespace

auto kmeans_plusplus_init(const float* data, std::size_t n, std::size_t dim,
                          std::uint32_t k, std::uint32_t seed)
    -> std::vector<std::vector<float>> {
    std::vector<std::vector<float>> centroids;
    centroids.reserve(k);

    std::mt19937 gen(seed);

    // Choose first centroid randomly
    std::uniform_int_distribution<std::size_t> first_dist(0, n - 1);
    const std::size_t first_idx = first_dist(gen);
    centroids.emplace_back(data + first_idx * dim, data + (first_idx + 1) * dim);

    // For remaining centroids, use D² weighting
    std::vector<float> min_distances(n, std::numeric_limits<float>::max());

    for (std::uint32_t c = 1; c < k; ++c) {
        // Update minimum distances to nearest centroid
        const auto& last_centroid = centroids.back();

        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(n); ++i) {
            const float dist = compute_distance(data + i * dim, last_centroid.data(), dim);
            min_distances[i] = std::min(min_distances[i], dist);
        }

        // Compute cumulative distribution
        std::vector<double> cumsum(n);
        cumsum[0] = min_distances[0];
        for (std::size_t i = 1; i < n; ++i) {
            cumsum[i] = cumsum[i - 1] + min_distances[i];
        }

        // Sample next centroid proportional to squared distance
        std::uniform_real_distribution<double> sample_dist(0.0, cumsum.back());
        const double target = sample_dist(gen);

        const auto it = std::lower_bound(cumsum.begin(), cumsum.end(), target);
        const std::size_t idx = std::distance(cumsum.begin(), it);

        centroids.emplace_back(data + idx * dim, data + (idx + 1) * dim);
    }

    return centroids;
}

auto kmeans_assign(const float* data, std::size_t n,
                  const std::vector<std::vector<float>>& centroids,
                  std::span<std::uint32_t> assignments) -> float {
    const std::size_t dim = centroids[0].size();
    double total_inertia = 0.0;

    #pragma omp parallel for reduction(+:total_inertia)
    for (int i = 0; i < static_cast<int>(n); ++i) {
        const auto [idx, dist] = find_nearest_centroid(data + i * dim, centroids, dim);
        assignments[i] = idx;
        total_inertia += dist;
    }

    return static_cast<float>(total_inertia);
}

auto kmeans_update_centroids(const float* data, std::size_t n, std::size_t dim,
                             std::span<const std::uint32_t> assignments,
                             std::uint32_t k,
                             std::vector<std::vector<float>>& centroids) -> void {
    // Initialize accumulators
    std::vector<std::vector<double>> sums(k, std::vector<double>(dim, 0.0));
    std::vector<std::uint32_t> counts(k, 0);

    // Accumulate points
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint32_t cluster = assignments[i];
        counts[cluster]++;

        const float* point = data + i * dim;
        for (std::size_t d = 0; d < dim; ++d) {
            sums[cluster][d] += point[d];
        }
    }

    // Update centroids
    for (std::uint32_t c = 0; c < k; ++c) {
        if (counts[c] > 0) {
            for (std::size_t d = 0; d < dim; ++d) {
                centroids[c][d] = static_cast<float>(sums[c][d] / counts[c]);
            }
        }
        // If cluster is empty, keep previous centroid
    }
}

auto kmeans_cluster(const float* data, std::size_t n, std::size_t dim,
                   const KmeansParams& params)
    -> std::expected<KmeansResult, core::error> {
    using core::error;
    using core::error_code;

    if (n < params.k) {
        return std::vesper_unexpected(error{error_code::precondition_failed,
                                    "Not enough data points for k clusters",
                                    "kmeans"});
    }

    if (params.k == 0) {
        return std::vesper_unexpected(error{error_code::precondition_failed,
                                    "k must be > 0",
                                    "kmeans"});
    }

    const auto start_time = std::chrono::steady_clock::now();

    KmeansResult best_result;
    float best_inertia = std::numeric_limits<float>::max();

    // Multiple runs with different initializations
    for (std::uint32_t redo = 0; redo < params.n_redo; ++redo) {
        const std::uint32_t seed = params.seed + redo;

        // Initialize centroids using k-means++
        auto centroids = kmeans_plusplus_init(data, n, dim, params.k, seed);

        std::vector<std::uint32_t> assignments(n);
        float prev_inertia = std::numeric_limits<float>::max();
        float last_inertia = std::numeric_limits<float>::infinity();
        std::uint32_t iter = 0;

        // Lloyd's algorithm iterations
        for (; iter < params.max_iter; ++iter) {
            // Assign points to nearest centroids
            const float inertia = kmeans_assign(data, n, centroids, assignments);
            last_inertia = inertia;

            // Check convergence
            const float change = std::abs(prev_inertia - inertia) / (prev_inertia + 1e-10f);
            if (change < params.epsilon) {
                if (params.verbose) {
                    // Log convergence
                }
                break;
            }

            prev_inertia = inertia;

            // Update centroids
            kmeans_update_centroids(data, n, dim, assignments, params.k, centroids);
        }

        // Ensure centroids correspond to final assignments and recompute inertia consistently
        kmeans_update_centroids(data, n, dim, assignments, params.k, centroids);
        {
            float final_inertia = 0.0f;
            #pragma omp parallel for reduction(+:final_inertia)
            for (int i = 0; i < static_cast<int>(n); ++i) {
                const float* point = data + static_cast<std::size_t>(i) * dim;
                const auto& centroid = centroids[assignments[static_cast<std::size_t>(i)]];
                float dist = 0.0f;
                for (std::size_t d = 0; d < dim; ++d) {
                    float diff = point[d] - centroid[d];
                    dist += diff * diff;
                }
                final_inertia += dist;
            }
            last_inertia = final_inertia;
        }

    // Final consistency: recompute inertia from returned centroids/assignments

        // Keep best result
        if (last_inertia < best_inertia) {
            best_inertia = last_inertia;

            // Compute cluster sizes
            std::vector<std::uint32_t> cluster_sizes(params.k, 0);
            for (std::uint32_t a : assignments) {
                cluster_sizes[a]++;
            }

            best_result = KmeansResult{
                .centroids = std::move(centroids),
                .assignments = std::move(assignments),
                .cluster_sizes = std::move(cluster_sizes),
                .inertia = last_inertia,
                .iterations = iter
            };
        }
    }

    const auto end_time = std::chrono::steady_clock::now();
    const auto duration = std::chrono::duration<float>(end_time - start_time);
    best_result.time_sec = duration.count();

    // Final consistency: recompute inertia for the selected best solution
    {
        float inertia = 0.0f;
        for (std::size_t i = 0; i < n; ++i) {
            const float* point = data + i * dim;
            const auto& centroid = best_result.centroids[best_result.assignments[i]];
            for (std::size_t d = 0; d < dim; ++d) {
                float diff = point[d] - centroid[d];
                inertia += diff * diff;
            }
        }
        best_result.inertia = inertia;
    }

    return best_result;
}

auto hierarchical_kmeans(const float* data, std::size_t n, std::size_t dim,
                         std::uint32_t k, std::uint32_t branching_factor,
                         const KmeansParams& params)
    -> std::expected<KmeansResult, core::error> {
    using core::error;
    using core::error_code;


    if (branching_factor < 2) {
        return std::vesper_unexpected(error{error_code::precondition_failed,
                                    "Branching factor must be >= 2",
                                    "kmeans"});
    }
    if (k < branching_factor) {
        // If k is less than branching factor, use regular k-means
        return kmeans_cluster(data, n, dim, params);
    }

    // Hierarchical k-means implementation
    struct HierarchicalNode {
        std::vector<float> centroid;
        std::vector<std::uint32_t> indices;  // Data point indices in this node
        std::vector<std::unique_ptr<HierarchicalNode>> children;
        std::uint32_t level;
        std::uint32_t cluster_id;  // Final cluster ID for leaf nodes
    };

    // Build hierarchical tree
    auto root = std::make_unique<HierarchicalNode>();
    root->level = 0;
    root->indices.resize(n);
    std::iota(root->indices.begin(), root->indices.end(), 0);

    // Queue for breadth-first tree construction
    std::queue<HierarchicalNode*> node_queue;
    node_queue.push(root.get());

    std::uint32_t next_cluster_id = 0;
    std::vector<HierarchicalNode*> leaf_nodes;

    // Calculate tree depth needed
    std::uint32_t tree_depth = static_cast<std::uint32_t>(
        std::ceil(std::log(k) / std::log(branching_factor))
    );

    while (!node_queue.empty() && next_cluster_id < k) {
        HierarchicalNode* current = node_queue.front();
        node_queue.pop();

        // Check if we should split this node
        bool should_split = current->level < tree_depth - 1 ||
                           (current->level == tree_depth - 1 && next_cluster_id + branching_factor <= k);

        if (!should_split || current->indices.size() < branching_factor) {
            // Make this a leaf node
            current->cluster_id = next_cluster_id++;
            leaf_nodes.push_back(current);

            // Compute centroid for this leaf
            current->centroid.resize(dim, 0.0f);
            for (std::uint32_t idx : current->indices) {
                for (std::size_t d = 0; d < dim; ++d) {
                    current->centroid[d] += data[idx * dim + d];
                }
            }
            if (!current->indices.empty()) {
                for (std::size_t d = 0; d < dim; ++d) {
                    current->centroid[d] /= current->indices.size();
                }
            }
            continue;
        }

        // Perform k-means clustering on this node's data
        std::size_t node_n = current->indices.size();
        std::vector<float> node_data(node_n * dim);

        // Copy data for this node
        for (std::size_t i = 0; i < node_n; ++i) {
            std::uint32_t idx = current->indices[i];
            std::memcpy(&node_data[i * dim], &data[idx * dim], dim * sizeof(float));
        }

        // Run k-means with branching_factor clusters
        KmeansParams local_params = params;
        local_params.k = std::min(branching_factor, static_cast<std::uint32_t>(node_n));
        local_params.max_iter = 10;  // Fewer iterations for intermediate nodes

        auto local_result = kmeans_cluster(node_data.data(), node_n, dim, local_params);
        if (!local_result) {
            return std::vesper_unexpected(local_result.error());
        }

        // Create child nodes
        current->children.resize(local_params.k);
        for (std::uint32_t c = 0; c < local_params.k; ++c) {
            current->children[c] = std::make_unique<HierarchicalNode>();
            current->children[c]->level = current->level + 1;
            current->children[c]->centroid = local_result->centroids[c];
        }

        // Assign data points to children
        for (std::size_t i = 0; i < node_n; ++i) {
            std::uint32_t cluster = local_result->assignments[i];
            current->children[cluster]->indices.push_back(current->indices[i]);
        }

        // Add non-empty children to queue
        for (auto& child : current->children) {
            if (!child->indices.empty()) {
                node_queue.push(child.get());
            }
        }
    }

    // Build final result from leaf nodes
    KmeansResult result;
    result.centroids.resize(leaf_nodes.size());
    result.assignments.resize(n);
    result.cluster_sizes.resize(leaf_nodes.size(), 0);
    result.inertia = 0.0f;

    // Extract centroids and compute assignments
    for (std::size_t i = 0; i < leaf_nodes.size(); ++i) {
        result.centroids[i] = leaf_nodes[i]->centroid;

        // Assign all points in this leaf to cluster i
        for (std::uint32_t idx : leaf_nodes[i]->indices) {
            result.assignments[idx] = static_cast<std::uint32_t>(i);
            result.cluster_sizes[i]++;

            // Compute distance for inertia
            float dist = 0.0f;
            for (std::size_t d = 0; d < dim; ++d) {
                float diff = data[idx * dim + d] - result.centroids[i][d];
                dist += diff * diff;
            }
            result.inertia += dist;
        }
    }

    // If we have fewer clusters than requested, add empty ones
    while (result.centroids.size() < k) {
        result.centroids.push_back(std::vector<float>(dim, 0.0f));
        result.cluster_sizes.push_back(0);
    }

    result.iterations = tree_depth;  // Use tree depth as iteration count

    return result;
}

auto evaluate_clustering(const float* data, std::size_t n, std::size_t dim,
                         const KmeansResult& result) -> ClusterMetrics {
    ClusterMetrics metrics;

    if (n == 0 || result.centroids.empty() || result.assignments.empty()) {
        return metrics;
    }

    const std::size_t k = result.centroids.size();

    // Compute cluster sizes and within-cluster sum of squares
    std::vector<std::size_t> cluster_sizes(k, 0);
    std::vector<float> within_ss(k, 0.0f);

    for (std::size_t i = 0; i < n; ++i) {
        std::uint32_t label = result.assignments[i];
        if (label >= k) continue;

        cluster_sizes[label]++;

        // Compute squared distance to centroid
        float dist_sq = 0.0f;
        for (std::size_t d = 0; d < dim; ++d) {
            float diff = data[i * dim + d] - result.centroids[label][d];
            dist_sq += diff * diff;
        }
        within_ss[label] += dist_sq;
    }

    // Compute global centroid
    std::vector<float> global_centroid(dim, 0.0f);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t d = 0; d < dim; ++d) {
            global_centroid[d] += data[i * dim + d];
        }
    }
    for (std::size_t d = 0; d < dim; ++d) {
        global_centroid[d] /= static_cast<float>(n);
    }

    // Compute between-cluster sum of squares
    float between_ss = 0.0f;
    for (std::size_t c = 0; c < k; ++c) {
        if (cluster_sizes[c] == 0) continue;

        float dist_sq = 0.0f;
        for (std::size_t d = 0; d < dim; ++d) {
            float diff = result.centroids[c][d] - global_centroid[d];
            dist_sq += diff * diff;
        }
        between_ss += static_cast<float>(cluster_sizes[c]) * dist_sq;
    }

    // Compute total within-cluster sum of squares
    float total_within_ss = 0.0f;
    for (std::size_t c = 0; c < k; ++c) {
        total_within_ss += within_ss[c];
    }

    // Calinski-Harabasz index = (between_ss / (k-1)) / (within_ss / (n-k))
    if (k > 1 && n > k && total_within_ss > 0) {
        metrics.calinski_harabasz = (between_ss / (k - 1)) / (total_within_ss / (n - k));
    }

    // Davies-Bouldin index - average similarity between each cluster and its most similar cluster
    metrics.davies_bouldin = 0.0f;
    for (std::size_t i = 0; i < k; ++i) {
        if (cluster_sizes[i] == 0) continue;

        float max_similarity = 0.0f;
        float avg_dist_i = within_ss[i] / cluster_sizes[i];

        for (std::size_t j = 0; j < k; ++j) {
            if (i == j || cluster_sizes[j] == 0) continue;

            float avg_dist_j = within_ss[j] / cluster_sizes[j];

            // Compute distance between centroids
            float centroid_dist = 0.0f;
            for (std::size_t d = 0; d < dim; ++d) {
                float diff = result.centroids[i][d] - result.centroids[j][d];
                centroid_dist += diff * diff;
            }
            centroid_dist = std::sqrt(centroid_dist);

            if (centroid_dist > 0) {
                float similarity = (std::sqrt(avg_dist_i) + std::sqrt(avg_dist_j)) / centroid_dist;
                max_similarity = std::max(max_similarity, similarity);
            }
        }

        metrics.davies_bouldin += max_similarity;
    }

    if (k > 0) {
        metrics.davies_bouldin /= k;
    }

    // Simplified silhouette coefficient (full computation is O(n²))
    // We'll compute an approximation using cluster centroids
    float total_silhouette = 0.0f;
    std::size_t valid_points = 0;

    for (std::size_t i = 0; i < n; ++i) {
        std::uint32_t label = result.assignments[i];
        if (label >= k || cluster_sizes[label] <= 1) continue;

        // a(i) = average distance to points in same cluster (approximated)
        float a_i = std::sqrt(within_ss[label] / cluster_sizes[label]);

        // b(i) = minimum average distance to points in other clusters
        float b_i = std::numeric_limits<float>::max();
        for (std::size_t c = 0; c < k; ++c) {
            if (c == label || cluster_sizes[c] == 0) continue;

            // Distance to other cluster's centroid
            float dist = 0.0f;
            for (std::size_t d = 0; d < dim; ++d) {
                float diff = data[i * dim + d] - result.centroids[c][d];
                dist += diff * diff;
            }
            b_i = std::min(b_i, std::sqrt(dist));
        }

        if (b_i < std::numeric_limits<float>::max()) {
            float s_i = (b_i - a_i) / std::max(a_i, b_i);
            total_silhouette += s_i;
            valid_points++;
        }
    }

    if (valid_points > 0) {
        metrics.silhouette = total_silhouette / valid_points;
    }

    return metrics;
}

} // namespace vesper::index