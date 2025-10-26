/** \file coarse_filter.cpp
 *  \brief Ultra-fast geometric filtering using skyline signatures.
 *
 * First phase of CGF pipeline - eliminates 99% of candidates using
 * cheap geometric tests based on random projections.
 */

#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>
#include <immintrin.h>
#include <vector>

#include "vesper/index/cgf.hpp"
#include "vesper/kernels/distance.hpp"

namespace vesper::index {

class CoarseFilter {
public:
    CoarseFilter(std::size_t dim, std::uint32_t n_projections)
        : dim_(dim), n_projections_(n_projections) {
        
        // Generate random projection axes
        projection_axes_.resize(n_projections_ * dim_);
        std::mt19937 gen(42);  // Fixed seed for reproducibility
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        for (auto& val : projection_axes_) {
            val = dist(gen);
        }
        
        // Normalize each projection axis
        for (std::uint32_t i = 0; i < n_projections_; ++i) {
            float* axis = projection_axes_.data() + i * dim_;
            float norm = 0.0f;
            for (std::size_t d = 0; d < dim_; ++d) {
                norm += axis[d] * axis[d];
            }
            norm = std::sqrt(norm);
            if (norm > 0) {
                for (std::size_t d = 0; d < dim_; ++d) {
                    axis[d] /= norm;
                }
            }
        }
    }
    
    /** Build skyline signatures from cluster centroids. */
    auto build_signatures(const std::vector<std::vector<float>>& cluster_centroids,
                         std::uint32_t n_super_clusters)
        -> std::vector<SkylineSignature> {
        
        const std::uint32_t n_clusters = static_cast<std::uint32_t>(cluster_centroids.size());
        
        // Determine super-cluster assignments using k-means
        std::vector<std::uint32_t> assignments(n_clusters);
        std::vector<std::vector<float>> super_centroids;
        
        if (n_super_clusters == 0) {
            // Auto-determine: fourth root of total clusters
            n_super_clusters = std::max(1u, static_cast<std::uint32_t>(
                std::pow(n_clusters, 0.25)));
        }
        
        // Hierarchical clustering of cluster centroids for better organization
        super_centroids = hierarchical_cluster(cluster_centroids, n_super_clusters, assignments);
        
        // Build skyline signatures
        std::vector<SkylineSignature> signatures(n_super_clusters);
        
        for (std::uint32_t sc = 0; sc < n_super_clusters; ++sc) {
            auto& sig = signatures[sc];
            sig.min_projections.resize(n_projections_, std::numeric_limits<float>::max());
            sig.max_projections.resize(n_projections_, std::numeric_limits<float>::lowest());
            
            // Copy super-centroid
            std::copy(super_centroids[sc].begin(), 
                     super_centroids[sc].end(),
                     sig.centroid.begin());
            
            // Find member clusters and compute projections
            float max_dist = 0.0f;
            for (std::uint32_t c = 0; c < n_clusters; ++c) {
                if (assignments[c] == sc) {
                    sig.member_clusters.push_back(c);
                    
                    // Compute projections for this cluster centroid
                    for (std::uint32_t p = 0; p < n_projections_; ++p) {
                        float proj = compute_projection(
                            cluster_centroids[c].data(),
                            projection_axes_.data() + p * dim_
                        );
                        sig.min_projections[p] = std::min(sig.min_projections[p], proj);
                        sig.max_projections[p] = std::max(sig.max_projections[p], proj);
                    }
                    
                    // Update bounding radius
                    float dist = compute_l2_distance(
                        cluster_centroids[c].data(),
                        sig.centroid.data(),
                        dim_
                    );
                    max_dist = std::max(max_dist, dist);
                }
            }
            
            sig.radius = max_dist * 1.1f;  // Add 10% margin
        }
        
        return signatures;
    }
    
    /** Filter super-clusters using geometric bounds. */
    auto filter_super_clusters(const float* query,
                              const std::vector<SkylineSignature>& signatures,
                              float search_radius = 0.0f) const
        -> std::vector<std::uint32_t> {
        
        std::vector<std::uint32_t> surviving;
        
        // Compute query projections once
        std::vector<float> query_projections(n_projections_);
        for (std::uint32_t p = 0; p < n_projections_; ++p) {
            query_projections[p] = compute_projection(
                query,
                projection_axes_.data() + p * dim_
            );
        }
        
        // If no search radius specified, use adaptive radius based on k-NN estimate
        if (search_radius <= 0.0f) {
            // Find k-th nearest cluster center for radius estimation
            std::vector<float> center_dists;
            center_dists.reserve(signatures.size());
            for (const auto& sig : signatures) {
                float dist = compute_l2_distance(query, sig.centroid.data(), dim_);
                center_dists.push_back(dist);
            }
            
            // Use median distance as conservative estimate
            std::size_t median_idx = center_dists.size() / 2;
            std::nth_element(center_dists.begin(), 
                           center_dists.begin() + median_idx,
                           center_dists.end());
            search_radius = center_dists[median_idx] * 1.5f;  // 50% margin
        }
        
        // Test each super-cluster with multi-stage filtering
        for (std::uint32_t sc = 0; sc < signatures.size(); ++sc) {
            const auto& sig = signatures[sc];
            
            // Stage 1: Coarse distance bound using triangle inequality
            float center_dist = compute_l2_distance(query, sig.centroid.data(), dim_);
            if (center_dist > sig.radius + search_radius) {
                continue;  // Too far from cluster boundary
            }
            
            // Stage 2: Fine-grained projection-based elimination
            // A cluster can be eliminated if the query is outside the expanded
            // bounding box in ANY projection (not all projections)
            bool can_eliminate = false;
            
            for (std::uint32_t p = 0; p < n_projections_; ++p) {
                float q_proj = query_projections[p];
                
                // Calculate projection bounds with margin
                float min_bound = sig.min_projections[p] - search_radius;
                float max_bound = sig.max_projections[p] + search_radius;
                
                // Check if query is outside bounds in this projection
                if (q_proj < min_bound || q_proj > max_bound) {
                    // Query is outside bounds for this projection
                    // Can safely eliminate based on this projection
                    can_eliminate = true;
                    break;  // No need to check other projections
                }
            }
            
            // Stage 3: Apply elimination decision
            // Keep the cluster if it cannot be safely eliminated OR
            // if it's very close to the center (override elimination)
            if (!can_eliminate || center_dist <= sig.radius * 0.5f) {
                surviving.push_back(sc);
            }
        }
        
        // Ensure we don't eliminate everything (safety check)
        if (surviving.empty() && !signatures.empty()) {
            // Return closest cluster as fallback
            std::uint32_t closest = 0;
            float min_dist = std::numeric_limits<float>::max();
            
            for (std::uint32_t sc = 0; sc < signatures.size(); ++sc) {
                float dist = compute_l2_distance(query, 
                                                signatures[sc].centroid.data(), 
                                                dim_);
                if (dist < min_dist) {
                    min_dist = dist;
                    closest = sc;
                }
            }
            surviving.push_back(closest);
        }
        
        return surviving;
    }
    
    /** Get clusters from surviving super-clusters. */
    auto get_probe_clusters(const std::vector<std::uint32_t>& super_clusters,
                           const std::vector<SkylineSignature>& signatures) const
        -> std::vector<std::uint32_t> {
        
        std::vector<std::uint32_t> clusters;
        
        for (std::uint32_t sc : super_clusters) {
            const auto& sig = signatures[sc];
            clusters.insert(clusters.end(),
                          sig.member_clusters.begin(),
                          sig.member_clusters.end());
        }
        
        // Remove duplicates and sort
        std::sort(clusters.begin(), clusters.end());
        clusters.erase(std::unique(clusters.begin(), clusters.end()), clusters.end());
        
        return clusters;
    }

private:
    std::size_t dim_;
    std::uint32_t n_projections_;
    std::vector<float> projection_axes_;
    
    /** Compute dot product projection. */
    float compute_projection(const float* vec, const float* axis) const {
        float sum = 0.0f;
        
        #ifdef __AVX2__
        // AVX2 vectorized dot product with reduced horizontal operations
        const std::size_t simd_width = 8;
        const std::size_t simd_iters = dim_ / simd_width;
        
        // Use multiple accumulators to reduce dependency chains
        __m256 sum_vec0 = _mm256_setzero_ps();
        __m256 sum_vec1 = _mm256_setzero_ps();
        __m256 sum_vec2 = _mm256_setzero_ps();
        __m256 sum_vec3 = _mm256_setzero_ps();
        
        // Process 4 vectors at a time for better ILP
        std::size_t i = 0;
        for (; i + 3 < simd_iters; i += 4) {
            __m256 v0 = _mm256_loadu_ps(vec + (i + 0) * simd_width);
            __m256 a0 = _mm256_loadu_ps(axis + (i + 0) * simd_width);
            sum_vec0 = _mm256_fmadd_ps(v0, a0, sum_vec0);
            
            __m256 v1 = _mm256_loadu_ps(vec + (i + 1) * simd_width);
            __m256 a1 = _mm256_loadu_ps(axis + (i + 1) * simd_width);
            sum_vec1 = _mm256_fmadd_ps(v1, a1, sum_vec1);
            
            __m256 v2 = _mm256_loadu_ps(vec + (i + 2) * simd_width);
            __m256 a2 = _mm256_loadu_ps(axis + (i + 2) * simd_width);
            sum_vec2 = _mm256_fmadd_ps(v2, a2, sum_vec2);
            
            __m256 v3 = _mm256_loadu_ps(vec + (i + 3) * simd_width);
            __m256 a3 = _mm256_loadu_ps(axis + (i + 3) * simd_width);
            sum_vec3 = _mm256_fmadd_ps(v3, a3, sum_vec3);
        }
        
        // Process remaining vectors
        for (; i < simd_iters; ++i) {
            __m256 v = _mm256_loadu_ps(vec + i * simd_width);
            __m256 a = _mm256_loadu_ps(axis + i * simd_width);
            sum_vec0 = _mm256_fmadd_ps(v, a, sum_vec0);
        }
        
        // Combine accumulators
        sum_vec0 = _mm256_add_ps(sum_vec0, sum_vec1);
        sum_vec2 = _mm256_add_ps(sum_vec2, sum_vec3);
        sum_vec0 = _mm256_add_ps(sum_vec0, sum_vec2);
        
        // Efficient horizontal sum using shuffles instead of hadd
        __m128 sum_high = _mm256_extractf128_ps(sum_vec0, 1);
        __m128 sum_low = _mm256_castps256_ps128(sum_vec0);
        __m128 sum_128 = _mm_add_ps(sum_high, sum_low);
        
        // Shuffle and add instead of hadd (faster on most CPUs)
        __m128 shuf = _mm_shuffle_ps(sum_128, sum_128, _MM_SHUFFLE(2, 3, 0, 1));
        sum_128 = _mm_add_ps(sum_128, shuf);
        shuf = _mm_movehl_ps(shuf, sum_128);
        sum_128 = _mm_add_ss(sum_128, shuf);
        sum = _mm_cvtss_f32(sum_128);
        
        // Handle remainder
        for (std::size_t j = simd_iters * simd_width; j < dim_; ++j) {
            sum += vec[j] * axis[j];
        }
        #else
        // Scalar fallback
        for (std::size_t i = 0; i < dim_; ++i) {
            sum += vec[i] * axis[i];
        }
        #endif
        
        return sum;
    }
    
    /** Compute L2 distance. */
    float compute_l2_distance(const float* a, const float* b, std::size_t dim) const {
        float sum = 0.0f;
        
        #ifdef __AVX2__
        // AVX2 vectorized L2 distance with reduced horizontal operations
        const std::size_t simd_width = 8;
        const std::size_t simd_iters = dim / simd_width;
        
        // Use multiple accumulators for better ILP
        __m256 sum_vec0 = _mm256_setzero_ps();
        __m256 sum_vec1 = _mm256_setzero_ps();
        __m256 sum_vec2 = _mm256_setzero_ps();
        __m256 sum_vec3 = _mm256_setzero_ps();
        
        // Process 4 vectors at a time
        std::size_t i = 0;
        for (; i + 3 < simd_iters; i += 4) {
            __m256 va0 = _mm256_loadu_ps(a + (i + 0) * simd_width);
            __m256 vb0 = _mm256_loadu_ps(b + (i + 0) * simd_width);
            __m256 diff0 = _mm256_sub_ps(va0, vb0);
            sum_vec0 = _mm256_fmadd_ps(diff0, diff0, sum_vec0);
            
            __m256 va1 = _mm256_loadu_ps(a + (i + 1) * simd_width);
            __m256 vb1 = _mm256_loadu_ps(b + (i + 1) * simd_width);
            __m256 diff1 = _mm256_sub_ps(va1, vb1);
            sum_vec1 = _mm256_fmadd_ps(diff1, diff1, sum_vec1);
            
            __m256 va2 = _mm256_loadu_ps(a + (i + 2) * simd_width);
            __m256 vb2 = _mm256_loadu_ps(b + (i + 2) * simd_width);
            __m256 diff2 = _mm256_sub_ps(va2, vb2);
            sum_vec2 = _mm256_fmadd_ps(diff2, diff2, sum_vec2);
            
            __m256 va3 = _mm256_loadu_ps(a + (i + 3) * simd_width);
            __m256 vb3 = _mm256_loadu_ps(b + (i + 3) * simd_width);
            __m256 diff3 = _mm256_sub_ps(va3, vb3);
            sum_vec3 = _mm256_fmadd_ps(diff3, diff3, sum_vec3);
        }
        
        // Process remaining vectors
        for (; i < simd_iters; ++i) {
            __m256 va = _mm256_loadu_ps(a + i * simd_width);
            __m256 vb = _mm256_loadu_ps(b + i * simd_width);
            __m256 diff = _mm256_sub_ps(va, vb);
            sum_vec0 = _mm256_fmadd_ps(diff, diff, sum_vec0);
        }
        
        // Combine accumulators
        sum_vec0 = _mm256_add_ps(sum_vec0, sum_vec1);
        sum_vec2 = _mm256_add_ps(sum_vec2, sum_vec3);
        sum_vec0 = _mm256_add_ps(sum_vec0, sum_vec2);
        
        // Efficient horizontal sum using shuffles
        __m128 sum_high = _mm256_extractf128_ps(sum_vec0, 1);
        __m128 sum_low = _mm256_castps256_ps128(sum_vec0);
        __m128 sum_128 = _mm_add_ps(sum_high, sum_low);
        
        // Shuffle and add instead of hadd
        __m128 shuf = _mm_shuffle_ps(sum_128, sum_128, _MM_SHUFFLE(2, 3, 0, 1));
        sum_128 = _mm_add_ps(sum_128, shuf);
        shuf = _mm_movehl_ps(shuf, sum_128);
        sum_128 = _mm_add_ss(sum_128, shuf);
        sum = _mm_cvtss_f32(sum_128);
        
        // Handle remainder
        for (std::size_t j = simd_iters * simd_width; j < dim; ++j) {
            float diff = a[j] - b[j];
            sum += diff * diff;
        }
        #else
        // Scalar fallback
        for (std::size_t i = 0; i < dim; ++i) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        #endif
        
        return std::sqrt(sum);
    }
    
    /** Hierarchical clustering for better super-cluster creation. */
    std::vector<std::vector<float>> hierarchical_cluster(
        const std::vector<std::vector<float>>& points,
        std::uint32_t k,
        std::vector<std::uint32_t>& assignments) {
        
        const std::size_t n = points.size();
        const std::size_t dim = points[0].size();
        
        // Use hierarchical approach: first create 2*k clusters, then merge
        if (k > 1 && n > 2 * k) {
            // Phase 1: Create fine-grained clusters
            std::uint32_t fine_k = std::min<std::uint32_t>(2 * k, static_cast<std::uint32_t>(n));
            std::vector<std::uint32_t> fine_assignments;
            auto fine_centroids = kmeans_cluster_impl(points, fine_k, fine_assignments);
            
            // Phase 2: Merge to final k clusters using balanced tree
            std::vector<std::uint32_t> final_assignments;
            auto final_centroids = kmeans_cluster_impl(fine_centroids, k, final_assignments);
            
            // Map fine assignments to final assignments
            assignments.resize(n);
            for (std::size_t i = 0; i < n; ++i) {
                assignments[i] = final_assignments[fine_assignments[i]];
            }
            
            return final_centroids;
        }
        
        // Fallback to simple k-means for small k or n
        return kmeans_cluster_impl(points, k, assignments);
    }
    
    /** Simple k-means implementation. */
    std::vector<std::vector<float>> kmeans_cluster_impl(
        const std::vector<std::vector<float>>& points,
        std::uint32_t k,
        std::vector<std::uint32_t>& assignments) {
        
        const std::size_t n = points.size();
        const std::size_t dim = points[0].size();
        
        // Initialize centroids with k-means++
        std::vector<std::vector<float>> centroids(k, std::vector<float>(dim));
        std::mt19937 gen(42);
        
        // First centroid: random point
        std::uniform_int_distribution<std::size_t> dist(0, n - 1);
        centroids[0] = points[dist(gen)];
        
        // Remaining centroids: weighted by distance
        for (std::uint32_t c = 1; c < k; ++c) {
            std::vector<float> min_dists(n, std::numeric_limits<float>::max());
            
            for (std::size_t i = 0; i < n; ++i) {
                for (std::uint32_t j = 0; j < c; ++j) {
                    float dist = compute_l2_distance(
                        points[i].data(),
                        centroids[j].data(),
                        dim
                    );
                    min_dists[i] = std::min(min_dists[i], dist * dist);
                }
            }
            
            // Weighted sampling
            std::discrete_distribution<std::size_t> weighted_dist(
                min_dists.begin(), min_dists.end()
            );
            centroids[c] = points[weighted_dist(gen)];
        }
        
        // Lloyd's iterations
        const std::uint32_t max_iters = 25;
        assignments.resize(n);
        
        for (std::uint32_t iter = 0; iter < max_iters; ++iter) {
            // Assign points to nearest centroid
            bool changed = false;
            for (std::size_t i = 0; i < n; ++i) {
                float min_dist = std::numeric_limits<float>::max();
                std::uint32_t best_c = 0;
                
                for (std::uint32_t c = 0; c < k; ++c) {
                    float dist = compute_l2_distance(
                        points[i].data(),
                        centroids[c].data(),
                        dim
                    );
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_c = c;
                    }
                }
                
                if (assignments[i] != best_c) {
                    changed = true;
                    assignments[i] = best_c;
                }
            }
            
            if (!changed) break;
            
            // Update centroids
            for (std::uint32_t c = 0; c < k; ++c) {
                std::fill(centroids[c].begin(), centroids[c].end(), 0.0f);
                std::uint32_t count = 0;
                
                for (std::size_t i = 0; i < n; ++i) {
                    if (assignments[i] == c) {
                        for (std::size_t d = 0; d < dim; ++d) {
                            centroids[c][d] += points[i][d];
                        }
                        count++;
                    }
                }
                
                if (count > 0) {
                    for (std::size_t d = 0; d < dim; ++d) {
                        centroids[c][d] /= count;
                    }
                }
            }
        }
        
        return centroids;
    }
};

} // namespace vesper::index