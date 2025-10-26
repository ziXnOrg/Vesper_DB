/** \file smart_ivf.cpp
 *  \brief Intelligent cluster selection using learned predictor.
 *
 * Second phase of CGF pipeline - uses a lightweight neural network
 * to predict which clusters are likely to contain k-NN, enabling
 * adaptive probing that stops when confidence threshold is reached.
 */

#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <queue>
#include <immintrin.h>
#include <unordered_set>
#include <expected>

#include "vesper/index/cgf.hpp"
#include "vesper/kernels/distance.hpp"
#include "vesper/error.hpp"

namespace vesper::index {

class SmartIVF {
public:
    /** Neural network for cluster prediction. */
    class ClusterPredictor {
    public:
        ClusterPredictor(std::size_t dim, std::uint32_t n_clusters)
            : dim_(dim), n_clusters_(n_clusters) {
            
            // 2-layer network: dim+n_clusters -> 128 -> n_clusters
            hidden_size_ = 128;
            
            // Initialize weights with Xavier initialization
            std::mt19937 gen(42);
            float xavier_scale1 = std::sqrt(2.0f / (dim_ + n_clusters_));
            float xavier_scale2 = std::sqrt(2.0f / hidden_size_);
            
            std::normal_distribution<float> dist1(0.0f, xavier_scale1);
            std::normal_distribution<float> dist2(0.0f, xavier_scale2);
            
            // Layer 1: (dim + n_clusters) -> hidden_size
            W1_.resize((dim_ + n_clusters_) * hidden_size_);
            b1_.resize(hidden_size_, 0.0f);
            for (auto& w : W1_) w = dist1(gen);
            
            // Layer 2: hidden_size -> n_clusters
            W2_.resize(hidden_size_ * n_clusters_);
            b2_.resize(n_clusters_, 0.0f);
            for (auto& w : W2_) w = dist2(gen);
        }
        
        /** Predict cluster probabilities for query. */
        auto predict(const float* query, 
                    const std::vector<float>& cluster_distances) const
            -> std::vector<float> {
            
            // Prepare input features
            std::vector<float> features;
            features.reserve(dim_ + n_clusters_);
            
            // Query features
            features.insert(features.end(), query, query + dim_);
            
            // Normalized cluster distances as features
            float min_dist = *std::min_element(cluster_distances.begin(), 
                                              cluster_distances.end());
            float max_dist = *std::max_element(cluster_distances.begin(), 
                                              cluster_distances.end());
            float range = max_dist - min_dist + 1e-6f;
            
            for (float dist : cluster_distances) {
                features.push_back((dist - min_dist) / range);
            }
            
            // Forward pass through network
            std::vector<float> hidden = forward_layer1(features.data());
            std::vector<float> output = forward_layer2(hidden.data());
            
            // Apply softmax
            return softmax(output);
        }
        
        /** Train predictor on query-cluster pairs. */
        auto train(const std::vector<TrainingExample>& examples,
                  std::uint32_t epochs = 10,
                  float learning_rate = 0.001f) -> void {
            
            for (std::uint32_t epoch = 0; epoch < epochs; ++epoch) {
                float total_loss = 0.0f;
                
                for (const auto& example : examples) {
                    // Forward pass
                    auto probs = predict(example.query.data(), 
                                       example.cluster_distances);
                    
                    // Compute cross-entropy loss
                    float loss = -std::log(probs[example.target_cluster] + 1e-10f);
                    total_loss += loss;
                    
                    // Backward pass (simplified SGD)
                    update_weights(example, probs, learning_rate);
                }
                
                // Decay learning rate
                learning_rate *= 0.95f;
            }
        }
        
        struct TrainingExample {
            std::vector<float> query;
            std::vector<float> cluster_distances;
            std::uint32_t target_cluster;
        };
        
        /** Get top-k clusters by predicted probability. */
        auto get_top_clusters(const float* query,
                             const std::vector<float>& cluster_distances,
                             float confidence_threshold = 0.95f) const
            -> std::vector<std::uint32_t> {
            
            auto probs = predict(query, cluster_distances);
            
            // Sort clusters by probability
            std::vector<std::pair<float, std::uint32_t>> scored;
            for (std::uint32_t i = 0; i < n_clusters_; ++i) {
                scored.emplace_back(probs[i], i);
            }
            std::sort(scored.begin(), scored.end(), std::greater<>());
            
            // Take clusters until confidence threshold
            std::vector<std::uint32_t> selected;
            float cumulative_prob = 0.0f;
            
            for (const auto& [prob, cluster_id] : scored) {
                selected.push_back(cluster_id);
                cumulative_prob += prob;
                if (cumulative_prob >= confidence_threshold) {
                    break;
                }
            }
            
            return selected;
        }
        
    private:
        std::size_t dim_;
        std::uint32_t n_clusters_;
        std::uint32_t hidden_size_;
        
        // Network weights
        std::vector<float> W1_, b1_;  // First layer
        std::vector<float> W2_, b2_;  // Second layer
        
        /** Forward pass through first layer. */
        std::vector<float> forward_layer1(const float* input) const {
            std::vector<float> hidden(hidden_size_);
            
            // Compute W1 * input + b1
            for (std::uint32_t h = 0; h < hidden_size_; ++h) {
                float sum = b1_[h];
                const float* w = W1_.data() + h * (dim_ + n_clusters_);
                
                // Vectorized dot product if available
                #ifdef __AVX2__
                std::size_t input_size = dim_ + n_clusters_;
                std::size_t simd_iters = input_size / 8;
                __m256 sum_vec = _mm256_setzero_ps();
                
                for (std::size_t i = 0; i < simd_iters; ++i) {
                    __m256 w_vec = _mm256_loadu_ps(w + i * 8);
                    __m256 in_vec = _mm256_loadu_ps(input + i * 8);
                    sum_vec = _mm256_fmadd_ps(w_vec, in_vec, sum_vec);
                }
                
                // Horizontal sum
                __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
                __m128 sum_low = _mm256_castps256_ps128(sum_vec);
                __m128 sum_128 = _mm_add_ps(sum_high, sum_low);
                sum_128 = _mm_hadd_ps(sum_128, sum_128);
                sum_128 = _mm_hadd_ps(sum_128, sum_128);
                sum += _mm_cvtss_f32(sum_128);
                
                // Handle remainder
                for (std::size_t i = simd_iters * 8; i < input_size; ++i) {
                    sum += w[i] * input[i];
                }
                #else
                for (std::size_t i = 0; i < dim_ + n_clusters_; ++i) {
                    sum += w[i] * input[i];
                }
                #endif
                
                // ReLU activation
                hidden[h] = std::max(0.0f, sum);
            }
            
            return hidden;
        }
        
        /** Forward pass through second layer. */
        std::vector<float> forward_layer2(const float* hidden) const {
            std::vector<float> output(n_clusters_);
            
            // Compute W2 * hidden + b2
            for (std::uint32_t c = 0; c < n_clusters_; ++c) {
                float sum = b2_[c];
                const float* w = W2_.data() + c * hidden_size_;
                
                for (std::uint32_t h = 0; h < hidden_size_; ++h) {
                    sum += w[h] * hidden[h];
                }
                
                output[c] = sum;
            }
            
            return output;
        }
        
        /** Apply softmax to logits. */
        std::vector<float> softmax(const std::vector<float>& logits) const {
            std::vector<float> probs(logits.size());
            
            // Numerical stability: subtract max
            float max_logit = *std::max_element(logits.begin(), logits.end());
            
            float sum_exp = 0.0f;
            for (std::size_t i = 0; i < logits.size(); ++i) {
                probs[i] = std::exp(logits[i] - max_logit);
                sum_exp += probs[i];
            }
            
            // Normalize
            for (auto& p : probs) {
                p /= sum_exp;
            }
            
            return probs;
        }
        
        /** Update weights using gradient descent. */
        void update_weights(const TrainingExample& example,
                           const std::vector<float>& predicted_probs,
                           float lr) {
            // Simplified backpropagation (placeholder)
            // In production, use proper autograd or manual differentiation
            
            // Output gradient
            std::vector<float> grad_output(n_clusters_);
            for (std::uint32_t c = 0; c < n_clusters_; ++c) {
                grad_output[c] = predicted_probs[c];
                if (c == example.target_cluster) {
                    grad_output[c] -= 1.0f;
                }
            }
            
            // Update W2 and b2
            std::vector<float> features(example.query);
            features.insert(features.end(), 
                          example.cluster_distances.begin(),
                          example.cluster_distances.end());
            
            std::vector<float> hidden = forward_layer1(features.data());
            
            for (std::uint32_t c = 0; c < n_clusters_; ++c) {
                b2_[c] -= lr * grad_output[c];
                
                float* w2 = W2_.data() + c * hidden_size_;
                for (std::uint32_t h = 0; h < hidden_size_; ++h) {
                    w2[h] -= lr * grad_output[c] * hidden[h];
                }
            }
            
            // Backprop to hidden layer (simplified)
            // ... (full implementation would continue here)
        }
    };
    
    SmartIVF(std::size_t dim, std::uint32_t n_clusters)
        : dim_(dim), n_clusters_(n_clusters),
          predictor_(dim, n_clusters) {}
    
    /** Train clustering and predictor. */
    auto train(const float* data, std::size_t n,
              const std::vector<std::uint64_t>& ids) -> void {
        
        // Train k-means clustering
        train_clusters(data, n);
        
        // Assign points to clusters
        assign_to_clusters(data, n, ids);
        
        // Generate training examples for predictor
        auto examples = generate_training_examples(data, n);
        
        // Train neural network predictor
        predictor_.train(examples);
    }
    
    /** Smart cluster selection for query. */
    auto select_clusters(const float* query,
                        const std::vector<std::uint32_t>& candidate_clusters,
                        float confidence = 0.95f) const
        -> std::vector<std::uint32_t> {
        
        // Compute distances to all cluster centroids
        std::vector<float> cluster_distances(n_clusters_);
        const auto& ops = kernels::select_backend_auto();
        
        for (std::uint32_t c = 0; c < n_clusters_; ++c) {
            cluster_distances[c] = ops.l2_sq(
                std::span(query, dim_),
                std::span(centroids_[c].data(), dim_)
            );
        }
        
        // Filter to candidate clusters if provided
        if (!candidate_clusters.empty()) {
            std::vector<float> filtered_distances(n_clusters_, 
                                                 std::numeric_limits<float>::max());
            for (std::uint32_t c : candidate_clusters) {
                filtered_distances[c] = cluster_distances[c];
            }
            cluster_distances = std::move(filtered_distances);
        }
        
        // Use predictor to select clusters
        return predictor_.get_top_clusters(query, cluster_distances, confidence);
    }
    
    /** Get vectors in selected clusters. */
    auto get_cluster_vectors(const std::vector<std::uint32_t>& clusters) const
        -> std::vector<std::uint64_t> {
        
        std::vector<std::uint64_t> vectors;
        
        for (std::uint32_t c : clusters) {
            const auto& members = cluster_members_[c];
            vectors.insert(vectors.end(), members.begin(), members.end());
        }
        
        // Remove duplicates
        std::sort(vectors.begin(), vectors.end());
        vectors.erase(std::unique(vectors.begin(), vectors.end()), 
                     vectors.end());
        
        return vectors;
    }
    
    /** Adaptive search with early stopping. */
    auto search_adaptive(const float* query,
                       std::uint32_t k,
                       float initial_confidence = 0.5f,
                       float confidence_step = 0.1f,
                       std::uint32_t min_candidates = 100) const
        -> std::vector<std::uint64_t> {
        
        std::vector<std::uint64_t> candidates;
        float confidence = initial_confidence;
        
        // Progressively increase confidence until enough candidates
        while (candidates.size() < min_candidates && confidence <= 1.0f) {
            auto clusters = select_clusters(query, {}, confidence);
            candidates = get_cluster_vectors(clusters);
            confidence += confidence_step;
        }
        
        // Ensure we have at least k candidates
        if (candidates.size() < k) {
            // Fall back to nearest clusters
            auto all_dists = compute_all_cluster_distances(query);
            std::vector<std::pair<float, std::uint32_t>> sorted;
            
            for (std::uint32_t c = 0; c < n_clusters_; ++c) {
                sorted.emplace_back(all_dists[c], c);
            }
            std::sort(sorted.begin(), sorted.end());
            
            for (const auto& [dist, c] : sorted) {
                const auto& members = cluster_members_[c];
                candidates.insert(candidates.end(), 
                                members.begin(), members.end());
                if (candidates.size() >= k * 10) break;
            }
        }
        
        return candidates;
    }
    
    /** Get cluster assignment for a vector. */
    auto get_cluster(std::uint64_t id) const 
        -> std::expected<std::uint32_t, core::error> {
        
        auto it = vector_to_cluster_.find(id);
        if (it == vector_to_cluster_.end()) {
            return std::vesper_unexpected(core::error{
                core::error_code::not_found,
                "Vector not found",
                "smart_ivf"
            });
        }
        return it->second;
    }
    
private:
    std::size_t dim_;
    std::uint32_t n_clusters_;
    ClusterPredictor predictor_;
    
    // Clustering data
    std::vector<std::vector<float>> centroids_;
    std::vector<std::vector<std::uint64_t>> cluster_members_;
    std::unordered_map<std::uint64_t, std::uint32_t> vector_to_cluster_;
    
    /** Train k-means clustering. */
    void train_clusters(const float* data, std::size_t n) {
        centroids_.resize(n_clusters_, std::vector<float>(dim_));
        cluster_members_.resize(n_clusters_);
        
        // Initialize centroids with k-means++
        std::mt19937 gen(42);
        std::uniform_int_distribution<std::size_t> dist(0, n - 1);
        
        // First centroid: random
        std::size_t first = dist(gen);
        std::copy(data + first * dim_, 
                 data + (first + 1) * dim_,
                 centroids_[0].begin());
        
        // Remaining centroids: weighted by distance
        const auto& ops = kernels::select_backend_auto();
        
        for (std::uint32_t c = 1; c < n_clusters_; ++c) {
            std::vector<float> min_dists(n, std::numeric_limits<float>::max());
            
            for (std::size_t i = 0; i < n; ++i) {
                for (std::uint32_t j = 0; j < c; ++j) {
                    float dist = ops.l2_sq(
                        std::span(data + i * dim_, dim_),
                        std::span(centroids_[j].data(), dim_)
                    );
                    min_dists[i] = std::min(min_dists[i], dist);
                }
            }
            
            std::discrete_distribution<std::size_t> weighted(
                min_dists.begin(), min_dists.end()
            );
            std::size_t selected = weighted(gen);
            std::copy(data + selected * dim_,
                     data + (selected + 1) * dim_,
                     centroids_[c].begin());
        }
        
        // Lloyd's iterations
        const std::uint32_t max_iters = 25;
        std::vector<std::uint32_t> assignments(n);
        
        for (std::uint32_t iter = 0; iter < max_iters; ++iter) {
            // Assign points
            bool changed = false;
            for (std::size_t i = 0; i < n; ++i) {
                float min_dist = std::numeric_limits<float>::max();
                std::uint32_t best_c = 0;
                
                for (std::uint32_t c = 0; c < n_clusters_; ++c) {
                    float dist = ops.l2_sq(
                        std::span(data + i * dim_, dim_),
                        std::span(centroids_[c].data(), dim_)
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
            for (auto& centroid : centroids_) {
                std::fill(centroid.begin(), centroid.end(), 0.0f);
            }
            std::vector<std::uint32_t> counts(n_clusters_, 0);
            
            for (std::size_t i = 0; i < n; ++i) {
                std::uint32_t c = assignments[i];
                for (std::size_t d = 0; d < dim_; ++d) {
                    centroids_[c][d] += data[i * dim_ + d];
                }
                counts[c]++;
            }
            
            for (std::uint32_t c = 0; c < n_clusters_; ++c) {
                if (counts[c] > 0) {
                    for (float& val : centroids_[c]) {
                        val /= counts[c];
                    }
                }
            }
        }
    }
    
    /** Assign vectors to clusters. */
    void assign_to_clusters(const float* data, std::size_t n,
                           const std::vector<std::uint64_t>& ids) {
        const auto& ops = kernels::select_backend_auto();
        
        for (std::size_t i = 0; i < n; ++i) {
            float min_dist = std::numeric_limits<float>::max();
            std::uint32_t best_c = 0;
            
            for (std::uint32_t c = 0; c < n_clusters_; ++c) {
                float dist = ops.l2_sq(
                    std::span(data + i * dim_, dim_),
                    std::span(centroids_[c].data(), dim_)
                );
                if (dist < min_dist) {
                    min_dist = dist;
                    best_c = c;
                }
            }
            
            cluster_members_[best_c].push_back(ids[i]);
            vector_to_cluster_[ids[i]] = best_c;
        }
    }
    
    /** Generate training examples for predictor. */
    std::vector<ClusterPredictor::TrainingExample> 
    generate_training_examples(const float* data, std::size_t n) {
        std::vector<ClusterPredictor::TrainingExample> examples;
        
        // Sample queries and find their true nearest clusters
        std::mt19937 gen(42);
        std::uniform_int_distribution<std::size_t> dist(0, n - 1);
        const std::size_t n_samples = std::min<std::size_t>(1000, n / 10);
        
        for (std::size_t s = 0; s < n_samples; ++s) {
            std::size_t idx = dist(gen);
            
            ClusterPredictor::TrainingExample ex;
            ex.query.assign(data + idx * dim_, data + (idx + 1) * dim_);
            ex.cluster_distances = compute_all_cluster_distances(ex.query.data());
            
            // Find cluster containing nearest neighbors
            // (simplified: use assigned cluster as target)
            for (const auto& [id, cluster] : vector_to_cluster_) {
                if (id == idx) {
                    ex.target_cluster = cluster;
                    break;
                }
            }
            
            examples.push_back(std::move(ex));
        }
        
        return examples;
    }
    
    /** Compute distances to all clusters. */
    std::vector<float> compute_all_cluster_distances(const float* query) const {
        std::vector<float> distances(n_clusters_);
        const auto& ops = kernels::select_backend_auto();
        
        for (std::uint32_t c = 0; c < n_clusters_; ++c) {
            distances[c] = ops.l2_sq(
                std::span(query, dim_),
                std::span(centroids_[c].data(), dim_)
            );
        }
        
        return distances;
    }
};

} // namespace vesper::index