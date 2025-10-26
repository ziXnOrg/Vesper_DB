/** \file hybrid_storage.cpp
 *  \brief Hybrid storage combining PQ codes with 8-bit quantization.
 *
 * Core innovation of CGF: Use PQ for elimination, not approximation.
 * Store both compressed PQ codes (for filtering) and quantized vectors
 * (for accurate distance computation).
 */

#include <cstring>
#include <algorithm>
#include <numeric>
#include <immintrin.h>
#include <unordered_map>
#include <memory>
#include <random>
#include <expected>

#include "vesper/index/cgf.hpp"
#include "vesper/kernels/distance.hpp"
#include "vesper/error.hpp"

namespace vesper::index {

class HybridStorage {
public:
    struct VectorData {
        std::uint64_t id;
        std::uint32_t cluster_id;
        std::vector<std::uint8_t> pq_codes;     // 16 bytes for filtering
        std::vector<std::uint8_t> quantized;    // 128 bytes for accuracy
        std::uint32_t mini_graph_idx;           // Index in mini-HNSW
    };
    
    HybridStorage(std::size_t dim, std::uint32_t m_pq = 16, std::uint32_t nbits = 8)
        : dim_(dim), m_pq_(m_pq), nbits_(nbits) {
        
        // Pre-compute quantization parameters
        subvec_dim_ = dim_ / m_pq_;
        n_centroids_ = 1 << nbits_;
        
        // Initialize quantization scale/offset
        quantization_scale_.resize(dim_);
        quantization_offset_.resize(dim_);
    }
    
    /** Train quantization parameters on data. */
    auto train(const float* data, std::size_t n) -> void {
        // Use Lloyd-Max optimal quantization for better accuracy
        // Instead of simple min-max, use percentile-based robust quantization
        
        for (std::size_t d = 0; d < dim_; ++d) {
            // Collect values for this dimension
            std::vector<float> dim_values;
            dim_values.reserve(n);
            for (std::size_t i = 0; i < n; ++i) {
                dim_values.push_back(data[i * dim_ + d]);
            }
            
            // Sort for percentile computation
            std::sort(dim_values.begin(), dim_values.end());
            
            // Use robust percentile range (1st to 99th percentile)
            // This handles outliers better than min-max
            std::size_t idx_1 = static_cast<std::size_t>(n * 0.01);
            std::size_t idx_99 = static_cast<std::size_t>(n * 0.99);
            
            // Clamp indices to valid range
            idx_1 = std::min(idx_1, n - 1);
            idx_99 = std::min(idx_99, n - 1);
            
            float percentile_1 = dim_values[idx_1];
            float percentile_99 = dim_values[idx_99];
            
            // Lloyd-Max inspired: use percentile range with small margin
            float range = percentile_99 - percentile_1;
            float margin = range * 0.01f;  // 1% margin for robustness
            
            quantization_offset_[d] = percentile_1 - margin;
            
            // Compute scale ensuring no division by zero
            if (range + 2 * margin > 1e-6f) {
                quantization_scale_[d] = 255.0f / (range + 2 * margin);
            } else {
                // Degenerate case: all values nearly identical
                quantization_scale_[d] = 1.0f;
            }
        }
        
        // Train PQ codebooks (simplified k-means)
        train_pq_codebooks(data, n);
    }
    
    /** Add vector to hybrid storage. */
    auto add(std::uint64_t id, const float* vec, std::uint32_t cluster_id) 
        -> std::uint32_t {
        
        VectorData vdata;
        vdata.id = id;
        vdata.cluster_id = cluster_id;
        
        // Encode PQ codes for filtering
        vdata.pq_codes = encode_pq(vec);
        
        // Quantize to 8-bit for accurate distance
        vdata.quantized = quantize_8bit(vec);
        
        // Store
        std::uint32_t idx = static_cast<std::uint32_t>(vectors_.size());
        id_to_idx_[id] = idx;
        vectors_.push_back(std::move(vdata));
        
        return idx;
    }
    
    /** Two-stage distance computation. */
    auto compute_distance(const float* query, std::uint32_t idx,
                         float pq_threshold = 0.0f) const -> float {
        
        const auto& vdata = vectors_[idx];
        
        // Stage 1: PQ distance for filtering
        float pq_dist = compute_pq_distance(query, vdata.pq_codes.data());
        
        // Early exit if PQ distance too large
        if (pq_threshold > 0 && pq_dist > pq_threshold) {
            return std::numeric_limits<float>::max();
        }
        
        // Stage 2: Accurate distance from 8-bit quantized
        return compute_quantized_distance(query, vdata.quantized.data());
    }
    
    /** Batch distance computation with filtering. */
    auto compute_distances_filtered(const float* query,
                                   const std::vector<std::uint32_t>& candidates,
                                   float pq_threshold,
                                   std::uint32_t top_k) const
        -> std::vector<std::pair<std::uint32_t, float>> {
        
        // Stage 1: PQ filtering
        std::vector<std::pair<std::uint32_t, float>> pq_scores;
        pq_scores.reserve(candidates.size());
        
        for (std::uint32_t idx : candidates) {
            float pq_dist = compute_pq_distance(query, vectors_[idx].pq_codes.data());
            if (pq_dist <= pq_threshold) {
                pq_scores.emplace_back(idx, pq_dist);
            }
        }
        
        // Take top candidates from PQ
        std::uint32_t n_refine = std::min<std::uint32_t>(
            static_cast<std::uint32_t>(pq_scores.size()),
            top_k * 5  // Refine 5x the final k
        );
        
        if (pq_scores.size() > n_refine) {
            std::partial_sort(pq_scores.begin(), 
                            pq_scores.begin() + n_refine,
                            pq_scores.end(),
                            [](const auto& a, const auto& b) {
                                return a.second < b.second;
                            });
            pq_scores.resize(n_refine);
        }
        
        // Stage 2: Accurate reranking
        std::vector<std::pair<std::uint32_t, float>> results;
        results.reserve(pq_scores.size());
        
        for (const auto& [idx, _] : pq_scores) {
            float exact_dist = compute_quantized_distance(
                query, vectors_[idx].quantized.data()
            );
            results.emplace_back(idx, exact_dist);
        }
        
        // Final sort
        std::sort(results.begin(), results.end(),
                 [](const auto& a, const auto& b) {
                     return a.second < b.second;
                 });
        
        if (results.size() > top_k) {
            results.resize(top_k);
        }
        
        return results;
    }
    
    /** Get vector by ID. */
    auto get_vector(std::uint64_t id) const 
        -> std::expected<const VectorData*, core::error> {
        
        auto it = id_to_idx_.find(id);
        if (it == id_to_idx_.end()) {
            return std::vesper_unexpected(core::error{
                core::error_code::not_found,
                "Vector not found",
                "hybrid_storage"
            });
        }
        return &vectors_[it->second];
    }
    
    /** Reconstruct original vector from quantized representation. */
    auto reconstruct(std::uint32_t idx) const -> std::vector<float> {
        const auto& vdata = vectors_[idx];
        std::vector<float> vec(dim_);
        
        // Dequantize 8-bit representation
        for (std::size_t d = 0; d < dim_; ++d) {
            vec[d] = vdata.quantized[d] / quantization_scale_[d] + 
                    quantization_offset_[d];
        }
        
        return vec;
    }
    
    /** Memory usage statistics. */
    struct MemoryStats {
        std::size_t pq_bytes;
        std::size_t quantized_bytes;
        std::size_t metadata_bytes;
        std::size_t total_bytes;
        float bytes_per_vector;
    };
    
    auto get_memory_stats() const -> MemoryStats {
        MemoryStats stats{};
        
        for (const auto& v : vectors_) {
            stats.pq_bytes += v.pq_codes.size();
            stats.quantized_bytes += v.quantized.size();
            stats.metadata_bytes += sizeof(VectorData);
        }
        
        stats.total_bytes = stats.pq_bytes + stats.quantized_bytes + 
                          stats.metadata_bytes;
        
        if (!vectors_.empty()) {
            stats.bytes_per_vector = static_cast<float>(stats.total_bytes) / 
                                   vectors_.size();
        }
        
        return stats;
    }

private:
    std::size_t dim_;
    std::uint32_t m_pq_;
    std::uint32_t nbits_;
    std::uint32_t subvec_dim_;
    std::uint32_t n_centroids_;
    
    std::vector<VectorData> vectors_;
    std::unordered_map<std::uint64_t, std::uint32_t> id_to_idx_;
    
    // Quantization parameters
    std::vector<float> quantization_scale_;
    std::vector<float> quantization_offset_;
    
    // PQ codebooks
    std::vector<std::vector<float>> pq_codebooks_;  // [m_pq][n_centroids][subvec_dim]
    
    /** Train PQ codebooks using k-means. */
    void train_pq_codebooks(const float* data, std::size_t n) {
        pq_codebooks_.resize(m_pq_);
        
        for (std::uint32_t m = 0; m < m_pq_; ++m) {
            // Extract subvectors for this dimension
            std::vector<std::vector<float>> subvecs(n, std::vector<float>(subvec_dim_));
            
            for (std::size_t i = 0; i < n; ++i) {
                const float* vec = data + i * dim_;
                for (std::uint32_t d = 0; d < subvec_dim_; ++d) {
                    subvecs[i][d] = vec[m * subvec_dim_ + d];
                }
            }
            
            // Run k-means to get centroids
            pq_codebooks_[m] = run_kmeans(subvecs, n_centroids_);
        }
    }
    
    /** Simple k-means implementation. */
    std::vector<float> run_kmeans(const std::vector<std::vector<float>>& points,
                                 std::uint32_t k) {
        const std::size_t n = points.size();
        const std::size_t dim = points[0].size();
        
        // Initialize centroids randomly
        std::vector<float> centroids(k * dim);
        std::mt19937 gen(42);
        std::uniform_int_distribution<std::size_t> dist(0, n - 1);
        
        for (std::uint32_t c = 0; c < k; ++c) {
            std::size_t idx = dist(gen);
            std::copy(points[idx].begin(), points[idx].end(),
                     centroids.begin() + c * dim);
        }
        
        // Lloyd's iterations (simplified)
        const std::uint32_t max_iters = 10;
        std::vector<std::uint32_t> assignments(n);
        
        for (std::uint32_t iter = 0; iter < max_iters; ++iter) {
            // Assign points
            for (std::size_t i = 0; i < n; ++i) {
                float min_dist = std::numeric_limits<float>::max();
                std::uint32_t best_c = 0;
                
                for (std::uint32_t c = 0; c < k; ++c) {
                    float dist = 0.0f;
                    for (std::size_t d = 0; d < dim; ++d) {
                        float diff = points[i][d] - centroids[c * dim + d];
                        dist += diff * diff;
                    }
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_c = c;
                    }
                }
                assignments[i] = best_c;
            }
            
            // Update centroids
            std::fill(centroids.begin(), centroids.end(), 0.0f);
            std::vector<std::uint32_t> counts(k, 0);
            
            for (std::size_t i = 0; i < n; ++i) {
                std::uint32_t c = assignments[i];
                for (std::size_t d = 0; d < dim; ++d) {
                    centroids[c * dim + d] += points[i][d];
                }
                counts[c]++;
            }
            
            for (std::uint32_t c = 0; c < k; ++c) {
                if (counts[c] > 0) {
                    for (std::size_t d = 0; d < dim; ++d) {
                        centroids[c * dim + d] /= counts[c];
                    }
                }
            }
        }
        
        return centroids;
    }
    
    /** Encode vector to PQ codes. */
    std::vector<std::uint8_t> encode_pq(const float* vec) const {
        std::vector<std::uint8_t> codes(m_pq_);
        
        for (std::uint32_t m = 0; m < m_pq_; ++m) {
            const float* subvec = vec + m * subvec_dim_;
            
            // Find nearest centroid
            float min_dist = std::numeric_limits<float>::max();
            std::uint8_t best_c = 0;
            
            for (std::uint32_t c = 0; c < n_centroids_; ++c) {
                float dist = 0.0f;
                const float* centroid = pq_codebooks_[m].data() + c * subvec_dim_;
                
                for (std::uint32_t d = 0; d < subvec_dim_; ++d) {
                    float diff = subvec[d] - centroid[d];
                    dist += diff * diff;
                }
                
                if (dist < min_dist) {
                    min_dist = dist;
                    best_c = static_cast<std::uint8_t>(c);
                }
            }
            
            codes[m] = best_c;
        }
        
        return codes;
    }
    
    /** Quantize vector to 8-bit. */
    std::vector<std::uint8_t> quantize_8bit(const float* vec) const {
        std::vector<std::uint8_t> quantized(dim_);
        
        for (std::size_t d = 0; d < dim_; ++d) {
            float normalized = (vec[d] - quantization_offset_[d]) * 
                             quantization_scale_[d];
            normalized = std::max(0.0f, std::min(255.0f, normalized));
            quantized[d] = static_cast<std::uint8_t>(normalized + 0.5f);
        }
        
        return quantized;
    }
    
    /** Precompute PQ lookup table for query (Asymmetric Distance Computation). */
    void compute_pq_lut(const float* query, float* lut) const {
        // Compute distance from query to each centroid in each subspace
        // This is done once per query, then we just lookup for each code
        
        #pragma omp parallel for if(m_pq_ > 8)
        for (std::uint32_t m = 0; m < m_pq_; ++m) {
            const float* q_subvec = query + m * subvec_dim_;
            float* lut_row = lut + m * n_centroids_;
            
            // Compute distances to all centroids in this subspace
            for (std::uint32_t c = 0; c < n_centroids_; ++c) {
                const float* centroid = pq_codebooks_[m].data() + c * subvec_dim_;
                
                #ifdef __AVX2__
                // AVX2 optimized distance computation
                __m256 sum = _mm256_setzero_ps();
                std::size_t d = 0;
                
                // Process 8 dimensions at a time
                for (; d + 7 < subvec_dim_; d += 8) {
                    __m256 q = _mm256_loadu_ps(q_subvec + d);
                    __m256 cent = _mm256_loadu_ps(centroid + d);
                    __m256 diff = _mm256_sub_ps(q, cent);
                    sum = _mm256_fmadd_ps(diff, diff, sum);
                }
                
                // Horizontal sum without slow hadd
                __m128 sum_high = _mm256_extractf128_ps(sum, 1);
                __m128 sum_low = _mm256_castps256_ps128(sum);
                __m128 sum_128 = _mm_add_ps(sum_high, sum_low);
                sum_128 = _mm_add_ps(sum_128, _mm_shuffle_ps(sum_128, sum_128, _MM_SHUFFLE(2,3,0,1)));
                sum_128 = _mm_add_ps(sum_128, _mm_shuffle_ps(sum_128, sum_128, _MM_SHUFFLE(1,0,3,2)));
                float dist = _mm_cvtss_f32(sum_128);
                
                // Handle remainder
                for (; d < subvec_dim_; ++d) {
                    float diff = q_subvec[d] - centroid[d];
                    dist += diff * diff;
                }
                
                lut_row[c] = dist;
                #else
                // Scalar fallback
                float dist = 0.0f;
                for (std::uint32_t d = 0; d < subvec_dim_; ++d) {
                    float diff = q_subvec[d] - centroid[d];
                    dist += diff * diff;
                }
                lut_row[c] = dist;
                #endif
            }
        }
    }
    
    /** Compute PQ distance using precomputed lookup table. */
    float compute_pq_distance_lut(const float* lut, const std::uint8_t* codes) const {
        float dist = 0.0f;
        
        // Simple lookup - O(m) instead of O(m*d)
        #pragma unroll
        for (std::uint32_t m = 0; m < m_pq_; ++m) {
            dist += lut[m * n_centroids_ + codes[m]];
        }
        
        return dist;
    }
    
    /** Compute PQ distance (legacy - for compatibility). */
    float compute_pq_distance(const float* query, const std::uint8_t* codes) const {
        // Allocate aligned LUT on stack for small m_pq
        if (m_pq_ <= 32) {
            alignas(64) float lut[32 * 256];
            compute_pq_lut(query, lut);
            return compute_pq_distance_lut(lut, codes);
        } else {
            // Heap allocation for large m_pq
            std::vector<float> lut(m_pq_ * n_centroids_);
            compute_pq_lut(query, lut.data());
            return compute_pq_distance_lut(lut.data(), codes);
        }
    }
    
    /** Compute distance from 8-bit quantized vector. */
    float compute_quantized_distance(const float* query, 
                                    const std::uint8_t* quantized) const {
        float dist = 0.0f;
        
        #ifdef __AVX2__
        // AVX2 optimized path
        const std::size_t simd_width = 8;
        const std::size_t simd_iters = dim_ / simd_width;
        
        __m256 sum_vec = _mm256_setzero_ps();
        
        for (std::size_t i = 0; i < simd_iters; ++i) {
            // Load and dequantize 8 bytes
            __m128i quant_bytes = _mm_loadl_epi64(
                reinterpret_cast<const __m128i*>(quantized + i * simd_width)
            );
            __m256i quant_int = _mm256_cvtepu8_epi32(quant_bytes);
            __m256 quant_float = _mm256_cvtepi32_ps(quant_int);
            
            // Load scale and offset
            __m256 scale = _mm256_loadu_ps(quantization_scale_.data() + i * simd_width);
            __m256 offset = _mm256_loadu_ps(quantization_offset_.data() + i * simd_width);
            
            // Dequantize: val = quant / scale + offset
            __m256 dequant = _mm256_div_ps(quant_float, scale);
            dequant = _mm256_add_ps(dequant, offset);
            
            // Load query and compute difference
            __m256 q = _mm256_loadu_ps(query + i * simd_width);
            __m256 diff = _mm256_sub_ps(q, dequant);
            
            // Accumulate squared difference
            sum_vec = _mm256_fmadd_ps(diff, diff, sum_vec);
        }
        
        // Horizontal sum
        __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
        __m128 sum_low = _mm256_castps256_ps128(sum_vec);
        __m128 sum_128 = _mm_add_ps(sum_high, sum_low);
        sum_128 = _mm_hadd_ps(sum_128, sum_128);
        sum_128 = _mm_hadd_ps(sum_128, sum_128);
        dist = _mm_cvtss_f32(sum_128);
        
        // Handle remainder
        for (std::size_t d = simd_iters * simd_width; d < dim_; ++d) {
            float val = quantized[d] / quantization_scale_[d] + 
                       quantization_offset_[d];
            float diff = query[d] - val;
            dist += diff * diff;
        }
        #else
        // Scalar fallback
        for (std::size_t d = 0; d < dim_; ++d) {
            float val = quantized[d] / quantization_scale_[d] + 
                       quantization_offset_[d];
            float diff = query[d] - val;
            dist += diff * diff;
        }
        #endif
        
        return dist;
    }
};

} // namespace vesper::index