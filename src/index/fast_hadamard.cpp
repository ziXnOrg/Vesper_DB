/** \file fast_hadamard.cpp
 *  \brief Implementation of Fast Walsh-Hadamard Transform
 */

#include "vesper/index/fast_hadamard.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cassert>

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64)
#include <immintrin.h>
#endif

namespace vesper::index {

namespace {
    // Round up to next power of 2
    constexpr std::uint32_t next_pow2(std::uint32_t n) {
        if (n == 0) return 1;
        n--;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        return n + 1;
    }
    
    // Count trailing zeros (log2 of power of 2)
    constexpr std::uint32_t log2_pow2(std::uint32_t n) {
        std::uint32_t count = 0;
        while ((n & 1) == 0) {
            n >>= 1;
            count++;
        }
        return count;
    }
}

FastHadamard::FastHadamard(std::uint32_t dim, std::uint32_t seed)
    : dim_(dim)
    , padded_dim_(next_pow2(dim))
    , log2_dim_(log2_pow2(padded_dim_))
    , workspace_(padded_dim_) {
    
    // Generate random sign flips
    signs_.resize(padded_dim_);
    std::mt19937 rng(seed);
    std::bernoulli_distribution dist(0.5);
    
    for (auto& sign : signs_) {
        sign = dist(rng) ? 1.0f : -1.0f;
    }
}

void FastHadamard::rotate_inplace(std::span<float> vec) const {
    assert(vec.size() >= dim_);
    
    // Copy to workspace and pad with zeros
    std::copy_n(vec.data(), dim_, workspace_.data());
    std::fill(workspace_.begin() + dim_, workspace_.end(), 0.0f);
    
    // Apply random sign flips (D matrix)
    for (std::uint32_t i = 0; i < padded_dim_; ++i) {
        workspace_[i] *= signs_[i];
    }
    
    // Apply Fast Walsh-Hadamard Transform
#if defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64)
    fwht_avx2(workspace_.data());
#else
    fwht_scalar(workspace_.data());
#endif
    
    // Normalize by sqrt(n) to maintain orthogonality
    const float scale = 1.0f / std::sqrt(static_cast<float>(padded_dim_));
    for (std::uint32_t i = 0; i < dim_; ++i) {
        vec[i] = workspace_[i] * scale;
    }
}

void FastHadamard::rotate(std::span<const float> input, std::span<float> output) const {
    assert(input.size() >= dim_ && output.size() >= dim_);
    
    // Copy input to output then rotate in-place
    std::copy_n(input.data(), dim_, output.data());
    rotate_inplace(output);
}

void FastHadamard::rotate_batch(float* vecs, std::size_t n, std::size_t stride) const {
    if (stride == 0) stride = dim_;
    
    for (std::size_t i = 0; i < n; ++i) {
        rotate_inplace(std::span<float>(vecs + i * stride, dim_));
    }
}

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64)
void FastHadamard::fwht_avx2(float* data) const {
    // Fast Walsh-Hadamard Transform with AVX2
    // Butterfly operations with increasing stride
    
    for (std::uint32_t len = 1; len < padded_dim_; len <<= 1) {
        const std::uint32_t len2 = len << 1;
        
        // Process 8 elements at a time with AVX2
        for (std::uint32_t i = 0; i < padded_dim_; i += len2) {
            std::uint32_t j = 0;
            
            // AVX2 loop for blocks of 8
            for (; j + 7 < len; j += 8) {
                __m256 a = _mm256_loadu_ps(&data[i + j]);
                __m256 b = _mm256_loadu_ps(&data[i + j + len]);
                
                __m256 sum = _mm256_add_ps(a, b);
                __m256 diff = _mm256_sub_ps(a, b);
                
                _mm256_storeu_ps(&data[i + j], sum);
                _mm256_storeu_ps(&data[i + j + len], diff);
            }
            
            // Handle remaining elements
            for (; j < len; ++j) {
                float a = data[i + j];
                float b = data[i + j + len];
                data[i + j] = a + b;
                data[i + j + len] = a - b;
            }
        }
    }
}
#endif

void FastHadamard::fwht_scalar(float* data) const {
    // Scalar fallback for Fast Walsh-Hadamard Transform
    for (std::uint32_t len = 1; len < padded_dim_; len <<= 1) {
        for (std::uint32_t i = 0; i < padded_dim_; i += len << 1) {
            for (std::uint32_t j = 0; j < len; ++j) {
                float a = data[i + j];
                float b = data[i + j + len];
                data[i + j] = a + b;
                data[i + j + len] = a - b;
            }
        }
    }
}

// FastRotationalQuantizer implementation

void FastRotationalQuantizer::train(const float* data, std::size_t n, std::size_t dim,
                                   const TrainParams& params) {
    dim_ = static_cast<std::uint32_t>(dim);
    padded_dim_ = next_pow2(dim_);
    workspace_.resize(padded_dim_);
    
    // Initialize multiple rotation blocks
    rotations_.clear();
    rotations_.reserve(params.num_rotations);
    for (std::uint32_t i = 0; i < params.num_rotations; ++i) {
        rotations_.emplace_back(dim_, params.seed + i * 1337);
    }
    
    // Compute per-dimension statistics for quantization
    scales_.resize(dim_);
    offsets_.resize(dim_);
    
    std::vector<float> rotated(dim_ * n);
    
    // Apply rotations to training data
    for (std::size_t i = 0; i < n; ++i) {
        std::copy_n(data + i * dim_, dim_, workspace_.data());
        
        // Apply all rotation blocks
        for (const auto& rotation : rotations_) {
            rotation.rotate_inplace(workspace_);
        }
        
        std::copy_n(workspace_.data(), dim_, rotated.data() + i * dim_);
    }
    
    // Compute per-dimension min/max for scalar quantization
    for (std::uint32_t d = 0; d < dim_; ++d) {
        float min_val = std::numeric_limits<float>::max();
        float max_val = std::numeric_limits<float>::lowest();
        
        for (std::size_t i = 0; i < n; ++i) {
            float val = rotated[i * dim_ + d];
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
        }
        
        // Avoid division by zero
        float range = max_val - min_val;
        if (range < 1e-6f) range = 1.0f;
        
        offsets_[d] = min_val;
        scales_[d] = 255.0f / range;
    }
}

std::pair<float, float> FastRotationalQuantizer::quantize(std::span<const float> vec,
                                                         std::uint8_t* output) const {
    assert(vec.size() >= dim_);
    
    // Apply rotations
    std::copy_n(vec.data(), dim_, workspace_.data());
    for (const auto& rotation : rotations_) {
        rotation.rotate_inplace(workspace_);
    }
    
    // Quantize to 8-bit with per-dimension scaling
    float global_scale = 0.0f;
    float global_offset = 0.0f;
    
    for (std::uint32_t i = 0; i < dim_; ++i) {
        float val = workspace_[i];
        float normalized = (val - offsets_[i]) * scales_[i];
        normalized = std::clamp(normalized, 0.0f, 255.0f);
        output[i] = static_cast<std::uint8_t>(normalized + 0.5f);
        
        // Track average scale/offset for distance estimation
        global_scale += scales_[i];
        global_offset += offsets_[i];
    }
    
    global_scale /= dim_;
    global_offset /= dim_;
    
    return {global_scale, global_offset};
}

void FastRotationalQuantizer::quantize_batch(const float* vecs, std::size_t n,
                                            std::uint8_t* output,
                                            float* scales, float* offsets) const {
    for (std::size_t i = 0; i < n; ++i) {
        auto [scale, offset] = quantize(
            std::span<const float>(vecs + i * dim_, dim_),
            output + i * dim_
        );
        if (scales) scales[i] = scale;
        if (offsets) offsets[i] = offset;
    }
}

float FastRotationalQuantizer::estimate_l2_distance(const std::uint8_t* q_codes,
                                                   const std::uint8_t* db_codes,
                                                   float q_scale, float q_offset,
                                                   float db_scale, float db_offset) const {
    // Fast distance estimation using quantized codes
    float sum = 0.0f;
    
#if defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64)
    // AVX2 optimized path
    __m256 sum_vec = _mm256_setzero_ps();
    
    std::uint32_t i = 0;
    for (; i + 31 < dim_; i += 32) {
        // Load 32 bytes and convert to int
        __m256i q = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q_codes + i));
        __m256i d = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(db_codes + i));
        
        // Compute differences (need to unpack to 16-bit first)
        __m256i q_lo = _mm256_unpacklo_epi8(q, _mm256_setzero_si256());
        __m256i q_hi = _mm256_unpackhi_epi8(q, _mm256_setzero_si256());
        __m256i d_lo = _mm256_unpacklo_epi8(d, _mm256_setzero_si256());
        __m256i d_hi = _mm256_unpackhi_epi8(d, _mm256_setzero_si256());
        
        __m256i diff_lo = _mm256_sub_epi16(q_lo, d_lo);
        __m256i diff_hi = _mm256_sub_epi16(q_hi, d_hi);
        
        // Square the differences (need to further unpack to 32-bit)
        __m256i diff_lo_lo = _mm256_unpacklo_epi16(diff_lo, diff_lo);
        __m256i diff_lo_hi = _mm256_unpackhi_epi16(diff_lo, diff_lo);
        __m256i diff_hi_lo = _mm256_unpacklo_epi16(diff_hi, diff_hi);
        __m256i diff_hi_hi = _mm256_unpackhi_epi16(diff_hi, diff_hi);
        
        // Multiply (squared) and accumulate
        __m256i sq_lo_lo = _mm256_madd_epi16(diff_lo_lo, _mm256_set1_epi16(1));
        __m256i sq_lo_hi = _mm256_madd_epi16(diff_lo_hi, _mm256_set1_epi16(1));
        __m256i sq_hi_lo = _mm256_madd_epi16(diff_hi_lo, _mm256_set1_epi16(1));
        __m256i sq_hi_hi = _mm256_madd_epi16(diff_hi_hi, _mm256_set1_epi16(1));
        
        // Convert to float and accumulate
        sum_vec = _mm256_add_ps(sum_vec, _mm256_cvtepi32_ps(sq_lo_lo));
        sum_vec = _mm256_add_ps(sum_vec, _mm256_cvtepi32_ps(sq_lo_hi));
        sum_vec = _mm256_add_ps(sum_vec, _mm256_cvtepi32_ps(sq_hi_lo));
        sum_vec = _mm256_add_ps(sum_vec, _mm256_cvtepi32_ps(sq_hi_hi));
    }
    
    // Horizontal sum
    __m128 hi = _mm256_extractf128_ps(sum_vec, 1);
    __m128 lo = _mm256_castps256_ps128(sum_vec);
    __m128 sum128 = _mm_add_ps(hi, lo);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum += _mm_cvtss_f32(sum128);
    
    // Handle remaining elements
    for (; i < dim_; ++i) {
        float diff = static_cast<float>(q_codes[i]) - static_cast<float>(db_codes[i]);
        sum += diff * diff;
    }
#else
    // Scalar fallback
    for (std::uint32_t i = 0; i < dim_; ++i) {
        float diff = static_cast<float>(q_codes[i]) - static_cast<float>(db_codes[i]);
        sum += diff * diff;
    }
#endif
    
    // Apply inverse scaling
    float avg_scale = (q_scale + db_scale) * 0.5f;
    if (avg_scale > 0) {
        sum /= (avg_scale * avg_scale);
    }
    
    return sum;
}

void FastRotationalQuantizer::estimate_distances_batch(const std::uint8_t* q_codes,
                                                      const std::uint8_t* db_codes,
                                                      std::size_t n,
                                                      float q_scale, float q_offset,
                                                      const float* db_scales,
                                                      const float* db_offsets,
                                                      float* distances) const {
    // Process multiple database vectors
    for (std::size_t i = 0; i < n; ++i) {
        distances[i] = estimate_l2_distance(
            q_codes,
            db_codes + i * dim_,
            q_scale, q_offset,
            db_scales ? db_scales[i] : q_scale,
            db_offsets ? db_offsets[i] : q_offset
        );
    }
}

} // namespace vesper::index