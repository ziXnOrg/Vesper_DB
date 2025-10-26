#pragma once

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64)

/** \file avx2.hpp
 *  \brief Production AVX2 SIMD kernels for distance computation.
 *
 * Implements L2 squared, inner product, and cosine similarity using AVX2 intrinsics.
 * Features:
 * - 8-wide float operations with FMA instructions
 * - Cache-line aware processing with prefetch hints
 * - Optimized tail handling for non-aligned dimensions
 * - Zero-overhead abstractions with force-inline
 *
 * Preconditions: a.size() == b.size() > 0; inputs finite; for cosine, norms > 0.
 * Complexity: O(d) with SIMD vectorization factor of 8
 * Thread-safety: Pure functions, no shared state
 */

#include <immintrin.h>
#include <vesper/span_polyfill.hpp>
#include <cmath>
#include "vesper/kernels/dispatch.hpp"

namespace vesper::kernels {

namespace detail {

/** \brief Horizontal sum of 8 floats in AVX2 register.
 *  Uses efficient shuffle+hadd pattern to minimize latency.
 */
#ifdef _MSC_VER
__forceinline
#else
[[gnu::always_inline]]
#endif
inline auto hsum_ps(__m256 v) noexcept -> float {
    const __m128 hi = _mm256_extractf128_ps(v, 1);
    const __m128 lo = _mm256_castps256_ps128(v);
    const __m128 sum = _mm_add_ps(hi, lo);
    const __m128 shuf = _mm_movehdup_ps(sum);
    const __m128 sums = _mm_add_ps(sum, shuf);
    const __m128 shuf2 = _mm_movehl_ps(sums, sums);
    const __m128 result = _mm_add_ss(sums, shuf2);
    return _mm_cvtss_f32(result);
}

/** \brief Software prefetch for next cache line.
 *  Hints L1 cache to load next iteration's data.
 */
#ifdef _MSC_VER
__forceinline
#else
[[gnu::always_inline]]
#endif
inline void prefetch_l1(const float* ptr) noexcept {
    _mm_prefetch(reinterpret_cast<const char*>(ptr + 16), _MM_HINT_T0);
}

} // namespace detail

/** \brief AVX2-optimized L2 squared distance.
 *  
 * Computes ||a - b||² using FMA instructions for optimal throughput.
 * Processes 8 elements per iteration with tail handling.
 *
 * \param a First vector
 * \param b Second vector  
 * \return L2 squared distance
 */
#ifdef _MSC_VER
[[maybe_unused]]
#else
[[gnu::hot]]
#endif
inline auto avx2_l2_sq(std::span<const float> a, std::span<const float> b) noexcept -> float {
    const std::size_t n = a.size();
    const float* pa = a.data();
    const float* pb = b.data();
    
    __m256 sum = _mm256_setzero_ps();
    std::size_t i = 0;
    
    // Main loop: process 8 elements at a time
    const std::size_t simd_end = n & ~7u;
    for (; i < simd_end; i += 8) {
        // Prefetch next cache line
        if (i + 16 < n) {
            detail::prefetch_l1(pa + i + 16);
            detail::prefetch_l1(pb + i + 16);
        }
        
        const __m256 va = _mm256_loadu_ps(pa + i);
        const __m256 vb = _mm256_loadu_ps(pb + i);
        const __m256 diff = _mm256_sub_ps(va, vb);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }
    
    float result = detail::hsum_ps(sum);
    
    // Tail: process remaining elements
    for (; i < n; ++i) {
        const float diff = pa[i] - pb[i];
        result += diff * diff;
    }
    
    return result;
}

/** \brief AVX2-optimized inner product.
 *
 * Computes a·b using FMA instructions for maximum throughput.
 * Achieves near-peak FLOPS on modern CPUs.
 *
 * \param a First vector
 * \param b Second vector
 * \return Inner product
 */
#ifdef _MSC_VER
[[maybe_unused]]
#else
[[gnu::hot]]
#endif
inline auto avx2_inner_product(std::span<const float> a, std::span<const float> b) noexcept -> float {
    const std::size_t n = a.size();
    const float* pa = a.data();
    const float* pb = b.data();
    
    __m256 sum = _mm256_setzero_ps();
    std::size_t i = 0;
    
    // Main loop: process 8 elements at a time
    const std::size_t simd_end = n & ~7u;
    for (; i < simd_end; i += 8) {
        // Prefetch next cache line
        if (i + 16 < n) {
            detail::prefetch_l1(pa + i + 16);
            detail::prefetch_l1(pb + i + 16);
        }
        
        const __m256 va = _mm256_loadu_ps(pa + i);
        const __m256 vb = _mm256_loadu_ps(pb + i);
        sum = _mm256_fmadd_ps(va, vb, sum);
    }
    
    float result = detail::hsum_ps(sum);
    
    // Tail: process remaining elements
    for (; i < n; ++i) {
        result += pa[i] * pb[i];
    }
    
    return result;
}

/** \brief AVX2-optimized cosine similarity.
 *
 * Computes (a·b) / (||a|| * ||b||) with single-pass norm computation.
 * Uses FMA for all dot products and norms.
 *
 * \param a First vector
 * \param b Second vector
 * \return Cosine similarity in [0, 1]
 */
#ifdef _MSC_VER
[[maybe_unused]]
#else
[[gnu::hot]]
#endif
inline auto avx2_cosine_similarity(std::span<const float> a, std::span<const float> b) noexcept -> float {
    const std::size_t n = a.size();
    const float* pa = a.data();
    const float* pb = b.data();
    
    __m256 dot_sum = _mm256_setzero_ps();
    __m256 norm_a_sum = _mm256_setzero_ps();
    __m256 norm_b_sum = _mm256_setzero_ps();
    std::size_t i = 0;
    
    // Main loop: compute dot product and norms in single pass
    const std::size_t simd_end = n & ~7u;
    for (; i < simd_end; i += 8) {
        // Prefetch next cache line
        if (i + 16 < n) {
            detail::prefetch_l1(pa + i + 16);
            detail::prefetch_l1(pb + i + 16);
        }
        
        const __m256 va = _mm256_loadu_ps(pa + i);
        const __m256 vb = _mm256_loadu_ps(pb + i);
        
        dot_sum = _mm256_fmadd_ps(va, vb, dot_sum);
        norm_a_sum = _mm256_fmadd_ps(va, va, norm_a_sum);
        norm_b_sum = _mm256_fmadd_ps(vb, vb, norm_b_sum);
    }
    
    float dot = detail::hsum_ps(dot_sum);
    float norm_a_sq = detail::hsum_ps(norm_a_sum);
    float norm_b_sq = detail::hsum_ps(norm_b_sum);
    
    // Tail: process remaining elements
    for (; i < n; ++i) {
        const float va = pa[i];
        const float vb = pb[i];
        dot += va * vb;
        norm_a_sq += va * va;
        norm_b_sq += vb * vb;
    }
    
    // Compute final similarity with fast reciprocal square root
    const float norm_prod = std::sqrt(norm_a_sq * norm_b_sq);
    return (norm_prod > 0.0f) ? (dot / norm_prod) : 0.0f;
}

/** \brief AVX2-optimized cosine distance.
 *
 * Computes 1.0 - cosine_similarity for use as distance metric.
 *
 * \param a First vector
 * \param b Second vector
 * \return Cosine distance in [0, 2]
 */
#ifdef _MSC_VER
[[maybe_unused]]
#else
[[gnu::hot]]
#endif
inline auto avx2_cosine_distance(std::span<const float> a, std::span<const float> b) noexcept -> float {
    return 1.0f - avx2_cosine_similarity(a, b);
}

/** \brief AVX2-optimized batch L2 squared distance.
 *
 * Computes L2 squared distances from query to multiple vectors.
 * Optimized for cache locality and SIMD throughput.
 *
 * \param query Query vector
 * \param vectors Packed vectors (row-major layout)
 * \param nvec Number of vectors
 * \param dim Dimension
 * \param distances Output distances array
 */
#ifdef _MSC_VER
[[maybe_unused]]
#else
[[gnu::hot]]
#endif
inline void avx2_batch_l2_sq(std::span<const float> query,
                             const float* vectors, size_t nvec, size_t dim,
                             float* distances) noexcept {
    const float* q = query.data();
    const size_t simd_dim = dim & ~7u;
    
    // Process 4 vectors at a time for better cache utilization
    size_t v = 0;
    for (; v + 3 < nvec; v += 4) {
        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
        __m256 sum2 = _mm256_setzero_ps();
        __m256 sum3 = _mm256_setzero_ps();
        
        // Main SIMD loop
        for (size_t d = 0; d < simd_dim; d += 8) {
            const __m256 vq = _mm256_loadu_ps(q + d);
            
            const __m256 v0 = _mm256_loadu_ps(vectors + (v + 0) * dim + d);
            const __m256 v1 = _mm256_loadu_ps(vectors + (v + 1) * dim + d);
            const __m256 v2 = _mm256_loadu_ps(vectors + (v + 2) * dim + d);
            const __m256 v3 = _mm256_loadu_ps(vectors + (v + 3) * dim + d);
            
            const __m256 diff0 = _mm256_sub_ps(vq, v0);
            const __m256 diff1 = _mm256_sub_ps(vq, v1);
            const __m256 diff2 = _mm256_sub_ps(vq, v2);
            const __m256 diff3 = _mm256_sub_ps(vq, v3);
            
            sum0 = _mm256_fmadd_ps(diff0, diff0, sum0);
            sum1 = _mm256_fmadd_ps(diff1, diff1, sum1);
            sum2 = _mm256_fmadd_ps(diff2, diff2, sum2);
            sum3 = _mm256_fmadd_ps(diff3, diff3, sum3);
        }
        
        distances[v + 0] = detail::hsum_ps(sum0);
        distances[v + 1] = detail::hsum_ps(sum1);
        distances[v + 2] = detail::hsum_ps(sum2);
        distances[v + 3] = detail::hsum_ps(sum3);
        
        // Handle tail dimensions
        for (size_t d = simd_dim; d < dim; ++d) {
            float qd = q[d];
            distances[v + 0] += (qd - vectors[(v + 0) * dim + d]) * (qd - vectors[(v + 0) * dim + d]);
            distances[v + 1] += (qd - vectors[(v + 1) * dim + d]) * (qd - vectors[(v + 1) * dim + d]);
            distances[v + 2] += (qd - vectors[(v + 2) * dim + d]) * (qd - vectors[(v + 2) * dim + d]);
            distances[v + 3] += (qd - vectors[(v + 3) * dim + d]) * (qd - vectors[(v + 3) * dim + d]);
        }
    }
    
    // Process remaining vectors
    for (; v < nvec; ++v) {
        __m256 sum = _mm256_setzero_ps();
        for (size_t d = 0; d < simd_dim; d += 8) {
            const __m256 vq = _mm256_loadu_ps(q + d);
            const __m256 vv = _mm256_loadu_ps(vectors + v * dim + d);
            const __m256 diff = _mm256_sub_ps(vq, vv);
            sum = _mm256_fmadd_ps(diff, diff, sum);
        }
        distances[v] = detail::hsum_ps(sum);
        
        for (size_t d = simd_dim; d < dim; ++d) {
            float diff = q[d] - vectors[v * dim + d];
            distances[v] += diff * diff;
        }
    }
}

/** \brief AVX2-optimized batch inner product.
 *
 * Computes inner products from query to multiple vectors.
 *
 * \param query Query vector
 * \param vectors Packed vectors (row-major layout)
 * \param nvec Number of vectors
 * \param dim Dimension
 * \param distances Output inner products array
 */
#ifdef _MSC_VER
[[maybe_unused]]
#else
[[gnu::hot]]
#endif
inline void avx2_batch_inner_product(std::span<const float> query,
                                     const float* vectors, size_t nvec, size_t dim,
                                     float* distances) noexcept {
    const float* q = query.data();
    const size_t simd_dim = dim & ~7u;
    
    // Process 4 vectors at a time
    size_t v = 0;
    for (; v + 3 < nvec; v += 4) {
        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
        __m256 sum2 = _mm256_setzero_ps();
        __m256 sum3 = _mm256_setzero_ps();
        
        for (size_t d = 0; d < simd_dim; d += 8) {
            const __m256 vq = _mm256_loadu_ps(q + d);
            
            const __m256 v0 = _mm256_loadu_ps(vectors + (v + 0) * dim + d);
            const __m256 v1 = _mm256_loadu_ps(vectors + (v + 1) * dim + d);
            const __m256 v2 = _mm256_loadu_ps(vectors + (v + 2) * dim + d);
            const __m256 v3 = _mm256_loadu_ps(vectors + (v + 3) * dim + d);
            
            sum0 = _mm256_fmadd_ps(vq, v0, sum0);
            sum1 = _mm256_fmadd_ps(vq, v1, sum1);
            sum2 = _mm256_fmadd_ps(vq, v2, sum2);
            sum3 = _mm256_fmadd_ps(vq, v3, sum3);
        }
        
        distances[v + 0] = detail::hsum_ps(sum0);
        distances[v + 1] = detail::hsum_ps(sum1);
        distances[v + 2] = detail::hsum_ps(sum2);
        distances[v + 3] = detail::hsum_ps(sum3);
        
        // Handle tail dimensions
        for (size_t d = simd_dim; d < dim; ++d) {
            float qd = q[d];
            distances[v + 0] += qd * vectors[(v + 0) * dim + d];
            distances[v + 1] += qd * vectors[(v + 1) * dim + d];
            distances[v + 2] += qd * vectors[(v + 2) * dim + d];
            distances[v + 3] += qd * vectors[(v + 3) * dim + d];
        }
    }
    
    // Process remaining vectors
    for (; v < nvec; ++v) {
        __m256 sum = _mm256_setzero_ps();
        for (size_t d = 0; d < simd_dim; d += 8) {
            const __m256 vq = _mm256_loadu_ps(q + d);
            const __m256 vv = _mm256_loadu_ps(vectors + v * dim + d);
            sum = _mm256_fmadd_ps(vq, vv, sum);
        }
        distances[v] = detail::hsum_ps(sum);
        
        for (size_t d = simd_dim; d < dim; ++d) {
            distances[v] += q[d] * vectors[v * dim + d];
        }
    }
}

/** \brief Get AVX2 kernel operations table.
 *
 * Returns static singleton of kernel function pointers.
 * Thread-safe through function-local static initialization.
 */
inline const KernelOps& get_avx2_ops() noexcept {
    static const KernelOps ops{
        &avx2_l2_sq,
        &avx2_inner_product,
        &avx2_cosine_similarity,
        &avx2_cosine_distance,
        &avx2_batch_l2_sq,
        &avx2_batch_inner_product
    };
    return ops;
}

} // namespace vesper::kernels

#endif // defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64)