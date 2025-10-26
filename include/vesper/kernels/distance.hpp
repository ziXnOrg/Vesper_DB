#pragma once

/** \file distance.hpp
 *  \brief Scalar reference distance kernels (L2^2, Inner Product, Cosine).
 *
 * Preconditions
 * - a.size() == b.size() > 0
 * - All inputs are finite
 * - For cosine: norms must be strictly positive (||a|| > 0 and ||b|| > 0); behavior is undefined otherwise
 * Determinism: pure functions, no allocations, no exceptions on hot paths.
 */

#include <cstddef>
#include <vesper/span_polyfill.hpp>
#include <cmath>

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64)
#include <immintrin.h>
#endif

namespace vesper::kernels {

namespace detail {

/** \brief Software prefetch hint for scalar loops. */
inline void scalar_prefetch(const float* ptr) noexcept {
#if defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64)
    _mm_prefetch(reinterpret_cast<const char*>(ptr + 16), _MM_HINT_T0);
#elif defined(__builtin_prefetch)
    __builtin_prefetch(ptr + 16, 0, 3);
#else
    (void)ptr; // No-op on platforms without prefetch
#endif
}

} // namespace detail

/** \brief Sum of squared differences: sum((a[i] - b[i])^2). O(d). */
inline float l2_sq(std::span<const float> a, std::span<const float> b) {
  const std::size_t n = a.size();
  const float* pa = a.data();
  const float* pb = b.data();
  
  // 4-way unrolled loop for better instruction-level parallelism
  float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
  std::size_t i = 0;
  
  const std::size_t unroll_end = n & ~3u;
  for (; i < unroll_end; i += 4) {
    // Prefetch next cache line
    if (i + 16 < n) {
      detail::scalar_prefetch(pa + i);
      detail::scalar_prefetch(pb + i);
    }
    
    float d0 = pa[i] - pb[i];
    float d1 = pa[i+1] - pb[i+1];
    float d2 = pa[i+2] - pb[i+2];
    float d3 = pa[i+3] - pb[i+3];
    
    s0 += d0 * d0;
    s1 += d1 * d1;
    s2 += d2 * d2;
    s3 += d3 * d3;
  }
  
  float s = s0 + s1 + s2 + s3;
  
  // Handle remaining elements
  for (; i < n; ++i) {
    float d = pa[i] - pb[i];
    s += d * d;
  }
  return s;
}

/** \brief Inner product: sum(a[i] * b[i]). O(d). */
inline float inner_product(std::span<const float> a, std::span<const float> b) {
  const std::size_t n = a.size();
  const float* pa = a.data();
  const float* pb = b.data();
  
  // 4-way unrolled loop
  float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
  std::size_t i = 0;
  
  const std::size_t unroll_end = n & ~3u;
  for (; i < unroll_end; i += 4) {
    // Prefetch next cache line
    if (i + 16 < n) {
      detail::scalar_prefetch(pa + i);
      detail::scalar_prefetch(pb + i);
    }
    
    s0 += pa[i] * pb[i];
    s1 += pa[i+1] * pb[i+1];
    s2 += pa[i+2] * pb[i+2];
    s3 += pa[i+3] * pb[i+3];
  }
  
  float s = s0 + s1 + s2 + s3;
  
  // Handle remaining elements
  for (; i < n; ++i) {
    s += pa[i] * pb[i];
  }
  return s;
}

/** \brief Cosine similarity: (a·b) / (||a|| * ||b||). Norms must be > 0. O(d). */
inline float cosine_similarity(std::span<const float> a, std::span<const float> b) {
  const std::size_t n = a.size();
  const float* pa = a.data();
  const float* pb = b.data();
  
  // 4-way unrolled loop for all three sums
  float dot0 = 0.0f, dot1 = 0.0f, dot2 = 0.0f, dot3 = 0.0f;
  float na0 = 0.0f, na1 = 0.0f, na2 = 0.0f, na3 = 0.0f;
  float nb0 = 0.0f, nb1 = 0.0f, nb2 = 0.0f, nb3 = 0.0f;
  std::size_t i = 0;
  
  const std::size_t unroll_end = n & ~3u;
  for (; i < unroll_end; i += 4) {
    // Prefetch next cache line
    if (i + 16 < n) {
      detail::scalar_prefetch(pa + i);
      detail::scalar_prefetch(pb + i);
    }
    
    float a0 = pa[i], b0 = pb[i];
    float a1 = pa[i+1], b1 = pb[i+1];
    float a2 = pa[i+2], b2 = pb[i+2];
    float a3 = pa[i+3], b3 = pb[i+3];
    
    dot0 += a0 * b0; na0 += a0 * a0; nb0 += b0 * b0;
    dot1 += a1 * b1; na1 += a1 * a1; nb1 += b1 * b1;
    dot2 += a2 * b2; na2 += a2 * a2; nb2 += b2 * b2;
    dot3 += a3 * b3; na3 += a3 * a3; nb3 += b3 * b3;
  }
  
  float dot = dot0 + dot1 + dot2 + dot3;
  float na2_total = na0 + na1 + na2 + na3;
  float nb2_total = nb0 + nb1 + nb2 + nb3;
  
  // Handle remaining elements
  for (; i < n; ++i) {
    float av = pa[i], bv = pb[i];
    dot += av * bv;
    na2_total += av * av;
    nb2_total += bv * bv;
  }
  
  float denom = std::sqrt(na2_total) * std::sqrt(nb2_total);
  return dot / denom; // UB if denom==0 per preconditions
}

/** \brief Cosine distance defined as 1 - cosine_similarity(a,b). O(d). */
inline float cosine_distance(std::span<const float> a, std::span<const float> b) {
  return 1.0f - cosine_similarity(a, b);
}

} // namespace vesper::kernels

