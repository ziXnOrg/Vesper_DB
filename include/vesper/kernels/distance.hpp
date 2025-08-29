#pragma once

/** \file distance.hpp
 *  \brief Scalar reference distance kernels (L2^2, Inner Product, Cosine).
 *
 * Preconditions: a.size() == b.size() > 0; inputs are finite; for cosine, norms > 0.
 * Determinism: pure functions, no allocations, no exceptions on hot paths.
 */

#include <cstddef>
#include <span>
#include <cmath>

namespace vesper::kernels {

/** \brief Sum of squared differences: sum((a[i] - b[i])^2). */
inline float l2_sq(std::span<const float> a, std::span<const float> b) {
  const std::size_t n = a.size();
  float s = 0.0f;
  for (std::size_t i = 0; i < n; ++i) {
    float d = a[i] - b[i];
    s += d * d;
  }
  return s;
}

/** \brief Inner product: sum(a[i] * b[i]). */
inline float inner_product(std::span<const float> a, std::span<const float> b) {
  const std::size_t n = a.size();
  float s = 0.0f;
  for (std::size_t i = 0; i < n; ++i) s += a[i] * b[i];
  return s;
}

/** \brief Cosine similarity: (a·b) / (||a|| * ||b||). Norms must be > 0. */
inline float cosine_similarity(std::span<const float> a, std::span<const float> b) {
  const std::size_t n = a.size();
  float dot = 0.0f, na2 = 0.0f, nb2 = 0.0f;
  for (std::size_t i = 0; i < n; ++i) {
    float av = a[i], bv = b[i];
    dot += av * bv; na2 += av * av; nb2 += bv * bv;
  }
  float denom = std::sqrt(na2) * std::sqrt(nb2);
  return dot / denom; // UB if denom==0 per preconditions
}

/** \brief Cosine distance defined as 1 - cosine_similarity(a,b). */
inline float cosine_distance(std::span<const float> a, std::span<const float> b) {
  return 1.0f - cosine_similarity(a, b);
}

} // namespace vesper::kernels

