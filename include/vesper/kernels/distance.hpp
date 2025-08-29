#pragma once

/** \file distance.hpp
 *  \brief Scalar reference distance kernels (L2^2 and Inner Product).
 *
 * Preconditions: a.size() == b.size() > 0; inputs are finite.
 * Determinism: pure functions, no allocations, no exceptions on hot paths.
 */

#include <cstddef>
#include <span>

namespace vesper::kernels {

inline float l2_sq(std::span<const float> a, std::span<const float> b) {
  const std::size_t n = a.size();
  float s = 0.0f;
  for (std::size_t i = 0; i < n; ++i) {
    float d = a[i] - b[i];
    s += d * d;
  }
  return s;
}

inline float inner_product(std::span<const float> a, std::span<const float> b) {
  const std::size_t n = a.size();
  float s = 0.0f;
  for (std::size_t i = 0; i < n; ++i) s += a[i] * b[i];
  return s;
}

} // namespace vesper::kernels

