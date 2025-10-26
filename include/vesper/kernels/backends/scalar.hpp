#pragma once

/** \file scalar.hpp
 *  \brief Scalar backend implementing KernelOps via distance.hpp reference kernels.
 */

#include <vesper/span_polyfill.hpp>
#include <string_view>
#include "vesper/kernels/dispatch.hpp"
#include "vesper/kernels/distance.hpp"
#include "vesper/kernels/backends/stub_avx2.hpp"
#include "vesper/kernels/backends/stub_neon.hpp"

#ifndef VESPER_ENABLE_BACKEND_STUB_AVX2
#define VESPER_ENABLE_BACKEND_STUB_AVX2 1
#endif
#ifndef VESPER_ENABLE_BACKEND_STUB_NEON
#define VESPER_ENABLE_BACKEND_STUB_NEON 1
#endif

namespace vesper::kernels {

inline float scalar_l2(std::span<const float> a, std::span<const float> b) noexcept { return l2_sq(a,b); }
inline float scalar_ip(std::span<const float> a, std::span<const float> b) noexcept { return inner_product(a,b); }
inline float scalar_cos(std::span<const float> a, std::span<const float> b) noexcept { return cosine_similarity(a,b); }
inline float scalar_cosd(std::span<const float> a, std::span<const float> b) noexcept { return cosine_distance(a,b); }

inline void scalar_batch_l2_sq(std::span<const float> query,
                               const float* vectors, size_t nvec, size_t dim,
                               float* distances) noexcept {
    for (size_t v = 0; v < nvec; ++v) {
        distances[v] = l2_sq(query, std::span<const float>(vectors + v * dim, dim));
    }
}

inline void scalar_batch_inner_product(std::span<const float> query,
                                       const float* vectors, size_t nvec, size_t dim,
                                       float* distances) noexcept {
    for (size_t v = 0; v < nvec; ++v) {
        distances[v] = inner_product(query, std::span<const float>(vectors + v * dim, dim));
    }
}

inline const KernelOps& get_scalar_ops() noexcept {
  static const KernelOps ops{ 
      &scalar_l2, &scalar_ip, &scalar_cos, &scalar_cosd,
      &scalar_batch_l2_sq, &scalar_batch_inner_product 
  };
  return ops;
}

// Implementation moved to dispatch.cpp for proper CPU feature detection

} // namespace vesper::kernels

