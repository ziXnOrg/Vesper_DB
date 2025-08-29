#pragma once

/** \file scalar.hpp
 *  \brief Scalar backend implementing KernelOps via distance.hpp reference kernels.
 */

#include <span>
#include "vesper/kernels/dispatch.hpp"
#include "vesper/kernels/distance.hpp"

namespace vesper::kernels {

inline float scalar_l2(std::span<const float> a, std::span<const float> b) noexcept { return l2_sq(a,b); }
inline float scalar_ip(std::span<const float> a, std::span<const float> b) noexcept { return inner_product(a,b); }
inline float scalar_cos(std::span<const float> a, std::span<const float> b) noexcept { return cosine_similarity(a,b); }
inline float scalar_cosd(std::span<const float> a, std::span<const float> b) noexcept { return cosine_distance(a,b); }

inline const KernelOps& get_scalar_ops() noexcept {
  static const KernelOps ops{
    &scalar_l2,
    &scalar_ip,
    &scalar_cos,
    &scalar_cosd
  };
  return ops;
}

inline const KernelOps& select_backend(std::string_view /*name*/) noexcept {
  // Only scalar backend for now; future backends ("avx2", "neon") will be matched here.
  return get_scalar_ops();
}

} // namespace vesper::kernels

