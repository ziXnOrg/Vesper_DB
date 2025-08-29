#pragma once

/** \file scalar.hpp
 *  \brief Scalar backend implementing KernelOps via distance.hpp reference kernels.
 */

#include <span>
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

inline const KernelOps& get_scalar_ops() noexcept {
  static const KernelOps ops{ &scalar_l2, &scalar_ip, &scalar_cos, &scalar_cosd };
  return ops;
}

inline const KernelOps& select_backend(std::string_view name) noexcept {
  if (name == "scalar" || name.empty()) return get_scalar_ops();
#if VESPER_ENABLE_BACKEND_STUB_AVX2
  if (name == "stub-avx2") return get_stub_avx2_ops();
#endif
#if VESPER_ENABLE_BACKEND_STUB_NEON
  if (name == "stub-neon") return get_stub_neon_ops();
#endif
  return get_scalar_ops();
}

} // namespace vesper::kernels

