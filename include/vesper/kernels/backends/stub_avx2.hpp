#pragma once

/** \file stub_avx2.hpp
 *  \brief Stub AVX2 backend (no intrinsics). Delegates to scalar reference kernels.
 *
 * Preconditions: a.size() == b.size() > 0; inputs finite; for cosine, norms > 0.
 * Determinism: pure functions, O(d), no allocations, noexcept.
 */

#include <vesper/span_polyfill.hpp>
#include "vesper/kernels/dispatch.hpp"
#include "vesper/kernels/distance.hpp"

namespace vesper::kernels {

inline float avx2_l2(std::span<const float> a, std::span<const float> b) noexcept { return l2_sq(a,b); }
inline float avx2_ip(std::span<const float> a, std::span<const float> b) noexcept { return inner_product(a,b); }
inline float avx2_cos(std::span<const float> a, std::span<const float> b) noexcept { return cosine_similarity(a,b); }
inline float avx2_cosd(std::span<const float> a, std::span<const float> b) noexcept { return cosine_distance(a,b); }

inline const KernelOps& get_stub_avx2_ops() noexcept {
  static const KernelOps ops{ &avx2_l2, &avx2_ip, &avx2_cos, &avx2_cosd };
  return ops;
}

} // namespace vesper::kernels

