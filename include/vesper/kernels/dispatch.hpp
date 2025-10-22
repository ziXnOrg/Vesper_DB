#pragma once

/** \file dispatch.hpp
 *  \brief SIMD-ready kernel interface and dispatcher. Scalar is the default backend.
 *
 * Preconditions for all ops: a.size() == b.size() > 0; inputs finite; for cosine, norms > 0.
 * Determinism: pure functions, O(d) complexity; no allocations; no exceptions on hot paths.
 */

#include <cstddef>
#include <vesper/span_polyfill.hpp>
#include <string_view>

namespace vesper::kernels {

struct KernelOps {
  float (*l2_sq)(std::span<const float>, std::span<const float>) noexcept;
  float (*inner_product)(std::span<const float>, std::span<const float>) noexcept;
  float (*cosine_similarity)(std::span<const float>, std::span<const float>) noexcept;
  float (*cosine_distance)(std::span<const float>, std::span<const float>) noexcept;
  
  // Batch operations for multiple vectors
  void (*batch_l2_sq)(std::span<const float> query, 
                      const float* vectors, size_t nvec, size_t dim,
                      float* distances) noexcept;
  void (*batch_inner_product)(std::span<const float> query,
                              const float* vectors, size_t nvec, size_t dim,
                              float* distances) noexcept;
};

// Returns a stable reference valid for the process lifetime. "scalar" is the default backend.
const KernelOps& select_backend(std::string_view name = "scalar") noexcept;

/** \brief Auto-selects a kernel backend based on CPU features.
 *  Thread-safe initialization and stable reference semantics apply.
 */
const KernelOps& select_backend_auto() noexcept;

} // namespace vesper::kernels

