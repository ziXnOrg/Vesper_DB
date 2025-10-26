/**
 * Copyright (c) 2025 Colin Macritchie / Ripple Group, LLC
 * Licensed under the Apache License, Version 2.0
 */

#pragma once

/** \file capq_dist.hpp
 *  \brief Scalar distance kernels for CAPQ stages (Hamming, q4, q8).
 */

#include <cstddef>
#include <cstdint>

#include "vesper/index/capq.hpp"
#include "vesper/index/capq_encode.hpp"
#include "vesper/index/capq_dist_avx2.hpp"
#include "vesper/index/capq_q4.hpp"

namespace vesper::index {

/** \brief Hamming distance with AVX2 dispatch. */
inline std::uint32_t hamming_distance_words(const std::uint64_t* a,
                                            const std::uint64_t* b,
                                            std::size_t words) noexcept {
  if (capq_avx2_available()) {
    return hamming_distance_avx2(a, b, words);
  }
  std::uint32_t dist = 0;
  for (std::size_t i = 0; i < words; ++i) {
    const std::uint64_t x = a[i] ^ b[i];
#if defined(__GNUG__) || defined(__clang__)
    dist += static_cast<std::uint32_t>(__builtin_popcountll(x));
#else
    dist += static_cast<std::uint32_t>(__popcnt64(x));
#endif
  }
  return dist;
}

/** \brief Stage 2: q4 distance (scalar).
 *  Diff of 4-bit nibbles, scaled by (16*scale[d])^2 and accumulated.
 */
inline float distance_q4_scalar(const std::uint8_t* q4_query,
                                const std::uint8_t* q4_db,
                                const CapqQ8Params& params) noexcept {
  float sum = 0.0f;
  for (int i = 0; i < 32; ++i) {
    const std::uint8_t qq = q4_query[i];
    const std::uint8_t dd = q4_db[i];
    // Map byte to two dims: idx 2*i (low nibble), 2*i+1 (high nibble)
    const int q_lo = static_cast<int>(qq & 0x0F);
    const int q_hi = static_cast<int>((qq >> 4) & 0x0F);
    const int d_lo = static_cast<int>(dd & 0x0F);
    const int d_hi = static_cast<int>((dd >> 4) & 0x0F);

    const int d0 = 2 * i;
    const int d1 = 2 * i + 1;
    const float s0 = params.scale[d0];
    const float s1 = params.scale[d1];

    const int diff0 = q_lo - d_lo;
    const int diff1 = q_hi - d_hi;
    // (16*scale)^2 = 256 * scale^2
    sum += static_cast<float>(diff0 * diff0) * (256.0f * s0 * s0);
    sum += static_cast<float>(diff1 * diff1) * (256.0f * s1 * s1);
  }
  return sum;
}

/** \brief Dispatching q4 distance: uses AVX2 when available, otherwise scalar. */
inline float distance_q4(const std::uint8_t* q4_query,
                         const std::uint8_t* q4_db,
                         const CapqQ8Params& params) noexcept {
  if (capq_avx2_available()) {
    return distance_q4_avx2(q4_query, q4_db, params);
  }
  return distance_q4_scalar(q4_query, q4_db, params);
}

/** \brief Dispatching ADE distance: AVX2 path if available. */
inline float distance_q4_ade_dispatch(const float zq[64],
                                      const std::uint8_t* q4_db,
                                      const CapqQ4Codebooks& cb) noexcept {
  if (capq_avx2_available()) {
    return distance_q4_ade_avx2(zq, q4_db, cb);
  }
  return distance_q4_ade(zq, q4_db, cb);
}

/** \brief Stage 3: q8 distance (scalar) using per-dim scales. */
inline float distance_q8_scalar(const std::int8_t* q8_query,
                                const std::int8_t* q8_db,
                                const CapqQ8Params& params) noexcept {
  float sum = 0.0f;
  for (int d = 0; d < 64; ++d) {
    const int diff = static_cast<int>(q8_query[d]) - static_cast<int>(q8_db[d]);
    const float s = params.scale[d];
    sum += static_cast<float>(diff * diff) * (s * s);
  }
  return sum;
}

/** \brief Dispatching q8 distance: uses AVX2 when available, otherwise scalar. */
inline float distance_q8(const std::int8_t* q8_query,
                         const std::int8_t* q8_db,
                         const CapqQ8Params& params) noexcept {
  if (capq_avx2_available()) {
    return distance_q8_avx2(q8_query, q8_db, params);
  }
  return distance_q8_scalar(q8_query, q8_db, params);
}

} // namespace vesper::index


