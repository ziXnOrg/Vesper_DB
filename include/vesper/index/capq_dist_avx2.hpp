/**
 * Copyright (c) 2025 Colin Macritchie / Ripple Group, LLC
 * Licensed under the Apache License, Version 2.0
 */

#pragma once

/** \file capq_dist_avx2.hpp
 *  \brief AVX2-optimized CAPQ distance kernels.
 */

#include <cstddef>
#include <cstdint>

#include "vesper/index/capq_encode.hpp"
#include "vesper/index/capq_q4.hpp"

namespace vesper::index {

/** \brief Runtime check for AVX2 support (compiled-in and CPU-available). */
bool capq_avx2_available() noexcept;

/** \brief AVX2 implementation of q8 distance with per-dimension scaling.
 *  Falls back to scalar internally if AVX2 not available.
 */
float distance_q8_avx2(const std::int8_t* q8_query,
                       const std::int8_t* q8_db,
                       const CapqQ8Params& params) noexcept;

/** \brief AVX2 implementation of Hamming distance over 256/384-bit sketches. */
std::uint32_t hamming_distance_avx2(const std::uint64_t* a,
                                    const std::uint64_t* b,
                                    std::size_t words) noexcept;

/** \brief AVX2 implementation of q4 distance using per-dim scales (monotone coarsening). */
float distance_q4_avx2(const std::uint8_t* q4_query,
                       const std::uint8_t* q4_db,
                       const CapqQ8Params& params) noexcept;

/** \brief AVX2 ADE distance: uses learned q4 codebooks and per-dim weights. */
float distance_q4_ade_avx2(const float zq[64],
                           const std::uint8_t* q4_db,
                           const CapqQ4Codebooks& cb) noexcept;

} // namespace vesper::index


