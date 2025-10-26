/**
 * Copyright (c) 2025 Colin Macritchie / Ripple Group, LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

/** \file capq_util.hpp
 *  \brief Utilities for CAPQ: FWHT-64, seeded permutation/signs, and Hamming sketches.
 */

#include <cstdint>
#include <array>
#include <expected>

#include "vesper/error.hpp"
#include "vesper/index/capq.hpp"

namespace vesper::index {

/** \brief In-place Fast Walsh-Hadamard Transform for 64 floats (scalar). */
void fwht64_inplace(float* data) noexcept;

/** \brief Compute a seed-derived permutation of [0,64).
 *  Deterministic Fisher–Yates using SplitMix64 for randomness.
 */
void seeded_permutation64(std::uint64_t seed, std::array<std::uint8_t, 64>& out_indices) noexcept;

/** \brief Compute CAPQ Hamming sketch planes.
 *
 *  Plane 0: direct sign bits of input z (no transform).
 *  Planes 1..N: FWHT-64(z), then apply seeded permutation and seeded sign flips.
 *
 *  \param z          Input projected/whitened 64-D vector
 *  \param seeds      At least words_per_vector seeds (first seed unused for plane0)
 *  \param bits       Target sketch width (256 or 384)
 *  \param out_words  Output buffer with size words_per_vector(bits)
 */
auto compute_hamming_sketch(const float z[64],
                            const std::array<std::uint64_t, 6>& seeds,
                            CapqHammingBits bits,
                            std::uint64_t* out_words) -> std::expected<void, core::error>;

/** \brief Helper: number of 64-bit words for a given sketch width. */
constexpr inline std::size_t capq_words_per_vector(CapqHammingBits bits) noexcept {
  return (bits == CapqHammingBits::B256) ? 4u : 6u;
}

} // namespace vesper::index


