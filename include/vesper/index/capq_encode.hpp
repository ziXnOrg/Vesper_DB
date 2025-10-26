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

/** \file capq_encode.hpp
 *  \brief CAPQ training (whitening, quantization) and encode utilities.
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>

#include "vesper/error.hpp"
#include "vesper/index/capq.hpp"

namespace vesper::index {

/** \brief Whitening model: mean and 64x64 transform (row-major). */
struct CapqWhiteningModel {
  std::array<float, 64> mean{};       /**< mean of projected vectors */
  std::array<float, 64 * 64> W{};     /**< whitening transform (W · (x-mean)) */
  float lambda_ratio{1e-3f};          /**< ridge ratio used for training */
};

/** \brief Train whitening model with ridge regularization.
 *  lambda = lambda_ratio * trace(Sigma)/64.
 */
auto train_whitening_model(const float* data, std::size_t n,
                           float lambda_ratio) -> std::expected<CapqWhiteningModel, core::error>;

/** \brief Apply whitening transform to a single 64D vector. */
void apply_whitening(const CapqWhiteningModel& model, const float x[64], float z[64]) noexcept;

/** \brief Per-dimension quantization parameters for q8. */
struct CapqQ8Params {
  std::array<float, 64> scale{};       /**< scale per dimension */
  std::array<float, 64> zero_point{};  /**< zero-point per dimension (float for rounding) */
  bool symmetric{true};
};

/** \brief Train q8 scales/zero-points with optional symmetric clipping.
 *  If symmetric==true, zero_point=0, scale = max_abs/127 with clipping at percentile.
 */
auto train_q8_params(const float* z_data, std::size_t n, bool symmetric,
                     float clip_percentile)
    -> std::expected<CapqQ8Params, core::error>;

/** \brief Encode one whitened vector to q8/q4 and compute residual energy. */
void encode_capq_payload(const float z[64], const CapqQ8Params& q8p,
                         std::int8_t q8_out[64], std::uint8_t q4_out[32],
                         std::uint8_t& residual_energy_byte) noexcept;

/** \brief Pack two 4-bit nibbles into a byte: hi<<4 | lo. */
inline std::uint8_t pack_nibbles(std::uint8_t hi, std::uint8_t lo) noexcept {
  return static_cast<std::uint8_t>((hi << 4) | (lo & 0x0F));
}

/** \brief Coarsen q8 into q4 in-place packed form (monotone right-shift). */
void coarsen_q8_to_q4(const std::int8_t q8[64], std::uint8_t q4_packed[32]) noexcept;

} // namespace vesper::index


