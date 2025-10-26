/**
 * Copyright (c) 2025 Colin Macritchie / Ripple Group, LLC
 * Licensed under the Apache License, Version 2.0
 */

#pragma once

/** \file capq_q4.hpp
 *  \brief Learned 4-bit per-dimension codebooks and ADE distance for CAPQ Stage 2.
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <algorithm>

#include "vesper/error.hpp"

namespace vesper::index {

/** \brief Learned per-dimension 16-level codebooks and weights. */
struct CapqQ4Codebooks {
  std::array<float, 64 * 16> centers{};  /**< row-major: centers[d*16 + c] */
  std::array<float, 64> weights{};       /**< per-dimension weight for ADE */
};

/** \brief Train 16-level per-dimension codebooks via 1D Lloyd iterations.
 *  - Initializes centers to quantiles per dimension
 *  - Runs a small fixed number of Lloyd updates
 *  - Computes per-dim weights (inverse variance with floor)
 */
auto train_q4_codebooks(const float* z_data, std::size_t n)
    -> std::expected<CapqQ4Codebooks, core::error>;

/** \brief Encode one 64D vector to packed q4 using learned codebooks. */
void encode_q4_learned(const float z[64], const CapqQ4Codebooks& cb,
                       std::uint8_t q4_packed[32]) noexcept;

/** \brief ADE distance: sum_d w[d] * (zq[d] - centers[d, code_db[d]])^2. */
inline float distance_q4_ade(const float zq[64], const std::uint8_t* q4_db,
                             const CapqQ4Codebooks& cb) noexcept {
  float sum = 0.0f;
  for (int i = 0; i < 32; ++i) {
    const std::uint8_t dd = q4_db[i];
    const int c0 = static_cast<int>(dd & 0x0F);
    const int c1 = static_cast<int>((dd >> 4) & 0x0F);
    const int d0 = 2 * i;
    const int d1 = 2 * i + 1;
    const float w0 = cb.weights[d0];
    const float w1 = cb.weights[d1];
    const float mu0 = cb.centers[d0 * 16 + c0];
    const float mu1 = cb.centers[d1 * 16 + c1];
    const float e0 = zq[d0] - mu0;
    const float e1 = zq[d1] - mu1;
    sum += w0 * (e0 * e0);
    sum += w1 * (e1 * e1);
  }
  return sum;
}

/** \brief Compute 64-dim ADE feature vector f where f[d] = (zq[d] - mu[d, code_d])^2.
 *  The features are independent of per-dimension weights and suitable for regression.
 */
inline void compute_q4_ade_features(const float zq[64], const std::uint8_t* q4_db,
                                    const CapqQ4Codebooks& cb, float out_features[64]) noexcept {
  for (int i = 0; i < 32; ++i) {
    const std::uint8_t dd = q4_db[i];
    const int c0 = static_cast<int>(dd & 0x0F);
    const int c1 = static_cast<int>((dd >> 4) & 0x0F);
    const int d0 = 2 * i;
    const int d1 = 2 * i + 1;
    const float mu0 = cb.centers[d0 * 16 + c0];
    const float mu1 = cb.centers[d1 * 16 + c1];
    const float e0 = zq[d0] - mu0;
    const float e1 = zq[d1] - mu1;
    out_features[d0] = e0 * e0;
    out_features[d1] = e1 * e1;
  }
}

/** \brief Fit per-dimension ADE weights via ridge regression.
 *
 *  Solves (F^T F + lambda I) w = F^T y, where each row of F is a 64-dim feature
 *  vector f with f[d] = (zq[d] - mu[d, code_d])^2 and y is the true L2 distance.
 *
 *  Parameters:
 *  - F: pointer to row-major features (n_samples x 64)
 *  - n_samples: number of samples
 *  - stride: distance in floats between consecutive rows in F (>=64)
 *  - y: pointer to target vector (length n_samples)
 *  - lambda_: ridge penalty (non-negative)
 *
 *  Returns learned weight vector w[64] on success.
 */
auto fit_ade_weights_ridge(const float* F, std::size_t n_samples, std::size_t stride,
                           const float* y, float lambda_)
    -> std::expected<std::array<float, 64>, core::error>;

} // namespace vesper::index


