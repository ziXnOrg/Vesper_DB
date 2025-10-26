/**
 * Copyright (c) 2025 Colin Macritchie / Ripple Group, LLC
 * Licensed under the Apache License, Version 2.0
 */

#pragma once

/** \file capq_opq.hpp
 *  \brief ITQ/OPQ-style 64x64 orthonormal rotation for CAPQ projection refinement.
 */

#include <array>
#include <cstddef>
#include <expected>

#include "vesper/error.hpp"

namespace vesper::index {

struct CapqRotationModel {
  std::array<float, 64 * 64> R{}; /**< row-major orthonormal rotation */
};

/** \brief Train an ITQ-style rotation on zero-mean 64D data via Procrustes + sign bins.
 *  Implementation uses polar decomposition Q = M (M^T M)^{-1/2} with Newton–Schulz.
 */
auto train_itq_rotation(const float* z_data, std::size_t n, int iters = 10)
    -> std::expected<CapqRotationModel, core::error>;

/** \brief Apply rotation R to 64D vector. */
inline void apply_rotation(const CapqRotationModel& model, const float in[64], float out[64]) noexcept {
  for (int r = 0; r < 64; ++r) {
    float acc = 0.0f;
    const float* rr = &model.R[r * 64];
    for (int c = 0; c < 64; ++c) acc += rr[c] * in[c];
    out[r] = acc;
  }
}

} // namespace vesper::index


