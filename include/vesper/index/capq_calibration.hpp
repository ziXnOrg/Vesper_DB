/**
 * Copyright (c) 2025 Colin Macritchie / Ripple Group, LLC
 * Licensed under the Apache License, Version 2.0
 */

#pragma once

/** \file capq_calibration.hpp
 *  \brief Isotonic regression calibration for distance correction.
 */

#include <vector>

namespace vesper::index {

/** \brief Piecewise-constant isotonic calibrator using PAV algorithm. */
struct IsotonicCalibrator {
  // Domain breakpoints (sorted, strictly increasing). Length == values.size().
  std::vector<float> breaks;
  // Calibrated values for intervals [breaks[i], breaks[i+1)).
  std::vector<float> values;

  float map(float x) const noexcept;
  bool empty() const noexcept { return breaks.empty() || values.empty(); }
};

/** \brief Fit monotone increasing calibration y=f(x) via Pool-Adjacent-Violators.
 *  Inputs should be paired measurements (x_i: raw distance, y_i: target distance).
 */
IsotonicCalibrator fit_isotonic(const std::vector<float>& x,
                                const std::vector<float>& y) noexcept;

} // namespace vesper::index


