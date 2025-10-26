/**
 * Copyright (c) 2025 Colin Macritchie / Ripple Group, LLC
 */

#include "vesper/index/capq_calibration.hpp"

#include <algorithm>
#include <numeric>

namespace vesper::index {

float IsotonicCalibrator::map(float x) const noexcept {
  if (empty()) return x;
  // Binary search interval
  auto it = std::upper_bound(breaks.begin(), breaks.end(), x);
  std::size_t idx = it == breaks.begin() ? 0 : static_cast<std::size_t>((it - breaks.begin()) - 1);
  if (idx >= values.size()) idx = values.size() - 1;
  return values[idx];
}

IsotonicCalibrator fit_isotonic(const std::vector<float>& x,
                                const std::vector<float>& y) noexcept {
  // Sort pairs by x
  std::vector<std::pair<float,float>> xy(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) xy[i] = {x[i], y[i]};
  std::sort(xy.begin(), xy.end(), [](auto& a, auto& b){ return a.first < b.first; });

  // PAV: start with each point in its own block
  struct Block { double sum_w; double sum_y; float left_x; float right_x; };
  std::vector<Block> blocks; blocks.reserve(xy.size());
  for (const auto& [xi, yi] : xy) {
    blocks.push_back({1.0, yi, xi, xi});
    // Merge while monotonicity violated (avg must be non-decreasing)
    while (blocks.size() >= 2) {
      const std::size_t n = blocks.size();
      double avg_prev = blocks[n-2].sum_y / blocks[n-2].sum_w;
      double avg_curr = blocks[n-1].sum_y / blocks[n-1].sum_w;
      if (avg_prev <= avg_curr) break;
      // merge last two
      blocks[n-2].sum_w += blocks[n-1].sum_w;
      blocks[n-2].sum_y += blocks[n-1].sum_y;
      blocks[n-2].right_x = blocks[n-1].right_x;
      blocks.pop_back();
    }
  }

  IsotonicCalibrator c;
  c.breaks.reserve(blocks.size());
  c.values.reserve(blocks.size());
  for (const auto& b : blocks) {
    c.breaks.push_back(b.left_x);
    c.values.push_back(static_cast<float>(b.sum_y / b.sum_w));
  }
  return c;
}

} // namespace vesper::index


