/**
 * Copyright (c) 2025 Colin Macritchie / Ripple Group, LLC
 */

#include "vesper/index/capq_q4.hpp"

#include <algorithm>
#include <cmath>
#include <vector>
#include <array>

namespace vesper::index {

static void init_quantiles(const float* z, std::size_t n, int dim, float centers[16]) {
  std::vector<float> vals; vals.reserve(n);
  for (std::size_t i = 0; i < n; ++i) vals.push_back(z[i * 64 + dim]);
  std::sort(vals.begin(), vals.end());
  for (int c = 0; c < 16; ++c) {
    const double q = (c + 0.5) / 16.0;
    const std::size_t idx = static_cast<std::size_t>(q * (vals.size() - 1));
    centers[c] = vals[idx];
  }
}

auto train_q4_codebooks(const float* z_data, std::size_t n)
    -> std::expected<CapqQ4Codebooks, core::error> {
  if (z_data == nullptr || n == 0) {
    return std::vesper_unexpected(core::error{core::error_code::invalid_argument,
                                       "q4 train: null/empty input",
                                       "index.capq.q4"});
  }
  CapqQ4Codebooks cb{};
  // Initialize centers by quantiles per dimension
  for (int d = 0; d < 64; ++d) init_quantiles(z_data, n, d, &cb.centers[d * 16]);
  // Lloyd iterations
  std::vector<int> assign(n * 64);
  const int iters = 8;
  for (int it = 0; it < iters; ++it) {
    // Assign
    for (int d = 0; d < 64; ++d) {
      float* cd = &cb.centers[d * 16];
      for (std::size_t i = 0; i < n; ++i) {
        const float v = z_data[i * 64 + d];
        int best = 0; float best_err = std::abs(v - cd[0]);
        for (int c = 1; c < 16; ++c) { const float e = std::abs(v - cd[c]); if (e < best_err) { best_err = e; best = c; } }
        assign[i * 64 + d] = best;
      }
    }
    // Update
    for (int d = 0; d < 64; ++d) {
      float sums[16]{}; int counts[16]{};
      for (std::size_t i = 0; i < n; ++i) { const int c = assign[i * 64 + d]; sums[c] += z_data[i * 64 + d]; counts[c]++; }
      for (int c = 0; c < 16; ++c) {
        if (counts[c] > 0) cb.centers[d * 16 + c] = sums[c] / static_cast<float>(counts[c]);
      }
      // Optional: sort centers to enforce monotonic codes
      std::sort(&cb.centers[d * 16], &cb.centers[d * 16] + 16);
    }
  }
  // Weights as inverse variance with floor
  for (int d = 0; d < 64; ++d) {
    double mean = 0.0; for (std::size_t i = 0; i < n; ++i) mean += z_data[i * 64 + d]; mean /= double(n);
    double var = 0.0; for (std::size_t i = 0; i < n; ++i) { double dv = double(z_data[i * 64 + d]) - mean; var += dv * dv; }
    var = (n > 1) ? var / double(n - 1) : var;
    const double inv = 1.0 / std::max(var, 1e-6);
    cb.weights[d] = static_cast<float>(inv);
  }
  return cb;
}

void encode_q4_learned(const float z[64], const CapqQ4Codebooks& cb,
                       std::uint8_t q4_packed[32]) noexcept {
  for (int i = 0; i < 32; ++i) {
    const int d0 = 2 * i, d1 = 2 * i + 1;
    const float* c0 = &cb.centers[d0 * 16];
    const float* c1 = &cb.centers[d1 * 16];
    const float w0 = cb.weights[d0] > 0.0f ? cb.weights[d0] : 1.0f;
    const float w1 = cb.weights[d1] > 0.0f ? cb.weights[d1] : 1.0f;
    int b0 = 0; float best0 = w0 * (z[d0] - c0[0]) * (z[d0] - c0[0]);
    for (int c = 1; c < 16; ++c) {
      const float err = w0 * (z[d0] - c0[c]) * (z[d0] - c0[c]);
      if (err < best0) { best0 = err; b0 = c; }
    }
    int b1 = 0; float best1 = w1 * (z[d1] - c1[0]) * (z[d1] - c1[0]);
    for (int c = 1; c < 16; ++c) {
      const float err = w1 * (z[d1] - c1[c]) * (z[d1] - c1[c]);
      if (err < best1) { best1 = err; b1 = c; }
    }
    q4_packed[i] = static_cast<std::uint8_t>((b1 << 4) | (b0 & 0x0F));
  }
}

// Solve 64x64 ridge (F^T F + lambda I) w = F^T y using simple Cholesky
static bool cholesky64(std::array<double, 64 * 64>& A) {
  for (int i = 0; i < 64; ++i) {
    for (int j = 0; j <= i; ++j) {
      double sum = A[i * 64 + j];
      for (int k = 0; k < j; ++k) sum -= A[i * 64 + k] * A[j * 64 + k];
      if (i == j) {
        if (sum <= 0.0) return false;
        A[i * 64 + j] = std::sqrt(sum);
      } else {
        A[i * 64 + j] = sum / A[j * 64 + j];
      }
    }
    for (int j = i + 1; j < 64; ++j) A[i * 64 + j] = 0.0; // store lower-triangular only
  }
  return true;
}

static void chol_solve64(const std::array<double, 64 * 64>& L, const std::array<double, 64>& b, std::array<double, 64>& x) {
  // Forward solve Ly = b
  std::array<double, 64> y{};
  for (int i = 0; i < 64; ++i) {
    double sum = b[i];
    for (int k = 0; k < i; ++k) sum -= L[i * 64 + k] * y[k];
    y[i] = sum / L[i * 64 + i];
  }
  // Backward solve L^T x = y
  for (int i = 63; i >= 0; --i) {
    double sum = y[i];
    for (int k = i + 1; k < 64; ++k) sum -= L[k * 64 + i] * x[k];
    x[i] = sum / L[i * 64 + i];
  }
}

auto fit_ade_weights_ridge(const float* F, std::size_t n_samples, std::size_t stride,
                           const float* y, float lambda_)
    -> std::expected<std::array<float, 64>, core::error> {
  if (!F || !y || n_samples == 0 || stride < 64) {
    return std::vesper_unexpected(core::error{core::error_code::invalid_argument,
                                       "fit_ade_weights_ridge: invalid inputs",
                                       "index.capq.q4"});
  }
  std::array<double, 64 * 64> G{}; // F^T F
  std::array<double, 64>       b{}; // F^T y
  for (std::size_t i = 0; i < n_samples; ++i) {
    const float* fi = F + i * stride;
    // b += fi * y[i]
    for (int d = 0; d < 64; ++d) b[d] += double(fi[d]) * double(y[i]);
    // G += fi^T fi
    for (int r = 0; r < 64; ++r) {
      const double fr = double(fi[r]);
      for (int c = 0; c < 64; ++c) G[r * 64 + c] += fr * double(fi[c]);
    }
  }
  // Add ridge lambda to diagonal
  const double lam = std::max(0.0, double(lambda_));
  for (int d = 0; d < 64; ++d) G[d * 64 + d] += lam;

  // Cholesky factorization (in-place G becomes L)
  if (!cholesky64(G)) {
    return std::vesper_unexpected(core::error{core::error_code::internal,
                                       "fit_ade_weights_ridge: Cholesky failed",
                                       "index.capq.q4"});
  }
  std::array<double, 64> x{};
  chol_solve64(G, b, x);
  std::array<float, 64> w{};
  for (int d = 0; d < 64; ++d) w[d] = static_cast<float>(x[d]);
  return w;
}

} // namespace vesper::index


