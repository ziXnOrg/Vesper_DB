/**
 * Copyright (c) 2025 Colin Macritchie / Ripple Group, LLC
 */

#include "vesper/index/capq_encode.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "vesper/core/platform_utils.hpp"

namespace vesper::index {

namespace {

// Jacobi eigen decomposition for 64x64 real symmetric matrix (single-precision)
// A is modified in-place to a diagonal matrix on success. eigvec will contain eigenvectors in columns.
static void jacobi_eigendecompose_64(float* A, float* eigvec, float* eigval) {
  // Initialize eigenvectors to identity
  for (int i = 0; i < 64 * 64; ++i) eigvec[i] = 0.0f;
  for (int i = 0; i < 64; ++i) eigvec[i * 64 + i] = 1.0f;

  const int max_sweeps = 60;
  const float tol = 1e-5f;
  for (int sweep = 0; sweep < max_sweeps; ++sweep) {
    // Find largest off-diagonal element
    int p = 0, q = 1;
    float max_off = 0.0f;
    for (int i = 0; i < 64; ++i) {
      for (int j = i + 1; j < 64; ++j) {
        float aij = std::fabs(A[i * 64 + j]);
        if (aij > max_off) { max_off = aij; p = i; q = j; }
      }
    }
    if (max_off < tol) break;

    const float app = A[p * 64 + p];
    const float aqq = A[q * 64 + q];
    const float apq = A[p * 64 + q];
    if (std::fabs(apq) < 1e-12f) continue;

    float tau = (aqq - app) / (2.0f * apq);
    float t = std::copysign(1.0f / (std::fabs(tau) + std::sqrt(1.0f + tau * tau)), tau);
    float c = 1.0f / std::sqrt(1.0f + t * t);
    float s = t * c;

    // Rotate A in p,q plane: A' = J^T A J
    for (int k = 0; k < 64; ++k) {
      if (k != p && k != q) {
        float aik = A[p * 64 + k];
        float akq = A[q * 64 + k];
        float vip = c * aik - s * akq;
        float viq = s * aik + c * akq;
        A[p * 64 + k] = vip;
        A[k * 64 + p] = vip;
        A[q * 64 + k] = viq;
        A[k * 64 + q] = viq;
      }
    }
    float new_app = c * c * app - 2.0f * s * c * apq + s * s * aqq;
    float new_aqq = s * s * app + 2.0f * s * c * apq + c * c * aqq;
    A[p * 64 + p] = new_app;
    A[q * 64 + q] = new_aqq;
    A[p * 64 + q] = 0.0f;
    A[q * 64 + p] = 0.0f;

    // Update eigenvectors V = V J
    for (int k = 0; k < 64; ++k) {
      float vip = eigvec[k * 64 + p];
      float viq = eigvec[k * 64 + q];
      eigvec[k * 64 + p] = c * vip - s * viq;
      eigvec[k * 64 + q] = s * vip + c * viq;
    }
  }
  for (int i = 0; i < 64; ++i) eigval[i] = A[i * 64 + i];

  // Ensure non-negative and small floor for stability
  for (int i = 0; i < 64; ++i) if (eigval[i] < 1e-12f) eigval[i] = 1e-12f;
}

} // namespace

auto train_whitening_model(const float* data, std::size_t n,
                           float lambda_ratio) -> std::expected<CapqWhiteningModel, core::error> {
  if (data == nullptr || n == 0) {
    return std::vesper_unexpected(core::error{core::error_code::invalid_argument,
                                       "whitening: null or empty input",
                                       "index.capq.encode"});
  }
  CapqWhiteningModel model;
  model.lambda_ratio = lambda_ratio;

  // Compute mean
  for (std::size_t i = 0; i < n; ++i) {
    const float* x = data + i * 64;
    for (int d = 0; d < 64; ++d) model.mean[d] += x[d];
  }
  for (int d = 0; d < 64; ++d) model.mean[d] = model.mean[d] / static_cast<float>(n);

  // Compute covariance Sigma = E[(x-mean)(x-mean)^T]
  float Sigma[64 * 64]{};
  for (std::size_t i = 0; i < n; ++i) {
    const float* x = data + i * 64;
    float c[64];
    for (int d = 0; d < 64; ++d) c[d] = x[d] - model.mean[d];
    for (int r = 0; r < 64; ++r) {
      const float cr = c[r];
      for (int cidx = 0; cidx < 64; ++cidx) Sigma[r * 64 + cidx] += cr * c[cidx];
    }
  }
  for (int i = 0; i < 64 * 64; ++i) Sigma[i] /= static_cast<float>(n);

  // trace(Sigma)
  float trace = 0.0f;
  for (int d = 0; d < 64; ++d) trace += Sigma[d * 64 + d];
  const float lambda = lambda_ratio * (trace / 64.0f);

  // Add ridge to diagonal
  for (int d = 0; d < 64; ++d) Sigma[d * 64 + d] += lambda;

  // Decide whitening mode: diagonal (default) or PCA/ZCA via env CAPQ_WHITENING={diag|zca}
  const auto mode_env = vesper::core::safe_getenv("CAPQ_WHITENING");
  const bool use_zca = (mode_env && (*mode_env == "zca" || *mode_env == "ZCA" || *mode_env == "pca"));
  if (!use_zca) {
    // Diagonal whitening W = diag(1/sqrt(var))
    for (int r = 0; r < 64; ++r) {
      for (int cidx = 0; cidx < 64; ++cidx) model.W[r * 64 + cidx] = 0.0f;
      const float var = Sigma[r * 64 + r];
      const float inv_std = (var > 0.0f) ? (1.0f / std::sqrt(var)) : 1.0f;
      model.W[r * 64 + r] = inv_std;
    }
  } else {
    // ZCA whitening: W = V * diag(1/sqrt(lambda)) * V^T
    alignas(64) float A[64 * 64];
    for (int i = 0; i < 64 * 64; ++i) A[i] = Sigma[i];
    alignas(64) float V[64 * 64];
    alignas(64) float lambda[64];
    jacobi_eigendecompose_64(A, V, lambda);
    // Build D^{-1/2}
    alignas(64) float inv_sqrt[64];
    for (int i = 0; i < 64; ++i) inv_sqrt[i] = 1.0f / std::sqrt(lambda[i]);
    // Compute W = V * D^{-1/2} * V^T
    // Temp T = V * D^{-1/2}
    alignas(64) float T[64 * 64];
    for (int r = 0; r < 64; ++r) {
      for (int c = 0; c < 64; ++c) {
        T[r * 64 + c] = V[r * 64 + c] * inv_sqrt[c];
      }
    }
    // W = T * V^T
    for (int r = 0; r < 64; ++r) {
      for (int c = 0; c < 64; ++c) {
        float acc = 0.0f;
        for (int k = 0; k < 64; ++k) acc += T[r * 64 + k] * V[c * 64 + k];
        model.W[r * 64 + c] = acc;
      }
    }
  }

  return model;
}

void apply_whitening(const CapqWhiteningModel& model, const float x[64], float z[64]) noexcept {
  float c[64];
  for (int d = 0; d < 64; ++d) c[d] = x[d] - model.mean[d];
  for (int r = 0; r < 64; ++r) {
    float acc = 0.0f;
    const float* Wr = &model.W[r * 64];
    for (int cidx = 0; cidx < 64; ++cidx) acc += Wr[cidx] * c[cidx];
    z[r] = acc;
  }
}

auto train_q8_params(const float* z_data, std::size_t n, bool symmetric,
                     float clip_percentile) -> std::expected<CapqQ8Params, core::error> {
  if (z_data == nullptr || n == 0) {
    return std::vesper_unexpected(core::error{core::error_code::invalid_argument,
                                       "q8 train: null or empty input",
                                       "index.capq.encode"});
  }
  CapqQ8Params p{};
  p.symmetric = symmetric;
  std::vector<float> vals;
  vals.reserve(n);
  for (int d = 0; d < 64; ++d) {
    vals.clear();
    for (std::size_t i = 0; i < n; ++i) vals.push_back(z_data[i * 64 + d]);
    std::sort(vals.begin(), vals.end());
    const auto clip_idx = static_cast<std::size_t>(clip_percentile * (vals.size() - 1));
    const float lo = vals[std::min<std::size_t>(clip_idx, vals.size() - 1)];
    const float hi = vals[std::max<std::size_t>(vals.size() - 1 - clip_idx, 0)];
    float max_abs = std::max(std::abs(lo), std::abs(hi));
    if (max_abs < 1e-8f) max_abs = 1.0f;
    if (symmetric) {
      p.scale[d] = max_abs / 127.0f;
      p.zero_point[d] = 0.0f;
    } else {
      // Asymmetric: map [lo, hi] to [0, 255]
      const float range = std::max(hi - lo, 1e-6f);
      p.scale[d] = range / 255.0f;
      p.zero_point[d] = -lo / p.scale[d];
    }
  }
  return p;
}

static inline std::uint8_t quantize_q8_scalar_asym(float v, float scale, float zp) {
  const float q = std::round(std::clamp(v / scale + zp, 0.0f, 255.0f));
  return static_cast<std::uint8_t>(q);
}

void encode_capq_payload(const float z[64], const CapqQ8Params& q8p,
                         std::int8_t q8_out[64], std::uint8_t q4_out[32],
                         std::uint8_t& residual_energy_byte) noexcept {
  // q8 encode per dim
  float energy = 0.0f;
  for (int d = 0; d < 64; ++d) {
    const float v = z[d];
    const float scale = q8p.scale[d];
    const float zp = q8p.zero_point[d];
    if (q8p.symmetric) {
      // Signed symmetric quantization in [-127,127]
      const int qs = static_cast<int>(std::round(std::clamp(v / scale, -127.0f, 127.0f)));
      q8_out[d] = static_cast<std::int8_t>(qs);
      const float recon = static_cast<float>(qs) * scale;
      const float err = v - recon;
      energy += err * err;
    } else {
      const std::uint8_t q = quantize_q8_scalar_asym(v, scale, zp);
      // Store biased value as signed int8_t by subtracting 128 for compactness
      q8_out[d] = static_cast<std::int8_t>(static_cast<int>(q) - 128);
      const float recon = (static_cast<int>(q) - zp) * scale;
      const float err = v - recon;
      energy += err * err;
    }
  }
  // Residual energy compression to byte via log-like companding
  const float e = std::min(energy, 1e6f);
  residual_energy_byte = static_cast<std::uint8_t>(std::min(255.0f, std::log1p(e)));

  // q4 as monotone coarsen: right-shift high 4 bits of signed q8 bytes
  coarsen_q8_to_q4(q8_out, q4_out);
}

void coarsen_q8_to_q4(const std::int8_t q8[64], std::uint8_t q4_packed[32]) noexcept {
  for (int i = 0; i < 32; ++i) {
    // Monotone coarsening: arithmetic right shift of signed q8 by 4, then bias to [0,15]
    const int n0 = ((static_cast<int>(q8[2 * i])     >> 4) + 8) & 0x0F; // dim 2*i
    const int n1 = ((static_cast<int>(q8[2 * i + 1]) >> 4) + 8) & 0x0F; // dim 2*i+1
    q4_packed[i] = static_cast<std::uint8_t>((n1 << 4) | n0);
  }
}

} // namespace vesper::index


