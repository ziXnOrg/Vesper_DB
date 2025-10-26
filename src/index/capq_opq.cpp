/**
 * Copyright (c) 2025 Colin Macritchie / Ripple Group, LLC
 */

#include "vesper/index/capq_opq.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace vesper::index {

// Newton–Schulz iteration for inverse square root of SPD matrix S (64x64)
static void inv_sqrt_spd_64(const float* S, float* S_invsqrt) {
  // Initialize X0 = I, Y0 = S / trace(S)
  std::vector<float> Y(64 * 64), Z(64 * 64), I(64 * 64);
  for (int i = 0; i < 64 * 64; ++i) { I[i] = 0.0f; }
  for (int i = 0; i < 64; ++i) I[i * 64 + i] = 1.0f;
  double tr = 0.0; for (int i = 0; i < 64; ++i) tr += S[i * 64 + i];
  const float alpha = (tr > 0.0) ? static_cast<float>(64.0 / tr) : 1.0f;
  for (int i = 0; i < 64 * 64; ++i) Y[i] = S[i] * alpha;
  // Xk starts as identity
  std::vector<float> X(64 * 64); X = I;
  const int iters = 10;
  auto matmul = [](const float* A, const float* B, float* C){
    for (int r = 0; r < 64; ++r) {
      for (int c = 0; c < 64; ++c) {
        double acc = 0.0; for (int k = 0; k < 64; ++k) acc += double(A[r * 64 + k]) * double(B[k * 64 + c]);
        C[r * 64 + c] = static_cast<float>(acc);
      }
    }
  };
  for (int it = 0; it < iters; ++it) {
    // Z = 0.5 * (3I - Y)
    for (int i = 0; i < 64 * 64; ++i) Z[i] = 0.5f * (3.0f * I[i] - Y[i]);
    // X = X * Z; Y = Z * Y
    std::vector<float> Xnext(64 * 64), Ynext(64 * 64);
    matmul(X.data(), Z.data(), Xnext.data());
    matmul(Z.data(), Y.data(), Ynext.data());
    X.swap(Xnext); Y.swap(Ynext);
  }
  // Now X ≈ S^{-1/2} scaled
  for (int i = 0; i < 64 * 64; ++i) S_invsqrt[i] = X[i] * std::sqrt(alpha);
}

auto train_itq_rotation(const float* z_data, std::size_t n, int iters)
    -> std::expected<CapqRotationModel, core::error> {
  if (z_data == nullptr || n == 0) {
    return std::vesper_unexpected(core::error{core::error_code::invalid_argument,
                                       "itq train: null/empty input",
                                       "index.capq.opq"});
  }
  CapqRotationModel model{};
  // Initialize R to identity
  for (int i = 0; i < 64 * 64; ++i) model.R[i] = 0.0f;
  for (int i = 0; i < 64; ++i) model.R[i * 64 + i] = 1.0f;

  std::vector<float> B(n * 64);
  for (int it = 0; it < iters; ++it) {
    // 1) Assign binary codes B = sign(Z * R^T)
    for (std::size_t i = 0; i < n; ++i) {
      for (int r = 0; r < 64; ++r) {
        double acc = 0.0; const float* Ri = &model.R[r * 64];
        for (int c = 0; c < 64; ++c) acc += double(z_data[i * 64 + c]) * double(Ri[c]);
        B[i * 64 + r] = (acc >= 0.0) ? 1.0f : -1.0f;
      }
    }
    // 2) Solve Procrustes: maximize Tr(R^T Z^T B) subject to R^T R = I
    // M = B^T Z  (64x64)
    std::vector<float> M(64 * 64);
    for (int r = 0; r < 64; ++r) {
      for (int c = 0; c < 64; ++c) {
        double acc = 0.0; for (std::size_t i = 0; i < n; ++i) acc += double(B[i * 64 + r]) * double(z_data[i * 64 + c]);
        M[r * 64 + c] = static_cast<float>(acc);
      }
    }
    // Polar decomposition via M = QH, Q = M (M^T M)^{-1/2}
    std::vector<float> MtM(64 * 64), invsqrt(64 * 64), Q(64 * 64);
    // MtM = M^T M
    for (int r = 0; r < 64; ++r) {
      for (int c = 0; c < 64; ++c) {
        double acc = 0.0; for (int k = 0; k < 64; ++k) acc += double(M[k * 64 + r]) * double(M[k * 64 + c]);
        MtM[r * 64 + c] = static_cast<float>(acc);
      }
    }
    inv_sqrt_spd_64(MtM.data(), invsqrt.data());
    // Q = M * invsqrt(M^T M)
    for (int r = 0; r < 64; ++r) {
      for (int c = 0; c < 64; ++c) {
        double acc = 0.0; for (int k = 0; k < 64; ++k) acc += double(M[r * 64 + k]) * double(invsqrt[k * 64 + c]);
        Q[r * 64 + c] = static_cast<float>(acc);
      }
    }
    // Update R = Q
    for (int i = 0; i < 64 * 64; ++i) model.R[i] = Q[i];
  }
  return model;
}

} // namespace vesper::index


