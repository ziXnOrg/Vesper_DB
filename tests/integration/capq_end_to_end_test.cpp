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

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <chrono>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <numeric>
#include <iostream>

#include "dataset_loader.hpp"
#include "vesper/core/platform_utils.hpp"

#include "vesper/index/capq.hpp"
#include "vesper/index/capq_util.hpp"
#include "vesper/index/capq_encode.hpp"
#include "vesper/index/capq_dist.hpp"
#include "vesper/index/capq_q4.hpp"
#include "vesper/index/capq_opq.hpp"
#include "vesper/index/capq_select.hpp"
#include "vesper/index/capq_calibration.hpp"

using vesper::index::CapqBuildParams;
using vesper::index::CapqHammingBits;
using vesper::index::CapqQ8Params;
using vesper::index::CapqSoAStorage;
using vesper::index::TopK;

namespace {

struct CapqPipeline {
  CapqBuildParams params;
  vesper::index::CapqWhiteningModel wm;
  CapqQ8Params q8p;
  vesper::index::CapqQ4Codebooks q4c;
  std::array<std::uint64_t, 6> seeds;
  std::array<std::uint64_t, 6> seeds2; // second independent sketch seeds
  CapqSoAStorage storage;
  std::vector<std::uint64_t> hamming_words2; // base hamming for second sketch (size: N * words)
};

static std::size_t env_to_size(const char* name, std::size_t fallback) {
  auto v = vesper::core::safe_getenv(name);
  if (!v || v->empty()) return fallback;
  try {
    return static_cast<std::size_t>(std::stoull(*v));
  } catch (...) {
    return fallback;
  }
}

static float env_to_float(const char* name, float fallback) {
  auto v = vesper::core::safe_getenv(name);
  if (!v || v->empty()) return fallback;
  try {
    return std::stof(*v);
  } catch (...) {
    return fallback;
  }
}

static void project_to_64(const float* src, std::size_t dim, float dst[64]) {
  if (dim == 64) { std::copy_n(src, 64, dst); return; }
  std::size_t pow2 = 1; while (pow2 < dim) pow2 <<= 1;
  std::vector<float> buf(pow2, 0.0f);
  std::copy_n(src, dim, buf.data());
  vesper::index::fwht64_inplace(buf.data());
  std::copy_n(buf.data(), 64, dst);
}

static CapqPipeline train_and_encode(const vesper::test::Dataset& ds, std::size_t train_n, std::size_t use_n, CapqHammingBits hbits) {
  CapqPipeline P{};
  P.seeds = P.params.hamming_seeds;
  // Derive second seed family deterministically
  auto splitmix = [](std::uint64_t& x){ std::uint64_t z = (x += 0x9E3779B97F4A7C15ULL); z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL; return z ^ (z >> 31); };
  {
    std::uint64_t base = 0xA02BDBF7BB3C0A7FULL;
    for (int i = 0; i < 6; ++i) { P.seeds2[i] = splitmix(base); }
  }

  // Training set projection
  const std::size_t T = std::min(train_n, ds.info.num_vectors);
  std::vector<float> train64(T * 64);
  for (std::size_t i = 0; i < T; ++i) {
    project_to_64(ds.base_vectors.data() + i * ds.info.dimension, ds.info.dimension, train64.data() + i * 64);
  }

  // Optional OPQ/ITQ rotation before whitening based on CAPQ_PROJECTION
  auto proj_env = vesper::core::safe_getenv("CAPQ_PROJECTION");
  const bool use_itq = proj_env && (*proj_env == "opq" || *proj_env == "itq" || *proj_env == "OPQ");
  vesper::index::CapqRotationModel rot;
  if (use_itq) {
    auto rr = vesper::index::train_itq_rotation(train64.data(), T, 10);
    REQUIRE(rr.has_value());
    rot = rr.value();
    for (std::size_t i = 0; i < T; ++i) {
      float tmp[64]; vesper::index::apply_rotation(rot, train64.data() + i * 64, tmp);
      std::copy_n(tmp, 64, train64.data() + i * 64);
    }
  }
  auto wm = vesper::index::train_whitening_model(train64.data(), T, P.params.whitening_lambda_ratio);
  REQUIRE(wm.has_value());
  P.wm = std::move(wm.value());

  // Train q8 params on whitened training
  std::vector<float> trainZ(T * 64);
  for (std::size_t i = 0; i < T; ++i) {
    vesper::index::apply_whitening(P.wm, train64.data() + i * 64, trainZ.data() + i * 64);
  }
  auto q8p = vesper::index::train_q8_params(trainZ.data(), T, /*symmetric=*/true, /*clip=*/0.9995f);
  REQUIRE(q8p.has_value());
  P.q8p = std::move(q8p.value());
  // Train q4 codebooks (learned) and weights
  auto q4c = vesper::index::train_q4_codebooks(trainZ.data(), T);
  REQUIRE(q4c.has_value());
  P.q4c = std::move(q4c.value());

  // Encode base
  const std::size_t N = std::min(use_n, ds.info.num_vectors);
  P.storage = CapqSoAStorage(N, hbits);
  auto view = P.storage.view();
  P.hamming_words2.resize(N * view.words_per_vector());
  for (std::size_t i = 0; i < N; ++i) {
    float x64[64], z[64];
    project_to_64(ds.base_vectors.data() + i * ds.info.dimension, ds.info.dimension, x64);
    if (use_itq) {
      float xr[64]; vesper::index::apply_rotation(rot, x64, xr);
      vesper::index::apply_whitening(P.wm, xr, z);
    } else {
      vesper::index::apply_whitening(P.wm, x64, z);
    }
    auto ham = view.hamming_ptr(i);
    auto hr = vesper::index::compute_hamming_sketch(z, P.seeds, hbits, ham);
    REQUIRE(hr.has_value());
    // Second sketch for base
    auto hr2 = vesper::index::compute_hamming_sketch(z, P.seeds2, hbits, P.hamming_words2.data() + i * view.words_per_vector());
    REQUIRE(hr2.has_value());
    std::int8_t* q8 = view.q8_ptr(i);
    std::uint8_t* q4 = view.q4_ptr(i);
    std::uint8_t* re = view.residual_ptr(i);
    // Encode q8 via standard path; q4 via learned codebooks
    vesper::index::encode_capq_payload(z, P.q8p, q8, q4, *re);
    vesper::index::encode_q4_learned(z, P.q4c, q4);
  }

  return P;
}

static std::vector<std::uint32_t> capq_search_topk(const CapqPipeline& P,
                                                   const vesper::test::Dataset& ds,
                                                   std::size_t query_index,
                                                   std::uint32_t k,
                                                   std::size_t stage1_keep,
                                                   std::size_t stage2_keep,
                                                   const vesper::index::IsotonicCalibrator* calib) {
  auto view = P.storage.view();

  // Prepare query encodings
  float q64[64], zq[64];
  const float* qsrc = ds.query_vectors.empty()
                        ? (ds.base_vectors.data() + query_index * ds.info.dimension)
                        : (ds.query_vectors.data() + query_index * ds.info.dimension);
  project_to_64(qsrc, ds.info.dimension, q64);
  vesper::index::apply_whitening(P.wm, q64, zq);
  std::array<std::uint64_t, 4> hamq{};
  auto hr = vesper::index::compute_hamming_sketch(zq, P.seeds, P.storage.hbits(), hamq.data());
  REQUIRE(hr.has_value());
  std::array<std::int8_t, 64> q8{}; std::array<std::uint8_t, 32> q4{}; std::uint8_t dummy{};
  vesper::index::encode_capq_payload(zq, P.q8p, q8.data(), q4.data(), dummy);

  // Stage 1: Hamming prune
  TopK top_s1(stage1_keep);
  for (std::size_t i = 0; i < view.num_vectors; ++i) {
    const auto* hamd = view.hamming_ptr(i);
    const float d = static_cast<float>(vesper::index::hamming_distance_words(hamq.data(), hamd, view.words_per_vector()));
    top_s1.push(static_cast<std::uint64_t>(i), d);
  }
  const auto s1 = top_s1.take_sorted();

  // Stage 2: q4 refine
  TopK top_s2(stage2_keep);
  for (const auto& [d, id] : s1) {
    (void)d;
    const float dist = vesper::index::distance_q4(q4.data(), view.q4_ptr(static_cast<std::size_t>(id)), P.q8p);
    top_s2.push(id, dist);
  }
  const auto s2 = top_s2.take_sorted();

  // Stage 3: q8 final
  TopK topk(k);
  for (const auto& [d, id] : s2) {
    (void)d;
    float dist = vesper::index::distance_q8(q8.data(), view.q8_ptr(static_cast<std::size_t>(id)), P.q8p);
    if (calib && !calib->empty()) dist = calib->map(dist);
    topk.push(id, dist);
  }
  const auto final = topk.take_sorted();
  std::vector<std::uint32_t> ids; ids.reserve(final.size());
  for (const auto& [dist, id] : final) { (void)dist; ids.push_back(static_cast<std::uint32_t>(id)); }
  return ids;
}

} // namespace

TEST_CASE("CAPQ E2E: scalar pipeline recall smoke", "[capq][e2e]") {
  // Dataset config
  auto data_dir = vesper::core::safe_getenv("VESPER_BENCH_DIR");
  std::filesystem::path dir = data_dir ? std::filesystem::path(*data_dir) : std::filesystem::path("data");
  auto dname = vesper::core::safe_getenv("VESPER_BENCH_DATASET");
  const std::string dataset_name = dname && !dname->empty() ? *dname : std::string("sift-128-euclidean");
  auto dsr = vesper::test::DatasetLoader::load_benchmark(dataset_name, dir);
  REQUIRE(dsr.has_value());
  const auto& ds = dsr.value();

  const std::size_t train_n = env_to_size("CAPQ_TRAIN_N", 10000);
  const std::size_t use_n   = env_to_size("CAPQ_USE_N",   50000);
  const std::size_t q_n     = env_to_size("CAPQ_Q_N",     100);
  const std::size_t s1_keep = env_to_size("CAPQ_S1_KEEP", 5000);  // ~90-99% drop
  const std::size_t s2_keep = env_to_size("CAPQ_S2_KEEP", 1000);  // ~10% of s1
  const std::uint32_t k     = static_cast<std::uint32_t>(env_to_size("CAPQ_K", 10));
  const bool rank_with_l2 = vesper::core::safe_getenv("CAPQ_RANK_L2").value_or("") == std::string("1");

  const bool use_b384 = vesper::core::safe_getenv("CAPQ_USE_B384").value_or("") == std::string("1");
  auto P = train_and_encode(ds, train_n, use_n, use_b384 ? CapqHammingBits::B384 : CapqHammingBits::B256);

  // Optional isotonic calibration using true L2 distances against the original space.
  vesper::index::IsotonicCalibrator calib;
  {
    const std::size_t env_calib_q = env_to_size("CAPQ_CALIB_Q", 200);
    const std::size_t calib_Q = std::min<std::size_t>(env_calib_q, q_n);
    std::vector<float> xs; xs.reserve(calib_Q * s2_keep);
    std::vector<float> ys; ys.reserve(calib_Q * s2_keep);
    const std::size_t calib_hn = env_to_size("CAPQ_CALIB_HN", 0); // hard negatives per query
    const std::string hn_src = vesper::core::safe_getenv("CAPQ_CALIB_HN_SRC").value_or("q8");
    const std::size_t folds = std::max<std::size_t>(1, env_to_size("CAPQ_CALIB_FOLDS", 1));
    auto spearman = [](const std::vector<float>& a, const std::vector<float>& b) -> float {
      if (a.size() != b.size() || a.empty()) return 0.0f;
      const std::size_t n = a.size();
      std::vector<std::size_t> idx(n), idy(n);
      std::iota(idx.begin(), idx.end(), 0); std::iota(idy.begin(), idy.end(), 0);
      std::sort(idx.begin(), idx.end(), [&](std::size_t i, std::size_t j){ return a[i] < a[j]; });
      std::sort(idy.begin(), idy.end(), [&](std::size_t i, std::size_t j){ return b[i] < b[j]; });
      std::vector<float> ra(n), rb(n);
      for (std::size_t r = 0; r < n; ++r) ra[idx[r]] = static_cast<float>(r);
      for (std::size_t r = 0; r < n; ++r) rb[idy[r]] = static_cast<float>(r);
      double mean_ra = 0.0, mean_rb = 0.0; for (std::size_t i = 0; i < n; ++i) { mean_ra += ra[i]; mean_rb += rb[i]; }
      mean_ra /= double(n); mean_rb /= double(n);
      double num = 0.0, da = 0.0, db = 0.0;
      for (std::size_t i = 0; i < n; ++i) { const double xa = ra[i] - mean_ra; const double xb = rb[i] - mean_rb; num += xa * xb; da += xa * xa; db += xb * xb; }
      if (da <= 0.0 || db <= 0.0) return 0.0f;
      return static_cast<float>(num / std::sqrt(da * db));
    };
    auto l2_dist = [&](const float* a, const float* b, std::size_t dim) -> float {
      double s = 0.0; for (std::size_t i = 0; i < dim; ++i) { const double d = double(a[i]) - double(b[i]); s += d * d; } return static_cast<float>(s);
    };
    auto view = P.storage.view();
    const std::size_t words = view.words_per_vector();
    const std::size_t calib_s2_keep = env_to_size("CAPQ_CALIB_S2_KEEP", s2_keep * 2);
    for (std::size_t qi = 0; qi < calib_Q; ++qi) {
      float q64[64], zq[64];
      const float* qsrc = ds.query_vectors.empty()
                            ? (ds.base_vectors.data() + qi * ds.info.dimension)
                            : (ds.query_vectors.data() + qi * ds.info.dimension);
      project_to_64(qsrc, ds.info.dimension, q64);
      vesper::index::apply_whitening(P.wm, q64, zq);
      std::vector<std::uint64_t> hamq(words);
      auto hr = vesper::index::compute_hamming_sketch(zq, P.seeds, P.storage.hbits(), hamq.data());
      REQUIRE(hr.has_value());
      TopK top_s1(s1_keep);
      for (std::size_t i = 0; i < view.num_vectors; ++i) {
        const auto* hamd = view.hamming_ptr(i);
        const float d = static_cast<float>(vesper::index::hamming_distance_words(hamq.data(), hamd, words));
        top_s1.push(static_cast<std::uint64_t>(i), d);
      }
      const auto s1 = top_s1.take_sorted();
      std::array<std::int8_t, 64> q8{}; std::array<std::uint8_t, 32> q4{}; std::uint8_t dummy{};
      vesper::index::encode_capq_payload(zq, P.q8p, q8.data(), q4.data(), dummy);
      TopK top_s2(calib_s2_keep);
      for (const auto& [d, id] : s1) {
        (void)d;
        const float dist_q4 = vesper::index::distance_q4(q4.data(), view.q4_ptr(static_cast<std::size_t>(id)), P.q8p);
        top_s2.push(id, dist_q4);
      }
      const auto s2 = top_s2.take_sorted();
      // Build a set for s2 membership to mine hard negatives from s1-s2
      std::unordered_set<std::uint64_t> s2set; s2set.reserve(s2.size());
      for (const auto& pr : s2) s2set.insert(static_cast<std::uint64_t>(pr.second));
      for (const auto& [d_unused, id] : s2) {
        (void)d_unused;
        const float raw_q8 = vesper::index::distance_q8(q8.data(), view.q8_ptr(static_cast<std::size_t>(id)), P.q8p);
        const float true_l2 = l2_dist(qsrc, ds.base_vectors.data() + static_cast<std::size_t>(id) * ds.info.dimension, ds.info.dimension);
        xs.push_back(raw_q8);
        ys.push_back(true_l2);
      }
      // Hard negatives: choose top calib_hn from (s1 \ s2) by q8 or ADE
      if (calib_hn > 0) {
        TopK hn_top(calib_hn);
        for (const auto& [ds1, id1] : s1) {
          (void)ds1;
          if (s2set.find(static_cast<std::uint64_t>(id1)) != s2set.end()) continue;
          float score;
          if (hn_src == "q4") {
            score = vesper::index::distance_q4_ade_dispatch(zq, view.q4_ptr(static_cast<std::size_t>(id1)), P.q4c);
          } else {
            score = vesper::index::distance_q8(q8.data(), view.q8_ptr(static_cast<std::size_t>(id1)), P.q8p);
          }
          hn_top.push(id1, score);
        }
        const auto hns = hn_top.take_sorted();
        for (const auto& [dh, idh] : hns) {
          (void)dh;
          const float raw_q8 = vesper::index::distance_q8(q8.data(), view.q8_ptr(static_cast<std::size_t>(idh)), P.q8p);
          const float true_l2 = l2_dist(qsrc, ds.base_vectors.data() + static_cast<std::size_t>(idh) * ds.info.dimension, ds.info.dimension);
          xs.push_back(raw_q8);
          ys.push_back(true_l2);
        }
      }
    }
    // Cross-validation (log-only): split into folds and compute Spearman on holdout
    if (folds > 1 && xs.size() > folds) {
      const std::size_t n = xs.size();
      const std::size_t fold_sz = std::max<std::size_t>(1, n / folds);
      float avg_spear = 0.0f;
      for (std::size_t f = 0; f < folds; ++f) {
        const std::size_t start = f * fold_sz;
        const std::size_t end = (f + 1 == folds) ? n : std::min(n, start + fold_sz);
        std::vector<float> xtr, ytr, xva, yva; xtr.reserve(n - (end - start)); ytr.reserve(n - (end - start)); xva.reserve(end - start); yva.reserve(end - start);
        for (std::size_t i = 0; i < n; ++i) {
          if (i >= start && i < end) { xva.push_back(xs[i]); yva.push_back(ys[i]); }
          else { xtr.push_back(xs[i]); ytr.push_back(ys[i]); }
        }
        auto cal = vesper::index::fit_isotonic(xtr, ytr);
        if (!cal.empty()) {
          std::vector<float> yhat; yhat.reserve(xva.size());
          for (float xv : xva) yhat.push_back(cal.map(xv));
          avg_spear += spearman(yhat, yva);
        }
      }
      avg_spear /= static_cast<float>(folds);
      INFO("Isotonic CV Spearman (" << folds << " folds) = " << avg_spear);
    }

    calib = vesper::index::fit_isotonic(xs, ys);
    INFO("Isotonic calibration bins: " << calib.breaks.size());
  }

  // Learn ADE per-dimension weights via ridge regression on a bounded subset
  {
    auto view = P.storage.view();
    const std::size_t ade_q = env_to_size("CAPQ_ADE_Q", 100);
    const std::size_t ade_keep = env_to_size("CAPQ_ADE_KEEP", 500);
    const float ade_lambda = env_to_float("CAPQ_ADE_LAMBDA", 1e-3f);
    const std::size_t Q_ade = std::min<std::size_t>(ade_q, q_n);
    const std::size_t rows = std::min<std::size_t>(Q_ade * ade_keep, std::size_t(1) * Q_ade * ade_keep);
    std::vector<float> F; F.resize(rows * 64);
    std::vector<float> y; y.resize(rows);
    std::size_t row = 0;
    for (std::size_t qi = 0; qi < Q_ade && row < rows; ++qi) {
      float q64[64], zq[64];
      const float* qsrc = ds.query_vectors.empty()
                            ? (ds.base_vectors.data() + qi * ds.info.dimension)
                            : (ds.query_vectors.data() + qi * ds.info.dimension);
      project_to_64(qsrc, ds.info.dimension, q64);
      vesper::index::apply_whitening(P.wm, q64, zq);
      // Stage 1
      const std::size_t words = view.words_per_vector();
      std::vector<std::uint64_t> hamq(words);
      auto hr = vesper::index::compute_hamming_sketch(zq, P.seeds, P.storage.hbits(), hamq.data());
      REQUIRE(hr.has_value());
      TopK top_s1(ade_keep * 4); // broader pre-prune for ADE sampling
      for (std::size_t i = 0; i < view.num_vectors; ++i) {
        const auto* hamd = view.hamming_ptr(i);
        const float d = static_cast<float>(vesper::index::hamming_distance_words(hamq.data(), hamd, words));
        top_s1.push(static_cast<std::uint64_t>(i), d);
      }
      const auto s1 = top_s1.take_sorted();
      // Stage 2 shortlist for ADE features
      std::array<std::int8_t, 64> q8{}; std::array<std::uint8_t, 32> q4{}; std::uint8_t dummy{};
      vesper::index::encode_capq_payload(zq, P.q8p, q8.data(), q4.data(), dummy);
      TopK top_s2(ade_keep);
      for (const auto& [d_unused, id] : s1) {
        (void)d_unused;
        const float dist_q4 = vesper::index::distance_q4_ade_dispatch(zq, view.q4_ptr(static_cast<std::size_t>(id)), P.q4c);
        top_s2.push(id, dist_q4);
      }
      const auto s2 = top_s2.take_sorted();
      for (const auto& [d2, id] : s2) {
        (void)d2;
        if (row >= rows) break;
        // Build feature row and target
        float* frow = F.data() + row * 64;
        vesper::index::compute_q4_ade_features(zq, view.q4_ptr(static_cast<std::size_t>(id)), P.q4c, frow);
        // True L2 target in original space
        const float* xb = ds.base_vectors.data() + static_cast<std::size_t>(id) * ds.info.dimension;
        double s = 0.0; for (std::size_t d = 0; d < ds.info.dimension; ++d) { const double dd = double(qsrc[d]) - double(xb[d]); s += dd * dd; }
        y[row] = static_cast<float>(s);
        ++row;
      }
    }
    if (row >= 64) {
      auto wr = vesper::index::fit_ade_weights_ridge(F.data(), row, 64, y.data(), ade_lambda);
      REQUIRE(wr.has_value());
      auto w = wr.value();
      // Clamp to non-negative and lightly normalize to avoid degenerate scaling
      float sumw = 0.0f;
      for (int d = 0; d < 64; ++d) { w[d] = std::max(w[d], 0.0f); sumw += w[d]; }
      const float scale = (sumw > 0.0f) ? (64.0f / sumw) : 1.0f;
      const bool use_learned = vesper::core::safe_getenv("CAPQ_ADE_WEIGHTS_USE").value_or("") == std::string("1");
      if (use_learned) {
        for (int d = 0; d < 64; ++d) P.q4c.weights[d] = w[d] * scale;
        INFO("ADE weights applied (rows=" << row << ", lambda=" << ade_lambda << ")");
      } else {
        INFO("ADE weights learned but not applied (set CAPQ_ADE_WEIGHTS_USE=1 to enable)");
      }
    } else {
      INFO("ADE weight learning skipped: insufficient rows (" << row << ")");
    }
  }

  // Compute recall on a subset (or all available query vectors)
  const std::size_t Q = std::min(q_n, ds.info.num_queries ? ds.info.num_queries : std::size_t(100));
  std::vector<std::uint32_t> all_results(Q * k);
  // Capture S1/S2 candidate ID lists per-query to compute Coverage@k aligned with the pipeline
  std::vector<std::vector<std::uint32_t>> s1_id_lists(Q);
  std::vector<std::vector<std::uint32_t>> s2_id_lists(Q);
  using clock = std::chrono::steady_clock;
  std::chrono::nanoseconds t_s1{0}, t_s2{0}, t_s3{0};
  for (std::size_t qi = 0; qi < Q; ++qi) {
    const auto t0 = clock::now();
    // Stage 1 timing measured inside search by splitting here for fairness
    float q64[64], zq[64];
    const float* qsrc = ds.query_vectors.empty()
                          ? (ds.base_vectors.data() + qi * ds.info.dimension)
                          : (ds.query_vectors.data() + qi * ds.info.dimension);
    project_to_64(qsrc, ds.info.dimension, q64);
    vesper::index::apply_whitening(P.wm, q64, zq);
    const std::size_t words = P.storage.view().words_per_vector();
    std::vector<std::uint64_t> hamq(words);
    auto hr = vesper::index::compute_hamming_sketch(zq, P.seeds, P.storage.hbits(), hamq.data());
    REQUIRE(hr.has_value());

    auto view = P.storage.view();
    // Second sketch for query
    std::vector<std::uint64_t> hamq2(words);
    auto hrq2 = vesper::index::compute_hamming_sketch(zq, P.seeds2, words == 4 ? CapqHammingBits::B256 : CapqHammingBits::B384, hamq2.data());
    REQUIRE(hrq2.has_value());
    // Build two S1 lists, then merge by minimum distance and dedup to s1_keep
    TopK s1_a(s1_keep), s1_b(s1_keep);
    for (std::size_t i = 0; i < view.num_vectors; ++i) {
      const auto* hamd1 = view.hamming_ptr(i);
      const auto* hamd2 = P.hamming_words2.data() + i * words;
      const float d1 = static_cast<float>(vesper::index::hamming_distance_words(hamq.data(), hamd1, words));
      const float d2 = static_cast<float>(vesper::index::hamming_distance_words(hamq2.data(), hamd2, words));
      s1_a.push(static_cast<std::uint64_t>(i), d1);
      s1_b.push(static_cast<std::uint64_t>(i), d2);
    }
    auto a = s1_a.take_sorted();
    auto b = s1_b.take_sorted();
    // Merge by min distance
    std::unordered_map<std::uint64_t, float> id_to_min;
    id_to_min.reserve(a.size() + b.size());
    for (const auto& [da, ida] : a) {
      float cur = static_cast<float>(da);
      auto it = id_to_min.find(ida);
      if (it == id_to_min.end()) id_to_min.emplace(ida, cur);
      else { it->second = (it->second < cur ? it->second : cur); }
    }
    for (const auto& [db, idb] : b) {
      float cur = static_cast<float>(db);
      auto it = id_to_min.find(idb);
      if (it == id_to_min.end()) id_to_min.emplace(idb, cur);
      else { it->second = (it->second < cur ? it->second : cur); }
    }
    // Optional 1-bit multi-probe on first sketch (env CAPQ_S1_PROBE1)
    const std::size_t probe1 = env_to_size("CAPQ_S1_PROBE1", 0);
    if (probe1 > 0) {
      const std::size_t words_per_vec = view.words_per_vector();
      const std::size_t total_bits = words * 64u;
      const std::size_t flips = std::min<std::size_t>(probe1, total_bits);
      // Base pointer for first sketch distances
      for (std::size_t t = 0; t < flips; ++t) {
        const std::size_t bit = t; // simple sequential selection
        const std::size_t widx = bit / 64u;
        const std::size_t bofs = bit % 64u;
        std::vector<std::uint64_t> hamq_flip = hamq; // copy original query sketch for first seed
        hamq_flip[widx] ^= (1ull << bofs);
        for (std::size_t i = 0; i < view.num_vectors; ++i) {
          const auto* hamd1 = view.hamming_ptr(i);
          const float d = static_cast<float>(vesper::index::hamming_distance_words(hamq_flip.data(), hamd1, words_per_vec));
          auto it = id_to_min.find(static_cast<std::uint64_t>(i));
          if (it == id_to_min.end()) id_to_min.emplace(static_cast<std::uint64_t>(i), d);
          else if (d < it->second) it->second = d;
        }
      }
    }
    std::vector<std::pair<float, std::uint64_t>> s1_merged; s1_merged.reserve(id_to_min.size());
    for (const auto& kv : id_to_min) s1_merged.emplace_back(kv.second, kv.first);
    std::nth_element(s1_merged.begin(), s1_merged.begin() + std::min<std::size_t>(s1_keep, s1_merged.size()) - 1, s1_merged.end(),
                     [](const auto& x, const auto& y){ return x.first < y.first; });
    s1_merged.resize(std::min<std::size_t>(s1_keep, s1_merged.size()));
    std::sort(s1_merged.begin(), s1_merged.end(), [](const auto& x, const auto& y){ return x.first < y.first; });
    // Persist S1 IDs for Coverage@k accounting
    {
      std::vector<std::uint32_t> s1_ids_for_qi; s1_ids_for_qi.reserve(s1_merged.size());
      for (const auto& pr : s1_merged) { s1_ids_for_qi.push_back(static_cast<std::uint32_t>(pr.second)); }
      s1_id_lists[qi] = std::move(s1_ids_for_qi);
    }
    const auto t1 = clock::now();
    t_s1 += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0);

    // Stage 2 timing
    std::array<std::int8_t, 64> q8{}; std::array<std::uint8_t, 32> q4{}; std::uint8_t dummy{};
    vesper::index::encode_capq_payload(zq, P.q8p, q8.data(), q4.data(), dummy);
    TopK top_s2(s2_keep);
    for (const auto& pr : s1_merged) {
      const auto id = pr.second;
      const float dist = vesper::index::distance_q4_ade_dispatch(zq, view.q4_ptr(static_cast<std::size_t>(id)), P.q4c);
      top_s2.push(id, dist);
    }
    const auto s2 = top_s2.take_sorted();
    // Persist S2 IDs for Coverage@k accounting
    {
      std::vector<std::uint32_t> s2_ids_for_qi; s2_ids_for_qi.reserve(s2.size());
      for (const auto& pr : s2) { s2_ids_for_qi.push_back(static_cast<std::uint32_t>(pr.second)); }
      s2_id_lists[qi] = std::move(s2_ids_for_qi);
    }
    const auto t2 = clock::now();
    t_s2 += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1);

    // Stage 3 timing
    TopK topk(k);
    for (const auto& [d, id] : s2) {
      (void)d;
      float dist;
      if (rank_with_l2) {
        const float* xb = ds.base_vectors.data() + static_cast<std::size_t>(id) * ds.info.dimension;
        double s = 0.0; for (std::size_t i = 0; i < ds.info.dimension; ++i) { const double dd = double(qsrc[i]) - double(xb[i]); s += dd * dd; }
        dist = static_cast<float>(s);
      } else {
        dist = vesper::index::distance_q8(q8.data(), view.q8_ptr(static_cast<std::size_t>(id)), P.q8p);
        if (!calib.empty()) dist = calib.map(dist);
      }
      topk.push(id, dist);
    }
    const auto final = topk.take_sorted();
    const auto t3 = clock::now();
    t_s3 += std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2);

    std::vector<std::uint32_t> ids; ids.reserve(final.size());
    for (const auto& [dist, id] : final) { (void)dist; ids.push_back(static_cast<std::uint32_t>(id)); }
    REQUIRE(ids.size() == k);
    std::copy(ids.begin(), ids.end(), all_results.begin() + qi * k);
  }

  const bool use_subset_gt = vesper::core::safe_getenv("CAPQ_USE_SUBSET_GT").value_or("") == std::string("1");
  if (use_subset_gt) {
    const std::size_t N = P.storage.view().num_vectors;
    double hits = 0.0;
    double s1_hits = 0.0, s2_hits = 0.0;
    for (std::size_t qi = 0; qi < Q; ++qi) {
      // Build subset-GT top-k via brute-force L2 on first N base vectors
      float q64[64];
      const float* qsrc = ds.query_vectors.empty()
                            ? (ds.base_vectors.data() + qi * ds.info.dimension)
                            : (ds.query_vectors.data() + qi * ds.info.dimension);
      std::vector<std::pair<float, std::uint32_t>> dists; dists.reserve(N);
      for (std::size_t i = 0; i < N; ++i) {
        const float* xb = ds.base_vectors.data() + i * ds.info.dimension;
        double s = 0.0; for (std::size_t d = 0; d < ds.info.dimension; ++d) { const double dd = double(qsrc[d]) - double(xb[d]); s += dd * dd; }
        dists.emplace_back(static_cast<float>(s), static_cast<std::uint32_t>(i));
      }
      std::nth_element(dists.begin(), dists.begin() + std::min<std::size_t>(k, dists.size()) - 1, dists.end(),
                       [](const auto& a, const auto& b){ return a.first < b.first; });
      dists.resize(std::min<std::size_t>(k, dists.size()));
      std::sort(dists.begin(), dists.end(), [](const auto& a, const auto& b){ return a.first < b.first; });
      // Build sets for overlap accounting
      std::vector<std::uint32_t> subset_gt_ids; subset_gt_ids.reserve(dists.size());
      for (const auto& pr : dists) subset_gt_ids.push_back(pr.second);

      // Count overlap with final results
      for (std::size_t j = 0; j < k; ++j) {
        const std::uint32_t id = all_results[qi * k + j];
        for (const auto& pr : dists) { if (pr.second == id) { hits += 1.0; break; } }
      }

      // Compute S1/S2 Coverage@k using lists captured during main search loop (dual-sketch + probe, ADE distances)
      {
        const auto& s1_ids_for_qi = s1_id_lists[qi];
        const auto& s2_ids_for_qi = s2_id_lists[qi];
        std::vector<std::uint32_t> s1_sorted = s1_ids_for_qi;
        std::vector<std::uint32_t> s2_sorted = s2_ids_for_qi;
        std::sort(s1_sorted.begin(), s1_sorted.end());
        std::sort(s2_sorted.begin(), s2_sorted.end());
        auto contains_sorted = [](const std::vector<std::uint32_t>& v, std::uint32_t id){
          return std::binary_search(v.begin(), v.end(), id);
        };
        for (std::uint32_t gt : subset_gt_ids) {
          if (contains_sorted(s1_sorted, gt)) s1_hits += 1.0;
          if (contains_sorted(s2_sorted, gt)) s2_hits += 1.0;
        }
      }
    }
    const float recall = static_cast<float>(hits / double(Q * k));
    const float recall_min = env_to_float("CAPQ_RECALL_MIN", 0.03f);
    REQUIRE(recall >= recall_min);
    INFO("CAPQ SubsetGT Recall@" << k << " = " << recall << ", min=" << recall_min);
    const float s1_cov = static_cast<float>(s1_hits / double(Q * k));
    const float s2_cov = static_cast<float>(s2_hits / double(Q * k));
    INFO("Coverage@" << k << ": S1=" << s1_cov << ", S2=" << s2_cov);
    // Force-print key metrics to stdout for automated parsing
    std::cout << "CAPQ Metrics: recall@" << k << "=" << recall
              << ", coverage@" << k << ": S1=" << s1_cov << ", S2=" << s2_cov
              << "\n";
  } else if (ds.info.has_groundtruth) {
    const float recall = vesper::test::SearchMetrics::compute_recall(
      all_results, ds.groundtruth, Q, k, ds.k);
    const float recall_min = env_to_float("CAPQ_RECALL_MIN", 0.03f);
    REQUIRE(recall >= recall_min);
    INFO("CAPQ Recall@" << k << " = " << recall << ", min=" << recall_min);
  }

  const double s1_ms = static_cast<double>(t_s1.count()) / 1e6;
  const double s2_ms = static_cast<double>(t_s2.count()) / 1e6;
  const double s3_ms = static_cast<double>(t_s3.count()) / 1e6;
  const std::size_t N = P.storage.view().num_vectors;
  const double s1_drop = N ? (1.0 - static_cast<double>(s1_keep) / static_cast<double>(N)) : 0.0;
  const double s2_drop = s1_keep ? (1.0 - static_cast<double>(s2_keep) / static_cast<double>(s1_keep)) : 0.0;
  INFO("CAPQ Telemetry: S1_keep=" << s1_keep << "/" << N
       << " (drop=" << s1_drop << ")"
       << ", S2_keep=" << s2_keep << "/" << s1_keep
       << " (drop=" << s2_drop << ")"
       << ", timings(ms): S1=" << s1_ms << ", S2=" << s2_ms << ", S3=" << s3_ms
       << ", per-query(ms): S1=" << s1_ms / Q << ", S2=" << s2_ms / Q << ", S3=" << s3_ms / Q);
}


