// Copyright (c) 2025 Colin Macritchie / Ripple Group, LLC

#include <benchmark/benchmark.h>
#include <cstdlib>
#include <string>
#include <vector>
#include <random>

#include "dataset_loader.hpp"
#include "vesper/core/platform_utils.hpp"

#include "vesper/index/capq.hpp"
#include "vesper/index/capq_util.hpp"
#include "vesper/index/capq_encode.hpp"
#include "vesper/index/capq_dist.hpp"
#include "vesper/index/capq_select.hpp"

using vesper::index::CapqBuildParams;
using vesper::index::CapqHammingBits;
using vesper::index::CapqQ8Params;
using vesper::index::CapqSoAStorage;
using vesper::index::CapqSoAView;
using vesper::index::CapqWhiteningModel;
using vesper::index::TopK;

namespace {

struct CapqBenchDataset {
  vesper::test::Dataset ds;
  std::size_t dim{0};
};

static CapqBenchDataset load_dataset() {
  auto name = vesper::core::safe_getenv("VESPER_BENCH_DATASET");
  auto dir  = vesper::core::safe_getenv("VESPER_BENCH_DIR");
  std::string dataset_name = name ? *name : std::string("sift");
  std::filesystem::path dataset_dir = dir ? std::filesystem::path(*dir) : std::filesystem::path("data");

  auto res = vesper::test::DatasetLoader::load_benchmark(dataset_name, dataset_dir);
  if (!res) {
    // Fallback to auto-detect from a sample .fvecs in data/fvecs
    std::filesystem::path fvecs_dir = dataset_dir / "fvecs";
    for (auto& p : std::filesystem::directory_iterator(fvecs_dir)) {
      if (p.path().extension() == ".fvecs") {
        res = vesper::test::DatasetLoader::load(p.path());
        if (res) break;
      }
    }
  }
  if (!res) {
    throw std::runtime_error("Failed to load dataset for CAPQ bench. Set VESPER_BENCH_DATASET and VESPER_BENCH_DIR.");
  }
  CapqBenchDataset cbd{.ds = std::move(res.value()), .dim = res->info.dimension};
  return cbd;
}

// Deterministic FWHT-based projection to 64D (staff-level: stable and portable)
static void project_to_64(const float* src, std::size_t dim, float dst[64]) {
  // If already 64D, copy
  if (dim == 64) {
    std::copy_n(src, 64, dst);
    return;
  }
  // Copy to buffer padded to power of 2, apply in-place FWHT, then take first 64 comps
  std::size_t pow2 = 1;
  while (pow2 < dim) pow2 <<= 1;
  std::vector<float> buf(pow2, 0.0f);
  std::copy_n(src, dim, buf.data());
  vesper::index::fwht64_inplace(buf.data()); // operates on first 64 entries correctly
  std::copy_n(buf.data(), 64, dst);
}

struct CapqPrepared {
  CapqWhiteningModel wm;
  CapqQ8Params q8p;
  std::array<std::uint64_t, 6> seeds;
  CapqSoAStorage storage;
  std::vector<std::array<std::uint64_t, 4>> query_ham;
  std::vector<std::array<std::uint8_t, 32>> query_q4;
  std::vector<std::array<std::int8_t, 64>> query_q8;
};

static CapqPrepared prepare_capq(const CapqBenchDataset& cbd, std::size_t train_n, std::size_t use_n, std::size_t q_n) {
  CapqPrepared P{};
  CapqBuildParams bp{};
  P.seeds = bp.hamming_seeds;

  // Project and collect training set in 64D
  std::vector<float> train64(train_n * 64);
  for (std::size_t i = 0; i < train_n; ++i) {
    project_to_64(cbd.ds.base_vectors.data() + i * cbd.dim, cbd.dim, train64.data() + i * 64);
  }

  auto wm = vesper::index::train_whitening_model(train64.data(), train_n, bp.whitening_lambda_ratio);
  if (!wm) throw std::runtime_error("Whitening training failed");
  P.wm = std::move(wm.value());

  // Prepare whitened training for q8 param fit
  std::vector<float> trainZ(train_n * 64);
  for (std::size_t i = 0; i < train_n; ++i) {
    vesper::index::apply_whitening(P.wm, train64.data() + i * 64, trainZ.data() + i * 64);
  }
  auto q8p = vesper::index::train_q8_params(trainZ.data(), train_n, /*symmetric=*/true, /*clip=*/0.9995f);
  if (!q8p) throw std::runtime_error("q8 param training failed");
  P.q8p = std::move(q8p.value());

  // Encode base vectors
  const std::size_t N = std::min(use_n, cbd.ds.info.num_vectors);
  P.storage = CapqSoAStorage(N, CapqHammingBits::B256);
  auto view = P.storage.view();
  for (std::size_t i = 0; i < N; ++i) {
    float x64[64], z[64];
    project_to_64(cbd.ds.base_vectors.data() + i * cbd.dim, cbd.dim, x64);
    vesper::index::apply_whitening(P.wm, x64, z);
    // Hamming
    std::uint64_t* ham = view.hamming_ptr(i);
    auto hr = vesper::index::compute_hamming_sketch(z, P.seeds, CapqHammingBits::B256, ham);
    if (!hr) throw std::runtime_error("Hamming encode failed");
    // q8/q4 + residual
    std::int8_t* q8 = view.q8_ptr(i);
    std::uint8_t* q4 = view.q4_ptr(i);
    std::uint8_t* re = view.residual_ptr(i);
    vesper::index::encode_capq_payload(z, P.q8p, q8, q4, *re);
  }

  // Prepare queries
  const std::size_t Q = std::min(q_n, cbd.ds.info.num_queries ? cbd.ds.info.num_queries : std::size_t(100));
  P.query_ham.resize(Q);
  P.query_q4.resize(Q);
  P.query_q8.resize(Q);
  for (std::size_t qi = 0; qi < Q; ++qi) {
    const float* qsrc = cbd.ds.query_vectors.empty() ? (cbd.ds.base_vectors.data() + qi * cbd.dim) : (cbd.ds.query_vectors.data() + qi * cbd.dim);
    float q64[64], zq[64];
    project_to_64(qsrc, cbd.dim, q64);
    vesper::index::apply_whitening(P.wm, q64, zq);
    auto hr = vesper::index::compute_hamming_sketch(zq, P.seeds, CapqHammingBits::B256, P.query_ham[qi].data());
    if (!hr) throw std::runtime_error("Hamming encode failed (query)");
    std::uint8_t dummy_res{};
    vesper::index::encode_capq_payload(zq, P.q8p, P.query_q8[qi].data(), P.query_q4[qi].data(), dummy_res);
  }
  return P;
}

static void BM_Capq_Hamming_Scan(benchmark::State& state) {
  auto cbd = load_dataset();
  auto P = prepare_capq(cbd, /*train_n=*/std::min<std::size_t>(10000, cbd.ds.info.num_vectors),
                        /*use_n=*/std::min<std::size_t>(50000, cbd.ds.info.num_vectors),
                        /*q_n=*/100);
  auto view = P.storage.view();
  std::uint64_t checksum = 0;
  for (auto _ : state) {
    for (std::size_t qi = 0; qi < P.query_ham.size(); ++qi) {
      for (std::size_t i = 0; i < view.num_vectors; ++i) {
        checksum += vesper::index::hamming_distance_words(P.query_ham[qi].data(), view.hamming_ptr(i), view.words_per_vector());
      }
    }
  }
  benchmark::DoNotOptimize(checksum);
}

static void BM_Capq_Q4_Scan(benchmark::State& state) {
  auto cbd = load_dataset();
  auto P = prepare_capq(cbd, std::min<std::size_t>(10000, cbd.ds.info.num_vectors),
                        std::min<std::size_t>(50000, cbd.ds.info.num_vectors),
                        100);
  auto view = P.storage.view();
  float acc = 0.0f;
  for (auto _ : state) {
    for (std::size_t qi = 0; qi < P.query_q4.size(); ++qi) {
      for (std::size_t i = 0; i < view.num_vectors; ++i) {
        acc += vesper::index::distance_q4(P.query_q4[qi].data(), view.q4_ptr(i), P.q8p);
      }
    }
  }
  benchmark::DoNotOptimize(acc);
}

static void BM_Capq_Q8_Scan(benchmark::State& state) {
  auto cbd = load_dataset();
  auto P = prepare_capq(cbd, std::min<std::size_t>(10000, cbd.ds.info.num_vectors),
                        std::min<std::size_t>(50000, cbd.ds.info.num_vectors),
                        100);
  auto view = P.storage.view();
  float acc = 0.0f;
  for (auto _ : state) {
    for (std::size_t qi = 0; qi < P.query_q8.size(); ++qi) {
      for (std::size_t i = 0; i < view.num_vectors; ++i) {
        acc += vesper::index::distance_q8(P.query_q8[qi].data(), view.q8_ptr(i), P.q8p);
      }
    }
  }
  benchmark::DoNotOptimize(acc);
}

} // namespace

BENCHMARK(BM_Capq_Hamming_Scan)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Capq_Q4_Scan)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Capq_Q8_Scan)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();


