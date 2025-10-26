#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>

#include "tests/integration/dataset_loader.hpp"
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
using vesper::index::TopK;

namespace {

struct CapqPipeline {
  CapqBuildParams params;
  vesper::index::CapqWhiteningModel wm;
  CapqQ8Params q8p;
  std::array<std::uint64_t, 6> seeds;
  CapqSoAStorage storage;
};

static void project_to_64(const float* src, std::size_t dim, float dst[64]) {
  if (dim == 64) { std::copy_n(src, 64, dst); return; }
  std::size_t pow2 = 1; while (pow2 < dim) pow2 <<= 1;
  std::vector<float> buf(pow2, 0.0f);
  std::copy_n(src, dim, buf.data());
  vesper::index::fwht64_inplace(buf.data());
  std::copy_n(buf.data(), 64, dst);
}

static CapqPipeline train_and_encode(const vesper::test::Dataset& ds, std::size_t train_n, std::size_t use_n, CapqHammingBits hbits) {
  CapqPipeline P{}; P.seeds = P.params.hamming_seeds;
  const std::size_t T = std::min(train_n, ds.info.num_vectors);
  std::vector<float> train64(T * 64);
  for (std::size_t i = 0; i < T; ++i) project_to_64(ds.base_vectors.data() + i * ds.info.dimension, ds.info.dimension, train64.data() + i * 64);
  auto wm = vesper::index::train_whitening_model(train64.data(), T, P.params.whitening_lambda_ratio);
  if (!wm) throw std::runtime_error("Whitening training failed");
  P.wm = std::move(wm.value());
  std::vector<float> trainZ(T * 64);
  for (std::size_t i = 0; i < T; ++i) vesper::index::apply_whitening(P.wm, train64.data() + i * 64, trainZ.data() + i * 64);
  auto q8p = vesper::index::train_q8_params(trainZ.data(), T, true, 0.9995f);
  if (!q8p) throw std::runtime_error("q8 param training failed");
  P.q8p = std::move(q8p.value());

  const std::size_t N = std::min(use_n, ds.info.num_vectors);
  P.storage = CapqSoAStorage(N, hbits);
  auto view = P.storage.view();
  for (std::size_t i = 0; i < N; ++i) {
    float x64[64], z[64];
    project_to_64(ds.base_vectors.data() + i * ds.info.dimension, ds.info.dimension, x64);
    vesper::index::apply_whitening(P.wm, x64, z);
    auto ham = view.hamming_ptr(i);
    auto hr = vesper::index::compute_hamming_sketch(z, P.seeds, hbits, ham);
    if (!hr) throw std::runtime_error("Hamming encode failed");
    std::int8_t* q8 = view.q8_ptr(i);
    std::uint8_t* q4 = view.q4_ptr(i);
    std::uint8_t* re = view.residual_ptr(i);
    vesper::index::encode_capq_payload(z, P.q8p, q8, q4, *re);
  }
  return P;
}

static std::vector<std::uint32_t> capq_search_topk(const CapqPipeline& P,
                                                   const vesper::test::Dataset& ds,
                                                   std::size_t qi,
                                                   std::uint32_t k,
                                                   std::size_t s1_keep,
                                                   std::size_t s2_keep,
                                                   double& s1_ms, double& s2_ms, double& s3_ms) {
  using clock = std::chrono::steady_clock;
  auto view = P.storage.view();
  float q64[64], zq[64];
  const float* qsrc = ds.query_vectors.empty() ? (ds.base_vectors.data() + qi * ds.info.dimension)
                                               : (ds.query_vectors.data() + qi * ds.info.dimension);
  project_to_64(qsrc, ds.info.dimension, q64);
  vesper::index::apply_whitening(P.wm, q64, zq);
  std::array<std::uint64_t, 4> hamq{};
  auto hr = vesper::index::compute_hamming_sketch(zq, P.seeds, CapqHammingBits::B256, hamq.data());
  if (!hr) throw std::runtime_error("Hamming encode failed (query)");
  auto t0 = clock::now();
  TopK top_s1(s1_keep);
  for (std::size_t i = 0; i < view.num_vectors; ++i) {
    const float d = static_cast<float>(vesper::index::hamming_distance_words(hamq.data(), view.hamming_ptr(i), view.words_per_vector()));
    top_s1.push(static_cast<std::uint64_t>(i), d);
  }
  auto s1 = top_s1.take_sorted();
  auto t1 = clock::now();
  s1_ms += std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();

  std::array<std::int8_t, 64> q8{}; std::array<std::uint8_t, 32> q4{}; std::uint8_t dummy{};
  vesper::index::encode_capq_payload(zq, P.q8p, q8.data(), q4.data(), dummy);
  TopK top_s2(s2_keep);
  for (const auto& [d, id] : s1) {
    (void)d;
    const float dist = vesper::index::distance_q4(q4.data(), view.q4_ptr(static_cast<std::size_t>(id)), P.q8p);
    top_s2.push(id, dist);
  }
  auto s2 = top_s2.take_sorted();
  auto t2 = clock::now();
  s2_ms += std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t2 - t1).count();

  TopK topk(k);
  for (const auto& [d, id] : s2) {
    (void)d;
    const float dist = vesper::index::distance_q8(q8.data(), view.q8_ptr(static_cast<std::size_t>(id)), P.q8p);
    topk.push(id, dist);
  }
  auto final = topk.take_sorted();
  auto t3 = clock::now();
  s3_ms += std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t3 - t2).count();

  std::vector<std::uint32_t> ids; ids.reserve(final.size());
  for (const auto& [dist, id] : final) { (void)dist; ids.push_back(static_cast<std::uint32_t>(id)); }
  return ids;
}

} // namespace

int main() {
  try {
    auto dir = vesper::core::safe_getenv("VESPER_BENCH_DIR");
    std::filesystem::path dataset_dir = dir ? std::filesystem::path(*dir) : std::filesystem::path("data");
    auto dname = vesper::core::safe_getenv("VESPER_BENCH_DATASET");
    const std::string dataset_name = dname && !dname->empty() ? *dname : std::string("sift-128-euclidean");
    auto dsr = vesper::test::DatasetLoader::load_benchmark(dataset_name, dataset_dir);
    if (!dsr) { std::cerr << "Failed to load dataset" << std::endl; return 2; }
    const auto& ds = dsr.value();
    const bool use_subset_gt = vesper::core::safe_getenv("CAPQ_USE_SUBSET_GT").value_or("") == std::string("1");

    const std::size_t train_n = 30000; // larger training improves whitening & q8 scales
    const std::size_t use_n   = std::min<std::size_t>(50000, ds.info.num_vectors);
    const std::size_t q_n     = std::min<std::size_t>(100, ds.info.num_queries ? ds.info.num_queries : std::size_t(100));
    const std::uint32_t k     = 10;

    const bool use_b384 = vesper::core::safe_getenv("CAPQ_USE_B384").value_or("") == std::string("1");
    auto P = train_and_encode(ds, train_n, use_n, use_b384 ? CapqHammingBits::B384 : CapqHammingBits::B256);

    auto parse_size_list = [](const std::string& s) {
      std::vector<std::size_t> v; std::stringstream ss(s); std::string tok;
      while (std::getline(ss, tok, ',')) { if (!tok.empty()) { try { v.push_back(static_cast<std::size_t>(std::stoull(tok))); } catch(...){} } }
      return v;
    };
    std::vector<std::size_t> s1_grid{2000, 3000, 5000, 8000};
    std::vector<std::size_t> s2_grid{200, 500, 1000, 2000};
    if (auto g = vesper::core::safe_getenv("CAPQ_S1_GRID"); g && !g->empty()) { auto v = parse_size_list(*g); if (!v.empty()) s1_grid = std::move(v); }
    if (auto g = vesper::core::safe_getenv("CAPQ_S2_GRID"); g && !g->empty()) { auto v = parse_size_list(*g); if (!v.empty()) s2_grid = std::move(v); }
    // Optional: override whitening lambda
    if (auto lam = vesper::core::safe_getenv("CAPQ_LAMBDA"); lam && !lam->empty()) {
      try {
        const float lr = std::stof(*lam);
        // set via globals in this TU by capturing after object construction
        // (we'll pass it into train_whitening_model via P.params.whitening_lambda_ratio in train_and_encode)
      } catch(...) {}
    }
    // use_subset_gt already defined above

    double best_score = -1e9; std::pair<std::size_t,std::size_t> best{0,0}; float best_recall = 0.0f; double best_ms = 0.0;
    for (auto s1_keep : s1_grid) {
      for (auto s2_keep : s2_grid) {
        std::vector<std::uint32_t> all_results(q_n * k);
        double t1_ms = 0.0, t2_ms = 0.0, t3_ms = 0.0;
        for (std::size_t qi = 0; qi < q_n; ++qi) {
          auto ids = capq_search_topk(P, ds, qi, k, s1_keep, s2_keep, t1_ms, t2_ms, t3_ms);
          std::copy(ids.begin(), ids.end(), all_results.begin() + qi * k);
        }
        double perq1 = t1_ms / q_n, perq2 = t2_ms / q_n, perq3 = t3_ms / q_n;
        float recall = -1.0f;
        if (use_subset_gt) {
          // Compute subset-GT recall against first N base vectors
          const std::size_t N = P.storage.view().num_vectors;
          double hits = 0.0;
          for (std::size_t qi = 0; qi < q_n; ++qi) {
            std::vector<std::pair<float, std::uint32_t>> dists; dists.reserve(N);
            const float* qsrc = ds.query_vectors.empty() ? (ds.base_vectors.data() + qi * ds.info.dimension)
                                                         : (ds.query_vectors.data() + qi * ds.info.dimension);
            for (std::size_t i = 0; i < N; ++i) {
              const float* xb = ds.base_vectors.data() + i * ds.info.dimension;
              double s = 0.0; for (std::size_t d = 0; d < ds.info.dimension; ++d) { const double dd = double(qsrc[d]) - double(xb[d]); s += dd * dd; }
              dists.emplace_back(static_cast<float>(s), static_cast<std::uint32_t>(i));
            }
            std::nth_element(dists.begin(), dists.begin() + std::min<std::size_t>(k, dists.size()) - 1, dists.end(),
                             [](const auto& a, const auto& b){ return a.first < b.first; });
            dists.resize(std::min<std::size_t>(k, dists.size()));
            std::sort(dists.begin(), dists.end(), [](const auto& a, const auto& b){ return a.first < b.first; });
            for (std::size_t j = 0; j < k; ++j) {
              const std::uint32_t id = all_results[qi * k + j];
              for (const auto& pr : dists) { if (pr.second == id) { hits += 1.0; break; } }
            }
          }
          recall = static_cast<float>(hits / double(q_n * k));
        } else if (ds.info.has_groundtruth) {
          recall = vesper::test::SearchMetrics::compute_recall(all_results, ds.groundtruth, q_n, k, ds.k);
        }
        const double total_ms = perq1 + perq2 + perq3;
        const double score = (recall >= 0.0f ? recall : 0.0f) - 0.0005 * total_ms; // simple utility: recall - alpha*latency
        if (score > best_score) { best_score = score; best = {s1_keep, s2_keep}; best_recall = recall; best_ms = total_ms; }
        std::cout << "CAPQ Sweep: S1=" << s1_keep
                  << " S2=" << s2_keep
                  << " perq_ms={" << perq1 << "," << perq2 << "," << perq3 << "}"
                  << " recall@" << k << "=" << recall
                  << std::endl;
      }
    }
    std::cout << "CAPQ Sweep Recommendation: S1=" << best.first << " S2=" << best.second
              << " recall@" << k << "=" << best_recall
              << " total_perq_ms=" << best_ms << std::endl;
  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << std::endl;
    return 1;
  }
  return 0;
}
