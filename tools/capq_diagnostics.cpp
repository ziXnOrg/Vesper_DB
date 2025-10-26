#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <unordered_set>
#include <string>
#include <sstream>
#include <vector>

#include "tests/integration/dataset_loader.hpp"
#include "vesper/core/platform_utils.hpp"

#include "vesper/index/capq.hpp"
#include "vesper/index/capq_util.hpp"
#include "vesper/index/capq_encode.hpp"
#include "vesper/index/capq_dist.hpp"
#include "vesper/index/capq_select.hpp"

namespace {

struct Corr { double pearson{0}, spearman{0}; };

static Corr corr_stats(const std::vector<double>& x, const std::vector<double>& y) {
  const size_t n = x.size();
  if (n == 0 || y.size() != n) return {};
  double mx = std::accumulate(x.begin(), x.end(), 0.0) / n;
  double my = std::accumulate(y.begin(), y.end(), 0.0) / n;
  double num = 0.0, dx2 = 0.0, dy2 = 0.0;
  for (size_t i = 0; i < n; ++i) { const double dx = x[i] - mx, dy = y[i] - my; num += dx * dy; dx2 += dx * dx; dy2 += dy * dy; }
  double pearson = (dx2 > 0 && dy2 > 0) ? (num / std::sqrt(dx2 * dy2)) : 0.0;
  // Spearman: rank both and compute Pearson of ranks
  std::vector<size_t> rx(n), ry(n);
  std::iota(rx.begin(), rx.end(), 0); std::iota(ry.begin(), ry.end(), 0);
  std::stable_sort(rx.begin(), rx.end(), [&](size_t a, size_t b){ return x[a] < x[b]; });
  std::stable_sort(ry.begin(), ry.end(), [&](size_t a, size_t b){ return y[a] < y[b]; });
  std::vector<double> xr(n), yr(n);
  for (size_t r = 0; r < n; ++r) { xr[rx[r]] = static_cast<double>(r); yr[ry[r]] = static_cast<double>(r); }
  double mxr = (n - 1) * 0.5, myr = mxr; num = dx2 = dy2 = 0.0;
  for (size_t i = 0; i < n; ++i) { const double dx = xr[i] - mxr, dy = yr[i] - myr; num += dx * dy; dx2 += dx * dx; dy2 += dy * dy; }
  double spearman = (dx2 > 0 && dy2 > 0) ? (num / std::sqrt(dx2 * dy2)) : 0.0;
  return {pearson, spearman};
}

static double auc_score(const std::vector<double>& scores, const std::vector<int>& labels) {
  // AUC via rank-sum (Mann–Whitney U)
  const size_t n = scores.size();
  std::vector<size_t> idx(n); std::iota(idx.begin(), idx.end(), 0);
  std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b){ return scores[a] < scores[b]; });
  size_t pos = 0, neg = 0; double rank_sum = 0.0;
  for (size_t r = 0; r < n; ++r) {
    size_t i = idx[r];
    if (labels[i] > 0) { ++pos; rank_sum += static_cast<double>(r + 1); } else { ++neg; }
  }
  if (pos == 0 || neg == 0) return 0.5;
  double U = rank_sum - static_cast<double>(pos) * (pos + 1) / 2.0;
  return U / (static_cast<double>(pos) * static_cast<double>(neg));
}

// SplitMix64
static inline std::uint64_t splitmix64(std::uint64_t& x) {
  std::uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

static std::array<std::uint64_t, 6> make_seeds(std::uint64_t base, int idx) {
  std::array<std::uint64_t, 6> s{};
  std::uint64_t x = base + static_cast<std::uint64_t>(idx) * 0x9E3779B185EBCA87ULL;
  for (int i = 0; i < 6; ++i) s[i] = splitmix64(x);
  return s;
}

static std::vector<float> parse_float_list(const std::string& s) {
  std::vector<float> vals; std::stringstream ss(s); std::string tok;
  while (std::getline(ss, tok, ',')) { if (!tok.empty()) vals.push_back(static_cast<float>(std::stod(tok))); }
  return vals;
}

static std::vector<std::string> parse_mode_list(const std::string& s) {
  std::vector<std::string> vals; std::stringstream ss(s); std::string tok;
  while (std::getline(ss, tok, ',')) { if (!tok.empty()) vals.push_back(tok); }
  return vals;
}

#if defined(_WIN32)
#  include <cstdlib>
static inline void set_env_var(const char* name, const char* value) { _putenv_s(name, value); }
#else
#  include <cstdlib>
static inline void set_env_var(const char* name, const char* value) { setenv(name, value, 1); }
#endif

struct JsonBuilder {
  std::ostream& os;
  bool first{true};
  void begin() { os << "{"; first = true; }
  void end() { os << "}\n"; }
  void kv(const char* key, double v) { if (!first) os << ","; first = false; os << "\n  \"" << key << "\": " << v; }
  void kvs(const char* key, const std::string& s) { if (!first) os << ","; first = false; os << "\n  \"" << key << "\": \"" << s << "\""; }
  void begin_obj(const char* key) { if (!first) os << ","; first = false; os << "\n  \"" << key << "\": {"; first = true; }
  void end_obj() { os << "\n  }"; first = false; }
};

} // namespace

int main() {
  using namespace vesper::index;
  try {
    auto dir = vesper::core::safe_getenv("VESPER_BENCH_DIR");
    std::filesystem::path dataset_dir = dir ? std::filesystem::path(*dir) : std::filesystem::path("data");
    auto dname = vesper::core::safe_getenv("VESPER_BENCH_DATASET");
    const std::string dataset_name = dname && !dname->empty() ? *dname : std::string("sift-128-euclidean");
    auto dsr = vesper::test::DatasetLoader::load_benchmark(dataset_name, dataset_dir);
    if (!dsr) { std::cerr << "Failed to load dataset" << std::endl; return 2; }
    const auto& ds = dsr.value();

    // Config (env overrides)
    const auto env_train = vesper::core::safe_getenv("CAPQ_TRAIN_N");
    const auto env_use   = vesper::core::safe_getenv("CAPQ_USE_N");
    const auto env_calq  = vesper::core::safe_getenv("CAPQ_CALIB_Q");
    std::size_t train_n = env_train ? std::min<std::size_t>(std::stoull(*env_train), ds.info.num_vectors)
                                    : std::min<std::size_t>(std::max<std::size_t>(30000, ds.info.num_vectors / 4), ds.info.num_vectors);
    std::size_t use_n   = env_use   ? std::min<std::size_t>(std::stoull(*env_use), ds.info.num_vectors)
                                    : std::min<std::size_t>(50000, ds.info.num_vectors);
    std::size_t calib_Q = env_calq  ? std::min<std::size_t>(std::stoull(*env_calq), (ds.info.num_queries ? ds.info.num_queries : size_t(1000)))
                                    : std::min<std::size_t>(1000, ds.info.num_queries ? ds.info.num_queries : size_t(1000));
    const bool use_b384 = vesper::core::safe_getenv("CAPQ_USE_B384").value_or("") == std::string("1");
    const auto env_sample = vesper::core::safe_getenv("CAPQ_SAMPLE_N");
    const std::size_t sample_cap = env_sample ? std::stoull(*env_sample) : 2000ULL;

    // Prepare training projection to 64D for all train vectors
    std::vector<float> train64(train_n * 64);
    for (std::size_t i = 0; i < train_n; ++i) {
      const float* x = ds.base_vectors.data() + i * ds.info.dimension;
      std::size_t pow2 = 1; while (pow2 < ds.info.dimension) pow2 <<= 1;
      std::vector<float> buf(pow2, 0.0f);
      std::copy_n(x, ds.info.dimension, buf.data());
      fwht64_inplace(buf.data());
      std::copy_n(buf.data(), 64, train64.data() + i * 64);
    }

    // Diagnostics over calib_Q queries
    std::vector<double> s1_dists; std::vector<double> s2_dists; std::vector<double> s3_dists; std::vector<double> true_l2;
    std::vector<int> s1_labels; s1_dists.reserve(calib_Q * 1000); s1_labels.reserve(calib_Q * 1000);

    const std::size_t seed_trials = std::max<std::size_t>(1, std::stoull(vesper::core::safe_getenv("CAPQ_SEED_SWEEP_N").value_or("1")));
    const std::uint64_t seed_base = std::stoull(vesper::core::safe_getenv("CAPQ_SEED_BASE").value_or("1337"));

    // Lambda and whitening sweeps
    const auto lambda_env = vesper::core::safe_getenv("CAPQ_LAMBDA_SWEEP").value_or("0.0005,0.001,0.002");
    const auto modes_env = vesper::core::safe_getenv("CAPQ_WHITENING_SWEEP").value_or("diag,zca");
    const auto lambdas = parse_float_list(lambda_env);
    const auto modes = parse_mode_list(modes_env);

    struct BestSummary { double s1_auc{0.0}; Corr s1{}; Corr s2{}; Corr s3{}; std::array<std::uint64_t,6> seeds{}; std::string mode{"diag"}; float lambda{1e-3f}; } best;

    // Allocate store once; re-use across candidates
    CapqSoAStorage store(use_n, use_b384 ? CapqHammingBits::B384 : CapqHammingBits::B256);
    auto view = store.view();
    std::vector<float> baseZ(use_n * 64);

    for (const auto& mode : modes) {
      set_env_var("CAPQ_WHITENING", mode.c_str());
      for (float lambda_ratio : lambdas) {
        // Train whitening and q8 for this candidate
        auto wm = train_whitening_model(train64.data(), train_n, lambda_ratio);
        if (!wm) { continue; }
        std::vector<float> trainZ(train_n * 64);
        for (std::size_t i = 0; i < train_n; ++i) apply_whitening(wm.value(), train64.data() + i * 64, trainZ.data() + i * 64);
        auto q8p = train_q8_params(trainZ.data(), train_n, /*symmetric*/true, 0.9995f);
        if (!q8p) { continue; }

        // Compute baseZ and encode q8/q4 for base
        for (std::size_t i = 0; i < use_n; ++i) {
          const float* xb = ds.base_vectors.data() + i * ds.info.dimension;
          std::size_t pow2 = 1; while (pow2 < ds.info.dimension) pow2 <<= 1;
          std::vector<float> buf(pow2, 0.0f);
          std::copy_n(xb, ds.info.dimension, buf.data());
          fwht64_inplace(buf.data());
          float* z = baseZ.data() + i * 64;
          apply_whitening(wm.value(), buf.data(), z);
          std::int8_t* q8 = view.q8_ptr(i);
          std::uint8_t* q4 = view.q4_ptr(i);
          std::uint8_t* re = view.residual_ptr(i);
          encode_capq_payload(z, q8p.value(), q8, q4, *re);
        }

        // Seed sweep for this candidate; recompute base hamming per seed
        Corr best_c1{}; double best_auc = 0.0; std::array<std::uint64_t, 6> best_seeds = CapqBuildParams{}.hamming_seeds;
        for (std::size_t trial = 0; trial < seed_trials; ++trial) {
          auto seeds = make_seeds(seed_base, static_cast<int>(trial));
          // recompute base hamming with these seeds
          for (std::size_t i = 0; i < use_n; ++i) {
            const float* z = baseZ.data() + i * 64;
            compute_hamming_sketch(z, seeds, store.hbits(), view.hamming_ptr(i));
          }
          std::vector<double> s1d; std::vector<double> tl2; std::vector<int> lbls;
          s1d.reserve(calib_Q * 1000); tl2.reserve(calib_Q * 1000); lbls.reserve(calib_Q * 1000);
          for (std::size_t qi = 0; qi < calib_Q; ++qi) {
            const float* qsrc = ds.query_vectors.empty() ? (ds.base_vectors.data() + qi * ds.info.dimension)
                                                         : (ds.query_vectors.data() + qi * ds.info.dimension);
            std::size_t pow2 = 1; while (pow2 < ds.info.dimension) pow2 <<= 1;
            std::vector<float> buf(pow2, 0.0f);
            std::copy_n(qsrc, ds.info.dimension, buf.data());
            fwht64_inplace(buf.data());
            float zq[64]; apply_whitening(wm.value(), buf.data(), zq);
            std::vector<std::uint64_t> hamq(view.words_per_vector());
            compute_hamming_sketch(zq, seeds, store.hbits(), hamq.data());
            const std::size_t sample = std::min<std::size_t>(sample_cap, use_n);
            for (std::size_t i = 0; i < sample; ++i) {
              double d1 = static_cast<double>(vesper::index::hamming_distance_words(hamq.data(), view.hamming_ptr(i), view.words_per_vector()));
              const float* xb = ds.base_vectors.data() + i * ds.info.dimension;
              double tl = 0.0; for (std::size_t d = 0; d < ds.info.dimension; ++d) { double dx = double(qsrc[d]) - double(xb[d]); tl += dx * dx; }
              s1d.push_back(d1); tl2.push_back(tl);
              int l = 0; if (ds.info.has_groundtruth) {
                const std::uint32_t* gt = ds.groundtruth.data() + static_cast<std::size_t>(qi) * static_cast<std::size_t>(ds.k);
                for (std::size_t g = 0; g < static_cast<std::size_t>(ds.k); ++g) if (gt[g] == static_cast<std::uint32_t>(i)) { l = 1; break; }
              }
              lbls.push_back(l);
            }
          }
          Corr c1 = corr_stats(s1d, tl2);
          std::vector<double> scores(s1d.size()); for (size_t i = 0; i < s1d.size(); ++i) scores[i] = -s1d[i];
          double auc = auc_score(scores, lbls);
          if (auc > best_auc) { best_auc = auc; best_c1 = c1; best_seeds = seeds; }
        }

        // With best seeds, compute full diagnostics (including s2/s3) on a subset
        std::array<std::uint64_t, 6> seeds = best_seeds;
        s1_dists.clear(); s2_dists.clear(); s3_dists.clear(); true_l2.clear(); s1_labels.clear();
        s1_dists.reserve(calib_Q * 1000); s1_labels.reserve(calib_Q * 1000);
        // FN@k coverage bookkeeping
        double s1_cov_hits = 0.0, s2_cov_hits = 0.0; std::size_t gt_total = 0; const std::size_t k = std::min<std::size_t>(static_cast<std::size_t>(ds.k), std::size_t(10));
        for (std::size_t qi = 0; qi < calib_Q; ++qi) {
          const float* qsrc = ds.query_vectors.empty() ? (ds.base_vectors.data() + qi * ds.info.dimension)
                                                       : (ds.query_vectors.data() + qi * ds.info.dimension);
          std::size_t pow2 = 1; while (pow2 < ds.info.dimension) pow2 <<= 1;
          std::vector<float> buf(pow2, 0.0f);
          std::copy_n(qsrc, ds.info.dimension, buf.data());
          fwht64_inplace(buf.data());
          float zq[64]; apply_whitening(wm.value(), buf.data(), zq);
          std::vector<std::uint64_t> hamq(view.words_per_vector());
          compute_hamming_sketch(zq, seeds, store.hbits(), hamq.data());
          std::array<std::int8_t, 64> q8q{}; std::array<std::uint8_t, 32> q4q{}; std::uint8_t dummy{};
          encode_capq_payload(zq, q8p.value(), q8q.data(), q4q.data(), dummy);
          const std::size_t sample = std::min<std::size_t>(sample_cap, use_n);
          // S1 top M and S2 top R for FN coverage; choose modest defaults
          const std::size_t s1_keep = std::min<std::size_t>(5000, sample);
          const std::size_t s2_keep = std::min<std::size_t>(1000, s1_keep);
          vesper::index::TopK top_s1(s1_keep);
          for (std::size_t i = 0; i < sample; ++i) {
            const auto* hamd = view.hamming_ptr(i);
            float d = static_cast<float>(vesper::index::hamming_distance_words(hamq.data(), hamd, view.words_per_vector()));
            top_s1.push(i, d);
          }
          const auto s1c = top_s1.take_sorted();
          std::unordered_set<std::uint32_t> s1set; s1set.reserve(s1c.size());
          for (const auto& p : s1c) s1set.insert(static_cast<std::uint32_t>(p.second));
          vesper::index::TopK top_s2(s2_keep);
          for (const auto& [d_unused, id] : s1c) {
            (void)d_unused;
            float d = vesper::index::distance_q4(q4q.data(), view.q4_ptr(static_cast<std::size_t>(id)), q8p.value());
            top_s2.push(id, d);
          }
          const auto s2c = top_s2.take_sorted();
          std::unordered_set<std::uint32_t> s2set; s2set.reserve(s2c.size());
          for (const auto& p : s2c) s2set.insert(static_cast<std::uint32_t>(p.second));

          for (std::size_t i = 0; i < sample; ++i) {
            const auto* hamd = view.hamming_ptr(i);
            double d1 = static_cast<double>(vesper::index::hamming_distance_words(hamq.data(), hamd, view.words_per_vector()));
            double d2 = static_cast<double>(vesper::index::distance_q4(q4q.data(), view.q4_ptr(i), q8p.value()));
            double d3 = static_cast<double>(vesper::index::distance_q8(q8q.data(), view.q8_ptr(i), q8p.value()));
            const float* xb = ds.base_vectors.data() + i * ds.info.dimension;
            double tl2 = 0.0; for (std::size_t d = 0; d < ds.info.dimension; ++d) { double dx = double(qsrc[d]) - double(xb[d]); tl2 += dx * dx; }
            s1_dists.push_back(d1); s2_dists.push_back(d2); s3_dists.push_back(d3); true_l2.push_back(tl2);
            int lbl = 0;
            if (ds.info.has_groundtruth) {
              const std::uint32_t* gt = ds.groundtruth.data() + static_cast<std::size_t>(qi) * static_cast<std::size_t>(ds.k);
              for (std::size_t g = 0; g < std::min<std::size_t>(k, static_cast<std::size_t>(ds.k)); ++g) if (gt[g] == static_cast<std::uint32_t>(i)) { lbl = 1; break; }
            }
            s1_labels.push_back(lbl);
          }
          if (ds.info.has_groundtruth) {
            // Compute coverage of GT@k by S1/S2 candidate sets (restricted to sample domain)
            const std::uint32_t* gt = ds.groundtruth.data() + static_cast<std::size_t>(qi) * static_cast<std::size_t>(ds.k);
            std::vector<std::uint32_t> gt_ids; gt_ids.reserve(k);
            for (std::size_t g = 0; g < std::min<std::size_t>(k, static_cast<std::size_t>(ds.k)); ++g) {
              std::uint32_t id = gt[g]; if (id < sample) gt_ids.push_back(id);
            }
            gt_total += gt_ids.size();
            for (std::uint32_t id : gt_ids) { if (s1set.find(id) != s1set.end()) s1_cov_hits += 1.0; if (s2set.find(id) != s2set.end()) s2_cov_hits += 1.0; }
          }
        }
        Corr c1 = corr_stats(s1_dists, true_l2);
        Corr c2 = corr_stats(s2_dists, true_l2);
        Corr c3 = corr_stats(s3_dists, true_l2);
        std::vector<double> s1_scores(s1_dists.size()); for (size_t i = 0; i < s1_dists.size(); ++i) s1_scores[i] = -s1_dists[i];
        double auc1 = auc_score(s1_scores, s1_labels);

        // Track best by AUC, then s3.pearson
        const bool better = (auc1 > best.s1_auc) || (std::abs(auc1 - best.s1_auc) < 1e-6 && c3.pearson > best.s3.pearson);
        if (better) { best = BestSummary{auc1, c1, c2, c3, best_seeds, mode, lambda_ratio}; }
      }
    }

    // Emit JSON (best summary on top-level + sweep summary)
    JsonBuilder jb{std::cout};
    jb.begin();
    jb.kvs("dataset", dataset_name);
    jb.kv("calib_q", static_cast<double>(calib_Q));
    jb.begin_obj("correlation");
    jb.kv("s1_pearson", best.s1.pearson); jb.kv("s1_spearman", best.s1.spearman);
    jb.kv("s2_pearson", best.s2.pearson); jb.kv("s2_spearman", best.s2.spearman);
    jb.kv("s3_pearson", best.s3.pearson); jb.kv("s3_spearman", best.s3.spearman);
    jb.end_obj();
    jb.kv("s1_auc", best.s1_auc);
    jb.begin_obj("lambda_sweep");
    jb.kvs("best_mode", best.mode);
    jb.kv("best_lambda", best.lambda);
    jb.kv("best_s1_auc", best.s1_auc);
    jb.kv("best_s1_pearson", best.s1.pearson);
    jb.kv("best_s1_spearman", best.s1.spearman);
    jb.kv("best_s2_pearson", best.s2.pearson);
    jb.kv("best_s2_spearman", best.s2.spearman);
    jb.kv("best_s3_pearson", best.s3.pearson);
    jb.kv("best_s3_spearman", best.s3.spearman);
    // seeds array (as u64)
    std::cout << ",\n  \"best_seeds\": [";
    for (int i = 0; i < 6; ++i) { if (i) std::cout << ", "; std::cout << best.seeds[i]; }
    std::cout << "]";
    jb.end_obj();
    // Note: Coverage metrics are computed within candidate loop; we can optionally print aggregates if desired in future
    jb.end();

    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << std::endl;
    return 1;
  }
}
