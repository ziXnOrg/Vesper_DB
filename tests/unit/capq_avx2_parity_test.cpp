/**
 * Copyright (c) 2025 Colin Macritchie / Ripple Group, LLC
 * Licensed under the Apache License, Version 2.0
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <random>
#include <vector>

#include "vesper/index/capq_dist.hpp"
#include "vesper/index/capq_dist_avx2.hpp"

using vesper::index::CapqQ8Params;
using Catch::Approx;

namespace {

static std::mt19937& rng() {
  static std::mt19937 gen(1234567);
  return gen;
}

static std::vector<std::uint64_t> random_ham(std::size_t words) {
  std::vector<std::uint64_t> v(words);
  std::uniform_int_distribution<std::uint64_t> dist;
  for (auto& x : v) x = dist(rng());
  return v;
}

static std::vector<std::uint8_t> random_q4(std::size_t bytes) {
  std::vector<std::uint8_t> v(bytes);
  std::uniform_int_distribution<int> dist(0, 255);
  for (auto& x : v) x = static_cast<std::uint8_t>(dist(rng()));
  return v;
}

static std::vector<std::int8_t> random_q8(std::size_t dims) {
  std::vector<std::int8_t> v(dims);
  std::uniform_int_distribution<int> dist(-127, 127);
  for (auto& x : v) x = static_cast<std::int8_t>(dist(rng()));
  return v;
}

static CapqQ8Params unit_scales() {
  CapqQ8Params p{};
  for (int i = 0; i < 64; ++i) p.scale[i] = 1.0f;
  return p;
}

} // namespace

TEST_CASE("CAPQ AVX2 parity: Hamming vs scalar", "[capq][simd]") {
  if (!vesper::index::capq_avx2_available()) return; // skip if no AVX2
  for (std::size_t words : {4ull, 6ull}) {
    auto a = random_ham(words);
    auto b = random_ham(words);
    const auto s = vesper::index::hamming_distance_words(a.data(), b.data(), words);
    const auto v = vesper::index::hamming_distance_avx2(a.data(), b.data(), words);
    REQUIRE(s == v);
  }
}

TEST_CASE("CAPQ AVX2 parity: q8 vs scalar", "[capq][simd]") {
  if (!vesper::index::capq_avx2_available()) return;
  auto q = random_q8(64);
  auto d = random_q8(64);
  auto params = unit_scales();
  const float s = vesper::index::distance_q8_scalar(q.data(), d.data(), params);
  const float v = vesper::index::distance_q8_avx2(q.data(), d.data(), params);
  REQUIRE(v == Approx(s).margin(1e-4f));
}

TEST_CASE("CAPQ AVX2 parity: q4 vs scalar", "[capq][simd]") {
  if (!vesper::index::capq_avx2_available()) return;
  auto q = random_q4(32);
  auto d = random_q4(32);
  auto params = unit_scales();
  const float s = vesper::index::distance_q4_scalar(q.data(), d.data(), params);
  const float v = vesper::index::distance_q4_avx2(q.data(), d.data(), params);
  REQUIRE(v == Approx(s).margin(1e-3f));
}


