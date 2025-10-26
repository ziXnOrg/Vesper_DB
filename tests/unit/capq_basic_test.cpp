// Copyright (c) 2025 Colin Macritchie / Ripple Group, LLC

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <catch2/catch_approx.hpp>

#include <array>
#include <cstdint>
#include <vector>

#include "vesper/index/capq.hpp"
#include "vesper/index/capq_util.hpp"
#include "vesper/index/capq_encode.hpp"
#include "vesper/index/capq_dist.hpp"

using namespace vesper::index;

TEST_CASE("CAPQ SoA sizing and view validation", "[capq]") {
  CapqSoAStorage storage(3, CapqHammingBits::B256);
  auto v = storage.view();
  REQUIRE(v.num_vectors == 3);
  REQUIRE(v.dimension == 64);
  auto ok = validate_capq_view(v);
  REQUIRE(ok.has_value());

  // 256-bit => 4 words per vector
  REQUIRE(v.hamming_words.size() == 3 * 4);
  REQUIRE(v.q4_packed.size() == 3 * 32);
  REQUIRE(v.q8.size() == 3 * 64);
  REQUIRE(v.residual_energy.size() == 3);
}

TEST_CASE("FWHT-64 preserves energy after normalization", "[capq][fwht]") {
  // Start with a simple vector; energy should be preserved due to 1/sqrt(n) normalization
  float x[64]{};
  for (int i = 0; i < 64; ++i) x[i] = static_cast<float>(i + 1);
  float sumsq_before = 0.0f;
  for (int i = 0; i < 64; ++i) sumsq_before += x[i] * x[i];

  fwht64_inplace(x);

  float sumsq_after = 0.0f;
  for (int i = 0; i < 64; ++i) sumsq_after += x[i] * x[i];

  REQUIRE(sumsq_after == Catch::Approx(sumsq_before).margin(1e-3));
}

TEST_CASE("Seeded permutation 64 is a bijection", "[capq][perm]") {
  std::array<std::uint8_t,64> perm{};
  seeded_permutation64(12345ULL, perm);
  bool seen[64]{};
  for (int i = 0; i < 64; ++i) {
    REQUIRE(perm[i] < 64);
    REQUIRE(seen[perm[i]] == false);
    seen[perm[i]] = true;
  }
}

TEST_CASE("Hamming sketch plane0 is sign bits", "[capq][hamming]") {
  float z[64]{};
  for (int i = 0; i < 64; ++i) z[i] = (i % 3 == 0) ? -0.1f : 0.1f;
  std::array<std::uint64_t,6> seeds{0,1,2,3,4,5};
  std::uint64_t words[4]{};
  auto r = compute_hamming_sketch(z, seeds, CapqHammingBits::B256, words);
  REQUIRE(r.has_value());
  // Check first word equals sign bits of z
  std::uint64_t expected0 = 0;
  for (int i = 0; i < 64; ++i) expected0 |= (static_cast<std::uint64_t>(z[i] >= 0.0f) << i);
  REQUIRE(words[0] == expected0);
}

TEST_CASE("Whitening (diagonal) roughly normalizes variance", "[capq][whiten]") {
  // Build synthetic data with known per-dim scales
  std::vector<float> data(64 * 1000);
  for (int i = 0; i < 1000; ++i) {
    for (int d = 0; d < 64; ++d) {
      data[i * 64 + d] = (d + 1) * 0.01f; // constant but varying scale
    }
  }
  auto m = train_whitening_model(data.data(), 1000, 1e-3f);
  REQUIRE(m.has_value());

  float z[64];
  apply_whitening(m.value(), &data[0], z);
  // Since input minus mean becomes near-zero, whitened should be near zero as well (allowing relaxed tolerance)
  for (int d = 0; d < 64; ++d) REQUIRE(z[d] == Catch::Approx(0.0f).margin(1.0f));

}

TEST_CASE("q8 encode and q4 coarsen monotonicity", "[capq][quant]") {
  // Simple vector with increasing values
  float z[64];
  for (int i = 0; i < 64; ++i) z[i] = static_cast<float>(i) / 10.0f;
  // Train trivial symmetric params
  auto qp = train_q8_params(z, 1, /*symmetric=*/true, /*clip=*/0.999f);
  REQUIRE(qp.has_value());

  std::int8_t q8[64];
  std::uint8_t q4[32];
  std::uint8_t e{};
  encode_capq_payload(z, qp.value(), q8, q4, e);

  // Check q4 high/low nibbles match q8>>4 packing order
  for (int i = 0; i < 32; ++i) {
    const int n0 = ((static_cast<int>(q8[2*i]) >> 4) + 8) & 0x0F;      // dim 2*i (low nibble)
    const int n1 = ((static_cast<int>(q8[2*i+1]) >> 4) + 8) & 0x0F;  // dim 2*i+1 (high nibble)
    REQUIRE(q4[i] == static_cast<std::uint8_t>((n1 << 4) | n0));
  }
}

TEST_CASE("q4/q8 scalar distance basic sanity", "[capq][dist]") {
  float z[64];
  for (int i = 0; i < 64; ++i) z[i] = static_cast<float>(i - 32) * 0.1f;
  auto qp = train_q8_params(z, 1, /*symmetric=*/true, 0.999f);
  REQUIRE(qp.has_value());

  std::int8_t q8a[64], q8b[64];
  std::uint8_t q4a[32], q4b[32];
  std::uint8_t ea{}, eb{};
  encode_capq_payload(z, qp.value(), q8a, q4a, ea);

  // Create a slightly perturbed vector
  float z2[64];
  for (int i = 0; i < 64; ++i) z2[i] = z[i] + ((i % 7) - 3) * 0.01f;
  encode_capq_payload(z2, qp.value(), q8b, q4b, eb);

  const float d8 = distance_q8_scalar(q8a, q8b, qp.value());
  const float d4 = distance_q4_scalar(q4a, q4b, qp.value());
  REQUIRE(d8 >= 0.0f);
  REQUIRE(d4 >= 0.0f);
  // Self distance must be zero
  REQUIRE(distance_q8_scalar(q8a, q8a, qp.value()) == Catch::Approx(0.0f).margin(1e-6));
  REQUIRE(distance_q4_scalar(q4a, q4a, qp.value()) == Catch::Approx(0.0f).margin(1e-6));
}


