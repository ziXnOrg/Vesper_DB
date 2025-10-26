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

#include "vesper/index/capq_util.hpp"

#include <algorithm>

namespace vesper::index {

void fwht64_inplace(float* data) noexcept {
  // 64 = 2^6, six butterfly stages
  // Scalar FWHT with in-place butterflies
  for (int len = 1; len < 64; len <<= 1) {
    for (int i = 0; i < 64; i += (len << 1)) {
      for (int j = 0; j < len; ++j) {
        const float a = data[i + j];
        const float b = data[i + j + len];
        data[i + j] = a + b;
        data[i + j + len] = a - b;
      }
    }
  }
  // Normalize by sqrt(64) = 8 to keep orthogonality
  for (int i = 0; i < 64; ++i) data[i] *= 0.125f;
}

// SplitMix64 for simple deterministic PRNG
static inline std::uint64_t splitmix64(std::uint64_t& x) noexcept {
  std::uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

void seeded_permutation64(std::uint64_t seed, std::array<std::uint8_t, 64>& out_indices) noexcept {
  for (int i = 0; i < 64; ++i) out_indices[i] = static_cast<std::uint8_t>(i);
  std::uint64_t s = seed;
  for (int i = 63; i > 0; --i) {
    const std::uint64_t r = splitmix64(s);
    const int j = static_cast<int>(r % (i + 1));
    std::swap(out_indices[i], out_indices[j]);
  }
}

auto compute_hamming_sketch(const float z_in[64],
                            const std::array<std::uint64_t, 6>& seeds,
                            CapqHammingBits bits,
                            std::uint64_t* out_words) -> std::expected<void, core::error> {
  if (out_words == nullptr) {
    return std::vesper_unexpected(core::error{core::error_code::invalid_argument,
                                       "out_words is null",
                                       "index.capq.util"});
  }
  const std::size_t wpv = capq_words_per_vector(bits);

  // Plane 0: direct sign bits of z
  std::uint64_t plane0 = 0;
  for (int i = 0; i < 64; ++i) {
    plane0 |= (static_cast<std::uint64_t>(z_in[i] >= 0.0f) << i);
  }
  out_words[0] = plane0;

  // Prepare working copy for FWHT and per-plane transforms
  float w[64];
  for (int i = 0; i < 64; ++i) w[i] = z_in[i];
  fwht64_inplace(w);

  // Subsequent planes: apply seeded permutation and seeded sign flips
  for (std::size_t p = 1; p < wpv; ++p) {
    std::array<std::uint8_t, 64> perm{};
    seeded_permutation64(seeds[p], perm);

    // Seeded signs: derive from SplitMix64 stream
    std::uint64_t s = seeds[p];
    std::uint64_t word = 0;
    for (int i = 0; i < 64; ++i) {
      // Pull randomness every 8 steps to reduce PRNG pressure
      if ((i & 7) == 0) (void)splitmix64(s);
      const bool sign_pos = (splitmix64(s) & 1ull) != 0ull;
      const int idx = perm[i];
      const float val = sign_pos ? w[idx] : -w[idx];
      word |= (static_cast<std::uint64_t>(val >= 0.0f) << i);
    }
    out_words[p] = word;
  }

  return {};
}

} // namespace vesper::index


