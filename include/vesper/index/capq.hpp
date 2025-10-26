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

#pragma once

/** \file capq.hpp
 *  \brief CAPQ (Cascade-Aware Progressive Quantization) payload layout (SoA) and views.
 *
 *  This header defines the memory layout and typed views for CAPQ’s
 *  128-byte payload per vector plus a 1-byte residual energy side array:
 *   - 32B Hamming sketch (256 bits; expandable to 384 bits)
 *   - 32B packed 4-bit quantization (64 dims × 4 bits)
 *   - 64B 8-bit quantization (64 × int8_t)
 *   - 1B  residual energy (separate metadata array)
 *
 *  Layout is Structure-of-Arrays (SoA) to maximize cache locality during
 *  stage scanning, SIMD-friendly access, and sequential prefetching.
 *
 *  Thread-safety: these are plain data containers/views. Concurrency control
 *  is the responsibility of higher layers (see ADR-0004).
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <array>
#include <string>
#include <expected>

#include "vesper/span_polyfill.hpp"
#include "vesper/error.hpp"

namespace vesper::index {

/** \brief Hamming sketch width for Stage 1. */
enum class CapqHammingBits : std::uint16_t {
  B256 = 256,
  B384 = 384
};

/** \brief CAPQ build/configuration parameters with sane defaults. */
struct CapqBuildParams {
  // Stage 1 sketch
  CapqHammingBits hbits{CapqHammingBits::B256};
  // Seeds for FWHT-based plane permutations/signs. Only the first
  // words_per_vector() entries are used depending on hbits.
  std::array<std::uint64_t, 6> hamming_seeds{ {
      0x9E3779B185EBCA87ULL, 0xC2B2AE3D27D4EB4FULL,
      0x165667B19E3779F9ULL, 0x85EBCA77C2B2AE63ULL,
      0x27D4EB2F165667C5ULL, 0x94D049BB133111EBULL } };

  // Whitening ridge factor: lambda = ratio * trace(Sigma)/64
  float whitening_lambda_ratio{1e-3f};

  // Quantization controls
  bool q8_symmetric{true};                 // symmetric [-127,127]
  float q8_clip_percentile{0.9995f};       // clip outliers before scale fit
  bool enable_q4_coarsen{true};            // q4 = q8 >> 4
};

/** \brief Non-owning SoA view over CAPQ payload arrays. */
struct CapqSoAView {
  std::size_t num_vectors{0};           /**< number of vectors in this view */
  std::size_t dimension{64};            /**< logical dimension after projection */
  CapqHammingBits hbits{CapqHammingBits::B256};

  // Hamming sketches as contiguous 64-bit words: [num_vectors * words_per_vec]
  // words_per_vec = 4 for 256-bit, 6 for 384-bit.
  std::span<std::uint64_t> hamming_words;    /**< size = num_vectors * words_per_vec */

  // Packed q4 codes: 32 bytes per vector (64 dims × 4 bits)
  std::span<std::uint8_t> q4_packed;         /**< size = num_vectors * 32 */

  // q8 vector: 64 bytes per vector (64 × int8)
  std::span<std::int8_t> q8;                 /**< size = num_vectors * 64 */

  // Residual energy: 1 byte per vector, separate side array for tight scans
  std::span<std::uint8_t> residual_energy;   /**< size = num_vectors */

  [[nodiscard]] constexpr std::size_t words_per_vector() const noexcept {
    return (hbits == CapqHammingBits::B256) ? 4u : 6u;
  }

  [[nodiscard]] constexpr std::size_t bytes_per_vector_payload() const noexcept {
    // 32B (hamming) + 32B (q4) + 64B (q8) = 128B
    return 128u;
  }

  [[nodiscard]] constexpr std::size_t bytes_per_vector_total() const noexcept {
    // payload + 1B residual side array
    return 129u;
  }

  [[nodiscard]] inline const std::uint64_t* hamming_ptr(std::size_t i) const noexcept {
    return hamming_words.data() + i * words_per_vector();
  }
  [[nodiscard]] inline std::uint64_t* hamming_ptr(std::size_t i) noexcept {
    return hamming_words.data() + i * words_per_vector();
  }

  [[nodiscard]] inline const std::uint8_t* q4_ptr(std::size_t i) const noexcept {
    return q4_packed.data() + i * 32u;
  }
  [[nodiscard]] inline std::uint8_t* q4_ptr(std::size_t i) noexcept {
    return q4_packed.data() + i * 32u;
  }

  [[nodiscard]] inline const std::int8_t* q8_ptr(std::size_t i) const noexcept {
    return q8.data() + i * 64u;
  }
  [[nodiscard]] inline std::int8_t* q8_ptr(std::size_t i) noexcept {
    return q8.data() + i * 64u;
  }

  [[nodiscard]] inline const std::uint8_t* residual_ptr(std::size_t i) const noexcept {
    return residual_energy.data() + i;
  }
  [[nodiscard]] inline std::uint8_t* residual_ptr(std::size_t i) noexcept {
    return residual_energy.data() + i;
  }
};

/** \brief Owning SoA storage for CAPQ payloads. */
class CapqSoAStorage {
public:
  CapqSoAStorage() = default;

  /** Construct storage for a given size and Hamming width. */
  explicit CapqSoAStorage(std::size_t n, CapqHammingBits bits = CapqHammingBits::B256)
      : num_vectors_(n), hbits_(bits) {
    allocate();
  }

  /** Resize to hold n vectors, preserving contents where possible. */
  void resize(std::size_t n) {
    num_vectors_ = n;
    allocate();
  }

  /** Return a non-owning view over the storage. */
  [[nodiscard]] CapqSoAView view() noexcept {
    return make_view();
  }
  [[nodiscard]] CapqSoAView view() const noexcept {
    return make_view();
  }

  [[nodiscard]] std::size_t size() const noexcept { return num_vectors_; }
  [[nodiscard]] std::size_t dimension() const noexcept { return 64u; }
  [[nodiscard]] CapqHammingBits hbits() const noexcept { return hbits_; }

  [[nodiscard]] std::size_t words_per_vector() const noexcept {
    return (hbits_ == CapqHammingBits::B256) ? 4u : 6u;
  }

private:
  std::size_t num_vectors_{0};
  CapqHammingBits hbits_{CapqHammingBits::B256};

  std::vector<std::uint64_t> hamming_words_;   // [num_vectors * words_per_vector]
  std::vector<std::uint8_t> q4_packed_;        // [num_vectors * 32]
  std::vector<std::int8_t> q8_;                // [num_vectors * 64]
  std::vector<std::uint8_t> residual_energy_;  // [num_vectors]

  void allocate() {
    const std::size_t wpv = words_per_vector();
    hamming_words_.assign(num_vectors_ * wpv, 0u);
    q4_packed_.assign(num_vectors_ * 32u, 0u);
    q8_.assign(num_vectors_ * 64u, 0);
    residual_energy_.assign(num_vectors_, 0u);
  }

  [[nodiscard]] CapqSoAView make_view() noexcept {
    CapqSoAView v;
    v.num_vectors = num_vectors_;
    v.dimension = 64u;
    v.hbits = hbits_;
    v.hamming_words = std::span<std::uint64_t>(hamming_words_.data(), hamming_words_.size());
    v.q4_packed = std::span<std::uint8_t>(q4_packed_.data(), q4_packed_.size());
    v.q8 = std::span<std::int8_t>(q8_.data(), q8_.size());
    v.residual_energy = std::span<std::uint8_t>(residual_energy_.data(), residual_energy_.size());
    return v;
  }
  [[nodiscard]] CapqSoAView make_view() const noexcept {
    CapqSoAView v;
    v.num_vectors = num_vectors_;
    v.dimension = 64u;
    v.hbits = hbits_;
    v.hamming_words = std::span<std::uint64_t>(const_cast<std::uint64_t*>(hamming_words_.data()), hamming_words_.size());
    v.q4_packed = std::span<std::uint8_t>(const_cast<std::uint8_t*>(q4_packed_.data()), q4_packed_.size());
    v.q8 = std::span<std::int8_t>(const_cast<std::int8_t*>(q8_.data()), q8_.size());
    v.residual_energy = std::span<std::uint8_t>(const_cast<std::uint8_t*>(residual_energy_.data()), residual_energy_.size());
    return v;
  }
};

/** \brief Validate that a SoA view has consistent sizes for the configured Hamming width. */
inline auto validate_capq_view(const CapqSoAView& v) -> std::expected<void, core::error> {
  const std::size_t wpv = (v.hbits == CapqHammingBits::B256) ? 4u : 6u;
  const std::size_t expected_hamming = v.num_vectors * wpv;
  const std::size_t expected_q4 = v.num_vectors * 32u;
  const std::size_t expected_q8 = v.num_vectors * 64u;
  const std::size_t expected_res = v.num_vectors;

  if (v.hamming_words.size() != expected_hamming) {
    return std::vesper_unexpected(core::error{core::error_code::precondition_failed,
                                       "CAPQ view: invalid hamming_words size",
                                       "index.capq"});
  }
  if (v.q4_packed.size() != expected_q4) {
    return std::vesper_unexpected(core::error{core::error_code::precondition_failed,
                                       "CAPQ view: invalid q4_packed size",
                                       "index.capq"});
  }
  if (v.q8.size() != expected_q8) {
    return std::vesper_unexpected(core::error{core::error_code::precondition_failed,
                                       "CAPQ view: invalid q8 size",
                                       "index.capq"});
  }
  if (v.residual_energy.size() != expected_res) {
    return std::vesper_unexpected(core::error{core::error_code::precondition_failed,
                                       "CAPQ view: invalid residual_energy size",
                                       "index.capq"});
  }
  return {};
}

} // namespace vesper::index


