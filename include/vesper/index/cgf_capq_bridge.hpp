/**
 * Copyright (c) 2025 Colin Macritchie / Ripple Group, LLC
 * Licensed under the Apache License, Version 2.0
 */

#pragma once

/** \file cgf_capq_bridge.hpp
 *  \brief Bridge adapter exposing CAPQ as a CGF-compatible filter stage.
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <chrono>

#include "vesper/index/capq.hpp"
#include "vesper/index/capq_encode.hpp"
#include "vesper/index/capq_util.hpp"
#include "vesper/index/capq_dist.hpp"
#include "vesper/index/capq_select.hpp"

namespace vesper::index {

/** \brief CAPQ-backed candidate filter for CGF pipelines. */
class CapqFilter {
public:
  CapqFilter() = default;

  // Initialize with a read-only CAPQ view (preferred)
  void initialize(const CapqSoAViewConst& storage_view,
                  const CapqWhiteningModel& whitening_model,
                  const CapqQ8Params& q8_params,
                  const std::array<std::uint64_t, 6>& seeds,
                  CapqHammingBits hbits) {
    storage_view_ = storage_view;
    whitening_model_ = whitening_model;
    q8_params_ = q8_params;
    seeds_ = seeds;
    hbits_ = hbits;
  }
  // Backward-compatible overload: accept mutable view and convert to const
  void initialize(const CapqSoAView& storage_view,
                  const CapqWhiteningModel& whitening_model,
                  const CapqQ8Params& q8_params,
                  const std::array<std::uint64_t, 6>& seeds,
                  CapqHammingBits hbits) {
    CapqSoAViewConst v{};
    v.num_vectors = storage_view.num_vectors;
    v.dimension = storage_view.dimension;
    v.hbits = storage_view.hbits;
    v.hamming_words = std::span<const std::uint64_t>(storage_view.hamming_words.data(), storage_view.hamming_words.size());
    v.q4_packed = std::span<const std::uint8_t>(storage_view.q4_packed.data(), storage_view.q4_packed.size());
    v.q8 = std::span<const std::int8_t>(storage_view.q8.data(), storage_view.q8.size());
    v.residual_energy = std::span<const std::uint8_t>(storage_view.residual_energy.data(), storage_view.residual_energy.size());
    initialize(v, whitening_model, q8_params, seeds, hbits);
  }

  /** \brief Search candidates using CAPQ three-stage cascade; returns top-k ids. */
  std::vector<std::uint32_t> search(const float* query, std::size_t original_dim,
                                    std::uint32_t k,
                                    std::size_t stage1_keep,
                                    std::size_t stage2_keep) const {
    // Project + whiten
    float q64[64], zq[64];
    if (original_dim == 64) {
      for (int i = 0; i < 64; ++i) q64[i] = query[i];
    } else {
      std::size_t pow2 = 1; while (pow2 < original_dim) pow2 <<= 1;
      std::vector<float> buf(pow2, 0.0f);
      for (std::size_t i = 0; i < original_dim; ++i) buf[i] = query[i];
      fwht64_inplace(buf.data());
      for (int i = 0; i < 64; ++i) q64[i] = buf[i];
    }
    apply_whitening(whitening_model_, q64, zq);

    // Encode query payloads
    const std::size_t words = (hbits_ == CapqHammingBits::B256) ? 4u : 6u;
    std::vector<std::uint64_t> hamq(words);
    auto hr = compute_hamming_sketch(zq, seeds_, hbits_, hamq.data());
    (void)hr;
    std::array<std::int8_t, 64> q8{}; std::array<std::uint8_t, 32> q4{}; std::uint8_t dummy{};
    encode_capq_payload(zq, q8_params_, q8.data(), q4.data(), dummy);

    // Stage 1
    TopK top_s1(stage1_keep);
    for (std::size_t i = 0; i < storage_view_.num_vectors; ++i) {
      const float d = static_cast<float>(hamming_distance_words(hamq.data(), storage_view_.hamming_ptr(i), words));
      top_s1.push(static_cast<std::uint64_t>(i), d);
    }
    const auto s1 = top_s1.take_sorted();

    // Stage 2
    TopK top_s2(stage2_keep);
    for (const auto& [d, id] : s1) {
      (void)d;
      const float dist = distance_q4(q4.data(), storage_view_.q4_ptr(static_cast<std::size_t>(id)), q8_params_);
      top_s2.push(id, dist);
    }
    const auto s2 = top_s2.take_sorted();

    // Stage 3
    TopK topk(k);
    for (const auto& [d, id] : s2) {
      (void)d;
      const float dist = distance_q8(q8.data(), storage_view_.q8_ptr(static_cast<std::size_t>(id)), q8_params_);
      topk.push(id, dist);
    }
    const auto final = topk.take_sorted();
    std::vector<std::uint32_t> ids; ids.reserve(final.size());
    for (const auto& [dist, id] : final) { (void)dist; ids.push_back(static_cast<std::uint32_t>(id)); }
    return ids;
  }

private:
  CapqSoAViewConst storage_view_{};
  CapqWhiteningModel whitening_model_{};
  CapqQ8Params q8_params_{};
  std::array<std::uint64_t, 6> seeds_{};
  CapqHammingBits hbits_{CapqHammingBits::B256};
};

} // namespace vesper::index


