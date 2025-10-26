/**
 * Copyright (c) 2025 Colin Macritchie / Ripple Group, LLC
 * Licensed under the Apache License, Version 2.0
 */

#pragma once

/** \file capq_select.hpp
 *  \brief Candidate scoring in 16-wide blocks and bounded TopK selection.
 */

#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <vector>
#include <queue>
#include <utility>

#include "vesper/index/capq.hpp"
#include "vesper/index/capq_dist.hpp"

namespace vesper::index {

/** \brief Simple bounded TopK selector (max-heap on distance). */
class TopK {
public:
  explicit TopK(std::size_t k) : k_(k) { heap_.reserve(k); }

  void push(std::uint64_t id, float distance) {
    if (k_ == 0) return;
    if (heap_.size() < k_) {
      heap_.emplace_back(distance, id);
      std::push_heap(heap_.begin(), heap_.end(), cmp);
    } else if (distance < heap_.front().first) {
      std::pop_heap(heap_.begin(), heap_.end(), cmp);
      heap_.back() = {distance, id};
      std::push_heap(heap_.begin(), heap_.end(), cmp);
    }
  }

  std::vector<std::pair<float, std::uint64_t>> take_sorted() {
    std::sort(heap_.begin(), heap_.end(), cmp);
    return heap_;
  }

  std::size_t size() const noexcept { return heap_.size(); }
  float worst_distance() const noexcept { return heap_.empty() ? std::numeric_limits<float>::infinity() : heap_.front().first; }

private:
  static bool cmp(const std::pair<float, std::uint64_t>& a,
                  const std::pair<float, std::uint64_t>& b) {
    return a.first > b.first; // min distance preferred; this makes a max-heap by distance
  }
  std::size_t k_;
  std::vector<std::pair<float, std::uint64_t>> heap_;
};

/** \brief Score a block of up to 16 candidates using q8 and push into TopK.
 *  \param view  CAPQ SoA view
 *  \param q8_query  pointer to 64 q8 codes for the query
 *  \param params  q8 scaling params
 *  \param ids  candidate ids array (length <= 16)
 *  \param count  number of candidates in this block (<= 16)
 */
inline void score_block_q8(const CapqSoAView& view,
                           const std::int8_t* q8_query,
                           const CapqQ8Params& params,
                           const std::uint64_t* ids,
                           std::size_t count,
                           TopK& topk) {
  for (std::size_t i = 0; i < count; ++i) {
    const std::uint64_t id = ids[i];
    // For skeleton, interpret id as index into SoA (callers ensure bounds)
    const std::int8_t* q8_db = view.q8_ptr(static_cast<std::size_t>(id));
    const float d = distance_q8(q8_query, q8_db, params);
    topk.push(id, d);
  }
}

/** \brief Score a block via q4 packed codes. */
inline void score_block_q4(const CapqSoAView& view,
                           const std::uint8_t* q4_query,
                           const CapqQ8Params& params,
                           const std::uint64_t* ids,
                           std::size_t count,
                           TopK& topk) {
  for (std::size_t i = 0; i < count; ++i) {
    const std::uint64_t id = ids[i];
    const std::uint8_t* q4_db = view.q4_ptr(static_cast<std::size_t>(id));
    const float d = distance_q4_scalar(q4_query, q4_db, params);
    topk.push(id, d);
  }
}

/** \brief Score a block via Hamming words. */
inline void score_block_hamming(const CapqSoAView& view,
                                const std::uint64_t* ham_query,
                                const std::uint64_t* ids,
                                std::size_t count,
                                TopK& topk) {
  const std::size_t words = view.words_per_vector();
  for (std::size_t i = 0; i < count; ++i) {
    const std::uint64_t id = ids[i];
    const std::uint64_t* ham_db = view.hamming_ptr(static_cast<std::size_t>(id));
    const float d = static_cast<float>(hamming_distance_words(ham_query, ham_db, words));
    topk.push(id, d);
  }
}

} // namespace vesper::index


