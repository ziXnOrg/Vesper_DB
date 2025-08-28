#pragma once

/**
 * \file segment.hpp
 * \brief Segment management (mutable, sealed) and lifecycle operations.
 */

#include <cstdint>
#include <expected>
#include <string>
#include <vector>

#include "vesper/error.hpp"

namespace vesper {

enum class segment_state : std::uint8_t { mutable_segment, sealed_segment };

struct segment_info {
  segment_state state{};
  std::uint64_t doc_count{};
  std::string index_family; // "ivf_pq", "hnsw", "disk_graph"
};

class segment {
public:
  static auto load(const std::string& path)
      -> std::expected<segment, core::error>;

  auto info() const -> segment_info;

  auto seal() -> std::expected<void, core::error>;
};

} // namespace vesper

