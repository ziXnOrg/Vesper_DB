#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vesper/error.hpp"

namespace vesper {

struct search_params {
  std::string metric; // "l2" | "ip" | "cosine"
  std::uint32_t k{10};
  float target_recall{0.95f};
  std::uint32_t nprobe{8};
  std::uint32_t ef_search{64};
  std::uint32_t rerank{0};
};

struct search_result {
  std::uint64_t id{};
  float score{}; // distance or similarity depending on metric
};

struct filter_expr; // fwd-decl

class collection {
public:
  // Doxygen: thread-safe for readers; single-writer semantics for mutating ops
  static auto open(const std::string& path) -> std::expected<collection, core::error>;

  auto insert(std::uint64_t id, const float* vec, std::size_t dim,
              /*metadata TBD*/)
      -> std::expected<void, core::error>;

  auto remove(std::uint64_t id) -> std::expected<void, core::error>;

  auto search(const float* query, std::size_t dim, const search_params& p,
              const filter_expr* filter)
      -> std::expected<std::vector<search_result>, core::error>;

  auto seal_segment() -> std::expected<void, core::error>;
  auto compact() -> std::expected<void, core::error>;
  auto snapshot() -> std::expected<void, core::error>;
  auto recover() -> std::expected<void, core::error>;
};

} // namespace vesper

