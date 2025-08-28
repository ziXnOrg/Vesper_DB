#include <expected>
#include <string>
#include "vesper/collection.hpp"

namespace vesper {

auto collection::open(const std::string& /*path*/) -> std::expected<collection, core::error> {
  // No-op stub for CI linkage; return a default-constructed collection handle
  return collection{};
}

auto collection::insert(std::uint64_t, const float*, std::size_t) -> std::expected<void, core::error> {
  return {};
}

auto collection::remove(std::uint64_t) -> std::expected<void, core::error> { return {}; }

auto collection::search(const float*, std::size_t, const search_params&, const filter_expr*)
    -> std::expected<std::vector<search_result>, core::error> {
  return std::vector<search_result>{};
}

auto collection::seal_segment() -> std::expected<void, core::error> { return {}; }
auto collection::compact() -> std::expected<void, core::error> { return {}; }
auto collection::snapshot() -> std::expected<void, core::error> { return {}; }
auto collection::recover() -> std::expected<void, core::error> { return {}; }

auto collection::list_segments() const -> std::expected<std::vector<segment_info>, core::error> {
  return std::vector<segment_info>{};
}

} // namespace vesper

