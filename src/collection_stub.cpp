#include <expected>
#include <string>
#include <algorithm>
#include <memory>
#include "vesper/collection.hpp"
#include "inmemory_index.hpp"

namespace vesper {

struct collection_impl {
  std::unique_ptr<detail::inmem_index> idx;
};

static collection_impl* impl_of(collection* c){
  return reinterpret_cast<collection_impl*>(c);
}

auto collection::open(const std::string& /*path*/) -> std::expected<collection, core::error> {
  collection c;
  auto* impl = new collection_impl{};
  impl->idx = std::make_unique<detail::inmem_index>();
  // store impl pointer into this object storage via aliasing (placeholder; real impl will differ)
  static_assert(sizeof(collection_impl*) <= sizeof(collection), "placeholder impl too big");
  std::memcpy(&c, &impl, sizeof(impl));
  return c;
}

auto collection::insert(std::uint64_t id, const float* vec, std::size_t dim)
    -> std::expected<void, core::error> {
  auto* impl = impl_of(this);
  std::vector<float> v(vec, vec + dim);
  impl->idx->store[id] = std::move(v);
  return {};
}

auto collection::remove(std::uint64_t id) -> std::expected<void, core::error> {
  auto* impl = impl_of(this);
  impl->idx->store.erase(id);
  return {};
}

auto collection::search(const float* q, std::size_t dim, const search_params& p, const filter_expr*)
    -> std::expected<std::vector<search_result>, core::error> {
  auto* impl = impl_of(this);
  std::vector<float> qv(q, q + dim);
  std::vector<search_result> out;
  out.reserve(p.k);
  for (auto& kv : impl->idx->store) {
    float d = detail::inmem_index::l2(qv, kv.second);
    out.push_back({kv.first, d});
  }
  std::sort(out.begin(), out.end(), [](auto& a, auto& b){ return a.score < b.score; });
  if (out.size() > p.k) out.resize(p.k);
  return out;
}

auto collection::seal_segment() -> std::expected<void, core::error> { return {}; }
auto collection::compact() -> std::expected<void, core::error> { return {}; }
auto collection::snapshot() -> std::expected<void, core::error> { return {}; }
auto collection::recover() -> std::expected<void, core::error> { return {}; }

auto collection::list_segments() const -> std::expected<std::vector<segment_info>, core::error> {
  return std::vector<segment_info>{ segment_info{segment_state::mutable_segment, 0, "inmem"} };
}

} // namespace vesper

