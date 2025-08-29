#include <algorithm>
#include <cstring>
#include <expected>
#include <memory>
#include <string>
#include <vector>

#include "vesper/collection.hpp"
#include "vesper/filter_eval.hpp"
#include "inmemory_index.hpp"

namespace vesper {

struct collection_impl {
  std::unique_ptr<detail::inmem_index> idx;
};

collection::~collection(){ delete reinterpret_cast<collection_impl*>(impl_); }
collection::collection(collection&& o) noexcept : impl_(o.impl_) { o.impl_ = nullptr; }
collection& collection::operator=(collection&& o) noexcept { if(this!=&o){ delete reinterpret_cast<collection_impl*>(impl_); impl_=o.impl_; o.impl_=nullptr; } return *this; }

auto collection::open(const std::string& /*path*/) -> std::expected<collection, core::error> {
  collection c;
  auto* impl = new collection_impl{};
  impl->idx = std::make_unique<detail::inmem_index>();
  c.impl_ = impl;
  return c;
}

auto collection::insert(std::uint64_t id, const float* vec, std::size_t dim)
    -> std::expected<void, core::error> {
  auto* impl = reinterpret_cast<collection_impl*>(impl_);
  std::vector<float> v(vec, vec + dim);
  detail::doc d{}; d.vec = std::move(v);
  impl->idx->store[id] = std::move(d);
  return {};
}

auto collection::remove(std::uint64_t id) -> std::expected<void, core::error> {
  auto* impl = reinterpret_cast<collection_impl*>(impl_);
  impl->idx->store.erase(id);
  return {};
}

auto collection::search(const float* q, std::size_t dim, const search_params& p, const filter_expr* fexpr)
    -> std::expected<std::vector<search_result>, core::error> {
  auto* impl = reinterpret_cast<collection_impl*>(impl_);
  std::vector<float> qv(q, q + dim);

  std::vector<filter_eval::id_view> views; views.reserve(impl->idx->store.size());
  for (auto& kv : impl->idx->store) {
    views.push_back({kv.first, &kv.second.tags, &kv.second.nums});
  }
  std::vector<std::uint64_t> candidates = filter_eval::apply_filter(fexpr, views);

  std::vector<search_result> out; out.reserve(std::min<std::size_t>(p.k, candidates.size()));
  for (auto id : candidates) {
    auto it = impl->idx->store.find(id);
    if (it == impl->idx->store.end()) continue;
    float d = kernels::l2_sq(qv, it->second.vec);
    out.push_back({id, d});
  }
  std::sort(out.begin(), out.end(), [](auto& a, auto& b){ if (a.score == b.score) return a.id < b.id; return a.score < b.score; });
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

