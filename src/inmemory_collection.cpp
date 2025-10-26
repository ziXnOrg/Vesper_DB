#include <algorithm>
#include <cstring>
#include <expected>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "vesper/collection.hpp"
#include "vesper/filter_eval.hpp"
#include "inmemory_index.hpp"

#include "vesper/wal.hpp"

#if VESPER_ENABLE_KERNEL_DISPATCH
#include "vesper/kernels/dispatch.hpp"
#include "vesper/kernels/backends/scalar.hpp"
#endif
#include "vesper/kernels/distance.hpp"

namespace vesper {

struct collection_impl {
  std::unique_ptr<detail::inmem_index> idx;
  std::optional<wal::WalWriter> wal;   // optional WAL writer (rotation mode)
  std::uint64_t next_lsn{1};           // monotonic LSN for TYPE in {1,2}
};

collection::~collection(){ delete reinterpret_cast<collection_impl*>(impl_); }
collection::collection(collection&& o) noexcept : impl_(o.impl_) { o.impl_ = nullptr; }
collection& collection::operator=(collection&& o) noexcept { if(this!=&o){ delete reinterpret_cast<collection_impl*>(impl_); impl_=o.impl_; o.impl_=nullptr; } return *this; }

auto collection::open(const std::string& path) -> std::expected<collection, core::error> {
  collection c;
  auto* impl = new collection_impl{};
  impl->idx = std::make_unique<detail::inmem_index>();

  // Best-effort: open WAL writer in rotation mode for the provided directory
  if (!path.empty()) {
    wal::WalWriterOptions opts{};
    opts.dir = path;
    opts.prefix = "wal-";
    opts.max_file_bytes = 16 * 1024 * 1024; // rotate every 16MB (tunable)
    opts.durability_profile = wal::DurabilityProfile::Rotation; // sync on rotation only
    if (auto wx = wal::WalWriter::open(opts); wx) {
      impl->wal = std::move(*wx);
    } else {
      delete impl;
      return std::vesper_unexpected(wx.error());
    }
  }

  c.impl_ = impl;
  return c;
}

auto collection::insert(std::uint64_t id, const float* vec, std::size_t dim)
    -> std::expected<void, core::error> {
  auto* impl = reinterpret_cast<collection_impl*>(impl_);

  // Apply to in-memory index first (current behavior)
  std::vector<float> v(vec, vec + dim);
  detail::doc d{}; d.vec = std::move(v);
  impl->idx->store[id] = std::move(d);

  // Best-effort WAL append (op=0x01 insert)
  if (impl->wal.has_value()) {
    std::vector<std::uint8_t> payload;
    payload.reserve(1 + 8 + 4 + dim * sizeof(float));
    payload.push_back(static_cast<std::uint8_t>(0x01));
    // id (8 bytes)
    for (int i = 0; i < 8; ++i) payload.push_back(static_cast<std::uint8_t>((id >> (i*8)) & 0xFF));
    // dim (4 bytes)
    std::uint32_t d32 = static_cast<std::uint32_t>(dim);
    for (int i = 0; i < 4; ++i) payload.push_back(static_cast<std::uint8_t>((d32 >> (i*8)) & 0xFF));
    // vector data
    const std::uint8_t* bytes = reinterpret_cast<const std::uint8_t*>(vec);
    payload.insert(payload.end(), bytes, bytes + dim * sizeof(float));
    (void)impl->wal->append(impl->next_lsn++, /*type=*/1, payload);
  }

  return {};
}

auto collection::remove(std::uint64_t id) -> std::expected<void, core::error> {
  auto* impl = reinterpret_cast<collection_impl*>(impl_);

  // Apply to in-memory index first (current behavior)
  impl->idx->store.erase(id);

  // Best-effort WAL append (op=0x02 remove)
  if (impl->wal.has_value()) {
    std::vector<std::uint8_t> payload;
    payload.reserve(1 + 8);
    payload.push_back(static_cast<std::uint8_t>(0x02));
    for (int i = 0; i < 8; ++i) payload.push_back(static_cast<std::uint8_t>((id >> (i*8)) & 0xFF));
    (void)impl->wal->append(impl->next_lsn++, /*type=*/1, payload);
  }

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

#if VESPER_ENABLE_KERNEL_DISPATCH
  const auto& ops = kernels::select_backend_auto();
  auto score_l2 = [&](const std::vector<float>& v){ return ops.l2_sq(qv, v); };
  auto score_ip = [&](const std::vector<float>& v){ return -ops.inner_product(qv, v); }; // lower is better
  auto score_cos = [&](const std::vector<float>& v){ return 1.0f - ops.cosine_similarity(qv, v); }; // distance form
#else
  auto score_l2 = [&](const std::vector<float>& v){ return vesper::kernels::l2_sq(qv, v); };
  auto score_ip = [&](const std::vector<float>& v){ return -vesper::kernels::inner_product(qv, v); }; // lower is better
  auto score_cos = [&](const std::vector<float>& v){ return 1.0f - vesper::kernels::cosine_similarity(qv, v); }; // distance form
#endif

  int mode = 0; // 0=l2, 1=ip, 2=cosine
  if (p.metric == "ip") mode = 1;
  else if (p.metric == "cosine") mode = 2;

  for (auto id : candidates) {
    auto it = impl->idx->store.find(id);
    if (it == impl->idx->store.end()) continue;
    float d = (mode==0)? score_l2(it->second.vec) : (mode==1? score_ip(it->second.vec) : score_cos(it->second.vec));
    out.push_back({id, d});
  }
  auto comp = [](const search_result& a, const search_result& b){
    if (a.score == b.score) return a.id < b.id;
    return a.score < b.score;
  };
  if (out.size() > p.k) {
    auto kth = out.begin() + p.k;
    std::nth_element(out.begin(), kth, out.end(), comp);
    std::sort(out.begin(), kth, comp);
    out.resize(p.k);
  } else {
    std::sort(out.begin(), out.end(), comp);
  }
  return out;
}

auto collection::seal_segment() -> std::expected<void, core::error> { return {}; }
auto collection::compact() -> std::expected<void, core::error> { return {}; }
auto collection::snapshot() -> std::expected<void, core::error> { return {}; }

auto collection::list_segments() const -> std::expected<std::vector<segment_info>, core::error> {
  return std::vector<segment_info>{ segment_info{segment_state::mutable_segment, 0, "inmem"} };
}

} // namespace vesper

