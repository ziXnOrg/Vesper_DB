#include "vesper/wal/retention.hpp"
#include "vesper/wal/manifest.hpp"
#include "vesper/wal/snapshot.hpp"

#include <set>

namespace vesper::wal {

auto purge_wal(const std::filesystem::path& dir, std::uint64_t up_to_lsn)
    -> std::expected<void, vesper::core::error> {
  using vesper::core::error; using vesper::core::error_code;
  auto mx = load_manifest(dir);
  if (!mx) return std::vesper_unexpected(mx.error());
  Manifest m = *mx;
  // Determine global last_lsn among manifest entries
  std::uint64_t global_last = 0; for (const auto& e : m.entries) if (e.end_lsn > global_last) global_last = e.end_lsn;
  Manifest kept;
  for (const auto& e : m.entries) {
    const bool remove = (e.end_lsn <= up_to_lsn); // inclusive delete at cutoff
    if (remove) {
      std::error_code ec; std::filesystem::remove(dir / e.file, ec);
      if (ec) return std::vesper_unexpected(error{error_code::io_failed, "remove failed", "wal.retention"});
    } else {
      kept.entries.push_back(e);
    }
  }
  if (auto sx = save_manifest(dir, kept); !sx) return std::vesper_unexpected(sx.error());
  // Persist snapshot at cutoff to define baseline for replay
  if (auto sp = save_snapshot(dir, Snapshot{up_to_lsn}); !sp) return std::vesper_unexpected(sp.error());
  return {};
}

auto purge_keep_last_n(const std::filesystem::path& dir, std::size_t keep_last_n)
    -> std::expected<void, vesper::core::error> {
  using vesper::core::error; using vesper::core::error_code;
  auto mx = load_manifest(dir);
  if (!mx) return std::vesper_unexpected(mx.error());
  Manifest m = *mx;
  if (m.entries.size() <= keep_last_n) return {};
  std::sort(m.entries.begin(), m.entries.end(), [](auto&a, auto&b){ return a.seq < b.seq; });
  std::size_t to_remove = m.entries.size() - keep_last_n;
  for (std::size_t i = 0; i < to_remove; ++i) {
    std::error_code ec; std::filesystem::remove(dir / m.entries[i].file, ec);
    if (ec) return std::vesper_unexpected(error{error_code::io_failed, "remove failed", "wal.retention"});
  }
  Manifest kept; kept.entries.assign(m.entries.end() - keep_last_n, m.entries.end());
  return save_manifest(dir, kept);
}

auto purge_keep_newer_than(const std::filesystem::path& dir, std::filesystem::file_time_type cutoff_time)
    -> std::expected<void, vesper::core::error> {
  using vesper::core::error; using vesper::core::error_code;
  auto mx = load_manifest(dir);
  if (!mx) return std::vesper_unexpected(mx.error());
  Manifest m = *mx; Manifest kept;

  // Collect candidates by timestamp relation to cutoff
  struct Cand { ManifestEntry e; std::filesystem::file_time_type ft; };
  std::vector<Cand> newer, equal;
  for (const auto& e : m.entries) {
    auto p = dir / e.file; std::error_code ec; auto ft = std::filesystem::last_write_time(p, ec);
    if (ec) {
      // If stat fails, best-effort: attempt removal (treat as older)
      std::error_code ec2; std::filesystem::remove(p, ec2);
      continue;
    }
    if (ft > cutoff_time) newer.push_back(Cand{e, ft});
    else if (ft == cutoff_time) equal.push_back(Cand{e, ft});
    else {
      std::error_code ec2; std::filesystem::remove(p, ec2);
    }
  }
  // Keep all strictly newer; for ties at exactly cutoff_time, keep only the highest-seq file
  for (auto& c : newer) kept.entries.push_back(c.e);
  if (!equal.empty()) {
    auto it = std::max_element(equal.begin(), equal.end(), [](const Cand& a, const Cand& b){ return a.e.seq < b.e.seq; });
    kept.entries.push_back(it->e);
    // Remove all other equal-timestamp files
    for (auto& c : equal) if (c.e.file != it->e.file) { std::error_code ec2; std::filesystem::remove(dir / c.e.file, ec2); }
  }
  // Save manifest with kept entries sorted by seq
  std::sort(kept.entries.begin(), kept.entries.end(), [](auto&a, auto&b){ return a.seq < b.seq; });
  return save_manifest(dir, kept);
}

auto purge_keep_total_bytes_max(const std::filesystem::path& dir, std::uint64_t max_total_bytes)
    -> std::expected<void, vesper::core::error> {
  using vesper::core::error; using vesper::core::error_code;
  auto mx = load_manifest(dir);
  if (!mx) return std::vesper_unexpected(mx.error());
  Manifest m = *mx;
  std::sort(m.entries.begin(), m.entries.end(), [](auto&a, auto&b){ return a.seq < b.seq; });
  // Start from newest and keep while under budget
  std::uint64_t acc = 0; Manifest kept;
  for (auto it = m.entries.rbegin(); it != m.entries.rend(); ++it) {
    if (acc + it->bytes <= max_total_bytes) { acc += it->bytes; kept.entries.push_back(*it); }
  }
  // Remove all not kept
  std::set<std::string> keep_names; for (auto& e : kept.entries) keep_names.insert(e.file);
  for (const auto& e : m.entries) {
    if (!keep_names.count(e.file)) { std::error_code ec; std::filesystem::remove(dir / e.file, ec); if (ec) return std::vesper_unexpected(error{error_code::io_failed, "remove failed", "wal.retention"}); }
  }
  // kept.entries currently in reverse order; restore ascending by seq
  std::sort(kept.entries.begin(), kept.entries.end(), [](auto&a, auto&b){ return a.seq < b.seq; });
  return save_manifest(dir, kept);
}


} // namespace vesper::wal

