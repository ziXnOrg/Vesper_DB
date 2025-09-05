#include "vesper/wal/retention.hpp"
#include "vesper/wal/manifest.hpp"

#include <set>

namespace vesper::wal {

auto purge_wal(const std::filesystem::path& dir, std::uint64_t up_to_lsn)
    -> std::expected<void, vesper::core::error> {
  using vesper::core::error; using vesper::core::error_code;
  auto mx = load_manifest(dir);
  if (!mx) return std::vesper_unexpected(mx.error());
  Manifest m = *mx;
  Manifest kept;
  for (const auto& e : m.entries) {
    if (e.end_lsn <= up_to_lsn) {
      std::error_code ec; std::filesystem::remove(dir / e.file, ec);
      if (ec) return std::vesper_unexpected(error{error_code::io_failed, "remove failed", "wal.retention"});
    } else {
      kept.entries.push_back(e);
    }
  }
  if (auto sx = save_manifest(dir, kept); !sx) return std::vesper_unexpected(sx.error());
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
  for (const auto& e : m.entries) {
    auto p = dir / e.file; std::error_code ec; auto ft = std::filesystem::last_write_time(p, ec);
    if (ec) return std::vesper_unexpected(error{error_code::io_failed, "stat failed", "wal.retention"});
    if (ft >= cutoff_time) kept.entries.push_back(e); else {
      std::error_code ec2; std::filesystem::remove(p, ec2);
      if (ec2) return std::vesper_unexpected(error{error_code::io_failed, "remove failed", "wal.retention"});
    }
  }
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

