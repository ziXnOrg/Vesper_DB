#include "vesper/wal/retention.hpp"
#include <numeric>

#include "vesper/wal/manifest.hpp"

#include <vector>
#include <algorithm>

namespace vesper::wal {

auto purge_keep_last_n(const std::filesystem::path& dir, std::size_t keep_last_n)
    -> std::expected<void, vesper::core::error> {
  using vesper::core::error; using vesper::core::error_code;
  auto mx = load_manifest(dir);
  if (!mx) return std::unexpected(mx.error());
  Manifest m = *mx;
  if (m.entries.size() <= keep_last_n) return {};
  const auto cutoff_index = (m.entries.size() - keep_last_n);
  for (std::size_t i = 0; i < cutoff_index; ++i) {
    std::error_code ec; std::filesystem::remove(dir / m.entries[i].file, ec);
    if (ec) return std::unexpected(error{error_code::io_failed, "remove failed", "wal.retention"});
  }
  Manifest kept;
  kept.entries.insert(kept.entries.end(), m.entries.begin() + cutoff_index, m.entries.end());
  if (auto sx = save_manifest(dir, kept); !sx) return std::unexpected(sx.error());
  return {};
}

auto purge_keep_newer_than(const std::filesystem::path& dir, std::filesystem::file_time_type cutoff_time)
    -> std::expected<void, vesper::core::error> {
  using vesper::core::error; using vesper::core::error_code;
  auto mx = load_manifest(dir);
  if (!mx) return std::unexpected(mx.error());
  Manifest m = *mx;
  Manifest kept;
  for (const auto& e : m.entries) {
    std::error_code ec; auto p = dir / e.file; auto ft = std::filesystem::last_write_time(p, ec);
    if (ec) return std::unexpected(error{error_code::io_failed, "stat failed", "wal.retention"});
    if (ft >= cutoff_time) {
      kept.entries.push_back(e);
    } else {
      std::error_code ec2; std::filesystem::remove(p, ec2);
      if (ec2) return std::unexpected(error{error_code::io_failed, "remove failed", "wal.retention"});
    }
  }
  if (auto sx = save_manifest(dir, kept); !sx) return std::unexpected(sx.error());
  return {};
}

} // namespace vesper::wal

auto purge_keep_total_bytes_max(const std::filesystem::path& dir, std::uint64_t max_total_bytes)
    -> std::expected<void, vesper::core::error> {
  using vesper::core::error; using vesper::core::error_code;
  auto mx = load_manifest(dir);
  if (!mx) return std::unexpected(mx.error());
  Manifest m = *mx;
  std::uint64_t acc = 0;
  std::vector<std::size_t> keep_indices;
  keep_indices.reserve(m.entries.size());
  for (std::size_t i = m.entries.size(); i > 0; --i) {
    const auto idx = i - 1;
    const auto bytes = m.entries[idx].bytes;
    if (acc + bytes > max_total_bytes && !keep_indices.empty()) break;
    keep_indices.push_back(idx);
    acc += bytes;
  }
  std::sort(keep_indices.begin(), keep_indices.end());

  Manifest kept;
  kept.entries.reserve(keep_indices.size());
  std::size_t next = 0;
  for (std::size_t i = 0; i < m.entries.size(); ++i) {
    if (next < keep_indices.size() && i == keep_indices[next]) {
      kept.entries.push_back(m.entries[i]);
      ++next;
    } else {
      std::error_code ec; std::filesystem::remove(dir / m.entries[i].file, ec);
      if (ec) return std::unexpected(error{error_code::io_failed, "remove failed", "wal.retention"});
    }
  }
  if (auto sx = save_manifest(dir, kept); !sx) return std::unexpected(sx.error());
  return {};
}


