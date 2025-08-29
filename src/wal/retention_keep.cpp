#include "vesper/wal/retention.hpp"
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

