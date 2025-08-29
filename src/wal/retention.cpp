#include "vesper/wal/retention.hpp"
#include "vesper/wal/manifest.hpp"

namespace vesper::wal {

auto purge_wal(const std::filesystem::path& dir, std::uint64_t up_to_lsn)
    -> std::expected<void, vesper::core::error> {
  using vesper::core::error; using vesper::core::error_code;
  auto mx = load_manifest(dir);
  if (!mx) return std::unexpected(mx.error());
  Manifest m = *mx;
  Manifest kept;
  for (const auto& e : m.entries) {
    if (e.end_lsn <= up_to_lsn) {
      std::error_code ec; std::filesystem::remove(dir / e.file, ec);
      if (ec) return std::unexpected(error{error_code::io_failed, "remove failed", "wal.retention"});
    } else {
      kept.entries.push_back(e);
    }
  }
  if (auto sx = save_manifest(dir, kept); !sx) return std::unexpected(sx.error());
  return {};
}

} // namespace vesper::wal

