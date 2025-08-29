#include "vesper/wal/snapshot.hpp"

#include <fstream>
#include <string>

namespace vesper::wal {

auto load_snapshot(const std::filesystem::path& dir)
    -> std::expected<Snapshot, vesper::core::error> {
  using vesper::core::error; using vesper::core::error_code;
  auto p = dir / "wal.snapshot";
  std::ifstream in(p);
  if (!in.good()) return std::unexpected(error{error_code::not_found, "snapshot open failed", "wal.snapshot"});
  std::string header; std::getline(in, header);
  if (header != std::string("vesper-wal-snapshot v1")) {
    return std::unexpected(error{error_code::data_integrity, "bad snapshot header", "wal.snapshot"});
  }
  std::string line; std::getline(in, line);
  if (line.rfind("last_lsn=", 0) != 0) {
    return std::unexpected(error{error_code::data_integrity, "missing last_lsn", "wal.snapshot"});
  }
  std::uint64_t last = 0;
  try {
    last = std::stoull(line.substr(9));
  } catch (...) {
    return std::unexpected(error{error_code::data_integrity, "malformed last_lsn", "wal.snapshot"});
  }
  return Snapshot{last};
}

auto save_snapshot(const std::filesystem::path& dir, const Snapshot& s)
    -> std::expected<void, vesper::core::error> {
  using vesper::core::error; using vesper::core::error_code;
  auto p = dir / "wal.snapshot";
  std::ofstream out(p, std::ios::binary | std::ios::trunc);
  if (!out.good()) return std::unexpected(error{error_code::io_failed, "snapshot write failed", "wal.snapshot"});
  out << "vesper-wal-snapshot v1\n";
  out << "last_lsn=" << s.last_lsn << "\n";
  return {};
}

} // namespace vesper::wal

