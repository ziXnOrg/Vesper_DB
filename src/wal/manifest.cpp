#include "vesper/wal/manifest.hpp"

#include <fstream>
#include <sstream>

#ifndef VESPER_ENABLE_MANIFEST_FSYNC
#define VESPER_ENABLE_MANIFEST_FSYNC 0
#endif

namespace vesper::wal {

auto load_manifest(const std::filesystem::path& dir)
    -> std::expected<Manifest, vesper::core::error> {
  using vesper::core::error; using vesper::core::error_code;
  Manifest m{};
  auto p = dir / "wal.manifest";
  std::ifstream in(p);
  if (!in.good()) {
    return std::unexpected(error{error_code::not_found, "manifest open failed", "wal.manifest"});
  }
  std::string header; std::getline(in, header);
  if (header != std::string("vesper-wal-manifest v1")) {
    return std::unexpected(error{error_code::data_integrity, "bad manifest header", "wal.manifest"});
  }
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    ManifestEntry e{};
    std::istringstream iss(line);
    std::string kv;
    while (iss >> kv) {
      auto eq = kv.find('='); if (eq == std::string::npos) continue;
      auto k = kv.substr(0, eq);
      auto v = kv.substr(eq + 1);
      if (k == "file") e.file = v;
      else if (k == "seq") e.seq = std::stoull(v);
      else if (k == "start_lsn") e.start_lsn = std::stoull(v);
      else if (k == "first_lsn") e.first_lsn = std::stoull(v);
      else if (k == "end_lsn") e.end_lsn = std::stoull(v);
      else if (k == "frames") e.frames = std::stoull(v);
      else if (k == "bytes") e.bytes = std::stoull(v);
    }
    if (e.first_lsn == 0) e.first_lsn = e.start_lsn; // backward compat
    m.entries.push_back(std::move(e));
  }
  return m;
}

auto save_manifest(const std::filesystem::path& dir, const Manifest& m)
    -> std::expected<void, vesper::core::error> {
  using vesper::core::error; using vesper::core::error_code;
  auto p = dir / "wal.manifest";
  std::ofstream out(p, std::ios::binary | std::ios::trunc);
  if (!out.good()) {
    return std::unexpected(error{error_code::io_failed, "manifest write failed", "wal.manifest"});
  }
  out << "vesper-wal-manifest v1\n";
  for (const auto& e : m.entries) {
    out << "file=" << e.file
        << " seq=" << e.seq
        << " start_lsn=" << e.start_lsn
        << " first_lsn=" << e.first_lsn
        << " end_lsn=" << e.end_lsn
        << " frames=" << e.frames
        << " bytes=" << e.bytes
        << "\n";
  }
#if VESPER_ENABLE_MANIFEST_FSYNC
  out.flush();
  // No cross-platform fsync for ofstream; this is a guard placeholder. We intentionally skip actual fsync in tests.
#endif
  return {};
}

} // namespace vesper::wal

