#include "vesper/wal/manifest.hpp"
#include "vesper/wal/io.hpp" // recover_scan, WalFrame


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


#include <regex>
#include <utility>
#include <algorithm>
#include <vector>

namespace {
using Pair = std::pair<std::uint64_t, std::filesystem::path>;
static std::vector<Pair> list_sorted(const std::filesystem::path& dir, const std::string& prefix){
  std::vector<Pair> v;
  std::regex rx(std::string("^") + prefix + "([0-9]{8})\\.log$");
  for (auto& de : std::filesystem::directory_iterator(dir)){
    if (!de.is_regular_file()) continue; auto name=de.path().filename().string();
    std::smatch m; if (std::regex_match(name, m, rx) && m.size()==2){
      try { auto seq = static_cast<std::uint64_t>(std::stoull(m[1].str())); v.emplace_back(seq, de.path()); } catch(...) {}
    }
  }
  std::sort(v.begin(), v.end(), [](auto&a, auto&b){ return a.first < b.first; });
  return v;
}
}

namespace vesper::wal {

auto rebuild_manifest(const std::filesystem::path& dir)
    -> std::expected<Manifest, vesper::core::error> {
  using vesper::core::error; using vesper::core::error_code;
  Manifest m{};
  auto files = list_sorted(dir, "wal-");
  for (auto& kv : files){
    // Scan this file and accumulate stats
    std::size_t frames=0; std::size_t bytes=0; std::uint64_t first_lsn=0; std::uint64_t last_lsn=0;
    auto st = recover_scan(kv.second.string(), [&](const WalFrame& f){
      frames++; bytes += f.len; if (first_lsn==0) first_lsn = f.lsn; last_lsn = f.lsn;
    });
    if (!st) return std::unexpected(st.error());
    ManifestEntry e{};
    e.file = kv.second.filename().string();
    e.seq = kv.first;
    e.start_lsn = first_lsn; e.first_lsn = first_lsn; e.end_lsn = last_lsn;
    e.frames = frames; e.bytes = bytes;
    m.entries.push_back(e);
  }
  return m;
}

} // namespace vesper::wal
