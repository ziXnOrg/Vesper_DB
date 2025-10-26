#include "vesper/wal/snapshot.hpp"

#include <fstream>
#include <string>

#if defined(__linux__) || defined(__APPLE__)
#include <fcntl.h>
#include <unistd.h>
#endif

#ifndef VESPER_ENABLE_ATOMIC_RENAME
#if defined(_WIN32) || defined(__linux__) || defined(__APPLE__)
#define VESPER_ENABLE_ATOMIC_RENAME 1
#else
#define VESPER_ENABLE_ATOMIC_RENAME 0
#endif
#endif

namespace vesper::wal {

auto load_snapshot(const std::filesystem::path& dir)
    -> std::expected<Snapshot, vesper::core::error> {
  using vesper::core::error; using vesper::core::error_code;
  auto p = dir / "wal.snapshot";
  std::ifstream in(p);
  if (!in.good()) return std::vesper_unexpected(error{error_code::not_found, "snapshot open failed", "wal.snapshot"});
  std::string header; std::getline(in, header);
  if (header != std::string("vesper-wal-snapshot v1")) {
    return std::vesper_unexpected(error{error_code::data_integrity, "bad snapshot header", "wal.snapshot"});
  }
  std::string line; std::getline(in, line);
  if (line.rfind("last_lsn=", 0) != 0) {
    return std::vesper_unexpected(error{error_code::data_integrity, "missing last_lsn", "wal.snapshot"});
  }
  std::uint64_t last = 0;
  try {
    last = std::stoull(line.substr(9));
  } catch (...) {
    return std::vesper_unexpected(error{error_code::data_integrity, "malformed last_lsn", "wal.snapshot"});
  }
  return Snapshot{last};
}

auto save_snapshot(const std::filesystem::path& dir, const Snapshot& s)
    -> std::expected<void, vesper::core::error> {
  using vesper::core::error; using vesper::core::error_code;
#if VESPER_ENABLE_ATOMIC_RENAME
  auto p_tmp = dir / "wal.snapshot.tmp";
  {
    std::ofstream out(p_tmp, std::ios::binary | std::ios::trunc);
    if (!out.good()) return std::vesper_unexpected(error{error_code::io_failed, "snapshot tmp write failed", "wal.snapshot"});
    out << "vesper-wal-snapshot v1\n";
    out << "last_lsn=" << s.last_lsn << "\n";
  }
  std::error_code ec;
  std::filesystem::rename(p_tmp, dir / "wal.snapshot", ec);
  if (ec) {
    // Fallback to simple write
    auto p = dir / "wal.snapshot";
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    if (!out.good()) return std::vesper_unexpected(error{error_code::io_failed, "snapshot write failed", "wal.snapshot"});
    out << "vesper-wal-snapshot v1\n";
    out << "last_lsn=" << s.last_lsn << "\n";
  } else {
    // Best-effort: ensure directory entry is durable on POSIX
    #if defined(__linux__) || defined(__APPLE__)
    int dfd = ::open(dir.c_str(), O_RDONLY);
    if (dfd >= 0) {
      (void)::fsync(dfd);
      (void)::close(dfd);
    }
    #endif
  }
#else
  auto p = dir / "wal.snapshot";
  std::ofstream out(p, std::ios::binary | std::ios::trunc);
  if (!out.good()) return std::vesper_unexpected(error{error_code::io_failed, "snapshot write failed", "wal.snapshot"});
  out << "vesper-wal-snapshot v1\n";
  out << "last_lsn=" << s.last_lsn << "\n";
#endif
  return {};
}

} // namespace vesper::wal

