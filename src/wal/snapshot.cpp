#include "vesper/wal/snapshot.hpp"

#include <fstream>
#include <string>

#if defined(__linux__) || defined(__APPLE__)
#include <fcntl.h>
#include <unistd.h>
#endif
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
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
  const auto p_tmp = dir / "wal.snapshot.tmp";
  const auto p_dst = dir / "wal.snapshot";
  // 1) Write tmp
  {
    std::ofstream out(p_tmp, std::ios::binary | std::ios::trunc);
    if (!out.good()) return std::vesper_unexpected(error{error_code::io_failed, "snapshot tmp write failed", "wal.snapshot"});
    out << "vesper-wal-snapshot v1\n";
    out << "last_lsn=" << s.last_lsn << "\n";
    out.flush();
  }
  // 2) Ensure tmp contents durable
  #if defined(__linux__) || defined(__APPLE__)
  {
    int fd = ::open(p_tmp.string().c_str(), O_RDONLY);
    if (fd >= 0) {
      (void)::fsync(fd);
      (void)::close(fd);
    } else {
      std::error_code rec; (void)std::filesystem::remove(p_tmp, rec);
      return std::vesper_unexpected(error{error_code::io_failed, "snapshot tmp fsync open failed", "wal.snapshot"});
    }
  }
  #elif defined(_WIN32)
  {
    HANDLE h = ::CreateFileW(p_tmp.wstring().c_str(), GENERIC_READ | GENERIC_WRITE,
                             FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                             nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h != INVALID_HANDLE_VALUE) {
      (void)::FlushFileBuffers(h);
      ::CloseHandle(h);
    }
  }
  #endif
  // 3) Atomic replace
  #if defined(_WIN32)
  {
    BOOL ok = ::MoveFileExW(p_tmp.wstring().c_str(), p_dst.wstring().c_str(),
                            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH);
    if (!ok) {
      std::error_code rec; (void)std::filesystem::remove(p_tmp, rec);
      return std::vesper_unexpected(error{error_code::io_failed, "snapshot replace failed", "wal.snapshot"});
    }
    // 4) Best-effort directory flush
    HANDLE dh = ::CreateFileW(dir.wstring().c_str(), GENERIC_READ,
                              FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                              nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr);
    if (dh != INVALID_HANDLE_VALUE) { (void)::FlushFileBuffers(dh); ::CloseHandle(dh); }
  }
  #else
  {
    std::error_code ec;
    std::filesystem::rename(p_tmp, p_dst, ec);
    if (ec) {
      (void)std::filesystem::remove(p_tmp, ec);
      return std::vesper_unexpected(error{error_code::io_failed, "snapshot rename failed", "wal.snapshot"});
    }
    int dfd = ::open(dir.string().c_str(), O_RDONLY);
    if (dfd >= 0) { (void)::fsync(dfd); (void)::close(dfd); }
  }
  #endif
#else
  // Non-atomic fallback when atomic rename is disabled at compile time
  const auto p = dir / "wal.snapshot";
  std::ofstream out(p, std::ios::binary | std::ios::trunc);
  if (!out.good()) return std::vesper_unexpected(error{error_code::io_failed, "snapshot write failed", "wal.snapshot"});
  out << "vesper-wal-snapshot v1\n";
  out << "last_lsn=" << s.last_lsn << "\n";
#endif
  return {};
}

} // namespace vesper::wal

