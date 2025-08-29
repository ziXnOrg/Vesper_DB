#pragma once

/** \file manifest.hpp
 *  \brief WAL manifest format and load/save helpers.
 *
 * Format (v1):
 *   header: "vesper-wal-manifest v1"\n
 *   lines:  file=<name> seq=<N> start_lsn=<u64> first_lsn=<u64> end_lsn=<u64> frames=<u64> bytes=<u64>\n
 */

#include <cstdint>
#include <expected>
#include <filesystem>
#include <string>
#include <vector>

#include "vesper/error.hpp"

namespace vesper::wal {

struct ManifestEntry {
  std::string file;          // filename (relative to dir)
  std::uint64_t seq{};       // sequence number
  std::uint64_t start_lsn{}; // first LSN in file (kept for backward compat)
  std::uint64_t first_lsn{}; // explicit alias for first LSN in file
  std::uint64_t end_lsn{};   // last LSN in file
  std::uint64_t frames{};    // number of frames
  std::uint64_t bytes{};     // total bytes
};

struct Manifest { std::vector<ManifestEntry> entries; };

auto load_manifest(const std::filesystem::path& dir)
    -> std::expected<Manifest, vesper::core::error>;

auto save_manifest(const std::filesystem::path& dir, const Manifest& m)
    -> std::expected<void, vesper::core::error>;

} // namespace vesper::wal

