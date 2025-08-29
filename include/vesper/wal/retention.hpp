#pragma once

/** \file retention.hpp
 *  \brief WAL retention helpers (purge rotated files by LSN, count, age, or size).
 */

#include <expected>
#include <filesystem>

#include "vesper/error.hpp"

namespace vesper::wal {

// Purge any wal-*.log fully covered by up_to_lsn (based on manifest end_lsn)
auto purge_wal(const std::filesystem::path& dir, std::uint64_t up_to_lsn)
    -> std::expected<void, vesper::core::error>;

// Keep only the last N files by sequence (manifest order), deleting older ones
auto purge_keep_last_n(const std::filesystem::path& dir, std::size_t keep_last_n)
    -> std::expected<void, vesper::core::error>;

// Keep files whose last_write_time >= cutoff_time; delete older ones
// The caller provides cutoff_time to avoid wall-clock usage in tests.
auto purge_keep_newer_than(const std::filesystem::path& dir, std::filesystem::file_time_type cutoff_time)
    -> std::expected<void, vesper::core::error>;

// Keep the newest files such that total bytes (from manifest) <= max_total_bytes
// Deletes older files until the constraint is met, and updates the manifest.
auto purge_keep_total_bytes_max(const std::filesystem::path& dir, std::uint64_t max_total_bytes)
    -> std::expected<void, vesper::core::error>;

} // namespace vesper::wal

