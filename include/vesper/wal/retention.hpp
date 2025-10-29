#pragma once

/** \file retention.hpp
 *  \brief WAL retention helpers (purge rotated files by LSN, count, age, or size).
 */

#include <expected>
#include <filesystem>

#include "vesper/error.hpp"

namespace vesper::wal {

// Purge any wal-*.log fully covered by up_to_lsn (based on manifest end_lsn)
[[nodiscard]] auto purge_wal(const std::filesystem::path& dir, std::uint64_t up_to_lsn)
    -> std::expected<void, vesper::core::error>;

/**
 * Keep only the N newest rotated WAL files.
 *
 * Deterministic ordering policy (applies to all retention helpers):
 * - Order entries by end_lsn (descending, newest first)
 * - For ties on end_lsn, order by filename lexicographically descending
 *
 * "Last N" means the N newest files according to the above order (not by
 * sequence number or manifest insertion order). Older files are deleted and
 * the manifest is updated accordingly.
 */
[[nodiscard]] auto purge_keep_last_n(const std::filesystem::path& dir, std::size_t keep_last_n)
    -> std::expected<void, vesper::core::error>;

/**
 * Keep all files with last_write_time > cutoff_time; among files exactly at cutoff_time,
 * keep exactly one chosen by unified order (highest end_lsn, then lexicographically last filename).
 *
 * The caller supplies cutoff_time to avoid wall-clock dependencies in tests.
 */
[[nodiscard]] auto purge_keep_newer_than(const std::filesystem::path& dir, std::filesystem::file_time_type cutoff_time)
    -> std::expected<void, vesper::core::error>;

/**
 * Keep the newest files such that the total bytes (from manifest) <= max_total_bytes.
 *
 * Always retains the newest file (by the unified order) even if its size alone
 * exceeds max_total_bytes; then accumulates additional files while
 * (accumulated_bytes + next.bytes) <= max_total_bytes. Older files are deleted
 * and the manifest is updated accordingly.
 */
[[nodiscard]] auto purge_keep_total_bytes_max(const std::filesystem::path& dir, std::uint64_t max_total_bytes)
    -> std::expected<void, vesper::core::error>;

} // namespace vesper::wal

