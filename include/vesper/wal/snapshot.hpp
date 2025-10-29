#pragma once

/** \file snapshot.hpp
 *  \brief Snapshot load/save helpers for WAL recovery.
 *
 * Format (v1):
 *   vesper-wal-snapshot v1\n
 *   last_lsn=<u64>\n
 * Atomic, durable save (platform-correct):
 * - Write contents to a temporary sibling file (wal.snapshot.tmp) in the same directory
 * - Flush stream buffers and ensure file-level durability:
 *     POSIX: fsync(tmp)
 *     Windows: FlushFileBuffers(tmp)
 * - Atomically replace the destination:
 *     POSIX: std::filesystem::rename(tmp, wal.snapshot) (rename(2), replaces if exists)
 *     Windows: MoveFileExW(tmp, wal.snapshot, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)
 * - Best-effort directory durability:
 *     POSIX: fsync(parent directory)
 *     Windows: FlushFileBuffers(directory handle opened with FILE_FLAG_BACKUP_SEMANTICS)
 * - On failure, wal.snapshot.tmp is removed and an io_failed error is returned (no non-atomic fallback).
 *
 * When VESPER_ENABLE_ATOMIC_RENAME is disabled at compile time, save_snapshot falls back
 * to a simple truncating write of wal.snapshot without atomic replacement.
 */

#include <cstdint>
#include <expected>
#include <filesystem>
#include <vesper/expected_polyfill.hpp>

#include "vesper/error.hpp"

namespace vesper::wal {

struct Snapshot { std::uint64_t last_lsn{}; };

auto load_snapshot(const std::filesystem::path& dir)
    -> std::expected<Snapshot, vesper::core::error>;

auto save_snapshot(const std::filesystem::path& dir, const Snapshot& s)
    -> std::expected<void, vesper::core::error>;

} // namespace vesper::wal

