#pragma once

/** \file snapshot.hpp
 *  \brief Snapshot load/save helpers for WAL recovery.
 *
 * Format (v1):
 *   vesper-wal-snapshot v1\n
 *   last_lsn=<u64>\n
 * Atomic save
 * - When VESPER_ENABLE_ATOMIC_RENAME is enabled (default on supported platforms), save_snapshot writes
 *   to a temporary file in the same directory and then renames it to wal.snapshot.
 * - Otherwise, it falls back to a simple truncating write.
 */

#include <cstdint>
#include <expected>
#include <filesystem>

#include "vesper/error.hpp"

namespace vesper::wal {

struct Snapshot { std::uint64_t last_lsn{}; };

auto load_snapshot(const std::filesystem::path& dir)
    -> std::expected<Snapshot, vesper::core::error>;

auto save_snapshot(const std::filesystem::path& dir, const Snapshot& s)
    -> std::expected<void, vesper::core::error>;

} // namespace vesper::wal

