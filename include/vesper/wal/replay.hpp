#pragma once

/** \file replay.hpp
 *  \brief Recovery replay API built atop directory scanning and snapshot cutoff.
 */

#include <expected>
#include <filesystem>
#include <functional>
#include <span>

#include "vesper/error.hpp"
#include "vesper/wal/frame.hpp"

#include "vesper/wal/io.hpp" // RecoveryStats, recover_scan_dir

namespace vesper::wal {

// Replays frames' payloads post-snapshot cutoff in order across rotated files.
// on_payload is invoked with (lsn, type, payload) for each delivered frame.
using ReplayCallback = std::function<void(std::uint64_t lsn, std::uint16_t type, std::span<const std::uint8_t> payload)>;

auto recover_replay(const std::filesystem::path& dir, ReplayCallback on_payload)
    -> std::expected<RecoveryStats, vesper::core::error>;

} // namespace vesper::wal

