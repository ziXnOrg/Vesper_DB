#pragma once

/** \file replay.hpp
 *  \brief Recovery replay API built atop directory scanning and snapshot cutoff.
 */

#include <expected>
#include <filesystem>
#include <functional>
#include <vesper/span_polyfill.hpp>

#include "vesper/error.hpp"
#include "vesper/wal/frame.hpp"

#include "vesper/wal/io.hpp" // RecoveryStats, recover_scan_dir

namespace vesper::wal {

/** \brief Callback invoked for each delivered frame during replay.
 *  \ingroup wal_api
 *  \param lsn     Frame log sequence number (strictly > snapshot cutoff)
 *  \param type    Frame type (1=data, 2=commit, 3=padding)
 *  \param payload Frame payload bytes
 */
using ReplayCallback = std::function<void(std::uint64_t lsn, std::uint16_t type, std::span<const std::uint8_t> payload)>;

/** \brief Replays frames' payloads post-snapshot cutoff across rotated files.
 *  \ingroup wal_api
 *  \return Aggregated RecoveryStats for delivered frames or error
 */
auto recover_replay(const std::filesystem::path& dir, ReplayCallback on_payload)
    -> std::expected<RecoveryStats, vesper::core::error>;


/** \brief Replay overload with type mask: only deliver frames with (1u << type) & type_mask.
 *  \ingroup wal_api
 *  \note RecoveryStats reflect delivered frames post-filter.
 */
auto recover_replay(const std::filesystem::path& dir, std::uint32_t type_mask, ReplayCallback on_payload)
    -> std::expected<RecoveryStats, vesper::core::error>;

} // namespace vesper::wal

