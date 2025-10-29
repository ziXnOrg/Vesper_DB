#pragma once

/** \file replay.hpp
 *  \brief Recovery replay API built atop directory scanning and snapshot cutoff.
 */

#include <expected>
#include <filesystem>
#include <functional>
#include <vesper/expected_polyfill.hpp>
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

/** \brief Result-callback enabling early stop/error propagation and precise stats.
 *  Return semantics:
 *  - DeliverAndContinue => payload accepted; counted; continue
 *  - DeliverAndStop     => accepted; counted; stop (ok(stats_so_far))
 *  - Skip/SkipAndStop   => not delivered; not counted; stop if *_AndStop
 *  - unexpected(error)  => stop with error
 *  Example (stop deterministically after N payloads):
 *  @code
 *  std::size_t n = 0;
 *  ReplayResultCallback cb = [&](std::uint64_t, std::uint16_t, std::span<const std::uint8_t>)
 *      -> std::expected<DeliverDecision, vesper::core::error> {
 *    return (++n == 5) ? DeliverDecision::DeliverAndStop
 *                      : DeliverDecision::DeliverAndContinue;
 *  };
 *  auto st = recover_replay(dir, cb);
 *  @endcode
 */
using ReplayResultCallback = std::function<std::expected<DeliverDecision, vesper::core::error>(std::uint64_t lsn, std::uint16_t type, std::span<const std::uint8_t> payload)>;

/** \brief Replays frames' payloads post-snapshot cutoff across rotated files.
 *  Stats reflect only delivered frames; early-stop not applicable for void-callback (always deliver).
 *  \ingroup wal_api
 *  \return Aggregated RecoveryStats for delivered frames or error */
[[nodiscard]] auto recover_replay(const std::filesystem::path& dir, ReplayCallback on_payload)
    -> std::expected<RecoveryStats, vesper::core::error>;

/// Result-callback overload (stats count only delivered frames; early-stop/error supported)
[[nodiscard]] auto recover_replay(const std::filesystem::path& dir, ReplayResultCallback on_payload)
    -> std::expected<RecoveryStats, vesper::core::error>;

/** \brief Replay overload with type mask: only deliver frames with (1u << type) & type_mask.
 *  \note RecoveryStats reflect post-filter delivery; with result-callback, early-stop semantics apply as above.
 *  \ingroup wal_api
 */
[[nodiscard]] auto recover_replay(const std::filesystem::path& dir, std::uint32_t type_mask, ReplayCallback on_payload)
    -> std::expected<RecoveryStats, vesper::core::error>;

/** Result-callback + type mask overload */
[[nodiscard]] auto recover_replay(const std::filesystem::path& dir, std::uint32_t type_mask, ReplayResultCallback on_payload)
    -> std::expected<RecoveryStats, vesper::core::error>;

} // namespace vesper::wal

