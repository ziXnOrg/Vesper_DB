#pragma once

/** \file io.hpp
 *  \brief WAL writer and recovery scan (binary file IO). Little-endian framing.
 *
 * Notes
 * - Writer is not thread-safe; one writer per file.
 * - recover_scan is read-only and reentrant for independent paths.
 * - fsync is optional and guarded; when enabled via knobs or flush(true), the writer performs OS-level syncs (fsync/FlushFileBuffers).
 */

#include <array>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <fstream>
#include <functional>
#include <vesper/expected_polyfill.hpp>
#include <vesper/span_polyfill.hpp>
#include <string>
#include <string_view>
#include <vector>

#include "vesper/error.hpp"
#include "vesper/wal/frame.hpp"
#include "vesper/wal/snapshot.hpp"
#include "vesper/wal/retention.hpp"

namespace vesper::wal {

/** \brief Statistics gathered during recovery scan.
 *  \ingroup wal_api
 */

/** \brief Delivery controls for scanning/replay (additive, optional).
 *  cutoff_lsn (>0) overrides snapshot cutoff; type_mask filters delivered types.
 *  max_frames/max_bytes (>0) limit delivery; max_bytes gating uses payload bytes only; stats reflect delivered frames.
 */
struct DeliveryLimits {
  std::uint64_t cutoff_lsn{0};       /**< overrides snapshot if >0 */
  std::uint32_t type_mask{~0u};      /**< bit t enables type==t; default all */
  std::size_t   max_frames{0};       /**< 0 = unlimited */
  std::size_t   max_bytes{0};        /**< 0 = unlimited; gating uses payload bytes only */
};

struct RecoveryStats {
  std::size_t frames{};             /**< number of delivered frames */
  std::size_t bytes{};              /**< total bytes of delivered frames (full frame bytes: header + payload + CRC) */
  std::uint64_t last_lsn{};         /**< LSN of the last valid frame */


  bool lsn_monotonic{true};         /**< strictly increasing LSN for TYPE in {1,2} */
  std::size_t lsn_violations{};     /**< count of non-monotonic transitions */
  std::uint32_t min_len{0};         /**< minimum LEN among valid frames (0 if no frames) */
  std::uint32_t max_len{0};         /**< maximum LEN among valid frames (0 if no frames) */
  std::array<std::uint64_t, 4> type_counts{}; /**< type histogram; indices 1..3 used */
};

struct WalWriterStats {
  std::uint64_t frames{};
  std::uint64_t rotations{};
  std::uint64_t flushes{};
  std::uint64_t syncs{};
};




/** \brief Rotation/open options for WAL writer. */

/** Durability profiles map to fsync knobs (fsync/FlushFileBuffers when enabled). */
enum class DurabilityProfile { None, Rotation, Flush, RotationAndFlush };

/// \brief Delivery decision for accepting-callback scans
/// - DeliverAndContinue: delivered; counted; continue
/// - DeliverAndStop: delivered; counted; stop after this frame
/// - Skip: not delivered; not counted; continue
/// - SkipAndStop: not delivered; not counted; stop
enum class DeliverDecision : std::uint8_t { DeliverAndContinue, DeliverAndStop, Skip, SkipAndStop };

struct WalWriterOptions {
  std::filesystem::path dir;     /**< directory to place wal-*.log files */
  std::string prefix{"wal-"};   /**< file prefix */
  std::uint64_t max_file_bytes{};/**< rotate before a frame would exceed this size; 0 disables rotation */
  bool strict_lsn_monotonic{false}; /**< if true, enforce strictly increasing LSN for TYPE in {1,2} */
  bool fsync_on_rotation{false};     /**< if true, issue a sync when rotating files (portable no-op in tests) */
  bool fsync_on_flush{false};        /**< if true, issue a sync on flush() (portable no-op in tests) */
  std::optional<DurabilityProfile> durability_profile; /**< optional alias to map to fsync knobs */
};

class WalWriter {
public:
  WalWriter() = default;
  ~WalWriter();
  WalWriter(WalWriter&&) noexcept;
  WalWriter& operator=(WalWriter&&) noexcept;
  WalWriter(const WalWriter&) = delete;
  WalWriter& operator=(const WalWriter&) = delete;

  /** If fsync_on_rotation is true, a rotation increments stats_.syncs. */

  static auto open(std::string_view path, bool create_if_missing = true)
      -> std::expected<WalWriter, vesper::core::error>;

  static auto open(const WalWriterOptions& opts)
      -> std::expected<WalWriter, vesper::core::error>;

  auto append(std::uint64_t lsn, std::uint16_t type, std::span<const std::uint8_t> payload)
      -> std::expected<void, vesper::core::error>;

  /** Flush buffered data. If sync=true or options.fsync_on_flush, performs OS-level sync (fsync/FlushFileBuffers) and increments stats_.syncs. */
  auto flush(bool sync = false) -> std::expected<void, vesper::core::error>;

  // Publish a snapshot in rotation mode (writes wal.snapshot in dir_). No fsync; deterministic.
  auto publish_snapshot(std::uint64_t last_lsn) -> std::expected<void, vesper::core::error>;

  const std::filesystem::path& path() const noexcept { return path_; }
  std::uint64_t index() const noexcept { return seq_index_; }
  const WalWriterStats& stats() const noexcept { return stats_; }

private:
  // single-file mode
  std::filesystem::path path_;
  // rotation mode
  std::filesystem::path dir_;
  std::string prefix_{"wal-"};
  std::uint64_t max_file_bytes_{};
  bool strict_lsn_monotonic_{};
  bool fsync_on_rotation_{};
  bool fsync_on_flush_{};
  std::uint64_t seq_index_{}; // current file sequence index


  std::uint64_t cur_bytes_{}; // current file size in bytes
  std::uint64_t cur_frames_{};
  std::uint64_t cur_start_lsn_{};
  std::uint64_t cur_end_lsn_{};
  std::uint64_t prev_lsn_{};
  bool have_prev_{false};
  WalWriterStats stats_{};

  std::ofstream out_;

  auto maybe_rotate(std::size_t next_frame_bytes) -> std::expected<void, vesper::core::error>;
  auto open_seq(std::uint64_t seq) -> std::expected<void, vesper::core::error>;
};

// Sequentially scans a WAL file and invokes on_frame for each valid frame.
// Stops on torn/truncated tail without error. Monotonicity is warn-only.
[[nodiscard]] auto recover_scan(std::string_view path, std::function<void(const WalFrame&)> on_frame)
    -> std::expected<RecoveryStats, vesper::core::error>;

/** \brief Accepting-callback variant.
 *  Callback returns expected<DeliverDecision,error> with semantics:
 *  - DeliverAndContinue: delivered; counted; continue
 *  - DeliverAndStop: delivered; counted; stop (ok(stats_so_far))
 *  - Skip/SkipAndStop: not delivered; not counted; stop if *_AndStop; continue otherwise
 *  - unexpected(error): stop with error
 *  Example:
 *  @code
 *  std::size_t n=0;
 *  auto cb = [&](const WalFrame&){ return (++n==5) ? DeliverAndStop : DeliverAndContinue; };
 *  auto st = recover_scan(path, cb);
 *  @endcode
 */
[[nodiscard]] auto recover_scan(std::string_view path, std::function<std::expected<DeliverDecision, vesper::core::error>(const WalFrame&)> on_frame)
    -> std::expected<RecoveryStats, vesper::core::error>;

// Scan a directory of rotated WAL files (manifest-aware). Aggregates stats across files.
// Snapshot semantics: if wal.snapshot exists and parses, frames with lsn <= snapshot.last_lsn are skipped.
[[nodiscard]] auto recover_scan_dir(const std::filesystem::path& dir, std::function<void(const WalFrame&)> on_frame)
    -> std::expected<RecoveryStats, vesper::core::error>;

/// Accepting-callback directory scan: early-stop/error propagation; stats count only delivered frames
[[nodiscard]] auto recover_scan_dir(const std::filesystem::path& dir, std::function<std::expected<DeliverDecision, vesper::core::error>(const WalFrame&)> on_frame)
    -> std::expected<RecoveryStats, vesper::core::error>;

/// Overload: filter delivered frames by type bitmask (bit t enables type==t); stats reflect post-filter delivery
[[nodiscard]] auto recover_scan_dir(const std::filesystem::path& dir, std::uint32_t type_mask, std::function<void(const WalFrame&)> on_frame)
    -> std::expected<RecoveryStats, vesper::core::error>;

// Overload: type mask with accepting-callback
[[nodiscard]] auto recover_scan_dir(const std::filesystem::path& dir, std::uint32_t type_mask,
                      std::function<std::expected<DeliverDecision, vesper::core::error>(const WalFrame&)> on_frame)
    -> std::expected<RecoveryStats, vesper::core::error>;

/// Overload: delivery controls (cutoff/type/limits); max_bytes gating uses payload bytes; stats reflect delivered frames
[[nodiscard]] auto recover_scan_dir(const std::filesystem::path& dir, const DeliveryLimits& limits, std::function<void(const WalFrame&)> on_frame)
    -> std::expected<RecoveryStats, vesper::core::error>;

// Overload: delivery controls with accepting-callback
[[nodiscard]] auto recover_scan_dir(const std::filesystem::path& dir, const DeliveryLimits& limits,
                      std::function<std::expected<DeliverDecision, vesper::core::error>(const WalFrame&)> on_frame)
    -> std::expected<RecoveryStats, vesper::core::error>;


} // namespace vesper::wal


