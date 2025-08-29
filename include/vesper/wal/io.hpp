#pragma once

/** \file io.hpp
 *  \brief WAL writer and recovery scan (binary file IO). Little-endian framing.
 *
 * Notes
 * - Writer is not thread-safe; one writer per file.
 * - recover_scan is read-only and reentrant for independent paths.
 * - fsync is optional and guarded; tests do not depend on it.
 */

#include <array>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <fstream>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "vesper/error.hpp"
#include "vesper/wal/frame.hpp"

namespace vesper::wal {

/** \brief Statistics gathered during recovery scan. */
struct RecoveryStats {
  std::size_t frames{};             /**< number of valid frames visited */
  std::size_t bytes{};              /**< total bytes of valid frames */
  std::uint64_t last_lsn{};         /**< LSN of the last valid frame */
  bool lsn_monotonic{true};         /**< strictly increasing LSN for TYPE in {1,2} */
  std::size_t lsn_violations{};     /**< count of non-monotonic transitions */
  std::uint32_t min_len{0};         /**< minimum LEN among valid frames (0 if no frames) */
  std::uint32_t max_len{0};         /**< maximum LEN among valid frames (0 if no frames) */
  std::array<std::uint64_t, 4> type_counts{}; /**< type histogram; indices 1..3 used */
};

/** \brief Rotation/open options for WAL writer. */
struct WalWriterOptions {
  std::filesystem::path dir;     /**< directory to place wal-*.log files */
  std::string prefix{"wal-"};   /**< file prefix */
  std::uint64_t max_file_bytes{};/**< rotate before a frame would exceed this size; 0 disables rotation */
};

class WalWriter {
public:
  WalWriter() = default;
  ~WalWriter();
  WalWriter(WalWriter&&) noexcept;
  WalWriter& operator=(WalWriter&&) noexcept;
  WalWriter(const WalWriter&) = delete;
  WalWriter& operator=(const WalWriter&) = delete;

  static auto open(std::string_view path, bool create_if_missing = true)
      -> std::expected<WalWriter, vesper::core::error>;

  static auto open(const WalWriterOptions& opts)
      -> std::expected<WalWriter, vesper::core::error>;

  auto append(std::uint64_t lsn, std::uint16_t type, std::span<const std::uint8_t> payload)
      -> std::expected<void, vesper::core::error>;

  auto flush(bool sync = false) -> std::expected<void, vesper::core::error>;

  const std::filesystem::path& path() const noexcept { return path_; }
  std::uint64_t index() const noexcept { return seq_index_; }

private:
  // single-file mode
  std::filesystem::path path_;
  // rotation mode
  std::filesystem::path dir_;
  std::string prefix_{"wal-"};
  std::uint64_t max_file_bytes_{};
  std::uint64_t seq_index_{}; // current file sequence index
  std::uint64_t cur_bytes_{}; // current file size in bytes
  std::uint64_t cur_frames_{};
  std::uint64_t cur_start_lsn_{};
  std::uint64_t cur_end_lsn_{};

  std::ofstream out_;

  auto maybe_rotate(std::size_t next_frame_bytes) -> std::expected<void, vesper::core::error>;
  auto open_seq(std::uint64_t seq) -> std::expected<void, vesper::core::error>;
};

// Sequentially scans a WAL file and invokes on_frame for each valid frame.
// Stops on torn/truncated tail without error. Monotonicity is warn-only.
auto recover_scan(std::string_view path, std::function<void(const WalFrame&)> on_frame)
    -> std::expected<RecoveryStats, vesper::core::error>;

// Scan a directory of rotated WAL files (manifest-aware). Aggregates stats across files.
// Snapshot semantics: if wal.snapshot exists and parses, frames with lsn <= snapshot.last_lsn are skipped.
auto recover_scan_dir(const std::filesystem::path& dir, std::function<void(const WalFrame&)> on_frame)
    -> std::expected<RecoveryStats, vesper::core::error>;

} // namespace vesper::wal

