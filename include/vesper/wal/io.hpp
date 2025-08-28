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
#include <string_view>

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

  auto append(std::uint64_t lsn, std::uint16_t type, std::span<const std::uint8_t> payload)
      -> std::expected<void, vesper::core::error>;

  auto flush(bool sync = false) -> std::expected<void, vesper::core::error>;

  const std::filesystem::path& path() const noexcept { return path_; }

private:
  std::filesystem::path path_;
  std::ofstream out_;
};

// Sequentially scans a WAL file and invokes on_frame for each valid frame.
// Stops on torn/truncated tail without error. Monotonicity is warn-only.
auto recover_scan(std::string_view path, std::function<void(const WalFrame&)> on_frame)
    -> std::expected<RecoveryStats, vesper::core::error>;

} // namespace vesper::wal

