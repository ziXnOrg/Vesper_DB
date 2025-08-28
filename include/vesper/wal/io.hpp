#pragma once

/** \file io.hpp
 *  \brief WAL writer and recovery scan (binary file IO). Little-endian framing.
 *
 * Notes
 * - Writer is not thread-safe; one writer per file.
 * - recover_scan is read-only and reentrant for independent paths.
 * - fsync is optional and guarded; tests do not depend on it.
 */

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

struct RecoveryStats {
  std::size_t frames{};
  std::size_t bytes{};
  std::uint64_t last_lsn{};
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
// Stops on torn/truncated tail without error.
auto recover_scan(std::string_view path, std::function<void(const WalFrame&)> on_frame)
    -> std::expected<RecoveryStats, vesper::core::error>;

} // namespace vesper::wal

