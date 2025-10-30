#pragma once

/** \file frame.hpp
 *  \brief WAL frame encode/decode and CRC32C verification (pure, in-memory).
 *
 * Endianness: little-endian framing on all platforms.
 * Thread-safety: functions are stateless and thread-safe.
 * Errors: returned via std::expected with vesper::core::error.
 */

#include <cstdint>
#include <expected>
#include <vesper/expected_polyfill.hpp>
#include <vesper/span_polyfill.hpp>
#include <string>
#include <vector>

#include "vesper/error.hpp"

namespace vesper::wal {

constexpr std::uint32_t WAL_MAGIC = 0x56535741u; // "VSW A"
constexpr std::size_t WAL_HEADER_SIZE = 4 + 4 + 2 + 2 + 8; // 20 bytes

struct WalFrame {
  std::uint32_t magic;
  std::uint32_t len;       // total length including header+payload+CRC
  std::uint16_t type;      // 1=data, 2=commit, 3=padding
  std::uint16_t reserved;  // 0
  std::uint64_t lsn;       // log sequence number
  std::span<const std::uint8_t> payload; // does not own memory
  std::uint32_t crc32c;    // Castagnoli over [magic..payload]
};

// CRC32C (Castagnoli) over the given bytes
auto crc32c(std::span<const std::uint8_t> bytes) -> std::uint32_t;

// Verify CRC32C of a full frame buffer (includes CRC at the end)
auto verify_crc32c(std::span<const std::uint8_t> full_frame) -> bool;

// Encode a frame into a contiguous byte vector (little-endian)
auto encode_frame(std::uint64_t lsn, std::uint16_t type, std::span<const std::uint8_t> payload)
    -> std::vector<std::uint8_t>;


  // Encode a frame; error-returning variant with overflow guards (preferred in writers)
  auto encode_frame_expected(std::uint64_t lsn, std::uint16_t type, std::span<const std::uint8_t> payload)
      -> std::expected<std::vector<std::uint8_t>, core::error>;

// Decode a frame from a contiguous buffer (no allocations for payload)
auto decode_frame(std::span<const std::uint8_t> bytes) -> std::expected<WalFrame, core::error>;

} // namespace vesper::wal

