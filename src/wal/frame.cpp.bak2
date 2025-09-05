#include "vesper/wal/frame.hpp"

#include <array>
#include <cstring>

namespace vesper::wal {

static constexpr std::array<std::uint32_t, 256> CRC32C_TABLE = []{
  std::array<std::uint32_t, 256> t{};
  const std::uint32_t poly = 0x1EDC6F41u; // Castagnoli
  for (std::uint32_t i = 0; i < 256; ++i) {
    std::uint32_t c = i;
    for (int k = 0; k < 8; ++k) {
      c = (c & 1u) ? (poly ^ (c >> 1)) : (c >> 1);
    }
    t[i] = c;
  }
  return t;
}();

auto crc32c(std::span<const std::uint8_t> bytes) -> std::uint32_t {
  std::uint32_t c = ~0u;
  for (auto b : bytes) {
    c = CRC32C_TABLE[(c ^ b) & 0xFFu] ^ (c >> 8);
  }
  return ~c;
}

static auto load_le32(const std::uint8_t* p) -> std::uint32_t {
  std::uint32_t v;
  std::memcpy(&v, p, 4);
  return v;
}
static auto load_le16(const std::uint8_t* p) -> std::uint16_t {
  std::uint16_t v;
  std::memcpy(&v, p, 2);
  return v;
}
static auto load_le64(const std::uint8_t* p) -> std::uint64_t {
  std::uint64_t v;
  std::memcpy(&v, p, 8);
  return v;
}

auto verify_crc32c(std::span<const std::uint8_t> full_frame) -> bool {
  if (full_frame.size() < WAL_HEADER_SIZE + 4) return false;
  const std::size_t n = full_frame.size();
  const std::uint32_t expect = load_le32(full_frame.data() + n - 4);
  const std::uint32_t got = crc32c(full_frame.first(n - 4));
  return expect == got;
}

auto encode_frame(std::uint64_t lsn, std::uint16_t type, std::span<const std::uint8_t> payload)
    -> std::vector<std::uint8_t> {
  const std::uint32_t len = static_cast<std::uint32_t>(WAL_HEADER_SIZE + payload.size() + 4);
  std::vector<std::uint8_t> out(len);
  std::uint8_t* p = out.data();
  auto store_le32 = [&](std::uint32_t v){ std::memcpy(p, &v, 4); p += 4; };
  auto store_le16 = [&](std::uint16_t v){ std::memcpy(p, &v, 2); p += 2; };
  auto store_le64 = [&](std::uint64_t v){ std::memcpy(p, &v, 8); p += 8; };

  store_le32(WAL_MAGIC);
  store_le32(len);
  store_le16(type);
  store_le16(0);
  store_le64(lsn);
  if (!payload.empty()) { std::memcpy(p, payload.data(), payload.size()); p += payload.size(); }
  const std::uint32_t c = crc32c({out.data(), out.size() - 4});
  store_le32(c);
  return out;
}

auto decode_frame(std::span<const std::uint8_t> bytes) -> std::expected<WalFrame, core::error> {
  using vesper::core::error;
  using vesper::core::error_code;
  if (bytes.size() < WAL_HEADER_SIZE + 4) {
    return std::unexpected(error{error_code::precondition_failed, "frame too short", "wal.frame"});
  }
  const std::uint8_t* p = bytes.data();
  const std::uint32_t magic = load_le32(p); p += 4;
  const std::uint32_t len = load_le32(p); p += 4;
  const std::uint16_t type = load_le16(p); p += 2;
  const std::uint16_t reserved = load_le16(p); p += 2;
  const std::uint64_t lsn = load_le64(p); p += 8;

  if (magic != WAL_MAGIC) {
    return std::unexpected(error{error_code::data_integrity, "bad magic", "wal.frame"});
  }
  if (len != bytes.size()) {
    return std::unexpected(error{error_code::precondition_failed, "len mismatch", "wal.frame"});
  }
  if (len < WAL_HEADER_SIZE + 4) {
    return std::unexpected(error{error_code::precondition_failed, "len too small", "wal.frame"});
  }
  if (reserved != 0) {
    return std::unexpected(error{error_code::precondition_failed, "reserved != 0", "wal.frame"});
  }
  if (!verify_crc32c(bytes)) {
    return std::unexpected(error{error_code::data_integrity, "crc mismatch", "wal.frame"});
  }
  const std::size_t payload_len = len - WAL_HEADER_SIZE - 4;
  std::span<const std::uint8_t> payload{bytes.data() + WAL_HEADER_SIZE, payload_len};
  WalFrame f{magic, len, type, reserved, lsn, payload, load_le32(bytes.data() + len - 4)};
  return f;
}

} // namespace vesper::wal

