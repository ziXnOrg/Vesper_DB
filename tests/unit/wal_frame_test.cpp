#include <catch2/catch_all.hpp>
#include <vesper/wal/frame.hpp>

TEST_CASE("crc32c round-trip and frame idempotence", "[wal]") {
  using namespace vesper::wal;
  std::vector<std::uint8_t> payload = {0xDE,0xAD,0xBE,0xEF};
  auto bytes = encode_frame(1, /*type=*/1, payload);
  REQUIRE(verify_crc32c(bytes));
  auto dec = decode_frame(bytes);
  REQUIRE(dec.has_value());
  REQUIRE(dec->magic == WAL_MAGIC);
  REQUIRE(dec->len == bytes.size());
  REQUIRE(dec->type == 1);
  REQUIRE(dec->lsn == 1);
  REQUIRE(dec->payload.size() == payload.size());
  auto re = encode_frame(dec->lsn, dec->type, dec->payload);
  REQUIRE(re == bytes);
}

TEST_CASE("crc mismatch yields data_integrity error", "[wal]") {
  using namespace vesper::wal;
  std::vector<std::uint8_t> payload = {0x01,0x02,0x03};
  auto bytes = encode_frame(2, 1, payload);
  bytes.back() ^= 0xFF; // corrupt CRC
  auto dec = decode_frame(bytes);
  REQUIRE_FALSE(dec.has_value());
  REQUIRE(dec.error().code == vesper::core::error_code::data_integrity);
}



#include <limits>
#include <array>
#include <cstring>

TEST_CASE("crc32c known-answer (Castagnoli, reflected)", "[wal][crc32c]") {
  using namespace vesper::wal;
  const char* s = "123456789";
  std::span<const std::uint8_t> bytes{reinterpret_cast<const std::uint8_t*>(s), 9};
  REQUIRE(crc32c(bytes) == 0xE3069283u);
}

TEST_CASE("encode_frame_expected guards length overflow", "[wal][safety]") {
  using namespace vesper::wal;
  // Construct a span with size just over the max representable payload for 32-bit length
  const std::size_t max_payload = std::numeric_limits<std::uint32_t>::max() - WAL_HEADER_SIZE - 4;
  const std::size_t too_big = max_payload + 1;
  std::uint8_t dummy = 0;
  std::span<const std::uint8_t> payload{&dummy, too_big};
  auto enc = encode_frame_expected(/*lsn=*/42, /*type=*/1, payload);
  REQUIRE_FALSE(enc.has_value());
  REQUIRE(enc.error().code == vesper::core::error_code::invalid_argument);
}

namespace { // test-local legacy CRC helper (mirrors historical bug for migration acceptance)
static std::uint32_t crc32c_legacy(std::span<const std::uint8_t> bytes) {
  static const std::array<std::uint32_t, 256> T = []{
    std::array<std::uint32_t, 256> t{};
    const std::uint32_t poly = 0x1EDC6F41u; // wrong for reflected update
    for (std::uint32_t i = 0; i < 256; ++i) {
      std::uint32_t c = i;
      for (int k = 0; k < 8; ++k) {
        c = (c & 1u) ? (poly ^ (c >> 1)) : (c >> 1);
      }
      t[i] = c;
    }
    return t;
  }();
  std::uint32_t c = ~0u;
  for (auto b : bytes) c = T[(c ^ b) & 0xFFu] ^ (c >> 8);
  return ~c;
}
} // namespace

TEST_CASE("verify accepts legacy CRC during migration", "[wal][crc32c][migration]") {
  using namespace vesper::wal;
  // Build a frame with CRC computed using legacy orientation
  const std::vector<std::uint8_t> payload = {'a','b','c'};
  const std::uint32_t len = static_cast<std::uint32_t>(WAL_HEADER_SIZE + payload.size() + 4);
  std::vector<std::uint8_t> frame(len);
  std::uint8_t* p = frame.data();
  auto store_le32 = [&](std::uint32_t v){ std::memcpy(p, &v, 4); p += 4; };
  auto store_le16 = [&](std::uint16_t v){ std::memcpy(p, &v, 2); p += 2; };
  auto store_le64 = [&](std::uint64_t v){ std::memcpy(p, &v, 8); p += 8; };
  store_le32(WAL_MAGIC);
  store_le32(len);
  store_le16(1);
  store_le16(0);
  store_le64(123);
  std::memcpy(p, payload.data(), payload.size()); p += payload.size();
  const std::uint32_t legacy = crc32c_legacy({frame.data(), frame.size() - 4});
  store_le32(legacy);
  REQUIRE(verify_crc32c(frame));
  auto dec = decode_frame(frame);
  REQUIRE(dec.has_value());
  REQUIRE(dec->len == len);
  REQUIRE(dec->payload.size() == payload.size());
}
