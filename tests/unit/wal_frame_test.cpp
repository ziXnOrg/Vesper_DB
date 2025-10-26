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

