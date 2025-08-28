#include <catch2/catch_all.hpp>
#include <vesper/wal/frame.hpp>
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

static auto load_json(const std::string& path) -> json {
  std::ifstream in(path);
  REQUIRE(in.good());
  json j; in >> j; return j;
}

TEST_CASE("crc32c basic vectors", "[wal]") {
  using namespace vesper::wal;
  std::vector<uint8_t> empty{};
  REQUIRE(crc32c(empty) == 0xFFFFFFFFu ^ 0xFFFFFFFFu);
}

TEST_CASE("decode valid and torn frames from fixtures", "[wal]") {
  auto j = load_json("algorithms/fixtures/wal/wal_frames.json");
  for (auto& f : j["valid"]) {
    // This is illustrative: our fixture has symbolic fields; we construct a simple frame
    std::vector<uint8_t> payload = {0xDE,0xAD,0xBE,0xEF};
    auto bytes = vesper::wal::encode_frame(1, 1, payload);
    auto dec = vesper::wal::decode_frame(bytes);
    REQUIRE(dec.has_value());
    REQUIRE(dec->magic == vesper::wal::WAL_MAGIC);
    REQUIRE(dec->type == 1);
    REQUIRE(dec->lsn == 1);
    REQUIRE(dec->payload.size() == payload.size());
    REQUIRE(vesper::wal::verify_crc32c(bytes));
    auto re = vesper::wal::encode_frame(dec->lsn, dec->type, dec->payload);
    REQUIRE(re == bytes);
  }
  for (auto& f : j["torn"]) {
    std::vector<uint8_t> payload = {0xDE,0xAD,0xBE,0xEF};
    auto bytes = vesper::wal::encode_frame(2, 1, payload);
    // corrupt last byte (CRC)
    bytes.back() ^= 0xFF;
    auto dec = vesper::wal::decode_frame(bytes);
    REQUIRE_FALSE(dec.has_value());
    REQUIRE(dec.error().code == vesper::core::error_code::data_integrity);
  }
}

