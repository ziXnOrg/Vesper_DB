#include <catch2/catch_all.hpp>
#include <vesper/wal/io.hpp>
#include <vesper/wal/frame.hpp>
#include <filesystem>

using namespace vesper;

TEST_CASE("WAL write/read happy path", "[wal][io]") {
  namespace fs = std::filesystem;
  auto tmp = fs::temp_directory_path() / "vesper_wal_test.bin";
  // Ensure clean file
  std::error_code ec; fs::remove(tmp, ec);

  auto writer = wal::WalWriter::open(tmp.string());
  REQUIRE(writer.has_value());

  std::vector<std::uint8_t> p1{0xAA,0xBB};
  std::vector<std::uint8_t> p2{0xCC,0xDD,0xEE};
  REQUIRE(writer->append(1, 1, p1).has_value());
  REQUIRE(writer->append(2, 1, p2).has_value());
  REQUIRE(writer->flush(false).has_value());

  std::vector<wal::WalFrame> frames;
  auto stats_exp = wal::recover_scan(writer->path().string(), [&](const wal::WalFrame& f){ frames.push_back(f); });
  REQUIRE(stats_exp.has_value());
  auto stats = *stats_exp;
  REQUIRE(stats.frames == 2);
  REQUIRE(frames.size() == 2);
  REQUIRE(frames[0].lsn == 1);
  REQUIRE(frames[1].lsn == 2);
  fs::remove(tmp, ec);
}

TEST_CASE("WAL torn tail is ignored", "[wal][io]") {
  namespace fs = std::filesystem;
  auto tmp = fs::temp_directory_path() / "vesper_wal_torn.bin";
  std::error_code ec; fs::remove(tmp, ec);

  auto writer = wal::WalWriter::open(tmp.string());
  REQUIRE(writer.has_value());
  std::vector<std::uint8_t> p{0xDE,0xAD,0xBE,0xEF};
  REQUIRE(writer->append(1, 1, p).has_value());
  REQUIRE(writer->flush(false).has_value());

  // Manually append a frame then truncate 1 byte
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::app);
    auto bytes = wal::encode_frame(2, 1, p);
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    REQUIRE(out.good());
    out.close();
    // Truncate by 1 byte
    auto sz = fs::file_size(tmp);
    fs::resize_file(tmp, sz - 1, ec);
    REQUIRE(!ec);
  }

  std::vector<wal::WalFrame> frames;
  auto stats_exp = wal::recover_scan(tmp.string(), [&](const wal::WalFrame& f){ frames.push_back(f); });
  REQUIRE(stats_exp.has_value());
  REQUIRE(frames.size() == 1);
  REQUIRE(frames[0].lsn == 1);
  fs::remove(tmp, ec);
}



TEST_CASE("WAL LSN monotonicity warn-only and stats", "[wal][io][stats]") {
  namespace fs = std::filesystem;
  auto tmp = fs::temp_directory_path() / "vesper_wal_lsn.bin";
  std::error_code ec; fs::remove(tmp, ec);
  auto w = wal::WalWriter::open(tmp.string());
  REQUIRE(w.has_value());
  std::vector<std::uint8_t> p{0x01};
  REQUIRE(w->append(10, 1, p).has_value());
  REQUIRE(w->append(8, 1, p).has_value()); // violation: 8 < 10
  REQUIRE(w->flush(false).has_value());
  std::size_t total_payload = 0;
  auto stats_exp = wal::recover_scan(tmp.string(), [&](const wal::WalFrame& f){ total_payload += f.payload.size(); });
  REQUIRE(stats_exp.has_value());
  auto s = *stats_exp;
  REQUIRE(s.frames == 2);
  REQUIRE(s.lsn_monotonic == false);
  REQUIRE(s.lsn_violations == 1);
  REQUIRE(s.type_counts[1] == 2);
  REQUIRE(s.min_len <= s.max_len);
  REQUIRE(s.last_lsn == 8);
  REQUIRE(total_payload == 2); // 1 byte each
  fs::remove(tmp, ec);
}
