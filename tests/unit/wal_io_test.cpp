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

