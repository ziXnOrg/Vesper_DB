#include <catch2/catch_all.hpp>
#include <vesper/wal/manifest.hpp>
#include <vesper/wal/io.hpp>
#include <filesystem>

using namespace vesper;

TEST_CASE("manifest includes first_lsn and start_lsn for entries", "[wal][manifest]") {
  namespace fs = std::filesystem;
  auto dir = fs::temp_directory_path() / "vesper_wal_manifest_firstlsn";
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);
  wal::WalWriterOptions opts{.dir=dir, .prefix="wal-", .max_file_bytes=64};
  auto w = wal::WalWriter::open(opts); REQUIRE(w.has_value());
  std::vector<uint8_t> p{0x01};
  REQUIRE(w->append(10, 1, p).has_value());
  REQUIRE(w->append(11, 1, p).has_value());
  REQUIRE(w->flush(false).has_value());

  auto m = wal::load_manifest(dir);
  REQUIRE(m.has_value());
  REQUIRE(m->entries.size() >= 1);
  for (const auto& e : m->entries) {
    REQUIRE(e.first_lsn >= e.start_lsn);
    REQUIRE(e.end_lsn >= e.first_lsn);
  }
  fs::remove_all(dir, ec);
}

