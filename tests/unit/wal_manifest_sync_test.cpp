#include <catch2/catch_all.hpp>
#include <vesper/wal/io.hpp>
#include <vesper/wal/manifest.hpp>
#include <filesystem>

using namespace vesper;

TEST_CASE("manifest is written/updated at rotation boundaries", "[wal][manifest]") {
  namespace fs = std::filesystem;
  auto dir = fs::temp_directory_path() / "vesper_wal_manifest_sync";
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);

  wal::WalWriterOptions opts{.dir=dir, .prefix="wal-", .max_file_bytes=64};
  auto w = wal::WalWriter::open(opts); REQUIRE(w.has_value());

  std::vector<uint8_t> p{0xAA};
  // Write enough frames to complete at least one file and rotate into next
  REQUIRE(w->append(1, 1, p).has_value());
  REQUIRE(w->append(2, 1, p).has_value());
  REQUIRE(w->append(3, 1, p).has_value());
  REQUIRE(w->flush(false).has_value());

  // Re-open writer to force a new sequence file if needed
  auto w2 = wal::WalWriter::open(opts); REQUIRE(w2.has_value());
  REQUIRE(w2->append(4, 1, p).has_value());
  REQUIRE(w2->flush(false).has_value());

  // Load manifest and assert entries
  auto m = wal::load_manifest(dir);
  REQUIRE(m.has_value());
  REQUIRE(m->entries.size() >= 1);
  for (const auto& e : m->entries) {
    REQUIRE(!e.file.empty());
    REQUIRE(e.frames > 0);
    REQUIRE(e.bytes > 0);
    REQUIRE(e.end_lsn >= e.start_lsn);
    REQUIRE(e.file.rfind("wal-", 0) == 0);
  }

  fs::remove_all(dir, ec);
}

