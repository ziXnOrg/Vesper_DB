#include <catch2/catch_all.hpp>
#include <vesper/wal/snapshot.hpp>
#include <filesystem>
#include <fstream>

using namespace vesper;

TEST_CASE("atomic snapshot save leaves only final file", "[wal][snapshot]") {
  namespace fs = std::filesystem;
  auto dir = fs::temp_directory_path() / "vesper_wal_snap_atomic";
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);

  wal::Snapshot s{.last_lsn=42};
  REQUIRE(wal::save_snapshot(dir, s).has_value());

  // Ensure wal.snapshot exists and tmp does not
  REQUIRE(fs::exists(dir / "wal.snapshot"));
  REQUIRE_FALSE(fs::exists(dir / "wal.snapshot.tmp"));

  // Verify content
  auto sx = wal::load_snapshot(dir);
  REQUIRE(sx.has_value());
  REQUIRE(sx->last_lsn == 42);

  fs::remove_all(dir, ec);
}

