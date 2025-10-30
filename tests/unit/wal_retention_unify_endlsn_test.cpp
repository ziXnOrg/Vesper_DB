#include <catch2/catch_all.hpp>
#include <vesper/wal/retention.hpp>
#include <vesper/wal/manifest.hpp>

#include <filesystem>
#include <fstream>

using namespace vesper;

namespace fs = std::filesystem;

TEST_CASE("purge_keep_last_n should select by end_lsn when seq order disagrees", "[wal][retention][unify]"){
  auto dir = fs::temp_directory_path() / "vesper_wal_unify_endlsn";
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);

  // Create two dummy wal files
  const std::string f1 = "wal-00000001.log";
  const std::string f2 = "wal-00000002.log";
  { std::ofstream(dir / f1).put('\n'); }
  { std::ofstream(dir / f2).put('\n'); }

  // Craft manifest where seq=2 has smaller end_lsn than seq=1
  wal::Manifest m;
  m.entries.push_back(wal::ManifestEntry{.file=f1, .seq=1, .start_lsn=100, .first_lsn=100, .end_lsn=200, .frames=2, .bytes=1});
  m.entries.push_back(wal::ManifestEntry{.file=f2, .seq=2, .start_lsn=50,  .first_lsn=50,  .end_lsn=50,  .frames=1, .bytes=1});
  REQUIRE(wal::save_manifest(dir, m).has_value());

  // Keep the last 1 according to unified semantics (by end_lsn, then filename)
  auto rx = wal::purge_keep_last_n(dir, 1);
  REQUIRE(rx.has_value());

  bool f1_exists = fs::exists(dir / f1);
  bool f2_exists = fs::exists(dir / f2);

  // Expect f1 (end_lsn=200) to be kept, and f2 to be removed
  REQUIRE(f1_exists == true);
  REQUIRE(f2_exists == false);

  fs::remove_all(dir, ec);
}

