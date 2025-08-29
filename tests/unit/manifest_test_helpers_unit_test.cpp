#include <catch2/catch_all.hpp>
#include <filesystem>
#include <fstream>
#include <vector>

#include <tests/support/manifest_test_helpers.hpp>

using namespace manifest_test_helpers;

TEST_CASE("manifest_test_helpers read/write and transforms are deterministic", "[wal][manifest][helpers]"){
  namespace fs = std::filesystem;
  auto dir = fs::temp_directory_path() / "vesper_manifest_helpers_unit";
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);
  auto mpath = dir / "wal.manifest";

  // Write a small manifest by hand using the helpers
  std::vector<std::string> lines{
    "file=wal-00000001.log seq=1 first_lsn=1 start_lsn=1 end_lsn=1 frames=1 bytes=10",
    "file=wal-00000002.log seq=2 first_lsn=2 start_lsn=2 end_lsn=2 frames=1 bytes=10",
    "file=wal-00000003.log seq=3 first_lsn=3 start_lsn=3 end_lsn=3 frames=1 bytes=10"
  };
  write_manifest_entries(mpath, lines);

  // Read and verify
  auto read_back = read_manifest_entries(mpath);
  REQUIRE(read_back == lines);

  // Reverse
  auto rev = entries_reversed(read_back);
  REQUIRE(rev.front().find("wal-00000003.log") != std::string::npos);

  // Without filename
  auto kept = entries_without_filename(read_back, std::string("wal-00000002.log"));
  REQUIRE(kept.size() == 2);
  for (auto& ln : kept) REQUIRE(ln.find("wal-00000002.log") == std::string::npos);

  // Duplicate filename
  auto dup = entries_with_duplicated_filename(read_back, std::string("wal-00000001.log"));
  REQUIRE(dup.size() == read_back.size() + 1);

  fs::remove_all(dir, ec);
}

