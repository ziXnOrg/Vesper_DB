#include <catch2/catch_all.hpp>
#include <vesper/wal/manifest.hpp>
#include <filesystem>
#include <fstream>
#include <vector>

#include <tests/support/manifest_test_helpers.hpp>

using namespace vesper;
using namespace manifest_test_helpers;

TEST_CASE("validate_manifest detects ordering/duplicates/missing/extra", "[wal][manifest][validate]"){
  namespace fs = std::filesystem;
  auto dir = fs::temp_directory_path() / "vesper_wal_manifest_validate";
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);
  auto mp = dir / "wal.manifest";

  // Write base manifest
  std::vector<std::string> lines{
    "file=wal-00000001.log seq=1 start_lsn=1 first_lsn=1 end_lsn=1 frames=1 bytes=10",
    "file=wal-00000002.log seq=2 start_lsn=2 first_lsn=2 end_lsn=2 frames=1 bytes=10",
    "file=wal-00000003.log seq=3 start_lsn=3 first_lsn=3 end_lsn=3 frames=1 bytes=10"
  }; write_manifest_entries(mp, lines);

  // Reverse order and duplicate first, and drop middle; create extra file on disk
  auto rev = entries_reversed(lines);
  auto dup = entries_with_duplicated_filename(rev, std::string("wal-00000001.log"));
  auto without_mid = entries_without_filename(dup, std::string("wal-00000002.log"));
  write_manifest_entries(mp, without_mid);
  // Create an extra file on disk not in manifest
  std::ofstream(dir / "wal-00000099.log").put('\n');

  auto v = wal::validate_manifest(dir);
  REQUIRE(v.has_value());
  REQUIRE(v->ok == false);
  bool seen_dup=false, seen_order=false, seen_missing=false, seen_extra=false;
  for (auto& is : v->issues) {
    if (is.code == wal::ManifestIssueCode::DuplicateFile) seen_dup=true;
    if (is.code == wal::ManifestIssueCode::OutOfOrderSeq) seen_order=true;
    if (is.code == wal::ManifestIssueCode::MissingFileOnDisk) seen_missing=true;
    if (is.code == wal::ManifestIssueCode::ExtraFileOnDisk) seen_extra=true;
  }
  REQUIRE(seen_dup);
  REQUIRE(seen_order);
  REQUIRE(seen_missing);
  REQUIRE(seen_extra);

  // enforce ordering
  REQUIRE(wal::enforce_manifest_order(dir).has_value());
  // After enforcement, lines should be sorted and duplicates remain (advisory)
  auto after = read_manifest_entries(mp);
  REQUIRE(after.size() == without_mid.size());
  // ascending seq
  std::uint64_t prev=0; for (auto& ln : after){ auto pos = ln.find("seq="); REQUIRE(pos != std::string::npos); auto seq = std::stoull(ln.substr(pos+4)); REQUIRE(seq >= prev); prev=seq; }

  fs::remove_all(dir, ec);
}

