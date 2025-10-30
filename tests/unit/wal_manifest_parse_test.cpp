#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <string>
#include <expected>

#include "vesper/wal/manifest.hpp"

namespace fs = std::filesystem;
using vesper::wal::load_manifest;
using vesper::wal::validate_manifest;

static fs::path make_dir(const std::string& name) {
  auto dir = fs::temp_directory_path() / ("vesper_wal_manifest_parse_" + name);
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);
  return dir;
}

static void write_manifest(const fs::path& dir, const std::string& body) {
  auto p = dir / "wal.manifest";
  std::ofstream out(p, std::ios::binary);
  REQUIRE(out.good());
  out << body;
  out.close();
}

static void touch_file(const fs::path& dir, const std::string& name) {
  auto p = dir / name;
  std::ofstream out(p, std::ios::binary);
  REQUIRE(out.good());
  out.close();
}

// Test IDs for audit traceability:
//   TID-WAL-MAN-PARSE-001: malformed numeric field returns expected error (no throw)
//   TID-WAL-MAN-PARSE-002: overflow numeric value is detected and rejected (no throw)
//   TID-WAL-MAN-PARSE-003: missing required fields are rejected (no throw)
//   TID-WAL-MAN-PARSE-004: invalid control chars in filename are rejected (no throw)
//   TID-WAL-MAN-PATH-001: '../wal-00000001.log' traversal is rejected (no throw)
//   TID-WAL-MAN-PATH-002: '..\\wal-00000001.log' traversal is rejected (no throw)
//   TID-WAL-MAN-PATH-003: POSIX absolute '/etc/passwd' is rejected (no throw)
//   TID-WAL-MAN-PATH-004: Windows absolute 'C:\\Windows\\System32\\file.log' is rejected (no throw)
//   TID-WAL-MAN-PATH-005: UNC '\\\\server\\share\\file.log' is rejected (no throw)
//   TID-WAL-MAN-PATH-006: mixed separators 'wal-00000001.log/other' rejected (no throw)
//   TID-WAL-MAN-PATH-007: valid 'wal-00000042.log' accepted

TEST_CASE("wal_manifest_parse: malformed numeric field returns error (no throw)", "[wal_manifest_parse]") {
  auto dir = make_dir("bad_numeric");
  std::string content;
  content += "vesper-wal-manifest v1\n";
  content += "file=wal-00000001.log seq=abc start_lsn=1 first_lsn=1 end_lsn=2 frames=3 bytes=100\n";
  write_manifest(dir, content);

  REQUIRE_NOTHROW(load_manifest(dir));
  auto r = load_manifest(dir);
  REQUIRE_FALSE(r.has_value());
}

TEST_CASE("wal_manifest_parse: overflow numeric value is rejected (no throw)", "[wal_manifest_parse]") {
  auto dir = make_dir("overflow_numeric");
  std::string big = "18446744073709551616"; // 2^64 (one above max uint64)
  std::string content;
  content += "vesper-wal-manifest v1\n";
  content += std::string("file=wal-00000001.log seq=") + big + " start_lsn=1 first_lsn=1 end_lsn=2 frames=3 bytes=100\n";
  write_manifest(dir, content);

  REQUIRE_NOTHROW(load_manifest(dir));
  auto r = load_manifest(dir);
  REQUIRE_FALSE(r.has_value());
}

TEST_CASE("wal_manifest_parse: missing required fields are rejected (no throw)", "[wal_manifest_parse]") {
  auto dir = make_dir("missing_fields");
  std::string content;
  content += "vesper-wal-manifest v1\n";
  // Missing frames and bytes
  content += "file=wal-00000001.log seq=1 start_lsn=1 first_lsn=1 end_lsn=2\n";
  write_manifest(dir, content);

  REQUIRE_NOTHROW(load_manifest(dir));
  auto r = load_manifest(dir);
  REQUIRE_FALSE(r.has_value());
}

TEST_CASE("wal_manifest_parse: invalid control chars in filename are rejected (no throw)", "[wal_manifest_parse]") {
  auto dir = make_dir("bad_filename_ctrl");
  std::string content;
  content += "vesper-wal-manifest v1\n";
  // Build filename with a control char 0x01 inside value
  content += std::string("file=wal-");
  content.push_back('\x01');
  content += std::string("bad.log seq=1 start_lsn=1 first_lsn=1 end_lsn=2 frames=3 bytes=100\n");
  write_manifest(dir, content);

  REQUIRE_NOTHROW(load_manifest(dir));
  auto r = load_manifest(dir);
  REQUIRE_FALSE(r.has_value());
}


TEST_CASE("wal_manifest_parse: path traversal '../' is rejected (no throw)", "[wal_manifest_parse]") {
  auto dir = make_dir("traversal_up_unix");
  std::string content;
  content += "vesper-wal-manifest v1\n";
  content += "file=../wal-00000001.log seq=1 start_lsn=1 first_lsn=1 end_lsn=2 frames=3 bytes=100\n";
  write_manifest(dir, content);
  REQUIRE_NOTHROW(load_manifest(dir));
  auto r = load_manifest(dir);
  REQUIRE_FALSE(r.has_value());
}

TEST_CASE("wal_manifest_parse: path traversal '..\\' is rejected (no throw)", "[wal_manifest_parse]") {
  auto dir = make_dir("traversal_up_win");
  std::string content;
  content += "vesper-wal-manifest v1\n";
  content += "file=..\\\\wal-00000001.log seq=1 start_lsn=1 first_lsn=1 end_lsn=2 frames=3 bytes=100\n";
  write_manifest(dir, content);
  REQUIRE_NOTHROW(load_manifest(dir));
  auto r = load_manifest(dir);
  REQUIRE_FALSE(r.has_value());
}

TEST_CASE("wal_manifest_parse: absolute paths are rejected (no throw)", "[wal_manifest_parse]") {
  auto dir = make_dir("absolute_paths");
  // POSIX-style absolute
  std::string content1;
  content1 += "vesper-wal-manifest v1\n";
  content1 += "file=/etc/passwd seq=1 start_lsn=1 first_lsn=1 end_lsn=2 frames=3 bytes=100\n";
  write_manifest(dir, content1);
  REQUIRE_NOTHROW(load_manifest(dir));
  auto r1 = load_manifest(dir);
  REQUIRE_FALSE(r1.has_value());
  // Windows-style absolute (drive)
  std::string content2;
  content2 += "vesper-wal-manifest v1\n";
  content2 += "file=C:\\Windows\\System32\\file.log seq=1 start_lsn=1 first_lsn=1 end_lsn=2 frames=3 bytes=100\n";
  write_manifest(dir, content2);
  REQUIRE_NOTHROW(load_manifest(dir));
  auto r2 = load_manifest(dir);
  REQUIRE_FALSE(r2.has_value());
}

TEST_CASE("wal_manifest_parse: UNC paths are rejected (no throw)", "[wal_manifest_parse]") {
  auto dir = make_dir("unc_paths");
  std::string content;
  content += "vesper-wal-manifest v1\n";
  content += "file=\\\\server\\share\\file.log seq=1 start_lsn=1 first_lsn=1 end_lsn=2 frames=3 bytes=100\n";
  write_manifest(dir, content);
  REQUIRE_NOTHROW(load_manifest(dir));
  auto r = load_manifest(dir);
  REQUIRE_FALSE(r.has_value());
}

TEST_CASE("wal_manifest_parse: mixed separators are rejected (no throw)", "[wal_manifest_parse]") {
  auto dir = make_dir("mixed_separators");
  std::string content;
  content += "vesper-wal-manifest v1\n";
  content += "file=wal-00000001.log/other seq=1 start_lsn=1 first_lsn=1 end_lsn=2 frames=3 bytes=100\n";
  write_manifest(dir, content);
  REQUIRE_NOTHROW(load_manifest(dir));
  auto r = load_manifest(dir);
  REQUIRE_FALSE(r.has_value());
}

TEST_CASE("wal_manifest_parse: valid filename pattern is accepted", "[wal_manifest_parse]") {
  auto dir = make_dir("valid_filename");
  std::string content;
  content += "vesper-wal-manifest v1\n";
  content += "file=wal-00000042.log seq=42 start_lsn=100 first_lsn=100 end_lsn=200 frames=3 bytes=100\n";
  write_manifest(dir, content);
  auto r = load_manifest(dir);
  REQUIRE(r.has_value());
  REQUIRE(r->entries.size() == 1);
  REQUIRE(r->entries[0].file == std::string("wal-00000042.log"));
}

TEST_CASE("wal_manifest_parse: invalid filename patterns are rejected (no throw)", "[wal_manifest_parse]") {
  auto dir = make_dir("invalid_patterns");
  // wal-123.log (too short)
  std::string c1 = std::string("vesper-wal-manifest v1\n") +
                   "file=wal-123.log seq=1 start_lsn=1 first_lsn=1 end_lsn=2 frames=3 bytes=100\n";
  write_manifest(dir, c1);
  REQUIRE_FALSE(load_manifest(dir).has_value());
  // other-00000001.log (wrong prefix)
  std::string c2 = std::string("vesper-wal-manifest v1\n") +
                   "file=other-00000001.log seq=1 start_lsn=1 first_lsn=1 end_lsn=2 frames=3 bytes=100\n";
  write_manifest(dir, c2);
  REQUIRE_FALSE(load_manifest(dir).has_value());
  // wal-00000001.txt (wrong extension)
  std::string c3 = std::string("vesper-wal-manifest v1\n") +
                   "file=wal-00000001.txt seq=1 start_lsn=1 first_lsn=1 end_lsn=2 frames=3 bytes=100\n";
  write_manifest(dir, c3);
  REQUIRE_FALSE(load_manifest(dir).has_value());
}

TEST_CASE("wal_manifest_validate: duplicate seq across different files is detected (no throw)", "[manifest]") {
  auto dir = make_dir("dup_seq_diff_files");
  std::string content;
  content += "vesper-wal-manifest v1\n";
  content += "file=wal-00000001.log seq=1 start_lsn=1 first_lsn=1 end_lsn=10 frames=1 bytes=1\n";
  content += "file=wal-00000002.log seq=1 start_lsn=11 first_lsn=11 end_lsn=20 frames=1 bytes=1\n";
  write_manifest(dir, content);
  touch_file(dir, "wal-00000001.log");
  touch_file(dir, "wal-00000002.log");
  auto r = validate_manifest(dir);
  REQUIRE(r.has_value());
  REQUIRE_FALSE(r->ok);
  bool found_dup_seq = false;
  for (auto& is : r->issues) {
    if (is.code == vesper::wal::ManifestIssueCode::DuplicateSeq && is.seq == 1) {
      found_dup_seq = true;
      REQUIRE(is.file == std::string("wal-00000002.log"));
      REQUIRE(is.message.find("wal-00000001.log") != std::string::npos);
    }
  }
  REQUIRE(found_dup_seq);
}

TEST_CASE("wal_manifest_validate: duplicate seq within same file reports DuplicateFile (no throw)", "[manifest]") {
  auto dir = make_dir("dup_seq_same_file");
  std::string content;
  content += "vesper-wal-manifest v1\n";
  content += "file=wal-00000003.log seq=2 start_lsn=1 first_lsn=1 end_lsn=10 frames=1 bytes=1\n";
  content += "file=wal-00000003.log seq=2 start_lsn=11 first_lsn=11 end_lsn=20 frames=1 bytes=1\n";
  write_manifest(dir, content);
  touch_file(dir, "wal-00000003.log");
  auto r = validate_manifest(dir);
  REQUIRE(r.has_value());
  REQUIRE_FALSE(r->ok);
  bool found_dup_file = false;
  for (auto& is : r->issues) {
    if (is.code == vesper::wal::ManifestIssueCode::DuplicateFile && is.file == std::string("wal-00000003.log")) {
      found_dup_file = true;
    }
  }
  REQUIRE(found_dup_file);
}

TEST_CASE("wal_manifest_validate: multiple duplicates are all detected (no throw)", "[manifest]") {
  auto dir = make_dir("dup_seq_multiple");
  std::string content;
  content += "vesper-wal-manifest v1\n";
  content += "file=wal-00000005.log seq=5 start_lsn=1 first_lsn=1 end_lsn=10 frames=1 bytes=1\n";
  content += "file=wal-00000006.log seq=5 start_lsn=11 first_lsn=11 end_lsn=20 frames=1 bytes=1\n";
  content += "file=wal-00000007.log seq=5 start_lsn=21 first_lsn=21 end_lsn=30 frames=1 bytes=1\n";
  write_manifest(dir, content);
  touch_file(dir, "wal-00000005.log");
  touch_file(dir, "wal-00000006.log");
  touch_file(dir, "wal-00000007.log");
  auto r = validate_manifest(dir);
  REQUIRE(r.has_value());
  REQUIRE_FALSE(r->ok);
  int dup_count = 0;
  for (auto& is : r->issues) {
    if (is.code == vesper::wal::ManifestIssueCode::DuplicateSeq && is.seq == 5) {
      ++dup_count;
    }
  }
  REQUIRE(dup_count >= 2);
}

TEST_CASE("wal_manifest_validate: unique seq manifest passes", "[manifest]") {
  auto dir = make_dir("unique_seq_ok");
  std::string content;
  content += "vesper-wal-manifest v1\n";
  content += "file=wal-00000001.log seq=1 start_lsn=1 first_lsn=1 end_lsn=10 frames=1 bytes=1\n";
  content += "file=wal-00000002.log seq=2 start_lsn=11 first_lsn=11 end_lsn=20 frames=1 bytes=1\n";
  content += "file=wal-00000003.log seq=3 start_lsn=21 first_lsn=21 end_lsn=30 frames=1 bytes=1\n";
  write_manifest(dir, content);
  touch_file(dir, "wal-00000001.log");
  touch_file(dir, "wal-00000002.log");
  touch_file(dir, "wal-00000003.log");
  auto r = validate_manifest(dir);
  REQUIRE(r.has_value());
  REQUIRE(r->ok);
}



// LSN range validation tests
//   TID-WAL-MAN-LSN-001..008

TEST_CASE("wal_manifest_validate: intra-entry LSN invalid start>first (no throw)", "[manifest]") {
  auto dir = make_dir("lsn_intra_start_gt_first");
  std::string content;
  content += "vesper-wal-manifest v1\n";
  content += "file=wal-00000001.log seq=1 start_lsn=10 first_lsn=5 end_lsn=10 frames=1 bytes=1\n";
  write_manifest(dir, content);
  touch_file(dir, "wal-00000001.log");
  auto r = validate_manifest(dir);
  REQUIRE(r.has_value());
  REQUIRE_FALSE(r->ok);
  bool found=false; for (auto& is : r->issues) {
    if (is.code == vesper::wal::ManifestIssueCode::LsnInvalid) { found=true; REQUIRE(is.file==std::string("wal-00000001.log")); }
  }
  REQUIRE(found);
}

TEST_CASE("wal_manifest_validate: intra-entry LSN invalid first>end (no throw)", "[manifest]") {
  auto dir = make_dir("lsn_intra_first_gt_end");
  std::string content;
  content += "vesper-wal-manifest v1\n";
  content += "file=wal-00000001.log seq=1 start_lsn=1 first_lsn=100 end_lsn=10 frames=1 bytes=1\n";
  write_manifest(dir, content);
  touch_file(dir, "wal-00000001.log");
  auto r = validate_manifest(dir);
  REQUIRE(r.has_value());
  REQUIRE_FALSE(r->ok);
  bool found=false; for (auto& is : r->issues) {
    if (is.code == vesper::wal::ManifestIssueCode::LsnInvalid) { found=true; }
  }
  REQUIRE(found);
}

TEST_CASE("wal_manifest_validate: intra-entry LSN invalid start>end (no throw)", "[manifest]") {
  auto dir = make_dir("lsn_intra_start_gt_end");
  std::string content;
  content += "vesper-wal-manifest v1\n";
  content += "file=wal-00000001.log seq=1 start_lsn=20 first_lsn=20 end_lsn=10 frames=1 bytes=1\n";
  write_manifest(dir, content);
  touch_file(dir, "wal-00000001.log");
  auto r = validate_manifest(dir);
  REQUIRE(r.has_value());
  REQUIRE_FALSE(r->ok);
  bool found=false; for (auto& is : r->issues) {
    if (is.code == vesper::wal::ManifestIssueCode::LsnInvalid) { found=true; }
  }
  REQUIRE(found);
}

TEST_CASE("wal_manifest_validate: intra-entry LSN boundary start==first==end passes", "[manifest]") {
  auto dir = make_dir("lsn_intra_boundary_ok");
  std::string content;
  content += "vesper-wal-manifest v1\n";
  content += "file=wal-00000001.log seq=1 start_lsn=100 first_lsn=100 end_lsn=100 frames=1 bytes=1\n";
  write_manifest(dir, content);
  touch_file(dir, "wal-00000001.log");
  auto r = validate_manifest(dir);
  REQUIRE(r.has_value());
  REQUIRE(r->ok);
}

TEST_CASE("wal_manifest_validate: cross-entry overlap detected (no throw)", "[manifest]") {
  auto dir = make_dir("lsn_overlap");
  std::string content;
  content += "vesper-wal-manifest v1\n";
  content += "file=wal-00000001.log seq=1 start_lsn=1 first_lsn=1 end_lsn=20 frames=1 bytes=1\n";
  content += "file=wal-00000002.log seq=2 start_lsn=20 first_lsn=20 end_lsn=30 frames=1 bytes=1\n";
  write_manifest(dir, content);
  touch_file(dir, "wal-00000001.log");
  touch_file(dir, "wal-00000002.log");
  auto r = validate_manifest(dir);
  REQUIRE(r.has_value());
  REQUIRE_FALSE(r->ok);
  bool found=false; for (auto& is : r->issues) {
    if (is.code == vesper::wal::ManifestIssueCode::LsnOverlap && is.seq==2) { found=true; }
  }
  REQUIRE(found);
}

TEST_CASE("wal_manifest_validate: cross-entry end_lsn decreasing detected (no throw)", "[manifest]") {
  auto dir = make_dir("lsn_end_decreasing");
  std::string content;
  content += "vesper-wal-manifest v1\n";
  content += "file=wal-00000001.log seq=1 start_lsn=1 first_lsn=1 end_lsn=50 frames=1 bytes=1\n";
  content += "file=wal-00000002.log seq=2 start_lsn=40 first_lsn=40 end_lsn=45 frames=1 bytes=1\n"; // also overlaps
  write_manifest(dir, content);
  touch_file(dir, "wal-00000001.log");
  touch_file(dir, "wal-00000002.log");
  auto r = validate_manifest(dir);
  REQUIRE(r.has_value());
  REQUIRE_FALSE(r->ok);
  bool found=false; for (auto& is : r->issues) {
    if (is.code == vesper::wal::ManifestIssueCode::LsnOrder && is.seq==2) { found=true; }
  }
  REQUIRE(found);
}

TEST_CASE("wal_manifest_validate: cross-entry gap flagged as warning (no throw)", "[manifest]") {
  auto dir = make_dir("lsn_gap_warning");
  std::string content;
  content += "vesper-wal-manifest v1\n";
  content += "file=wal-00000001.log seq=1 start_lsn=1 first_lsn=1 end_lsn=20 frames=1 bytes=1\n";
  content += "file=wal-00000002.log seq=2 start_lsn=25 first_lsn=25 end_lsn=30 frames=1 bytes=1\n";
  write_manifest(dir, content);
  touch_file(dir, "wal-00000001.log");
  touch_file(dir, "wal-00000002.log");
  auto r = validate_manifest(dir);
  REQUIRE(r.has_value());
  REQUIRE(r->ok); // warnings do not flip ok
  bool found=false; for (auto& is : r->issues) {
    if (is.code == vesper::wal::ManifestIssueCode::LsnGap && is.seq==2) { found=true; REQUIRE(is.severity==vesper::wal::Severity::Warning); }
  }
  REQUIRE(found);
}

TEST_CASE("wal_manifest_validate: valid cross-entry ordering passes", "[manifest]") {
  auto dir = make_dir("lsn_valid_ok");
  std::string content;
  content += "vesper-wal-manifest v1\n";
  content += "file=wal-00000001.log seq=1 start_lsn=10 first_lsn=10 end_lsn=20 frames=1 bytes=1\n";
  content += "file=wal-00000002.log seq=2 start_lsn=21 first_lsn=21 end_lsn=30 frames=1 bytes=1\n";
  write_manifest(dir, content);
  touch_file(dir, "wal-00000001.log");
  touch_file(dir, "wal-00000002.log");
  auto r = validate_manifest(dir);
  REQUIRE(r.has_value());
  REQUIRE(r->ok);
}
