#include <catch2/catch_all.hpp>
#include <vesper/wal/io.hpp>
#include <vesper/wal/manifest.hpp>
#include <filesystem>
#include <vector>

#include <tests/support/replayer_payload.hpp>
#include <tests/support/wal_replay_helpers.hpp>
#include <tests/support/manifest_test_helpers.hpp>

using namespace vesper;
using namespace test_support;
using namespace manifest_test_helpers;

TEST_CASE("rebuild_manifest restores correct order and content", "[wal][manifest][rebuild][roundtrip]"){
  namespace fs = std::filesystem;
  auto dir = fs::temp_directory_path() / "vesper_wal_manifest_rebuild";
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);

  // Write deterministic frames across >=2 files
  wal::WalWriterOptions opts{.dir=dir, .prefix="wal-", .max_file_bytes=48, .strict_lsn_monotonic=false};
  auto w = wal::WalWriter::open(opts); REQUIRE(w.has_value());
  auto up1 = make_upsert(501, std::vector<float>{1.0f}, {});
  auto up2 = make_upsert(502, std::vector<float>{2.0f}, {});
  auto up3 = make_upsert(503, std::vector<float>{3.0f}, {});
  REQUIRE(w->append(1, 1, up1).has_value());
  REQUIRE(w->append(2, 1, up2).has_value());
  REQUIRE(w->append(3, 1, up3).has_value());
  REQUIRE(w->flush(false).has_value());

  auto files = list_wal_files_sorted(dir); REQUIRE(files.size() >= 2);
  auto manifest_path = dir / "wal.manifest";
  REQUIRE(fs::exists(manifest_path));

  // Mangle the manifest: reverse order and drop middle if any, then duplicate first
  {
    auto lines = read_manifest_entries(manifest_path);
    auto rev = entries_reversed(lines);
    if (files.size() >= 3) {
      auto mid = files[files.size()/2].second.filename().string();
      rev = entries_without_filename(rev, mid);
    }
    auto first_name = files.front().second.filename().string();
    auto dup = entries_with_duplicated_filename(rev, first_name);
    write_manifest_entries(manifest_path, dup);
  }

  // Build expected manifest from files via rebuild and assert it sorts and contains all
  auto mx = wal::rebuild_manifest(dir);
  REQUIRE(mx.has_value());
  const wal::Manifest& m = *mx;
  REQUIRE(m.entries.size() == files.size());
  for (size_t i=0;i<m.entries.size();++i){
    REQUIRE(m.entries[i].file == files[i].second.filename().string());
    REQUIRE(m.entries[i].seq == files[i].first);
    // Basic sanity: frames>0; bytes>0; end_lsn>=first_lsn>=start_lsn
    REQUIRE(m.entries[i].frames > 0);
    REQUIRE(m.entries[i].bytes > 0);
    REQUIRE(m.entries[i].end_lsn >= m.entries[i].first_lsn);
    REQUIRE(m.entries[i].first_lsn >= m.entries[i].start_lsn);
  }

  // Write the rebuilt manifest
  REQUIRE(wal::save_manifest(dir, m).has_value());

  // Stats/replay using the rebuilt manifest should be consistent
  std::size_t frames=0; auto st = wal::recover_scan_dir(dir, [&](const wal::WalFrame&){ frames++; });
  REQUIRE(st.has_value());
  REQUIRE(frames == 3);
  REQUIRE(st->last_lsn == 3);
  ToyIndex idx = build_toy_index_baseline_then_replay(dir, /*cutoff=*/0);
  REQUIRE(idx.count(501) == 1);
  REQUIRE(idx.count(502) == 1);
  REQUIRE(idx.count(503) == 1);

  fs::remove_all(dir, ec);
}



TEST_CASE("rebuild_manifest: valid LSN ranges pass validate_manifest", "[wal][manifest][rebuild]"){
  namespace fs = std::filesystem;
  auto dir = fs::temp_directory_path() / "vesper_wal_manifest_rebuild_valid";
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);

  wal::WalWriterOptions opts{.dir=dir, .prefix="wal-", .max_file_bytes=48, .strict_lsn_monotonic=true};
  auto w = wal::WalWriter::open(opts); REQUIRE(w.has_value());
  auto up = make_upsert(600, std::vector<float>{4.0f}, {});
  REQUIRE(w->append(10, 1, up).has_value());
  REQUIRE(w->append(11, 1, up).has_value());
  REQUIRE(w->append(12, 1, up).has_value());
  REQUIRE(w->flush(false).has_value());

  auto mx = wal::rebuild_manifest(dir);
  REQUIRE(mx.has_value());
  REQUIRE(wal::save_manifest(dir, *mx).has_value());
  auto v = wal::validate_manifest(dir);
  REQUIRE(v.has_value());
  REQUIRE(v->ok);
  fs::remove_all(dir, ec);
}

TEST_CASE("rebuild_manifest: LSN overlap across files is rejected (no throw)", "[wal][manifest][rebuild]"){
  namespace fs = std::filesystem;
  auto dir = fs::temp_directory_path() / "vesper_wal_manifest_rebuild_overlap";
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);

  wal::WalWriterOptions opts{.dir=dir, .prefix="wal-", .max_file_bytes=32, .strict_lsn_monotonic=false};
  auto w = wal::WalWriter::open(opts); REQUIRE(w.has_value());
  auto up = make_upsert(601, std::vector<float>{5.0f}, {});
  REQUIRE(w->append(10, 1, up).has_value());
  REQUIRE(w->append(20, 1, up).has_value());
  // Force another file and overlap start_lsn <= prev end_lsn
  REQUIRE(w->append(15, 1, up).has_value());
  REQUIRE(w->flush(false).has_value());

  auto mx = wal::rebuild_manifest(dir);
  REQUIRE_FALSE(mx.has_value());
  fs::remove_all(dir, ec);
}

TEST_CASE("rebuild_manifest: intra-entry start_lsn > end_lsn is rejected (no throw)", "[wal][manifest][rebuild]"){
  namespace fs = std::filesystem;
  auto dir = fs::temp_directory_path() / "vesper_wal_manifest_rebuild_intra";
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);

  // Keep a single file; write higher LSN then lower LSN to make start_lsn > end_lsn
  wal::WalWriterOptions opts{.dir=dir, .prefix="wal-", .max_file_bytes=4096, .strict_lsn_monotonic=false};
  auto w = wal::WalWriter::open(opts); REQUIRE(w.has_value());
  auto up = make_upsert(602, std::vector<float>{6.0f}, {});
  REQUIRE(w->append(20, 1, up).has_value());
  REQUIRE(w->append(10, 1, up).has_value());
  REQUIRE(w->flush(false).has_value());

  auto mx = wal::rebuild_manifest(dir);
  REQUIRE_FALSE(mx.has_value());
  fs::remove_all(dir, ec);
}

TEST_CASE("rebuild_manifest: LSN gap across files allowed; validate warns", "[wal][manifest][rebuild]"){
  namespace fs = std::filesystem;
  auto dir = fs::temp_directory_path() / "vesper_wal_manifest_rebuild_gap";
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);

  wal::WalWriterOptions opts{.dir=dir, .prefix="wal-", .max_file_bytes=32, .strict_lsn_monotonic=true};
  auto w = wal::WalWriter::open(opts); REQUIRE(w.has_value());
  auto up = make_upsert(603, std::vector<float>{7.0f}, {});
  REQUIRE(w->append(10, 1, up).has_value());
  REQUIRE(w->append(11, 1, up).has_value());
  // Rotate and create a gap
  REQUIRE(w->append(100, 1, up).has_value());
  REQUIRE(w->flush(false).has_value());

  auto mx = wal::rebuild_manifest(dir);
  REQUIRE(mx.has_value());
  REQUIRE(wal::save_manifest(dir, *mx).has_value());
  auto v = wal::validate_manifest(dir);
  REQUIRE(v.has_value());
  REQUIRE(v->ok);
  bool seen_gap=false; for (auto& is : v->issues) { if (is.code == wal::ManifestIssueCode::LsnGap) { seen_gap=true; REQUIRE(is.severity==wal::Severity::Warning); } }
  REQUIRE(seen_gap);
  fs::remove_all(dir, ec);
}


TEST_CASE("rebuild_manifest_lenient: all valid files -> complete manifest, no issues", "[wal][manifest][rebuild][lenient]"){
  namespace fs = std::filesystem;
  auto dir = fs::temp_directory_path() / "vesper_wal_manifest_rebuild_lenient_ok";
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);
  wal::WalWriterOptions opts{.dir=dir, .prefix="wal-", .max_file_bytes=48, .strict_lsn_monotonic=true};
  auto w = wal::WalWriter::open(opts); REQUIRE(w.has_value());
  auto up = make_upsert(700, std::vector<float>{1.0f}, {});
  REQUIRE(w->append(1, 1, up).has_value());
  REQUIRE(w->append(2, 1, up).has_value());
  REQUIRE(w->append(3, 1, up).has_value());
  REQUIRE(w->flush(false).has_value());
  auto files = list_wal_files_sorted(dir);
  auto rr = wal::rebuild_manifest_lenient(dir);
  REQUIRE(rr.has_value());
  REQUIRE(rr->issues.empty());
  REQUIRE(rr->manifest.entries.size() == files.size());
  fs::remove_all(dir, ec);
}

#if defined(_WIN32)
#include <windows.h>
static HANDLE lock_file_exclusive(const std::filesystem::path& p){
  auto w = std::wstring(p.wstring());
  return CreateFileW(w.c_str(), GENERIC_READ|GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
}
#endif

TEST_CASE("rebuild_manifest_lenient: IO error on one file -> partial manifest + 1 issue", "[wal][manifest][rebuild][lenient]"){
  namespace fs = std::filesystem;
  auto dir = fs::temp_directory_path() / "vesper_wal_manifest_rebuild_lenient_io";
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);
  {
    wal::WalWriterOptions opts{.dir=dir, .prefix="wal-", .max_file_bytes=32, .strict_lsn_monotonic=true};
    auto w = wal::WalWriter::open(opts); REQUIRE(w.has_value());
    auto up = make_upsert(701, std::vector<float>{1.0f}, {});
    REQUIRE(w->append(10, 1, up).has_value());
    REQUIRE(w->append(11, 1, up).has_value());
    REQUIRE(w->append(12, 1, up).has_value());
    REQUIRE(w->flush(false).has_value());
  }
  auto files = list_wal_files_sorted(dir); REQUIRE(files.size() >= 2);

#if defined(_WIN32)
  // Lock the second file exclusively to try to induce open failure in recover_scan
  HANDLE h = lock_file_exclusive(files[1].second);
#endif

  auto rr = wal::rebuild_manifest_lenient(dir);
  REQUIRE(rr.has_value());
#if defined(_WIN32)
  if (h != INVALID_HANDLE_VALUE) {
    REQUIRE(rr->issues.size() == 1);
    REQUIRE(rr->manifest.entries.size() == files.size()-1);
    CloseHandle(h);
  } else {
    // Could not acquire exclusive lock; ensure lenient rebuild still succeeds
    REQUIRE(rr->manifest.entries.size() == files.size());
  }
#else
  // On non-Windows we cannot force IO error portably; just ensure function succeeds
  REQUIRE(rr->manifest.entries.size() == files.size());
#endif
  fs::remove_all(dir, ec);
}

TEST_CASE("rebuild_manifest_lenient: intra-entry invalid LSN -> skipped with issue", "[wal][manifest][rebuild][lenient]"){
  namespace fs = std::filesystem;
  auto dir = fs::temp_directory_path() / "vesper_wal_manifest_rebuild_lenient_intra";
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);
  wal::WalWriterOptions opts{.dir=dir, .prefix="wal-", .max_file_bytes=4096, .strict_lsn_monotonic=false};
  auto w = wal::WalWriter::open(opts); REQUIRE(w.has_value());
  auto up = make_upsert(702, std::vector<float>{1.0f}, {});
  // Write descending LSN within a single file to produce start_lsn > end_lsn
  REQUIRE(w->append(20, 1, up).has_value());
  REQUIRE(w->append(10, 1, up).has_value());
  REQUIRE(w->flush(false).has_value());
  auto rr = wal::rebuild_manifest_lenient(dir);
  REQUIRE(rr.has_value());
  REQUIRE(rr->manifest.entries.empty());
  REQUIRE(rr->issues.size() == 1);
  fs::remove_all(dir, ec);
}

TEST_CASE("rebuild_manifest_lenient: multiple corrupt files -> multiple issues", "[wal][manifest][rebuild][lenient]"){
  namespace fs = std::filesystem;
  auto dir = fs::temp_directory_path() / "vesper_wal_manifest_rebuild_lenient_multi";
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);
  wal::WalWriterOptions opts{.dir=dir, .prefix="wal-", .max_file_bytes=32, .strict_lsn_monotonic=false};
  auto w = wal::WalWriter::open(opts); REQUIRE(w.has_value());
  auto up = make_upsert(703, std::vector<float>{1.0f}, {});
  // File 1: valid
  REQUIRE(w->append(1, 1, up).has_value());
  REQUIRE(w->append(2, 1, up).has_value());
  // File 2: overlap with file 1
  REQUIRE(w->append(2, 1, up).has_value());
  // File 3: descending within file (invalid intra)
  REQUIRE(w->append(20, 1, up).has_value());
  REQUIRE(w->append(10, 1, up).has_value());
  REQUIRE(w->flush(false).has_value());

  auto rr = wal::rebuild_manifest_lenient(dir);
  REQUIRE(rr.has_value());
  REQUIRE(rr->issues.size() >= 2);
  // At least the valid first file should be present, later may be skipped
  REQUIRE(rr->manifest.entries.size() >= 1);
  fs::remove_all(dir, ec);
}

#if defined(_WIN32)
TEST_CASE("rebuild_manifest_lenient: all files cause IO errors -> empty manifest, issues for all", "[wal][manifest][rebuild][lenient]"){
  namespace fs = std::filesystem;
  auto dir = fs::temp_directory_path() / "vesper_wal_manifest_rebuild_lenient_all_bad";
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);
  {
    wal::WalWriterOptions opts{.dir=dir, .prefix="wal-", .max_file_bytes=32, .strict_lsn_monotonic=true};
    auto w = wal::WalWriter::open(opts); REQUIRE(w.has_value());
    auto up = make_upsert(704, std::vector<float>{1.0f}, {});
    REQUIRE(w->append(10, 1, up).has_value());
    REQUIRE(w->append(11, 1, up).has_value());
    REQUIRE(w->append(12, 1, up).has_value());
    REQUIRE(w->flush(false).has_value());
  }
  auto files = list_wal_files_sorted(dir); REQUIRE(files.size() >= 2);
  // Try to lock all files to force IO open failures
  std::vector<HANDLE> locks; locks.reserve(files.size());
  bool all_locked = true;
  for (auto& kv : files) { HANDLE h = lock_file_exclusive(kv.second); if (h == INVALID_HANDLE_VALUE) { all_locked = false; } else { locks.push_back(h); } }
  auto rr = wal::rebuild_manifest_lenient(dir);
  REQUIRE(rr.has_value());
  if (all_locked) {
    REQUIRE(rr->manifest.entries.empty());
    REQUIRE(rr->issues.size() == files.size());
  }
  for (auto h : locks) CloseHandle(h);
  fs::remove_all(dir, ec);
}
#endif

TEST_CASE("rebuild_manifest_lenient: partial manifest validates ok (warnings allowed)", "[wal][manifest][rebuild][lenient]"){
  namespace fs = std::filesystem;
  auto dir = fs::temp_directory_path() / "vesper_wal_manifest_rebuild_lenient_validate";
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);
  wal::WalWriterOptions opts{.dir=dir, .prefix="wal-", .max_file_bytes=32, .strict_lsn_monotonic=false};
  auto w = wal::WalWriter::open(opts); REQUIRE(w.has_value());
  auto up = make_upsert(705, std::vector<float>{1.0f}, {});
  // File 1: 1,2 valid
  REQUIRE(w->append(1, 1, up).has_value());
  REQUIRE(w->append(2, 1, up).has_value());
  // File 2: overlap start 2 (invalid)
  REQUIRE(w->append(2, 1, up).has_value());
  // File 3: 5 valid (creates a gap)
  REQUIRE(w->append(5, 1, up).has_value());
  REQUIRE(w->flush(false).has_value());

  auto rr = wal::rebuild_manifest_lenient(dir);
  REQUIRE(rr.has_value());
  // Save only the valid entries manifest and validate
  REQUIRE(wal::save_manifest(dir, rr->manifest).has_value());
  auto v = wal::validate_manifest(dir);
  REQUIRE(v.has_value());
  REQUIRE(v->ok);
  fs::remove_all(dir, ec);
}
