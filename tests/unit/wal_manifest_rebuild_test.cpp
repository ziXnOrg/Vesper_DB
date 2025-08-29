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

