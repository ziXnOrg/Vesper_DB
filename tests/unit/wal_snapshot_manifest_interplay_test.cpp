#include <catch2/catch_all.hpp>
#include <vesper/wal/io.hpp>
#include <vesper/wal/replay.hpp>
#include <vesper/wal/snapshot.hpp>
#include <filesystem>
#include <vector>

#include <tests/support/replayer_payload.hpp>
#include <tests/support/wal_replay_helpers.hpp>
#include <tests/support/manifest_test_helpers.hpp>

using namespace vesper;
using namespace test_support;
using namespace manifest_test_helpers;

namespace {
namespace fs = std::filesystem;

static void write_sequence(const fs::path& dir){
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);
  wal::WalWriterOptions opts{.dir=dir, .prefix="wal-", .max_file_bytes=48, .strict_lsn_monotonic=false};
  auto w = wal::WalWriter::open(opts); REQUIRE(w.has_value());
  auto up201 = make_upsert(201, std::vector<float>{1.0f}, {});
  auto up202 = make_upsert(202, std::vector<float>{2.0f}, {});
  auto del201 = make_delete(201);
  auto up203 = make_upsert(203, std::vector<float>{3.0f}, {});
  auto up204 = make_upsert(204, std::vector<float>{4.0f}, {});
  auto up205 = make_upsert(205, std::vector<float>{5.0f}, {});
  REQUIRE(w->append(1, /*type=*/1, up201).has_value());
  REQUIRE(w->append(2, /*type=*/1, up202).has_value());
  REQUIRE(w->append(3, /*type=*/1, del201).has_value());
  REQUIRE(w->append(4, /*type=*/1, up203).has_value());
  REQUIRE(w->append(5, /*type=*/1, up204).has_value());
  REQUIRE(w->append(6, /*type=*/1, up205).has_value());
  REQUIRE(w->flush(false).has_value());
}

static void make_manifest_stale_drop_last(const fs::path& dir){
  auto files = list_wal_files_sorted(dir); if (files.empty()) return;
  auto last_file = files.back().second.filename().string();
  auto manifest_path = dir / "wal.manifest";
  auto lines = read_manifest_entries(manifest_path);
  auto kept = entries_without_filename(lines, last_file);
  write_manifest_entries(manifest_path, kept);
}

} // namespace

TEST_CASE("snapshot cutoff respected across rotations with/without manifest completeness", "[wal][snapshot][manifest][replay]"){
  namespace fs = std::filesystem;
  auto dir = fs::temp_directory_path() / "vesper_wal_snapshot_manifest";
  write_sequence(dir);

  // Subcase 1: cutoff1=3, manifest up-to-date
  {
    const std::uint64_t cutoff1 = 3;
    REQUIRE(wal::save_snapshot(dir, wal::Snapshot{.last_lsn = cutoff1}).has_value());
    std::size_t frames=0; auto st = wal::recover_scan_dir(dir, [&](const wal::WalFrame&){ frames++; });
    REQUIRE(st.has_value()); REQUIRE(frames == 3); REQUIRE(st->last_lsn == 6); REQUIRE(st->lsn_monotonic == true);
    ToyIndex idx = build_toy_index_baseline_then_replay(dir, cutoff1);
    REQUIRE(idx.count(201) == 0);
    REQUIRE(idx.count(202) == 1);
    REQUIRE(idx.count(203) == 1);
    REQUIRE(idx.count(204) == 1);
    REQUIRE(idx.count(205) == 1);
  }

  // Subcase 2: cutoff2=5, manifest up-to-date
  {
    const std::uint64_t cutoff2 = 5;
    REQUIRE(wal::save_snapshot(dir, wal::Snapshot{.last_lsn = cutoff2}).has_value());
    std::size_t frames=0; auto st = wal::recover_scan_dir(dir, [&](const wal::WalFrame&){ frames++; });
    REQUIRE(st.has_value()); REQUIRE(frames == 1); REQUIRE(st->last_lsn == 6);
    ToyIndex idx = build_toy_index_baseline_then_replay(dir, cutoff2);
    REQUIRE(idx.count(205) == 1);
    REQUIRE(idx.count(202) == 1);
    REQUIRE(idx.count(203) == 1);
    REQUIRE(idx.count(204) == 1);
    REQUIRE(idx.count(201) == 0);
  }

  // Subcase 3: cutoff2=5 with stale manifest (drop last entry)
  {
    write_sequence(dir); // reset content
    const std::uint64_t cutoff2 = 5;
    make_manifest_stale_drop_last(dir);
    REQUIRE(wal::save_snapshot(dir, wal::Snapshot{.last_lsn = cutoff2}).has_value());
    std::size_t frames=0; auto st = wal::recover_scan_dir(dir, [&](const wal::WalFrame&){ frames++; });
    REQUIRE(st.has_value()); REQUIRE(frames == 1); REQUIRE(st->last_lsn == 6);
    ToyIndex idx = build_toy_index_baseline_then_replay(dir, cutoff2);
    REQUIRE(idx.count(205) == 1);
    REQUIRE(idx.count(202) == 1);
    REQUIRE(idx.count(203) == 1);
    REQUIRE(idx.count(204) == 1);
    REQUIRE(idx.count(201) == 0);
  }

  std::error_code ec; fs::remove_all(dir, ec);
}

