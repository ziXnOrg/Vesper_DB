#include <catch2/catch_all.hpp>
#include <vesper/wal/io.hpp>
#include <vesper/wal/replay.hpp>
#include <filesystem>
#include <vector>

#include <tests/support/replayer_payload.hpp>
#include <tests/support/wal_replay_helpers.hpp>

using namespace vesper;
using namespace test_support;

namespace {
namespace fs = std::filesystem;

static std::vector<std::pair<std::uint64_t, fs::path>> wal_files_sorted(const fs::path& dir){
  std::vector<std::pair<std::uint64_t, fs::path>> v;
  for (auto& de : fs::directory_iterator(dir)){
    if (!de.is_regular_file()) continue; auto name = de.path().filename().string();
    if (name.rfind("wal-", 0)==0 && name.size() >= 4+8) { try { auto seq = static_cast<std::uint64_t>(std::stoull(name.substr(4,8))); v.emplace_back(seq, de.path()); } catch(...){} }
  }
  std::sort(v.begin(), v.end(), [](auto&a, auto&b){ return a.first < b.first; });
  return v;
}

static void make_sequence_dir(const fs::path& dir){
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);
  wal::WalWriterOptions opts{.dir=dir, .prefix="wal-", .max_file_bytes=24, .strict_lsn_monotonic=false};
  auto w = wal::WalWriter::open(opts); REQUIRE(w.has_value());
  auto up101 = make_upsert(101, std::vector<float>{1.0f}, {});
  auto up102 = make_upsert(102, std::vector<float>{2.0f}, {});
  auto del101 = make_delete(101);
  auto up103 = make_upsert(103, std::vector<float>{3.0f}, {});
  auto up104 = make_upsert(104, std::vector<float>{4.0f}, {});
  REQUIRE(w->append(1, /*type=*/1, up101).has_value());
  REQUIRE(w->append(2, /*type=*/1, up102).has_value());
  REQUIRE(w->append(3, /*type=*/2, del101).has_value());
  REQUIRE(w->append(4, /*type=*/1, up103).has_value());
  REQUIRE(w->append(5, /*type=*/1, up104).has_value());
  REQUIRE(w->flush(false).has_value());
}

} // namespace

TEST_CASE("purge_wal handles file-boundary cutoffs deterministically", "[wal][manifest][purge][boundaries]"){
  namespace fs = std::filesystem;
  fs::path dir = fs::temp_directory_path() / "vesper_wal_purge_boundaries";
  make_sequence_dir(dir);
  auto files = wal_files_sorted(dir); REQUIRE(files.size() >= 3);

  SECTION("cutoff equals end_lsn of the first file: removes first file only"){
    // We wrote frames 1..5; with tight rotation there should be ~1-2 frames per file. Choose cutoff=2 so first file fully <= cutoff.
    std::uint64_t cutoff = 2;
    REQUIRE(wal::purge_wal(dir, cutoff).has_value());
    auto after = wal_files_sorted(dir);
    REQUIRE(after.size() < files.size());
    // Remaining stats and replay
    auto st = wal::recover_scan_dir(dir, [&](const wal::WalFrame&){ /*noop*/ }); REQUIRE(st.has_value());
    REQUIRE(st->last_lsn == 5);
    REQUIRE(st->lsn_monotonic == true);
    ToyIndex idx = build_toy_index_baseline_then_replay(dir, cutoff);
    REQUIRE(idx.count(101) == 0);
    REQUIRE(idx.count(102) == 1);
    REQUIRE(idx.count(103) == 1);
    REQUIRE(idx.count(104) == 1);
  }

  SECTION("cutoff less than the first file's first_lsn: no files removed"){
    std::uint64_t cutoff = 0;
    auto before = wal_files_sorted(dir);
    REQUIRE(wal::purge_wal(dir, cutoff).has_value());
    auto after = wal_files_sorted(dir);
    REQUIRE(after.size() == before.size());
    auto st = wal::recover_scan_dir(dir, [&](const wal::WalFrame&){ /*noop*/ }); REQUIRE(st.has_value());
    REQUIRE(st->frames >= 5);
    ToyIndex idx = build_toy_index_baseline_then_replay(dir, cutoff);
    REQUIRE(idx.count(102) == 1);
    REQUIRE(idx.count(103) == 1);
    REQUIRE(idx.count(104) == 1);
  }

  SECTION("cutoff equals the global last_lsn: all files removed"){
    std::uint64_t cutoff = 5;
    REQUIRE(wal::purge_wal(dir, cutoff).has_value());
    auto after = wal_files_sorted(dir);
    REQUIRE(after.size() == 0);
    auto st = wal::recover_scan_dir(dir, [&](const wal::WalFrame&){ /*noop*/ });
    REQUIRE(st.has_value());
    REQUIRE(st->frames == 0);
    // Replay should be a no-op
    ToyIndex idx = build_toy_index_baseline_then_replay(dir, cutoff);
    REQUIRE(idx.size() == 0);
  }

  std::error_code ec; fs::remove_all(dir, ec);
}

