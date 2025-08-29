#include <catch2/catch_all.hpp>
#include <vesper/wal/io.hpp>
#include <vesper/wal/replay.hpp>
#include <vesper/wal/snapshot.hpp>
#include <filesystem>
#include <fstream>
#include <vector>
#include <algorithm>

#include <tests/support/replayer_payload.hpp>
#include <tests/support/wal_replay_helpers.hpp>

using namespace vesper;
using namespace test_support;

namespace {
namespace fs = std::filesystem;

static std::vector<fs::path> list_wal_files_sorted(const fs::path& dir){
  std::vector<std::pair<std::uint64_t, fs::path>> v;
  for (auto& de : fs::directory_iterator(dir)){
    if (!de.is_regular_file()) continue;
    auto name = de.path().filename().string();
    if (name.rfind("wal-", 0)==0 && name.size() >= 4+8) {
      try { auto seq = static_cast<std::uint64_t>(std::stoull(name.substr(4,8))); v.emplace_back(seq, de.path()); } catch(...) {}
    }
  }
  std::sort(v.begin(), v.end(), [](auto& a, auto& b){ return a.first < b.first; });
  std::vector<fs::path> out; out.reserve(v.size());
  for (auto& kv : v) out.push_back(kv.second);
  return out;
}

static void make_sequence_dir(const fs::path& dir){
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);
  wal::WalWriterOptions opts{.dir=dir, .prefix="wal-", .max_file_bytes=32, .strict_lsn_monotonic=false};
  auto w = wal::WalWriter::open(opts); REQUIRE(w.has_value());
  auto up101 = make_upsert(101, std::vector<float>{1.0f, 2.0f}, {{"a","b"}});
  auto up102 = make_upsert(102, std::vector<float>{2.0f, 3.0f}, {{"c","d"}});
  auto del101 = make_delete(101);
  auto up103 = make_upsert(103, std::vector<float>{3.0f, 4.0f}, {{"e","f"}});
  auto up104 = make_upsert(104, std::vector<float>{4.0f, 5.0f}, {{"g","h"}});
  REQUIRE(w->append(1, /*type=*/1, up101).has_value());
  REQUIRE(w->append(2, /*type=*/1, up102).has_value());
  REQUIRE(w->append(3, /*type=*/2, del101).has_value());
  REQUIRE(w->append(4, /*type=*/1, up103).has_value());
  REQUIRE(w->append(5, /*type=*/1, up104).has_value());
  REQUIRE(w->flush(false).has_value());
}

static void make_manifest_stale_drop_last(const fs::path& dir){
  auto files = list_wal_files_sorted(dir); REQUIRE(files.size() >= 2);
  auto last_file = files.back().filename().string();
  auto manifest_path = dir / "wal.manifest";
  REQUIRE(fs::exists(manifest_path));
  std::ifstream in(manifest_path);
  REQUIRE(in.good());
  std::string header; std::getline(in, header);
  REQUIRE(header == std::string("vesper-wal-manifest v1"));
  std::vector<std::string> lines; std::string line;
  while (std::getline(in, line)) lines.push_back(line);
  in.close();
  std::vector<std::string> kept;
  for (const auto& ln : lines){ if (ln.find(last_file) == std::string::npos) kept.push_back(ln); }
  {
    std::ofstream out(manifest_path, std::ios::binary | std::ios::trunc);
    REQUIRE(out.good());
    out << "vesper-wal-manifest v1\n";
    for (const auto& ln : kept) out << ln << "\n";
  }
  // verify last_file is absent
  {
    std::ifstream chk(manifest_path);
    REQUIRE(chk.good()); std::string x; bool seen=false; while (std::getline(chk, x)) if (x.find(last_file)!=std::string::npos) seen=true; REQUIRE(!seen);
  }
}

static void assert_post_purge_stats_and_state(const fs::path& dir, std::uint64_t cutoff){
  // Stats
  std::size_t delivered=0; auto r = wal::recover_scan_dir(dir, [&](const wal::WalFrame& f){ delivered += f.payload.size(); });
  REQUIRE(r.has_value()); auto st = *r;
  REQUIRE(st.frames == 2);
  REQUIRE(st.last_lsn == 5);
  REQUIRE(st.type_counts[1] == 2);
  REQUIRE(st.type_counts[2] == 0);
  REQUIRE(st.lsn_monotonic == true);
  REQUIRE(delivered > 0);
  // Replay state
  ToyIndex idx = build_toy_index_baseline_then_replay(dir, cutoff);
  REQUIRE(idx.count(101) == 0);
  REQUIRE(idx.count(102) == 1);
  REQUIRE(idx.count(103) == 1);
  REQUIRE(idx.count(104) == 1);
  REQUIRE(idx.at(102).vec[0] == Catch::Approx(2.0f));
  REQUIRE(idx.at(103).vec[0] == Catch::Approx(3.0f));
  REQUIRE(idx.at(104).vec[0] == Catch::Approx(4.0f));
}

} // namespace

TEST_CASE("purge_wal honors cutoff with manifest present and stale", "[wal][manifest][purge]"){
  namespace fs = std::filesystem;
  const std::uint64_t cutoff = 3;

  // Case 1: Up-to-date manifest
  fs::path dir1 = fs::temp_directory_path() / "vesper_wal_purge_manifest";
  make_sequence_dir(dir1);
  auto files1_before = list_wal_files_sorted(dir1); REQUIRE(files1_before.size() >= 3);
  REQUIRE(wal::purge_wal(dir1, cutoff).has_value());
  auto files1_after = list_wal_files_sorted(dir1);
  // Only files with frames lsn>3 remain; since we wrote one frame per file, last two files remain
  REQUIRE(files1_after.size() == 2);
  assert_post_purge_stats_and_state(dir1, cutoff);

  // Case 2: Stale manifest (drop last file entry), then purge
  fs::path dir2 = fs::temp_directory_path() / "vesper_wal_purge_manifest_stale";
  make_sequence_dir(dir2);
  auto files2_before = list_wal_files_sorted(dir2); REQUIRE(files2_before.size() >= 3);
  make_manifest_stale_drop_last(dir2);
  REQUIRE(wal::purge_wal(dir2, cutoff).has_value());
  auto files2_after = list_wal_files_sorted(dir2);
  REQUIRE(files2_after.size() == files1_after.size());
  // Compare filename sets
  std::vector<std::string> names1, names2;
  for (auto& p : files1_after) names1.push_back(p.filename().string());
  for (auto& p : files2_after) names2.push_back(p.filename().string());
  REQUIRE(names1 == names2);
  assert_post_purge_stats_and_state(dir2, cutoff);

  // Case 3: Idempotency
  auto files_before_idem = list_wal_files_sorted(dir2);
  REQUIRE(wal::purge_wal(dir2, cutoff).has_value());
  auto files_after_idem = list_wal_files_sorted(dir2);
  REQUIRE(files_before_idem == files_after_idem);
  assert_post_purge_stats_and_state(dir2, cutoff);

  std::error_code ec; fs::remove_all(dir1, ec); fs::remove_all(dir2, ec);
}

