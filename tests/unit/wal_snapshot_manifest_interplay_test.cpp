#include <catch2/catch_all.hpp>
#include <vesper/wal/io.hpp>
#include <vesper/wal/replay.hpp>
#include <vesper/wal/snapshot.hpp>
#include <filesystem>
#include <fstream>
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
  auto files = wal_files_sorted(dir); REQUIRE(!files.empty());
  auto last_file = files.back().second.filename().string();
  auto manifest_path = dir / "wal.manifest"; REQUIRE(fs::exists(manifest_path));
  std::ifstream in(manifest_path); REQUIRE(in.good()); std::string header; std::getline(in, header); REQUIRE(header == std::string("vesper-wal-manifest v1"));
  std::vector<std::string> lines; std::string line; while (std::getline(in, line)) lines.push_back(line);
  in.close();
  std::vector<std::string> kept; for (auto& ln : lines) if (ln.find(last_file)==std::string::npos) kept.push_back(ln);
  std::ofstream out(manifest_path, std::ios::binary | std::ios::trunc); REQUIRE(out.good());
  out << "vesper-wal-manifest v1\n"; for (auto& ln : kept) out << ln << "\n";
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

