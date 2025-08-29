#include <catch2/catch_all.hpp>
#include <vesper/wal/io.hpp>
#include <vesper/wal/replay.hpp>
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
  auto up101 = make_upsert(101, std::vector<float>{1.0f, 2.0f}, {{"a","b"}});
  auto up102 = make_upsert(102, std::vector<float>{2.0f, 3.0f}, {{"c","d"}});
  auto del101 = make_delete(101);
  auto up103 = make_upsert(103, std::vector<float>{3.0f, 4.0f}, {{"e","f"}});
  REQUIRE(w->append(1, /*type=*/1, up101).has_value());
  REQUIRE(w->append(2, /*type=*/1, up102).has_value());
  REQUIRE(w->append(3, /*type=*/2, del101).has_value());
  REQUIRE(w->append(4, /*type=*/1, up103).has_value());
  REQUIRE(w->flush(false).has_value());
}

static std::vector<std::string> read_manifest_entries(const fs::path& p){
  std::ifstream in(p); REQUIRE(in.good());
  std::string header; std::getline(in, header); REQUIRE(header == std::string("vesper-wal-manifest v1"));
  std::vector<std::string> lines; std::string line; while (std::getline(in, line)) lines.push_back(line); return lines;
}

static void write_manifest(const fs::path& p, const std::vector<std::string>& lines){
  std::ofstream out(p, std::ios::binary | std::ios::trunc); REQUIRE(out.good());
  out << "vesper-wal-manifest v1\n"; for (auto& ln : lines) out << ln << "\n";
}

static void assert_all_frames_ok(const fs::path& dir){
  std::size_t bytes=0; auto st = wal::recover_scan_dir(dir, [&](const wal::WalFrame& f){ bytes += f.payload.size(); });
  REQUIRE(st.has_value()); auto s = *st;
  REQUIRE(s.frames == 4);
  REQUIRE(s.last_lsn == 4);
  REQUIRE(s.type_counts[1] == 3);
  REQUIRE(s.type_counts[2] == 1);
  REQUIRE(s.lsn_monotonic == true);
  REQUIRE(bytes > 0);
}

} // namespace

TEST_CASE("manifest tolerates order/dup/staleness and still scans correctly", "[wal][manifest][roundtrip][edges]"){
  namespace fs = std::filesystem;
  auto dir = fs::temp_directory_path() / "vesper_wal_manifest_edges";
  write_sequence(dir);
  auto files = wal_files_sorted(dir); REQUIRE(files.size() >= 2);
  auto manifest_path = dir / "wal.manifest"; REQUIRE(fs::exists(manifest_path));

  SECTION("out-of-order manifest entries"){
    auto lines = read_manifest_entries(manifest_path);
    std::reverse(lines.begin(), lines.end());
    write_manifest(manifest_path, lines);
    assert_all_frames_ok(dir);
  }

  SECTION("duplicate manifest entry for first file"){
    auto lines = read_manifest_entries(manifest_path);
    // Duplicate the first file entry line (find via filename)
    auto first_name = files.front().second.filename().string();
    auto it = std::find_if(lines.begin(), lines.end(), [&](const std::string& ln){ return ln.find(first_name) != std::string::npos; });
    REQUIRE(it != lines.end());
    lines.push_back(*it);
    write_manifest(manifest_path, lines);
    assert_all_frames_ok(dir);
  }

  SECTION("missing middle-file entry (stale subset)"){
    auto lines = read_manifest_entries(manifest_path);
    REQUIRE(files.size() >= 3);
    auto middle_name = files[files.size()/2].second.filename().string();
    std::vector<std::string> kept; kept.reserve(lines.size());
    for (auto& ln : lines) { if (ln.find(middle_name)==std::string::npos) kept.push_back(ln); }
    write_manifest(manifest_path, kept);
    assert_all_frames_ok(dir);
  }

  SECTION("header only: no entries"){
    std::vector<std::string> empty; write_manifest(manifest_path, empty);
    assert_all_frames_ok(dir);
    // Also assert replay once
    ToyIndex idx = build_toy_index_baseline_then_replay(dir, /*cutoff=*/0);
    REQUIRE(idx.count(101) == 0);
    REQUIRE(idx.count(102) == 1);
    REQUIRE(idx.count(103) == 1);
  }

  std::error_code ec; fs::remove_all(dir, ec);
}

