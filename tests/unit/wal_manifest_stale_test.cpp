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

TEST_CASE("stale manifest still scans highest-seq file", "[wal][manifest][replay][stale]"){
  namespace fs = std::filesystem;
  auto dir = fs::temp_directory_path() / "vesper_wal_manifest_stale";
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);

  wal::WalWriterOptions opts{.dir=dir, .prefix="wal-", .max_file_bytes=64, .strict_lsn_monotonic=false};
  auto w = wal::WalWriter::open(opts); REQUIRE(w.has_value());

  // Deterministic frames ensuring at least one rotation
  auto up101 = make_upsert(101, std::vector<float>{1.0f, 2.0f}, {{"a","b"}});
  auto up102 = make_upsert(102, std::vector<float>{2.0f, 3.0f}, {{"c","d"}});
  auto del101 = make_delete(101);
  auto up103 = make_upsert(103, std::vector<float>{3.0f, 4.0f}, {{"e","f"}});

  REQUIRE(w->append(1, /*type=*/1, up101).has_value());
  REQUIRE(w->append(2, /*type=*/1, up102).has_value());
  REQUIRE(w->append(3, /*type=*/2, del101).has_value());
  REQUIRE(w->append(4, /*type=*/1, up103).has_value());
  REQUIRE(w->flush(false).has_value());

  // Enumerate wal-* files and ensure rotation occurred
  std::vector<std::pair<std::uint64_t, fs::path>> files;
  for (auto& de : fs::directory_iterator(dir)){
    if (!de.is_regular_file()) continue; auto name = de.path().filename().string();
    if (name.rfind("wal-", 0)==0 && name.size() >= 4+8) {
      try { auto seq = static_cast<std::uint64_t>(std::stoull(name.substr(4,8))); files.emplace_back(seq, de.path()); } catch (...) {}
    }
  }
  std::sort(files.begin(), files.end(), [](auto& a, auto& b){ return a.first < b.first; });
  REQUIRE(files.size() >= 2);
  auto last_file = files.back().second.filename().string();

  // Make manifest stale by removing last file's entry
  auto manifest_path = dir / "wal.manifest";
  REQUIRE(fs::exists(manifest_path));
  std::ifstream in(manifest_path);
  REQUIRE(in.good());
  std::string header; std::getline(in, header);
  REQUIRE(header == std::string("vesper-wal-manifest v1"));
  std::vector<std::string> lines; std::string line;
  while (std::getline(in, line)) lines.push_back(line);
  in.close();
  // Filter out the line with file=<last_file>
  std::vector<std::string> kept;
  for (const auto& ln : lines){
    if (ln.rfind(std::string("file=") + last_file + " ", 0) == 0) continue;
    kept.push_back(ln);
  }
  // Rewrite manifest without the last entry
  {
    std::ofstream out(manifest_path, std::ios::binary | std::ios::trunc);
    REQUIRE(out.good());
    out << "vesper-wal-manifest v1\n";
    for (const auto& ln : kept) out << ln << "\n";
  }
  // Verify last_file is absent from manifest
  {
    std::ifstream chk(manifest_path);
    REQUIRE(chk.good());
    std::string x; bool seen=false;
    while (std::getline(chk, x)) { if (x.find(last_file) != std::string::npos) seen=true; }
    REQUIRE(seen == false);
  }

  // Scan directory: even with stale manifest, highest-seq file must be included
  std::size_t delivered_bytes = 0; std::vector<std::uint64_t> lsns;
  auto st_dir = wal::recover_scan_dir(dir, [&](const wal::WalFrame& f){ delivered_bytes += f.payload.size(); lsns.push_back(f.lsn); });
  REQUIRE(st_dir.has_value());
  auto st = *st_dir;
  REQUIRE(st.frames == 4);
  REQUIRE(st.last_lsn == 4);
  REQUIRE(st.type_counts[1] == 3);
  REQUIRE(st.type_counts[2] == 1);
  REQUIRE(st.lsn_monotonic == true);
  REQUIRE(delivered_bytes == up101.size() + up102.size() + del101.size() + up103.size());
  // Prove we saw frames from the highest-seq file by asserting LSN 4 was delivered
  REQUIRE(std::find(lsns.begin(), lsns.end(), 4) != lsns.end());

  // Replay correctness
  ToyIndex idx_expected; apply_frame_payload(up101, idx_expected); apply_frame_payload(up102, idx_expected); apply_frame_payload(del101, idx_expected); apply_frame_payload(up103, idx_expected);
  ToyIndex idx_replayed = build_toy_index_baseline_then_replay(dir, /*cutoff=*/0);
  auto st_replay = wal::recover_replay(dir, [&](std::uint64_t, std::uint16_t, std::span<const std::uint8_t>){ /* noop */ });
  REQUIRE(st_replay.has_value());
  REQUIRE(idx_replayed.count(101) == 0);
  REQUIRE(idx_replayed.count(102) == 1);
  REQUIRE(idx_replayed.count(103) == 1);
  REQUIRE(idx_replayed.at(102).vec.size() == 2);
  REQUIRE(idx_replayed.at(103).vec.size() == 2);
  REQUIRE(idx_replayed.at(102).vec[0] == Catch::Approx(2.0f));
  REQUIRE(idx_replayed.at(103).vec[0] == Catch::Approx(3.0f));

  fs::remove_all(dir, ec);
}

