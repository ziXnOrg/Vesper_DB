#include <catch2/catch_all.hpp>
#include <vesper/wal/io.hpp>
#include <vesper/wal/replay.hpp>
#include <vesper/wal/snapshot.hpp>
#include <filesystem>

using namespace vesper;

TEST_CASE("e2e replay reconstructs state deterministically", "[wal][replay][e2e]") {
  namespace fs = std::filesystem;
  auto dir = fs::temp_directory_path() / "vesper_wal_replay_e2e";
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);
  wal::WalWriterOptions opts{.dir=dir, .prefix="wal-", .max_file_bytes=64};
  auto w = wal::WalWriter::open(opts); REQUIRE(w.has_value());

  // Write N frames with known payloads across rotations
  const int N = 7; std::vector<std::vector<uint8_t>> payloads;
  for (int i=1;i<=N;++i) { std::vector<uint8_t> p(3, static_cast<uint8_t>(i)); payloads.push_back(p); REQUIRE(w->append(i, 1, p).has_value()); }
  REQUIRE(w->flush(false).has_value());

  // Snapshot cutoff at M
  wal::Snapshot s{.last_lsn=3}; REQUIRE(wal::save_snapshot(dir, s).has_value());

  // Truncate last file by 1 byte (torn tail allowed)
  std::filesystem::path last; std::uint64_t max_seq=0;
  for (auto& de : fs::directory_iterator(dir)){
    if (!de.is_regular_file()) continue; auto name = de.path().filename().string();
    if (name.rfind("wal-", 0)==0) { auto seq = std::stoull(name.substr(4,8)); if (seq>max_seq){ max_seq=seq; last=de.path(); } }
  }
  REQUIRE(!last.empty()); auto sz = fs::file_size(last); fs::resize_file(last, sz-1, ec); REQUIRE(!ec);

  // Replay and accumulate state
  std::vector<uint8_t> state;
  auto stats_exp = wal::recover_replay(dir, [&](std::uint64_t lsn, std::uint16_t, std::span<const uint8_t> pl){ state.insert(state.end(), pl.begin(), pl.end()); });
  REQUIRE(stats_exp.has_value());

  // Expected concatenation of payloads strictly after cutoff; last file was torn by 1 byte, so the final frame may be incomplete and is ignored.
  std::vector<uint8_t> expected; for (int i=s.last_lsn+1;i<=N-1;++i) expected.insert(expected.end(), payloads[i-1].begin(), payloads[i-1].end());
  REQUIRE(state == expected);

  fs::remove_all(dir, ec);
}

