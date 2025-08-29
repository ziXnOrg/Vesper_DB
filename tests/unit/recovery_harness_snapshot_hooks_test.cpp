#include <catch2/catch_all.hpp>
#include <vesper/wal/io.hpp>
#include <vesper/wal/replay.hpp>
#include <filesystem>

using namespace vesper;

TEST_CASE("publish snapshot hook and replay harness", "[wal][snapshot][replay]") {
  namespace fs = std::filesystem;
  auto dir = fs::temp_directory_path() / "vesper_recovery_harness";
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);

  wal::WalWriterOptions opts{.dir=dir, .prefix="wal-", .max_file_bytes=64};
  auto w = wal::WalWriter::open(opts); REQUIRE(w.has_value());

  std::vector<std::vector<uint8_t>> payloads;
  for (int i=1;i<=5;++i){ std::vector<uint8_t> p(2, static_cast<uint8_t>(i)); payloads.push_back(p); REQUIRE(w->append(i, 1, p).has_value()); }
  REQUIRE(w->flush(false).has_value());

  // Publish snapshot up to LSN 3
  REQUIRE(w->publish_snapshot(3).has_value());

  // Replay should emit only LSNs 4 and 5
  std::vector<uint8_t> state;
  auto st = wal::recover_replay(dir, [&](std::uint64_t lsn, std::uint16_t, std::span<const uint8_t> pl){ state.insert(state.end(), pl.begin(), pl.end()); });
  REQUIRE(st.has_value());
  std::vector<uint8_t> expected; for (int i=4;i<=5;++i) expected.insert(expected.end(), payloads[i-1].begin(), payloads[i-1].end());
  REQUIRE(state == expected);

  fs::remove_all(dir, ec);
}

