#include <catch2/catch_all.hpp>
#include <vesper/wal/io.hpp>
#include <vesper/wal/replay.hpp>
#include <vesper/wal/snapshot.hpp>
#include <vesper/wal/retention.hpp>
#include <filesystem>

using namespace vesper;

TEST_CASE("recover_replay reconstructs payloads post-snapshot", "[wal][replay]") {
  namespace fs = std::filesystem;
  auto dir = fs::temp_directory_path() / "vesper_wal_replay";
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);
  wal::WalWriterOptions opts{.dir=dir, .prefix="wal-", .max_file_bytes=64};
  auto w = wal::WalWriter::open(opts); REQUIRE(w.has_value());

  std::vector<std::vector<uint8_t>> payloads;
  const int N = 5;
  for (int i=1;i<=N;++i){ std::vector<uint8_t> p{static_cast<uint8_t>(i)}; payloads.push_back(p); REQUIRE(w->append(i, 1, p).has_value()); }
  REQUIRE(w->flush(false).has_value());

  // Snapshot at 2
  wal::Snapshot s{.last_lsn=2}; REQUIRE(wal::save_snapshot(dir, s).has_value());

  std::vector<uint8_t> collected;
  auto stats_exp = wal::recover_replay(dir, [&](std::uint64_t lsn, std::uint16_t, std::span<const uint8_t> pl){ collected.insert(collected.end(), pl.begin(), pl.end()); });
  REQUIRE(stats_exp.has_value());
  // Expect payloads for lsn 3..5
  std::vector<uint8_t> expected; for (int i=3;i<=5;++i) expected.push_back(static_cast<uint8_t>(i));
  REQUIRE(collected == expected);

  fs::remove_all(dir, ec);
}

TEST_CASE("purge_wal removes files fully covered by cutoff LSN", "[wal][retention]") {
  namespace fs = std::filesystem;
  auto dir = fs::temp_directory_path() / "vesper_wal_purge";
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);
  wal::WalWriterOptions opts{.dir=dir, .prefix="wal-", .max_file_bytes=64};
  auto w = wal::WalWriter::open(opts); REQUIRE(w.has_value());
  std::vector<uint8_t> p{0xAA};
  REQUIRE(w->append(1, 1, p).has_value());
  REQUIRE(w->append(2, 1, p).has_value());
  REQUIRE(w->flush(false).has_value());
  REQUIRE(w->append(3, 1, p).has_value());
  REQUIRE(w->flush(false).has_value());

  // Save manifest should exist due to rotation; if only one file, skip test
  int files = 0; for (auto& de : fs::directory_iterator(dir)) if (de.is_regular_file()) ++files;
  REQUIRE(files >= 1);

  // Purge up to LSN 2
  REQUIRE(wal::purge_wal(dir, 2).has_value());
  // Verify that any file with end_lsn <=2 is gone by reloading manifest
  auto m = wal::load_manifest(dir); REQUIRE(m.has_value());
  for (auto& e : m->entries) REQUIRE(e.end_lsn > 2);

  fs::remove_all(dir, ec);
}

TEST_CASE("writer strict LSN monotonic option", "[wal][writer]") {
  namespace fs = std::filesystem;
  auto dir = fs::temp_directory_path() / "vesper_wal_strict";
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);
  wal::WalWriterOptions opts{.dir=dir, .prefix="wal-", .max_file_bytes=0, .strict_lsn_monotonic=true};
  auto w = wal::WalWriter::open(opts); REQUIRE(w.has_value());
  std::vector<uint8_t> p{0x01};
  REQUIRE(w->append(10, 1, p).has_value());
  auto st = w->append(9, 1, p);
  REQUIRE_FALSE(st.has_value());
  fs::remove_all(dir, ec);
}

