#include <catch2/catch_all.hpp>
#include <vesper/wal/io.hpp>
#include <vesper/wal/replay.hpp>
#include <filesystem>
#include <tests/support/replayer_payload.hpp>
#include <tests/support/wal_replay_helpers.hpp>

using namespace vesper;
using namespace test_support;

TEST_CASE("non-monotonic LSN across rotation boundary", "[wal][io][replay][nonmonotonic][rotation]"){
  namespace fs = std::filesystem;
  auto dir = fs::temp_directory_path() / "vesper_wal_nonmono_rot";
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);

  // Small rotation threshold to force new file after first frame
  wal::WalWriterOptions opts{.dir=dir, .prefix="wal-", .max_file_bytes=64, .strict_lsn_monotonic=false};
  auto w = wal::WalWriter::open(opts); REQUIRE(w.has_value());

  // Frame 1: upsert id=200 at lsn=10 (approx 48B frame with dim=1; should consume most of the first file budget)
  auto up200 = make_upsert(200, std::vector<float>{1.0f}, {});
  REQUIRE(w->append(10, /*type=*/1, up200).has_value());

  // Frame 2: upsert id=201 at lsn=5 (non-monotonic vs previous); rotation should put this into a new file
  auto up201 = make_upsert(201, std::vector<float>{2.0f}, {});
  REQUIRE(w->append(5, /*type=*/1, up201).has_value());

  // Frame 3: upsert id=202 at lsn=11
  auto up202 = make_upsert(202, std::vector<float>{3.0f}, {});
  REQUIRE(w->append(11, /*type=*/1, up202).has_value());

  REQUIRE(w->flush(false).has_value());

  // Directory scan stats should detect a single monotonicity violation across files
  std::size_t bytes_acc = 0;
  auto stx = wal::recover_scan_dir(dir, [&](const wal::WalFrame& f){ bytes_acc += f.payload.size(); });
  REQUIRE(stx.has_value());
  auto s = *stx;
  REQUIRE(s.frames == 3);
  REQUIRE(s.type_counts[1] == 3);
  REQUIRE(s.lsn_monotonic == false);
  REQUIRE(s.lsn_violations == 1);
  REQUIRE(s.last_lsn == 11);

  // Replay correctness: with cutoff=0, baseline empty and replay applies all frames in delivery order
  ToyIndex idx = build_toy_index_baseline_then_replay(dir, /*cutoff=*/0);
  // Expect all three ids present with their vectors
  REQUIRE(idx.count(200) == 1);
  REQUIRE(idx.count(201) == 1);
  REQUIRE(idx.count(202) == 1);
  REQUIRE(idx.at(200).vec.size() == 1);
  REQUIRE(idx.at(201).vec.size() == 1);
  REQUIRE(idx.at(202).vec.size() == 1);
  REQUIRE(idx.at(200).vec[0] == Catch::Approx(1.0f));
  REQUIRE(idx.at(201).vec[0] == Catch::Approx(2.0f));
  REQUIRE(idx.at(202).vec[0] == Catch::Approx(3.0f));

  fs::remove_all(dir, ec);
}

