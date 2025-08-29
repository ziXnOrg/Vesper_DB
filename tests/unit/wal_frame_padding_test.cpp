#include <catch2/catch_all.hpp>
#include <vesper/wal/io.hpp>
#include <vesper/wal/replay.hpp>
#include <vesper/wal/snapshot.hpp>
#include <filesystem>
#include <tests/support/replayer_payload.hpp>
#include <tests/support/wal_replay_helpers.hpp>

using namespace vesper;
using namespace test_support;

TEST_CASE("padding frames (type=3) are ignored by monotonicity and replay", "[wal][io][replay][padding]"){
  namespace fs = std::filesystem;

  // Part 1: LSN monotonicity exclusion for type=3 in a single WAL file
  {
    auto tmp = fs::temp_directory_path() / "vesper_wal_padding_single.bin";
    std::error_code ec; fs::remove(tmp, ec);

    auto w = wal::WalWriter::open(tmp.string());
    REQUIRE(w.has_value());
    std::vector<std::uint8_t> p1{0xAA};
    std::vector<std::uint8_t> ppad{0x00};

    REQUIRE(w->append(10, /*type=*/1, p1).has_value());
    REQUIRE(w->append( 9, /*type=*/3, ppad).has_value()); // padding; non-monotonic vs previous but should be ignored in monotonicity
    REQUIRE(w->append(11, /*type=*/1, p1).has_value());
    REQUIRE(w->flush(false).has_value());

    std::size_t delivered = 0;
    auto stats_exp = wal::recover_scan(tmp.string(), [&](const wal::WalFrame& f){ delivered += f.payload.size(); });
    REQUIRE(stats_exp.has_value());
    auto s = *stats_exp;
    REQUIRE(s.frames == 3);
    REQUIRE(s.lsn_monotonic == true);
    REQUIRE(s.lsn_violations == 0);
    REQUIRE(s.type_counts[1] == 2);
    REQUIRE(s.type_counts[3] == 1);
    REQUIRE(s.last_lsn == 11);
    REQUIRE(delivered == 3);

    fs::remove(tmp, ec);
  }

  // Part 2: Replay tolerance of type=3 (no effect on ToyIndex), with rotation + snapshot cutoff
  {
    auto dir = fs::temp_directory_path() / "vesper_wal_padding";
    std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);

    wal::WalWriterOptions opts{.dir=dir, .prefix="wal-", .max_file_bytes=64, .strict_lsn_monotonic=false};
    auto w = wal::WalWriter::open(opts);
    REQUIRE(w.has_value());

    // Baseline (<= cutoff)
    auto pl_up_101 = make_upsert(101, std::vector<float>{1.0f, 2.0f}, {{"k","v"}});
    auto pl_del_101 = make_delete(101);
    REQUIRE(w->append(1, /*type=*/1, pl_up_101).has_value());
    REQUIRE(w->append(2, /*type=*/2, pl_del_101).has_value());

    // Post-cutoff (> cutoff)
    std::vector<std::uint8_t> ppad{0xFF};
    REQUIRE(w->append(3, /*type=*/3, ppad).has_value());
    auto pl_up_102 = make_upsert(102, std::vector<float>{3.0f, 4.0f}, {{"t","x"}});
    REQUIRE(w->append(4, /*type=*/1, pl_up_102).has_value());

    REQUIRE(w->flush(false).has_value());

    const std::uint64_t cutoff = 2;
    REQUIRE(wal::save_snapshot(dir, wal::Snapshot{.last_lsn=cutoff}).has_value());

    // Expected index: baseline (upsert then delete id=101) => id=101 absent; then upsert id=102
    ToyIndex idx_expected;
    apply_frame_payload(pl_up_101, idx_expected);
    apply_frame_payload(pl_del_101, idx_expected);
    apply_frame_payload(pl_up_102, idx_expected);

    // Replayed index using helper
    ToyIndex idx_replayed = build_toy_index_baseline_then_replay(dir, cutoff);

    // Also capture stats from recover_replay to verify type_counts includes padding
    auto stats_exp = wal::recover_replay(dir, [&](std::uint64_t, std::uint16_t, std::span<const std::uint8_t>){ /* no-op */ });
    REQUIRE(stats_exp.has_value());
    auto st = *stats_exp;
    REQUIRE(st.type_counts[3] == 1); // padding at lsn=3 (>cutoff)
    REQUIRE(st.type_counts[1] == 1); // upsert at lsn=4 (>cutoff)
    REQUIRE(st.frames == 2);
    REQUIRE(st.last_lsn == 4);

    // Assertions on index state
    REQUIRE(idx_replayed.count(101) == 0);
    REQUIRE(idx_replayed.count(102) == 1);
    const auto& d = idx_replayed.at(102);
    REQUIRE(d.vec.size() == 2);
    REQUIRE(d.vec[0] == Catch::Approx(3.0f));
    REQUIRE(d.vec[1] == Catch::Approx(4.0f));
    REQUIRE(d.tags.at("t") == "x");

    fs::remove_all(dir, ec);
  }
}

