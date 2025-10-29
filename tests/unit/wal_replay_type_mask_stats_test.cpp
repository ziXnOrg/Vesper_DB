#include <catch2/catch_all.hpp>
#include <vesper/wal/io.hpp>
#include <vesper/wal/replay.hpp>
#include <filesystem>
#include <vector>

using namespace vesper;

TEST_CASE("recover_replay stats reflect post-filter delivery for type mask", "[wal][replay][mask][stats]"){
  namespace fs = std::filesystem;
  auto dir = fs::temp_directory_path() / "vesper_wal_replay_mask_stats";
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);

  wal::WalWriterOptions opts{.dir=dir, .prefix="wal-", .max_file_bytes=0, .strict_lsn_monotonic=false};
  auto w = wal::WalWriter::open(opts); REQUIRE(w.has_value());

  // Known payload sizes
  std::vector<std::uint8_t> p1(2, 0);   // type=1
  std::vector<std::uint8_t> p2;         // type=2 (commit), zero-length payload
  std::vector<std::uint8_t> p3(5, 0);   // type=3 (padding)

  REQUIRE(w->append(1, 1, p1).has_value());
  REQUIRE(w->append(2, 2, p2).has_value());
  REQUIRE(w->append(3, 3, p3).has_value());
  REQUIRE(w->flush(false).has_value());

  // Mask for {1,2}: expect only two frames delivered; stats must reflect delivered only
  std::vector<std::pair<std::uint64_t, std::uint16_t>> seen;
  auto st = wal::recover_replay(dir, /*type_mask=*/((1u<<1)|(1u<<2)),
    [&](std::uint64_t lsn, std::uint16_t type, std::span<const std::uint8_t>){ seen.emplace_back(lsn, type); }
  );
  REQUIRE(st.has_value());
  REQUIRE(seen.size() == 2);
  REQUIRE(seen[0].second == 1);
  REQUIRE(seen[1].second == 2);

  // Expected bytes: LEN = payload + header(20) + CRC(4) => payload+24
  const std::size_t expected_bytes = (2 + 24) + (0 + 24);
  REQUIRE(st->frames == 2);                 // should reflect post-filter delivery
  REQUIRE(st->bytes == expected_bytes);     // only types {1,2}
  REQUIRE(st->last_lsn == 2);               // last delivered among mask {1,2}
  REQUIRE(st->type_counts[1] == 1);
  REQUIRE(st->type_counts[2] == 1);
  REQUIRE(st->type_counts[3] == 0);

  fs::remove_all(dir, ec);
}

