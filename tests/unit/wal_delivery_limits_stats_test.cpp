#include <catch2/catch_all.hpp>
#include <vesper/wal/io.hpp>
#include <filesystem>
#include <vector>

using namespace vesper;

TEST_CASE("recover_scan_dir stats reflect post-filter delivery for DeliveryLimits", "[wal][replay][scan][limits][stats]"){
  namespace fs = std::filesystem;
  auto dir = fs::temp_directory_path() / "vesper_wal_limits_stats";
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

  // Case 1: cutoff at 1, mask {1,2}, max_frames=1 => deliver only lsn=2 (type 2)
  {
    wal::DeliveryLimits lim{}; lim.cutoff_lsn = 1; lim.type_mask = (1u<<1)|(1u<<2); lim.max_frames = 1;
    auto st = wal::recover_scan_dir(dir, lim, [&](const wal::WalFrame&){ /* no-op */ });
    REQUIRE(st.has_value());
    REQUIRE(st->frames == 1);
    REQUIRE(st->bytes == (0 + 24));
    REQUIRE(st->last_lsn == 2);
    REQUIRE(st->type_counts[1] == 0);
    REQUIRE(st->type_counts[2] == 1);
    REQUIRE(st->type_counts[3] == 0);
  }

  // Case 2: mask {3}, max_bytes fits exactly first padding frame
  {
    wal::DeliveryLimits lim{}; lim.type_mask = (1u<<3); lim.max_bytes = (5 + 24);
    auto st = wal::recover_scan_dir(dir, lim, [&](const wal::WalFrame&){ /* no-op */ });
    REQUIRE(st.has_value());
    REQUIRE(st->frames == 1);
    REQUIRE(st->bytes == (5 + 24));
    REQUIRE(st->last_lsn == 3);
    REQUIRE(st->type_counts[1] == 0);
    REQUIRE(st->type_counts[2] == 0);
    REQUIRE(st->type_counts[3] == 1);
  }

  fs::remove_all(dir, ec);
}

