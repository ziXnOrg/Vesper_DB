#include <catch2/catch_all.hpp>
#include <vesper/wal/io.hpp>
#include <vesper/wal/replay.hpp>
#include <filesystem>
#include <vector>

#include <tests/support/replayer_payload.hpp>
#include <tests/support/wal_replay_helpers.hpp>

using namespace vesper;
using namespace test_support;

TEST_CASE("replay type mask filters delivered frames and stats reflect post-filter", "[wal][replay][mask]"){
  namespace fs = std::filesystem;
  auto dir = fs::temp_directory_path() / "vesper_wal_replay_mask";
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);

  wal::WalWriterOptions opts{.dir=dir, .prefix="wal-", .max_file_bytes=48, .strict_lsn_monotonic=false};
  auto w = wal::WalWriter::open(opts); REQUIRE(w.has_value());
  auto up1 = make_upsert(901, std::vector<float>{1.0f}, {});
  auto com = make_upsert(902, std::vector<float>{2.0f}, {}); // treat as type=2 for test; payload ignored
  auto pad = std::vector<std::uint8_t>(5, 0);
  REQUIRE(w->append(1, 1, up1).has_value());
  REQUIRE(w->append(2, 2, com).has_value());
  REQUIRE(w->append(3, 3, pad).has_value());
  REQUIRE(w->flush(false).has_value());

  // Mask for {1,2}: expect 2 frames delivered and ToyIndex includes only upsert+commit payload (commit ignored by ToyIndex helper)
  std::vector<std::pair<uint64_t,uint16_t>> seen;
  auto st12 = wal::recover_replay(dir, /*type_mask=*/((1u<<1)|(1u<<2)), [&](uint64_t lsn, uint16_t type, std::span<const uint8_t>){ seen.emplace_back(lsn,type); });
  REQUIRE(st12.has_value());
  REQUIRE(seen.size() == 2);
  REQUIRE(seen[0].second == 1);
  REQUIRE(seen[1].second == 2);

  // Mask for {3}: delivered frames are padding only; ToyIndex unchanged
  std::vector<std::pair<uint64_t,uint16_t>> seen3;
  auto st3 = wal::recover_replay(dir, /*type_mask=*/(1u<<3), [&](uint64_t lsn, uint16_t type, std::span<const uint8_t>){ seen3.emplace_back(lsn,type); });
  REQUIRE(st3.has_value());
  REQUIRE(seen3.size() == 1);
  REQUIRE(seen3[0].second == 3);

  fs::remove_all(dir, ec);
}

