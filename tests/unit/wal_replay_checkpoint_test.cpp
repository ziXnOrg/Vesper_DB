#include <catch2/catch_all.hpp>
#include <vesper/wal/checkpoint.hpp>
#include <vesper/wal/io.hpp>
#include <filesystem>

#include <tests/support/replayer_payload.hpp>

using namespace vesper;

TEST_CASE("replay_from_checkpoint loads/saves and delivers post-cutoff by mask", "[wal][replay][checkpoint]"){
  namespace fs = std::filesystem;
  auto dir = fs::temp_directory_path() / "vesper_wal_ckpt";
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);

  wal::WalWriterOptions opts{.dir=dir, .prefix="wal-", .max_file_bytes=48, .strict_lsn_monotonic=false};
  auto w = wal::WalWriter::open(opts); REQUIRE(w.has_value());
  auto up1 = test_support::make_upsert(8101, std::vector<float>{1.0f}, {});
  auto up2 = test_support::make_upsert(8102, std::vector<float>{2.0f}, {});
  std::vector<uint8_t> pad(3, 0);
  REQUIRE(w->append(1, 1, up1).has_value());
  REQUIRE(w->append(2, 3, pad).has_value());
  REQUIRE(w->append(3, 2, up2).has_value());
  REQUIRE(w->flush(false).has_value());

  // Case 1: no checkpoint -> deliver all with mask {1,2}; save last_lsn==3
  {
    std::vector<std::pair<uint64_t,uint16_t>> seen;
    auto st = wal::checkpoint::replay_from_checkpoint(dir, "c1", (1u<<1)|(1u<<2), [&](uint64_t lsn, uint16_t type, std::span<const uint8_t>){ seen.emplace_back(lsn,type); });
    REQUIRE(st.has_value());
    REQUIRE(seen.size() == 2);
    REQUIRE(seen.front().first == 1);
    REQUIRE(seen.back().first == 3);
    auto ck = wal::checkpoint::load(dir, "c1"); REQUIRE(ck.has_value()); REQUIRE(ck->last_lsn == 3);
  }

  // Case 2: with checkpoint at 3 and mask {1,2} -> no delivery
  {
    std::vector<std::pair<uint64_t,uint16_t>> seen;
    auto st = wal::checkpoint::replay_from_checkpoint(dir, "c1", (1u<<1)|(1u<<2), [&](uint64_t lsn, uint16_t type, std::span<const uint8_t>){ seen.emplace_back(lsn,type); });
    REQUIRE(st.has_value());
    REQUIRE(seen.empty());
  }

  // Case 3: malformed checkpoint -> error
  {
    auto p = dir / "wal.checkpoints" / "c2.ckpt";
    std::error_code ec2; std::filesystem::create_directories(p.parent_path(), ec2);
    std::ofstream out(p); out << "bogus=1\n"; out.close();
    auto ck = wal::checkpoint::load(dir, "c2");
    REQUIRE(!ck.has_value());
  }

  fs::remove_all(dir, ec);
}

