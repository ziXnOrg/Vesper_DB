#include <catch2/catch_all.hpp>
#include <vesper/wal/io.hpp>
#include <vesper/wal/replay.hpp>
#include <filesystem>

#include <tests/support/replayer_payload.hpp>

using namespace vesper;
using namespace test_support;

TEST_CASE("DeliveryLimits cutoff/type/limits control delivery deterministically", "[wal][replay][scan][limits]"){
  namespace fs = std::filesystem;
  auto dir = fs::temp_directory_path() / "vesper_wal_limits";
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);

  wal::WalWriterOptions opts{.dir=dir, .prefix="wal-", .max_file_bytes=48, .strict_lsn_monotonic=false};
  auto w = wal::WalWriter::open(opts); REQUIRE(w.has_value());
  auto up1 = make_upsert(1001, std::vector<float>{1.0f}, {});
  auto up2 = make_upsert(1002, std::vector<float>{2.0f}, {});
  std::vector<uint8_t> pad(4, 0);
  REQUIRE(w->append(1, 1, up1).has_value());
  REQUIRE(w->append(2, 3, pad).has_value());
  REQUIRE(w->append(3, 2, up2).has_value());
  REQUIRE(w->flush(false).has_value());

  // cutoff override at 1: expect only lsn>1
  {
    wal::DeliveryLimits lim{}; lim.cutoff_lsn = 1; std::vector<uint64_t> lsns;
    auto st = wal::recover_scan_dir(dir, lim, [&](const wal::WalFrame& f){ lsns.push_back(f.lsn); });
    REQUIRE(st.has_value());
    REQUIRE(lsns.size() == 2);
    REQUIRE(lsns.front() == 2);
    REQUIRE(lsns.back() == 3);
  }

  // type mask {1,2} with max_frames=1: only first matching frame delivered
  {
    wal::DeliveryLimits lim{}; lim.type_mask = (1u<<1)|(1u<<2); lim.max_frames = 1; std::vector<uint16_t> types;
    auto st = wal::recover_scan_dir(dir, lim, [&](const wal::WalFrame& f){ types.push_back(f.type); });
    REQUIRE(st.has_value());
    REQUIRE(types.size() == 1);
    REQUIRE(types[0] == 1);
  }

  // type mask {3} with max_bytes that fits exactly padding frame
  {
    wal::DeliveryLimits lim{}; lim.type_mask = (1u<<3); lim.max_bytes = 8; std::vector<uint16_t> types;
    auto st = wal::recover_scan_dir(dir, lim, [&](const wal::WalFrame& f){ types.push_back(f.type); });
    REQUIRE(st.has_value());
    REQUIRE(types.size() == 1);
    REQUIRE(types[0] == 3);
  }

  fs::remove_all(dir, ec);
}

