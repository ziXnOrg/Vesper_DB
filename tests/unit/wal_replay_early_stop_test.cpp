#include <catch2/catch_all.hpp>
#include <vesper/wal/replay.hpp>
#include <vesper/wal/io.hpp>
#include <filesystem>
#include <vector>

using namespace vesper;

TEST_CASE("recover_replay supports early stop and error propagation via result-callback", "[wal][replay][early_stop][stats]"){
  namespace fs = std::filesystem;
  auto dir = fs::temp_directory_path() / "vesper_wal_replay_early_stop";
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);

  wal::WalWriterOptions opts{.dir=dir, .prefix="wal-", .max_file_bytes=0, .strict_lsn_monotonic=false};
  auto w = wal::WalWriter::open(opts); REQUIRE(w.has_value());

  std::vector<std::uint8_t> p1(1, 0);   // type=1
  std::vector<std::uint8_t> p2(1, 0);   // type=2
  std::vector<std::uint8_t> p3(1, 0);   // type=3
  REQUIRE(w->append(1, 1, p1).has_value());
  REQUIRE(w->append(2, 2, p2).has_value());
  REQUIRE(w->append(3, 3, p3).has_value());
  REQUIRE(w->flush(false).has_value());

  // Case 1: early stop (return false after first delivered frame)
  {
    std::size_t delivered = 0;
    // New overload: callback returns expected<bool,error>
    auto st = wal::recover_replay(
      dir,
      static_cast<wal::ReplayResultCallback>(
        [&](std::uint64_t, std::uint16_t, std::span<const std::uint8_t>) -> std::expected<wal::DeliverDecision, vesper::core::error> {
          delivered++;
          return std::expected<wal::DeliverDecision, vesper::core::error>{wal::DeliverDecision::DeliverAndStop};
        }));
    REQUIRE(st.has_value());
    REQUIRE(delivered == 1);
    REQUIRE(st->frames == 1);
  }

  // Case 2: error propagation (return unexpected(error))
  {
    std::size_t delivered = 0;
    auto st = wal::recover_replay(
      dir,
      static_cast<wal::ReplayResultCallback>(
        [&](std::uint64_t, std::uint16_t, std::span<const std::uint8_t>) -> std::expected<wal::DeliverDecision, vesper::core::error> {
          delivered++;
          return std::vesper_unexpected(vesper::core::error{vesper::core::error_code::io_failed, "injected", "wal.replay.test"});
        }));
    REQUIRE_FALSE(st.has_value());
    REQUIRE(delivered == 1);
  }

  fs::remove_all(dir, ec);
}

