#include <catch2/catch_all.hpp>
#include <vesper/wal/io.hpp>
#include <filesystem>

#include <tests/support/replayer_payload.hpp>

using namespace vesper;

TEST_CASE("WalWriter fsync policy controls increment stats without changing semantics", "[wal][fsync][writer]"){
  namespace fs = std::filesystem;
  auto dir = fs::temp_directory_path() / "vesper_wal_fsync";
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);

  wal::WalWriterOptions base{.dir=dir, .prefix="wal-", .max_file_bytes=48, .strict_lsn_monotonic=false};

  // Baseline: no syncs
  {
    auto w = wal::WalWriter::open(base); REQUIRE(w.has_value());
    for (int i=1;i<=3;++i) { auto up = test_support::make_upsert(600+i, std::vector<float>{float(i)}, {}); REQUIRE(w->append(i, 1, up).has_value()); }
    REQUIRE(w->flush(false).has_value());
    REQUIRE(w->stats().syncs == 0);
  }

  // fsync on flush enabled
  {
    auto opts = base; opts.fsync_on_flush = true;
    auto w = wal::WalWriter::open(opts); REQUIRE(w.has_value());
    auto up = test_support::make_upsert(700, std::vector<float>{1.0f}, {});
    REQUIRE(w->append(1, 1, up).has_value());
    REQUIRE(w->flush(false).has_value());
    REQUIRE(w->stats().syncs >= 1);
  }

  // fsync on rotation enabled
  {
    auto opts = base; opts.fsync_on_rotation = true; opts.max_file_bytes = 32; // force rotation
    auto w = wal::WalWriter::open(opts); REQUIRE(w.has_value());
    for (int i=1;i<=4;++i) { auto up = test_support::make_upsert(800+i, std::vector<float>{float(i)}, {}); REQUIRE(w->append(i, 1, up).has_value()); }
    REQUIRE(w->flush(false).has_value());
    REQUIRE(w->stats().rotations >= 1);
    REQUIRE(w->stats().syncs >= 1);
  }

  fs::remove_all(dir, ec);
}

