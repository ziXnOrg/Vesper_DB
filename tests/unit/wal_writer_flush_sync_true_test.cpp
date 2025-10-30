#include <catch2/catch_all.hpp>
#include <vesper/wal/io.hpp>
#include <filesystem>

#include <tests/support/replayer_payload.hpp>

using namespace vesper;

TEST_CASE("WalWriter flush(true) triggers sync regardless of fsync_on_flush knob", "[wal][writer][flush][sync]"){
  namespace fs = std::filesystem;
  auto dir = fs::temp_directory_path() / "vesper_wal_flush_true";
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);

  wal::WalWriterOptions opts{.dir=dir, .prefix="wal-", .max_file_bytes=0, .strict_lsn_monotonic=false};
  auto w = wal::WalWriter::open(opts);
  REQUIRE(w.has_value());

  auto up = test_support::make_upsert(1234, std::vector<float>{1.0f}, {});
  REQUIRE(w->append(1, 1, up).has_value());

  // flush(true) should perform a sync and increment stats even when fsync_on_flush is false
  REQUIRE(w->flush(true).has_value());
  REQUIRE(w->stats().flushes >= 1);
  REQUIRE(w->stats().syncs >= 1);

  fs::remove_all(dir, ec);
}

