#include <catch2/catch_all.hpp>
#include <vesper/wal/io.hpp>
#include <filesystem>

#include <tests/support/replayer_payload.hpp>

using namespace vesper;

static void write_three(wal::WalWriter& w){
  auto up = test_support::make_upsert(7000, std::vector<float>{1.0f}, {});
  REQUIRE(w.append(1, 1, up).has_value());
  REQUIRE(w.append(2, 1, up).has_value());
  REQUIRE(w.append(3, 1, up).has_value());
  REQUIRE(w.flush(false).has_value());
}

TEST_CASE("WalWriter durability profiles map to fsync knobs and stats", "[wal][writer][durability]"){
  namespace fs = std::filesystem;
  auto dir = fs::temp_directory_path() / "vesper_wal_durability";
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);

  wal::WalWriterOptions base{.dir=dir, .prefix="wal-", .max_file_bytes=32, .strict_lsn_monotonic=false};

  // None
  {
    auto opts = base; opts.durability_profile = wal::DurabilityProfile::None;
    auto w = wal::WalWriter::open(opts); REQUIRE(w.has_value());
    write_three(*w);
    REQUIRE(w->stats().syncs == 0);
  }

  // Rotation
  {
    auto opts = base; opts.durability_profile = wal::DurabilityProfile::Rotation;
    auto w = wal::WalWriter::open(opts); REQUIRE(w.has_value());
    // force rotation by small max_file_bytes
    write_three(*w);
    REQUIRE(w->stats().rotations >= 1);
    REQUIRE(w->stats().syncs >= 1);
  }

  // Flush
  {
    auto opts = base; opts.durability_profile = wal::DurabilityProfile::Flush;
    auto w = wal::WalWriter::open(opts); REQUIRE(w.has_value());
    write_three(*w);
    REQUIRE(w->stats().syncs >= 1);
  }

  // RotationAndFlush
  {
    auto opts = base; opts.durability_profile = wal::DurabilityProfile::RotationAndFlush;
    auto w = wal::WalWriter::open(opts); REQUIRE(w.has_value());
    write_three(*w);
    REQUIRE(w->stats().rotations >= 1);
    REQUIRE(w->stats().syncs >= 1);
  }

  fs::remove_all(dir, ec);
}

