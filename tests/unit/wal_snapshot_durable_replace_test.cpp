#include <catch2/catch_all.hpp>
#include <vesper/wal/snapshot.hpp>
#include <filesystem>
#include <fstream>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

using namespace vesper;

TEST_CASE("save replaces existing snapshot atomically and cleans tmp", "[wal][snapshot][durable]") {
  namespace fs = std::filesystem;
  std::error_code ec;
  auto dir = fs::temp_directory_path() / "vesper_wal_snap_durable_replace";
  fs::remove_all(dir, ec); fs::create_directories(dir, ec);

  // Pre-create an existing wal.snapshot with last_lsn=1
  {
    std::ofstream out(dir / "wal.snapshot", std::ios::binary | std::ios::trunc);
    REQUIRE(out.good());
    out << "vesper-wal-snapshot v1\n";
    out << "last_lsn=1\n";
  }

  // Now save a new snapshot with last_lsn=2 — should atomically replace and remove tmp
  wal::Snapshot s{.last_lsn=2};
  auto r = wal::save_snapshot(dir, s);
  REQUIRE(r.has_value());

  // Verify final file exists and tmp is cleaned up
  REQUIRE(fs::exists(dir / "wal.snapshot"));
  REQUIRE_FALSE(fs::exists(dir / "wal.snapshot.tmp"));

  // Verify content reflects the new snapshot
  auto loaded = wal::load_snapshot(dir);
  REQUIRE(loaded.has_value());
  REQUIRE(loaded->last_lsn == 2);

  fs::remove_all(dir, ec);
}

#if defined(_WIN32)
TEST_CASE("windows: tmp cleaned when destination is locked (rename failure path)", "[wal][snapshot][durable][win]") {
  namespace fs = std::filesystem;
  std::error_code ec;
  auto dir = fs::temp_directory_path() / "vesper_wal_snap_durable_replace_locked";
  fs::remove_all(dir, ec); fs::create_directories(dir, ec);

  // Pre-create destination and lock it without FILE_SHARE_DELETE to force rename failure
  auto dest = dir / "wal.snapshot";
  {
    std::ofstream out(dest, std::ios::binary | std::ios::trunc);
    REQUIRE(out.good());
    out << "vesper-wal-snapshot v1\n";
    out << "last_lsn=1\n";
  }
  HANDLE h = ::CreateFileW(dest.wstring().c_str(), GENERIC_READ | GENERIC_WRITE,
                           FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  REQUIRE(h != INVALID_HANDLE_VALUE);

  // Attempt save; current implementation may fall back to direct write and leave tmp behind
  wal::Snapshot s{.last_lsn=2};
  (void)wal::save_snapshot(dir, s);

  // Regardless of success/failure, tmp should not linger after the operation in the durable design
  REQUIRE_FALSE(fs::exists(dir / "wal.snapshot.tmp"));

  ::CloseHandle(h);
  fs::remove_all(dir, ec);
}
#endif

