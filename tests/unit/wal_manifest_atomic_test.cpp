#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <expected>

#include "vesper/wal/manifest.hpp"

using vesper::wal::Manifest;
using vesper::wal::ManifestEntry;
using vesper::wal::load_manifest;
using vesper::wal::save_manifest_atomic;

namespace fs = std::filesystem;

// Test IDs for audit traceability:
//   TID-WAL-MAN-ATOMIC-001: atomic manifest save leaves only final file
//   TID-WAL-MAN-ATOMIC-002: manifest save succeeds when tmp file is missing
//   TID-WAL-MAN-ATOMIC-003: (Windows) fallback to unique tmp name when default tmp is locked
//   TID-WAL-MAN-ATOMIC-004: (POSIX) save succeeds when stale tmp exists

static Manifest make_sample_manifest() {
  Manifest m{};
  ManifestEntry a{};
  a.file = "wal-00000001.log";
  a.seq = 1; a.start_lsn = 100; a.first_lsn = 100; a.end_lsn = 199; a.frames = 25; a.bytes = 8192;
  ManifestEntry b{};
  b.file = "wal-00000002.log";
  b.seq = 2; b.start_lsn = 200; b.first_lsn = 200; b.end_lsn = 299; b.frames = 26; b.bytes = 16384;
  m.entries.push_back(a);
  m.entries.push_back(b);
  return m;
}

static fs::path make_test_dir(const std::string& name) {
  auto base = fs::temp_directory_path() / "vesper_wal_manifest_atomic";
  fs::create_directories(base);
  auto dir = base / name;
  std::error_code ec;
  fs::remove_all(dir, ec);
  fs::create_directories(dir);
  return dir;
}

static void cleanup_dir(const fs::path& dir) {
  std::error_code ec;
  fs::remove_all(dir, ec);
}

TEST_CASE("wal_manifest_atomic: atomic manifest save leaves only final file", "[wal_manifest_atomic]") {
  auto dir = make_test_dir("atomic_final_only");
  auto res = save_manifest_atomic(dir, make_sample_manifest());
  REQUIRE(res.has_value());

  auto p_final = dir / "wal.manifest";
  auto p_tmp = dir / "wal.manifest.tmp";
  REQUIRE(fs::exists(p_final));
  REQUIRE_FALSE(fs::exists(p_tmp));
  {
    std::ifstream in(p_final);
    REQUIRE(in.good());
    std::string hdr; std::getline(in, hdr);
    REQUIRE(hdr == std::string("vesper-wal-manifest v1"));
    std::string l2; std::getline(in, l2);
    CAPTURE(l2);
  }

  auto loaded = load_manifest(dir);
  if (!loaded) { auto e = loaded.error(); INFO("err: " << e.message); }
  REQUIRE(loaded.has_value());
  auto m = loaded.value();
  REQUIRE(m.entries.size() == 2);
  REQUIRE(m.entries[0].file == "wal-00000001.log");
  REQUIRE(m.entries[1].file == "wal-00000002.log");

  cleanup_dir(dir);
}

TEST_CASE("wal_manifest_atomic: manifest save succeeds when tmp file is missing", "[wal_manifest_atomic]") {
  auto dir = make_test_dir("tmp_missing");
  auto p_tmp = dir / "wal.manifest.tmp";
  std::error_code ec; fs::remove(p_tmp, ec);

  auto res = save_manifest_atomic(dir, make_sample_manifest());
  REQUIRE(res.has_value());

  auto p_final = dir / "wal.manifest";
  REQUIRE(fs::exists(p_final));
  {
    std::ifstream in(p_final);
    REQUIRE(in.good());
    std::string hdr; std::getline(in, hdr);
    REQUIRE(hdr == std::string("vesper-wal-manifest v1"));
    std::string l2; std::getline(in, l2);
    CAPTURE(l2);
  }

  auto loaded = load_manifest(dir);
  if (!loaded) { auto e = loaded.error(); INFO("err: " << e.message); }
  REQUIRE(loaded.has_value());
  REQUIRE(loaded->entries.size() == 2);

  cleanup_dir(dir);
}

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
TEST_CASE("wal_manifest_atomic: manifest save falls back to unique tmp name when default tmp is locked", "[wal_manifest_atomic][windows]") {
  auto dir = make_test_dir("win_locked_tmp");
  auto p_tmp = dir / L"wal.manifest.tmp";

  // Create and lock wal.manifest.tmp exclusively (no delete sharing) to force fallback path
  HANDLE h = ::CreateFileW(p_tmp.wstring().c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  REQUIRE(h != INVALID_HANDLE_VALUE);

  // Call save; it should fall back to unique tmp and still succeed
  auto res = save_manifest_atomic(dir, make_sample_manifest());
  REQUIRE(res.has_value());

  // Final manifest exists and is valid
  auto p_final = dir / "wal.manifest";
  REQUIRE(fs::exists(p_final));
  {
    std::ifstream in(p_final);
    REQUIRE(in.good());
    std::string hdr; std::getline(in, hdr);
    REQUIRE(hdr == std::string("vesper-wal-manifest v1"));
    std::string l2; std::getline(in, l2);
    CAPTURE(l2);
  }
  auto loaded = load_manifest(dir);
  if (!loaded) { auto e = loaded.error(); INFO("err: " << e.message); }
  REQUIRE(loaded.has_value());
  REQUIRE(loaded->entries.size() == 2);

  // The unique tmp (wal.manifest.tmp.<pid>.<tick>) should not remain; ensure no files with that prefix exist
  size_t prefixed = 0;
  for (const auto& de : fs::directory_iterator(dir)) {
    auto name = de.path().filename().wstring();
    if (name.rfind(L"wal.manifest.tmp.", 0) == 0) ++prefixed;
  }
  REQUIRE(prefixed == 0);

  // Close our lock handle; then cleanup dir
  ::CloseHandle(h);
  cleanup_dir(dir);
}
#endif

#if defined(__linux__) || defined(__APPLE__)
TEST_CASE("wal_manifest_atomic: manifest save succeeds when stale tmp exists", "[wal_manifest_atomic][posix]") {
  auto dir = make_test_dir("posix_stale_tmp");
  auto p_tmp = dir / "wal.manifest.tmp";
  {
    std::ofstream out(p_tmp, std::ios::binary | std::ios::trunc);
    REQUIRE(out.good());
    out << "garbage\n";
  }

  auto res = save_manifest_atomic(dir, make_sample_manifest());
  REQUIRE(res.has_value());

  auto p_final = dir / "wal.manifest";
  REQUIRE(fs::exists(p_final));
  // Stale tmp should be cleaned up
  REQUIRE_FALSE(fs::exists(p_tmp));

  auto loaded = load_manifest(dir);
  REQUIRE(loaded.has_value());
  REQUIRE(loaded->entries.size() == 2);

  cleanup_dir(dir);
}
#endif

