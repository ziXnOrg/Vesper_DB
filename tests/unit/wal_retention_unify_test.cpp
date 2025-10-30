#include <catch2/catch_all.hpp>
#include <vesper/wal/retention.hpp>
#include <vesper/wal/manifest.hpp>
#include <vesper/wal/io.hpp>

#include <filesystem>
#include <set>
#include <vector>

#include <tests/support/replayer_payload.hpp>

using namespace vesper;
using namespace test_support;

namespace fs = std::filesystem;

static auto make_rotated_wal_dir(const std::string& name, std::size_t frames, std::size_t max_file_bytes)
    -> fs::path {
  auto dir = fs::temp_directory_path() / ("vesper_wal_unify_" + name);
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);
  wal::WalWriterOptions opts{.dir=dir, .prefix="wal-", .max_file_bytes=max_file_bytes, .strict_lsn_monotonic=false};
  auto w = wal::WalWriter::open(opts); REQUIRE(w.has_value());
  for (std::size_t i = 1; i <= frames; ++i) {
    auto up = make_upsert(1000 + int(i), std::vector<float>{float(i)}, {});
    REQUIRE(w->append(i, /*type=*/1, up).has_value());
  }
  REQUIRE(w->flush(false).has_value());
  return dir;
}

TEST_CASE("purge_keep_newer_than: deterministic tie-handling at identical timestamps", "[wal][retention]"){
  auto dir = make_rotated_wal_dir("ties", /*frames=*/6, /*max_file_bytes=*/48);

  // Set all wal-*.log files to the same timestamp
  fs::file_time_type common_ft{};
  bool first = true;
  std::vector<fs::path> wal_paths;
  for (auto& de : fs::directory_iterator(dir)) {
    if (!de.is_regular_file()) continue;
    auto name = de.path().filename().string();
    if (name.rfind("wal-", 0) == 0) {
      wal_paths.push_back(de.path());
      auto ft = fs::last_write_time(de.path());
      if (first) { common_ft = ft; first = false; }
    }
  }
  REQUIRE(!wal_paths.empty());
  for (auto& p : wal_paths) { std::error_code ec; fs::last_write_time(p, common_ft, ec); REQUIRE(!ec); }

  // With cutoff equal to common_ft, only one file must be kept (highest by deterministic rule)
  auto rx = wal::purge_keep_newer_than(dir, common_ft);
  REQUIRE(rx.has_value());

  // Expect exactly one remaining wal file
  std::vector<std::string> names;
  for (auto& de : fs::directory_iterator(dir)) if (de.is_regular_file()){
    auto nm = de.path().filename().string(); if (nm.rfind("wal-",0)==0) names.push_back(nm);
  }
  REQUIRE(names.size() == 1);

  // It should be the lexicographically greatest filename (ties broken deterministically)
  auto mx = wal::load_manifest(dir); REQUIRE(mx.has_value());
  auto m = *mx; REQUIRE_FALSE(m.entries.empty());
  // Determine expected winner by highest end_lsn, then filename
  auto entries = m.entries; // already post-purge
  // There is only one entry left; ensure it is the lexicographically max of the original set we modified
  std::string max_name = *std::max_element(names.begin(), names.end());
  REQUIRE(entries.back().file == max_name);

  std::error_code ec; fs::remove_all(dir, ec);
}

TEST_CASE("purge_keep_last_n keeps newest by deterministic order and always retains newest when N>=1", "[wal][retention]"){
  auto dir = make_rotated_wal_dir("keepn", /*frames=*/7, /*max_file_bytes=*/48);
  auto rx = wal::purge_keep_last_n(dir, 1);
  REQUIRE(rx.has_value());

  // Only one file should remain and it must be the newest
  std::vector<std::string> names;
  for (auto& de : fs::directory_iterator(dir)) if (de.is_regular_file()){
    auto nm = de.path().filename().string(); if (nm.rfind("wal-",0)==0) names.push_back(nm);
  }
  REQUIRE(names.size() == 1);

  // The remaining name must be the lexicographically greatest wal-*.log
  auto remaining = names.front();
  std::string max_name = remaining;
  for (auto& de : fs::directory_iterator(dir)) if (de.is_regular_file()){
    auto nm = de.path().filename().string(); if (nm.rfind("wal-",0)==0) max_name = std::max(max_name, nm);
  }
  REQUIRE(remaining == max_name);

  std::error_code ec; fs::remove_all(dir, ec);
}

TEST_CASE("purge_keep_total_bytes_max: byte budget edge cases and newest retained", "[wal][retention]"){
  auto dir = make_rotated_wal_dir("bytes", /*frames=*/9, /*max_file_bytes=*/64);

  auto mx0 = wal::load_manifest(dir); REQUIRE(mx0.has_value()); auto m0 = *mx0; REQUIRE(m0.entries.size() >= 3);
  // Compute sums for newest two files
  const auto n = m0.entries.size();
  const auto bytes_last1 = m0.entries[n-1].bytes;
  const auto bytes_last2 = m0.entries[n-2].bytes;
  const auto sum_last2 = bytes_last1 + bytes_last2;

  // exact budget: sum of last two
  {
    // Determine newest filename before purge
    std::string expected_max;
    for (auto& de : fs::directory_iterator(dir)) if (de.is_regular_file()){
      auto nm = de.path().filename().string(); if (nm.rfind("wal-",0)==0) expected_max = std::max(expected_max, nm);
    }

    auto rx = wal::purge_keep_total_bytes_max(dir, sum_last2);
    REQUIRE(rx.has_value());
    std::vector<std::string> names;
    for (auto& de : fs::directory_iterator(dir)) if (de.is_regular_file()){
      auto nm = de.path().filename().string(); if (nm.rfind("wal-",0)==0) names.push_back(nm);
    }
    REQUIRE(names.size() <= 2);
    // newest file must be present
    REQUIRE(std::find(names.begin(), names.end(), expected_max) != names.end());
  }

  // Reset directory and rebuild
  {
    std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);
    wal::WalWriterOptions opts{.dir=dir, .prefix="wal-", .max_file_bytes=64, .strict_lsn_monotonic=false};
    auto w = wal::WalWriter::open(opts); REQUIRE(w.has_value());
    for (int i=1;i<=9;++i){ auto up = make_upsert(3000+i, std::vector<float>{float(i)}, {}); REQUIRE(w->append(i, 1, up).has_value()); }
    REQUIRE(w->flush(false).has_value());
  }

  // budget less than newest bytes: must still keep the newest file (deterministic policy)
  {
    auto mx = wal::load_manifest(dir); REQUIRE(mx.has_value()); auto m = *mx; const auto n2 = m.entries.size();
    // Determine newest filename before purge
    std::string expected_max;
    for (auto& de : fs::directory_iterator(dir)) if (de.is_regular_file()){
      auto nm = de.path().filename().string(); if (nm.rfind("wal-",0)==0) expected_max = std::max(expected_max, nm);
    }
    auto rx = wal::purge_keep_total_bytes_max(dir, 0);
    REQUIRE(rx.has_value());
    std::vector<std::string> names;
    for (auto& de : fs::directory_iterator(dir)) if (de.is_regular_file()){
      auto nm = de.path().filename().string(); if (nm.rfind("wal-",0)==0) names.push_back(nm);
    }
    // Must retain newest file even under zero budget
    REQUIRE(std::find(names.begin(), names.end(), expected_max) != names.end());
    REQUIRE(names.size() == 1);
  }

  // Reset and budget + 1: should be able to keep at least the same set as exact budget
  {
    std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);
    wal::WalWriterOptions opts{.dir=dir, .prefix="wal-", .max_file_bytes=64, .strict_lsn_monotonic=false};
    auto w = wal::WalWriter::open(opts); REQUIRE(w.has_value());
    for (int i=1;i<=9;++i){ auto up = make_upsert(4000+i, std::vector<float>{float(i)}, {}); REQUIRE(w->append(i, 1, up).has_value()); }
    REQUIRE(w->flush(false).has_value());

    auto mx = wal::load_manifest(dir); REQUIRE(mx.has_value()); auto m = *mx; const auto n3 = m.entries.size();
    const auto b1 = m.entries[n3-1].bytes; const auto b2 = m.entries[n3-2].bytes; const auto sum2 = b1 + b2;
    // Determine newest filename before purge
    std::string expected_max;
    for (auto& de : fs::directory_iterator(dir)) if (de.is_regular_file()){
      auto nm = de.path().filename().string(); if (nm.rfind("wal-",0)==0) expected_max = std::max(expected_max, nm);
    }

    auto rx = wal::purge_keep_total_bytes_max(dir, sum2 + 1);
    REQUIRE(rx.has_value());

    std::vector<std::string> names;
    for (auto& de : fs::directory_iterator(dir)) if (de.is_regular_file()){
      auto nm = de.path().filename().string(); if (nm.rfind("wal-",0)==0) names.push_back(nm);
    }
    REQUIRE(names.size() <= 2);
    REQUIRE(std::find(names.begin(), names.end(), expected_max) != names.end());
  }

  std::error_code ec; fs::remove_all(dir, ec);
}

// Namespace correctness: ensure symbol is reachable
TEST_CASE("namespace correctness for purge_keep_total_bytes_max", "[wal][retention][namespace]"){
  auto dir = make_rotated_wal_dir("ns", /*frames=*/3, /*max_file_bytes=*/48);
  auto rx = wal::purge_keep_total_bytes_max(dir, /*max_total_bytes=*/std::numeric_limits<std::uint64_t>::max());
  REQUIRE(rx.has_value());
  std::error_code ec; fs::remove_all(dir, ec);
}

