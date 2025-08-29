#include <catch2/catch_all.hpp>
#include <vesper/wal/io.hpp>
#include <filesystem>
#include <chrono>

using namespace vesper;

namespace {
namespace fs = std::filesystem;

static std::size_t count_wal_files(const fs::path& dir){
  std::size_t c = 0;
  for (auto& de : fs::directory_iterator(dir)){
    if (!de.is_regular_file()) continue;
    auto name = de.path().filename().string();
    if (name.rfind("wal-", 0) == 0 && name.size() >= 4+8) {
      // Best-effort parse; accept if numeric
      try { (void)std::stoull(name.substr(4,8)); ++c; } catch (...) {}
    }
  }
  return c;
}

static void make_dir_with_rotated_files(const fs::path& dir, std::size_t target_files){
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);
  wal::WalWriterOptions opts{.dir=dir, .prefix="wal-", .max_file_bytes=64, .strict_lsn_monotonic=false};
  auto w = wal::WalWriter::open(opts); REQUIRE(w.has_value());
  std::vector<std::uint8_t> payload{0xAB};
  std::uint64_t lsn = 1;
  while (count_wal_files(dir) < target_files) {
    REQUIRE(w->append(lsn++, /*type=*/1, payload).has_value());
  }
  REQUIRE(w->flush(false).has_value());
  REQUIRE(count_wal_files(dir) >= target_files);
}

static std::chrono::duration<double> time_scan_dir(const fs::path& dir){
  auto t0 = std::chrono::steady_clock::now();
  auto stats = wal::recover_scan_dir(dir, [&](const wal::WalFrame&){ /*noop*/ });
  auto t1 = std::chrono::steady_clock::now();
  REQUIRE(stats.has_value());
  return t1 - t0;
}
}

TEST_CASE("recover_scan_dir scales ~linearly on small dirs", "[wal][manifest][perf]"){
  const std::size_t N = 32; // small, deterministic
  const fs::path dirN  = fs::temp_directory_path() / "vesper_wal_perf_N";
  const fs::path dir2N = fs::temp_directory_path() / "vesper_wal_perf_2N";

  make_dir_with_rotated_files(dirN,  N);
  make_dir_with_rotated_files(dir2N, 2*N);

  // Warm-up to avoid first-run overhead skewing ratio
  (void)time_scan_dir(dirN);
  (void)time_scan_dir(dir2N);

  auto tN  = time_scan_dir(dirN);
  auto t2N = time_scan_dir(dir2N);

  // Ratio sanity: allow generous factor to accommodate runner variance
  double ratio = t2N.count() / std::max(1e-9, tN.count());
  REQUIRE(ratio <= 3.0);

  // Optional absolute bounds (keep generous; can remove if flaky in CI)
  REQUIRE(tN.count()  < 1.0);
  REQUIRE(t2N.count() < 1.0);

  std::error_code ec; fs::remove_all(dirN, ec); fs::remove_all(dir2N, ec);
}

