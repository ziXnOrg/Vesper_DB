#include <catch2/catch_all.hpp>
#include <vesper/wal/io.hpp>
#include <filesystem>
#include <fstream>
#include <vector>

#include <tests/support/replayer_payload.hpp>

using namespace vesper;
using namespace test_support;
using vesper::core::error_code;


namespace {
namespace fs = std::filesystem;

static std::vector<std::pair<std::uint64_t, fs::path>> wal_files_sorted(const fs::path& dir){
  std::vector<std::pair<std::uint64_t, fs::path>> v;
  for (auto& de : fs::directory_iterator(dir)){
    if (!de.is_regular_file()) continue;
    auto name = de.path().filename().string();
    if (name.rfind("wal-", 0)==0 && name.size() >= 4+8) {
      try { auto seq = static_cast<std::uint64_t>(std::stoull(name.substr(4,8))); v.emplace_back(seq, de.path()); } catch(...) {}
    }
  }
  std::sort(v.begin(), v.end(), [](auto& a, auto& b){ return a.first < b.first; });
  return v;
}

static void build_three_frames_two_files(const fs::path& dir){
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);
  wal::WalWriterOptions opts{.dir=dir, .prefix="wal-", .max_file_bytes=48, .strict_lsn_monotonic=false};
  auto w = wal::WalWriter::open(opts); REQUIRE(w.has_value());
  auto up101 = make_upsert(101, std::vector<float>{1.0f}, {});
  auto up102 = make_upsert(102, std::vector<float>{2.0f}, {});
  auto up103 = make_upsert(103, std::vector<float>{3.0f}, {});
  REQUIRE(w->append(1, /*type=*/1, up101).has_value());
  REQUIRE(w->append(2, /*type=*/1, up102).has_value());
  REQUIRE(w->append(3, /*type=*/1, up103).has_value());
  REQUIRE(w->flush(false).has_value());
  // Ensure at least two files
  REQUIRE(wal_files_sorted(dir).size() >= 2);
}

}

TEST_CASE("recover_scan_dir propagates integrity and I/O errors deterministically", "[wal][replay][errors]"){
  namespace fs = std::filesystem;
  auto dir = fs::temp_directory_path() / "vesper_wal_scan_errors";
  build_three_frames_two_files(dir);

  SECTION("corruption in non-last file returns data_integrity"){
    auto files = wal_files_sorted(dir);
    REQUIRE(files.size() >= 2);
    auto corrupt = files.front().second;
    // Truncate the older file by 1 byte (non-last): should trigger data_integrity
    std::error_code ec; auto sz = fs::file_size(corrupt, ec); REQUIRE(!ec); REQUIRE(sz>0);
    fs::resize_file(corrupt, sz-1, ec); REQUIRE(!ec);

    std::size_t delivered=0;
    auto st = wal::recover_scan_dir(dir, [&](const wal::WalFrame& f){ delivered += f.payload.size(); });
    REQUIRE_FALSE(st.has_value());
    REQUIRE(st.error().code == error_code::data_integrity);
    // Delivered might include frames from the first intact file or none, but must not process beyond the error point deterministically.
  }

  SECTION("missing wal file yields io error"){
    auto files = wal_files_sorted(dir);
    REQUIRE(files.size() >= 2);
    auto missing = files.front().second;
    // Rename the first file so manifest (if present) points to a missing file
    auto backup = missing; backup += ".bak";
    std::error_code ec; fs::rename(missing, backup, ec); REQUIRE(!ec);

    auto st = wal::recover_scan_dir(dir, [&](const wal::WalFrame&){ /*noop*/ });
    REQUIRE_FALSE(st.has_value());
    REQUIRE(st.error().code == error_code::io_failed);

    // Restore to not affect other sections
    fs::rename(backup, missing, ec);
  }

  SECTION("torn tail on last file is tolerated"){
    auto files = wal_files_sorted(dir);
    REQUIRE(files.size() >= 2);
    auto last = files.back().second;
    std::error_code ec; auto sz = fs::file_size(last, ec); REQUIRE(!ec); REQUIRE(sz>0);
    fs::resize_file(last, sz-1, ec); REQUIRE(!ec);

    std::size_t frames=0; std::uint64_t last_lsn=0; bool lsn_mono=true; std::size_t upserts=0;
    auto st = wal::recover_scan_dir(dir, [&](const wal::WalFrame& f){ frames++; last_lsn=f.lsn; if (f.type==1) upserts++; });
    REQUIRE(st.has_value());
    auto s = *st;
    REQUIRE(s.frames == frames);
    REQUIRE(s.last_lsn == last_lsn);
    REQUIRE(s.type_counts[1] == upserts);
    REQUIRE(s.lsn_monotonic == true);
  }

  std::error_code ec; fs::remove_all(dir, ec);
}

