#include <catch2/catch_all.hpp>
#include <vesper/wal/io.hpp>
#include <vesper/wal/manifest.hpp>
#include <vesper/wal/snapshot.hpp>
#include <filesystem>

using namespace vesper;

TEST_CASE("snapshot skip happy path", "[wal][snapshot]") {
  namespace fs = std::filesystem;
  auto dir = fs::temp_directory_path() / "vesper_wal_snap_ok";
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);
  wal::WalWriterOptions opts{.dir=dir, .prefix="wal-", .max_file_bytes=64};
  auto w = wal::WalWriter::open(opts); REQUIRE(w.has_value());
  std::vector<std::uint8_t> p{0xAA};
  const int N = 6;
  for (int i=1;i<=N;++i) REQUIRE(w->append(i, 1, p).has_value());
  REQUIRE(w->flush(false).has_value());
  // Snapshot at M
  wal::Snapshot s{.last_lsn=3}; REQUIRE(wal::save_snapshot(dir, s).has_value());

  std::size_t total_payload = 0;
  auto stats_exp = wal::recover_scan_dir(dir, [&](const wal::WalFrame& f){ total_payload += f.payload.size(); });
  REQUIRE(stats_exp.has_value());
  auto st = *stats_exp;
  REQUIRE(st.frames == static_cast<std::size_t>(N - s.last_lsn));
  REQUIRE(st.last_lsn == static_cast<std::uint64_t>(N));
  REQUIRE(st.type_counts[1] == static_cast<std::uint64_t>(N - s.last_lsn));
  REQUIRE(total_payload == static_cast<std::size_t>(N - s.last_lsn));
  fs::remove_all(dir, ec);
}

TEST_CASE("snapshot yields zero results when at or beyond last", "[wal][snapshot]") {
  namespace fs = std::filesystem;
  auto dir = fs::temp_directory_path() / "vesper_wal_snap_zero";
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);
  wal::WalWriterOptions opts{.dir=dir, .prefix="wal-", .max_file_bytes=64};
  auto w = wal::WalWriter::open(opts); REQUIRE(w.has_value());
  std::vector<std::uint8_t> p{0xBB};
  REQUIRE(w->append(1, 1, p).has_value());
  REQUIRE(w->append(2, 1, p).has_value());
  REQUIRE(w->flush(false).has_value());
  wal::Snapshot s{.last_lsn=2}; REQUIRE(wal::save_snapshot(dir, s).has_value());
  auto stats_exp = wal::recover_scan_dir(dir, [&](const wal::WalFrame&){ /*noop*/ });
  REQUIRE(stats_exp.has_value());
  REQUIRE(stats_exp->frames == 0);
  fs::remove_all(dir, ec);
}

TEST_CASE("torn tail on last remains allowed with snapshot", "[wal][snapshot]") {
  namespace fs = std::filesystem;
  auto dir = fs::temp_directory_path() / "vesper_wal_snap_torn";
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);
  wal::WalWriterOptions opts{.dir=dir, .prefix="wal-", .max_file_bytes=64};
  auto w = wal::WalWriter::open(opts); REQUIRE(w.has_value());
  std::vector<std::uint8_t> p{0xCC};
  REQUIRE(w->append(1, 1, p).has_value());
  REQUIRE(w->append(2, 1, p).has_value());
  REQUIRE(w->flush(false).has_value());
  // New file with one frame then truncate a byte
  REQUIRE(w->append(3, 1, p).has_value());
  REQUIRE(w->flush(false).has_value());
  // Truncate last file
  std::filesystem::path last; std::uint64_t max_seq=0;
  for (auto& de : fs::directory_iterator(dir)){
    if (!de.is_regular_file()) continue; auto name = de.path().filename().string();
    if (name.rfind("wal-", 0)==0) { auto seq = std::stoull(name.substr(4,8)); if (seq>max_seq){ max_seq=seq; last=de.path(); } }
  }
  REQUIRE(!last.empty()); auto sz = fs::file_size(last); fs::resize_file(last, sz-1, ec); REQUIRE(!ec);
  wal::Snapshot s{.last_lsn=1}; REQUIRE(wal::save_snapshot(dir, s).has_value());
  auto stats_exp = wal::recover_scan_dir(dir, [&](const wal::WalFrame&){ /*noop*/ });
  REQUIRE(stats_exp.has_value());
  fs::remove_all(dir, ec);
}

TEST_CASE("corruption in non-last file fails despite snapshot", "[wal][snapshot]") {
  namespace fs = std::filesystem;
  auto dir = fs::temp_directory_path() / "vesper_wal_snap_corrupt";
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);
  wal::WalWriterOptions opts{.dir=dir, .prefix="wal-", .max_file_bytes=64};
  auto w = wal::WalWriter::open(opts); REQUIRE(w.has_value());
  std::vector<std::uint8_t> p{0xDD};
  REQUIRE(w->append(1, 1, p).has_value());
  REQUIRE(w->append(2, 1, p).has_value());
  REQUIRE(w->flush(false).has_value());
  // Write more to ensure at least two files
  REQUIRE(w->append(3, 1, p).has_value());
  REQUIRE(w->append(4, 1, p).has_value());
  REQUIRE(w->flush(false).has_value());

  // Corrupt the first file
  std::filesystem::path first; std::uint64_t min_seq=~0ull;
  for (auto& de : fs::directory_iterator(dir)){
    if (!de.is_regular_file()) continue; auto name = de.path().filename().string();
    if (name.rfind("wal-", 0)==0) { auto seq = std::stoull(name.substr(4,8)); if (seq<min_seq){ min_seq=seq; first=de.path(); } }
  }
  REQUIRE(!first.empty()); auto sz = fs::file_size(first); REQUIRE(sz>10); // ensure not empty
  // Flip a byte in the middle
  {
    std::fstream f(first, std::ios::binary | std::ios::in | std::ios::out);
    REQUIRE(f.good()); f.seekp(10); char c=0; f.read(&c,1); c^=0xFF; f.seekp(10); f.write(&c,1); f.flush();
  }

  wal::Snapshot s{.last_lsn=2}; REQUIRE(wal::save_snapshot(dir, s).has_value());
  auto stats_exp = wal::recover_scan_dir(dir, [&](const wal::WalFrame&){ /*noop*/ });
  REQUIRE_FALSE(stats_exp.has_value());
  REQUIRE(stats_exp.error().code == vesper::core::error_code::data_integrity);
  fs::remove_all(dir, ec);
}

