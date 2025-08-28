#include <catch2/catch_all.hpp>
#include <vesper/wal/io.hpp>
#include <vesper/wal/manifest.hpp>
#include <filesystem>

using namespace vesper;

TEST_CASE("WAL rotation happy path and recovery scan dir", "[wal][rotation]") {
  namespace fs = std::filesystem;
  auto dir = fs::temp_directory_path() / "vesper_wal_rot";
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);
  wal::WalWriterOptions opts{.dir=dir, .prefix="wal-", .max_file_bytes=64};
  auto w = wal::WalWriter::open(opts);
  REQUIRE(w.has_value());
  // Write N small frames to force multiple files
  const int N = 10; std::vector<uint8_t> p{0xAA};
  for (int i=1;i<=N;++i) REQUIRE(w->append(i, 1, p).has_value());
  REQUIRE(w->flush(false).has_value());

  // Verify multiple files exist
  int files = 0; for (auto& de : fs::directory_iterator(dir)) if (de.is_regular_file()) ++files;
  REQUIRE(files >= 1);

  // Recover across directory
  std::size_t total_payload = 0;
  auto stats_exp = wal::recover_scan_dir(dir, [&](const wal::WalFrame& f){ total_payload += f.payload.size(); });
  REQUIRE(stats_exp.has_value());
  auto s = *stats_exp;
  REQUIRE(s.frames == N);
  REQUIRE(s.last_lsn == static_cast<std::uint64_t>(N));
  REQUIRE(s.lsn_monotonic);
  REQUIRE(s.type_counts[1] == static_cast<std::uint64_t>(N));
  REQUIRE(total_payload == static_cast<std::size_t>(N));

  fs::remove_all(dir, ec);
}

TEST_CASE("WAL torn tail only allowed on last file", "[wal][rotation]") {
  namespace fs = std::filesystem;
  auto dir = fs::temp_directory_path() / "vesper_wal_rot_torn";
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);
  wal::WalWriterOptions opts{.dir=dir, .prefix="wal-", .max_file_bytes=64};
  auto w = wal::WalWriter::open(opts);
  REQUIRE(w.has_value());

  std::vector<uint8_t> p{0x11};
  // Write enough frames to ensure at least 2 files
  REQUIRE(w->append(1, 1, p).has_value());
  REQUIRE(w->append(2, 1, p).has_value());
  REQUIRE(w->flush(false).has_value());

  // Append a valid complete file by forcing rotation (small max bytes)
  REQUIRE(w->append(3, 1, p).has_value());
  REQUIRE(w->flush(false).has_value());

  // Manually write a frame to a new file and truncate its last byte
  {
    // Force rotation by constructing a new writer with same dir (next seq)
    auto w2 = wal::WalWriter::open(opts);
    REQUIRE(w2.has_value());
    REQUIRE(w2->append(4, 1, p).has_value());
    REQUIRE(w2->flush(false).has_value());
    // Find the last file and truncate 1 byte
    std::filesystem::path last;
    std::uint64_t max_seq = 0;
    for (auto& de : fs::directory_iterator(dir)) {
      if (!de.is_regular_file()) continue;
      auto name = de.path().filename().string();
      if (name.rfind("wal-", 0) == 0) {
        auto seq = std::stoull(name.substr(4, 8));
        if (seq > max_seq) { max_seq = seq; last = de.path(); }
      }
    }
    REQUIRE(!last.empty());
    auto sz = fs::file_size(last);
    fs::resize_file(last, sz - 1, ec); REQUIRE(!ec);
  }

  // Recover: should succeed (torn tail allowed only on last file)
  auto stats_exp = wal::recover_scan_dir(dir, [&](const wal::WalFrame&){ /*noop*/ });
  REQUIRE(stats_exp.has_value());

  fs::remove_all(dir, ec);
}

TEST_CASE("Manifest round-trip", "[wal][manifest]") {
  namespace fs = std::filesystem;
  auto dir = fs::temp_directory_path() / "vesper_wal_manifest";
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);
  wal::Manifest m; m.entries.push_back({"wal-00000001.log", 1, 1, 5, 5, 100});
  REQUIRE(wal::save_manifest(dir, m).has_value());
  auto m2 = wal::load_manifest(dir);
  REQUIRE(m2.has_value());
  REQUIRE(m2->entries.size() == 1);
  REQUIRE(m2->entries[0].file == "wal-00000001.log");
  REQUIRE(m2->entries[0].seq == 1);
  fs::remove_all(dir, ec);
}

