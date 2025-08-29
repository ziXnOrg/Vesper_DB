#include <catch2/catch_all.hpp>
#include <vesper/wal/io.hpp>
#include <vesper/wal/retention.hpp>
#include <filesystem>

#include <tests/support/replayer_payload.hpp>

using namespace vesper;
using namespace test_support;

TEST_CASE("purge_keep_last_n keeps only newest N rotated files", "[wal][retention][purge][boundaries]"){
  namespace fs = std::filesystem;
  auto dir = fs::temp_directory_path() / "vesper_wal_keep_last_n";
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);

  wal::WalWriterOptions opts{.dir=dir, .prefix="wal-", .max_file_bytes=48, .strict_lsn_monotonic=false};
  auto w = wal::WalWriter::open(opts); REQUIRE(w.has_value());
  for (int i=1;i<=6;++i){ auto up = make_upsert(100+i, std::vector<float>{float(i)}, {}); REQUIRE(w->append(i, 1, up).has_value()); }
  REQUIRE(w->flush(false).has_value());

  auto mx = wal::purge_keep_last_n(dir, 2);
  REQUIRE(mx.has_value());

  // Only last two files should remain by sequence
  std::size_t count=0; for (auto& de: fs::directory_iterator(dir)) if (de.is_regular_file() && de.path().filename().string().rfind("wal-",0)==0) count++;
  REQUIRE(count == 2);

  fs::remove_all(dir, ec);
}

TEST_CASE("purge_keep_newer_than respects provided cutoff time", "[wal][retention][purge]"){
  namespace fs = std::filesystem;
  auto dir = fs::temp_directory_path() / "vesper_wal_keep_time";
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);

  wal::WalWriterOptions opts{.dir=dir, .prefix="wal-", .max_file_bytes=48, .strict_lsn_monotonic=false};
  auto w = wal::WalWriter::open(opts); REQUIRE(w.has_value());
  for (int i=1;i<=4;++i){ auto up = make_upsert(200+i, std::vector<float>{float(i)}, {}); REQUIRE(w->append(i, 1, up).has_value()); }
  REQUIRE(w->flush(false).has_value());

  // Compute a cutoff equal to the last_write_time of the newest file
  std::filesystem::file_time_type max_ft{}; std::filesystem::path maxp;
  for (auto& de : fs::directory_iterator(dir)){
    if (!de.is_regular_file()) continue; auto name=de.path().filename().string(); if(name.rfind("wal-",0)==0){ auto ft = fs::last_write_time(de.path()); if(ft>max_ft){ max_ft=ft; maxp=de.path(); }}
  }
  REQUIRE(!maxp.empty());

  // Keep files new enough (>= newest) -> only newest survives
  auto rx = wal::purge_keep_newer_than(dir, max_ft);
  REQUIRE(rx.has_value());

  std::size_t count=0; for (auto& de: fs::directory_iterator(dir)) if (de.is_regular_file() && de.path().filename().string().rfind("wal-",0)==0) count++;
  REQUIRE(count == 1);

  fs::remove_all(dir, ec);
}



TEST_CASE("retention and snapshot interplay preserves replay semantics", "[wal][retention][purge][snapshot][manifest][replay]"){
  namespace fs = std::filesystem;
  auto dir = fs::temp_directory_path() / "vesper_wal_keep_snapshot";
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);

  wal::WalWriterOptions opts{.dir=dir, .prefix="wal-", .max_file_bytes=48, .strict_lsn_monotonic=false};
  auto w = wal::WalWriter::open(opts); REQUIRE(w.has_value());
  // Frames across >=3 files
  for (int i=1;i<=6;++i){ auto up = make_upsert(300+i, std::vector<float>{float(i)}, {}); REQUIRE(w->append(i, 1, up).has_value()); }
  REQUIRE(w->flush(false).has_value());

  // Case 1: cutoff1=3 then keep last 2 files -> replay should deliver lsns {4,5,6}
  {
    REQUIRE(wal::save_snapshot(dir, wal::Snapshot{.last_lsn = 3}).has_value());
    REQUIRE(wal::purge_keep_last_n(dir, 2).has_value());
    std::vector<std::uint64_t> lsns; auto st = wal::recover_scan_dir(dir, [&](const wal::WalFrame& f){ lsns.push_back(f.lsn); });
    REQUIRE(st.has_value());
    REQUIRE(lsns.size() == 3);
    REQUIRE(lsns.front() == 4);
    REQUIRE(lsns.back() == 6);
    REQUIRE(st->last_lsn == 6);
    REQUIRE(st->lsn_monotonic == true);
    // ToyIndex replay check
    ToyIndex idx = test_support::build_toy_index_baseline_then_replay(dir, 3);
    REQUIRE(idx.count(301) == 1);
    REQUIRE(idx.count(302) == 1);
    REQUIRE(idx.count(303) == 1);
  }

  // Reset and repeat for cutoff2=5 then keep only newest file by time -> only lsn 6 should be delivered
  {
    // Rebuild directory
    fs::remove_all(dir, ec); fs::create_directories(dir, ec);
    auto w2 = wal::WalWriter::open(opts); REQUIRE(w2.has_value());
    for (int i=1;i<=6;++i){ auto up = make_upsert(400+i, std::vector<float>{float(i)}, {}); REQUIRE(w2->append(i, 1, up).has_value()); }
    REQUIRE(w2->flush(false).has_value());

    REQUIRE(wal::save_snapshot(dir, wal::Snapshot{.last_lsn = 5}).has_value());

    // Determine newest file's timestamp to keep only that file
    std::filesystem::file_time_type max_ft{}; for (auto& de : fs::directory_iterator(dir)){
      if (!de.is_regular_file()) continue; auto name=de.path().filename().string(); if(name.rfind("wal-",0)==0){ auto ft = fs::last_write_time(de.path()); if(ft>max_ft) max_ft=ft; }
    }
    REQUIRE(wal::purge_keep_newer_than(dir, max_ft).has_value());

    std::vector<std::uint64_t> lsns; auto st = wal::recover_scan_dir(dir, [&](const wal::WalFrame& f){ lsns.push_back(f.lsn); });
    REQUIRE(st.has_value());
    REQUIRE(lsns.size() == 1);
    REQUIRE(lsns.front() == 6);
    REQUIRE(st->last_lsn == 6);

    ToyIndex idx = test_support::build_toy_index_baseline_then_replay(dir, 5);
    REQUIRE(idx.count(406) == 1);
  }

  fs::remove_all(dir, ec);
}
