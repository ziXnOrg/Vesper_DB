#include <catch2/catch_all.hpp>
#include <vesper/wal/io.hpp>
#include <vesper/wal/retention.hpp>
#include <filesystem>

#include <tests/support/replayer_payload.hpp>

using namespace vesper;
using namespace test_support;

TEST_CASE("purge_keep_total_bytes_max keeps newest files within byte budget", "[wal][retention][purge][boundaries]"){
  namespace fs = std::filesystem;
  auto dir = fs::temp_directory_path() / "vesper_wal_keep_bytes";
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);

  wal::WalWriterOptions opts{.dir=dir, .prefix="wal-", .max_file_bytes=64, .strict_lsn_monotonic=false};
  auto w = wal::WalWriter::open(opts); REQUIRE(w.has_value());
  // Create enough frames across multiple files; small byte sizes
  for (int i=1;i<=8;++i){ auto up = make_upsert(500+i, std::vector<float>{float(i)}, {}); REQUIRE(w->append(i, 1, up).has_value()); }
  REQUIRE(w->flush(false).has_value());

  // Compute manifest and bytes of last two files
  auto mx = wal::load_manifest(dir); REQUIRE(mx.has_value()); auto m = *mx; REQUIRE(m.entries.size() >= 3);
  auto last_bytes = m.entries[m.entries.size()-1].bytes + m.entries[m.entries.size()-2].bytes;

  // Purge to keep at most last two files' bytes
  REQUIRE(wal::purge_keep_total_bytes_max(dir, last_bytes).has_value());

  std::size_t count=0; std::vector<std::string> names;
  for (auto& de: fs::directory_iterator(dir)) if (de.is_regular_file()){
    auto name = de.path().filename().string(); if (name.rfind("wal-",0)==0) { count++; names.push_back(name); }
  }
  REQUIRE(count <= 2);

  fs::remove_all(dir, ec);
}

