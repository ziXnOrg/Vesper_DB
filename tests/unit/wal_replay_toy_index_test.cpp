#include <catch2/catch_all.hpp>
#include <vesper/wal/io.hpp>
#include <vesper/wal/replay.hpp>
#include <filesystem>
#include <tests/support/replayer_payload.hpp>

using namespace vesper;
using namespace test_support;

using Catch::Approx;

static void append_upsert(wal::WalWriter& w, std::uint64_t lsn, std::uint64_t id, std::initializer_list<std::pair<std::string,std::string>> tags, std::initializer_list<float> vec){
  std::vector<float> v(vec); auto payload = make_upsert(id, v, tags); REQUIRE(w.append(lsn, 1, payload).has_value()); }
static void append_delete(wal::WalWriter& w, std::uint64_t lsn, std::uint64_t id){ auto payload = make_delete(id); REQUIRE(w.append(lsn, 2, payload).has_value()); }

TEST_CASE("toy replayer happy path with rotation", "[wal][replay][toy]"){
  namespace fs = std::filesystem;
  auto dir = fs::temp_directory_path() / "vesper_toy_replay_rot";
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);
  wal::WalWriterOptions opts{.dir=dir, .prefix="wal-", .max_file_bytes=64};
  auto w = wal::WalWriter::open(opts); REQUIRE(w.has_value());

  append_upsert(*w, 1, 100, {{"color","red"}}, {1.0f,2.0f});
  append_upsert(*w, 2, 101, {{"shape","circle"}}, {0.0f,1.0f});
  append_delete(*w, 3, 100);
  append_upsert(*w, 4, 101, {{"shape","square"}}, {5.0f,6.0f}); // update
  REQUIRE(w->flush(false).has_value());

  ToyIndex idx;
  auto st = wal::recover_replay(dir, [&](std::uint64_t, std::uint16_t, std::span<const std::uint8_t> pl){ apply_frame_payload(pl, idx); });
  REQUIRE(st.has_value());

  REQUIRE(idx.count(100) == 0);
  REQUIRE(idx.count(101) == 1);
  auto& d = idx.at(101);
  REQUIRE(d.vec.size() == 2);
  REQUIRE(d.vec[0] == Approx(5.0f));
  REQUIRE(d.tags.at("shape") == "square");

  fs::remove_all(dir, ec);
}

TEST_CASE("toy replayer with snapshot cutoff", "[wal][replay][toy]"){
  namespace fs = std::filesystem;
  auto dir = fs::temp_directory_path() / "vesper_toy_replay_snap";
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);
  wal::WalWriterOptions opts{.dir=dir, .prefix="wal-", .max_file_bytes=64};
  auto w = wal::WalWriter::open(opts); REQUIRE(w.has_value());

  append_upsert(*w, 1, 200, {{"a","1"}}, {1.0f});
  append_upsert(*w, 2, 201, {{"b","2"}}, {2.0f});
  append_upsert(*w, 3, 200, {{"a","3"}}, {3.0f});
  REQUIRE(w->flush(false).has_value());
  REQUIRE(w->publish_snapshot(2).has_value());

  ToyIndex idx;
  // Reconstruct baseline snapshot state by applying frames up to and including cutoff
  const std::uint64_t cutoff = 2;
  std::vector<std::pair<std::uint64_t, std::filesystem::path>> files;
  for (auto& de : fs::directory_iterator(dir)){
    if (!de.is_regular_file()) continue; auto name = de.path().filename().string();
    if (name.rfind("wal-", 0)==0) { auto seq = std::stoull(name.substr(4,8)); files.emplace_back(seq, de.path()); }
  }
  std::sort(files.begin(), files.end(), [](auto& a, auto& b){ return a.first < b.first; });
  for (auto& kv : files){
    auto st0 = wal::recover_scan(kv.second.string(), [&](const wal::WalFrame& f){ if (f.lsn <= cutoff) apply_frame_payload(f.payload, idx); });
    REQUIRE(st0.has_value());
  }
  // Now replay frames strictly after cutoff
  auto st = wal::recover_replay(dir, [&](std::uint64_t, std::uint16_t, std::span<const std::uint8_t> pl){ apply_frame_payload(pl, idx); });
  REQUIRE(st.has_value());
  REQUIRE(idx.count(201) == 1);
  REQUIRE(idx.count(200) == 1);
  REQUIRE(idx.at(200).vec[0] == Approx(3.0f));

  fs::remove_all(dir, ec);
}

TEST_CASE("toy replayer tolerates torn tail on last file", "[wal][replay][toy]"){
  namespace fs = std::filesystem;
  auto dir = fs::temp_directory_path() / "vesper_toy_replay_torn";
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);
  wal::WalWriterOptions opts{.dir=dir, .prefix="wal-", .max_file_bytes=64};
  auto w = wal::WalWriter::open(opts); REQUIRE(w.has_value());

  append_upsert(*w, 1, 300, {{"x","y"}}, {9.0f});
  append_upsert(*w, 2, 301, {{"y","z"}}, {8.0f});
  REQUIRE(w->flush(false).has_value());

  // Append another and truncate last file by 1 byte
  append_upsert(*w, 3, 302, {{"z","q"}}, {7.0f});
  REQUIRE(w->flush(false).has_value());
  std::filesystem::path last; std::uint64_t max_seq=0;
  for (auto& de : fs::directory_iterator(dir)){
    if (!de.is_regular_file()) continue; auto name = de.path().filename().string();
    if (name.rfind("wal-", 0)==0) { auto seq = std::stoull(name.substr(4,8)); if (seq>max_seq){ max_seq=seq; last=de.path(); } }
  }
  REQUIRE(!last.empty()); auto sz = fs::file_size(last); fs::resize_file(last, sz-1, ec); REQUIRE(!ec);

  ToyIndex idx;
  auto st = wal::recover_replay(dir, [&](std::uint64_t, std::uint16_t, std::span<const std::uint8_t> pl){ apply_frame_payload(pl, idx); });
  REQUIRE(st.has_value());
  REQUIRE(idx.count(300) == 1);
  REQUIRE(idx.count(301) == 1);

  fs::remove_all(dir, ec);
}

