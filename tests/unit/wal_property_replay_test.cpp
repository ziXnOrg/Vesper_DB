#include <catch2/catch_all.hpp>
#include <vesper/wal/io.hpp>
#include <vesper/wal/replay.hpp>
#include <vesper/wal/snapshot.hpp>
#include <filesystem>
#include <unordered_set>
#include <tests/support/wal_replay_helpers.hpp>

#include <tests/support/replayer_payload.hpp>

using namespace vesper;
using namespace test_support;
using test_support::build_toy_index_baseline_then_replay;


using Catch::Approx;

namespace {
struct Op { std::uint64_t lsn; std::uint16_t type; std::uint64_t id; std::vector<std::uint8_t> payload; };

static std::vector<Op> make_ops_deterministic(int N){
  std::vector<Op> ops; ops.reserve(N);
  for (int i=1;i<=N;++i){
    std::uint64_t id = 100 + static_cast<std::uint64_t>(((i-1)%3)+1); // 101,102,103
    if (i % 4 == 0) {
      auto pl = make_delete(id);
      ops.push_back(Op{static_cast<std::uint64_t>(i), /*type=*/2, id, std::move(pl)});
    } else {
      float a = static_cast<float>(i)*0.5f;
      float b = static_cast<float>(id)*0.01f;
      std::vector<float> vec{a,b};
      if ((i % 2) == 0) {
        auto pl = make_upsert(id, vec, {{"parity","even"}});
        ops.push_back(Op{static_cast<std::uint64_t>(i), /*type=*/1, id, std::move(pl)});
      } else {
        auto pl = make_upsert(id, vec, {});
        ops.push_back(Op{static_cast<std::uint64_t>(i), /*type=*/1, id, std::move(pl)});
      }
    }
  }
  return ops;
}

static void apply_ops_to_index(const std::vector<Op>& ops, ToyIndex& idx){
  for (const auto& op : ops) {
    (void)op.id; // id is encoded in payload schema; apply by payload only
    apply_frame_payload(op.payload, idx);
  }
}
}

TEST_CASE("WAL property replay across rotations (deterministic)", "[wal][replay][property]"){
  namespace fs = std::filesystem;
  const bool torn_tail = true; // deterministic: enable torn tail
  const int N = 24;
  const std::uint64_t cutoff = 10;

  auto dir = fs::temp_directory_path() / "vesper_wal_prop";
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);

  wal::WalWriterOptions opts{.dir=dir, .prefix="wal-", .max_file_bytes=64, .strict_lsn_monotonic=false};
  auto w = wal::WalWriter::open(opts); REQUIRE(w.has_value());

  auto ops = make_ops_deterministic(N);
  for (const auto& op : ops) {
    REQUIRE(w->append(op.lsn, op.type, op.payload).has_value());
  }
  REQUIRE(w->flush(false).has_value());

  // Optional torn tail: truncate the last file by 1 byte
  if (torn_tail) {
    fs::path last; std::uint64_t max_seq=0;
    for (auto& de : fs::directory_iterator(dir)){
      if (!de.is_regular_file()) continue; auto name = de.path().filename().string();
      if (name.rfind("wal-", 0)==0) { auto seq = std::stoull(name.substr(4,8)); if (seq>max_seq){ max_seq=seq; last=de.path(); } }
    }
    REQUIRE(!last.empty());
    auto sz = fs::file_size(last); REQUIRE(sz>0);
    fs::resize_file(last, sz-1, ec); REQUIRE(!ec);
  }

  // Snapshot cutoff
  REQUIRE(wal::save_snapshot(dir, wal::Snapshot{.last_lsn=cutoff}).has_value());

  // Ground truth expected: baseline (<=cutoff) then apply (>cutoff, with torn-tail dropping last frame if present)
  ToyIndex idx_expected;
  {
    std::vector<Op> baseline, post;
    baseline.reserve(N); post.reserve(N);
    for (const auto& op : ops) {
      if (op.lsn <= cutoff) baseline.push_back(op);
      else post.push_back(op);
    }
    if (torn_tail && !post.empty()) {
      // Drop the last frame (N)
      post.pop_back();
    }
    apply_ops_to_index(baseline, idx_expected);
    apply_ops_to_index(post, idx_expected);
  }

  // Stats: recover_scan_dir should deliver frames strictly after cutoff and ignore torn tail
  std::size_t exp_frames = 0; std::size_t exp_upserts=0; std::size_t exp_deletes=0; std::uint64_t exp_last_lsn=0;
  for (const auto& op : ops) {
    if (op.lsn > cutoff) { exp_frames++; if (op.type==1) exp_upserts++; else if (op.type==2) exp_deletes++; exp_last_lsn = op.lsn; }
  }
  if (torn_tail) {
    // Dropped the last frame
    if (exp_frames>0) {
      const auto& last = ops.back();
      if (last.lsn > cutoff) { exp_frames--; if (last.type==1) exp_upserts--; else if (last.type==2) exp_deletes--; exp_last_lsn = exp_last_lsn - 1; }
    }
  }

  std::size_t delivered_payload_bytes = 0;
  auto stats_exp = wal::recover_scan_dir(dir, [&](const wal::WalFrame& f){ delivered_payload_bytes += f.payload.size(); });
  REQUIRE(stats_exp.has_value());
  auto stats = *stats_exp;
  REQUIRE(stats.frames == exp_frames);
  REQUIRE(stats.last_lsn == exp_last_lsn);
  REQUIRE(stats.type_counts[1] == exp_upserts);
  REQUIRE(stats.type_counts[2] == exp_deletes);
  REQUIRE(stats.lsn_monotonic == true);

  // Replayed index: build baseline via recover_scan (<=cutoff) then apply recover_replay (>cutoff)
  ToyIndex idx_replayed;
  {
    // Use test helper to build state deterministically
    idx_replayed = build_toy_index_baseline_then_replay(dir, cutoff);
  }

  // Equivalence: same keys and values
  REQUIRE(idx_replayed.size() == idx_expected.size());
  for (const auto& [id, doc] : idx_expected) {
    REQUIRE(idx_replayed.count(id) == 1);
    const auto& d2 = idx_replayed.at(id);
    REQUIRE(d2.vec.size() == doc.vec.size());
    for (std::size_t i=0;i<doc.vec.size();++i) REQUIRE(d2.vec[i] == Approx(doc.vec[i]).margin(1e-6));
    REQUIRE(d2.tags.size() == doc.tags.size());
    for (const auto& kv : doc.tags) {
      REQUIRE(d2.tags.count(kv.first) == 1);
      REQUIRE(d2.tags.at(kv.first) == kv.second);
    }
  }

  fs::remove_all(dir, ec);
}



TEST_CASE("WAL property replay with non-monotonic LSN (deterministic)", "[wal][replay][property][nonmonotonic]"){
  namespace fs = std::filesystem;
  const int N = 24;
  const std::uint64_t cutoff = 10;

  auto dir = fs::temp_directory_path() / "vesper_wal_prop_nonmono";
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);

  wal::WalWriterOptions opts{.dir=dir, .prefix="wal-", .max_file_bytes=0, .strict_lsn_monotonic=false};
  auto w = wal::WalWriter::open(opts); REQUIRE(w.has_value());

  auto ops = make_ops_deterministic(N);
  // Inject one non-monotonic event: at write position 16 (1-based), set lsn to 11 (< previous delivered lsn) and > cutoff to be delivered
  REQUIRE(ops.size() >= 16);
  ops[15].lsn = cutoff + 1; // 11

  for (const auto& op : ops) REQUIRE(w->append(op.lsn, op.type, op.payload).has_value());
  REQUIRE(w->flush(false).has_value());

  // Snapshot cutoff
  REQUIRE(wal::save_snapshot(dir, wal::Snapshot{.last_lsn=cutoff}).has_value());

  // Expected stats computed from mutated ops (>cutoff only)
  std::size_t exp_frames = 0; std::size_t exp_upserts=0; std::size_t exp_deletes=0; std::uint64_t exp_last_lsn=0;
  for (const auto& op : ops) {
    if (op.lsn > cutoff) { exp_frames++; if (op.type==1) exp_upserts++; else if (op.type==2) exp_deletes++; exp_last_lsn = op.lsn; }
  }

  // Run recover_scan_dir to gather stats; also ensure it flags the non-monotonicity
  std::size_t delivered_payload_bytes = 0;
  auto stats_exp = wal::recover_scan_dir(dir, [&](const wal::WalFrame& f){ delivered_payload_bytes += f.payload.size(); });
  REQUIRE(stats_exp.has_value());
  auto stats = *stats_exp;
  REQUIRE(stats.frames == exp_frames);
  REQUIRE(stats.last_lsn == exp_last_lsn);
  REQUIRE(stats.type_counts[1] == exp_upserts);
  REQUIRE(stats.type_counts[2] == exp_deletes);
  REQUIRE(stats.lsn_monotonic == false);
  REQUIRE(stats.lsn_violations == 1);

  // Build expected and replayed states
  ToyIndex idx_expected;
  {
    std::vector<Op> baseline, post; baseline.reserve(N); post.reserve(N);
    for (const auto& op : ops) { if (op.lsn <= cutoff) baseline.push_back(op); else post.push_back(op); }
    apply_ops_to_index(baseline, idx_expected);
    apply_ops_to_index(post, idx_expected);
  }

  ToyIndex idx_replayed;
  {
    // Use test helper to build state deterministically
    idx_replayed = build_toy_index_baseline_then_replay(dir, cutoff);
  }

  // Equivalence check
  REQUIRE(idx_replayed.size() == idx_expected.size());
  for (const auto& [id, doc] : idx_expected) {
    REQUIRE(idx_replayed.count(id) == 1);
    const auto& d2 = idx_replayed.at(id);
    REQUIRE(d2.vec.size() == doc.vec.size());
    for (std::size_t i=0;i<doc.vec.size();++i) REQUIRE(d2.vec[i] == Approx(doc.vec[i]).margin(1e-6));
    REQUIRE(d2.tags.size() == doc.tags.size());
    for (const auto& kv : doc.tags) {
      REQUIRE(d2.tags.count(kv.first) == 1);
      REQUIRE(d2.tags.at(kv.first) == kv.second);
    }
  }

  fs::remove_all(dir, ec);
}
