# Contributing WAL tests

This guide summarizes how to add deterministic, cross-platform WAL tests.

Principles
- Deterministic: no wall clock, no sleeps, fixed inputs; use std::filesystem temp dirs and clean up
- Cross-platform: std::filesystem only; avoid platform-specific permissions
- Fast: keep each test well under 1s; small payloads; minimal rotations
- No production changes or deps for tests

Helpers to use
- wal_replay_helpers.*
  - Build baseline (LSN ≤ cutoff) via per-file scan + apply replay (LSN > cutoff)
  - Example:
    - ToyIndex idx = test_support::build_toy_index_baseline_then_replay(dir, cutoff);
- manifest_test_helpers.*
  - Read/write wal.manifest and apply simple transforms
  - Examples:
    - auto lines = manifest_test_helpers::read_manifest_entries(manifest);
    - auto kept  = manifest_test_helpers::entries_without_filename(lines, filename);
    - manifest_test_helpers::write_manifest_entries(manifest, kept);

Tags to apply
- Always include [wal], and add specific tags such as [replay], [manifest], [snapshot], [purge], [errors], [padding], [roundtrip], [boundaries], [perf]

Test template (minimal)

```cpp
#include <catch2/catch_all.hpp>
#include <vesper/wal/io.hpp>
#include <tests/support/replayer_payload.hpp>
#include <tests/support/wal_replay_helpers.hpp>
#include <filesystem>

using namespace vesper;
using namespace test_support;

TEST_CASE("describe behavior", "[wal][manifest]"){
  namespace fs = std::filesystem;
  auto dir = fs::temp_directory_path() / "vesper_wal_minimal";
  std::error_code ec; fs::remove_all(dir, ec); fs::create_directories(dir, ec);

  wal::WalWriterOptions opts{.dir=dir, .prefix="wal-", .max_file_bytes=64, .strict_lsn_monotonic=false};
  auto w = wal::WalWriter::open(opts); REQUIRE(w.has_value());

  auto up = make_upsert(101, std::vector<float>{1.0f}, {});
  REQUIRE(w->append(1, /*type=*/1, up).has_value());
  REQUIRE(w->flush(false).has_value());

  ToyIndex idx = build_toy_index_baseline_then_replay(dir, /*cutoff=*/0);
  REQUIRE(idx.count(101) == 1);

  fs::remove_all(dir, ec);
}
```

Quick commands
- Build (Release):
  - cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release -j
- Run all:
  - ctest --test-dir build --output-on-failure
- Run WAL subsets:
  - ctest --test-dir build --output-on-failure -R "wal|replay|manifest|snapshot|purge|errors|padding|roundtrip|boundaries|perf"

