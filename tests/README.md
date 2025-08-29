# Tests — Quick start

This repo uses Catch2 v3 and CTest. The commands below are deterministic and work on macOS/Linux/Windows shells.

## Build (Release)

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

## Run all tests

```sh
ctest --test-dir build --output-on-failure
```

## Run WAL-related subsets (by tag or name)

```sh
# All WAL-tagged tests (io, replay, snapshot, manifest, rotation)
ctest --test-dir build --output-on-failure -R "wal"

# Specific groupings
ctest --test-dir build --output-on-failure -R "replay|snapshot|manifest|rotation"
```

## Run property tests specifically

```sh
# Property tests for WAL
ctest --test-dir build --output-on-failure -R "wal_property_replay_test|property"
```

## Determinism notes

- Tests write under std::filesystem::temp_directory_path() and clean up afterward.
- No environment variables or clocks are used; inputs are deterministic.
- For faster local iterations, use -R to filter by tags/patterns, e.g.:

```sh
ctest --test-dir build --output-on-failure -R "wal.*(replay|snapshot)"
```

## Test helpers

- wal_replay_helpers.*
  - Purpose: reconstruct baseline (LSN ≤ cutoff) via per-file scan and apply replay (LSN > cutoff)
  - Example:
    - ToyIndex idx = test_support::build_toy_index_baseline_then_replay(dir, cutoff);

- manifest_test_helpers.*
  - Purpose: deterministic wal.manifest read/write and simple transforms
  - Common APIs:
    - auto lines = manifest_test_helpers::read_manifest_entries(manifest);
    - manifest_test_helpers::write_manifest_entries(manifest, lines);
    - auto files = manifest_test_helpers::list_wal_files_sorted(dir);
    - auto kept = manifest_test_helpers::entries_without_filename(lines, files.back().second.filename().string());
  - Avoid duplicating parsing/rewrites in tests; keep logic declarative.


### Retention helpers quick reference

- APIs:
  - wal::purge_wal(dir, up_to_lsn)
  - wal::purge_keep_last_n(dir, keep_last_n)
  - wal::purge_keep_newer_than(dir, cutoff_time)

- Example (keep last 2 files, then replay > cutoff):
  - REQUIRE(wal::save_snapshot(dir, wal::Snapshot{.last_lsn = 3}).has_value());
  - REQUIRE(wal::purge_keep_last_n(dir, 2).has_value());
  - auto st = wal::recover_scan_dir(dir, [&](const wal::WalFrame& f){ /* use f */ });

- Run only retention tests:
  - ctest --test-dir build --output-on-failure -R "retention|purge"


## WAL stabilization summary (Phase 4)

- Coverage added (deterministic, cross-platform):
  - Property tests: rotations + optional torn tail; non-monotonic LSN variant
  - Padding frames (type=3): ignored by monotonicity; tolerated by replay
  - Non-monotonic across rotation boundary
  - Stale manifest scanning: highest-seq file included even if unlisted
  - Perf sanity: recover_scan_dir scales ~O(N) (ratio ≤ 3)
  - Error propagation: data_integrity on non-last corruption; io on missing file; torn tail on last tolerated
  - Purge: file-boundary cutoffs; stale-manifest parity; idempotency
  - Helpers: wal_replay_helpers.*, manifest_test_helpers.* (refactors removed duplication)

- Quick subsets (copy/paste):
  - ctest --test-dir build --output-on-failure -R "wal|replay|manifest|snapshot|purge|errors|padding|roundtrip|boundaries|perf"
  - ctest --test-dir build --output-on-failure -R "wal_property_replay_test|property"

- Validation commands:
  - cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release -j
  - ctest --test-dir build --output-on-failure
  - ctest --test-dir build --output-on-failure -R "wal|replay|manifest|snapshot|purge|errors|padding|roundtrip|boundaries|perf"

- No production behavior or API/ABI changes

