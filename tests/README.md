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

- Prefer using tests/support/wal_replay_helpers.* to reconstruct baseline (≤cutoff) and apply replay (>cutoff) in WAL replay tests.

