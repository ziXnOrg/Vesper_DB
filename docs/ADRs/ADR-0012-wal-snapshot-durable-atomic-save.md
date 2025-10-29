# ADR-0012: WAL snapshot durable atomic save (fsync + replace semantics)

Date: 2025-10-27
Status: Accepted

Context
- The WAL snapshot (`wal.snapshot`) records `last_lsn` to bound recovery.
- Prior implementation wrote `wal.snapshot.tmp` then `std::filesystem::rename` to `wal.snapshot` without fsync/FlushFileBuffers on the temp file. On Windows, `rename` can fail if destination exists; fallback rewrote `wal.snapshot` non-atomically and leaked `wal.snapshot.tmp`.
- This risks lost/corrupt snapshots on power loss and violates atomic replace guarantees.

Decision
- Implement platform-correct durable sequence when `VESPER_ENABLE_ATOMIC_RENAME` is enabled:
  1) Write snapshot to sibling temp file `wal.snapshot.tmp` and flush the stream
  2) Ensure file durability:
     - POSIX: `fsync(tmp)`
     - Windows: `FlushFileBuffers(tmp)`
  3) Atomically replace the destination:
     - POSIX: `std::filesystem::rename(tmp, dst)` (rename(2) semantics replace if exists)
     - Windows: `MoveFileExW(tmp, dst, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)`
  4) Best-effort directory durability:
     - POSIX: `fsync(parent directory)`
     - Windows: `FlushFileBuffers(directory handle opened with FILE_FLAG_BACKUP_SEMANTICS)`
  5) On any failure, remove `wal.snapshot.tmp` and return an error; do not fallback to non-atomic rewrite.

- When `VESPER_ENABLE_ATOMIC_RENAME` is disabled at compile time, fall back to a simple truncating write of `wal.snapshot`.

Consequences
- Snapshot updates are atomic and durable across supported platforms, matching SQLite/LevelDB patterns.
- Windows destination replacement works even when the destination exists; failure paths no longer leak `wal.snapshot.tmp`.
- Error model preserved (`std::expected<void, error>`); no exceptions introduced on hot paths.

Implementation
- File: `src/wal/snapshot.cpp`
  - Added Windows includes guarded by `_WIN32`.
  - Implemented fsync/FlushFileBuffers on temp, atomic replace (`MoveFileExW` on Windows), best-effort directory flush; removed non-atomic fallback; cleanup tmp on all paths.
- Header docs: `include/vesper/wal/snapshot.hpp` updated to document durability sequence and platform behavior.
- Tests: `tests/unit/wal_snapshot_durable_replace_test.cpp`
  - Validates replacement when destination exists and tmp is cleaned up.
  - Windows-only test holds destination open to force rename failure and verifies tmp cleanup (no lingering tmp on failure paths).

Validation
- Targeted tests: `[wal][snapshot][durable]` pass.
- Full suites: `[wal]` and `[manifest]` pass locally.
- Zero new compiler warnings introduced by these changes.
- Deterministic tests with fixed seeds.

References
- SQLite/LevelDB/RocksDB atomic update pattern: write tmp → fsync file → rename/replace → fsync dir.

