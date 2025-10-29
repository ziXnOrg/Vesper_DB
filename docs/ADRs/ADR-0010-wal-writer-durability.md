# ADR-0010 — WAL Writer Durability (fsync/FlushFileBuffers)

Status: Accepted
Date: 2025-10-27
Component: WAL (WalWriter), src/wal/io.cpp, include/vesper/wal/io.hpp

## Context

Prior to this change, the WAL durability knobs (`DurabilityProfile`, `fsync_on_rotation`, `fsync_on_flush`) were stats-only. The code did not issue OS-level persistence calls (fsync/fdatasync on POSIX; FlushFileBuffers on Windows), nor did it fsync the parent directory after creating/rotating files. This violated crash‑safety: power loss after a reported flush/rotation could lose acknowledged frames or manifest updates.

## Decision

Implement real, platform-correct durability in `WalWriter`:

- flush(sync)
  - Always flush C++ stream buffers first.
  - POSIX: open the file by path with `O_RDONLY` and call `fsync(fd)`.
  - Windows: best‑effort `FlushFileBuffers` on a separate handle; if a sharing restriction prevents opening or flushing, fall back to close → `FlushFileBuffers` → reopen the stream in append mode. Treat sharing/access‑denied violations as success (best‑effort semantics). Increment `stats_.syncs` on success.
- rotation
  - On rotation, flush and close the old file first, then ensure durability via `fsync/FlushFileBuffers` before counting the rotation, and then open the next sequence file.
  - When `fsync_on_rotation` is enabled, ensure directory metadata durability after creating the new file:
    - POSIX: open parent dir with `O_DIRECTORY` and `fsync`.
    - Windows: open directory with `FILE_FLAG_BACKUP_SEMANTICS` and call `FlushFileBuffers` (best‑effort; semantics vary by FS).
- Error propagation
  - Fail early with `std::expected<…, core::error>` on hard IO failures; sharing/access‑denied cases on Windows are treated as best‑effort success to avoid spurious failures caused by standard stream share modes.
- Observability
  - `stats_.syncs` now counts actual OS‑level sync attempts; `stats_.flushes` counts stream flushes.
- No behavior change when knobs are disabled.

## Rationale

- POSIX
  - `fsync` on a descriptor to the file is sufficient for data durability; directory fsync is needed for metadata/rename durability on rotation/creation.
- Windows
  - Standard `std::ofstream` does not expose a stable way to retrieve the underlying `HANDLE` for `FlushFileBuffers`. Opening a second handle to the same path can fail due to share flags chosen by the iostream implementation. The design therefore:
    - Tries best‑effort `FlushFileBuffers` on a separate handle.
    - Falls back to close → flush → reopen in append mode when needed.
    - Treats sharing/access‑denied cases as success for the best‑effort path to avoid fatal errors caused purely by stream share policy.
  - This keeps the API exception‑free on hot paths and preserves deterministic behavior in tests.

## Alternatives considered

1) Rework WalWriter to use low‑level OS descriptors/handles instead of `std::ofstream`, enabling direct `fsync/_commit/FlushFileBuffers` on the active handle.
   - Pros: Stronger guarantees; fewer sharing caveats on Windows.
   - Cons: Larger refactor; platform‑specific code in hot paths; more error‑prone stream state management.

2) Use non‑standard hooks to obtain the native FD/HANDLE from `std::ofstream`.
   - Rejected: Non‑portable and undefined across standard library implementations.

3) Omit directory fsync on rotation.
   - Rejected: Violates durability for metadata updates.

## Consequences

- Slight additional latency when `sync=true` or `fsync_on_*` is enabled (expected and controlled by knobs).
- Windows semantics are best‑effort when a separate durable handle cannot be obtained due to share flags; rotation path avoids share conflicts by syncing after close.
- Stats now reflect actual sync attempts, improving observability.

## Compatibility

- No ABI changes. Public headers updated to document OS‑level sync behavior when durability knobs are enabled.
- Behavior unchanged when durability profile is `None`.

## Tests

- wal_writer_durability_profile_test.cpp — durability profiles map to fsync knobs and stats.
- wal_writer_fsync_test.cpp — fsync-on-flush policy control.
- wal_writer_flush_sync_true_test.cpp — explicit `flush(true)` triggers durable sync regardless of profile.

All tests pass on Windows; no new warnings.

## References

- SQLite durable rename and directory fsync
- LevelDB/RocksDB WAL and MANIFEST durability patterns
- Windows FlushFileBuffers and directory handle caveats

