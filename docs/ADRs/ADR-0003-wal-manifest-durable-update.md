# ADR-0003: WAL Manifest Durable Atomic Update

Status: Accepted
Date: 2025-10-27
Authors: Colin MacRitchie - ziX Labs

## Context

The WAL manifest (wal.manifest) tracks WAL segment files and their metadata. The existing implementation writes directly to wal.manifest via truncating std::ofstream without file or directory fsync, and without atomic replace. This risks:
- Torn or partially written manifest on crash/power loss
- Lost updates due to lack of durability guarantees
- Inconsistent recovery/retention behavior across platforms

## Decision

Adopt a cross-platform durable update protocol for the manifest:
1) Write new contents to a temporary file wal.manifest.tmp
2) Flush file contents (fsync/FlushFileBuffers)
3) Atomically replace wal.manifest with the tmp file
   - POSIX: std::filesystem::rename (atomic within same filesystem)
   - Windows: Prefer ReplaceFileW (preserves attributes/ACLs); fallback to MoveFileExW(MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)
4) POSIX: fsync the parent directory to persist the rename; Windows: FlushFileBuffers on the final file handle
5) Remove any leftover wal.manifest.tmp from prior crashes; on Windows, if deletion fails due to sharing violations, fall back to a unique tmp name (wal.manifest.tmp.<pid>.<tick>) for this write

Writers use this protocol by default via save_manifest() delegating to save_manifest_atomic().

## Feature toggle

Reuse VESPER_ENABLE_ATOMIC_RENAME (also used by snapshot) with default enabled on Windows, Linux, macOS. This controls the atomic replace path. Non-atomic fallback remains for unsupported platforms.

## Consequences

- Crash-safety and durability improved; manifest updates are atomic and survive typical crashes
- Slight overhead from fsync/FlushFileBuffers; controllable via build flags if needed
- Platform caveats:
  - POSIX: rename is atomic on same filesystem; directory fsync best-effort may not be available on exotic filesystems
  - Windows: directory fsync is not generally supported; we flush the file handle and use MOVEFILE_WRITE_THROUGH; document as best-effort

## Alternatives considered

- Direct in-place write with fsync: still risks torn files on crash during overwrite
- Write + rename without fsync: improves atomicity but not durability; updates can be lost

## Testing

- Unit tests (tests/unit/wal_manifest_atomic_test.cpp):
  - TID-WAL-MAN-ATOMIC-001: atomic manifest save leaves only final file and round-trips via load_manifest()
  - TID-WAL-MAN-ATOMIC-002: manifest save succeeds when tmp file is missing
  - TID-WAL-MAN-ATOMIC-003 (Windows): fallback to unique tmp when default tmp is locked; cleanup succeeds
  - TID-WAL-MAN-ATOMIC-004 (POSIX): save succeeds when stale tmp exists; tmp cleaned

## Build and Test Results (2025-10-27)

- Build: Debug (Windows). Exit code 0. Zero warnings.
- Tests: 3/3 new wal_manifest_atomic tests passed on Windows; existing manifest/snapshot tests continue to pass.

## Migration

No file format changes. Behavior is backward compatible; existing tools reading wal.manifest continue to work.

