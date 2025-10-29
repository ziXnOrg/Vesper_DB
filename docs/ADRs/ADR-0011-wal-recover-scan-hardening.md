# ADR-0011: WAL recover_scan hardening (header validation and length bounds)

Status: Accepted  
Date: 2025-10-27  
Component: WAL recovery (src/wal/io.cpp)

## Context

The WAL recovery scanner `recover_scan()` previously:
- Read the 20-byte header, then immediately trusted the 32-bit LEN field to allocate/read the remainder.
- Validated CRC and decoding only after the full frame was read.
- Did not check the 4-byte magic (`WAL_MAGIC`) before allocation, and had no explicit upper bound on LEN nor any remaining-file-size check.

This permitted corrupt/malicious files to trigger large allocations (OOM/DoS risk) or long blocking reads during recovery.

## Decision

Harden `recover_scan()` with strict, early validation before any large allocation or read:
- Validate `WAL_MAGIC` immediately after reading the header; on mismatch, stop scanning (torn/corrupt tail semantics).
- Enforce a maximum total frame length `MAX_FRAME_LEN = 32 MiB`.
- Compute remaining bytes in the file (using `file_size(path) - tellg()`) and stop if the declared remainder exceeds the remaining file bytes.
- On any guard violation, stop scanning cleanly without error (consistent with tolerant end-of-file behavior). Directory scans retain existing torn-middle-file detection and error signaling.

## Rationale

- Prevents OOM/DoS from adversarial LEN values in corrupt headers.
- Aligns with robust WAL designs (SQLite/LevelDB/RocksDB) that avoid trusting unvalidated length fields.
- Preserves compatibility and caller-facing error behavior: single-file scans stop silently at torn tails; directory scans continue to report data_integrity for torn middle files.

## Consequences

- Recovery is safer against corruption; no oversized allocations are attempted.
- Valid files are unaffected; performance overhead is negligible (constant-time checks and one filesystem size query per frame when available).
- The fixed cap (32 MiB) can be exposed/configured in a future change; producer-side caps can be mirrored in `encode_frame_expected`.

## Testing

- Added tests/unit/wal_recover_scan_len_magic_guard_test.cpp with cases:
  - Invalid magic with huge LEN → no allocation, scan stops with 0 frames.
  - LEN > 32 MiB → scan stops with 0 frames.
  - LEN exceeds remaining bytes → scan stops with 0 frames.
- All existing [wal] and [manifest] tests pass; zero new warnings.

## References

- `src/wal/io.cpp` (recover_scan guards)
- `include/vesper/wal/frame.hpp` (WAL_MAGIC, header layout)
- Related: ADR-0010 (WAL writer durability)

