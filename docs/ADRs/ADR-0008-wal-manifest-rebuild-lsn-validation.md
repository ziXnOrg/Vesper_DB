# ADR-0008: WAL manifest rebuild LSN validation

Status: Accepted
Date: 2025-10-27

## Context
`rebuild_manifest()` scans wal-*.log files and constructs Manifest entries by reading frames and deriving LSN bounds. After ADR-0007, the system defines LSN invariants: intra-entry `start_lsn <= first_lsn <= end_lsn`, and cross-entry constraints across increasing `seq`: no overlaps (`prev.end_lsn < curr.start_lsn`) and end monotonicity (`curr.end_lsn >= prev.end_lsn`). Previously, `rebuild_manifest()` returned entries without validating these invariants, risking generation of invalid manifests that later fail validation or lead to ambiguous recovery.

## Decision
- Validate LSN invariants in `rebuild_manifest()` before returning the Manifest:
  - Intra-entry: enforce `start_lsn <= first_lsn <= end_lsn`.
  - Cross-entry (sorted by filename-derived `seq`): enforce no overlaps and monotonic `end_lsn`.
  - Gaps are allowed and not treated as errors at rebuild time (operators will see them as `LsnGap` warnings via `validate_manifest()`).
- Error handling: fail-fast on first violation with `error_code::data_integrity` and actionable diagnostics (include previous/current file names, seq, and LSN values where applicable).
- No-throw contract preserved via `std::expected`.

## Consequences
- Rebuilt manifests are guaranteed to satisfy LSN invariants from ADR-0007, increasing recovery safety and determinism.
- Operators receive immediate feedback when on-disk WAL files encode overlapping or invalid LSN ranges.
- No on-disk format changes; behavior is stricter and safer.

## Alternatives Considered
1) Post-generation validation by calling `validate_manifest()` on a temp manifest file. Rejected due to indirection and unnecessary IO; prefer in-memory, fail-fast checks.
2) Best-effort rebuild with warnings and partial Manifest. Rejected for default path; non-deterministic recovery risk. A future "lenient" mode could be considered under a separate ADR.

## Implementation
- `src/wal/manifest.cpp` (`rebuild_manifest()`): after scanning each file to compute `(start, first, end)`, perform intra-entry checks and then cross-entry checks against the prior entry. On violation, return `unexpected(error{data_integrity, ...})` with detailed context.
- No interface changes.

## Testing
- Unit tests in `tests/unit/wal_manifest_rebuild_test.cpp`:
  - TID-WAL-MAN-REBUILD-LSN-001: Valid LSN ranges across files → rebuild ok; saved manifest validates ok.
  - TID-WAL-MAN-REBUILD-LSN-002: Overlap across files (curr.start <= prev.end) → rebuild error.
  - TID-WAL-MAN-REBUILD-LSN-003: Intra-entry start_lsn > end_lsn within a single file → rebuild error.
  - TID-WAL-MAN-REBUILD-LSN-004: Gap across files (curr.start > prev.end+1) → rebuild ok; validate reports `LsnGap` Warning; ok=true.

## Migration
- None required. Existing valid WAL directories rebuild successfully. Invalid directories (overlaps/invalid LSN ranges) will fail with explicit diagnostics and require operator remediation (e.g., pruning or replaying to regenerate consistent segments).

