# Research: Lenient WAL Manifest Rebuild (Best-Effort)

Date: 2025-10-27
Author: Vesper Team
Related: ADR-0009

## Goal
Survey industry practices for recovering metadata/manifests/WAL state in the presence of corruption, and derive a safe "best-effort" strategy that preserves invariants for included segments while documenting skipped items.

## Findings (Industry)

- RocksDB Repairer
  - Intent: "Recover as much data as possible after a disaster without compromising consistency"
  - Approach: scans SSTs/WALs, reconstructs metadata; skips corrupt files; surfaces issues
  - Notes: requires explicit operator invocation; does not silently mask errors

- SQLite
  - WAL handles torn/truncated tails naturally; `.recover` command exists for schema/data reconstruction
  - Emphasis on deterministic behavior and explicit tooling rather than implicit auto-repair

- PostgreSQL
  - `pg_resetwal` and `zero_damaged_pages` are powerful and dangerous; they are explicit, last-resort tools and require operator acknowledgement
  - Not a model for default library behavior; informs our choice to make lenient rebuild opt-in and explicit via a separate API

## Vesper Constraints and Prior Art

- ADR-0007 defines LSN invariants; manifests we produce must satisfy these among included entries
- ADR-0008 added strict rebuild validation during generation
- `recover_scan()` semantics: torn/truncated tails are tolerated; only open failures return errors; non-monotonic LSNs counted in stats

## Design Implications

- Provide explicit lenient API: `rebuild_manifest_lenient(dir)` that returns partial `Manifest` + `issues`
- Preserve strict behavior as default: `rebuild_manifest(dir)` unchanged (fail-fast)
- Guarantee that included entries in lenient mode satisfy invariants; gaps are allowed and later warned by `validate_manifest()`
- No on-disk mutation in rebuild functions; callers persist via `save_manifest()` when desired

## Edge Cases and Diagnostics

- Per-file IO/open errors: skip file and emit `RebuildIssue{file, seq, code, message}`
- Intra-entry violations: `start_lsn > end_lsn`, `first_lsn < start_lsn`, `first_lsn > end_lsn`  skip and emit issue
- Cross-entry violations: overlap with previous included entry; `end_lsn` decrease  skip and emit issue
- Diagnostics should include optional LSN context for actionable insight: `start/first/end/prev_end`

## Testing Strategy

- Unit tests covering: all-valid; IO error on single file; intra-entry invalid; multiple corrupt; all corrupt; partial manifest validates ok
- Deterministic data generation using `WalWriter` with fixed seeds and bounded frame sizes to control file rotation
- Windows-only IO error simulation via `CreateFileW` exclusive lock (best-effort; conditional assertions if lock not acquired)

## Conclusion

The lenient rebuild mode aligns with industry best practices and Vesper's safety constraints by providing a deterministic, opt-in, best-effort recovery path that never compromises the invariants of the returned manifest while surfacing machine-readable issues for operator action.

