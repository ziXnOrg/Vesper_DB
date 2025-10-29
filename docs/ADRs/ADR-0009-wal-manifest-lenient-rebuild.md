# ADR-0009: WAL Manifest Lenient Rebuild Mode (Best-Effort)

Date: 2025-10-27
Status: Accepted
Authors: Vesper Team
Related: ADR-0003 (durable manifest update), ADR-0007 (LSN invariants), ADR-0008 (strict rebuild validation)

## Context

After ADR-0008, `rebuild_manifest()` validates LSN invariants during generation and fails fast on violations. This is safe-by-default, but operationally brittle when a subset of WAL segment files is corrupt. Operators need a recovery path that salvages all valid segments, documenting what was skipped, without compromising consistency of the resulting manifest.

Constraints (Vesper):
- No exceptions on hot paths; use `std::expected<T, error>`
- Deterministic, actionable diagnostics
- Invariants (ADR-0007) must hold for any manifest we return as valid
- No on-disk mutations from rebuild utilities (read-only)

Industry references:
- RocksDB Repairer: best-effort recovery while preserving consistency
- PostgreSQL: `pg_resetwal` and `zero_damaged_pages` are explicit and dangerous (not our model)
- SQLite: allows torn tails; provides `.recover` tooling

## Decision

Introduce an explicit lenient (best-effort) rebuild API alongside strict rebuild.

- Strict (default, unchanged):
  - `auto rebuild_manifest(const std::filesystem::path&) -> expected<Manifest, error>`
  - Behavior: fail-fast on any per-file scan error or LSN invariant violation; returns `unexpected(error{data_integrity,...})` with precise context.

- Lenient (new):
  - `auto rebuild_manifest_lenient(const std::filesystem::path&) -> expected<LenientRebuildResult, error>`
  - `struct LenientRebuildResult { Manifest manifest; std::vector<RebuildIssue> issues; }`
  - `struct RebuildIssue { std::string file; std::uint64_t seq; error_code code; std::string message; /* optional LSN context: start/first/end/prev_end */ }`
  - Behavior: iterate files in sequence order, scan each; on IO/parse error or invariant violation, skip the file/entry and append a `RebuildIssue`. Only include entries that satisfy:
    - Intra-entry: `start_lsn <= first_lsn <= end_lsn`
    - Cross-entry: no overlap with previous included entry; `end_lsn` monotonic non-decreasing
  - Gaps in LSN ranges are allowed; these surface as `Severity::Warning` via `validate_manifest()`.

Implementation approach:
- Shared internal implementation (`rebuild_manifest_impl(dir, RebuildMode mode)`) to avoid duplication and guarantee identical logic for invariant checks across modes.
- No mutation of files on disk. Persisting the result is the caller’s choice via `save_manifest(dir, result.manifest)`.

## Consequences

- Operators get a deterministic, documented path to salvage valid WAL segments after partial corruption.
- Strict mode remains the default and safest path for automated recovery; lenient is opt-in and explicit.
- Recovery code can trust that returned manifests (strict or lenient) satisfy invariants for included entries.
- Diagnostics (`RebuildIssue`) are machine-readable and suitable for logs/telemetry.

## Alternatives Considered

1) Strict-only (status quo) — Rejected: prevents recovery when partial data is salvageable.
2) Auto-repair/merge of overlapping files — Rejected: ambiguous semantics; risks hidden data loss.
3) Environment-variable toggles — Rejected: implicit behavior is risky; prefer explicit API.
4) Return `ManifestValidation` from lenient — Considered; chosen to return `LenientRebuildResult` for direct access to partial manifest plus issues.

## Testing and Validation

- Added 6 unit tests:
  - TID-WAL-MAN-REBUILD-LENIENT-001: all valid files → no issues
  - TID-WAL-MAN-REBUILD-LENIENT-002: one file IO error (Windows-specific attempt) → partial + 1 issue (best-effort if lock not acquired)
  - TID-WAL-MAN-REBUILD-LENIENT-003: intra-entry invalid LSN → skipped with issue
  - TID-WAL-MAN-REBUILD-LENIENT-004: multiple corrupt files → multiple issues
  - TID-WAL-MAN-REBUILD-LENIENT-005: all files IO errors (Windows-only attempt) → empty manifest + issues for all (conditional assertion)
  - TID-WAL-MAN-REBUILD-LENIENT-006: partial manifest validates ok (warnings allowed)
- Verified strict `[rebuild]` behavior unchanged and `[wal]` suite still green.
- No new compiler warnings.

## Migration & Usage Guidance

- Default to strict mode in automated recovery and CI paths.
- Use lenient mode when corruption is suspected and operator goal is to recover as much as possible without manual triage.
- Always inspect and persist the returned partial manifest explicitly; run `validate_manifest(dir)` after saving.

## Security & Safety Considerations

- No silent data loss: entries are either included in full or skipped with explicit `RebuildIssue`.
- No mutation side effects: read-only operations.
- Determinism: processing order is fixed by `seq`; diagnostics include filenames/seq for audit.

