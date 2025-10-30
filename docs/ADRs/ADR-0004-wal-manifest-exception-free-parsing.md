# ADR-0004: WAL Manifest Exception-Free Parsing

Date: 2025-10-27
Status: Accepted

## Context

The WAL manifest loader `load_manifest()` in `src/wal/manifest.cpp` used throwing conversions (`std::stoull`). On malformed or out-of-range numeric fields, these throw and can terminate recovery paths that expect `std::expected` error propagation. This violates Vesper rules: no exceptions on hot paths, crash-safety, and explicit error models. Additional throwing parses existed in `validate_manifest()` and `list_sorted()` helpers.

## Decision

- Replace all numeric parsing in the manifest module with non-throwing `std::from_chars` (base-10) and explicit validation.
- `load_manifest(dir) -> expected<Manifest, error>`:
  - Parse each `key=value` token, using `from_chars` for u64 fields, verifying full consumption (no trailing chars) and range.
  - Validate `file` does not contain control characters; reject invalid filenames.
  - Require presence of mandatory fields: `file, seq, start_lsn, end_lsn, frames, bytes`; `first_lsn` optional and defaults to `start_lsn` (back-compat).
  - On any violation, return `unexpected(error{data_integrity, ... , "wal.manifest"})` with actionable context (line number, field).
- `validate_manifest(dir)`:
  - Switch to non-throwing parse; on malformed numeric, mark `ok=false` and push `BadHeader` issue with line info; continue processing other lines.
- `list_sorted(dir, prefix)`:
  - Switch from try/catch `stoull` to `from_chars`; ignore entries that fail to parse.
- Keep file format unchanged (v1 header + kv lines). Error handling only; no schema changes.

## Consequences

- Recovery and manifest loading are robust to malformed inputs; no exceptions propagate from parsing.
- Errors are explicit and actionable via `vesper::core::error`.
- Fuzzer now runs without a try/catch wrapper for manifest paths; any unexpected throw will crash the fuzzer, surfacing regressions quickly.

## Alternatives Considered

- Guarded `strtoull` with `errno` checks: viable but `from_chars` is faster, locale-free, and header-only.
- Swallow invalid lines in `load_manifest` and continue: rejected. Recovery semantics should not silently accept malformed metadata.

## Testing

- Unit tests (tests/unit/wal_manifest_parse_test.cpp):
  - TID-WAL-MAN-PARSE-001: malformed numeric field returns error (no throw)
  - TID-WAL-MAN-PARSE-002: overflow numeric value is rejected (no throw)
  - TID-WAL-MAN-PARSE-003: missing required fields are rejected (no throw)
  - TID-WAL-MAN-PARSE-004: invalid control chars in filename are rejected (no throw)
- Fuzzing (tests/fuzz/wal_manifest_fuzz.cpp):
  - Removed try/catch; exercises `load_manifest`, `validate_manifest`, `enforce_manifest_order` directly.
- Regression suite:
  - Catch2: [wal] subset passed after changes; Windows Debug build.

## Migration

- No file format change; no migration required.
- Callers already using `expected` continue to work; exception paths removed.
- Downstream: none.

