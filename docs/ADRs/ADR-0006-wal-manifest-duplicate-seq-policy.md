# ADR-0006: WAL manifest duplicate sequence policy

Status: Accepted
Date: 2025-10-27

## Context
A WAL manifest lists WAL segment files with monotonically increasing sequence numbers (`seq`). The previous validation only checked for out-of-order sequences and gaps, but did not explicitly detect duplicate `seq` values. Duplicate sequence numbers across different files create recovery ambiguity (two segments claim the same position), violating Vesper's data-integrity and determinism requirements.

## Decision
- Add explicit detection for duplicate `seq` values in `validate_manifest()`.
- Report each duplicate as `Severity::Error` with `ManifestIssueCode::DuplicateSeq`. The issue includes the current file, the duplicate `seq`, and mentions the first file seen for that `seq` in the message for actionable diagnosis.
- Keep `load_manifest()` as parse/format validation only; do not implement duplicate detection there.
- Recovery policy: fail-closed when validation reports any `Severity::Error` (including `DuplicateSeq`). Do not attempt auto-resolution (e.g., choosing higher LSN range or later file order). Operators should remediate the manifest or regenerate it via `rebuild_manifest()`.

## Consequences
- Ambiguous manifests are rejected deterministically; recovery will not proceed with undefined precedence rules.
- No behavior change for valid manifests; no on-disk format change.

## Alternatives considered
1) Best-effort resolution (pick highest end_lsn or latest mtime). Rejected: introduces non-determinism and risks data loss.
2) Detect and warn only. Rejected: would allow ambiguous recovery to proceed.

## Implementation
- Header: add `DuplicateSeq` to `ManifestIssueCode` (include/vesper/wal/manifest.hpp).
- Validation: track first file per `seq` during `validate_manifest()`; on encountering a duplicate, emit `DuplicateSeq` error with message `"duplicate seq: also in <first-file>"`.
- Leave `load_manifest()` unchanged aside from prior exception-free parsing.

## Testing
- Unit tests (tests/unit/wal_manifest_parse_test.cpp):
  - TID-WAL-MAN-DUP-SEQ-001: duplicate seq across different files → `DuplicateSeq` error
  - TID-WAL-MAN-DUP-SEQ-002: duplicate seq within same file → `DuplicateFile` error (existing)
  - TID-WAL-MAN-DUP-SEQ-003: 3 files with same seq → two `DuplicateSeq` errors
  - TID-WAL-MAN-DUP-SEQ-004: unique seq manifest passes

## Migration
- No file format changes. Existing valid manifests unaffected.
- If duplicates are discovered in legacy manifests, rectify by removing the offending entries or regenerating via `rebuild_manifest()`.

