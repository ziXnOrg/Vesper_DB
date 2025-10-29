# ADR-0007: WAL manifest LSN range validation

Status: Accepted
Date: 2025-10-27

## Context
Each WAL manifest entry records LSN range bounds via start_lsn, first_lsn, and end_lsn. Previously, validate_manifest() did not enforce intra-entry invariants (start_lsn <= first_lsn <= end_lsn) nor cross-entry constraints (no overlapping ranges across increasing seq, end_lsn monotonicity). Invalid LSNs risk recovery corruption or ambiguous replay bounds.

## Decision
- Enforce intra-entry invariants in validate_manifest(): start_lsn <= first_lsn <= end_lsn. Violations are Severity::Error with ManifestIssueCode::LsnInvalid.
- Enforce cross-entry constraints across increasing seq order:
  - No overlaps: current.start_lsn must be strictly greater than previous.end_lsn. Violations are Severity::Error with ManifestIssueCode::LsnOverlap.
  - End monotonicity: current.end_lsn must be >= previous.end_lsn. Violations are Severity::Error with ManifestIssueCode::LsnOrder.
  - Gaps: when current.start_lsn > previous.end_lsn + 1, emit Severity::Warning with ManifestIssueCode::LsnGap. Gaps are permitted to allow retention/snapshot pruning but surfaced to operators.
- No-throw contract preserved; advisory validation returns ManifestValidation with issues list.

## Consequences
- Ambiguous or invalid LSN ranges are rejected deterministically (ok=false) and recovery can fail-closed on Severity::Error.
- Gaps are allowed but visible for operational awareness.
- No on-disk format change; valid manifests unaffected.

## Alternatives considered
1) Treat LSN gaps as errors. Rejected: legitimate scenarios (snapshots/retention) can create gaps.
2) Allow overlaps with best-effort resolution. Rejected: introduces non-determinism and potential double-replay or data loss.

## Implementation
- enum ManifestIssueCode extended with: LsnInvalid, LsnOverlap, LsnOrder, LsnGap (include/vesper/wal/manifest.hpp).
- validate_manifest() updated to check intra- and cross-entry rules and to populate issues with actionable messages including seq/file context.

## Testing
- Unit tests (tests/unit/wal_manifest_parse_test.cpp):
  - TID-WAL-MAN-LSN-001: start_lsn > first_lsn → LsnInvalid (Error)
  - TID-WAL-MAN-LSN-002: first_lsn > end_lsn → LsnInvalid (Error)
  - TID-WAL-MAN-LSN-003: start_lsn > end_lsn → LsnInvalid (Error)
  - TID-WAL-MAN-LSN-004: start==first==end passes
  - TID-WAL-MAN-LSN-005: overlap between consecutive entries → LsnOverlap (Error)
  - TID-WAL-MAN-LSN-006: end_lsn decreasing across entries → LsnOrder (Error)
  - TID-WAL-MAN-LSN-007: gap between entries → LsnGap (Warning), ok=true
  - TID-WAL-MAN-LSN-008: valid cross-entry ordering passes

## Migration
- No file format changes. Existing valid manifests unaffected.
- Legacy manifests with invalid LSN ranges will be detected; recovery should treat Severity::Error as fail-closed and require remediation or manifest rebuild.

