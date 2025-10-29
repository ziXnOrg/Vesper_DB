# ADR-0005: WAL Manifest Filename Validation (Defense-in-Depth)

## Status
Accepted — 2025-10-27

## Context
The WAL manifest contains entries of the form `file=...` that are later combined as `dir / file` (e.g., in `src/wal/io.cpp`). Previously, manifest filenames were not constrained beyond basic control-character checks. An attacker-controlled or corrupted manifest could include path traversal or absolute paths (e.g., `../x`, `..\\x`, `/etc/passwd`, `C:\\Windows\\...`, `\\\\server\\share\\x`), risking reads outside the WAL directory. Even though the manifest is an internal artifact, Vesper enforces defense-in-depth.

## Decision
- Enforce strict filename validation during parsing (`load_manifest`) and validation (`validate_manifest`):
  - Accept only filenames matching the exact pattern: `^wal-[0-9]{8}\.log$`.
  - Reject any of the following:
    - Path separators: `/` or `\\` anywhere in the name
    - Parent directory references: `..` at the beginning
    - Absolute roots: names starting with `/` or `\\`
    - Windows drive prefixes: `[A-Za-z]:`
    - UNC paths: `\\\\` at the start
- On `load_manifest()` violation: return `std::unexpected(error{error_code::data_integrity, ...})` with actionable context.
- On `validate_manifest()` violation: record `Severity::Error` with `ManifestIssueCode::BadHeader` and message `invalid filename` including the offending line number.
- No schema change (manifest v1 remains). This is a strict subset of allowed names.

## Consequences
- Recovery paths cannot be tricked into reading outside the WAL directory via manifest filenames.
- Legitimate WAL segment names following `wal-########.log` continue to work unchanged.
- Invalid or corrupted manifests fail fast with explicit, actionable diagnostics.

## Alternatives Considered
- Whitelisting by glob without rejecting absolute/UNC forms — rejected; insufficient for security.
- Normalizing via `std::filesystem::weakly_canonical` — rejected; normalization still exposes risk and is slower on hot paths.
- Allowing subdirectories under WAL dir — rejected; not required, complicates security posture.

## Testing
- Unit tests (tests/unit/wal_manifest_parse_test.cpp):
  - TID-WAL-MAN-PATH-001: Reject `../wal-00000001.log` (no throw)
  - TID-WAL-MAN-PATH-002: Reject `..\\wal-00000001.log` (no throw)
  - TID-WAL-MAN-PATH-003: Reject absolute POSIX `/etc/passwd` (no throw)
  - TID-WAL-MAN-PATH-004: Reject absolute Windows `C:\\Windows\\System32\\file.log` (no throw)
  - TID-WAL-MAN-PATH-005: Reject UNC `\\\\server\\share\\file.log` (no throw)
  - TID-WAL-MAN-PATH-006: Reject mixed separators `wal-00000001.log/other` (no throw)
  - TID-WAL-MAN-PATH-007: Accept valid `wal-00000042.log`
- Fuzz tests (tests/fuzz/wal_manifest_fuzz.cpp): continue to exercise load/validate/enforce without try/catch to enforce no-throw contract.

## Migration
- No migration required. Existing valid manifests remain valid. Manifests containing disallowed filenames are now rejected with explicit errors at parse/validate time; operators should regenerate affected manifests using the standard WAL publishing flow.

