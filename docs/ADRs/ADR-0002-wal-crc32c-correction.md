# ADR-0002: WAL CRC-32C Orientation Correction and Length Overflow Guard

- Status: Accepted
- Date: 2025-10-27
- Components: wal/frame.cpp, wal/io.cpp, tests/unit/wal_frame_test.cpp

## Context

The WAL frame format appends a CRC-32C (Castagnoli) checksum over the frame body `[magic||len||type||reserved||lsn||payload]`.

Two issues were identified in `src/wal/frame.cpp`:

1) CRC polynomial orientation bug
- The byte-wise reflected update algorithm was used with the standard (non-reversed) polynomial `0x1EDC6F41`. Correct reflected CRC-32C requires the reversed polynomial `0x82F63B78`.
- Consequence: checksums produced did not match standard CRC-32C values and would fail interop/verification with external tooling. Round-trip tests passed only because both encode and verify shared the same bug.

2) Length overflow risk
- `len = uint32_t(WAL_HEADER_SIZE + payload.size() + 4)` could truncate for payloads exceeding 2^32-1 minus header and CRC. The vector was allocated using the truncated `len` while the subsequent `memcpy` used the full `payload.size()`, leading to potential out-of-bounds write.

These violate Vesper serialization and reliability gates (serialization integrity, security/memory safety).

## Decision

- Correct CRC-32C orientation for reflected algorithm:
  - Generate the CRC table with the reversed polynomial `0x82F63B78` and compute checksums with the reflected update.
  - Introduce dual-acceptance during migration: `verify_crc32c()` first validates against the corrected CRC, then (if that fails) validates against the legacy orientation. This allows reading existing logs while writers emit only the corrected CRC.

- Add explicit length overflow guard for frame encoding:
  - Provide `encode_frame_expected(lsn, type, payload) -> expected<vector<uint8_t>, error>` that validates `payload.size() <= UINT32_MAX - WAL_HEADER_SIZE - 4` and returns `invalid_argument` on overflow.
  - Keep the existing `encode_frame(...) -> vector<uint8_t>` as a thin wrapper returning `{}` on error to avoid API break, while internal writers (WalWriter) use the expected-returning variant.

- Update `WalWriter::append` to use `encode_frame_expected` and propagate errors via `std::expected`.

## Rationale

- Aligns WAL with the standard CRC-32C (Castagnoli) reflected algorithm, enabling interoperability and external validation.
- The migration path avoids breaking existing data by accepting legacy checksums on read while ensuring all new frames are correct.
- Overflow guard eliminates a potential OOB write and follows Vesper’s explicit error model (no exceptions on hot paths, use `std::expected`).

## Consequences

- New frames produced after this change carry standard CRC-32C values (reflected, poly=0x82F63B78).
- Verification of legacy frames remains supported during migration (dual-accept in `verify_crc32c`).
- Writers now fail fast with a clear error on oversized payloads; downstream code is not exposed to truncated/unsafe frames.
- Minimal performance impact on verify: one extra CRC pass only when the first (correct) check fails.

## Compatibility & Migration

- Backward read compatibility: Yes (verify accepts both correct and legacy CRCs).
- Forward read compatibility: Old readers will reject new frames (expected); this is acceptable as long as all components are updated in lockstep or replay paths are upgraded first.
- API: Non-breaking; a new expected-returning encode function is introduced. Existing `encode_frame` remains for convenience/tests.

## Testing

- Unit tests added:
  - Known-answer CRC: `crc32c("123456789") == 0xE3069283` (standard reflected CRC-32C).
  - Overflow guard: `encode_frame_expected` rejects payloads exceeding 32-bit length capacity.
  - Migration: `verify_crc32c` accepts a frame whose CRC was computed using the legacy orientation.
- Existing round-trip tests continue to pass; fuzz target unaffected and continues exercising decode/verify.

## Implementation Notes

- New symbols:
  - `encode_frame_expected(...) -> expected<vector<uint8_t>, error>` (preferred writer API)
  - `CRC32C_TABLE_REFLECTED` (correct), `CRC32C_TABLE_LEGACY` (read-side only)
- `WalWriter::append` switched to the expected-returning variant and propagates errors.

## References

- Castagnoli CRC-32C polynomial: 0x1EDC6F41 (standard), 0x82F63B78 (reversed for reflected algorithm)
- Known-answer vector: `"123456789" -> 0xE3069283`
- Intel SSE4.2 CRC32C, RFCs and Wikipedia for CRC-32C (Castagnoli)

