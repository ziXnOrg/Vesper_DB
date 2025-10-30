# IMPL: FastScanPq trained-state preconditions (Task 21)

Thought framework: First Principles (with targeted ReAct lookups)

## Scope
Add checked variants to enforce trained-state preconditions for FastScanPq without penalizing hot paths:
- encode_checked(...)->expected<void,error>
- decode_checked(...)->expected<void,error>
- compute_lookup_tables_checked(...)->expected<AlignedCentroidBuffer,error>

Also: document \pre on existing fast methods and (optionally) add debug-only assertions.

## Modules/Files
- include/vesper/index/pq_fastscan.hpp (new APIs + Doxygen updates)
- src/index/pq_fastscan.cpp (implement checked variants; optional asserts)
- tests/unit/pq_fastscan_preconditions_test.cpp (new tests)
- docs/Implementation/AUDIT_2025_Codebase_Review.md (mark item resolved)

## Invariants (First Principles)
- trained_ == true after successful train() or import_pretrained(); false otherwise
- codebooks_ != nullptr iff trained_ is true
- dsub_ > 0 when trained (dim divisible by m)

## Error model
- Untrained usage of checked variants → error_code::precondition_failed
- Successful trained usage → expected has_value

## Performance expectations
- Checked variants: single predictable branch on trained_
- Existing void/fast methods remain unchanged (no new overhead)

## API Signatures
- encode_checked(const float* data, size_t n, uint8_t* codes) const -> expected<void, core::error>
- decode_checked(const uint8_t* codes, size_t n, float* data) const -> expected<void, core::error>
- compute_lookup_tables_checked(const float* query) const -> expected<AlignedCentroidBuffer, core::error>

## Doxygen (contract-based)
For both fast and checked methods:
- \brief concise, \param with shapes, \pre trained, \thread_safety, \complexity, \see train(), import_pretrained(), is_trained()

## Optional debug assertions
- In fast methods (Debug only): assert(trained_ && codebooks_)

## Testing plan (TDD)
- Untrained → encode_checked/decode_checked/compute_lookup_tables_checked return precondition_failed
- After train() → all checked variants succeed
- After import_pretrained() → all checked variants succeed

## Risks & Mitigations
- ABI surface growth: additive only (no signature changes)
- Perf: fast paths untouched; checked variants branch once

## Out-of-scope
- Converting fast methods to expected-return (future discussion)
- Adding compute_distances_checked (optional in future)

