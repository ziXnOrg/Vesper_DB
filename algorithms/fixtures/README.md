# Fixtures (Phase 2)

This directory will store golden inputs/outputs used by unit and property tests:
- distance_kernels/: small vectors with known L2/IP results
- pq/: codebooks, LUTs, encoded vectors
- wal/: valid/torn frames for recovery tests
- filters/: bitmap masks for typical predicates

Files should be small, textual (JSON/YAML/CSV) where possible, and deterministic. Seeds are documented in `../spec/tolerances.md`.

