# IVF-PQ v1.1 Metadata JSON Policy (Design Note)

Status: Draft (P0 for Phase 1)
Applies to: IVF-PQ v1.1 sectioned binary format; optional Metadata JSON section

## Purpose
Define strict, implementation-visible limits and forward-compat behavior for the optional metadata JSON section so that:
- Writers and loaders agree on constraints
- Fuzz targets can generate precise edge cases
- Docs and API helpers remain stable

## Encoding & Placement
- Encoding: UTF-8 JSON text; no BOM
- Placement: Single, optional dedicated section in the v1.1 container (after core mandatory sections); listed in trailer
- Length: Explicit section length in trailer; checked against maximum (below) prior to allocation/read

## Size Limit
- Maximum uncompressed JSON size: 64 KiB (65,536 bytes)
- Rationale: Keeps metadata bounded for mmap-friendly loads and avoids large heap allocations on parse
- Enforcement:
  - Writer rejects payloads > 64 KiB (hard error)
  - Loader rejects sections with length > 64 KiB (hard error)
  - Fuzz guidance: generate cases at 0, 1, 64 KiB-1, 64 KiB, 64 KiB+1, and very large declared lengths

## Forward-Compatibility Policy
- Unknown keys: Must be tolerated and preserved by loaders that expose pass-through metadata
- Key naming: snake_case recommended; stable keys should be documented in API docs
- Ordering: JSON object key order is not significant; implementations must not rely on it
- Types: Standard JSON types only; avoid NaN/Inf; numbers are IEEE-754 doubles in parsers
- Canonicalization: Not required; round-trip equality means semantic equivalence, not byte-for-byte
- Schema evolution: Optional schema hook may be provided by applications; core library remains permissive

## Validation & Error Handling
- Parse failures: Hard error; fail to load index if metadata cannot be parsed as valid UTF-8 JSON
- Size violations: Hard error at writer and loader as above
- Depth/complexity caps (defensive):
  - Recommended max nesting depth: 64
  - Recommended max total keys: 4,096
  - Parsers may enforce these limits to avoid pathological inputs; violations are hard errors

## API Helper Requirements
- Provide helper to set/get metadata as a JSON string and as a structured object (where available)
- Enforce 64 KiB limit at set-time; expose error with clear message
- Optional schema check hook: callable that validates object; failure aborts set/save
- Round-trip tests: save → load → semantic equality (object compare) including unknown keys

## Fuzz Targets (Guidance)
Generate and mutate metadata sections to cover:
- Size edges: 0, 1, 64KiB-1, 64KiB, 64KiB+1, huge declared length with small body, small length with larger body (truncation)
- Structural: deeply nested objects/arrays up to depth 64; many keys (~4096); empty objects; nulls
- Unicode: valid multi-byte UTF-8, combining characters, emoji; invalid byte sequences
- Keys: duplicate keys; unknown keys; long keys; non-ASCII keys
- Types: large integers, precise decimals, booleans, nulls, arrays of mixed types
- Corruptions: flipped bytes in trailer length; truncated sections; random noise

## Compatibility & Docs
- This policy complements docs/IVFPQ_Serialization_v11.md
- API docs must state:
  - 64 KiB max uncompressed JSON
  - Unknown keys tolerated; schema hook optional
  - Error behavior on parse/size/depth violations
  - Round-trip semantics (object-level equality)

## Acceptance (Phase 1)
- Tests: Round-trip success within limits; failure on limit violations; unknown-keys preserved
- Fuzz: Corpus includes metadata-bearing cases per guidance; loader is crash-free with <1% false-accept on 10k random corruptions (overall)
- Docs: API reference updated; examples of setting metadata, using optional schema hook

