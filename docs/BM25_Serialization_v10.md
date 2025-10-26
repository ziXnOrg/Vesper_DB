# BM25 Serialization Format v1.0-dev (Vesper, pre-release)

Status: Pre-release (read/write), format may change
Compatibility: No backward-compatibility guarantees until first stable release; version fields retained for future-proofing
Checksum: FNV-1a 64-bit over all bytes prior to trailer

## Overview
Binary, portable, deterministic format for persisting BM25 sparse index state.
We serialize only the canonical state (vocabulary, documents, parameters, and
aggregate stats) and deterministically rebuild inverted lists on load.

## Layout
```
[ MAGIC (8) = "BM25v100" ]
[ VERSION (2+2) = major=1, minor=0 ]

# Parameters
[ k1 (float32) ]
[ b  (float32) ]
[ lowercase (u8) ]
[ remove_stopwords (u8) ]
[ min_term_length (u32) ]
[ max_term_length (u32) ]

# Counts / globals
[ vocab_size (u32) ]
[ num_docs (u64) ]
[ total_tokens (usize; 64-bit on all supported platforms) ]
[ avg_doc_length (float32) ]

# Vocabulary (deterministic order by term_id)
repeat vocab_size times:
  [ term_len (u32) ] [ term_bytes (term_len) ]

# Documents (deterministic order by doc_id asc)
repeat num_docs times:
  [ doc_id (u64) ]
  [ length (u32) ]
  [ nnz (u32) ]
  repeat nnz times:
    [ term_id (u32) ] [ tf_value (float32) ]

# Trailer
[ 'C','H','K','S' ] [ checksum_fnv1a64 (u64) ]
```

Notes:
- All integers are little-endian.
- Strings are raw UTF-8 bytes with length prefix.
- We intentionally do not serialize the inverted index; it is rebuilt losslessly
  from document sparse vectors during load. This reduces file size and removes
  duplication.

## Determinism
- Vocabulary is emitted in increasing `term_id` order.
- Documents are emitted in increasing `doc_id` order.
- Within each document, `(term_id, tf)` pairs are sorted by `term_id`.
These rules guarantee bit-for-bit identical files for identical index state.

## Integrity
- Checksum uses 64-bit FNV-1a starting at offset `1469598103934665603` and prime
  `1099511628211`. We hash every byte from the start of file up to (but not
  including) the trailer. Loader validates and fails-closed on mismatch.

## Versioning
- v1.0-dev is a pre-release development format. The schema may change without backward compatibility until the first stable release.
- Magic and version fields are included to enable future evolution; the current loader only accepts (1,0) and is not required to accept older dev snapshots.
- Once a stable 1.x is declared, backward compatibility will be enforced going forward.

## Scoring and IDF definition (pre-release)
- We currently use a BM25 variant with non-negative IDF:
  - IDF(t) = log(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
- Document length normalization follows the standard BM25 form:
  - norm(d) = (1 - b) + b * (|d| / avg_doc_length)
- Term contribution per query term t is:
  - score_t(d) = IDF(t) * ((tf(t,d) * (k1 + 1)) / (tf(t,d) + k1 * norm(d)))
- Overall score is the sum over query terms. Parameters k1 and b come from BM25Params.
- Rationale: the +1 inside the logarithm avoids negative IDF when df > N/2 and stabilizes ranking. This choice is v1.0-dev and may change prior to the first stable release.

## Rationale
- Simplicity: The minimal state suffices to reconstruct search behavior.
- Portability: Only fixed-size arithmetic types and length-prefixed UTF-8.
- Robustness: Checksum-guarded; loader validates magic/version and fails-closed.
- Determinism: Stable iteration orders ensure reproducible artifacts.

## Reference Implementation
- Writer/reader implemented in `src/index/bm25.cpp` using the same FNV-1a helper
  pattern as IVF-PQ. See `docs/IVFPQ_Serialization_v11.md` for prior art.


