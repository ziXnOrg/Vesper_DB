# IVF-PQ v1.1 Serialization Format and Usage Guide

This document describes the binary serialization format used by Vesper's IVF-PQ index (v1.1), the versioning and compatibility policy, and practical usage examples for saving/loading indexes, including optional metadata.

## Overview

v1.1 introduces a sectioned file layout to enable optional compression, memory-mapped loading, and targeted parsing. The layout:

- File header with magic and version identifiers
- Fixed header fields describing core parameters
- Zero or more sections, each with a small header and payload
- Trailer carrying a file-wide checksum

Checksums are FNV-1a 64-bit. Optional Zstandard (Zstd) compression can be applied per section when enabled at build-time and via environment variable.

## Magic and Version

- Magic (8 bytes): ASCII `IVFPQv11`
- Version: 16-bit major and minor (currently: major=1, minor=1)

Compatibility policy:
- Readers use `(major, minor)` to determine how to parse and whether to reject.
- Minor bumps (1.x) add fields/sections in a backward-compatible way.
- Major bumps (2.x) may break compatibility.

## Header Layout (v1.1)

Following the magic and version, the file stores:
- flags (uint32): bitfield
  - bit 0: OPQ rotation present
  - bit 1: RaBitQ in use (reserved)
- dim (uint32)
- nlist (uint32)
- m (uint32)
- nbits (uint32)
- dsub (uint32)
- nvec (uint64)
- code_size (uint32) — equals `m`
- build_ts (uint64) — UNIX timestamp of build time
- meta_len (uint32) — reserved for future inline metadata (v1.1 keeps this 0)

All multi-byte integers are little-endian.

## Sections

Each section is preceded by a fixed header. Section header structure:
- type (uint32)
- uncompressed_size (uint64) — `unc`
- compressed_size (uint64) — `comp` (equals `unc` when not compressed)
- shash (uint64) — FNV-1a of the uncompressed payload

Known section `type` values (current):
- 1: Centroids (coarse quantizer)
- 2: PQ codebooks
- 3: Inverted lists (IDs + codes)
- 4: OPQ rotation (optional)
- 5: Metadata JSON (optional)

When compression is enabled and effective for a section, `comp < unc` and the payload holds Zstd-compressed bytes. Otherwise, payload is stored raw and `comp == unc`.

## Checksums and Trailer

- Per-section checksum `shash` validates the uncompressed payload integrity.
- A trailer stores a file-wide FNV-1a checksum computed over all bytes preceding the trailer. This ensures end-to-end integrity to quickly reject truncated or corrupted files.

Loaders verify:
- Magic/version
- Basic header sanity
- Section boundaries and `comp/unc` consistency
- `shash` for each section
- Trailer checksum

## Versioning and Backward Compatibility

- Writers default to legacy v1.0 (monolithic) unless explicitly requested to write v1.1 by environment configuration.
- Readers accept both v1.0 and v1.1, with additional defensive checks in v1.1.
- Future 1.x versions may introduce new optional sections. Unknown sections are skipped safely (length is known from the header).
- The presence of OPQ is signaled via the `flags` bit; RaBitQ is reserved.

Recommended policy when evolving the format:
- Add only optional fields/sections in minor updates.
- Make critical new requirements part of a major bump.
- Keep readers robust to unknown sections and conservative on resource allocation.

## Environment Variables

- VESPER_IVFPQ_SAVE_V11=1 — write v1.1 sectioned format (default is v1.0)
- VESPER_IVFPQ_ZSTD_LEVEL=1..3 — enable Zstd compression for sections (if built with Zstd)
- VESPER_IVFPQ_LOAD_MMAP=1 — prefer memory-mapped loading when supported by platform

## CLI Usage Examples (save/load)

Although most users use the C++ API, you can exercise save/load paths with the provided tools.

- Save an index in v1.1 format without Zstd:

```bash
# build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target ivfpq_persist_bench

# run save with v1.1
export VESPER_IVFPQ_SAVE_V11=1
./build/ivfpq_persist_bench --mode save --out ./index_dir --nlist 2048 --dim 128 --n 100000
```

- Load an index with memory-mapped I/O:

```bash
export VESPER_IVFPQ_LOAD_MMAP=1
./build/ivfpq_persist_bench --mode load --in ./index_dir
```

- Enable Zstd (if compiled with VESPER_WITH_ZSTD=ON):

```bash
export VESPER_IVFPQ_SAVE_V11=1
export VESPER_IVFPQ_ZSTD_LEVEL=3
./build/ivfpq_persist_bench --mode save --out ./index_dir
```

## Public API Examples

```cpp
#include <vesper/index/ivf_pq.hpp>
using namespace vesper::index;

// Training
IvfPqIndex index;
IvfPqTrainParams p; p.nlist = 2048; p.m = 16; p.use_opq = true;
auto ts = index.train(train_data, dim, n_train, p);

// Optional metadata JSON
index.set_metadata_json(R"({\"dataset\":\"sift1m\",\"owner\":\"team\"})");

// Add vectors
auto ok = index.add(ids, data, n);

// Save (path is a directory; file will be path/ivfpq.bin)
(void)index.save("./index_dir");

// Load
auto loaded = IvfPqIndex::load("./index_dir");
std::string meta = loaded->get_metadata_json();
```

Coarse assigner defaults and ANN toggles
- Default coarse quantizer: KDTree (exact). This is the recommended default for correctness.
- `use_centroid_ann = true` by default. For KDTree this enables the KD assignment path (exact). HNSW/Projection ANN parameters only apply when those assigners are selected.
- HNSW defaults (if selected): `ef_search=96`, `ef_construction=200`, `M=16`, `refine_k=96`.



Notes:
- `save(path)` writes `ivfpq.bin` under `path` (directory). The method ensures the directory exists.
- `set_metadata_json` enforces a size limit of 1 MiB and is serialized as section type 5 in v1.1.
- The index transparently handles v1.0 and v1.1 on load.

