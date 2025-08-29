# Vesper — Crash-Safe, Embeddable Vector Search Engine (C++20, CPU-only)

Vesper brings advanced vector search to wherever your data lives—offering uncompromising speed, durability, and privacy for on-device and air-gapped environments.

Vesper is a single-library, embeddable vector search engine purpose-built for edge, offline, and privacy-sensitive scenarios. Achieve ultra-low latency approximate nearest neighbor (ANN) search with blazing-fast metadata filters and deterministic persistence—no GPUs, cloud, or network IO required. Designed for regulated or crash-critical deployments, Vesper enables recoverable, predictable, and secure AI search on commodity CPUs.

---

## Key Features

- **Pluggable Index Families per Collection**  
  - IVF-PQ/OPQ: Compact, SSD-friendly
  - HNSW: In-memory hot segments
  - Disk-graph: DiskANN-style, billion-scale
- **Early, Fast Metadata Filtering:** Roaring bitmaps (AND/intersection during traversal)
- **Crash-Safety:** Checksummed WAL, atomic snapshot publishing, strict fsync and parent-dir persistence discipline
- **SIMD-Accelerated Kernels:** AVX2/AVX-512 for top performance, with scalar fallback and pmr arenas
- **Optional At-Rest Encryption:** XChaCha20-Poly1305 (default), AES-GCM (FIPS-friendly mode)
- **Portable & Flexible:** Pure C++20, runs on Linux, macOS, Windows. Stable C ABI for bindings—ideal for embedding or extension

---

## Who Should Use Vesper

Choose Vesper if you need:
- **Edge, Embedded, or Offline AI/RAG workflows:** Run powerful similarity and hybrid search anywhere, without network or cloud dependencies
- **Crash-Safe and Durable Retrieval:** For medical, IoT, field robotics, or critical business apps that cannot afford data loss
- **Compliant, Privacy-First Search:** Vesper’s local-only storage and optional encryption support strict regulatory requirements and data sovereignty
- **Low-Latency AI on Commodity Hardware:** Get <20ms tail latency with no GPU, simply by linking the library

Vesper is built for industries, researchers, and makers who need trustworthy vector search beyond the cloud.

---

## Supported Platforms and CPU Baseline

- **OS:** Linux, macOS, Windows  
- **Compilers (minimums):** GCC 12+, Clang 15+/AppleClang 15+, MSVC 19.36+  
- **CPU:** Baseline AVX2 assumed for best performance; auto runtime dispatch with scalar fallback

---

## Quick Start

See `docs/SETUP.md` for prerequisites, build instructions, and micro-benchmarks.  
Sample schemas/examples are in `experiments/`; validation workflow in `experiments/VALIDATION.md`.

---

## Technical Architecture

- **High-Level Design, Data Model, and Performance:** See [`blueprint.md`](./blueprint.md) for details and diagrams
- **API Notes:** Out-of-code documentation in [`api-notes.md`](./api-notes.md)
- **Performance Targets (Initial):**
    - Latency: `p50 1–3 ms`, `p99 10–20 ms` (128–1536D)
    - Quality: `recall@10 ≈ 0.95` (tunable)
    - Recovery: seconds to minutes (WAL replay / snapshot restore)

Full details in: `blueprint.md`, `benchmark-plan.md`

---

## Safety Stance and Privacy

- **No network IO by default:** Library operates only on local files
- **Optional, strong encryption at rest:** With strict fsync/rename for durability
- **Threat Model:** See [`threat-model.md`](./threat-model.md) for assets, adversaries, controls, and validation

---

## License

Apache License 2.0 — see [`LICENSE`](./LICENSE) for details.

---

**Questions?**  
Open an issue or use discussions to get involved.
