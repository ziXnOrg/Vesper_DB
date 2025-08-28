# Vesper — Crash‑Safe, Embeddable Vector Search Engine (C++20, CPU‑only)

Vesper is a single‑library, embeddable vector search engine for on‑device and air‑gapped scenarios. It delivers ultra‑low latency approximate nearest neighbor (ANN) search with metadata filters, deterministic persistence (WAL + snapshots), and predictable performance on commodity CPUs—no GPUs or network IO required by default.

## Key features
- Pluggable index families per collection: IVF‑PQ/OPQ (compact, SSD‑friendly), HNSW (in‑memory hot segments), and Disk‑graph (DiskANN‑style, billion‑scale)
- Early, fast metadata filtering with Roaring bitmaps (AND/intersection during traversal)
- Crash‑safety: checksummed WAL, atomic snapshot publish, parent‑dir fsync discipline
- SIMD‑accelerated distance kernels (AVX2/AVX‑512) with scalar fallback and pmr arenas
- Optional at‑rest encryption (XChaCha20‑Poly1305; AES‑GCM for FIPS‑friendly mode)
- Portable C++20, Linux/macOS/Windows, stable C ABI for bindings

## Supported platforms and CPU baseline
- OS: Linux, macOS, Windows
- Compilers (minimums): GCC 12+, Clang 15+/AppleClang 15+, MSVC 19.36+
- CPU: Baseline AVX2 assumed for best performance; runtime feature‑gated dispatch with scalar fallback

## Quick start
See docs/SETUP.md for prerequisites, build instructions, and how to run tests and micro‑benchmarks.

Phase 0 schemas and examples live under `experiments/` (see `experiments/VALIDATION.md`).

## Architecture overview
High‑level design, data model, and performance targets are specified in blueprint.md. Start here for a deep technical tour and diagrams.
- Technical Blueprint: ./blueprint.md
- API Notes (no code): ./api-notes.md

## Performance targets (initial)
- Latency: p50 1–3 ms, p99 10–20 ms (128–1536D)
- Quality: recall@10 ≈ 0.95 (tunable)
- Recovery: seconds to minutes (WAL replay / snapshot restore)
For full details, see blueprint.md and benchmark-plan.md.

## Safety stance and privacy
- No network IO by default; the library operates on local files only
- Optional encryption at rest; strict fsync/rename discipline for durability
- See threat-model.md for assets, adversaries, controls, and validation

## Roadmap and development process
- Project source of truth: ./blueprint.md
- Prompt Blueprint (methodology reference for prompts/evals/safety): ./prompt-blueprint.md
- Execution plan (phases, gates, milestones): ./prompt-dev-roadmap.md

## Contributing
We use a deterministic, tests‑first, prompt‑first workflow (temperature=0.0, top_p=1.0, fixed seed). Please read CONTRIBUTING.md for branching/PR flow, local checks, and schema/CI gates.

## License
Apache License 2.0 — see LICENSE for details.

