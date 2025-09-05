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

## Build configuration and performance knobs

Platform-aware defaults are applied automatically by CMake. You can override any option at configure time.

- macOS defaults
  - VESPER_ENABLE_ACCELERATE=ON (links Apple Accelerate; enables vectorized L2 kernels)
  - VESPER_ENABLE_OPENMP=OFF (OpenMP requires Homebrew LLVM; optional)
  - VESPER_SERIALIZE_BASE_LAYER=ON (safer HNSW base-layer connect; can be turned OFF for throughput)
- Linux/UNIX (non-Apple) defaults
  - VESPER_ENABLE_OPENMP=ON (links OpenMP)
  - VESPER_ENABLE_ACCELERATE=OFF
  - VESPER_SERIALIZE_BASE_LAYER=ON

Common configure examples:
<augment_code_snippet mode="EXCERPT">
````bash
# Default Release build with platform defaults
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# macOS: disable Accelerate or try OpenMP (requires Homebrew LLVM/omp)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DVESPER_ENABLE_ACCELERATE=OFF -DVESPER_ENABLE_OPENMP=ON

# Linux: turn off OpenMP (optional) or experiment with base-layer parallelism
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DVESPER_ENABLE_OPENMP=OFF -DVESPER_SERIALIZE_BASE_LAYER=OFF
````
</augment_code_snippet>

Notes
- Base-layer serialization (VESPER_SERIALIZE_BASE_LAYER)
  - ON (default): serializes search+connect at level 0 for maximum connectivity stability
  - OFF: serializes only the search at level 0; connect_node() runs without the global graph lock for higher throughput
  - Connectivity target: ≥95% reachable in our tests with either mode; OFF can be faster but may contend at high thread counts
- OpenMP on macOS
  - Use Homebrew LLVM (clang++) and libomp if you want OpenMP; otherwise leave it OFF
- Accelerate (macOS)
  - When ON, distance kernels route to vDSP (sum of squares + dot) for L2^2

### Run tests and micro-benchmarks
<augment_code_snippet mode="EXCERPT">
````bash
cmake --build build --target test_hnsw_batch hnsw_connectivity_test -j
./build/test_hnsw_batch
./build/hnsw_connectivity_test
````
</augment_code_snippet>

The two test executables accept environment overrides:
- VESPER_NUM_THREADS: number of build threads (0=auto)
- VESPER_EFC: efConstruction for build (beam width)
- VESPER_EFC_UPPER: ef for upper layers when adaptive is enabled (0=auto)
- VESPER_ADAPTIVE_EF: set to 1/true to enable adaptive ef for upper layers

Example:
<augment_code_snippet mode="EXCERPT">
````bash
VESPER_NUM_THREADS=4 VESPER_EFC=150 VESPER_ADAPTIVE_EF=1 ./build/hnsw_connectivity_test
````
</augment_code_snippet>

### Benchmark matrix (optional)
We provide a small matrix runner to compare combinations of flags and thread counts. It rebuilds per configuration and prints a summary table.

- Direct run:
<augment_code_snippet mode="EXCERPT">
````bash
python3 tools/bench_matrix.py
````
</augment_code_snippet>

- Via CTest (if Python3 is available):
<augment_code_snippet mode="EXCERPT">
````bash
ctest -R hnsw_bench_matrix -V
````
</augment_code_snippet>

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

