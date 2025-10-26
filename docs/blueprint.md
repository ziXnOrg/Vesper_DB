
# Vesper — A Crash‑Safe, Embeddable Vector Search Engine (C++20, CPU‑only)

**Revision:** v0.1 (Technical Blueprint)  
**Authors:** ShaiKKO (Product/Architecture), *[You]* (Research & Systems Design)  
**Target stack:** C++20, portable (Linux/macOS/Windows), POSIX I/O + `mmap`, no specialized hardware required (CPU‑only).  
**Primary goals:** ultra‑low latency top‑k ANN search with metadata filters, deterministic persistence & recovery (WAL + snapshots), small memory footprint, and predictable performance on commodity CPUs.

---

## 0) Executive Summary

Vesper is an *embedded* vector search engine designed for **on‑device** and **air‑gapped** scenarios: laptops, workstations, edge boxes, servers without GPUs, or compliance‑sensitive environments. It ships as a single C++20 library that links into host applications. The system emphasizes **crash‑safety** (write‑ahead logging + copy‑on‑write snapshots), **deterministic performance** (NUMA‑aware memory, SIMD‑accelerated distance computation, PQ/OPQ compression), and **filtered ANN** (Roaring bitmaps + per‑segment filter indexes).

The index tier supports three *pluggable* index families (selectable per collection):

1. **IVF‑PQ/OPQ** (coarse quantizer + compressed codes) for compact, scalable disk‑resident retrieval with fast *asymmetric distance computation (ADC)*.
2. **HNSW (in‑memory) with PQ‑coded bases** for hot/real‑time segments, high‑recall low‑latency search, and small working sets.
3. **Disk‑oriented flat graph (Vamana/DiskANN‑style)** for very large indices with **SSD‑resident** neighbor lists (RAM used as cache), but still CPU‑only.  

A *log‑structured* storage layer (mutable → immutable segments) provides fast ingestion, *crash‑consistent* compaction, and *atomic cut‑over* via `rename()`; a *WAL* guarantees durability, while *snapshots* give fast recovery and consistent point‑in‑time reads.

> **Why this niche?** Most vector DBs target server deployments with optional GPUs and relaxed crash semantics managed by a remote store. For **on‑device RAG**, compliant desktops, or air‑gapped labs, you need an embeddable engine with **predictable CPU performance**, **crash‑safety**, and **no external dependencies**. Vesper fills that gap.

---

## 1) Non‑Goals

* Distributed clustering/sharding (can be added later); this blueprint scopes a single‑node embedded library.
* GPU acceleration (later optional backends).  
* SQL query engine; we expose a C++ API that integrates with host apps or bindings.

---

## 2) Data Model & Capabilities

**Entity:** `(id, vector<float32 | float16 | int8>, metadata)`  
* `id`: 64‑bit integer (user‑supplied or auto).  
* `vector`: dimension `d` (64–4096 typical; support for >4k via blocking).  
* `metadata`: small typed columns (bool/int/float/string/enum/timestamp) + key‑value tags.  
* **Filters:** boolean expressions on metadata; accelerated by **Roaring bitmaps** per field and per segment, plus optional columnar dictionaries for string/enum.  
* **Distance metrics:** L2 (squared Euclidean), Inner‑Product (IP). Cosine handled by pre‑normalization (cosine == IP on L2‑normalized vectors).  
* **Mutability:** upserts, hard deletes (tombstones), *incremental* compaction (background, coroutine‑driven).  
* **Encryption at rest (optional):** stream‑oriented AEAD (XChaCha20‑Poly1305) with per‑file nonces and authenticated headers; envelope keying pluggable.

---

## 3) Performance Targets (Initial)

* P50 ≤ **1–3 ms**, P99 ≤ **10–20 ms** on 128–1536‑D embeddings (SIFT1M/SBERT‑like) on 8–16 core CPUs, **recall@10 ≈ 0.95** (tunable).  
* Build/ingest ≥ **50–200k vectors/s** (CPU‑bound; dataset dependent).  
* Index size: 10^6–10^9 vectors, scaling by choosing IVF‑PQ or DiskANN‑style graphs.  
* Crash recovery time: seconds (WAL replay) to minutes (rare full snapshot restore).

---

## 4) High‑Level Architecture

```
+------------------+      +------------------+
|  Host App        |      |  Tools           |
|  (C++/FFI)       |      |  (trainer, CLI)  |
+---------+--------+      +---------+--------+
          |                          |
          v                          v
+---------+--------------------------+----------------------+
|                 VESPER LIB (C++20)                         |
|  Query Planner  |  Index APIs  | Storage |  Crypto | Tele |
|                 |              |  (WAL)  | (AEAD)  |       |
+--+---------+----+----+----+----+----+----+----+-----+-----+
   |         |         |         |         |          |
   v         v         v         v         v          v
  IVF-PQ   HNSW     Disk-Graph  Filters   Snapshots   Metrics
 (ADC)     (RAM)      (SSD)     (Roaring)  (COW)      (perf)
```

### 4.1 Components

* **Query Planner:** chooses index family per collection, configures `efSearch / beam / nprobe`, switches to brute‑force if tiny postings, fuses **filters** as early pruning (bitmap AND) before and during candidate expansion.  
* **Index APIs:** uniform `search`, `insert`, `delete`, `seal_segment`, `compact`, `snapshot`, `recover`.  
* **Storage Layer:** *append‑only WAL* + *immutable segment files*. *Atomic* publish via `rename()`; *fsync/fdatasync()* on WAL and parent directory for crash‑safety.  
* **Memory:** `std::pmr` arenas for thread‑local scratch (distance tables, heaps, bitsets), NUMA‑aware pools; explicit cache line alignment (64B) to avoid false sharing.  
* **SIMD:** AVX2/AVX‑512 FMA distance kernels; int8 codepaths for PQ tables; software prefetch and cache blocking.

---

## 5) Index Families & Search Algorithms

### 5.1 IVF‑PQ / OPQ (Compact & Fast)

* **Idea:** train a coarse quantizer (k‑means or hierarchical balanced k‑means) to partition space into `nlist` cells; encode vectors with **Product Quantization (PQ)** codes (`m` subquantizers, `nbits` each). Use **Asymmetric Distance Computation (ADC)** at query time: compute lookup tables between the query’s subvectors and sub‑codebooks, then sum table entries to estimate distances without decoding. **OPQ** learns a rotation to reduce quantization error.  
* **When to choose:** space‑efficient indexes (≤ 16–32 bytes/vector), SSD‑resident, predictable latency.  
* **Search:** normalize for cosine→IP equivalence (if needed), select *nprobe* closest coarse cells, intersect with filter bitmaps to produce candidates, **ADC** with cached tables, then exact re‑rank on a short shortlist (optional).  
* **References:** Jégou et al., *Product Quantization for NN* (TPAMI 2011); Ge et al., *Optimized PQ (OPQ)* (CVPR/PAMI 2013).

### 5.2 HNSW (In‑Memory, Hot Segments)

* **Idea:** a multi‑layer proximity graph; greedy search from top layer, beam expansion with `efSearch`, per‑node bounded out‑degree `M`, `efConstruction` during build. Excellent recall/latency for RAM‑resident “hot” shards.  
* **Search:** filtered candidates enforced by bitmap checks when expanding neighbors; maintain visited set; reuse per‑thread heaps.  
* **References:** Malkov & Yashunin, *HNSW* (2016/2020).

### 5.3 Disk‑Oriented Flat Graph (Vamana / DiskANN‑style)

* **Idea:** build a single‑layer proximity graph with **RobustPrune** and a second pass for longer‑range edges; store neighbor lists on SSD; keep a small RAM cache (pinned medoids, entry points, PQ codebooks, Bloom of live ids). This yields billion‑scale indexes on a single node with limited RAM.  
* **Search:** SSD‑aware ordered adjacency fetch, beam expansion, PQ‑assisted distance estimation; *Filtered‑DiskANN* techniques adapt traversal to metadata predicates.  
* **References:** Subramanya et al., *DiskANN* (NeurIPS’19); Gollapudi et al., *Filtered‑DiskANN* (SIGMOD’23).

> Choice is per collection & even per segment: e.g., hot mutable segment = **HNSW**; sealed cold segments = **IVF‑PQ** for compactness or **DiskANN‑style** for scale.

---

## 6) Storage Engine, Persistence & Crash‑Safety

### 6.1 Files & Layout

**Directory structure (per collection):**

```
/vesper/collections/<name>/
  wal/0000000001.log, 0000000002.log, ...
  seg/<segid>-manifest.json
  seg/<segid>.idx    (index structure: IVF/HNSW/Vamana)
  seg/<segid>.vec    (raw or PQ-coded vectors)
  seg/<segid>.meta   (columnar metadata + roaring bitmaps)
  seg/<segid>.map    (mmap hint / bloom / id map)
  snapshots/<ts>-<seq>.snap/
    manifest, files...
```

*Write path:* append change records to `WAL`, update in‑memory mutable segment; flush checkpoints; on seal/compact, emit new immutable segment files to a temp path, **fsync** each, then *atomic publish* via `rename()` and **fsync parent directory**. This guarantees crash‑consistency.  

*Key ops:* `open(O_DIRECT|O_CLOEXEC)` where helpful, `posix_fallocate()` to avoid ENOSPC mid‑write, `mmap()` with read‑mostly flags and `MADV_WILLNEED` prefetch during warmup.  

### 6.2 WAL Format (durable, verifiable)

* Framing: 32‑bit little‑endian length, 32‑bit type, 64‑bit LSN (monotonic), payload, 32‑bit CRC32C or xxHash64 checksum.  
* Types: `Insert{id, vec, meta}`, `Delete{id}`, `Commit{seq}`, `Barrier`, `TruncateTo{seq}`.  
* Flush policy: group commit; `fdatasync()` on commit; WAL segment rollover at size/age thresholds.  
* Recovery: scan & verify checksums; apply idempotently to reconstruct the latest mutable segment; atomically load the latest published snapshot + segments.

### 6.3 Snapshots & Compaction

* **Snapshot:** consistent frozen view of live segments + tombstones; stored under `/snapshots/<ts>-<seq>`; published by atomic `rename()`.  
* **Compaction:** LSM‑like: union several immutable segments, merge tombstones, rebuild filter bitmaps, rewrite a *fresh* single segment (same index family or switch); crash‑safe staging then publish. Coroutines drive IO overlapped with CPU.

---

## 7) Filtering & Payload Indexing

* **Roaring bitmaps** per field value encode doc id sets (fast AND for multi‑term filters). Numeric ranges use tiered bitmaps (blocks) + min/max to prune; strings/enums use dictionary‑encoded bitmaps. During ANN candidate expansion, we **intersect** the current candidate set with the filter mask to avoid wasted distance evaluations.  
* *Filtered traversal* for DiskANN: maintain per‑beam *filter‑aware* frontier; skip neighbors failing predicates, avoiding SSD reads when possible.

---

## 8) Distance, Similarity & Metrics

* **L2** and **Inner Product (IP)** are primary metrics. **Cosine** is supported via L2‑normalization: for unit‑norm vectors, maximizing IP equals maximizing cosine and minimizing L2^2.  
* **ADC**: for PQ/OPQ, precompute `m` lookup tables per query; distance ∑ table[code[i]], using int8/uint8 indices and SIMD horizontal sums.  
* **Top‑k selection:** partial selection using **Quickselect/Floyd‑Rivest** for stable O(n) expected time; use small fixed‑size heaps for beam maintenance.

---

## 9) SIMD, Caches & NUMA

* **SIMD kernels** (AVX2/AVX‑512F): unrolled fused multiply‑add for L2 and IP; for PQ, vectorized table lookups with gather or SWAR techniques; software prefetch of adjacency lists and PQ code words.  
* **Padding & alignment:** 64‑byte cache lines, dimension padded to 32/64 elements to ease unrolling; interleave hot scalar metadata in separate arrays (SoA) to avoid polluting cache lines.  
* **NUMA:** pin worker threads, allocate scratch arenas on local nodes, *first‑touch* policy for bulk mmaps; optional page migration disabled for immutable segments.

---

## 10) Ingestion & Building

* **k‑means++** for coarse quantizer (IVF) + Lloyd iterations; **OPQ** rotation trained offline or online mini‑batch (option).  
* **HNSW**: `M`, `efConstruction` autotuned to memory budget and target recall; maintain visited sets with lock‑free bitsets.  
* **Vamana/Disk‑graph**: two‑pass robust prune, adding longer edges for navigability; neighbor lists serialized in ascending id for locality.  
* **Concurrency:** parallel construction (shards/partitions) → merge; robust spills to temp files when RAM bound.

---

## 11) Query Planner & Execution

1. **Parse & Plan:** metric, filters, target recall, time budget.  
2. **Compute filter mask:** per segment intersection of roaring bitmaps.  
3. **Pick index family:** small segment → brute‑force (SIMD); hot mutable → HNSW; cold sealed and compact → IVF‑PQ; very large → Disk‑graph.  
4. **Execute:** run candidate generation (graph traversal or IVF `nprobe`), **ADC** or exact distance on shortlist, **top‑k** with partial selection.  
5. **Rerank (optional):** exact float32 distance on top‑R results (R ≪ k).

---

## 12) Concurrency & Correctness

* **Readers:** RCU‑style epoch pin for segment maps; immutable segments are wait‑free for readers.  
* **Writers:** append WAL under a single writer mutex; mutable segment guarded by coarse shard locks.  
* **Memory reclamation:** hazard pointers or epoch‑based reclamation for transient graph nodes/visited sets to avoid ABA and UAF in lock‑free structures.  
* **Threading:** work‑stealing pools; per‑thread scratch `pmr::monotonic_buffer_resource` to eliminate allocator contention.

---

## 13) File Format Details (v1)

* **Endianness:** little‑endian scalar fields; explicit length prefixes; contiguous pages for vectors/PQ codes.  
* **Checksums:** CRC32C (SSE4.2) for small records, xxHash64 for large blobs; stored per‑block.  
* **Encryption (optional):** XChaCha20‑Poly1305 streaming; encrypted regions: `.vec`, `.meta`; headers include KDF salt (Argon2id) and nonce; AEAD additional data binds file UUID & segment id.

---

## 14) Security & Key Management

* **Default:** libsodium Secretstream (XChaCha20‑Poly1305) for large files; **nonce misuse‑resistant** choice to avoid catastrophic IV reuse risks seen with AES‑GCM when misused.  
* **Keying:** envelope (master key from OS KMS/TPM/HSM), per‑collection data keys; *rotate* via re‑encryption during compaction.  
* **Integrity:** per‑block AEAD tags; verify on read path; fail‑closed semantics.

---

## 15) Observability & Tooling

* Structured logging (JSON), counters (QPS, P50/P99, cache hit, fsync latency), histograms for distance kernel cycles, SSD read sizes, and bitmap densities.  
* `vesper inspect` CLI: dump manifests, verify checksums, compute recall on a probe set, print parameter suggestions.  
* `vesper bench` CLI: run ANN‑Benchmarks datasets and filtered‑retrieval datasets; export CSV + flamegraphs.

---

## 16) Testing, Verification & Benchmarks

* **Correctness:** randomized property tests (idempotent recovery, WAL truncation at arbitrary offsets, power‑fail fuzzing via fault‑injection).  
* **Performance:** `ANN‑Benchmarks` harness across SIFT1M/GIST1M/Deep; include **filtered** variants (category/range filters).  
* **Reproducibility:** pin compiler flags, CPU affinity, NUMA policy; publish config + seeds.  
* **Comparative:** HNSW vs IVF‑PQ vs Disk‑graph under identical recall targets and filters.

---

## 17) Parameter Heuristics (Initial)

* *IVF:* `nlist ≈ 4*sqrt(N)`; *nprobe* tuned to target recall & filter selectivity.  
* *PQ:* choose `m` such that `d % m == 0`; start with `m=16..64`, `nbits=8`; OPQ on by default for d≥256.  
* *HNSW:* `M=16..32`, `efC=200..600`, `efS≈ 2..8*k`.  
* *Disk‑graph:* beam width 4–16×k; neighbor cap 32–64; SSD page 4–16 KiB aligned; warm cache of entry points.

---

## 18) Roadmap

* v0: IVF‑PQ + HNSW + WAL/snapshots + Roaring filters + POSIX backends.  
* v1: DiskANN‑style graph + Filtered traversal; OPQ online trainer; encrypted segments.  
* v2: Incremental OPQ, dynamic graph maintenance; NUMA auto‑tuning; coroutine pipelines.  
* v3: Optional GPU build acceleration (not runtime), windows overlapped I/O backend.

---

## 19) Bibliography (selected, with notes)

**Approximate NN & Graphs**  
* HNSW: Malkov & Yashunin, *Efficient and robust ANN using HNSW* (2016/2020).  
* NSG: Fu et al., *Navigating Spreading‑out Graph* (2017/2019).  
* DiskANN (Vamana): Subramanya et al., *NeurIPS’19.*  
* Filtered‑DiskANN: Gollapudi et al., *SIGMOD’23.*  
* SPANN (hybrid IVF on SSD): Chen et al., *arXiv 2021.*  
* Survey: Wang et al., *VLDB 2021.*

**Quantization & IVF**  
* Jégou et al., *Product Quantization for NN* (TPAMI 2011).  
* Ge et al., *Optimized Product Quantization* (CVPR 2013; PAMI 2014).  
* IMI: Babenko & Lempitsky, *Inverted Multi‑Index* (2012).  
* FAISS overview & metric notes: FAISS docs/paper (2017–2024).

**Filtering**  
* Roaring bitmaps: Lemire et al., 2016+ (paper + format).

**Distance & Selection**  
* Cosine/IP equivalence on unit‑norm vectors: standard identity; see FAISS docs + references.  
* Partial selection: Hoare *Quickselect* (1961) and Floyd–Rivest *SELECT* (1975).

**Persistence & OS**  
* POSIX `fsync(2)`, `rename(2)`, `mmap(2)`, `posix_fallocate(3p)`; ARIES (1992) for WAL concepts (inspiration).

**SIMD, Caches, NUMA**  
* Intel Optimization Reference Manual; Ulrich Drepper’s *What Every Programmer Should Know About Memory*; Linux NUMA docs.

**Security**  
* NIST SP 800‑38D (AES‑GCM) cautionary limits; Libsodium XChaCha20‑Poly1305 Secretstream docs; Argon2id (RFC 9106).

---

## 20) Appendix A — Algorithmic Details (Engineering Notes)

### A.1 Product Quantization (PQ) & OPQ

* **Encoding:** split `d`‑dim vector into `m` subspaces of size `d/m`; for each, learn `K=2^b` codewords (k‑means). Store `m` codes (1 byte each for `b=8`).  
* **ADC:** per‑query, compute `m` LUTs of size `K` by measuring distances between query sub‑vectors (or rotated query for OPQ) and each sub‑codebook. Compress LUTs to `float16` if accurate enough; accumulate distances with SIMD, using integer indices to gather.  
* **OPQ:** learn orthogonal rotation `R` that minimizes reconstruction error over training data; store `R` in segment header. For cosine/IP on normalized inputs, apply `R` to normalized queries.  
* **Training:** k‑means++ seeding; mini‑batch Lloyd iterations; early stop by relative inertia change; shard + merge centroids for scalability.

### A.2 IVF

* **Coarse quantizer:** vanilla k‑means or *balanced* variant to avoid long tails; index stores centroid vectors.  
* **Search:** choose `nprobe` nearest centroids to the query (SIMD distance); scan their postings (PQ codes), apply **filters** first (bitmap AND against posting’s id set), then ADC; maintain a bounded candidate heap; optional exact recheck on top‑R raw vectors (if stored).

### A.3 HNSW

* **Construction:** random layer assignment (exponential), neighbor selection via heuristic (diversification) and bounded `M`.  
* **Search:** greedy descent from top, then beam/best‑first on bottom layer with `efSearch`; stop when queue cannot be improved.  
* **Filtering:** when expanding neighbors, only push ids in the **filter mask**; this yields fewer distance calls and better P99.

### A.4 Disk Graph (Vamana‑style)

* **RobustPrune:** iterate dataset; for each point `p`, run greedy search from a seed set to form candidate neighbors `V`, then prune edges using angular/L2 criteria with parameter `α`; second pass adds longer edges to ensure navigability. Store neighbor arrays on SSD; group by contiguous id ranges for locality.  
* **Search:** SSD‑aware beam: prefetch blocks of neighbor lists; when a **filter** is present, maintain filtered beam and avoid expanding disqualified nodes. Cache entry points and routing vectors in RAM.  
* **Engineering:** align adjacency blocks to 4–16 KiB; store PQ codes alongside ids to evaluate ADC before reading full vector payload (if any); use read‑ahead.

### A.5 Filters with Roaring Bitmaps

* **Structure:** per column/value roaring bitmap (runs + bitsets); segment‑scoped. For ranges, keep tiered bitmaps + value min/max to cut postings.  
* **Fusion:** compute `mask = AND(field1=v1, field2∈S, range(x))` once per query per segment; apply during candidate enumeration (IVF postings, HNSW expansion, Vamana beam).

### A.6 Distance Kernels (CPU)

* **L2:** unrolled FMA with AVX2/AVX‑512; accumulate partial sums in registers; handle tail with masked ops.  
* **IP:** same kernel without subtract; for cosine, pre‑normalize database vectors at build‑time or normalize queries online.  
* **PQ ADC:** integer code indices index into LUTs; use gather+horizontal add; on AVX2 stick to 8‑bit indices, on AVX‑512 leverage wider gathers; prefetch next posting list/page.  
* **Top‑k:** if candidate count ≲ few × k, use small max‑heap; otherwise apply partial selection (Floyd–Rivest) on contiguous distance buffer.

### A.7 NUMA & Memory

* Pin threads (`sched_setaffinity`); allocate per‑thread scratch arenas from local node; `mmap` immutable files with first‑touch on intended node; disable automatic migration for sealed segments.  
* Reserve hugepages optional for LUTs to reduce TLB pressure (tunable).

### A.8 WAL & Recovery Details

* **Commit protocol:** append records, write `Commit{seq}`, `fdatasync(fd)`, then `fsync(dirfd)` when publishing new segment directories via `rename()`.  
* **Replay:** verify each frame checksum; ignore trailing partial frames after a crash; apply idempotently (set‑last‑writer‑wins for upserts).  
* **Snapshots:** copy manifests + hardlink sealed segment files where supported; otherwise re‑copy with checksums verified.

### A.9 Security & Crypto Engineering

* **AEAD choice:** XChaCha20‑Poly1305 via libsodium Secretstream to remove nonce‑reuse footguns and permit essentially unbounded streams; Argon2id KDF for passphrase‑derived keys.  
* **Integrity first:** fail reads on tag mismatch; layer checksums inside AEAD for defense‑in‑depth.  
* **Zero‑copy:** decrypt in fixed‑size chunks aligned to page boundaries to keep IO streaming.

---

## 21) Appendix B — Implementation Sketch (no code)

* CMake options: `VESPER_WITH_AESGCM`, `VESPER_WITH_SODIUM`, `VESPER_WITH_AVX512`.  
* Public API surface (concepts): `Collection`, `Builder`, `Segment`, `SearchParams`, `FilterExpr`.  
* Memory: `pmr::monotonic_buffer_resource` per thread; custom resource over hugepage `mmap` for LUTs.  
* Error handling: `expected<T, error_code>`, never throw in hot paths; debug builds with contracts.  
* Portable fallbacks for SSE4.2 CRC32C and non‑SSE platforms (software CRC).

---

## 22) Open Questions & Future Research

* Combining **OPQ** with **DiskANN** beam scoring on SSD to reduce random reads further.  
* Learned coarse quantizers (e.g., PCA‑k‑means hybrids) to reduce `nprobe` for high‑dimensional transformer embeddings.  
* Adaptive filter‑aware beam control (learned beam size vs. bitmap density).  
* SIMD table‑driven ADC using compressed LUTs (`fp16`/`bf16`) without recall loss.

---

## 23) Verification of Truth & Citations (selected)

* **HNSW:** Malkov & Yashunin (2016/2020).  
* **DiskANN/Vamana:** Subramanya et al. (NeurIPS 2019); FreshDiskANN (2021); Filtered‑DiskANN (SIGMOD 2023).  
* **PQ/OPQ/IVF:** Jégou et al. (TPAMI 2011); Ge et al. (CVPR/PAMI 2013); FAISS docs/paper (2017–2024).  
* **Cosine ↔ IP equivalence (unit‑norm):** FAISS docs; standard identity.  
* **Roaring Bitmaps:** Lemire et al. (2016+).  
* **ANN Benchmarks:** ann‑benchmarks (Aumüller et al., 2018) + Big‑ANN (NeurIPS’21).  
* **SIMD & memory:** Intel Optimization Manual; Drepper (2007).  
* **Persistence primitives:** `fsync(2)`, `rename(2)`, `mmap(2)`, `posix_fallocate(3p)`.  
* **Crypto:** NIST SP 800‑38D (AES‑GCM); libsodium Secretstream (XChaCha20‑Poly1305); RFC 9106 (Argon2id).

(See the accompanying chat message for direct source links.)

---

*End of blueprint.*
