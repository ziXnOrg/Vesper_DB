# Vesper Codebase Audit - 2025

## Summary
- Total files reviewed: 66
- High-priority issues: 42
- Medium-priority issues: 152
- Low-priority issues: 104

## High-Priority Issues (Correctness, Safety, ABI Stability)
- [ ] (TBD)

## Medium-Priority Issues (Refactoring, Performance, Documentation)
- [ ] (TBD)

## Pass 1 Findings (Public Headers)

### include/vesper/index/ivf_pq.hpp
- [ ] Medium: ABI stability — IvfPqTrainParams uses nested enum KmeansElkan::Config::InitMethod
  - **Location**: lines ~55–65, ~140–166 (train API)
  - **Details**: Public struct depends on nested enum from another header; underlying values are not documented as stable. If InitMethod expands/reorders, ABI could be impacted across DLL boundaries.
  - **Recommendation**: Add explicit note that enum values are API-stable, or define an IvfKmeansInitMethod enum locally with fixed values and convert internally.
- [ ] Medium: Runtime validation clarity
  - **Location**: train() doc lines ~158–164
  - **Details**: Preconditions listed (n >= nlist; dim divisible by m). Ensure implementation returns explicit error codes when violated.
  - **Recommendation**: Cross-check ivf_pq.cpp to confirm checks; add “Errors” paragraph listing possible error codes.

### include/vesper/index/product_quantizer.hpp
- [ ] Medium: Validation/error model documentation
  - **Location**: train/encode/decode docs lines ~69–131
  - **Details**: Preconditions are stated; error codes are not enumerated in header.
  - **Recommendation**: Add explicit error conditions (e.g., dim % m != 0, untrained quantizer) and corresponding core::error values.
- [ ] Low: Performance note
  - **Location**: encode/decode docs lines ~101–131
  - **Details**: Mark expected throughput and memory access pattern (SoA) to guide users on batch sizing.
  - **Recommendation**: Add brief performance guidance.


- [x] High: ABI stability — STL and std::expected in public C++ API — RESOLVED (2025-10-29)
  - **Location**: train/encode/decode/compute_* declarations (74–162), save/load (202–207)
  - **Details**: The header exposes std::expected and std::string in public signatures. These are not ABI-stable across compilers/STL implementations.
  - **Recommendation**: Document that the C++ API is not ABI-stable across DSOs; prefer the C API for cross-boundary use. Optionally add filesystem::path overloads for save/load.
  - **Resolution**: Added ABI guidance note to include/vesper/index/product_quantizer.hpp; implemented std::filesystem::path overloads for save/load while preserving existing std::string overloads; added unit test tests/unit/product_quantizer_save_load_path_test.cpp; validated with Ninja+MSVC Debug build — [pq][io] subset and full suite passing; no new warnings.

- [ ] Medium: Preconditions mismatch (n ≥ ksub vs implementation n ≥ ksub·m)
  - **Location**: Header train() docs (71–73); Implementation src/index/product_quantizer.cpp (43–50)
  - **Details**: Header states n ≥ ksub, but implementation requires n ≥ ksub·m. Mismatch can confuse users and tests.
  - **Recommendation**: Align header docs to implementation and tests; consider enforcing explicit check for n ≥ ksub·m in train() docs and examples.

- [ ] Medium: Buffer-size preconditions not specified
  - **Location**: encode/decode/compute_* docs (95–162)
  - **Details**: Callers must provide buffers sized as: codes ≥ n·code_size(), distances ≥ n, table ≥ m·ksub, code/code1/code2 ≥ m.
  - **Recommendation**: Document exact buffer sizing; add an “Errors” paragraph enumerating core::error codes for violations.

- [ ] Medium: Serialization/versioning docs incomplete
  - **Location**: save()/load() (202–207)
  - **Details**: Implementation writes magic "VSPQ" and version=1 with no checksum; header doesn’t state format/version or integrity guarantees.
  - **Recommendation**: Document file format (magic/version), integrity (checksum TBD), and backward-compat policy.

- [ ] Low: Null pointer expectations
  - **Location**: All pointer parameters
  - **Details**: Header should explicitly state non-null requirements for data/codes/table pointers.
  - **Recommendation**: Add precondition notes to function docs.

- [ ] Low: Unused include
  - **Location**: #include <optional> (24)
  - **Recommendation**: Remove in Phase 2 to reduce transitive surface.


### include/vesper/index/pq_fastscan.hpp

- [ ] High: Training error propagation missing (silent success)
  - **Location**: train_subquantizer() returns void; train() ignores failures (src/index/pq_fastscan.cpp 62–71; 12–40)
  - **Details**: If k-means fails, codebook entries remain default-initialized but train() still returns success. Leads to degraded accuracy and undefined behavior in downstream SIMD paths.
  - **Recommendation**: Propagate errors via std::expected from train_subquantizer() and aggregate in train(); document possible error codes in header.

- [ ] High: encode/decode/compute_lookup_tables require trained state; not validated
  - **Location**: encode()/decode()/compute_lookup_tables declarations (128–176); implementations use codebooks_ without trained_ check (src/index/pq_fastscan.cpp 97–109, 279–291, 136–157)
  - **Details**: Calling before train/import_pretrained may dereference null codebooks_.
  - **Recommendation**: Document precondition “must be trained”. In Phase 2, add runtime guards and return errors (or asserts in non-API hot paths).

- [x] High: compute_batch_distances buffer sizing and empty-block handling — RESOLVED (2025-10-29)
  - **Location**: inline compute_batch_distances() (270–284)
  - **Details**: Uses blocks[0].size() to size per-query slice. For n < block_size or when blocks is empty, this under-allocates or dereferences out-of-bounds. compute_distances() uses stride = blocks.size()·block_size, creating mismatch.
  - **Recommendation**: Document expected output layout (stride = blocks.size()·block_size); validate non-empty blocks; compute total_codes from per-block sizes or pad consistently; add tests for n < block_size.
  - **Resolution**: Added empty-input guard; fixed per-query stride to use `pq.config().block_size`; updated documentation for both compute_batch_distances and compute_distances to state layout and padding semantics.
  - **Tests**: Added `tests/unit/pq_fastscan_batch_distances_layout_test.cpp` covering empty blocks, partial single block, and multi-block partial tail with multi-query stride verification.
  - **Validation**: Ninja+MSVC Debug build succeeded; targeted tests passed (137 assertions in 3 test cases, filter [pq][fastscan][batch]); full suite passed (8507 assertions in 188 test cases); zero new warnings.

- [ ] High: ABI stability — public API exposes STL containers and std::span
  - **Location**: encode_blocks() returns std::vector<PqCodeBlock> (136–138); compute_distances() takes std::vector<PqCodeBlock> (153–155); import_pretrained() takes std::span (202)
  - **Details**: These types aren’t ABI-stable across DSOs/toolchains.
  - **Recommendation**: Document ABI boundary; recommend C API for cross-DSO use.

- [ ] Medium: Parameter validation incomplete
  - **Location**: train() (119–121), FastScanPqConfig (34–39)
  - **Details**: Missing checks: n ≥ ksub, block_size ∈ {16,32}, m > 0, nbits supported. Only dim % m is validated.
  - **Recommendation**: Add validations and explicit error codes; assert invariants in debug.

- [ ] Medium: SIMD dispatch clarity and OS XCR0 gating
  - **Location**: AVX2/AVX-512 functions (157–169, 165–169) and config.use_avx512 (38)
  - **Details**: config.use_avx512 is unused; AVX-512 functions are compiled when __AVX512F__ and can SIGILL if OS XCR0 doesn’t enable ZMM state and the functions are called.
  - **Recommendation**: Document gating; add runtime checks for AVX-512 availability (CPUID + XCR0) and respect config flag.

- [ ] Low: Heavy temporaries/allocations in hot paths
  - **Location**: compute_lookup_tables() returns AlignedCentroidBuffer by value (176–178); AVX2 path allocates aligned_dists per call (src/index/pq_fastscan.cpp 200–222)
  - **Recommendation**: Provide scratch-buffer overloads or thread-local scratch; document cost.

- [ ] Low: Complexity and thread-safety docs
  - **Location**: Class preamble (100–106) and methods
  - **Details**: Add O() complexity for encode/decode/compute_distances; clarify that post-training read-only ops are thread-safe; training parallelizes per subquantizer.
  - **Recommendation**: Add notes consistent with Vesper doc style.


### include/vesper/index/hnsw.hpp
- [ ] Medium: Doc clarity around threading
  - **Location**: header preamble lines ~8–15; HnswBuildParams at lines ~30–42
  - **Details**: Preamble says “Build is single-threaded”, yet params include num_threads and add_batch notes internal parallelization.
  - **Recommendation**: Clarify: build graph connectivity is inherently serialized per insertion, but add_batch may parallelize data prep; document exact guarantees.
- [ ] Low: Determinism note
  - **Location**: HnswBuildParams lines ~30–42
  - **Details**: seed is present; document determinism expectations (same seed → identical graph) and any platform caveats.

### include/vesper/index/kmeans_elkan.hpp
- [ ] Low: Numeric stability bounds
  - **Location**: Config and cluster() docs lines ~45–91
  - **Details**: We now use squared-distance Elkan with s_quarter; add a note on no NaN/Inf allowed and bound propagation uses sqrt then squares back.
  - **Recommendation**: Add 1–2 sentences in header for future maintainers.

### include/vesper/index/kmeans.hpp
- [ ] Low: Quality metrics doc
  - **Location**: ClusterMetrics lines ~139–146
  - **Details**: Provide typical ranges and guidance for interpreting metrics.
  - **Recommendation**: Add brief notes (e.g., silhouette in [-1,1], higher is better).

### include/vesper/index/capq.hpp

- [ ] High: Const-correctness breach — const view exposes mutable spans
  - Location: `make_view() const` (≈197–205); `CapqSoAView` fields are `std::span<std::uint64_t>` and `std::span<std::uint8_t>`
  - Details: The const overload constructs a `CapqSoAView` with mutable spans via `const_cast`, allowing mutation through a const reference. This violates const-correctness and can enable accidental writes in read-only contexts, undermining thread-safety assumptions.
  - Web validation: C++ Core Guidelines (Const-correctness); `std::span<const T>` should be used for read-only views.
  - Recommendation: Introduce a `CapqSoAViewConst` with `std::span<const std::uint64_t>` and `std::span<const std::uint8_t>` (or templatize `CapqSoAView<T>`); have `make_view() const` return the const-view type. Remove `const_cast` and ensure the API enforces read-only access from const.
- [ ] Medium: bytes_per_vector_* ignore 384-bit configuration
  - Location: `CapqSoAView::bytes_per_vector_payload()`/`bytes_per_vector_total()` (≈97–105)
  - Details: Functions return 128 and 129 bytes respectively regardless of `CapqHammingBits` (`B256` vs `B384`). For 384-bit configuration, payload should be 144 bytes and total 145 bytes. Current constants can lead to buffer mis-sizing and unsafe copies if used by callers.
  - Web validation: FWHT/Hadamard compression preserves energy with orthonormal scaling; storage size must match configured bit budget.
  - Recommendation: Compute `payload = words_per_vector()*sizeof(std::uint64_t)`; `total = payload + 1` (residual energy byte). Ensure consistency with `CapqSoAStorage::words_per_vector()`. Add unit tests for both `B256` and `B384`.
- [ ] Low: Thread-safety/docs reference to ADR lacks explicit guarantees
  - Location: File preamble notes “plain data containers; concurrency control in higher layers (ADR-0004)” (≈13–20)
  - Details: The header defers thread-safety to an ADR reference without stating concrete guarantees for read-only vs writable views. Given the const-view mutability bug above, documentation should explicitly state read-only guarantees and view lifetimes.
  - Recommendation: Document: (a) `CapqSoAViewConst` is read-only and thread-safe for concurrent reads; (b) `CapqSoAView` is writable and not safe for concurrent mutation; (c) lifetime/aliasing rules. Link actual ADR once available.

## Pass 2 Findings (Index Implementations)

### src/index/ivf_pq.cpp
- [ ] Medium: Portability of prefetch macros
  - **Location**: lines ~1426–1435
  - **Details**: Uses `#if defined(__builtin_prefetch)` directly; on MSVC this is never defined.
  - **Recommendation**: Route prefetch through platform/intrinsics.hpp; no-op on unsupported compilers.
- [ ] Medium: Coarse top-k selection performance
  - **Location**: lines ~1690–1704
  - **Details**: Inner nested loops do branchless insertion into top-k buffers (O(n*k^2) worst case). For large centroid counts this can be costly.
  - **Recommendation**: Consider a small fixed-size min-heap or SIMD-accelerated partial selection to reduce constant factors.
- [ ] Medium: OPQ training allocations
  - **Location**: lines ~2565–2582
  - **Details**: Re-creates `PqImpl` each iteration; repeated allocations and potential cache misses.
  - **Recommendation**: Reuse a single instance or pool; measure impact.
- [ ] Medium: Windows mmap disabled
  - **Location**: lines ~4528–4533
  - **Details**: `mmap_supported = false` under _WIN32, forcing streaming path.
  - **Recommendation**: Implement Windows file mapping (CreateFileMapping/MapViewOfFile) behind a uniform abstraction; add tests.
- [ ] Low: Env-controlled streaming path
  - **Location**: lines ~4895–4900
  - **Details**: Behavior toggled by `VESPER_IVFPQ_LOAD_STREAM_CENTROIDS`.
  - **Recommendation**: Document env var in developer docs; ensure determinism unaffected.

- [ ] Medium: OPQ rotation application contract (ownership in IvfPq)
  - Location: add() applies R before encode (3325–3329); search() applies R to residual query before LUT (3603–3606); decode paths apply R^T for reconstruction (3769–3775, 4052–4059); update path re-applies R before re-encode (5305–5309)
  - Details: IvfPq performs all OPQ rotations externally around ProductQuantizer (train/add/search/rerank). This makes IvfPq correct even if ProductQuantizer does not apply rotations internally (as currently observed). However, if ProductQuantizer later fixes OPQ internal handling, double-rotation risk exists unless an explicit contract is documented.
  - Web validation: OPQ per He et al., PAMI 2013; FAISS OPQMatrix applies rotation at encode and query LUT computation.
  - Recommendation: Document cross-component contract: IvfPq owns OPQ rotations; ProductQuantizer must not auto-rotate on encode/decode when used under IvfPq, or expose an explicit flag to prevent double-rotation.
- [ ] Medium: Numerical stability gates (NaN/Inf) are debug-only
  - Location: add() residuals debug validation (3292–3313); compute_adc_distance() (3476–3487); search() accumulation and rerank; train_product_quantizer() main loops (2398–2767)
  - Details: Only debug logging checks for NaN/Inf on small samples. No enforcement in normal paths; bad inputs could propagate NaN/Inf and violate Vesper numerical stability gates.
  - Web validation: Vesper gates require zero NaN/Inf in distance kernels and search pipelines.
  - Recommendation: Add finite checks at public entry points (train/add/search/search_batch) and early-return errors on non-finite inputs; optionally guard compute_lookup_tables/ADC accumulation in debug builds with asserts.
- [ ] Medium: Rerank cluster lookup is O(nlist × list_size) per candidate set
  - Location: search() rerank phase builds id→cluster via scanning all inverted lists (4009–4026); then reconstructs and computes exact L2 (4028–4074)
  - Details: For large indices and bigger rerank_k, repeated list scans increase latency. Correctness is fine; performance may degrade.
  - Recommendation: Maintain an auxiliary reverse index (unordered_map id→cluster) updated on add(); or store cluster_id alongside each candidate in the initial pool to avoid scans.
- [ ] Low: Rotation multiply is naive O(d^2) per vector
  - Location: apply_rotation() (2947–2967); apply_rotation_T() (5369–5386)
  - Details: Triple loop without blocking/BLAS. For d∈{768,1536} this can dominate per-query LUT prep when OPQ enabled.
  - Recommendation: Consider blocked GEMM or optional CBLAS path under feature flag; ensure determinism preserved. Measure before changing.
- [ ] Low: Environment toggles undocumented (diagnostics/perf)
  - Location: pool size/env (3544–3579, 3571–3577); probe-all (3530–3535); scalar PQ path (3611–3626); KD approximate toggle (3021–3047)
  - Details: Numerous env vars control validation and performance. Useful for debugging but not documented for developers.
  - Recommendation: Add a developer doc section listing IVFPQ env toggles, their effects, and determinism impact.
- [ ] Info: ADC LUT sum parity fix is present and correct
  - Location: compute_adc_distance() (3476–3487)
  - Details: Uses per-sub LUT pointer and sums lut_m[code]; aligns with LUT layout and fixes prior stride bug.
  - Recommendation: Keep; add a unit test that cross-checks FastScan SIMD vs scalar LUT sum on random codes and queries (when we enter Phase 2).
- [ ] Medium: Determinism and ANN toggles documentation
  - Location: KDTree approximate mode (3021–3047); HNSW refine path (3113–3166); projection screening (3180–3231)
  - Details: Approximate KD is opt-in via env; projection screening followed by exact refinement. Determinism expectations should be explicit (defaults deterministic; env toggles may change behavior).
  - Recommendation: Document determinism guarantees and randomness sources (seeds, ordering); add notes to API reference.
- [ ] Low: Inverted list code storage has per-entry heap overhead
  - Location: add() insertion (3416–3424)
  - Details: Each InvertedListEntry stores codes in std::vector<uint8_t>, causing many small allocations and fragmentation.
  - Recommendation: Consider contiguous code arenas per list or fixed-size small-vector optimization to reduce overhead (future perf phase).
- [ ] Medium: OPQ rotation orthogonality fallback tolerance is coarse
  - Location: learn_opq_rotation() orthogonality check and fallback (2916–2945)
  - Details: Accepts |dot(ei, ej)| ≤ 0.01; on failure, falls back to identity. This may silently degrade OPQ quality.
  - Web validation: Orthogonal Procrustes solution recommends enforcing R orthonormality (U·V^T) with stricter tolerances.
  - Recommendation: Add explicit re-orthonormalization (Gram-Schmidt or SVD) and a tighter tolerance (e.g., 1e-3); log diagnostic metrics; keep identity fallback as last resort.


### src/index/ivf_pq.cpp (second pass)
- [ ] Medium: Input/param validation gaps in train/add/search
  - Location: train() prechecks (414–428) cover n>=nlist and dim% m==0; add()/search() validation spread across method; no explicit bounds for some fields
  - Details: Validate remaining constraints: m>0; nbits in supported set (e.g., {4,8}); opq_iter/opq_sample_n >=0; projection_dim ∈ [1, dim]; search: k>0; nprobe∈[1,nlist]; cand_k≥k; if use_exact_rerank then rerank_k≥k and cand_k≥rerank_k. Prefer early precondition errors over implicit clamping/heuristics.
  - Web validation: FAISS and literature enforce param sanity to avoid undefined behavior; defensive checks ease debugging.
  - Recommendation: Add explicit precondition checks with error_code::config_invalid or precondition_failed and actionable messages.
- [ ] Medium: add() edge cases and error surface at entry
  - Location: add() entry (Impl::add)
  - Details: Ensure preconditions: trained=true; n>0; ids/data non-null; dim matches training; avoid partial adds on failure; document duplicate-ID policy (allowed vs rejected).
  - Recommendation: Document and enforce at entry; return detailed error on violations.
- [ ] Low: Search parameter behavior and docs
  - Location: pool sizing and rerank logic (3548–3556)
  - Details: pool_k growth via env and rerank heuristics is correct but implicit; no explicit mention that pool_k≥k and may be increased by rerank. Determinism implications of env gates should be documented.
  - Recommendation: Document parameter interactions and determinism notes in header/API reference.

### src/index/kmeans_elkan.cpp

- [ ] Info: Stats accounting
  - **Location**: lines ~395–405
  - **Details**: Stats computed from atomic counters; looks correct post-fix. No action.


### src/index/capq.cpp
- [ ] Info: Walking skeleton TU (no algorithmic content yet)
  - Location: file overview (17–31)
  - Details: This TU includes the CAPQ header to ensure build integration; all algorithmic stages (FWHT, whitening, quantization, distances) live under `capq_util.hpp`, `capq_encode.hpp`, and `capq_dist.hpp`. No correctness/perf issues identified here.
  - Recommendation: None for this TU. See separate audits for the utility/encode/dist files (future passes).

### include/vesper/index/pq_fastscan.hpp


- [ ] Medium: Config validation — block_size should be enforced to {16,32}
  - **Location**: lines ~33–39, 52–59
  - **Details**: Header documents 16 or 32, but neither constructor nor train/encode validates; misuse may corrupt layout/padding assumptions.
  - **Web validation**: Confirmed FAISS “Fast-scan index” relies on block-friendly layouts; see “The Faiss Library” (2024/2025) and Quick ADC (2017) for SIMD LUT assumptions.
  - **Recommendation**: Validate at construction or train(); return precondition_failed on invalid values.
- [ ] Medium: API preconditions/returns — compute_distances* require trained state and non-empty blocks

### include/vesper/index/ivf_pq.hpp (second pass)

- [ ] Medium: Parameter constraints not fully specified in header docs
  - Location: IvfPqTrainParams (58–106); IvfPqSearchParams (108–117)
  - Details: Document explicit bounds/relationships: nlist≥1; m≥1 and dim% m==0; nbits allowed set (e.g., {4,8}); opq_iter/opq_sample_n reasonable ranges; projection_dim ∈ [1,dim]; kd_leaf_size in [16,1024]; search: k>0; nprobe∈[1,nlist]; cand_k≥k; if use_exact_rerank then rerank_k≥k and cand_k≥rerank_k.
  - Web validation: Jégou 2011 PQ typical nbits=8; FAISS recommends explicit param validation.
  - Recommendation: Expand header preconditions and cross-link API_REFERENCE.md; align C API docs.
- [ ] Low: Thread-safety wording is ambiguous
  - Location: File header (13–15) and search/train comments (181–205)
  - Details: “Training is single-threaded” can be misconstrued (internally parallel loops exist). The intent is operation-level concurrency: train/add/save/load not concurrent with search on same instance; search is safe concurrently.
  - Recommendation: Rephrase per C API doc (docs/C_API_Reference.md Thread Safety) for consistency.
- [ ] Low: API surface consistency and noexcept
  - Location: Getters and trivial methods (248–257, 250–255)
  - Details: Some getters are noexcept, others return expected; ensure consistency and document exceptions policy (no exceptions on hot paths; expected for errors).
  - Recommendation: Clarify in header comments; no code change in Phase 1.

- [ ] High: ABI stability — STL types in public C++ API across shared-library boundaries
  - Location: save/load (349–356), metadata setters/getters (206–216, 209–216), search/search_batch (193–219), reconstruct/get_vector (276–292), PqCode (128–133)
  - Details: Public API exposes std::string, std::string_view, std::vector and std::pair in method signatures. These are not guaranteed ABI-stable across compilers/STL implementations or differing ABI versions. The pImpl hides internals, but the signatures themselves constrain cross-DSO compatibility. A stable C API exists and should be the recommended boundary for plugins/FFI.
  - Web validation: General library guidance discourages STL types across binary boundaries; MSVC/libstdc++ ABI notes; Vesper rule “C API stability over C++ ABI”.
  - Recommendation: Document that the C++ API is source-compatible but not ABI-stable across DSOs; recommend using the C API for cross-boundary use. Consider overloads with const char* or std::filesystem::path for path parameters; keep STL-only overloads for in-process convenience.

- [ ] Low: Determinism & serialization toggles via environment
  - Location: Header notes (44–46)
  - Details: VESPER_IVFPQ_LOAD_MMAP and VESPER_IVFPQ_SAVE_V11 alter IO strategy/format, leading to environment-dependent outputs. This can harm reproducibility across machines/CI.
  - Recommendation: Document env var effects and provide programmatic setters on IvfPqIndex to pin behavior for deterministic pipelines.

- [ ] Low: Path type portability
  - Location: save()/load() params (349–356)
  - Details: std::string for paths may be insufficient on Windows (UTF-16). Using std::filesystem::path improves portability and intent.
  - Recommendation: Add std::filesystem::path overloads; document UTF‑8 expectation on POSIX.

- [ ] Medium: compute_distances* preconditions not documented
  - **Location**: lines ~147–176, 199–222
  - **Details**: Public API returns void; preconditions not stated; misuse could segfault (e.g., empty blocks, untrained PQ).
  - **Web validation**: ADC per Jégou et al. 2011 requires LUTs defined over trained codebooks; calling on untrained state is undefined.
  - **Recommendation**: Document preconditions; optionally provide status-returning overloads or asserts in debug.
- [ ] Medium: SIMD dispatch clarity — config.use_avx512 not used for routing
  - **Location**: lines ~33–39; 157–169
  - **Details**: AVX2/AVX‑512 methods exist but are not used by compute_distances/compute_batch_distances; flag may mislead users.
  - **Web validation**: FAISS reports explicit fast-scan SIMD paths; our API should route based on CPU features + config.
  - **Recommendation**: Add runtime dispatch (CPU feature probe) and honor config flag; or remove flag and document separate call.
- [ ] Low: Docs — add complexity and thread-safety notes
  - **Location**: class docs lines ~100–107; methods ~119–205
  - **Details**: Add O() notes, thread-safety (encode/decode/compute* are parallel-for safe), and memory layout guarantees.
  - **Recommendation**: Align with Vesper doc style.
### src/index/pq_fastscan.cpp

- [ ] High: Training error propagation — subquantizer k-means failures ignored
  - **Location**: lines 62–71; 12–40
  - **Details**: train_subquantizer drops failure; train() always returns success and sets trained_=true, potentially leaving uninitialized codebooks.
  - **Web validation**: Jégou 2011 PQ requires ksub centroids per subspace; failure should abort training.
  - **Recommendation**: Collect sub-results; if any error, return that error; do not set trained_.
- [ ] Medium: Preconditions — enforce n ≥ ksub for stable k-means
  - **Location**: lines 12–26, 55–61
  - **Details**: No guard for n < ksub (2^nbits); k-means may degenerate.
  - **Web validation**: PQ training requires sufficient samples per centroid; see Jégou 2011 and FAISS guidance.
  - **Recommendation**: Validate and return precondition_failed; document.
- [ ] Medium: compute_batch_distances — empty blocks handling
  - **Location**: header lines ~270–284 (stride uses blocks[0].size())
  - **Details**: Dereferences blocks[0] without guard; empty input can UB.
  - **Web validation**: ADC requires non-empty code sets; guard is standard.
  - **Recommendation**: If blocks.empty(), return early; document.
- [ ] Medium: SIMD paths consistency and dispatch
  - **Location**: lines 189–276
  - **Details**: AVX2 path writes to aligned temp then copies; AVX‑512 writes directly. No runtime dispatch from compute_distances.
  - **Web validation**: FAISS fast-scan uses explicit SIMD kernels with feature checks.
  - **Recommendation**: Unify approach; add runtime dispatch from compute_distances based on CPU and config.
- [ ] Low: Encoding pipeline allocs
  - **Location**: lines 111–134
  - **Details**: encode_blocks creates all_codes then transposes; could stream codes into blocks to reduce peak memory.
  - **Recommendation**: Optional refactor; measure before changing.

### include/vesper/index/projection_assigner.hpp

- [ ] Medium: API preconditions and error model not documented
  - **Location**: lines 9–24, 33–34
  - **Details**: Inputs require precomputed qproj/qnorm and optionally centroids_pack8; shapes/strides and alignment are implicit. Behavior when n==0, L==0 (clamped), or null buffers is not specified. Return type is void with no error surface.
  - **Web validation**: FAISS coarse assignment uses SGEMM + distance formula d(q,c)=||q||²+||c||²−2 q·c (The FAISS Library, 2025). Johnson–Lindenstrauss projections are standard for screening; preconditions must be explicit for correctness.
  - **Recommendation**: Document shapes, clamping (L in [1,C]), and nullability. Consider a status-returning overload or debug asserts.
- [ ] Medium: centroids_pack8 packing contract is undocumented
  - **Location**: lines 14–18 (centroids_rm, centroid_norms, centroids_pack8); 31–33 (notes)
  - **Details**: AVX2 kernel requires packing 8 centroids × 16-d panel with layout pack[blk*(16*8)+k*8+lane]; currently only test describes it. Missing guidance may cause UB.
  - **Recommendation**: Add packing layout diagram and example; state that AVX2 path requires p==16 and non-null pack, else falls back.
- [ ] Low: Output order and sorting guarantees
  - **Location**: function docs (add) and impl lines 367–375
  - **Details**: Output shortlist is not sorted; selection merges per-block top-T with cached worst. Expose this guarantee in header.
  - **Recommendation**: Document “unsorted shortlist” to avoid caller assumptions.

### src/index/projection_assigner.cpp
- [x] High: AVX2 tile correctness bug — per-row dist_buf overwritten across rows
  - **Location**: lines 162–193 (compute per-row distances inside cjblk loop), 195–235 (selection uses shared dist_buf for all rows)
  - **Details**: dist_buf is updated for each r inside the centroid-block loop, then a single shared dist_buf is reused for selection across r=0..15. This leaves dist_buf containing distances for only the last processed row when selection starts, corrupting candidates for other rows.
  - **Web validation**: Correct block GEMM + per-row selection pattern is described in FAISS (2025) for IVF coarse assignment; per‑row buffers or immediate selection are required.
  - **Recommendation**: Either (a) move selection for a row inside the r loop right after finishing all cjblk updates for that row, or (b) allocate dist_buf[16][jb] per tile. Add tests to compare AVX2 vs CBLAS/scalar outputs for small n, p=16.

  **Resolution**: Fixed by allocating per-row distance buffer `dist_buf16[16 * jb]` to preserve all rows' distances throughout tile processing. Validated with AVX2 vs scalar parity test in `tests/unit/projection_assigner_avx2_parity_test.cpp` (all 512 assertions passed).

- [ ] Medium: Comment/implementation mismatch for block shortlist size T
  - **Location**: comments lines 79–81, 194–200, 250–256 vs code setting T=min(L,jb)
  - **Details**: Comments say “T=L/2” but code uses T=min(L,jb). Mismatch causes confusion; also T impacts perf/recall tradeoff.
  - **Recommendation**: Align comment to implementation or implement T=L/2 as intended with justification; surface T as a tunable if needed.
- [ ] Low: Dead variable and minor alloc pattern
  - **Location**: lines 145 (idx_buf unused); 146 (dist_buf re-sized redundantly)
  - **Details**: idx_buf is never used; dist_buf.reserve/jb-sized arrays could avoid churn.
  - **Recommendation**: Remove dead variables; preallocate/reuse aligned buffers.
- [ ] Medium: API behavior — outputs not sorted and may include duplicates across blocks
  - **Location**: lines 367–375
  - **Details**: Merge strategy does not de-duplicate candidates across blocks. Header should document possible duplicates; caller may sort/unique.
  - **Recommendation**: Document; optional de-duplication pass if needed by downstream.

---

## Summary (updated)
- Total files reviewed (this pass): 23
- High-priority issues: 6
- Medium-priority issues: 63
- Low-priority issues: 35

### include/vesper/index/bm25.hpp
- [ ] Medium: API/doc gaps on threading and memory
  - **Location**: lines 13–15 (thread-safety/memory), 118–131 (add_document), 142–145 (add_batch), 147–160 (search)
  - **Details**: Header states search is thread-safe but does not specify read-write concurrency model (e.g., searching during add); memory big-O lacks assumptions (e.g., vocabulary capping). add_batch claims internal parallelization in header, but impl is sequential.
  - **Web validation**: Lucene BM25Similarity provides read concurrency during search with commit/snapshot semantics; IMR literature emphasizes clear concurrency guarantees.
  - **Recommendation**: Document read/write concurrency guarantees; align add_batch docs with implementation or mark as future work.
- [ ] Low: Tokenizer options vs BM25Params overlap
  - **Location**: BM25Params (lines 33–40) and Tokenizer::Options (241–248)
  - **Details**: Duplicate controls (lowercase, stopwords). Potential for divergence.
  - **Recommendation**: Note source of truth (BM25Params) and that Tokenizer::Options are derived internally.
- [ ] Medium: encode_text/get_document_vector semantics
  - **Location**: 162–179
  - **Details**: Returns TF-IDF weights, but score_document multiplies query_tf with BM25 doc-term contribution; clarify intended hybrid usage and normalization expectations.
  - **Recommendation**: Document units and examples for combining sparse with dense.

### src/index/bm25.cpp
- [ ] Medium: IDF variant choice should be documented
  - **Location**: compute_idf() lines 354–357
  - **Details**: Uses idf = log(1 + (N - df + 0.5)/(df + 0.5)), consistent with popular Lucene-style variants; not documented in header.
  - **Web validation**: Wikipedia Okapi BM25; Lucene BM25Similarity discussions; Kamphuis et al. “Which BM25 Do You Mean?” (ECIR 2020) note multiple variants. Our formula matches a non-negative Lucene-like form.
  - **Recommendation**: Document IDF formula and rationale in header; add note on reproducibility vs Lucene.
- [ ] Low: TODOs present (stemming; parallelize tokenization)
  - **Location**: lines 128–129; 809–810
  - **Details**: Clear placeholders; not harmful but should be tracked in backlog.
  - **Recommendation**: Record in roadmap; reference issue IDs if available.
- [ ] Medium: Concurrency model clarity
  - **Location**: add_document locks unique (269), search shared (407)
  - **Details**: Supports concurrent search, serialized mutation; behavior for search during add (snapshot consistency) not specified; inverted index and df updates happen incrementally.
  - **Recommendation**: Document consistency model (e.g., readers may or may not see in-flight adds).
- [ ] Medium: Scoring denominator stability
  - **Location**: score_document lines 371–389
  - **Details**: doc_len_norm uses avg_doc_length_ which can be 0.0 for empty corpus; pre-checks ensure non-empty index before search, but direct score_document calls can divide by zero.
  - **Recommendation**: Document precondition (non-empty index) at score_document; consider guard or expected-return overload (future fix phase).
- [ ] Low: Memory accounting is approximate
  - **Location**: get_stats lines 541–551
  - **Details**: Uses coarse estimates; acceptable but should be labeled approximate.
  - **Recommendation**: Annotate as estimate; optionally expose breakdown.
- [ ] Medium: Serialization format v1.0 lacks explicit endianness/versioned schema sectioning
  - **Location**: save/load lines 570–780
  - **Details**: Uses fixed magic and checksum; endianness not stated; no per-section checksums or compression flags.
  - **Web validation**: Vesper serialization gates require checksums and backward compat; sectioned v1.1 preferred.
  - **Recommendation**: Document v1.0 legacy status; plan v1.1 sectioned format per repo standard (future phase).


### include/vesper/index/product_quantizer.hpp (deeper review)
- [ ] Medium: OPQ usage contract and semantics not fully specified
  - **Location**: class docs lines 13–17, 78–94, 132–163
  - **Details**: Header advertises optional OPQ, but encode/decode/compute_distance_table docs do not state whether rotation is applied during those operations or if outputs are in rotated/original space.
  - **Web validation**: Spec algorithms/spec/ivf_pq_opq.md (§Build §Search) requires applying R during encode/search; FAISS OPQMatrix applies rotation consistently.
  - **Recommendation**: Document that when OPQ is enabled, encode/search operate in rotated space (q' = R·q, x' = R·x) and decode returns reconstructed vectors in original space (apply R^T).
- [ ] Low: Complexity and thread-safety notes could be expanded
  - **Location**: 95–163
  - **Details**: Add notes on O() per op, typical throughput expectations, and that read paths are thread-safe while training is single-threaded.
  - **Recommendation**: Align with Vesper header doc style (complexity, threading, memory).

### src/index/product_quantizer.cpp

- [ ] High: OPQ rotation learned but not applied in encode/decode/ADC
  - **Location**: encode (173–198), encode_one (200–217), compute_distance_table (243–274), decode_one_impl (547–551); OPQ flag set at 168
  - **Details**: After train_opq sets has_rotation_=true, subsequent encode/decode and distance table computation do not apply rotation (nor inverse rotation). This breaks OPQ correctness: codebooks trained in rotated space but encoding/search use unrotated vectors.
  - **Web validation**: Spec (§Search step 1) requires q' = R·q; OPQ literature (He et al., PAMI 2013) and FAISS implementations apply R during code assignment and LUT computation.
  - **Recommendation**: When has_rotation_ is true, apply R before subspace assignment and LUT computation; apply R^T when reconstructing in decode. Update docs and add parity tests.
- [ ] Medium: initialize_pca_rotation() is a stub (identity), mismatching comment
  - **Location**: 554–574
  - **Details**: Function name/comment suggests PCA init; code sets identity without computing PCA/covariance.
  - **Web validation**: OPQ commonly initializes with PCA or random orthonormal; proper PCA improves convergence.
  - **Recommendation**: Either implement PCA-based init or explicitly document identity fallback to avoid misleading readers.
- [ ] Medium: Procrustes/SVD update uses custom Jacobi SVD; numerical robustness unclear
  - **Location**: 590–751
  - **Details**: Homegrown SVD via Jacobi sweeps; orthogonality and convergence tolerances are ad‑hoc; no validation of R orthonormality/det(R)=+1.
  - **Web validation**: Orthogonal Procrustes solution is R = U·V^T from SVD of X^T·Y; use a well‑tested SVD or add sanity checks and re‑orthonormalization.
  - **Recommendation**: Add post‑update checks (||R^TR−I||_F, det>0); consider a simpler eigen/SVD routine or reduce to smaller dim blocks.
- [ ] Medium: Helper compute_pq_recall assumes 8‑bit codes
  - **Location**: 951–956
  - **Details**: Allocates LUT as m×(1<<8) regardless of nbits. If trained with nbits≠8, this is incorrect and may read/write out of bounds.
  - **Recommendation**: Use info.ksub to size tables; propagate ksub to this helper.
- [ ] Medium: Training precondition stricter than necessary
  - **Location**: 43–50
  - **Details**: Requires n ≥ ksub·m for training; literature and FAISS typically require n ≥ ksub (per subspace). Over‑restrictive precondition may reject adequate training sets.
  - **Web validation**: Jégou et al. 2011 PQ; FAISS ProductQuantizer training guidance.
  - **Recommendation**: Document rationale for stricter bound or relax to n ≥ ksub; ensure k‑means failure paths return errors (already handled).
- [ ] Low: Avoid per‑subquantizer reallocation/copies during training
  - **Location**: 67–74
  - **Details**: Reallocates and memcpy subvectors for each subquantizer; can reuse a single buffer/view to reduce churn.
  - **Recommendation**: Refactor for reuse; measure impact.
- [ ] Medium: Input validity and NaN/Inf handling not enforced
  - **Location**: train/encode/decode/compute paths
  - **Details**: No explicit checks for finite inputs; kernels may propagate NaN/Inf silently, violating Vesper numerical stability gates.
  - **Recommendation**: Add finite checks in training and public entry points; document behavior.

## Low-Priority Issues (Code Quality, TODOs, Minor Improvements)

- [ ] (TBD)

## Per-File Detailed Findings

### include/vesper/index/kmeans_elkan.hpp


- [x] Fixed (2025-10-22): Elkan bounds corrected for squared distances; `s_half` -> `s_quarter`; default math now correct
- [x] Fixed (2025-10-22): Public k-means|| initializer exposed; init controls added to Config

### src/index/kmeans_elkan.cpp
- [x] Fixed (2025-10-22): Elkan squared-distance pruning correctness; tightened inertia accounting

### src/index/ivf_pq.cpp
- [x] Fixed (2025-10-22): Projection coarse assigner now sets ANN telemetry (ann_enabled) and increments counters unconditionally when Projection is selected
- [ ] Low: Multiple compiler warnings about hidden/unused locals (e.g., lines ~1076, 1378, 1514, 1542, 1771, 3027, 3296, 3297, 3761, 4316, 4614, 4782, 4812, 4922). Clean up unused variables; consider scoping/renaming to avoid shadowing.

### Benchmarks
- [x] Added (2025-10-22): `bench/micro/kmeans_init_bench.cpp` – measures k-means++ vs k-means|| init time; exposes final inertia and iterations as counters
- [ ] Medium: Add runbook + scripts to emit CSV/JSON artifacts in CI with fixed seeds and environment manifests

### Documentation
- [x] Added (2025-10-22): API reference section for IVF-PQ k-means initialization; guidance on when to use k-means||; field docs for IvfPqTrainParams

---

Notes:
- This audit document will be expanded in subsequent passes:
  1) Public headers in `/include/vesper/` (correctness, ABI, docs)
  2) Core index implementations under `/src/index/`
  3) Tests (unit/integration): coverage gaps, missing invariants
  4) Benchmarks: gaps vs acceptance gates (perf, determinism artifacts)

Acceptance for this PR iteration:
- Benchmarks and docs completed for k-means||
- Audit skeleton created; next pass will enumerate public headers under `/include/vesper/`.


### include/vesper/index/hnsw.hpp (deeper pass)

- [ ] Medium: Concurrency docs and "lock-free" claim are ambiguous
  - Location: header preamble (11–15); add/add_batch/search docs (96–112, 115–139)
  - Details: Header states "Lock-free concurrent search operations" and "Build is single-threaded". Implementation uses per-node mutexes during search unless read_only_graph is set, and provides internally parallelized add_batch. Wording should reflect operation-level guarantees (search concurrent; structural mods serialized) and conditional lock elision.
  - Web validation: hnswlib and FAISS expose thread-safe search with serialized build; no claims of unconditional lock-free unless using specialized lock-free designs.
  - Recommendation: Rephrase thread-safety section; document set_read_only() semantics and constraints.
- [ ] Medium: Parameter bounds incomplete in header docs
  - Location: HnswBuildParams (30–42); HnswSearchParams (44–50)
  - Details: Specify constraints: M≥2; efConstruction≥M; max_M∈[M, …]; max_M0∈[2M, …] or auto; num_threads≥0; efConstructionUpper = 0 or ≥1. For search: efSearch≥1; k≥1; if filter_mask!=nullptr then filter_size≥ceil(n/8).
  - Web validation: Malkov & Yashunin (2018), hnswlib parameter guidance; FAISS HNSW recommends efSearch≥k.
  - Recommendation: Expand preconditions; add “Errors” paragraphs in header matching implementation error codes.
- [ ] Low: Metric is L2-only (not documented explicitly)
  - Location: Class docs (62–66)
  - Details: Implementation hardcodes L2 squared; clarify that current HNSW variant is L2-only.
  - Recommendation: Document metric support; future-proof by noting possible IP/COS variants.

### src/index/hnsw.cpp (deep review)

- [ ] High: Neighbor selection deviates from HNSW Algorithm 4 (Robust Prune)
  - Location: robust_prune() (1506–1570)
  - Details: Heuristic uses only ratio vs nearest (dist_c ≤ 1.5×closest) and lacks pairwise diversity check against already selected set R. This diverges from Algorithm 4 where a candidate c is discarded if ∃r∈R with dist(r,c) < dist(q,c). Can harm connectivity/diversity and recall.
  - Web validation: Malkov & Yashunin, 2018 (Algorithm 4); hnswlib selectNeighborsHeuristic; FAISS HNSW neighbor selection.
  - Recommendation: Implement pairwise diversity check over R (with early-exit), keep extend_candidates/keep_pruned semantics, maintain deterministic tie-breaking.
- [ ] High: Serialization incomplete/unstable
  - Location: save() (1422–1465); load() (1467–1470)
  - Details: save() writes raw HnswBuildParams struct and node payloads with magic "HNSW0001" but no checksum/version fields beyond magic; load() is stub returning default index. Raw struct write risks ABI/layout drift (bool size, padding, endianness), violating serialization gates.
  - Web validation: Vesper serialization rules (v1.0/v1.1), checksum and backward-compat requirements.
  - Recommendation: Define versioned schema (v1.1) with fixed-width fields, explicit endianness, checksums; implement full load(); avoid raw struct dumps.
- [ ] Medium: init() overrides user-specified max_M/max_M0
  - Location: init() (189–191)
  - Details: state_.params.max_M = M; max_M0 = max(2*M, M) unconditionally, ignoring user overrides provided in HnswBuildParams.
  - Recommendation: Honor caller-provided max_M/max_M0 when non-zero; otherwise derive from M.
- [ ] Medium: Base-layer acceptance bias not in spec
  - Location: connect_node() (457–465)
  - Details: Applies 0.95× bias to dist_to_new when neighbor base list saturated to prefer accepting the new edge. Not in HNSW papers; may skew degree distribution and determinism.
  - Web validation: Malkov & Yashunin; hnswlib does not bias distances this way.
  - Recommendation: Remove or guard behind a debug/experimental flag; measure connectivity/recall impact; keep deterministic tie-breaking.
- [ ] Medium: Entry validation gaps in add()/search()
  - Location: add() (582–647); search() (1019–1061)
  - Details: add() lacks explicit null-pointer guard for data and dimension check before assignment; search() doesn’t validate efSearch≥1, k≥1 (though it truncates to candidates.size()).
  - Recommendation: Add precondition checks with error_code::precondition_failed and actionable messages.
- [ ] Medium: Concurrency semantics vs docs; determinism notes
  - Location: search_layer neighbor copy (300–315); add_batch_parallel (757–820)
  - Details: Search uses per-node locks unless read_only_graph=true; add_batch uses chunk-based RNG seeding (seed+start) for deterministic levels. These behaviors aren’t documented in header.
  - Recommendation: Document conditional lock elision and determinism properties; specify that different num_threads/chunking could affect build order if policies change.
- [ ] Low: Hot-path backend selection repeated per call
  - Location: search_layer() (349–355)
  - Details: Re-selects kernels::select_backend_auto() inside loop instead of using cached state_.ops.
  - Recommendation: Use cached state_.ops for distance loop.
- [ ] Low: Memory layout fragmentation
  - Location: HnswNode (51–57); nodes_ allocations
  - Details: Per-node std::vector\<float\> and neighbor vectors lead to many small allocations; hurts cache locality.
  - Recommendation: Consider SoA/arena for node data and adjacency lists; measure before changes.
- [ ] Info: Level generator matches standard exponential scheme
  - Location: level_multiplier set (196–200); select_level() (223–227)
  - Details: Uses ml = 1/log(M) and level = floor(-log(U) * ml), consistent with hnswlib/FAISS.
  - Recommendation: Keep; document in header.

### include/vesper/index/hnsw.hpp (second pass)

- [ ] Medium: Parameter constraints not fully specified in header docs
  - Location: HnswBuildParams (30–42), HnswSearchParams (45–50)
  - Details: Add explicit bounds and relationships: M≥2; max_M≥M; max_M0≥2*M; efConstruction≥M; efSearch≥k; k≥1; dim>0; filter_mask either null or filter_size>0. Clarify determinism expectations with seed (same seed → stable graph given fixed toolchain and thread schedule).
  - Recommendation: Expand header preconditions and cross-link API_REFERENCE.md; enumerate error codes on violations.
- [ ] Medium: Concurrency/thread-safety docs are ambiguous
  - Location: Preamble (8–15), add/add_batch docs (96–114)
  - Details: Preamble says "Build is single-threaded; search operations are thread-safe," yet add_batch claims internal parallelization. Clarify that insertion is serialized per node but data preparation and search_batch may parallelize; document read-only mode semantics and any lock elision.
  - Recommendation: Tighten docs; document memory model for concurrent search (no locks on read paths; per-thread visited tables; no mutation when read-only).
- [ ] Low: Metric is L2-only (implicit)
  - Location: Class preamble (62–66)
  - Details: Make explicit that current implementation uses squared L2 distance; non-L2 metrics are unsupported.
  - Recommendation: Add a sentence and TODO for future metrics if planned.

### src/index/hnsw.cpp (deep review)

- [ ] High: RobustPrune deviates from HNSW Algorithm 4 (pairwise diversity missing)
  - Location: robust_prune() (1505–1577)
  - Details: Current heuristic keeps nearest, then rejects candidates only if dist_c ≤ 1.5×closest. It does not check pairwise distance against each already-selected neighbor R as required by Algorithm 4 (keep c only if ∀r∈R: dist(c,r) > dist(c,q)). This can reduce graph diversity/connectivity and harm recall.
  - Web validation: Malkov & Yashunin 2016/2018 Algorithm 4; hnswlib getNeighborsByHeuristic2; FAISS HNSW::shrink_neighbor_list (pairwise check dist(v1,v2) < dist(v1,q) → discard).
  - Recommendation: In Phase 2, implement proper pairwise diversity check (batch distance OK) and add unit tests comparing against FAISS/hnswlib behavior on small graphs.
- [ ] High: Serialization incomplete/unstable; load() stubbed
  - Location: save()/load() (1422–1470)
  - Details: save() writes magic "HNSW0001" + raw HnswBuildParams and nodes without checksum/versioned sections; load() returns a default index. This violates Vesper serialization gates (v1.0/v1.1 roundtrip, checksums, backward compatibility).
  - Web validation: Vesper rules (serialization gates); FAISS serializes with versioning and structured sections.
  - Recommendation: Phase 2: implement v1.1 sectioned format with header, schema version, checksums; validate on load; add roundtrip tests and backward-compat fixtures.
- [ ] Medium: User-provided max_M/max_M0 silently overridden
  - Location: init() (189–191)
  - Details: Derives max_M=max_M and max_M0=max(2*M, M) unconditionally, ignoring user overrides. Surprising API behavior.
  - Recommendation: Only default these when explicitly zero; otherwise honor caller-provided values.
- [ ] Medium: Base-layer acceptance bias (not in spec)
  - Location: connect/prune path (457–465)
  - Details: Applies a 0.95× bias to dist_to_new when base layer neighbor list is saturated to "keep new". This heuristic is undocumented and not in HNSW spec; may skew degree distribution and determinism.
  - Recommendation: Remove or gate behind a parameter with docs; measure recall/latency impact before adopting.
- [ ] Medium: Entry validation gaps in add()/search()/search_batch
  - Location: add (582–737), search (1014–1062), search_batch (1064–1110)
  - Details: Some preconditions are indirectly enforced, but explicit checks are missing (non-empty index before search; k>0; efSearch≥k; ids/data non-null; dim match). Exceptions are used in search_batch for SEH translation; prefer consistent error propagation.
  - Recommendation: Add explicit precondition checks with core::error codes; document behavior.
- [ ] Medium: Concurrency/determinism notes
  - Location: search_batch uses OpenMP (1074–1109); global state usage
  - Details: Parallel search is safe, but batch scheduling may be non-deterministic across runtimes. Document determinism guarantees (per-query results deterministic; cross-thread execution order not).
  - Recommendation: Document; add deterministic test with fixed seeds and repeated searches.
- [ ] Low: Hot-path backend selection repeated per call
  - Location: search layer distance loop (349–355)
  - Details: Re-selects kernels::select_backend_auto() per batch compute despite caching state_.ops in init. Minor overhead.
  - Recommendation: Use cached ops pointer/reference.
- [ ] Low: Memory layout fragmentation
  - Location: Node owns std::vector per level for neighbors
  - Details: Many small allocations; cache locality could suffer on large graphs.
  - Recommendation: Consider contiguous adjacency arenas or slab allocators in a future perf pass.
- [ ] Info: Level generator matches exponential scheme
  - Location: init() (196–199)
  - Details: Uses ml = 1/log(M) and exponential distribution; aligns with FAISS/hnswlib.

- Test recommendations (Phase 2):
  - Add RobustPrune parity tests vs FAISS/hnswlib on toy graphs (assert pairwise-diversity property).
  - Serialization roundtrip/backward-compat tests with checksums (v1.0/v1.1) and corruption detection.
  - Determinism tests: fixed seed → identical graph/levels; repeated searches → identical results; cross-platform tolerances documented.
  - Connectivity invariants: base-layer reachability ≥99% on random datasets; reciprocal edge checks within degree bounds.

- Cross-references:
  - Tests: tests/unit/hnsw_test.cpp; tests/unit/hnsw_basic_invariant_test.cpp; tests/unit/hnsw_connectivity_test.cpp; tests/unit/hnsw_lockfree_test.cpp
  - External refs: Malkov & Yashunin (arXiv:1603.09320); hnswlib (getNeighborsByHeuristic2/shrink); FAISS HNSW (shrink_neighbor_list, add_link)

### include/vesper/index/disk_graph.hpp (deep review)

- [ ] Medium: Parameter and precondition docs incomplete
  - Location: VamanaBuildParams (44–57), VamanaSearchParams (60–69)
  - Details: Specify constraints and relationships: degree≥1; alpha>1 (DiskANN uses α>1 to retain long edges); L≥degree; max_iters≥1; pq_m≥1; pq_bits∈{4,8}; shard_size≥1; num_threads≥0; seed fixed for determinism. For search: beam_width≥1; L≥beam_width; k≥1; cache_nodes≥0; if filter_mask!=nullptr then filter_size≥ceil(N/8).
  - Web validation: DiskANN (NeurIPS 2019, Microsoft Research) and Filtered‑DiskANN (SIGMOD 2023) parameter guidance; MSR DiskANN repo README.
  - Recommendation: Expand header preconditions and enumerate error codes returned on violations.
- [ ] Medium: API surface completeness — several declared methods lack implementation in .cpp
  - Location: Methods declared (183–187 add_vectors), (207–211 batch_search), (260–261 warmup_cache), (275–276 clear_cache)
  - Details: No corresponding definitions in src/index/disk_graph.cpp. Linking will fail if these APIs are used.
  - Recommendation: In Phase 2, either implement or temporarily remove from public surface; document availability.
- [ ] Low: Thread‑safety and metric support wording
  - Location: File header (13–15)
  - Details: Preamble states “Build is single‑threaded; search operations are thread‑safe,” but build params include num_threads and sharding; metric is implicitly L2 (squared) only.
  - Recommendation: Clarify concurrency guarantees; explicitly document metric = L2 squared at present; add notes about PQ‑assisted distance flag semantics.

### src/index/disk_graph.cpp (deep review)

- [ ] High: Serialization format inconsistent and likely corrupted (duplicate writes; mismatched readers)
  - Location: save() mmaps and writes header+offsets (795–814) and adjacency data (815–825), then also streams to the same files again (adjacency 874–879; vectors 893–896). get_neighbors() expects an offsets header (664–682), but load() populates in‑memory graph by reading degree+adjacency sequentially (971–977) and does not initialize graph_offsets_.
  - Details: Writing the same path (graph.adj, vectors.bin) via both mmap and iostreams produces conflicting layouts. Readers mix two incompatible formats (header/offset vs degree‑prefixed adjacency). Memory‑mapped read path is unused/uninitialized on load.
  - Web validation: Vesper serialization gates (v1.1 sectioned, checksums); MSR DiskANN persists explicit headers and adjacency blocks with versioning.
  - Recommendation: Define a single, versioned on‑disk format with explicit magic, endianness, per‑section sizes, and checksums. Write once via a single IO path; align readers. Populate graph_offsets_ on load when using the header/offset layout. Add roundtrip + corruption tests.
- [ ] High: RobustPrune α‑diversity check mixes Euclidean semantics with squared distances
  - Location: robust_prune_internal() (1229–1236); free robust_prune() (1671–1678)
  - Details: Distances are squared L2, but domination uses inter_dist < alpha · dist_i. For Euclidean‑intended α, the correct squared‑distance analogue is inter_dist < (alpha^2) · dist_i. Current test is stricter (uses √α effectively), over‑pruning neighbors and harming connectivity/recall.
  - Web validation: Vamana pruning (DiskANN) uses α on Euclidean distances; when operating in squared space, thresholds must be adjusted accordingly. See MSR DiskANN paper and reference implementations.
  - Recommendation: In Phase 2, either operate in Euclidean space for the α check or scale threshold by alpha^2 when using squared distances; add unit tests on toy graphs verifying diversity and degree bounds.
- [ ] Medium: Disk I/O path does synchronous per‑neighbor vector loads in beam_search; no prefetching
  - Location: beam_search() (1270–1314, 1346–1365)
  - Details: For each expansion, vectors are loaded via std::ifstream with no batched prefetch or readahead; neighbor lists are fetched from cache or in‑memory graph only (async neighbor I/O pathways exist but are unused). Severe random‑IO amplification on HDD/SSD.
  - Web validation: DiskANN designs batching/prefetch (PQ distance first, then exact refine) to reduce random I/O.
  - Recommendation: Introduce batched async prefetch of neighbor vectors (queue up next frontier); consider PQ‑assisted screening before exact loads when use_pq_distance=true.
- [ ] Medium: Search parameters are not enforced/used
  - Location: VamanaSearchParams.use_pq_distance, filter_mask/filter_size, cache_nodes, init_list_size (60–69 in header); usage in Impl::search/beam_search (468–510, 1250–1367)
  - Details: PQ‑assisted distance, filtering, and cache sizing are unused; init_list_size used only when entry_points are empty.
  - Recommendation: Validate and honor these params; implement bitmap filtering and optional PQ‑screened beam.
- [ ] Medium: Parameter validation gaps in build()
  - Location: build() (402–422)
  - Details: Checks only basic shape; missing degree≥1; L≥degree; alpha>1; max_iters≥1; pq_m≥1; pq_bits∈{4,8}; shard_size≥1.
  - Recommendation: Add explicit precondition errors with actionable messages.
- [ ] Medium: Incomplete mmap/read path
  - Location: load() (917–1007); get_neighbors() mmap path (644–651)
  - Details: load() does not mmap existing files nor populate graph_offsets_; get_neighbors() mmap branch depends on offsets populated elsewhere.
  - Recommendation: Add open_mmap_read() and initialize offsets/counters; or remove mmap read path until complete.
- [ ] Medium: Graph construction quality concerns (baseline OK but simplistic)
  - Location: build_initial_graph() (1049–1069); refine_graph() (1071–1113); greedy_search_internal() (1133–1197)
  - Details: Initial graph is random; refinement uses greedy L‑list search + RobustPrune with undirected reciprocal insertion. Reasonable baseline but may diverge from Vamana nuances (e.g., medoid seeding, candidate list management). No tests cover this.
  - Web validation: Vamana recommends careful candidate maintenance and α‑diversity; MSR codebases document additional heuristics.
  - Recommendation: Add correctness/quality tests (connectivity, recall vs brute force) and compare against reference implementations on small datasets.
- [ ] Medium: Determinism and threading
  - Location: VamanaBuildParams.num_threads/two_pass/shard_size unused; search is read‑only but uses caches without documented determinism guarantees.
  - Recommendation: Document determinism expectations (same seed → same graph); note platform and thread‑schedule caveats; either use or remove thread/shard/two_pass params.
- [ ] Medium: API surface gaps vs usage
  - Location: IndexManager usage (src/index/index_manager.cpp ~1046–1060; 1753–1761)
  - Details: DiskGraphIndex is instantiated and build() is used in conversion/creation flows. Absence of add_vectors/batch_search may block planned features.
  - Recommendation: Align API with IndexManager expectations; stub or implement missing pieces.
- [ ] Low: Performance hot spots
  - Location: greedy_search_internal visited set (1148–1156, 1188–1196)
  - Details: std::unordered_set for visited incurs heap traffic; typical ANN implementations use fixed‑size bitsets or epoch‑tagged arrays for O(1) marking.
  - Recommendation: Consider bitset/epoch table for visited; micro‑benchmark.
- [ ] Low: Unused/placeholder cache hooks
  - Location: comments “cache temporarily disabled” (1167–1169), (1486–1493), (1495–1500)
  - Details: Indicates planned LRU cache integrations currently disabled.
  - Recommendation: Track with TODOs and tests before re‑enabling.

- Cross‑references
  - Tests: No disk_graph unit tests found under tests/unit (directory listing reviewed)
  - Usage sites: src/index/index_manager.cpp (conversion to DiskANN; build path)
  - External refs: Microsoft Research DiskANN (https://github.com/microsoft/DiskANN); Filtered‑DiskANN (SIGMOD 2023); NeurIPS 2019 DiskANN paper.


### include/vesper/index/cgf.hpp

- [ ] Medium: Parameter constraints and preconditions not fully specified
  - Location: CgfBuildParams (33–60); CgfSearchParams (63–83)
  - Details: Bounds/relationships not stated: n_super_clusters≥1 or 0=auto; n_projections≥1; nlist≥1; pq_m≥1 and dim%pq_m==0 at train/add; pq_nbits∈{4,8}; mini_hnsw_M≥2; mini_hnsw_ef≥mini_hnsw_M; cluster_graph_threshold≥1; num_threads≥0; training_sample_size≥1; predictor_epochs≥0. Search: k≥1; max_probe≥0; rerank_pool_size≥k when use_exact_rerank.
  - Web validation: FAISS IVF/HNSW parameter guidance; Vesper header style in ivf_pq.hpp and hnsw.hpp requires explicit preconditions.
  - Recommendation: Expand header docs with explicit constraints and enumerate error codes returned on violations.

- [ ] Medium: Metric support not documented (implicitly L2 squared)
  - Location: Class preamble and search docs (184–239)
  - Details: CGF currently assumes L2 squared throughout the cascade (coarse projections, PQ filtering, mini‑HNSW). Header does not state metric; callers may assume cosine/IP support.
  - Web validation: FAISS documents metric per index type; HNSW (L2/IP) variants documented explicitly.
  - Recommendation: State metric = L2 squared for v1; note roadmap for other metrics.

- [ ] Medium: Determinism and threading guarantees not specified
  - Location: Class docs (184–291)
  - Details: Build/search concurrency and determinism are unspecified: whether search is thread‑safe; whether add/save/load may run concurrently; determinism of learned probing and mini‑graph builds given seeds.
  - Web validation: Vesper rules require explicit thread‑safety and determinism notes in public headers.
  - Recommendation: Document: search is thread‑safe; train/add/optimize/save/load are exclusive; same seeds → identical behavior (subject to fixed toolchain) or explicitly caveat experimental status.

- [ ] Low: “confidence_threshold” naming conflates recall/probe semantics
  - Location: CgfBuildParams (41–42) and CgfSearchParams (74–75)
  - Details: Field name implies statistical confidence but used as recall/probing target internally; could confuse users.
  - Recommendation: Clarify semantics or rename in a future version; keep API stable for now and document mapping (e.g., used to cap cluster probes).

- [ ] Low: Serialization/ABI notes missing in header
  - Location: save()/load() docs (253–267)
  - Details: No mention of versioning/checksums/endianness; implementation is stubbed.
  - Recommendation: Document intended v1.1 sectioned format and backward‑compat expectations.

### include/vesper/index/cgf_capq_bridge.hpp

- [ ] Medium: Stage‑1/2/3 operate over full storage view (no cluster pre‑filter)
  - Location: CapqFilter::search (42–96)
  - Details: The CAPQ bridge scans all vectors for Hamming/q4/q8 stages. In CGF, CAPQ should be applied after cluster pre‑selection to avoid O(N) scans.
  - Web validation: Multi‑stage ANN systems (FAISS rerank, ScaNN) filter candidates before refinement.
  - Recommendation: Note in docs that bridge is a building block; when used under CGF, pass cluster‑restricted views.

- [ ] Medium: ID width truncation in bridge
  - Location: search() returns std::vector<std::uint32_t> (42–46, 93–95)
  - Details: Bridge downcasts IDs to 32‑bit; the top‑level CgfIndex upcasts back to 64‑bit. For datasets with >2^32 IDs this will collide.
  - Recommendation: Return 64‑bit IDs from the bridge or make width configurable; document current limitation.

- [ ] Medium: Projection path uses FWHT‑64 on padded input without normalization notes
  - Location: 47–57
  - Details: Non‑64D inputs are zero‑padded to next power‑of‑two and FWHT‑64 applied to first 64 entries. No note on normalization/energy scaling.
  - Web validation: FWHT is orthogonal up to normalization; CAPQ whitening follows, but clarify assumptions.
  - Recommendation: Document the projection behavior; in Phase 2 consider explicit normalization to preserve scale.

- [ ] Low: Determinism/thread‑safety not documented
  - Location: class docs (25–41), search (42–96)
  - Details: Seeds imply determinism; thread‑safety for read‑only search should be stated.
  - Recommendation: Add brief notes (read‑only operations are thread‑safe; initializer must happen before concurrent use).

### src/index/cascaded_geometric_filter.cpp (deep review)

- [ ] Medium: Non‑standard build structure — aggregates .cpps via #include
  - Location: includes of component TUs (24–27)
  - Details: TU includes other .cpp files (coarse_filter.cpp, hybrid_storage.cpp, smart_ivf.cpp, mini_hnsw.cpp). This complicates build and violates layering; CMake also comments out these sources.
  - Recommendation: In Phase 2, refactor to proper headers/implementation units and add to CMake; keep file local while auditing.

- [ ] Medium: Missing header for std::unordered_set
  - Location: use in search() progressive refinement (267–289)
  - Details: Uses std::unordered_set without including <unordered_set>. May compile by transitive includes but not guaranteed.
  - Recommendation: Include <unordered_set> explicitly.

- [ ] Medium: Candidate refinement and distance paths are stubs/mis‑specified
  - Location: search() Stage‑4 refinement (262–307)
  - Details: MiniHNSW search returns results but mapping back to storage uses hybrid_storage_->get_vector(id) and .mini_graph_idx without defined contract. Exact distance recompute path is incomplete and performs O(N·clusters) scans.
  - Recommendation: Specify reverse lookup contracts in HybridStorage (id→storage_idx); implement efficient mapping; add tests.

- [ ] Low: Statistics/telemetry are inconsistent
  - Location: CascadeStats (31–50) and usage (220–231, 260–321)
  - Details: get_stage1_reduction divides by total_queries*1000; overall_reduction computes stage4/total_queries. Units/methodology unclear; avg_recall unused.
  - Recommendation: Define telemetry precisely (per‑query counts, true elimination ratios, recall estimates) and reset/aggregate correctly.

- [ ] Low: Memory accounting underestimates dynamic allocations
  - Location: get_memory_usage() (340–352)
  - Details: Uses sizeof(SkylineSignature) × count; ignores internal vectors’ heap memory.
  - Recommendation: Expose detailed memory stats from components; label totals as estimates.

### src/index/cgf.cpp (deep review)

- [ ] High: Compile‑time bug — calls non‑existent CgfBuildParams::max_probe_clusters()
  - Location: train() config mapping (197–209), specifically 201
  - Details: CgfBuildParams has no member function max_probe_clusters(); helper exists as Impl::max_probe_clusters() at 421–427. This will not compile if this TU is built.
  - Recommendation: Phase 2 fix: use max_probe_clusters() helper or map from params.max_probe with fallback.

- [ ] High: search_batch parallel path writes into un‑sized vector with data race
  - Location: search_batch() (267–311)
  - Details: results.reserve(n_queries) without resize; threads assign results[q] under a mutex but ‘q’ index may be out‑of‑range; also pushes in sequential branch leading to inconsistent shape.
  - Recommendation: Resize results to n_queries; in parallel path write to pre‑sized slots; consider per‑thread buffers and join.

- [ ] High: Serialization is stubbed and not versioned
  - Location: save() (313–339); load() (341–368)
  - Details: Writes “CGF1” + dim only; no schema/versioning/checksums; load() sets trained_=true without reconstructing components.
  - Web validation: Vesper serialization gates (v1.1 sectioned, checksums, backward compat).
  - Recommendation: Implement v1.1 sectioned format; add roundtrip/compat tests.

- [ ] Medium: Temporary CAPQ‑only prefilter path in public search
  - Location: search() (254–263)
  - Details: Returns CAPQ prefilter IDs with distance=0.0 for diagnostics; violates API contract (distances are not meaningful).
  - Recommendation: Gate behind debug flag or keep internal; ensure public search always returns meaningful distances.

- [ ] Medium: Predictor and mini‑graph stubs lack contracts/tests
  - Location: ClusterPredictor::Impl (18–94); MiniHnsw::Impl (118–151)
  - Details: Both are scaffolding with simplified logic; no deterministic seeding documented; stats() fabricates edges from counts only.
  - Recommendation: Document experimental status; add contracts for determinism and accuracy gates before enabling in main build.

- [ ] Low: Non‑standard TU include of cascaded orchestrator
  - Location: #include "cascaded_geometric_filter.cpp" (10)
  - Details: Couples TU ordering and increases compile units’ size.
  - Recommendation: Refactor to headers/implementation in Phase 2.

### include/vesper/index/index_manager.hpp

- [ ] Medium: QueryConfig lacks target_recall; planner uses hardcoded 0.95
  - Location: QueryConfig (79–101); QueryPlanner uses fixed 0.95 target (see src/index/query_planner.cpp:328)
  - Details: Public QueryPlanner tests/spec anticipate a configurable target recall (e.g., config.target_recall), but QueryConfig does not expose it. The planner currently hardcodes 0.95, reducing tunability and causing header/tests/spec divergence.
  - Web validation: FAISS/ScaNN expose recall/accuracy vs. latency tradeoffs via parameters; configurable targets are common in cost-based planners.
  - Recommendation: Add target_recall to QueryConfig (header-only change) and thread it through planner scoring. Update docs and tests accordingly.

- [ ] Low: Quantization feature flags exposed in build config but integration unspecified
  - Location: IndexBuildConfig (62–76)
  - Details: enable_rabitq/enable_matryoshka and related fields are exposed, but their semantics at the manager level (e.g., how they affect add/search) are not documented.
  - Recommendation: Document contracts at IndexManager level (what transforms are applied; persistence/compat guarantees; determinism and seeds).

- [ ] Low: Thread-safety guarantees not fully specified for hybrid mode
  - Location: Class doc/preamble and method comments (135–199)
  - Details: Search is documented thread-safe; add/update are safe with concurrent reads. When multiple indexes are active (Hybrid), clarify cross-index consistency guarantees during concurrent updates and search.
  - Recommendation: Expand header docs: define read/write epoch model; state that results reflect last committed epoch; document rebuild/optimization effects.

### src/index/index_manager.cpp

- [ ] High: QueryPlanner tuning discarded; plan.config not applied
  - Location: search() (351–366 selects plan.index; 381–418/493–503 use config.ef_search/nprobe/l_search instead of plan.config)
  - Details: Planner returns QueryPlan with tuned parameters (ef_search/nprobe/l_search), but IndexManager ignores these and uses the caller’s base config. This defeats cost-based optimization and makes planner tests misleading.
  - Web validation: Cost-based planners must execute the chosen plan, including tuned parameters.
  - Recommendation: Apply plan.config when invoking the selected index. Record actual time/recall back into planner stats.

- [ ] High: ID semantics inconsistent for DiskGraph (internal 32-bit vs external 64-bit)
  - Location: search() conversion (498–503); apply_diskgraph_batch() ignores all_ids (842–854); update() existence check (631–634); tombstone calls cast to uint32 (833, 649, 673, 789)
  - Details: DiskGraph uses contiguous internal IDs [0..N), but IndexManager surfaces external 64-bit IDs elsewhere. Manager returns DiskGraph internal IDs to users, checks existence by querying get_vector(external_id), and rebuilds ignoring the preserved all_ids list. Tombstone and metadata paths down-cast to 32-bit. This breaks cross-index coherence and can corrupt ID space once updates occur.
  - Web validation: FAISS and vector DBs maintain consistent external IDs with explicit mapping layers.
  - Recommendation: Introduce a stable external↔internal ID mapping for DiskGraph within IndexManager (or inside DiskGraph); rebuild must preserve mapping; use 64-bit throughout public surfaces; avoid narrowing casts.

- [ ] High: optimize() returns early; conversion/repair logic is dead code
  - Location: optimize() (956 returns {}; code at 961–1076 unreachable)
  - Details: The method exits before executing compaction/repair/index-conversion logic; comments reference an incremental repair coordinator. As written, optimize() is a no-op aside from tombstone compaction pre-checks.
  - Recommendation: Remove early return or refactor phases behind feature gates; ensure conversion logic executes and is tested.

- [ ] High: Serialization manifest uses raw binary of enums/structs; no checksums or endianness
  - Location: save()/load() (1094–1161, 1189–1323)
  - Details: Writes enum/struct bytes via reinterpret_cast without endianness/versioned sections or integrity checks. This violates Vesper’s serialization/ABI stability gates and is not portable across compilers/architectures.
  - Web validation: Vesper v1.1 sectioned formats with checksums/backward-compat gates.
  - Recommendation: Move to sectioned v1.1 format with explicit field tags, sizes, checksum; validate on load; document compatibility.

- [ ] Medium: Planner/manager stats never updated; memory enforcement heuristics ineffective
  - Location: stats fields (1813–1817); get_stats() usage (871–906); enforce_memory_budget() usage_score (1447–1464)
  - Details: query_count/total_time_ms/measured_recall are never incremented/updated during search, so usage_score=0 and adaptation never triggers.
  - Recommendation: In search paths, record timings and optionally measured recall when known; update per-index stats and feed back into planner via update_stats().

- [ ] Medium: Hardcoded temporary path breaks portability
  - Location: enforce_memory_budget() temp save path (≈1531; “/tmp/…”)
  - Details: POSIX-specific path; fails on Windows and is not configurable.
  - Recommendation: Use platform utils (safe_getenv("TMPDIR"/"TEMP") with fallback), or accept a tmp directory in config.

- [ ] Medium: Environment overrides applied per call can break determinism
  - Location: apply_rerank_env_overrides() used in search() (372–377)
  - Details: Reading env vars at query time can change behavior mid-run and across processes, violating determinism guarantees unless documented.
  - Recommendation: Snap env-derived settings at build() or constructor time; document determinism; optionally guard behind a debug/diagnostic flag.

- [ ] Medium: DiskGraph batch rebuild ignores IDs and tombstones width
  - Location: apply_diskgraph_batch() (831–846)
  - Details: Rebuild collects all_ids but never uses them; existing/tombstoned handling narrows to uint32. External IDs are lost.
  - Recommendation: Preserve external IDs and apply tombstone filter in 64-bit; ensure rebuilt index maintains mapping.

- [ ] Low: search_batch is sequential; no parallelization
  - Location: search_batch() (566–581)
  - Details: Simple for-loop over queries; misses easy speedup via parallel_for with bounded threads. Correctness OK.
  - Recommendation: Consider parallelization with stable ordering or return-permutation.

- [ ] Low: Exact rerank uses local kernels instead of centralized dispatch
  - Location: l2_sqr_scalar/avx2/l2_sqr_fast in anonymous namespace (46–132); used at 449–453
  - Details: Duplicates distance logic rather than using kernels/dispatch.hpp, risking divergence.
  - Recommendation: Route through kernels dispatcher for consistency, SIMD caps, and testing coverage.

### include/vesper/kernels/distance.hpp

- [ ] Medium: Cosine zero-norm semantics not uniform across backends
  - Location: cosine_similarity (110–155)
  - Details: Scalar path assumes precondition (norms > 0) and divides by denom without guard. AVX backends explicitly guard and return 0.0 when denom==0. This creates backend-dependent behavior if precondition is violated (e.g., zero vectors), harming determinism.
  - Web validation: FAISS rejects zero-norm for cosine or normalizes; numerical best practice is to define explicit behavior for degenerate norms.
  - Recommendation: Specify and enforce a single contract for zero-norm cases (e.g., return 0 similarity and 1 distance) across all backends; add tests.

- [ ] Low: No non-negativity clamp for L2^2; Accelerate backend clamps
  - Location: l2_sq (38–75)
  - Details: Scalar/AVX paths can return tiny negative values due to FP round-off; Accelerate path clamps to ≥0.0. Cross-backend drift possible.
  - Web validation: Kahan/Higham on stable summation; common practice clamps small negatives to 0 for squared norms.
  - Recommendation: Either document current behavior or clamp non-negative consistently across backends; add parity tests.

- [ ] Low: Missing noexcept on reference kernels
  - Location: l2_sq (38), inner_product (77), cosine_similarity (110), cosine_distance (157)
  - Details: Hot-path kernels are not marked noexcept; wrapper functions in scalar backend are noexcept. Minor API inconsistency.
  - Recommendation: Mark reference kernels noexcept for consistency with KernelOps signature.

- [ ] Medium: Numeric robustness — cosine denom can underflow to 0 → NaN/Inf
  - Location: 153–155 (denominator computation and division)
  - Details: `denom = sqrt(na2_total) * sqrt(nb2_total); return dot/denom;` relies on precondition (norms > 0). For subnormal inputs, `na2_total` or `nb2_total` may underflow to 0, yielding division-by-zero → Inf/NaN, violating Vesper numeric policy (no NaN/Inf from distance kernels).
  - Web validation: IEEE 754-2019 (division-by-zero and NaN propagation) — authoritative reference.
  - Recommendation: Keep scalar path as reference but add a documented "safe" variant (e.g., `cosine_similarity_safe`/`cosine_distance_safe`) that clamps the norm with epsilon (documented) to avoid NaN/Inf; alternatively, clamp denom via `std::max(denom, eps)` with a comment about semantics. Ensure unit tests cover tiny‑norm vectors.

- [ ] Low: Cosine range not clamped; cosine_distance may become slightly negative
  - Location: 157–160 (`return 1.0f - cosine_similarity(...)`)
  - Details: Due to FP error, cosine can be slightly >1 or <−1. Then `1 - cos` can be negative or >2 by tiny epsilons, which may surprise callers that assume distance ∈ [0,2].
  - Recommendation: Optionally clamp cosine to [−1,1] (or clamp distance to [0,2]) in the safe variant; document that the raw variant may have small ULP excursions and that ranking is unaffected within tolerances.

- [ ] Low: Prefetch stride hard-coded to +16 elements
  - Location: 50–54 (`scalar_prefetch(pa + i)`/`scalar_prefetch(pb + i)` with `i+16 < n`)
  - Details: The prefetch helper adds `+16` elements internally; guarded by `i+16 < n`. Functionally correct, but the stride is implicit and not dimension-aware.
  - Recommendation: Add a brief comment documenting the prefetch distance and cache-line intent; consider centralizing prefetch distance in platform constants to avoid duplication with intrinsics wrappers.

### src/kernels/dispatch.cpp

- [ ] Medium: AVX-512 backend header included on all x86_64 builds
  - Location: Includes (11–14)
  - Details: avx512.hpp is included even when `__AVX512F__` is not defined; although calls are gated, some toolchains may still parse intrinsics and require target flags. Portability risk.
  - Web validation: Typical pattern is to guard includes with feature macros or provide stub headers.
  - Recommendation: Wrap avx512.hpp include with `__AVX512F__` or provide a stub get_avx512_ops() declaration when not available.

- [ ] Medium: Accelerate l2 uses aa+bb−2ab identity; potential cancellation
  - Location: l2_sq_accel (27–38)
  - Details: Identity can lose precision for near-identical vectors (catastrophic cancellation) relative to FMA(diff,diff). Behavior differs from scalar/AVX paths.
  - Web validation: Intel/FAISS implementations favor direct diff^2 accumulation with FMA for stability.
  - Recommendation: Consider using the same accumulation scheme or document/cover with tolerances; clamp ≥0 retained.

- [ ] Low: Env-based backend overrides affect reproducibility across runs
  - Location: get_simd_override()/get_backend_name_override() (143–188)
  - Details: Backend selection via env vars can change results across executions; values are snapped once but behavior is externally mutable.
  - Recommendation: Document determinism implications; allow explicit configuration via API to supersede env.

### include/vesper/kernels/backends/stub_avx2.hpp (cross-ref)

- [ ] High: KernelOps initializer missing batch function pointers (compile-time error)
  - Location: get_stub_avx2_ops() (21–23)
  - Details: KernelOps currently has 6 members (incl. batch funcs) but stub initializer provides 4; this will not compile in any TU including scalar.hpp.
  - Web validation: N/A (API conformance).
  - Recommendation: Update stub to include batch_l2_sq and batch_inner_product pointers (delegating to scalar), or remove/guard stub include.

### include/vesper/kernels/backends/stub_neon.hpp (cross-ref)

- [ ] High: KernelOps initializer missing batch function pointers (compile-time error)
  - Location: get_stub_neon_ops() (21–23)
  - Details: Same mismatch as AVX2 stub; breaks builds including scalar.hpp on platforms with these headers.
  - Recommendation: Mirror scalar backend by wiring batch function pointers; optionally mark NEON stub behind feature guard.

### include/vesper/kernels/backends/avx2.hpp (doc)

- [ ] Low: Cosine similarity doc claims range [0,1]
  - Location: Docstring for avx2_cosine_similarity (156–165; line 163)
  - Details: Cosine similarity range is [-1,1].
  - Recommendation: Fix documentation; add bounds test like in simd_kernels_comprehensive_test.

### include/vesper/kernels/backends/avx512.hpp (doc)

- [ ] Low: Cosine similarity doc claims range [0,1]
  - Location: Docstring for avx512_cosine_similarity (169–177)
  - Details: Cosine similarity range is [-1,1].
  - Recommendation: Fix documentation for correctness.

### include/vesper/index/fast_hadamard.hpp

- [ ] Medium: Unconditional <immintrin.h> include in public header
  - Location: includes (16)
  - Details: Header includes SIMD intrinsics unguarded; breaks portability on non‑x86 toolchains and violates header minimality. Implementation file already guards by platform.
  - Web validation: Intel Intrinsics require target flags; common practice is to guard includes or confine intrinsics to .cpp.
  - Recommendation: Guard with architecture macros or remove from header; keep intrinsics confined to .cpp to avoid leaking platform specifics.

- [ ] Medium: Concurrency/thread-safety not documented; mutable workspace shared
  - Location: class members (72–74); methods (42–57)
  - Details: Methods are const but mutate `workspace_` via `mutable`. Concurrent calls on the same instance race on shared buffer. Header claims “orthogonal transformations” but provides no thread-safety guarantees.
  - Web validation: Vesper concurrency policy requires explicit thread-safety docs.
  - Recommendation: Document: FastHadamard is not thread-safe for concurrent calls on the same instance; per-thread instances required. Optionally note that separate instances are safe.

- [ ] Low: `log2_dim_` appears unused
  - Location: member (71)
  - Details: Stored but not read in current implementation; minor maintenance smell.
  - Recommendation: Remove or use in loop setup; otherwise document intent.

- [ ] Medium: API/docs imply “orthogonal rotation” for non‑power‑of‑two dims without caveat
  - Location: class doc (20–25), ctor doc (28–33)
  - Details: Header says non‑power‑of‑two dims “will be padded”, but does not disclose that only first `dim_` outputs are written back. This truncation breaks orthonormality/invertibility for `dim_`≠power‑of‑two.
  - Web validation: FWHT is orthonormal on R^{2^k} with 1/√n scaling; truncating outputs is not energy‑preserving on the original R^{dim}.
  - Recommendation: Either (a) restrict to power‑of‑two dims at API level, or (b) document that for non‑power‑of‑two dims, transform is HD on padded space and results are truncated (not strictly orthonormal on R^{dim}). Consider returning padded outputs or adding a projection stage.

### src/index/fast_hadamard.cpp

- [ ] High: Non‑power‑of‑two handling discards energy and violates claimed orthogonality
  - Location: rotate_inplace (57–81)
  - Details: Input is zero‑padded to `padded_dim_`, transformed with HD, then only first `dim_` outputs are written back. For `dim_`≠power‑of‑two this is not an orthonormal mapping on R^{dim_}; energy spreads across all `padded_dim_` outputs and truncation distorts norms despite 1/√n scaling.
  - Web validation: FWHT orthogonality holds on full length with normalization; partial truncation is not energy preserving. See “Walsh–Hadamard transform” (orthonormal basis on length 2^k).
  - Recommendation: Document limitation prominently; in Phase 2 consider: (1) restrict to power‑of‑two dims; (2) produce full `padded_dim_` output; or (3) follow FJLT‑style random projection that preserves norms in expectation.

- [ ] Medium: Public header claims AVX2; implementation uses guards correctly but header leaks intrinsics
  - Location: fwht_avx2 (99–132) and header include (fast_hadamard.hpp:16)
  - Details: Implementation guards AVX2 path; header’s unconditional intrinsics include still risks portability.
  - Recommendation: Mirror dispatch pattern used in kernels (guard includes and/or confine to TU).

- [ ] Medium: Concurrency — shared `workspace_` across calls
  - Location: rotate_inplace (57–81); members (72–74 in header)
  - Details: `workspace_` is reused per call; using the same instance across threads causes data races.
  - Recommendation: Document non‑thread‑safe behavior; consider per‑call scratch (passed by caller) or thread‑local buffers (documented determinism impact) in future.

- [ ] Low: Parameter validation and docs (dim==0, stride semantics) are implicit
  - Location: ctor (41–55), rotate_batch (91–97)
  - Details: `dim==0` results in no‑op; `stride==0` defaults to dim_. These behaviors aren’t documented in header.
  - Recommendation: Add header notes on edge cases; optionally enforce `dim>0` as precondition.

### Cross‑references and CAPQ integration

- [ ] Info: CAPQ uses a dedicated FWHT‑64 path with correct normalization
  - Location: include/vesper/index/capq_util.hpp (32–34); src/index/capq_util.cpp (23–38)
  - Details: `fwht64_inplace` implements 6 butterfly stages and 1/8 scaling, matching orthonormal FWHT‑64. This is independent of FastHadamard and used by CAPQ sketch computation.
  - Recommendation: Keep CAPQ path; ensure any future unification maintains exact FWHT‑64 parity.

- [ ] Medium: No unit tests target FastHadamard/FastRotationalQuantizer directly
  - Location: tests/unit/ (no FWHT tests found; `capq_*` tests exist)
  - Details: SIMD and scalar kernels have comprehensive parity tests; FWHT classes lack direct tests (power‑of‑two dims, non‑power‑of‑two behavior, determinism with fixed seeds).
  - Recommendation: Add tests: (a) FWHT‑64 parity vs scalar reference; (b) norm preservation for power‑of‑two; (c) document and test non‑power‑of‑two behavior; (d) seed determinism; (e) concurrency negative test (UB avoided by design).

- [ ] Medium: FastRotationalQuantizer distance estimation is heuristic and undocumented
  - Location: estimate_l2_distance (246–319)
  - Details: Computes sum of squared code differences and divides by average scale squared; ignores per‑dimension offsets/scales in reconstruction. Accuracy trade‑offs undocumented.
  - Recommendation: Document approximation and expected error bounds; add tests comparing against exact L2 on reconstructed floats; consider per‑dim scaling in Phase 2.

### include/vesper/index/aligned_buffer.hpp (header-only)

- [ ] Medium: set_centroid()/from_vectors() allow partial copies without validation
  - Location: set_centroid (117–126), from_vectors (139–143)
  - Details: Copies source length without enforcing source.size()==dim_. Shorter inputs leave tail of centroid row unchanged (stale data); longer inputs silently truncate. Leads to correctness risk depending on call sites.
  - Web validation: C++ container best practices recommend validating sizes for fixed-size views; FAISS centroid I/O enforces exact dimension lengths.
  - Recommendation: Document precondition (source length must equal dim_). In Phase 2, add asserts or explicit error paths; optionally zero-fill remainder when shorter to avoid stale data.

- [ ] Medium: Index bounds and invariants not documented
  - Location: operator[]/get_centroid (98–114) and dimension/stride invariants (176–189)
  - Details: No bounds checking for i<k_; assumes stride_>=dim_ and alignment invariants. Callers can trigger OOB access if misused.
  - Recommendation: Document invariants and preconditions in header; consider debug-mode asserts.

- [ ] Medium: Concurrency/thread-safety unspecified
  - Location: class AlignedCentroidBuffer (87–189)
  - Details: API returns raw pointers/spans; concurrent writers/readers across threads may race; prefetch functions hint multi-thread use but no guidance on false sharing or per-thread partitioning.
  - Web validation: Vesper concurrency policy requires explicit thread-safety docs; cache-line alignment is per-row but contention can still occur.
  - Recommendation: Document: not thread-safe for concurrent writes to same row; rows start at 64B boundaries to reduce false sharing; per-thread disjoint row updates are safe.

- [ ] Medium: AlignedDistanceMatrix from_vectors() does not validate shape
  - Location: 236–243
  - Details: Uses min sizes and copies sub-rectangles without enforcing matrix[i].size()==k_. Risk of partial/garbage data in remaining entries.
  - Recommendation: Document precondition; in Phase 2 enforce consistent shapes or zero remaining cells.

- [ ] Low: Documentation oversell — “NUMA-aware allocation support” not implemented here
  - Location: File header (13–14)
  - Details: This header uses std::vector with AlignedAllocator (64B alignment). NUMA features live under include/vesper/memory/numa_allocator.hpp and are not integrated.
  - Recommendation: Reword to “compatible with NUMA allocators”; optionally add constructors taking allocator to integrate NUMA in Phase 2.

- [ ] Low: Prefetch addresses may cross row end
  - Location: prefetch_read/prefetch_write (163–175)
  - Details: Prefetches ptr and ptr+16 floats (two cache lines). For small dims this may prefetch beyond row boundary. Prefetch is generally safe but can trip analyzers/tools.
  - Web validation: Intel Software Optimization Manuals indicate prefetch of invalid addresses typically does not fault but is not guaranteed across all environments/tools.
  - Recommendation: Guard second prefetch when dim_<16; or document analyzer expectations; keep as perf hint.

- [ ] Low: Memory overhead from stride padding not documented
  - Location: align_to_cache_line/stride (178–189)
  - Details: Memory use is k × ceil(dim/16) × 16 floats. For small dims memory overhead can be significant.
  - Recommendation: Add note in docs and usage sites; expose accessor for padded stride (already present) and recommend using it for iteration.

- [ ] Low: AlignedAllocator compile-time constraints not documented
  - Location: AlignedAllocator (30–80)
  - Details: Requires Alignment to be power-of-two and >= alignof(T); enforced at runtime by platform::aligned_allocate. Not stated in template doc.
  - Recommendation: Add template docs and/or static_asserts; reference platform::aligned_allocate contract.


- [ ] Medium: AlignedAllocator::allocate throws in allocation paths used within training loops
  - Location: AlignedAllocator::allocate (49–61); used by std::vector in AlignedCentroidBuffer (87–189) and AlignedDistanceMatrix (195–258)
  - Details: Allocation failure throws std::bad_alloc / std::bad_array_new_length. While most allocations occur at construction, k-means paths create AlignedDistanceMatrix during updates; exceptions on performance-critical loops violate Vesper “no exceptions on hot paths”.
  - Recommendation: In Phase 2, offer non‑throwing construction via caller‑supplied buffers or PMR with error return (std::expected). Document allocation behavior; pre‑allocate where possible.

- [ ] Low: Public header uses std::span in API
  - Location: get_centroid()/row() (107–114, 220–223)
  - Details: std::span in public headers is fine for header-only consumption but is not a stable binary ABI across DSOs on older toolchains. Aligns with policy to prefer C API for cross‑DSO use.
  - Recommendation: Add note that C++ API is not ABI‑stable; prefer C API for cross‑boundary scenarios.

- [ ] Low: to_vectors()/from_vectors() are heavy utility helpers
  - Location: 128–143; 225–243
  - Details: Allocate/copy O(k·dim) data; not suitable for hot paths.
  - Recommendation: Mark as compatibility helpers only; recommend span‑based row access in performance‑sensitive code.

### Cross-references and usage

- ProductQuantizer codebooks
  - Location: src/index/product_quantizer.cpp (lines ~60–66, ~481–489)
  - Details: Uses AlignedCentroidBuffer for PQ codebooks; serialization reads/writes only dim elements per row (not stride). Ensure callers never write beyond dim.

- KmeansElkan aligned distance matrix
  - Location: include/vesper/index/kmeans_elkan.hpp (112–116); src/index/kmeans_elkan.cpp (135–147)
  - Details: Uses AlignedCentroidBuffer with AlignedDistanceMatrix and SIMD matrix routines.

- Kernels batch distances integration
  - Location: include/vesper/kernels/batch_distances.hpp (114–126, 482–498)
  - Details: AVX2 path assumes dimension multiple-of-8 for vectorized portion and handles remainder; benefits from per-row 64B alignment.

### include/vesper/core/memory_pool.hpp

- [ ] High: Overflow-unsafe exhaustion check can wrap and allow OOB
  - Location: MemoryArena::allocate (63–74), specifically `if (offset + bytes > size_)` at 68
  - Details: Uses `offset + bytes > size_` which can overflow `size_t` and pass the check, leading to out-of-bounds writes. Safer pattern is `if (bytes > size_ - offset)` after verifying `offset <= size_`.
  - Web validation: Common bump-allocator guidance requires power-of-two align and overflow-safe checks (see StackOverflow align discussion; Nick Fitzgerald “Always Bump Downwards”).
  - Recommendation: In Phase 2, change the condition to `if (offset > size_ - bytes)` or `if (bytes > size_ - offset)`; validate `alignment != 0` and power-of-two.

- [ ] High: Scope-escape hazard for pooled containers can cause UAF
  - Location: PoolScope dtor resets arena (210–212); pooled helpers (PooledVector/make_pooled_vector at 227–239)
  - Details: `PoolScope` calls `reset()` in its destructor, invalidating all storage from the arena. Any `PooledVector`/pmr containers allocated in that scope must not escape it. Returning/moving a pooled container beyond the scope → dangling storage and use‑after‑free.
  - Web validation: pmr containers rely on the lifetime of their upstream memory_resource; monotonic/arena resources require non‑escaping allocations.
  - Recommendation: Document strict non‑escape rule in header; add debug assertions in helpers (Phase 2) and usage guidance (examples). Consider `[[nodiscard]]` helper that returns a scope‑bound handle.

- [ ] Medium: Alignment handling assumes power‑of‑two; not validated
  - Location: MemoryArena::align_up (97–100); allocate(alignment) (63–74)
  - Details: `align_up` uses bit‑masking which is only correct if `alignment` is a non‑zero power of two. PMR calls satisfy this precondition, but direct `ThreadLocalPool::allocate(bytes, alignment)` is public and undocumented.
  - Web validation: C++ pmr do_allocate requires `alignment` to be a power of two (cppreference; WG21 N3916/N4617).
  - Recommendation: Document precondition; add defensive `assert((alignment & (alignment-1))==0 && alignment!=0)` in debug builds (Phase 2).

- [ ] Medium: Thread affinity and safety of allocator not documented
  - Location: ThreadLocalPool::allocator (148–151); class preamble (136–144)
  - Details: Allocator and resource are thread‑local and not safe to use across threads. Copying a `polymorphic_allocator` to another thread can cause data races.
  - Recommendation: Document allocator is valid only on the creating thread; do not share across threads; prefer per‑thread construction.

- [ ] Medium: TempBuffer does not run element destructors
  - Location: TempBuffer (243–280)
  - Details: Raw arena allocation with pointer semantics; no destructor calls for contained elements on scope end. Safe only for trivially destructible Ts.
  - Recommendation: Document constraint explicitly; for non‑trivial Ts, use pmr containers so destructors run even if deallocate is a no‑op.

- [ ] Low: usage_ratio division by zero if arena size is zero
  - Location: ThreadLocalPool::stats (171–177)
  - Details: If `MemoryArena` were constructed with `size==0`, `usage_ratio` divides by zero. Default path uses 64MB, so practical risk is low.
  - Recommendation: Document `size>0` precondition; optionally guard in Phase 2.

- [ ] Low: align_up duplication across modules
  - Location: MemoryArena::align_up (97–100) vs vesper::platform::align_up
  - Details: Duplicated helpers increase drift risk.
  - Recommendation: Reuse `vesper::platform::align_up`; remove local duplicate (Phase 2).

- [ ] Low: Memory budget/per‑thread footprint not documented
  - Location: MemoryArena::DEFAULT_SIZE (33–36); ThreadLocalPool ctor (181–184)
  - Details: 64MB per thread by default; many threads can consume significant RAM before reuse/reset.
  - Recommendation: Document knob and provide env/param override pattern in Phase 2.

- [ ] High: Exceptions thrown in allocation paths violate no-exceptions-on-hot-paths
  - Location: MemoryArena ctor (36–44; throw at 42); ArenaResource::do_allocate (114–121; throw at 117); TempBuffer (245–252; throw at 250)
  - Details: Throwing std::bad_alloc in pool/arena paths used by k-means Elkan M-step and potentially kernel staging makes these code paths exceptionful; Vesper policy prohibits exceptions on hot paths.
  - Web validation: Vesper Coding Standards — hot paths use std::expected; C++ pmr guidance allows throwing in do_allocate, but Vesper policy overrides for hot paths.
  - Recommendation: Provide non-throwing variants returning std::expected or nullptr and require caller checks; reserve throw-based APIs for non-hot admin tooling; add noexcept to fast-path helpers.

- [ ] Medium: TempBuffer alignment should honor alignof(T)
  - Location: TempBuffer (245–252), allocate(count*sizeof(T)) with default alignment=64 at ThreadLocalPool::allocate
  - Details: For types requiring stricter alignment, 64 may be insufficient in principle; safer to use alignment = max(64, alignof(T)).
  - Recommendation: In Phase 2, change TempBuffer to request alignment = std::max<std::size_t>(64, alignof(T)); document assumption and add debug assert.

- [ ] Low: Missing noexcept on ThreadLocalPool::allocate/allocator/stats
  - Location: allocate (154–157), allocator (148–151), stats (171–177)
  - Details: These should be non-throwing; marking noexcept clarifies the contract and enables better codegen.
  - Recommendation: Annotate noexcept where appropriate; ensure any allocations route through non-throwing arena path.

### include/vesper/core/platform_utils.hpp

- [ ] High: noexcept on safe_getenv can terminate on allocation failure
  - Location: safe_getenv signature (14) and `std::string` construction (25, 31)
  - Details: Function is marked `noexcept` but performs dynamic allocation to build `std::string`. On OOM, `std::bad_alloc` would violate `noexcept` and call `std::terminate`.
  - Web validation: C++ exception guarantees — `noexcept` must not throw; string constructors may throw on allocation failure.
  - Recommendation: Remove `noexcept` or catch `std::bad_alloc` and return `std::nullopt` (Phase 2). Document allocation behavior.

- [ ] Medium: Inconsistent/tacit env parsing across code; no bool helper
  - Location: Consumers (e.g., src/kernels/dispatch.cpp 168–175; core/cpu_features.hpp 64–75)
  - Details: Call sites parse booleans by checking first char `'0'`/`'1'`; other strings ("true"/"false") treated as unset. Inconsistent parsing and duplicated logic.
  - Recommendation: Add `parse_env_bool(std::string_view)` utility with accepted forms {1/0, true/false, on/off, yes/no}; standardize call sites (Phase 2).

- [ ] Low: Thread‑safety note missing for POSIX getenv
  - Location: safe_getenv comment block (9–13)
  - Details: While `getenv` returns a pointer to static storage, concurrent `setenv/putenv` in other threads is unsafe.
  - Web validation: POSIX discussions highlight races with concurrent `setenv` affecting `getenv` readers.
  - Recommendation: Document that Vesper never mutates env at runtime; avoid concurrent env mutation in host apps.
- [ ] Low: Missing `errno.h` include for errno_t on Windows
  - Location: safe_getenv Windows branch (16–27)
  - Details: Uses errno_t but only includes `cstdlib`; MSVC defines errno_t in `errno.h`. Relying on transitive includes is brittle.
  - Recommendation: Explicitly include `errno.h` on Windows builds or avoid errno_t by checking return via _dupenv_s and GetLastError mapping.

- [ ] Low: Document empty-variable semantics difference across platforms
  - Location: safe_getenv docs (9–13) and tests
  - Details: On Windows, setting a var to empty removes it (nullopt); on POSIX, an empty var returns engaged optional with empty string.
  - Recommendation: Expand docstring to state this explicitly; keep current behavior.


### Cross‑references and usage

- Pooled memory in k‑means Elkan
  - Location: src/index/kmeans_elkan.cpp (346–354)
  - Details: Uses `core::PoolScope` and `make_pooled_vector` to group per‑iteration allocations under a scope — correct usage pattern.

- Backend selection via environment
  - Location: src/kernels/dispatch.cpp (160–175); include/vesper/core/cpu_features.hpp (64–75)
  - Details: `safe_getenv` used to pick SIMD backend or AVX‑512 policy; behavior depends on env parsing noted above.

### include/vesper/core/cpu_features.hpp (header-only)

- [ ] High: AVX-512 selection elsewhere ignores OS XCR0 enabling (illegal instruction risk)
  - Location: Cross-ref src/kernels/dispatch.cpp select_backend_auto (260–266) and detect_cpu_features (74–112)
  - Details: cpu_features.hpp correctly checks OSXSAVE and XCR0 required bits (16–49), but dispatch’s feature detection sets has_avx512f from CPUID.7:EBX[16] without verifying XCR0 (OS support). If compiled with `__AVX512F__`, select_backend_auto may choose AVX-512 and execute AVX-512 on OSes that didn’t enable ZMM state.
  - Web validation: Intel SDM — AVX-512 requires OS enabling XCR0: XMM(1), YMM(2), OPMASK(5), ZMM_HI256(6), HI16_ZMM(7) via XGETBV(0); CPUID leaf 7 EBX bit 16 indicates hardware, not OS support.
  - Recommendation: Unify detection: either reuse core::cpu_supports_avx512_runtime() in dispatch, or extend detect_cpu_features to include XCR0 checks. Gate selection on both hardware and OS support.

- [ ] Medium: Environment override may return true when AVX-512 is not compiled
  - Location: decide_use_avx512_from_env_and_cpu (64–75), env path (65–69)
  - Details: Returns true when VESPER_AVX512="1" even if `__AVX512F__` is not defined; relies on call sites to guard. Risk of misconfiguration (e.g., passing true into configs on non-AVX512 builds).
  - Recommendation: Under !__AVX512F__, ignore env “1” and return false; or return tri-state and let callers decide.

- [ ] Medium: GCC/Clang path does not preflight max CPUID leaf before leaf 7
  - Location: Inline asm cpuid for leaf 7 (51–55)
  - Details: Best practice is to query leaf 0 first to ensure max level ≥ 7. On x86_64 this is almost always true, but adding guard improves robustness and forward-compat.
  - Recommendation: Query cpuid(0) first and conditionally call leaf 7.

- [ ] Low: Prefer compiler helpers over inline asm for CPUID/XGETBV
  - Location: Inline asm (38–57)
  - Details: GCC/Clang provide __get_cpuid/__cpuid_count and _xgetbv intrinsics; inline asm with EBX constraints can be fragile across ABIs.
  - Recommendation: Replace with standard helpers; add noexcept to both functions.

### include/vesper/platform/memory.hpp (header-only)

- [ ] High: align_up may overflow causing undersized allocation
  - Location: align_up (42–44), usage in aligned_allocate (66–74)
  - Details: (size + alignment - 1) can wrap for large sizes. The resulting aligned_size may be < size, leading to returning a pointer to insufficient storage and potential OOB writes by callers.
  - Web validation: Common guidance for alignment rounding requires overflow-safe checks before addition/masking.
  - Recommendation: Guard with `if (size > std::numeric_limits<size_t>::max() - (alignment - 1)) return nullptr;` and assert aligned_size ≥ size.

- [ ] High: aligned_unique_ptr destroys elements that may never have been constructed
  - Location: ctor (128–134) allocates raw storage only; reset/destructor (176–185) calls destructors for non-trivial T over count_
  - Details: For non-trivially-destructible T, this is UB if elements weren’t placement-constructed. Current design is “uninitialized storage + element-wise dtor” without tracking constructed range.
  - Recommendation: Either (a) restrict to trivially-destructible T via static_assert, (b) add construction helpers and track constructed count for safe destruction, or (c) provide separate make_aligned_unique_array that value-initializes.

- [ ] Medium: typed allocation may overflow on bytes = count * sizeof(T)
  - Location: aligned_allocate_typed (86–93)
  - Details: No overflow check before multiplication; wrap leads to undersized allocation.
  - Recommendation: Add `if (count > max/sizeof(T)) return nullptr;` or throw; mirror allocator patterns used elsewhere.

- [ ] Medium: Error reporting is nullptr-only; lacks rich status
  - Location: aligned_allocate (59–75), aligned_allocate_typed (86–93)
  - Details: In safety-critical paths we prefer explicit error codes. nullptr conflates invalid alignment and OOM.
  - Recommendation: Consider std::expected<void*, error_code> in non-hot paths; keep fast noexcept path for hot code.

- [ ] Low: Zero-size allocation semantics undocumented
  - Location: aligned_allocate docs (46–58)
  - Details: Behavior for size==0 differs across platforms; current code returns a (possibly aligned) pointer or nullptr.
  - Recommendation: Document contract (allowed; may return nullptr; pointer must be passable to aligned_deallocate).

- [ ] Low: Consider posix_memalign alternative on POSIX
  - Location: aligned_allocate (72–74)
  - Details: std::aligned_alloc enforces size % alignment == 0; posix_memalign returns int error codes and doesn’t require rounding by caller.
  - Recommendation: Evaluate posix_memalign for clearer error handling; keep current path if portability constraints prefer aligned_alloc.

### include/vesper/wal.hpp

- [ ] Low: Umbrella header widens transitive includes
  - Location: 19–23
  - Details: Re-exports frame/io/replay/manifest/snapshot. Including this widely in hot-path TUs increases compile times and broadens the apparent API surface.
  - Recommendation: Keep as a convenience include; prefer including narrower headers (frame.hpp, io.hpp, replay.hpp, manifest.hpp, snapshot.hpp) directly in hot-path translation units.

### include/vesper/wal/frame.hpp + src/wal/frame.cpp

- [ ] High: CRC32C polynomial orientation likely incorrect (mismatch vs standard CRC-32C)
  - Location: src/wal/frame.cpp (10–18 table gen; 21–27 update)
  - Details: Table is generated using polynomial 0x1EDC6F41 (standard form) while the byte-wise, LSB-first update uses right shifts (reflected algorithm). The reflected algorithm requires the reversed polynomial 0x82F63B78. Using 0x1EDC6F41 with the reflected update yields non-standard CRC-32C values. Round-trip tests pass because encode/verify share the same bug, but interoperability and external verification (e.g., tooling) will fail.
  - Web validation: CRC-32C uses polynomial 0x1EDC6F41, reversed 0x82F63B78 (StackOverflow: "CRC32 vs CRC32C?"); many reference implementations use 0x82F63B78 for table-based reflected updates.
  - Recommendation: Regenerate the table using the reversed polynomial 0x82F63B78 for the reflected update, or switch to a non-reflected formulation consistently. Add a known-answer test (e.g., "123456789" -> 0xE3069283) to catch regressions.
  - Status: RESOLVED
  - Resolution reference: ADR-0002 (docs/ADRs/ADR-0002-wal-crc32c-correction.md)
  - Test evidence: Test IDs #131 (crc32c known-answer), #132 (overflow guard), #133 (migration acceptance)
  - Date resolved: 2025-10-27

- [ ] High: Length computation may overflow, leading to undersized allocation and OOB write
  - Location: src/wal/frame.cpp (55, 67)
  - Details: `len` is computed as `uint32_t(WAL_HEADER_SIZE + payload.size() + 4)` and used to size `out`. If `payload.size()` exceeds 2^32-1-24, the cast truncates and `out` is undersized. The subsequent `memcpy(p, payload.data(), payload.size())` copies `payload.size()` bytes, causing buffer overflow.
  - Recommendation: Validate `payload.size() <= UINT32_MAX - WAL_HEADER_SIZE - 4` before allocation; return error on overflow. Consider a configurable maximum frame size; document it.
  - Status: RESOLVED
  - Resolution reference: ADR-0002 (docs/ADRs/ADR-0002-wal-crc32c-correction.md)
  - Test evidence: Test IDs #131 (crc32c known-answer), #132 (overflow guard), #133 (migration acceptance)
  - Date resolved: 2025-10-27

- [ ] Medium: Endianness portability not implemented despite little-endian framing claim
  - Location: include/vesper/wal/frame.hpp (6–8 doc); src/wal/frame.cpp (29–43 loads, 58–66 stores)
  - Details: load/store helpers use `memcpy` without byte-swapping. On big-endian platforms, magic check, length, and CRC will misparse. The doc states "little-endian framing on all platforms", but implementation is little-endian only.
  - Web validation: Cross-platform serialization typically defines byte order and performs explicit conversion (network byte order) to ensure portability.
  - Recommendation: Implement le32/le64 encode/decode with conditional bswap for big-endian; or explicitly document little-endian-only support and add static_assert to fail on big-endian builds.

- [ ] Medium: Frame schema lacks version field; reserved not used for compatibility
  - Location: include/vesper/wal/frame.hpp (24–32)
  - Details: Header has `magic,len,type,reserved,lsn` but no explicit version. Future changes to header semantics (e.g., adding flags) may not be distinguishable.
  - Web validation: Frame-based protocols often include a version or flags field; SQLite WAL has stable, documented headers and checksums.
  - Recommendation: Allocate `reserved` bits to a version/flags field; document semantics and forward/ backward compatibility policy.

- [ ] Medium: Type value not validated against allowed set {1=data,2=commit,3=padding}
  - Location: src/wal/frame.cpp (82–86 decode)
  - Details: Decoding accepts any `type`. Downstream modules treat types 1/2 specially and ignore 3 (padding). Unknown types could skew stats/histograms or bypass monotonicity accounting in higher layers.
  - Recommendation: Validate `type` ∈ {1,2,3} (or defined set) and return error for unknown types; alternatively, reserve ranges and document behavior.

- [ ] Low: `crc32c()`/verify could be marked noexcept and documented with coverage
  - Location: include/vesper/wal/frame.hpp (34–41); src/wal/frame.cpp (21–27, 45–51)
  - Details: Functions are pure and cannot throw; documenting coverage (header+payload vs payload-only) improves clarity.
  - Recommendation: Mark `crc32c` and `verify_crc32c` noexcept; clarify coverage in comments.

- [ ] Low: Zero-length payload and min-length semantics not explicitly documented
  - Location: include/vesper/wal/frame.hpp (40–46); src/wal/frame.cpp (45–51, 92–95)
  - Details: Behavior is safe (len ≥ header+CRC enforced, payload=0 allowed), but not documented.
  - Recommendation: Document that zero-length payloads are permitted and frames smaller than header+CRC are invalid.

- [ ] Low: Performance note — potential hardware CRC acceleration
  - Location: src/wal/frame.cpp (21–27)
  - Details: SSE4.2 on x86-64 exposes CRC32C instructions; large payloads could benefit.
  - Recommendation: Consider optional CRC32C acceleration path (e.g., dispatch via cpu_features) with fallbacks; verify parity with known answers.

- [ ] Low: [[nodiscard]] recommended on return types that convey status/verification
  - Location: include/vesper/wal/frame.hpp (34–46)
  - Details: `verify_crc32c` (bool) and `decode_frame` (expected) results may be inadvertently ignored.
  - Recommendation: Add `[[nodiscard]]` to `verify_crc32c`, `encode_frame` (optional), and `decode_frame` to encourage correct use.

- [ ] Medium: C++ ABI hazard — public struct uses std::span
  - Location: include/vesper/wal/frame.hpp (24–32)
  - Details: WalFrame embeds std::span, which is not ABI-stable across compilers/standard libraries. Passing this type across shared-library/toolchain boundaries is unsafe.
  - Recommendation: Document that the C++ API is intended for same-toolchain use; provide a C-compatible view (ptr+len) or C API for cross-boundary consumers.

#### Cross-references and tests

- Usage/integration: wal/io.cpp uses encode_frame/verify_crc32c/decode_frame for file IO and scanning (lines 171–194, 226–281). Endianness assumptions also appear in IO when reading `len` via memcpy.
- Tests: tests/unit/wal_frame_test.cpp validates encode→verify→decode round-trip; does not compare against standard CRC-32C known values (would not catch the polynomial orientation issue). tests/fuzz/wal_frame_fuzz.cpp fuzzes decode/verify entry points.
- Related: Presence of `include/vesper/wal/*.bak` files suggests legacy versions lingering in tree; consider cleanup (repo hygiene).

### include/vesper/wal/manifest.hpp + src/wal/manifest.cpp

- [x] High: Non-atomic manifest writes; no fsync — RESOLVED (2025-10-27)
  - Resolution: Implemented save_manifest_atomic() with durable temp-write → fsync file → atomic replace → fsync parent dir; on Windows, prefer ReplaceFileW with fallback MoveFileExW(MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) and FlushFileBuffers on the final handle
  - ADR: docs/ADRs/ADR-0003-wal-manifest-durable-update.md
  - Tests: TID-WAL-MAN-ATOMIC-001/002/003/004 in tests/unit/wal_manifest_atomic_test.cpp passed
  - Notes: Removes leftover wal.manifest.tmp from prior crashes (ignore ENOENT/ERROR_FILE_NOT_FOUND); on Windows, falls back to unique tmp name if default tmp is locked
  - Build/Test: Debug build exit code 0; 3/3 new tests passed on Windows; zero warnings

- [x] High: load_manifest may throw on malformed lines — RESOLVED (2025-10-27)
  - Resolution: `load_manifest()` now uses exception-free parsing via `std::from_chars` for all numeric fields, validates filenames (no control chars), and enforces required keys. On malformed input, it returns `unexpected(error{data_integrity, ...})` with line/field context. `validate_manifest()` and `list_sorted()` were updated to use `from_chars` (advisory issues on malformed numeric) to eliminate throw paths. Fuzzer updated to drop try/catch around manifest paths.
  - ADR: docs/ADRs/ADR-0004-wal-manifest-exception-free-parsing.md
  - Tests: TID-WAL-MAN-PARSE-001/002/003/004 in tests/unit/wal_manifest_parse_test.cpp; fuzz `tests/fuzz/wal_manifest_fuzz.cpp` exercises load/validate/enforce; WAL subset [wal] passed on Windows
  - Build/Test: Debug build exit code 0; [wal] subset: 49/49 test cases passed; zero warnings

- [x] High: Path traversal risk from unsanitized ManifestEntry.file — RESOLVED (2025-10-27)
  - Resolution: Strict filename validation in `load_manifest()` and `validate_manifest()`:
    - Accept only `^wal-[0-9]{8}\.log$`
    - Reject any separators ('/' or '\\'), parent refs ('..'), absolute roots ('/' or '\\'), Windows drives (`[A-Za-z]:`), and UNC (`\\\\`)
    - On violation: return `unexpected(error{data_integrity,...})` from load; `validate_manifest()` reports `Severity::Error` (BadHeader: "invalid filename") with line context
  - ADR: docs/ADRs/ADR-0005-wal-manifest-filename-validation.md
  - Tests: TID-WAL-MAN-PATH-001/002/003/004/005/006/007 in tests/unit/wal_manifest_parse_test.cpp
  - Build/Test: Debug build exit code 0; [wal]/[manifest] subsets passed; zero warnings

- [x] High: Duplicate sequence numbers not detected — RESOLVED (2025-10-27)
  - Resolution: validate_manifest() now tracks seen seq values and reports duplicates across different files as Severity::Error with ManifestIssueCode::DuplicateSeq. Message includes the first filename for actionable diagnostics. Duplicate within the same file is already covered by DuplicateFile.
  - Policy: load_manifest() remains parse-only; recovery enforces fail-closed on any Severity::Error from validate_manifest(), including DuplicateSeq. No auto-resolution is attempted to avoid ambiguous recovery; operator remediation required if encountered.
  - ADR: docs/ADRs/ADR-0006-wal-manifest-duplicate-seq-policy.md
  - Tests: TID-WAL-MAN-DUP-SEQ-001/002/003/004 in tests/unit/wal_manifest_parse_test.cpp
  - Build/Test: Debug build exit code 0; [wal]/[manifest] subsets passed

- [x] High: LSN range invariants not validated (first_lsn/end_lsn) — RESOLVED (2025-10-27)
  - Resolution: validate_manifest() now enforces intra-entry start_lsn <= first_lsn <= end_lsn and cross-entry checks (no overlaps, end_lsn monotonicity, gaps warned). New codes: LsnInvalid (Error), LsnOverlap (Error), LsnOrder (Error), LsnGap (Warning).
  - ADR: docs/ADRs/ADR-0007-wal-manifest-lsn-validation.md
  - Tests: TID-WAL-MAN-LSN-001/002/003/004/005/006/007/008 in tests/unit/wal_manifest_parse_test.cpp
  - Build/Test: Debug build exit 0; [manifest] subset extended and passing; no new warnings
  - Details: No checks that first_lsn > 0 and end_lsn ≥ first_lsn. start_lsn is a legacy alias; divergence between start_lsn and first_lsn is not flagged.
  - Recommendation: Validate invariants and, for v1, enforce start_lsn == first_lsn; emit error on violations.


- [x] High: rebuild_manifest must enforce LSN invariants on generated entries — RESOLVED (2025-10-27)
  - Resolution: rebuild_manifest() now validates intra-entry start_lsn <= first_lsn <= end_lsn and cross-entry constraints (no overlaps; end_lsn monotonic across increasing seq) during generation and fails fast with error_code::data_integrity and actionable diagnostics. LSN gaps across files are allowed and surfaced later as warnings by validate_manifest().
  - ADR: docs/ADRs/ADR-0008-wal-manifest-rebuild-lsn-validation.md
  - Tests: TID-WAL-MAN-REBUILD-LSN-001/002/003/004 in tests/unit/wal_manifest_rebuild_test.cpp
  - Build/Test: Debug build exit 0; [rebuild] subset passing; [wal] subset passing; no new warnings

- [x] High: Provide lenient (best-effort) WAL manifest rebuild mode — RESOLVED (2025-10-27)
  - Resolution: Added `rebuild_manifest_lenient(dir) -> expected<LenientRebuildResult, error>` alongside strict `rebuild_manifest(dir)`. Lenient mode skips corrupt files/entries and accumulates structured `RebuildIssue` diagnostics while guaranteeing that included entries satisfy LSN invariants among themselves (gaps allowed). Strict behavior unchanged (fail-fast with actionable `data_integrity` errors). Internals share logic via a single implementation path.
  - ADR: docs/ADRs/ADR-0009-wal-manifest-lenient-rebuild.md
  - Tests: TID-WAL-MAN-REBUILD-LENIENT-001/002/003/004/005/006 in tests/unit/wal_manifest_rebuild_test.cpp
  - Build/Test: Debug build exit 0; [rebuild]/[wal] subsets passing; zero new warnings


- [ ] Medium: frames/bytes fields are not checked for plausibility
  - Location: src/wal/manifest.cpp (97)
  - Details: Missing/zero values pass silently; negative not representable but overflow of stoull is not handled. Values should be >0 for non-empty files and correlate (bytes ≥ frames*(min_frame_size)).
  - Recommendation: Add bounds/consistency checks (e.g., frames > 0, bytes ≥ 24*frames). Treat gross inconsistencies as errors.

- [x] Medium: VESPER_ENABLE_MANIFEST_FSYNC macro was defined but unused — RESOLVED (2025-10-27)
  - Action: Removed the unused macro from src/wal/manifest.cpp; we standardize on VESPER_ENABLE_ATOMIC_RENAME for atomic replace behavior (see ADR-0003)
  - Rationale: Avoid dead flags and confusion; durability is controlled by the atomic-rename flow and per-platform fsync/FlushFileBuffers already implemented
  - Impact: None on behavior; build/tests unaffected (Debug build 0 warnings; WAL suite green)

- [ ] Medium: rebuild_manifest aborts on first scan error (reduced recoverability)
  - Location: src/wal/manifest.cpp (168–170)
  - Details: On any per-file scan error, rebuild_manifest returns an error; subsequent files are skipped.
  - Recommendation: Consider best-effort rebuild: collect issues per-file, skip corrupted files, and return Manifest plus a list of per-file errors (or a ManifestValidation). Provide a strict mode toggle.

- [ ] Medium: Concurrency/lost update risk in upsert_manifest
  - Location: src/wal/io.cpp (127–139, 210–214)
  - Details: Read-modify-write without locking can interleave writers (e.g., multiple WalWriter instances), losing entries. Combined with non-atomic save, torn/lost updates more likely.
  - Recommendation: Serialize via atomic temp+rename; optionally add coarse file lock (advisory flock/LockFileEx) around updates. Writes are infrequent; simple coarse locking is acceptable.

- [ ] Medium: C++ ABI hazard — public structs use std::string/std::vector
  - Location: include/vesper/wal/manifest.hpp (21–32)
  - Details: ManifestEntry and Manifest expose std::string/std::vector in the public C++ API. These types are not ABI-stable across different standard libraries or compilers.
  - Recommendation: Document that the C++ API is not a stable cross-DSO boundary; prefer the C API or an opaque handle for cross-toolchain consumption. For C++ consumers, keep as-is but note boundary constraints.

- [ ] Low: Save order not specified by API; relies on caller to sort
  - Location: src/wal/manifest.cpp (63–71)
  - Details: save_manifest writes entries in provided order. Upsert path sorts, but other callers could pass unsorted entries.
  - Recommendation: Either document that callers must sort by seq, or sort defensively in save_manifest.

- [ ] Low: [[nodiscard]] recommended on validate/enforce APIs
  - Location: include/vesper/wal/manifest.hpp (49–54)
  - Details: validate_manifest returns a status+issues struct; enforce_manifest_order returns `expected<void>`. Both results are easy to ignore inadvertently.
  - Recommendation: Add [[nodiscard]] to validate_manifest and enforce_manifest_order.

- [ ] Low: Text parsing assumes space-delimited tokens; filenames with spaces unsupported
  - Location: src/wal/manifest.cpp (34–47, 63–71)
  - Details: Parser splits on whitespace; save path writes with single spaces. Current filenames are restricted (wal-########.log), so not an issue in practice.
  - Recommendation: Document limitation; keep as-is unless filenames change.

#### Cross-references and tests (manifest)

- Integration: upsert_manifest is called on rotation and on flush (src/wal/io.cpp 163–166, 210–214). Recovery prefers manifest when present but falls back to directory listing when missing/empty/bad (src/wal/io.cpp 293–333).
- Snapshot interplay: recover_scan_dir uses snapshot to compute cutoff_lsn and may skip files entirely if manifest.end_lsn ≤ cutoff (src/wal/io.cpp 349–355).
- Tests: tests/unit/wal_manifest_validate_test.cpp covers ordering/duplicates/missing/extra detection; wal_manifest_roundtrip_edges_test.cpp validates that even bad or header-only manifests still lead to correct frame scanning; wal_manifest_sync_test.cpp ensures entries appear after rotation/flush. Fuzz target tests/fuzz/wal_manifest_fuzz.cpp writes arbitrary wal.manifest bytes and exercises load/validate/enforce.

#### Web validation

- RocksDB MANIFEST: write-ahead metadata updates via temp file + fsync + rename (atomic), with directory fsync for durability.
- LevelDB VersionEdit/MANIFEST: similar atomic rename pattern for metadata; robust parsing and recovery tools.
- SQLite: atomic rename and directory fsync guidance; manifest-like metadata consistency across crashes.

### include/vesper/wal/io.hpp + src/wal/io.cpp

- [x] High: Durability knobs are stats-only; no actual fsync/fdatasync/FlushFileBuffers → not crash-safe — RESOLVED (2025-10-27)
  - Resolution: Implemented OS-level durability in WalWriter.
    - flush(sync): flush stream buffers, then ensure durability.
      - POSIX: open by path (O_RDONLY) and fsync; propagate errors.
      - Windows: best-effort FlushFileBuffers on a separate handle; if sharing prevents it, fall back to close → FlushFileBuffers → reopen in append mode. Treat sharing-violation/access-denied cases as success (best-effort semantics consistent with Windows handle sharing). Increment stats.syncs on success.
    - rotation: flush and close the old file first, then ensure durability via fsync/FlushFileBuffers before counting rotation; open the new file next. When fsync_on_rotation is enabled, fsync the parent directory after creating the new file (POSIX: open(O_DIRECTORY)+fsync; Windows: CreateFileW on directory with FILE_FLAG_BACKUP_SEMANTICS + FlushFileBuffers, best-effort).
  - ADR: docs/ADRs/ADR-0010-wal-writer-durability.md
  - Tests: TID-WAL-WRITER-DURABILITY-PROFILE (wal_writer_durability_profile_test.cpp), TID-WAL-WRITER-FSYNC (wal_writer_fsync_test.cpp), TID-WAL-WRITER-FLUSH-SYNC-TRUE (wal_writer_flush_sync_true_test.cpp) — all passing on Windows.
  - Build/Test: Debug build; [wal] subset 60/60 passing; [manifest] 35/35 passing; zero new warnings.
  - Notes: When durability knobs are disabled, behavior is unchanged. Windows path adopts best-effort semantics due to std::ofstream share-mode limitations; alternatives and rationale documented in ADR-0010.

- [x] High: recover_scan trusts LEN to allocate/read without validating header magic first (OOM/DoS risk) — RESOLVED (2025-10-27)
  - Resolution: In recover_scan(), validate WAL_MAGIC before trusting LEN; on mismatch, stop scanning without allocation (torn-tail semantics).
  - ADR: docs/ADRs/ADR-0011-wal-recover-scan-hardening.md
  - Tests: TID-WAL-RECOVER-GUARDS (wal_recover_scan_len_magic_guard_test.cpp)

- [x] High: No upper bound or remaining-file-size check on LEN before allocation/read — RESOLVED (2025-10-27)
  - Resolution: Enforce MAX_FRAME_LEN = 32 MiB and check remaining bytes (file_size - tellg()) before allocation/read; stop scan on violation.
  - ADR: docs/ADRs/ADR-0011-wal-recover-scan-hardening.md
  - Tests: TID-WAL-RECOVER-GUARDS (wal_recover_scan_len_magic_guard_test.cpp)

- [ ] Medium: DeliveryLimits::max_bytes semantics inconsistent (payload used for threshold, but stats track full frame bytes)
  - Location: include/vesper/wal/io.hpp (38–43 doc says "payload+header bytes"); src/wal/io.cpp (414–421 uses `payload.size()` for limit but increments delivered_b by `f.len`)
  - Details: A frame may be delivered even if its total bytes would exceed max_bytes because the pre-check uses only payload bytes. This deviates from the header contract and can surprise callers.
  - Recommendation: Apply the threshold against the same metric used for accounting (prefer `f.len` both for the pre-check and increment). Document whether the limit is inclusive or exclusive.

- [ ] Medium: Multi-writer/process collision risk in rotation mode (no locking/coordinator)
  - Location: src/wal/io.cpp (103–119 scan for max seq, then open_seq on first append)
  - Details: Two WalWriter instances pointed at the same dir/prefix can both observe the same `max_seq` and start writing the same next sequence concurrently, leading to clobber/race. Header notes "one writer per file" but rotation mode is effectively one-writer-per-dir.
  - Recommendation: Document process-safety constraints clearly (one writer per dir/prefix) and consider coarse advisory locking (LockFileEx/flock) or a create+link/rename based sequencer to ensure unique next sequence.

- [ ] Medium: recover_scan_dir two-pass per file (torn check + delivery) doubles IO
  - Location: src/wal/io.cpp (357–365 first pass; 391–393 second pass)
  - Details: Each file is scanned twice (first pass for torn middle file detection, second for delivery). This doubles I/O on large WALs.
  - Recommendation: Single-pass approach: track bytes consumed and compare to file_size at the end for non-last files; or buffer minimal per-file stats during delivery to avoid a full noop pass.

- [ ] Medium: Snapshot cutoff semantics depend on manifest for per-file skipping; header-only/no-manifest path may read more than necessary
  - Location: src/wal/io.cpp (349–355 uses manifest end_lsn to skip entire file; otherwise per-frame filter)
  - Details: Without manifest, we still open and scan files that are fully ≤ cutoff_lsn. This is correct but suboptimal.
  - Recommendation: Consider deriving a coarse per-file end_lsn by peeking the last frame (e.g., scan backward to last valid header) when manifest is missing, or accept as performance tradeoff.

- [ ] Low: `open(path, create_if_missing=false)` still appends to a non-existent file on some platforms
  - Location: src/wal/io.cpp (69–80)
  - Details: The out_.open uses `std::ios::app` regardless; even with create_if_missing=false, some libstdc++/MSVC may create the file. API contract is ambiguous.
  - Recommendation: If `create_if_missing=false`, open with `std::ios::in|std::ios::out` and error if file does not exist; document exact behavior.

- [ ] Low: Missing [[nodiscard]] on functions returning expected results
  - Location: include/vesper/wal/io.hpp (142–155)
  - Details: `recover_scan*` return status that can be ignored; same for `WalWriter::open/append/flush/publish_snapshot` in usage. Marking as [[nodiscard]] encourages correct error handling.
  - Recommendation: Add [[nodiscard]] to recovery APIs and writer factory.

- [ ] Low: Per-frame heap allocations in recover_scan for header and rest
  - Location: src/wal/io.cpp (238–251)
  - Details: Repeated `std::vector` resize/insert per frame causes churn. The header is fixed-size and can be on stack; the frame buffer can reuse capacity.
  - Recommendation: Use a fixed-size stack buffer for header and reuse a single `std::vector<uint8_t>` for frame body to reduce allocations.

#### Cross-references and tests (I/O)

- Integration: Uses frame encode/verify/decode (src/wal/io.cpp 180–189, 252–256). Manifest updates on rotation/flush (163–166, 210–214). Snapshot cutoff applied in directory scan (300–305, 368–389).
- Tests validate behaviors: torn tail tolerated only on last file; non-last torn returns data_integrity; stale manifest still scans highest-seq file; DeliveryLimits filters frames deterministically; fsync policy tests only assert stats increments (no durability).

#### Web validation (I/O)

- PostgreSQL/SQLite/RocksDB durability discussions: fsync/fdatasync/FlushFileBuffers and directory fsync requirements for crash-safe metadata and WAL.
- SQLite WAL torn-page handling and EOF truncation guidance (stop at first invalid frame).
- LevelDB/RocksDB: single-writer constraints and manifest/WAL atomics via temp file + fsync + rename.

### include/vesper/wal/snapshot.hpp + src/wal/snapshot.cpp

- [x] Fixed (2025-10-27): Atomic save lacked file fsync and durable replace semantics (implemented durable fsync + atomic replace on POSIX/Windows) — ADR-0012; tests: wal_snapshot_durable_replace_test.cpp
  - Location: src/wal/snapshot.cpp (44–73)
  - Details: Atomic path writes wal.snapshot.tmp and renames to wal.snapshot without fsync/fdatasync on the temp file prior to rename. On Windows, there is no FlushFileBuffers. Directory fsync is attempted only on POSIX and only after rename. Power loss can result in a zero-length or partially written wal.snapshot after an apparent "successful" save.
  - Web validation: SQLite/LevelDB/RocksDB durable rename patterns require fsync(temp) → rename → fsync(dir). On Windows, use FlushFileBuffers on the file handle, then ReplaceFileW/MoveFileExW with replace semantics, then flush directory handle.
  - Recommendation: Implement platform-correct durable sequence: write → flush → fsync file → atomic replace (rename with replace semantics) → fsync parent directory. Provide a feature flag to disable in tests.

- [x] Fixed (2025-10-27): Windows replace semantics — use MoveFileExW(MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH); removed non-atomic fallback; tmp cleanup on all paths — ADR-0012; tests: wal_snapshot_durable_replace_test.cpp
  - Location: src/wal/snapshot.cpp (56–64, 75–80)
  - Details: On Windows, std::filesystem::rename fails if destination exists. The code falls back to truncating write of wal.snapshot (non-atomic), undermining the intended atomic-save guarantee. Also, the temp file is not removed on fallback.
  - Recommendation: On Windows, use ReplaceFileW or MoveFileExW(MOVEFILE_REPLACE_EXISTING) for atomic replacement. Ensure the temp file is removed on failure paths. Keep behavior symmetric across platforms.

- [ ] Medium: Windows lacks directory fsync; POSIX-only path uses open(dir,O_RDONLY) without O_DIRECTORY
  - Location: src/wal/snapshot.cpp (66–72)
  - Details: Durable directory entry persistence is not attempted on Windows (no flush of directory handle). On POSIX, opening the directory without O_DIRECTORY can be error-prone on some platforms.
  - Recommendation: Add Windows code path to open the directory with FILE_FLAG_BACKUP_SEMANTICS and FlushFileBuffers, or document acceptable durability caveat. On POSIX, prefer open(dir, O_RDONLY|O_DIRECTORY) when available.

- [ ] Medium: Snapshot format has no checksum/integrity marker
  - Location: include/vesper/wal/snapshot.hpp (6–13); src/wal/snapshot.cpp (50–54, 78–79)
  - Details: A single-bit flip in wal.snapshot will not be detected until parsing fails; silent mis-reads (e.g., altered last_lsn that still parses) are possible.
  - Recommendation: Add a simple CRC32C over the payload line (or whole file) and validate on load. Keep v1 backward-compatible by accepting files without checksum.

- [ ] Medium: API results lack [[nodiscard]] annotations
  - Location: include/vesper/wal/snapshot.hpp (25–29)
  - Details: load_snapshot/save_snapshot return expected<…> that can be ignored by callers.
  - Recommendation: Add [[nodiscard]] to both functions to encourage correct error handling.

- [ ] Medium: Concurrency/process-safety is not documented (multi-writer hazard)
  - Location: include/vesper/wal/snapshot.hpp (3–13); src/wal/snapshot.cpp (44–81)
  - Details: If two processes call save_snapshot on the same dir concurrently, races can produce lost updates or mixed states, particularly on Windows where rename fallback is non-atomic.
  - Recommendation: Document one-writer-per-dir/prefix invariant (align with WalWriter) and consider coarse advisory locking around snapshot update.

- [ ] Low: Fallback path leaves wal.snapshot.tmp orphaned on rename failure
  - Location: src/wal/snapshot.cpp (56–64)
  - Details: When rename fails and code falls back to direct write, the wal.snapshot.tmp file is not removed.
  - Recommendation: Remove temp file on all non-success paths; ensure tests cover fallback.

- [ ] Low: Mixed binary/text modes may lead to platform-specific newline handling
  - Location: src/wal/snapshot.cpp (25 text input; 50, 60, 78 write with std::ios::binary)
  - Details: load_snapshot opens without std::ios::binary while save uses binary. In practice, getline handles CRLF, but mode mismatch can be surprising.
  - Recommendation: Either open both in text mode consistently (relying on iostream translation) or both in binary; document end-of-line expectations.

#### Cross-references and tests (Snapshot)

- Integration: WalWriter::publish_snapshot delegates to save_snapshot (src/wal/io.cpp 196–202). recover_scan_dir consults wal.snapshot when present (src/wal/io.cpp 300–305) and ignores it on parsing error.
- Tests: wal_snapshot_atomic_test.cpp validates tmp cleanup and load round-trip; wal_snapshot_manifest_test.cpp checks skip behavior; wal_snapshot_manifest_interplay_test.cpp combines manifest completeness and cutoff. No tests currently exercise Windows replace semantics or durable fsync ordering.

#### Web validation (Snapshot)

- SQLite/LevelDB/RocksDB: atomic update pattern — write temp → fsync file → rename/replace → fsync directory (metadata). Windows: ReplaceFileW/MoveFileExW(MOVEFILE_REPLACE_EXISTING) and FlushFileBuffers on file and directory handles.

### include/vesper/wal/replay.hpp + src/wal/replay.cpp

- [x] High: RecoveryStats reflect post-filter delivery (RESOLVED by Task 15)
  - Location: include/vesper/wal/replay.hpp, src/wal/replay.cpp, src/wal/io.cpp
  - Resolution: Added accepting-callback variants that return `std::expected<DeliverDecision,error>` and routed mask/limits overloads through them. `RecoveryStats` now count only delivered frames post-filter. Added unit tests asserting stats correctness.
  - Tests: wal_replay_type_mask_stats_test.cpp; wal_delivery_limits_stats_test.cpp.

- [x] High: ReplayCallback early termination/error propagation without exceptions (RESOLVED by Task 15)
  - Location: include/vesper/wal/replay.hpp, src/wal/replay.cpp, src/wal/io.cpp
  - Resolution: Introduced `DeliverDecision` with four states (DeliverAndContinue, DeliverAndStop, Skip, SkipAndStop) and `ReplayResultCallback = std::function<std::expected<DeliverDecision,error>(...)>`. Replay/scan honor early-stop and error without exceptions; void-callback overloads preserved for backward compatibility.
  - Tests: wal_replay_early_stop_test.cpp; wal_property_replay_test.cpp updated scenarios verified.

- [ ] Medium: Duplicate filtering logic risks divergence from scan API
  - Location: src/wal/replay.cpp (13–17); io.hpp (150–155)
  - Details: Filtering by mask is implemented locally in replay instead of using the existing recover_scan_dir(type_mask, ...). Any future change to filtering semantics in scan risks replay drift.
  - Recommendation: Remove duplicate filtering; delegate to the scan overload that already applies masks and aggregates stats consistently.

- [ ] Medium: Bitmask shift uses 32-bit literal with unbounded `type`
  - Location: src/wal/replay.cpp (16)
  - Details: `(1u << f.type)` is undefined if `f.type >= 32`. While current types are 1..3, future extensions could cause UB. The mask type is 32-bit, but `type` is 16-bit.
  - Recommendation: Guard `f.type <= 31` (or clamp) and document supported range; alternatively use 64-bit mask and static_assert on max type.

- [ ] Medium: Missing [[nodiscard]] on recover_replay results
  - Location: include/vesper/wal/replay.hpp (31–40)
  - Details: Callers can accidentally discard the `expected<RecoveryStats,error>` result.
  - Recommendation: Add [[nodiscard]] to both recover_replay overloads.

- [ ] Medium: Payload span lifetime not documented for callback
  - Location: include/vesper/wal/replay.hpp (19–26)
  - Details: `std::span<const uint8_t>` typically aliases an internal buffer valid only during the callback. Storing the span or its pointer beyond the call is unsafe, but this is not stated.
  - Recommendation: Document lifetime: payload is valid only during callback; copy if needed.

- [ ] Low: No replay overload for DeliveryLimits (cutoff override, max frames/bytes)
  - Location: include/vesper/wal/replay.hpp (API surface); io.hpp (150–155)
  - Details: recover_scan_dir supports DeliveryLimits; recover_replay does not. Callers must drop down to scan API and rewrap payload extraction, duplicating code.
  - Recommendation: Add `recover_replay(dir, const DeliveryLimits&, ReplayCallback)` delegating to the scan overload.

- [ ] Low: Callback emptiness not validated
  - Location: src/wal/replay.cpp (6–17)
  - Details: An empty `std::function` will crash on invocation. While misuse, a guard would produce a clearer error.
  - Recommendation: Validate `on_payload` and return an `invalid_argument` error if empty.

#### Cross-references and tests (Replay)

- Integration: Delegates to recover_scan_dir which applies snapshot cutoff and manifest ordering; replay adds payload extraction and optional type filtering.
- Tests: wal_replay_type_mask_test exercises filtering but does not assert stats; property tests (wal_property_replay_test.cpp) validate determinism and LSN monotonicity metrics via recover_scan_dir, and end-to-end equivalence via ToyIndex helpers.

#### Web validation (Replay)

- Database replay patterns: PostgreSQL/WAL redo, SQLite recovery, RocksDB log replay – callback/error propagation patterns typically allow early stop and error return, and stats reflect delivered (post-filter) operations.


### include/vesper/wal/retention.hpp + src/wal/retention.cpp + src/wal/retention_keep.cpp

- [x] High: Duplicate implementations (ODR hazard) and divergent semantics across TUs — RESOLVED
  - Location (was): src/wal/retention.cpp: purge_keep_last_n (33–48), purge_keep_newer_than (50–84), purge_keep_total_bytes_max (86–105); src/wal/retention_keep.cpp: purge_keep_last_n (11–27), purge_keep_newer_than (29–48), purge_keep_total_bytes_max (52–84)
  - Resolution: Deleted src/wal/retention_keep.cpp; unified single-source implementation in src/wal/retention.cpp with deterministic semantics.
  - Tests: Added tests/unit/wal_retention_unify_test.cpp (4 cases: timestamp tie-handling; keep-last-N; byte-budget edges incl. zero budget; namespace visibility) and extended tests/unit/wal_retention_keep_test.cpp (1 case: end_lsn ordering vs seq).
  - Unified semantics: sort by end_lsn (descending, newest first), then filename (lexicographically descending) for ties; in byte-budget mode the newest is always retained even if it exceeds the budget alone.

- [x] High: Namespace mismatch for purge_keep_total_bytes_max in retention_keep.cpp — RESOLVED
  - Location (was): src/wal/retention_keep.cpp (function defined after closing namespace block, lines 52–84)
  - Resolution: File removed; all exported symbols defined within namespace vesper::wal in src/wal/retention.cpp.
  - Notes: Header and tests validate correct namespacing; duplicate TU removal prevents future mismatch.

- [ ] High: Risk of deleting an active/open WAL file (platform-specific hazards)
  - Location: src/wal/retention.cpp purge_wal (18–23)
  - Details: Purge deletes any file with end_lsn ≤ cutoff with no guard against the current writer's open file. On POSIX, unlink on an open file detaches the directory entry while the writer continues to append to an unlinked inode (data loss on crash; leak). On Windows, DeleteFile typically fails with ERROR_SHARING_VIOLATION unless handles were opened with FILE_SHARE_DELETE. Behavior is thus non-deterministic and unsafe without coordination.
  - Web validation: POSIX durability and directory fsync guidance ["Files are hard"; Linux fsync/rename discussions]. Windows CreateFile/DeleteFile share semantics [MSDN: CreateFile*, DeleteFileW]. SQLite documents directory fsync and careful ordering for durability [sqlite.org/wal.html].
  - Recommendation: Define and enforce an invariant: retention must not run while a WalWriter is active on the same dir; or explicitly exclude the newest/active file (using manifest/lock). Document this precondition in the header and tests. Consider writer-managed retention to guarantee coordination.

- [ ] Medium: keep_last_n relies on manifest order in retention_keep.cpp (no sort)
  - Location: src/wal/retention_keep.cpp purge_keep_last_n (11–27)
  - Details: Function assumes manifest entries are already in ascending seq. If load_manifest ever returns out-of-order entries, "keep last N" may remove the wrong files. retention.cpp variant sorts explicitly.
  - Recommendation: Always sort by seq before computing the cutoff. Add a determinism note and a unit test that shuffles manifest entries before purge.

- [ ] Medium: Time-based retention tie-handling is inconsistent
  - Location: retention.cpp purge_keep_newer_than (50–84) vs retention_keep.cpp (29–48)
  - Details: retention.cpp keeps only the highest-seq file among equal timestamps and removes others; retention_keep.cpp keeps all files with ft ≥ cutoff. Divergence leads to inconsistent outcomes across builds/platforms, especially on coarse timestamp filesystems.
  - Web validation: Deterministic retention is preferred; SQLite WAL checkpointing defines deterministic truncation rules [sqlite.org/wal.html].
  - Recommendation: Adopt a single deterministic policy (e.g., keep strictly newer; for ties keep highest seq only). Document it in the header and add tests creating multiple files with identical timestamps.

- [ ] Medium: Byte-budget policy diverges and edge cases are unspecified
  - Location: retention.cpp purge_keep_total_bytes_max (86–105) vs retention_keep.cpp (52–84)
  - Details: retention_keep.cpp guarantees at least one newest file is kept even if budget < smallest file; retention.cpp may keep none if all files exceed the budget. The desired invariant (“never end up with zero WAL files unless an explicit snapshot baseline exists”) is not documented.
  - Recommendation: Specify and enforce the invariant (e.g., keep at least newest one). Add tests for budget=0 and budget < smallest file cases.

- [ ] Medium: Crash-consistency and durability ordering not specified
  - Location: retention.cpp purge_wal (27–30), all keep-* functions after deletion
  - Details: Safe publish on POSIX typically requires writing new manifest/snapshot, fsyncing files, then fsyncing parent directory (for rename durability). The current code deletes files first, then saves manifest/snapshot, and there is no explicit directory fsync. Windows lacks directory fsync; FlushFileBuffers applies to files/volumes and directory handles behave differently.
  - Web validation: Linux durability guidance (fsync directory after rename) [StackOverflow 12990180; renameio issue]; SQLite explains directory fsync needs [sqlite.org/tempfiles.html, wal.html]. Windows FlushFileBuffers/FILE_SHARE_DELETE semantics [MSDN: FlushFileBuffers, CreateFileW].
  - Recommendation: Document the intended ordering and durability model for Vesper (tests skip fsync). For crash-safety: commit manifest/snapshot via atomic rename, fsync files, then fsync parent directory before deleting old files; on Windows, document limitations and chosen guarantees.

- [ ] Medium: Parameter validation and API contracts not fully specified
  - Location: retention.hpp (14–30); keep_last_n, keep_total_bytes_max entry checks
  - Details: Behavior for keep_last_n==0, max_total_bytes==0, negative-equivalent overflows, and empty manifest is implicit. Error surface uses a generic io_failed.
  - Recommendation: Document preconditions and outcomes in the header; return precondition_failed for invalid configs; enumerate possible error codes.

- [ ] Low: Unused variable and minor quality issues
  - Location: retention.cpp (15–16 global_last computed but unused)
  - Details: dead/local variable and minor style inconsistency vs. other WAL TUs.
  - Recommendation: Remove unused computation; run static analysis/clang-tidy to prevent reintroduction.

- [ ] Low: Error-code specificity and docs
  - Location: all keep-* and purge_wal removal/stat failures map to error_code::io_failed
  - Details: Coarse error obscures causes (e.g., busy/open file vs. permission vs. not found). Header lacks detailed error model and cross-platform caveats (POSIX unlink open; Windows sharing violations).
  - Recommendation: Expand error taxonomy (busy/open, not_found, permission_denied) and document platform-specific behavior in retention.hpp with examples.

- [ ] Low: Documentation completeness
  - Location: include/vesper/wal/retention.hpp (3–31)
  - Details: Header briefly lists functions but lacks invariants and integration notes: relation to snapshots/manifest, cutoff LSN rule (“delete only fully covered files”), determinism guarantees, and Windows vs POSIX differences.
  - Recommendation: Add an "Invariants & Examples" block with: cutoff LSN semantics; coordination requirement with active writer; determinism notes; examples of count/time/bytes policies; and platform caveats.

- Cross-references (tests/usage):
  - tests/unit/wal_purge_boundaries_test.cpp — verifies inclusive cutoff and determinism at file boundaries.
  - tests/unit/wal_purge_manifest_test.cpp — checks manifest interplay and idempotency.
  - tests/unit/wal_retention_keep_test.cpp — tests keep_last_n and keep_newer_than; snapshot interplay.
  - tests/unit/wal_retention_bytes_test.cpp — tests byte-budget behavior; expand to include budget edge cases and tie-handling.

- Web validation references:
  - SQLite WAL: https://sqlite.org/wal.html; WAL format/tempfiles: https://sqlite.org/walformat.html, https://sqlite.org/tempfiles.html
  - RocksDB retention controls: https://github.com/facebook/rocksdb/wiki/basic-operations (wal_ttl_seconds, wal_size_limit_mb)
  - POSIX durability guidance and directory fsync: https://stackoverflow.com/questions/12990180/what-does-it-take-to-be-durable-on-linux, https://github.com/google/renameio/issues/11, https://danluu.com/file-consistency/
  - Windows file deletion and sharing semantics: https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-deletefilew, https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-createfilew, https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-flushfilebuffers

### include/vesper/wal/checkpoint.hpp + src/wal/checkpoint.cpp

- [ ] High: Consumer name not sanitized → path traversal/reserved-name risk
  - Location: src/wal/checkpoint.cpp path_for() (7–9)
  - Details: `consumer` is concatenated into `<dir>/wal.checkpoints/<consumer>.ckpt` without validation. A crafted consumer like `../foo` or `A/B` can escape the directory; Windows reserved device names (e.g., `CON`, `PRN`) may also fail.
  - Recommendation: Sanitize `consumer` to a conservative whitelist (e.g., `[A-Za-z0-9._-]{1,128}`) and reject/encode others. Document in header.

- [ ] Medium: Checkpoint save is not atomic/durable (crash can corrupt file)
  - Location: src/wal/checkpoint.cpp save() (31–38)
  - Details: Writes directly with `std::ofstream(..., trunc)` and no fsync/rename. Power loss or crash can leave empty/partial files that parse as data_integrity errors. No directory fsync noted.
  - Web validation: POSIX durability guidance (fsync after rename); SQLite documents directory fsync and atomic rename for WAL/checkpoints.
  - Recommendation: Write to temp (e.g., `<name>.ckpt.tmp`), flush+fsync file, then atomic rename to `<name>.ckpt`; fsync parent dir on POSIX. On Windows, use FlushFileBuffers on the file handle; document directory metadata limits.

- [ ] Medium: Type-mask semantics coupled to consumer but not recorded → missed delivery if mask changes
  - Location: src/wal/checkpoint.cpp replay_from_checkpoint() (49–58)
  - Details: `last` is advanced only when frames pass `type_mask` (callback invoked). If a consumer later broadens `type_mask`, earlier frames (with LSN ≤ saved last) that were previously filtered out will be skipped forever.
  - Recommendation: Define invariant: “consumer uses a stable mask.” Alternatively, record mask in the checkpoint file or advance `last` using the last visited valid frame (not only delivered). Document explicitly in header.

- [ ] Medium: Exceptions used in library code (parsing) rather than `from_chars`
  - Location: src/wal/checkpoint.cpp load() (24)
  - Details: Uses `std::stoull` in a try/catch block in library code. Vesper policy allows try/catch in tooling/tests; libraries prefer exception-free paths.
  - Recommendation: Parse with `std::from_chars` (no exceptions) and return `data_integrity` on failure. Keep library exception‑free where possible.

- [ ] Medium: No concurrency/locking contract for checkpoint file
  - Location: src/wal/checkpoint.cpp save()/load() (28–38, 11–26)
  - Details: Two processes/threads updating the same consumer’s checkpoint can race (last‑write‑wins), potentially regressing the high‑watermark. No file locks or atomic scheme defined.
  - Recommendation: Document single‑writer per consumer invariant; or use lock files/advisory locks; or versioned writes with monotonicity checks (reject if new last_lsn < existing).

- [ ] Medium: Callback error model and exception safety
  - Location: replay_from_checkpoint() (51)
  - Details: User callback is invoked directly; if it throws, behavior is unspecified and may unwind through replay. Vesper hot paths avoid exceptions.
  - Recommendation: Document that callbacks must be noexcept; or adapt the replay API to propagate a status via expected from the callback to stop early/error out deterministically.

- [ ] Low: Documentation completeness (invariants, durability, determinism)
  - Location: include/vesper/wal/checkpoint.hpp (14–25)
  - Details: Header lacks: (a) consumer name constraints; (b) checkpoint durability model (write‑then‑rename; fsync); (c) mask invariants; (d) determinism notes; (e) concurrency contract (single writer per consumer).
  - Recommendation: Add an “Invariants & Examples” block with the above; include example usage and caveats.

- [ ] Low: Text file format is ad‑hoc (single line)
  - Location: save()/load() (20–24, 35–38)
  - Details: Simple and fine, but lacks magic/version/checksum. Corruption detection is via parse failure only.
  - Recommendation: Optional: prefix with `CKPT1` magic and consider a trailing checksum for defense‑in‑depth. Not required if write‑then‑rename is adopted.

- Cross-references (tests/usage):
  - tests/unit/wal_replay_checkpoint_test.cpp — verifies basic load/save and mask‑filtered delivery.
  - tests/README.md (Replay checkpoint quick reference) — summarizes format and API usage.

- Web validation references:
  - SQLite WAL checkpointing and directory fsync: [SQLite WAL](https://sqlite.org/wal.html), [Temporary files](https://sqlite.org/tempfiles.html)
  - PostgreSQL checkpoints and WAL semantics: [PostgreSQL Docs](https://www.postgresql.org/docs/current/)
  - RocksDB checkpoints: [RocksDB Checkpoint](https://github.com/facebook/rocksdb/wiki/Read-only-and-Secondary-instances)

### include/vesper/c/vesper.h + src/c/vesper_c_api.cpp

- [ ] Medium: Error-code mapping too coarse (blurs NOT_TRAINED/IO vs INVALID_PARAM)
  - Location: src/c/vesper_c_api.cpp (train 186–191; add 211–215; search 239–242; save 313–317; load 337–340)
  - Details: Many underlying failures (e.g., not trained, I/O) are surfaced as VESPER_ERROR_INVALID_PARAM. This reduces diagnosability and makes programmatic handling harder. Header defines VESPER_ERROR_NOT_TRAINED but it is not used.
  - Recommendation: Map underlying categories: NOT_TRAINED → VESPER_ERROR_NOT_TRAINED; I/O → VESPER_ERROR_IO; invalid inputs → VESPER_ERROR_INVALID_PARAM; unexpected → VESPER_ERROR_INTERNAL/UNKNOWN. Document per‑function error codes in header.

- [ ] Medium: Two‑call metadata pattern returns INVALID_PARAM on short buffer
  - Location: src/c/vesper_c_api.cpp vesper_ivfpq_get_metadata_json() (118–121)
  - Details: When buffer too small, function returns VESPER_ERROR_INVALID_PARAM. Common practice is to signal a distinct precondition/short‑buffer error and still populate out_required_size. Docs should state exact behavior.
  - Recommendation: Clarify docs; optionally introduce a dedicated status (e.g., PRECONDITION_FAILED) for short buffer, or always require size‑query first and document this strictly.

- [ ] Medium: Precondition enforcement delegated fully to C++ layer (dim/m divisibility)
  - Location: src/c/vesper_c_api.cpp vesper_ivfpq_train() (177–185)
  - Details: Header docs specify `m` must divide `dim`; wrapper does not pre‑validate. Relying on deeper errors results in generic INVALID_PARAM.
  - Recommendation: Validate cheap preconditions in the wrapper and map to specific error codes with actionable messages.

- [ ] Medium: ABI/versioning surface is underspecified in public header
  - Location: include/vesper/c/vesper.h (1–122); vesper_version() returns "dev" (31–33)
  - Details: No explicit C ABI version macro in the new header (contrast: legacy include/vesper/vesper_c.h defines VESPER_C_ABI_VERSION). `vesper_version()` returns a non‑semantic placeholder.
  - Recommendation: Add `#define VESPER_C_ABI_VERSION <int>`; return a semantic version string from `vesper_version()`; document compatibility policy.

- [ ] Low: Missing cstring include for std::memcpy
  - Location: src/c/vesper_c_api.cpp vesper_ivfpq_get_metadata_json() (122–124)
  - Details: Uses std::memcpy without including the C header cstring; may compile via transitive includes but is non‑portable.
  - Recommendation: Include the C header cstring explicitly in this TU.

- [ ] Low: Header docs incomplete (thread‑safety, ownership, examples)
  - Location: include/vesper/c/vesper.h (63–118)
  - Details: The reference doc (docs/C_API_Reference.md) covers these, but the public header lacks a concise “Invariants & Usage” block.
  - Recommendation: Add a short section documenting: opaque handle ownership; thread‑safety (search concurrent; train/add not); padding semantics for <k results; metadata two‑call pattern.

- [ ] Low: Unused status enumerator (NOT_TRAINED) not exercised by current functions
  - Location: include/vesper/c/vesper.h (24–31)
  - Details: Code paths do not return VESPER_ERROR_NOT_TRAINED even when applicable (pre‑search/add before train).
  - Recommendation: Use or remove; preferably use where applicable and document.

- Cross‑references (examples/tests/usages):
  - examples/c/vesper_c_example.c — exercises most functions including two‑call metadata and save/load.
  - CMake: vesper_c shared target with VESPER_C_API_EXPORTS defined for symbol visibility (CMakeLists.txt: 255–263).

- Web validation references:
  - Microsoft: Exporting from a DLL (dllexport/dllimport): <https://learn.microsoft.com/en-us/cpp/build/exporting-from-a-dll>
  - Two‑call pattern precedents (snprintf/Windows API style): <https://en.cppreference.com/w/c/io/fprintf>, <https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-getfullpathnamea>

### include/vesper/c/vesper_manager.h + src/c/vesper_manager_c_api.cpp

- [ ] Medium: JSON filter feature flagging and error mapping
  - Location: src/c/vesper_manager_c_api.cpp to_query_config() (196–208)
  - Details: If JSON parsing is disabled or invalid, errors map to VESPER_ERROR_INVALID_PARAM with a message from the parser. This is acceptable but should be explicitly documented as build‑dependent behavior.
  - Recommendation: Document in header: feature may be disabled; error surfaces via INVALID_PARAM and last_error string; no partial state retained.

- [ ] Medium: Input bounds not validated in wrapper (k>0, epsilon>=0, etc.)
  - Location: src/c/vesper_manager_c_api.cpp to_query_config() (179–209), search entrypoints (212–246, 248–285)
  - Details: Relies entirely on underlying C++ layer for validation. Some trivial checks (k>0, nq>0 already checked) could be enforced early for clearer errors.
  - Recommendation: Add minimal prechecks or document that underlying layer validates and returns specific codes; align docs with behavior.

- [ ] Medium: Two‑call stats pattern requires inout_capacity when out_stats non‑NULL
  - Location: src/c/vesper_manager_c_api.cpp vesper_mgr_get_stats() (371–387)
  - Details: Correctly errors if out_stats provided but inout_capacity is NULL. Header mentions two‑call pattern but not this precondition explicitly.
  - Recommendation: Document explicit preconditions and example call sequence in header (mirroring docs/C_API_Reference.md 169–173).

- [ ] Low: Header lacks condensed thread‑safety/ownership section
  - Location: include/vesper/c/vesper_manager.h (78–118)
  - Details: Public header omits thread‑safety model, per‑handle ownership, and filter_json lifetime rules (copied per call).
  - Recommendation: Add concise notes (copy semantics for filter_json; search concurrency permitted; mutating ops require exclusive access).

- [ ] Low: Enum/version stability not stated (ABI evolution)
  - Location: include/vesper/c/vesper_manager.h (12–24, 67–76)
  - Details: Values appear stable but no statement on ABI compatibility across minor versions.
  - Recommendation: Add ABI/versioning policy reference and reserved ranges for future enums.

- Cross‑references (examples/tests/usages):
  - examples/c/vesper_manager_example.c — two‑call stats, filter_json, persistence, update/remove, memory budget.
  - examples/python/vesper_ctypes_example.py — FFI usage via ctypes aligns with opaque handle + status‑code design.

- Web validation references:
  - FFI and opaque handles (libgit2 style error handling): <https://libgit2.org/docs/guides/error-handling/>

### include/vesper/vesper_c.h (legacy C API header)

- [ ] High: Conflicting legacy C API present without implementation in vesper_c library
  - Location: include/vesper/vesper_c.h (56–73); CMake vesper_c sources (257–263); tests/unit/c_api_smoke_test.c (1–13)
  - Details: Legacy header defines a distinct API (collections: open/search/close) and a different vesper_status_t range. The shared library target only builds src/c/vesper_c_api.cpp and src/c/vesper_manager_c_api.cpp; no implementation matches legacy symbols. Consumers including this header will fail to link or pick conflicting types.
  - Recommendation: Deprecate or move legacy header under experimental/ with clear warning; unify on the new C API; provide a migration guide; update examples/tests to avoid including the legacy header.

- [ ] Medium: Type/enum collisions across headers (vesper_status_t)
  - Location: include/vesper/vesper_c.h (16–28) vs include/vesper/c/vesper.h (24–31)
  - Details: Both define vesper_status_t with different enumerators/values; including both leads to ODR/type conflicts.
  - Recommendation: Remove/rename legacy typedef or isolate; ensure a single canonical C API surface is shipped.

- [ ] Medium: Stale docs/example references
  - Location: include/vesper/examples.md (20–34); tests/unit/c_api_smoke_test.c (1–13)
  - Details: Examples/tests reference legacy header/functions not provided by the current vesper_c library build.
  - Recommendation: Update examples/tests to the current C API; ensure CI compiles and runs examples (already added as CTest targets for examples).

- [ ] Low: Versioning macro divergence
  - Location: include/vesper/vesper_c.h (14) vs include/vesper/c/vesper.h (no ABI macro)
  - Details: Legacy header defines VESPER_C_ABI_VERSION whereas the new header does not; creates confusion over the authoritative ABI contract.
  - Recommendation: Consolidate on a single ABI/version scheme in the new header and deprecate the legacy macro.

- Web validation references:
  - SemVer vs ABI compatibility: <https://abi-laboratory.pro/>
  - C API stability guidance (general): <https://nullprogram.com/blog/2016/12/22/>

### Enumeration — include/vesper (non‑C API)

Scope excludes: `include/vesper/c/*` and `include/vesper/vesper_c.h`.

- Top‑level headers
  - include/vesper/collection.hpp
  - include/vesper/error.hpp
  - include/vesper/error_mapping.hpp
  - include/vesper/filter_eval.hpp
  - include/vesper/filter_expr.hpp
  - include/vesper/segment.hpp
  - include/vesper/span_polyfill.hpp
  - include/vesper/wal.hpp
- cache
  - include/vesper/cache/lru_cache.hpp
- core
  - include/vesper/core/cpu_features.hpp
  - include/vesper/core/memory_pool.hpp
  - include/vesper/core/platform_utils.hpp
- filter
  - include/vesper/filter/bitmap_filter.hpp
  - include/vesper/filter/roaring_bitmap_filter.hpp
- index
  - include/vesper/index/aligned_buffer.hpp
  - include/vesper/index/bm25.hpp
  - include/vesper/index/capq.hpp
  - include/vesper/index/capq_calibration.hpp
  - include/vesper/index/capq_dist.hpp
  - include/vesper/index/capq_dist_avx2.hpp
  - include/vesper/index/capq_encode.hpp
  - include/vesper/index/capq_opq.hpp
  - include/vesper/index/capq_q4.hpp
  - include/vesper/index/capq_select.hpp
  - include/vesper/index/capq_util.hpp
  - include/vesper/index/cgf.hpp
  - include/vesper/index/cgf_capq_bridge.hpp
  - include/vesper/index/disk_graph.hpp
  - include/vesper/index/fast_hadamard.hpp
  - include/vesper/index/hnsw.hpp
  - include/vesper/index/hnsw_thread_pool.hpp
  - include/vesper/index/index_manager.hpp
  - include/vesper/index/ivf_pq.hpp
  - include/vesper/index/kmeans.hpp
  - include/vesper/index/kmeans_elkan.hpp
  - include/vesper/index/matryoshka.hpp
  - include/vesper/index/pq_fastscan.hpp
  - include/vesper/index/pq_simple.hpp
  - include/vesper/index/product_quantizer.hpp
  - include/vesper/index/projection_assigner.hpp
  - include/vesper/index/rabitq_quantizer.hpp
- io
  - include/vesper/io/async_io.hpp
  - include/vesper/io/io_uring.hpp
  - include/vesper/io/prefetch_manager.hpp
- kernels
  - include/vesper/kernels/batch_distances.hpp
  - include/vesper/kernels/dispatch.hpp
  - include/vesper/kernels/distance.hpp
  - include/vesper/kernels/backends/ (subdirectory)
- memory
  - include/vesper/memory/numa_allocator.hpp
- metadata
  - include/vesper/metadata/metadata_store.hpp
- platform
  - include/vesper/platform/compiler.hpp
  - include/vesper/platform/filesystem.hpp
  - include/vesper/platform/intrinsics.hpp
  - include/vesper/platform/memory.hpp
  - include/vesper/platform/parallel.hpp
  - include/vesper/platform/platform.hpp
- search
  - include/vesper/search/fusion_algorithms.hpp
  - include/vesper/search/hybrid_searcher.hpp
- tombstone
  - include/vesper/tombstone/tombstone_manager.hpp
- wal
  - include/vesper/wal/checkpoint.hpp
  - include/vesper/wal/frame.hpp
  - include/vesper/wal/io.hpp
  - include/vesper/wal/manifest.hpp
  - include/vesper/wal/replay.hpp
  - include/vesper/wal/retention.hpp
  - include/vesper/wal/snapshot.hpp

---

### include/vesper/error.hpp (public C++ header)

- [ ] High: ABI boundary risk — `struct error` contains `std::string`
  - Location: lines 37–42
  - Details: Exposing `std::string` in public structs crossing shared-library boundaries risks ABI/allocator mismatch across toolchains/CRT (especially MSVC vs MinGW/Clang). Returning/accepting this type in public APIs can fail when client and library are built with different runtimes.
  - Web validation: C++ ABI stability concerns for STL types across DLL boundaries (MSVC CRT rules; Itanium ABI notes). Many libraries avoid STL in ABI; use PImpl or C ABI wrappers.
  - Recommendation (Phase 2): Either document that the C++ API has no cross-compiler ABI guarantee and is intended for static linking/same-toolchain only, or move `error` to internal and expose only `error_code` at the boundary. For human-readable diagnostics, use C API last_error or logging sinks.
- [ ] Medium: Error taxonomy usage and documentation
  - Location: enum class `error_code` (18–35)
  - Details: Additional codes (`invalid_argument`, `not_initialized`, `out_of_range`) exist but are not mentioned in mapping docs. Clarify semantics and when each is used in public APIs.
  - Recommendation: Add doc section enumerating codes with examples and typical call-sites; cross-link ADR‑0005.
- [ ] Low: Header docs could state exceptions policy explicitly
  - Location: file preamble (7–10)
  - Details: Mentions std::expected in design notes; add explicit statement “no exceptions on hot paths; APIs return `std::expected<T, error>` or `std::expected<T, error_code>`”.

### include/vesper/error_mapping.hpp (public C++ header)

- [ ] High: Public header couples to legacy C API
  - Location: include line 4; mapping functions 8–23 and 25–39
  - Details: Includes `vesper/vesper_c.h` (legacy C API). This creates ODR/enum conflicts with the new C API (`include/vesper/c/*`) and leaks deprecated types into the public C++ surface.
  - Recommendation (Phase 2): Remove this header from the public surface or decouple it from legacy. If mapping is needed, define it at the C API boundary in `src/c/*` and target the new `vesper/c/vesper.h` status codes. Align with the approved deprecation plan (Stage 3).
- [ ] Medium: Incomplete mapping for some `error_code` values
  - Location: to_c_status (8–23)
  - Details: `invalid_argument`, `not_initialized`, `out_of_range`, `io_error`, `out_of_memory` do not have explicit mappings; current default falls back to INTERNAL. This loses diagnosability.
  - Recommendation: Provide explicit mappings (e.g., INVALID_PARAM, INTERNAL, IO) consistent with the new C API’s `vesper_status_t`, or document the fallback behavior.
- [ ] Medium: Semantics of `io_eof → OK` must be documented
  - Location: line 20
  - Details: Treating EOF as non-fatal may be correct for bounded reads but could mask truncated files if used broadly.
  - Recommendation: Document which functions consider EOF non-fatal and under which invariants (size pre-known, checksum verified). Otherwise surface as IO error.
- [ ] Low: Transitive include hygiene
  - Details: Pulling a C header into public C++ headers widens the surface and complicates include graphs/compile times.
  - Recommendation: Keep mapping in implementation files at the C boundary; avoid exposing into public C++ headers.

Cross-references

- Depracation plan entries: docs/Implementation/C_API_Legacy_Deprecation_Plan.md §11–§13 recommend isolating or removing the legacy coupling and standardizing C API status codes.
- Usage sites: Mapping is currently not used by new C API implementations (they use direct status enums). Ensure removal does not break internal tools/tests.

### include/vesper/platform/compiler.hpp

- [ ] Medium: TLS macro may be unsafe across DLL boundaries on MSVC
  - Location: 215–219 (`VESPER_THREAD_LOCAL`)
  - Details: Uses `__declspec(thread)` which historically has limitations with dynamically loaded DLLs and non-static CRTs. Modern MSVC supports it, but there are caveats; `thread_local` is generally safer and portable.
  - Web validation: MS Docs (Thread Local Storage), C++ standard thread storage duration (cppreference)
  - Recommendation: Prefer `thread_local` where possible; if keeping `__declspec(thread)`, document constraints (static linking, loader behavior) in platform notes.
- [ ] Medium: Duplicate prefetch APIs (macro vs. functions) risks divergence
  - Location: compiler.hpp 183–193 (`VESPER_PREFETCH`); intrinsics.hpp 57–141 (`prefetch_read/write`)
  - Details: Two distinct user-facing prefetch interfaces increase inconsistency risk (different hint semantics, call sites). Some code uses builtin directly as well.
  - Recommendation: Pick one public abstraction (suggest: `vesper::platform::prefetch_*` functions) and make macros internal or remove. Update caller guidance.
- [ ] Low: `VESPER_ASSUME_ALIGNED` has no effect on MSVC
  - Location: 75–85
  - Details: On MSVC it returns `ptr` unchanged, providing no assumption to the optimizer. On GCC/Clang it uses `__builtin_assume_aligned`.
  - Recommendation: Document as a no-op on MSVC or add `_assume(((uintptr_t)ptr % n) == 0)` guarded by UB-safe preconditions.
- [ ] Low: Packed struct macros are easy to misuse
  - Location: 221–234 (`VESPER_PACKED_*`)
  - Details: Mixing pragma pack and `__attribute__((packed))` via macros across translation units can lead to subtle ABI mismatches if macros are used inconsistently.
  - Recommendation: Restrict usage to tightly scoped internal headers; document required include discipline and pair `PACKED_BEGIN/END` correctly.

### include/vesper/platform/intrinsics.hpp

- [ ] Medium: Public header pulls in `<windows.h>` (transitive include bloat and macro pollution)
  - Location: 20–25
  - Details: Including `<windows.h>` in a public header increases build times and leaks macros despite `NOMINMAX`. `MemoryBarrier` can be replaced with standard C++ or compiler intrinsics.
  - Web validation: MS Docs recommend minimizing `<windows.h>` in public headers; prefer narrow wrappers. C++ `std::atomic_thread_fence` provides portable full-fence.
  - Recommendation: Remove `<windows.h>` from the header; use `std::atomic_thread_fence(std::memory_order_seq_cst)` for full fences and `_ReadWriteBarrier`/`__asm__` for compiler barriers. If OS fence is required, isolate to a `.cpp` or a Windows-only internal header.
- [ ] Medium: Missing `<atomic>` include for fallback fence paths
  - Location: 172–186 (`atomic_signal_fence`), 193–201 (`atomic_thread_fence`)
  - Details: The fallback branches use `<atomic>` symbols without including the header, which can fail on non-MSVC/non-GCC/Clang toolchains.
  - Recommendation: Add `#include <atomic>` in the header unconditionally (harmless on other branches) or guard includes appropriately.
- [ ] Low: Hard-coded cache line size (64) duplicates platform constant
  - Location: 153–170 (`prefetch_range`)
  - Details: Uses `constexpr std::size_t CACHE_LINE_SIZE = 64` instead of `VESPER_CACHE_LINE_SIZE`, risking divergence if the global constant changes.
  - Recommendation: Replace with `VESPER_CACHE_LINE_SIZE` for consistency.
- [ ] Low: `read_timestamp_counter()` not serialized; determinism/accuracy caveats
  - Location: 230–245
  - Details: `rdtsc` is not serializing; readings can be reordered and are not synchronized across cores. Also affected by frequency scaling.
  - Web validation: Intel® 64 and IA-32 Architectures Optimization Reference Manual; use `lfence; rdtsc` or `rdtscp` for ordered reads.
  - Recommendation: Document caveats; for precise timing provide ordered variants or prefer `std::chrono::steady_clock` in portable code paths.

### include/vesper/platform/platform.hpp

- [ ] Medium: Missing `<cstdio>` include for `std::printf`
  - Location: 104–117 (`print_platform_info`)
  - Details: Header uses `std::printf` without including `<cstdio>`. Reliance on transitive includes can break consumers.
  - Recommendation: Add `#include <cstdio>` in the header or replace with iostreams (less preferred in headers).
- [ ] Low: Compile-time SIMD feature flags may mislead consumers
  - Location: 85–96
  - Details: `has_avx2/has_avx512` reflect compile-time macros only; comment notes “real detection should use CPUID”.
  - Recommendation: Wire these fields to the centralized runtime CPU-feature detector used by `kernels::select_backend_auto()` or explicitly document compile-time nature in the struct docs.
- [ ] Low: `num_cores` uses `get_num_threads()` which may reflect OMP threads, not hardware cores
  - Location: 98–101
  - Details: Naming implies physical/logical cores; the function may return maximum OpenMP threads or a policy value.
  - Recommendation: Clarify semantics or switch to `std::thread::hardware_concurrency()` (with documented caveats) for this field; expose OMP threads separately if helpful.
- [ ] Low: Aggregator header includes heavy subsystems
  - Location: 10–16
  - Details: `platform.hpp` re-exports memory/filesystem/parallel; including it widely increases compilation time.
  - Recommendation: Keep as convenience include but prefer including narrower headers in hot-path translation units.


### include/vesper/kernels/batch_distances.hpp

- [ ] High: Exceptions on hot paths due to dynamic allocations
  - Location: 219 (`std::vector<float> all_distances`), 230–246 (full materialization path), 327–413 (fused top‑k: `std::vector<Pair> heap`, `idx`), plus other temporaries
  - Details: Public kernels allocate on the heap inside hot loops. `std::vector` can throw `std::bad_alloc` and violates the "no exceptions on hot paths" policy. Also increases latency variance.
  - Recommendation: Provide APIs that accept caller-provided scratch buffers (or PMR arenas), or return `std::expected<void, error_code>` if allocation fails; avoid materializing full matrices on hot paths. Keep the materializing variant but mark it deprecated and non-hot-path.

- [ ] Medium: Missing `noexcept` on pure compute wrappers
  - Location: 458–481/483–508/510–533 (distance_matrix dispatchers and wrappers)
  - Details: These wrappers do not allocate and call `noexcept` function pointers. They can and should be `noexcept` to align with policy and enable better codegen.
  - Recommendation: Annotate `noexcept` where no allocation/throwing is possible; propagate through all backends.

- [ ] Low: Documentation comments embedded mid-function
  - Location: 420–453 (multi-line doc blocks inside function body)
  - Details: Large doc blocks appear inside the body of `find_nearest_centroids_batch_fused`, which harms readability.
  - Recommendation: Move doc blocks above the function or to a dedicated comment section.

- [ ] Low: Minor overhead constructing `std::span` inside inner loops
  - Location: 458–481 and similar loops
  - Details: Constructing spans in the innermost loops adds tiny overhead. Likely negligible, but can be hoisted when micro-optimizing.
  - Recommendation: Hoist span construction or use raw pointers in the inner loop when targeting maximum throughput; keep clarity-first until profiling demands otherwise.

  Determinism & numeric notes:
  - Reductions use fixed loop orders per (i,j); differences vs scalar are bounded by ULPs. Tests cover parity/tolerances.
  - Tie-breaking is explicit and stable (smaller index wins), ensuring reproducible top‑k ordering for equal scores.

  Web validation notes:
  - Deterministic FP reductions: order-dependence and reproducibility guidance (HPC reproducibility literature)

### include/vesper/kernels/dispatch.hpp

- [ ] Medium: C++ ABI hazard — std::span in public function pointer types
  - Location: 16–29 (`KernelOps` signature)
  - Details: `std::span` layout is not guaranteed to be stable across compilers/standard libraries. Exposing it across a shared-library boundary risks ABI issues for mixed-toolchain consumers.
  - Recommendation: Document that the C++ API is intended for same-toolchain/static linking scenarios; for cross-toolchain/DLL boundaries, prefer the stable C API. Optionally provide C-compatible wrappers for kernel ops.

- [ ] Low: Partial batch coverage in `KernelOps`
  - Location: 22–29
  - Details: Batch ops exist for L2/IP but not for cosine variants; this may be intentional, but the asymmetry can surprise users.
  - Recommendation: Either add batch cosine variants (documenting preconditions/epsilon handling) or explicitly document that batch support is limited to L2/IP.
