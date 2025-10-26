# C API Legacy Deprecation Plan — Stage 1: Impact Analysis & Inventory

Status: Draft (Phase 1, Audit-Only)
Owner: Vesper Engineering (Pre-release, correctness-first)
Scope: Consolidate conflicting C API surfaces; deprecate legacy `include/vesper/vesper_c.h` safely.

---

## 0) Context & Problem Statement

Two conflicting C API designs exist:
- Active/implemented API: `include/vesper/c/vesper.h` + `include/vesper/c/vesper_manager.h` with implementations in `src/c/vesper_c_api.cpp` and `src/c/vesper_manager_c_api.cpp` (target `vesper_c`).
- Legacy/unimplemented API: `include/vesper/vesper_c.h` (collection-based; different status codes; conflicting `vesper_status_t`).

Risks:
- ODR/type conflicts (`vesper_status_t` defined twice with different enumerators/values)
- Link failures (legacy functions not implemented in `vesper_c`)
- Consumer confusion (docs/examples/tests referencing legacy header)

This document covers Stage 1 only (impact analysis). No code changes.

---

## 1) Inventory of References to Legacy Header and Symbols

Authoritative findings (paths are repository-relative):

- Code/tests including legacy header
  - tests/unit/c_api_smoke_test.c (lines 1–13) — includes `<vesper/vesper_c.h>`; calls `vesper_open_collection`, `vesper_close_collection`, uses `vesper_collection_t`, `VESPER_OK`.

- Documentation/examples referencing legacy header or symbols
  - include/vesper/examples.md (C ABI example block) — includes `<vesper/vesper_c.h>`; calls `vesper_open_collection`, `vesper_search`, `vesper_close_collection`; uses `vesper_search_params_t`, `vesper_search_result_t`.
  - docs/Open_issues.md (mentions public C API header at include/vesper/vesper_c.h)
  - IMMEDIATE_ACTION_PLAN.md (references include/vesper/vesper_c.h among restoration hints)
  - architecture/adr-0006-api-surface.md (proposed stable C ABI under include/vesper/vesper_c.h)

- Header coupling (internal)
  - include/vesper/error_mapping.hpp — includes `vesper/vesper_c.h` and maps `core::error_code` ↔ legacy `vesper_status_t` enumerators.

Notes:
- No production source files in src/ include the legacy header (only the mapping header above, which appears unused by current build targets).

---

## 2) Legacy C API Surface (as declared in include/vesper/vesper_c.h)

Opaque handle and POD structs:
- `typedef struct vesper_collection_t_ vesper_collection_t;`
- `typedef struct { const char* metric; uint32_t k; float target_recall; uint32_t nprobe; uint32_t ef_search; uint32_t rerank; } vesper_search_params_t;`
- `typedef struct { uint64_t id; float score; } vesper_search_result_t;`

Functions:
- Lifecycle: `vesper_open_collection(const char* path, vesper_collection_t** out)`, `vesper_close_collection(vesper_collection_t* c)`
- Ingest/maintenance: `vesper_insert(...)`, `vesper_remove(...)`, `vesper_seal_segment(...)`, `vesper_compact(...)`, `vesper_snapshot(...)`, `vesper_recover(...)`
- Search: `vesper_search(vesper_collection_t* c, const float* query, size_t dim, const vesper_search_params_t* p, vesper_search_result_t* out, size_t out_cap, size_t* out_size)`

Error codes (legacy):
- `vesper_status_t` enumerators in the 1001/2001/... ranges (e.g., `VESPER_E_IO_FAILED=1001`, `VESPER_E_CONFIG_INVALID=2001`, ..., `VESPER_E_INTERNAL=9001`)

Contrast with active C API:
- New `vesper_status_t` uses compact codes: `VESPER_OK=0`, `VESPER_ERROR_UNKNOWN=1`, `VESPER_ERROR_INVALID_PARAM=2`, `VESPER_ERROR_NOT_TRAINED=3`, `VESPER_ERROR_IO=4`, `VESPER_ERROR_INTERNAL=5`.
- New API is index- or manager-centric, not collection-centric; examples and CMake targets point to it.

---

## 3) CMake Exposure & Build Reality

- Shared library: `add_library(vesper_c SHARED src/c/vesper_c_api.cpp src/c/vesper_manager_c_api.cpp)`
  - Public include: `${PROJECT_SOURCE_DIR}/include`
  - Windows export define: `VESPER_C_API_EXPORTS`
- No translation units in `vesper_c` implement legacy functions (`vesper_open_collection`, `vesper_search`, etc.).
- Examples wired into CI link to `vesper_c` and include the new headers:
  - `examples/c/vesper_c_example.c` → includes `vesper/c/vesper.h`
  - `examples/c/vesper_manager_example.c` → includes `vesper/c/vesper.h` and `vesper/c/vesper_manager.h`
- Tests under Catch2 aggregate do not compile `tests/unit/c_api_smoke_test.c` (legacy C); that file exists but is not built by default.

Implication: Any consumer including the legacy header and linking against `vesper_c` will fail at link time due to missing symbols.

---

## 4) External Consumers / Examples Depending on Legacy API

- Built/linked examples in this repo depend on the new API (OK).
- A documentation example (`include/vesper/examples.md`) depends on legacy API (stale; misleading).
- The `tests/unit/c_api_smoke_test.c` depends on legacy API but is not wired to CMake’s test targets (latent hazard).
- No evidence of third-party bindings in-tree that include the legacy header (Python/Rust examples reference the new API path).

---

## 5) Risks and Impact Summary

- API confusion: Two different `vesper_status_t` definitions and incompatible function sets under the same include tree cause ambiguity and potential ODR/type conflicts for consumers.
- Link-time failures: Legacy API symbols are not present in `vesper_c`.
- Documentation drift: Examples and ADRs point to the legacy header, not the implemented API.
- Hidden coupling: `include/vesper/error_mapping.hpp` codifies the legacy error code mapping; keeping it alongside the new C API increases divergence.

Severity: High (public API surface area).
Likelihood: Medium (because defaults build/link the new examples/tests).
Blast radius: External FFI consumers and docs readers; internal developers may also be confused.

---

## 6) Evidence (pointers)

- Legacy header and symbols: `include/vesper/vesper_c.h` (opaque handle, search params, result, legacy status codes; functions open/close/insert/remove/search/seal/compact/snapshot/recover)
- New C API headers: `include/vesper/c/vesper.h`, `include/vesper/c/vesper_manager.h`
- C API implementation sources: `src/c/vesper_c_api.cpp`, `src/c/vesper_manager_c_api.cpp`
- CMake: `CMakeLists.txt` lines ~255–267 define `vesper_c` target (only new API sources)
- Legacy references:
  - `tests/unit/c_api_smoke_test.c` — includes legacy header, calls legacy functions
  - `include/vesper/examples.md` — C snippet includes legacy header and uses legacy symbols
  - `include/vesper/error_mapping.hpp` — includes legacy header and maps `core::error_code` ↔ legacy `vesper_status_t`
  - Docs: `docs/Open_issues.md`, `IMMEDIATE_ACTION_PLAN.md`, `architecture/adr-0006-api-surface.md`

---

## 7) Stage 1 Conclusions (Audit-Only)

- The legacy header is present and referenced by at least one test and one documentation example but has no corresponding implementation in the `vesper_c` shared library.
- The active C API (new headers + implementations) is the only C API actually built and shipped by the repository’s CMake.
- Immediate risk is primarily confusion and incorrect consumer usage rather than CI breakage (since legacy references are not compiled). However, the presence of conflicting headers in `include/vesper/` violates clarity and increases support burden.

Recommended direction (to be refined in Stage 3):
- Consolidate on the new C API as the canonical surface.
- Deprecate and isolate the legacy header (move under `experimental/` or `deprecated/` with clear banner), or provide a thin compatibility shim that forwards to the new API where feasible.
- Update all docs/examples/tests to align with the new API.

---

## 8) Checkpoint Gate (STOP)

This completes Stage 1 (Impact Analysis & Inventory).
Requesting approval to proceed to Stage 2: API Mapping & Migration Guide (legacy → new), including a function-by-function mapping and error-code conversion table, plus identification of any semantic gaps.



---

## 9) Stage 2: API Mapping & Migration Guide (Audit-Only)

Status: Draft for review (no code changes). All mappings verified against:

- New headers: include/vesper/c/vesper.h, include/vesper/c/vesper_manager.h
- Implementations: src/c/vesper_c_api.cpp, src/c/vesper_manager_c_api.cpp
- Legacy header: include/vesper/vesper_c.h

### 9.1 Function-by-function mapping (legacy → new)

| Legacy function | New API equivalent | Notes |
|---|---|---|
| vesper_open_collection(path, out) | Option A (single-index): vesper_ivfpq_create(&idx) → vesper_ivfpq_load(idx, file_path) | Loads an IVF-PQ index file. No “collection” container. |
|  | Option B (multi-index): vesper_mgr_create(dim, &mgr) → vesper_mgr_load(mgr, dir_path) | Requires dim at create; persisted metadata must match. Consider helper/open in Phase 2. |
| vesper_close_collection(c) | vesper_ivfpq_destroy(idx) or vesper_mgr_destroy(mgr) | Depends on chosen path (A or B). |
| vesper_insert(c, id, vec, dim) | vesper_ivfpq_add(idx, &id, vec, 1) | IVFPQ assumes trained index; dim is implicit from training. |
|  | vesper_mgr_add_batch(mgr, &id, vec, 1) | Manager provides routing over index family. |
| vesper_remove(c, id) | vesper_mgr_remove(mgr, id) | No remove in IVFPQ C API; use manager. |
| vesper_search(c, query, dim, params, out, out_cap, out_size) | vesper_ivfpq_search(idx, query, {k,nprobe}, out_ids, out_dists) | No dim arg; dim fixed by index. params.metric/target_recall ignored. |
|  | vesper_mgr_search(mgr, query, vesper_query_config_t{...}, out_ids, out_dists) | Map params: k→k; nprobe→nprobe; ef_search→ef_search; target_recall→epsilon (approx); rerank→use_exact_rerank (+rerank fields). |
| vesper_seal_segment(c) | No direct equivalent | Internal maintenance; not exposed in new C API. |
| vesper_compact(c) | No direct equivalent | Compaction internal; not exposed in new C API. |
| vesper_snapshot(c) | Approx: vesper_mgr_save(mgr, dir) or vesper_ivfpq_save(idx, file) | Not identical semantics to snapshot; save persists durable state. |
| vesper_recover(c) | Approx: vesper_mgr_load(mgr, dir) or vesper_ivfpq_load(idx, file) | Load triggers recovery from persisted artifacts; not a WAL replay API. |

### 9.2 Error code mapping (legacy → new)

| Legacy vesper_status_t | New vesper_status_t | Notes |
|---|---|---|
| VESPER_OK | VESPER_OK | Identical success. |
| VESPER_E_IO_FAILED (1001) | VESPER_ERROR_IO | Direct IO category mapping. |
| VESPER_E_CONFIG_INVALID (2001) | VESPER_ERROR_INVALID_PARAM | Configuration/argument issues. |
| VESPER_E_DATA_INTEGRITY (3001) | VESPER_ERROR_INTERNAL | Loss of specificity; consider expanding new enum later. |
| VESPER_E_PRECONDITION_FAILED (4001) | VESPER_ERROR_INVALID_PARAM or VESPER_ERROR_NOT_TRAINED | Case-based: training/ready state → NOT_TRAINED; otherwise INVALID_PARAM. |
| VESPER_E_RESOURCE_EXHAUSTED (5001) | VESPER_ERROR_INTERNAL | Loss of specificity (OOM/limits). |
| VESPER_E_NOT_FOUND (6001) | VESPER_ERROR_INVALID_PARAM | Loss of specificity; could merit distinct NOT_FOUND in future. |
| VESPER_E_UNAVAILABLE (7001) | VESPER_ERROR_UNKNOWN | Loss of specificity; transient/infra status not modeled. |
| VESPER_E_CANCELLED (8001) | VESPER_ERROR_UNKNOWN | No explicit CANCELLED. |
| VESPER_E_INTERNAL (9001) | VESPER_ERROR_INTERNAL | Direct mapping. |

Observation: New API compresses categories, losing granularity for programmatic handling. If compatibility needs arise, Stage 3 can propose extending the new enum (non-breaking if values appended and headers updated).

### 9.3 Type/struct mapping

- vesper_collection_t → choose one:
  - vesper_index_t (IVFPQ path): single-index handle
  - vesper_manager_t (manager path): multi-index facade

- vesper_search_params_t →
  - IVFPQ: map to vesper_ivfpq_search_params_t { k, nprobe } (ignore ef_search/metric/target_recall/rerank)
  - Manager: map to vesper_query_config_t: k, nprobe, ef_search, epsilon≈target_recall, use_exact_rerank (+ rerank_k/alpha/cand_ceiling as needed). Metric is a build-time/index attribute, not a per-call field.

- vesper_search_result_t {id, score} →
  - New API returns parallel arrays: out_ids[], out_dists[] of length k (or nq*k). Caller owns buffers; pad semantics defined in implementations.

### 9.4 Semantic differences (conceptual)

- Model: Legacy is collection-centric with mixed responsibilities (WAL/segments/maintenance). New API separates concerns into index (IVFPQ) and an IndexManager facade (build, routing, search, persistence).
- Lifecycle: Legacy open/close vs. new create/destroy + save/load. New IVFPQ requires explicit train() before add/search. Manager build() chooses/constructs index family.
- Parameters: Legacy search takes metric/target_recall per call; new API sets metric at build time and uses epsilon (approximate) and rerank knobs in manager configs; IVFPQ search only exposes k and nprobe in C API.
- Thread-safety: New docs imply search is safe concurrently; train/add not safe on same handle. Legacy stated single-writer semantics. Behavior is functionally aligned but needs explicit C API header docs (tracked in audit).

### 9.5 Feature gap analysis

- Maintenance ops (seal_segment, compact, snapshot, recover): not exposed in new C API. Current equivalents are save/load (persistence) and internal background maintenance. If external control is required, Stage 3 can propose minimal, well-specified maintenance APIs on the manager with deterministic semantics and durability notes.
- Fine-grained error classes (NOT_FOUND, RESOURCE_EXHAUSTED, UNAVAILABLE, CANCELLED, DATA_INTEGRITY): not represented in new status enum. Consider extending in a backward-compatible way if needed by FFI consumers.

### 9.6 Migration code patterns (before → after)

Pattern A: Open existing on-disk artifact and search (single-index/IVFPQ)

```c
/* Legacy */
vesper_collection_t* c = NULL;
vesper_open_collection("/tmp/vesper", &c);
vesper_search_params_t p = {"l2", 10, 0.95f, 8, 0, 0};
vesper_search_result_t out[10]; size_t out_sz=0;
vesper_search(c, q, dim, &p, out, 10, &out_sz);
vesper_close_collection(c);

/* New (IVFPQ) */
vesper_index_t* idx = NULL;
vesper_ivfpq_create(&idx);
vesper_ivfpq_load(idx, "/tmp/vesper.ivfpq");
vesper_ivfpq_search_params_t sp = { .k=10, .nprobe=8 };
uint64_t ids[10]; float dists[10];
vesper_ivfpq_search(idx, q, &sp, ids, dists);
vesper_ivfpq_destroy(idx);
```

Pattern B: Build via manager, add, search (multi-index)

```c
/* Legacy */
vesper_collection_t* c = NULL; vesper_open_collection(dir, &c);
vesper_insert(c, id, vec, dim);
vesper_search_params_t p = {"l2", 10, 0.0f, 8, 64, 0};
vesper_search(c, q, dim, &p, out, 10, &out_sz);
vesper_close_collection(c);

/* New (Manager) */
vesper_manager_t* mgr = NULL; vesper_mgr_create(dim, &mgr);
vesper_manager_build_config_t cfg = { .type=VESPER_INDEX_IVF_PQ };
vesper_mgr_build(mgr, base, n, &cfg);
vesper_mgr_add_batch(mgr, &id, vec, 1);
vesper_query_config_t qc = { .k=10, .nprobe=8, .ef_search=64, .use_exact_rerank=0 };
uint64_t ids[10]; float dists[10];
vesper_mgr_search(mgr, q, &qc, ids, dists);
vesper_mgr_destroy(mgr);
```

---

## 10) Checkpoint Gate (STOP)

Stage 2 deliverables (mapping tables, type mapping, semantic differences, gaps, and migration patterns) are complete for review.
Requesting approval to proceed to Stage 3: Deprecation Strategy Document (timeline, migration path, compatibility options, communication plan, validation).

---

## 11) Stage 3: Deprecation Strategy Document (Audit-Only Plan)

Status: Draft for approval (no code changes yet). Assumptions: pre-release; Stage 1 indicates limited external consumers (legacy usage only in docs + an unbuilt test); active C API is already canonical in build.

### 11.1 Deprecation timeline (phased)

- Immediate (Phase 2 – next PR)
  - Canonicalize docs to the new API: update docs/C_API_Reference.md, include/vesper/examples.md, and examples to reference `vesper/c/` headers exclusively
  - Fix tests/examples: remove or migrate `tests/unit/c_api_smoke_test.c` to the new API; confirm examples build/link/run
  - Isolate legacy header to prevent accidental use: either remove `include/vesper/vesper_c.h` from install/exports or move to `include/vesper/deprecated/` with a banner comment; ensure it is not referenced by any build target
  - Resolve internal coupling: replace or remove `include/vesper/error_mapping.hpp` dependency on legacy status codes; align with the new status enum
  - Add MIGRATIONS.md (C API) pointing to Stage 2 mapping and patterns

- Short-term (next release cycle)
  - If legacy header remains for discoverability, add compile-time deprecation banner and explicit guidance to migrate; do not ship it in install artifacts
  - Provide optional helper utilities (if needed) to smooth migration (e.g., thin convenience wrappers using the new API), documented as non-stable
  - Confirm packaging/export sets contain only new C API headers and a single shared library target (`vesper_c`)

- Medium-term (1–2 releases)
  - If a deprecation period was chosen, remove any temporary compatibility aids and delete the legacy header from the tree
  - Freeze any convenience helpers or replace with clearly supported APIs if adoption indicates need

- Long-term (final removal)
  - Ensure zero references to `include/vesper/vesper_c.h` across repo and docs; remove migration notes from the primary docs (retain in historical notes/CHANGELOG)

### 11.2 Migration path (recommended)

Default recommendation: migrate to the Manager C API (`vesper_manager_t`) for forward-compatibility across index families and query planning. For minimal/single-index deployments, the IVFPQ C API is acceptable and lighter-weight.

Step-by-step (manager path):

1) Replace `#include <vesper/vesper_c.h>` with `#include "vesper/c/vesper.h"` and `#include "vesper/c/vesper_manager.h"`
2) Replace `vesper_collection_t*` with `vesper_manager_t*`
3) Replace open/close with `vesper_mgr_create(dim, &mgr)` + `vesper_mgr_load(mgr, dir)` and `vesper_mgr_destroy(mgr)`
4) Replace ingest/maintenance calls with `vesper_mgr_add_batch`, `vesper_mgr_update[_batch]`, `vesper_mgr_remove[_batch]`
5) Replace search with `vesper_mgr_search`/`vesper_mgr_search_batch` using `vesper_query_config_t` (map legacy k, nprobe, ef_search; set epsilon≈target_recall; set rerank knobs as needed)
6) Replace snapshot/recover with `vesper_mgr_save`/`vesper_mgr_load` (note: not identical to WAL snapshot semantics; see Stage 2)
7) Update error handling to use the new `vesper_status_t` codes; where legacy code differentiated NOT_FOUND/RESOURCE_EXHAUSTED/etc., handle via return code + logs until/if new enum is extended

Step-by-step (IVFPQ path):

1) Replace include with `#include "vesper/c/vesper.h"`
2) Replace handle with `vesper_index_t*`; use `vesper_ivfpq_create`/`vesper_ivfpq_destroy`
3) Replace open with `vesper_ivfpq_load`; close with destroy
4) Train if building from scratch: `vesper_ivfpq_train`; then add with `vesper_ivfpq_add`
5) Search with `vesper_ivfpq_search`/`_batch` using `vesper_ivfpq_search_params_t {k,nprobe}`
6) Persist with `vesper_ivfpq_save`/`vesper_ivfpq_load`

Compatibility aids (only if truly needed):

- A thin, header-only shim that maps `vesper_open_collection` → manager/IVFPQ create+load, `vesper_search` → mgr/IVFPQ search, etc. Risks: semantic mismatch, maintenance burden, confusion. Recommend avoiding unless external users are discovered.

### 11.3 Compatibility options (evaluation + recommendation)

- Option A — Clean break (Recommended)
  - Action: Remove legacy header from install and codebase (or move under deprecated/ not shipped); update all references to new API; no shim
  - Pros: eliminates ODR/status conflicts, reduces confusion and maintenance; aligns with pre-release status and current build reality
  - Cons: immediate breaking change for any out-of-tree consumer (none identified in Stage 1)

- Option B — Deprecation period with compatibility layer
  - Action: Keep a shim or banner-deprecated header; forward calls to new API; remove after N releases
  - Pros: softer landing for unknown consumers; can emit deprecation diagnostics
  - Cons: added complexity and risk (semantic drift), extends confusion, delays consolidation

- Option C — Namespace isolation (legacy renames)
  - Action: Rename legacy types/enums to avoid ODR conflicts and keep both side-by-side while deprecating
  - Pros: avoids ODR at cost of duplicate surfaces
  - Cons: maximal complexity and confusion; not justified for pre-release

Recommendation: Option A (Clean break) given pre-release status, Stage 1 impact analysis (no active external consumers), and the risk of conflicting `vesper_status_t` definitions.

### 11.4 Breaking changes (surface, behavior, build)

- Header paths: `include/vesper/vesper_c.h` → `include/vesper/c/vesper.h` (+ `vesper/c/vesper_manager.h` for manager)
- Handles/types: `vesper_collection_t` → `vesper_manager_t` or `vesper_index_t`; `vesper_search_params_t` → `vesper_query_config_t` or `vesper_ivfpq_search_params_t`
- Functions: open/close/insert/remove/search/seal/compact/snapshot/recover → manager/IVFPQ create/destroy/build/add/remove/search/save/load
- Error codes: legacy 1001+/2001+ categories collapse into new compact enum; programmatic distinction for NOT_FOUND/EXHAUSTED/etc. is lost (Stage 2 mapping provided)
- Semantics: per-call metric/target_recall removed; metric becomes build-time; snapshot/recover map to save/load (not WAL snapshot semantics)
- Build/link: Consumers must include `vesper/c/*` and link against the `vesper_c` shared library; legacy symbols do not exist

### 11.5 Communication plan

- Docs: Update docs/C_API_Reference.md to declare `include/vesper/c/*` as canonical; add “Deprecation of legacy header” section linking to this plan and MIGRATIONS.md
- Examples: Replace legacy example block in `include/vesper/examples.md` with new API example; ensure examples compile and run in CI
- ADRs/roadmap: Update architecture/adr-0006-api-surface.md and docs/Open_issues.md to reference the new C API; add ADR addendum noting the deprecation decision
- Release notes: Add CHANGELOG/release notes entry; highlight breaking changes and migration steps
- In-tree notices: If legacy header is kept briefly, add a top-of-file banner comment declaring deprecation and pointing to MIGRATIONS.md (do not ship it in installs)

### 11.6 Validation and rollback plan (acceptance criteria)

Acceptance criteria for Phase 2 PR(s):

- Build & link: `vesper_c` builds cleanly; no references to `include/vesper/vesper_c.h` remain in the tree (except, if temporarily retained under `deprecated/`, it is not included by any target)
- Tests/examples: legacy `c_api_smoke_test.c` replaced/migrated; examples for both IVFPQ and Manager compile and run
- Lints/sanitizers: zero warnings (`-Werror`), ASan/TSan/UBSan clean; static analysis clean (no new violations)
- Docs: C API reference and examples updated; ADR and Open_issues updated; MIGRATIONS.md added
- API correctness: New/updated smoke tests cover create/load/search/save for both IVFPQ and Manager; determinism unaffected; no ODR/enum conflicts

Rollback plan:

- If unforeseen external consumers surface, revert removal/move of the legacy header and ship a short-lived shim behind a build option (e.g., `VESPER_ENABLE_C_LEGACY_SHIM=ON`) while communicating the migration schedule; maintain for a single release then remove

---

## 12) Checkpoint Gate (STOP)

Stage 3 plan is ready for review.
Requesting approval to proceed to Stage 4: Correctness Verification Plan (tests, CI gates, ABI/serialization checks, performance/determinism validation, and explicit acceptance criteria mapping).

## 13) Stage 4: Correctness Verification Plan (Audit-Only)

Status: Draft for approval (no code changes). This plan specifies tests, CI gates, ABI/serialization checks, perf/determinism validation, acceptance mapping, docs verification, and rollback.

### 13.1 Test coverage plan (Phase 2 implementation targets)

- Smoke tests (C API lifecycle)
  - IVFPQ: create → (optional train) → add → search → save → destroy; load existing → search → destroy
  - Manager: create(dim) → build(base) → add/update/remove → search → save → destroy; load existing → search → destroy
  - Batch search coverage for both paths
- Migrate legacy test
  - Replace `tests/unit/c_api_smoke_test.c` (legacy) with new C API smoke tests (two files: `c_api_ivfpq_smoke.c`, `c_api_manager_smoke.c`)
- Error handling tests
  - Null pointers for in/out parameters return `VESPER_ERROR_INVALID_PARAM`
  - Not-trained search returns `VESPER_ERROR_NOT_TRAINED`
  - Invalid k/nprobe/ef_search detected as `VESPER_ERROR_INVALID_PARAM`
  - Two-call buffer sizing for metadata JSON returns required size and fills buffer correctly
  - Persistence error surfaces as `VESPER_ERROR_IO`
- Thread‑safety validation
  - Concurrent search on same handle (multiple threads) yields correct results; no data races (TSan clean)
  - Single-writer constraints: concurrent add/train on same handle is rejected or serialized (documented behavior); add while searching on separate handles validated where supported
- Determinism tests
  - Fixed seeds; repeated build/search runs yield identical IDs and distances within FP tolerance (document epsilon/ULP if needed)
- Examples build & run
  - `examples/c/vesper_c_example.c` and `examples/c/vesper_manager_example.c` compile, link, and run (return 0) via CTest targets

### 13.2 CI gates and build validation

- Zero warnings, `-Werror`: MSVC 2022, GCC ≥ 12, Clang ≥ 15 across OS matrix
- Sanitizers: ASan/UBSan/TSan clean on Debug builds; failures block
- Static analysis: Clang‑Tidy bundles (modernize/performance/readability/bugprone/concurrency/security); no new violations
- Build/link: `vesper_c` shared library links with no unresolved symbols; exported symbol check includes expected `vesper_*` functions
- Install/export verification
  - Installed headers include only `include/vesper/c/*`
  - Legacy `include/vesper/vesper_c.h` excluded or isolated under `deprecated/` and not installed

### 13.3 ABI stability and serialization checks

- ODR/enum conflicts eliminated
  - Ensure only the new `vesper_status_t` is present in shipped headers; confirm no references to legacy header in build targets
- Opaque handles
  - No public struct layout exposure; attempts to take `sizeof(vesper_index_t)` should not be possible in headers
- Two‑call buffer pattern correctness
  - `vesper_ivfpq_get_metadata_json`: first call with buffer_size = 0 returns required size; second call fills; verify NUL termination
- Serialization round‑trip
  - IVFPQ: save → load → stats match (n_vectors, n_lists, m, memory_bytes within tolerance) and search parity on a fixed query set
  - Manager: save → load → selected index type preserved; search parity
- Cross‑platform
  - Windows/Linux/macOS matrix: compile, link, run smoke tests; DLL/export visibility validated on Windows via symbol inspection

### 13.4 Performance and determinism validation

- Bench smoke (budget guardrails; not full perf suite)
  - Query latency: P50 ≤ 1–3 ms; P99 ≤ 10–20 ms on representative synthetic dataset (document hardware/flags)
  - Ingestion throughput (where applicable): 50–200k vec/s target band
- Determinism
  - Re-run fixed‑seed build/search 3×; identical top‑k IDs and distances (within documented FP tolerance)
- Memory correctness
  - ASan/UBSan/TSan clean; optional Valgrind run on Linux shows zero leaks for example binaries
- Migration impact
  - No expected perf change (implementation unchanged); record evidence from smoke measurements

### 13.5 Acceptance criteria mapping (to Stage 3 §11.6)

- Build & link clean
  - Tests: build of `vesper_c`; link + symbol check; examples compile/run
  - CI: Build jobs per compiler/OS; artifact/symbol inspection step
  - Local: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j`
- Tests/examples updated
  - Tests: `c_api_ivfpq_smoke.c`, `c_api_manager_smoke.c` green; examples return 0 under CTest
  - CI: `ctest --output-on-failure` on all matrix jobs
  - Local: `cd build && ctest -R c_api_`
- Lints/sanitizers/static analysis
  - CI: Tidy gating; ASan/UBSan/TSan jobs must pass; zero warnings enforced
  - Local: run sanitizers build and unit tests; verify no diagnostics
- Docs updated
  - Files: C_API_Reference.md, include/vesper/examples.md, ADR(s), Open_issues.md, MIGRATIONS.md, CHANGELOG.md
  - CI: doc lint (if enabled) and a docs verification checklist; reviewer sign‑off required
- API correctness/determinism/ODR
  - Tests: two‑call buffer tests, determinism tests, round‑trip save/load parity
  - CI: determinism tests (fixed seeds) included in unit suite; ODR check implicit via single shipped header set

### 13.6 Documentation verification checklist

- docs/C_API_Reference.md
  - Declares `include/vesper/c/*` as canonical; adds legacy deprecation section linking MIGRATIONS.md
- include/vesper/examples.md
  - Legacy example replaced with new API example; snippet compiles (as part of example program); run in CI
- architecture/adr-0006-api-surface.md
  - Updated to reference new C API; add note on deprecation decision (Option A)
- docs/Open_issues.md
  - Legacy references updated/closed; link to migration plan
- MIGRATIONS.md (new)
  - Contains Stage 2 mapping tables and patterns; referenced from docs and release notes
- CHANGELOG.md / release notes
  - Breaking changes and migration steps documented

Verification: reviewer checklist + CI job that builds and runs examples, and (optionally) parses header install set to ensure only `include/vesper/c/*` is present.

### 13.7 Rollback procedure

- Triggers
  - Discovery of external consumer blocked by removal; critical bug in new API surfaced late; ABI break uncovered externally
- Steps
  - Revert removal/move commits of legacy header; reintroduce a controlled, short‑lived shim behind a build option (e.g., `VESPER_ENABLE_C_LEGACY_SHIM=ON`)
  - Restore legacy example/test (gated behind the same option) while communicating the migration timeline
  - Cut a patch release with deprecation warnings; schedule final removal after one release cycle
- Validation
  - CI green with shim option ON and OFF; examples/tests pass in both modes; docs updated to reflect temporary shim and sunset date

---

## 14) Final Checkpoint Gate (STOP)

Stage 4 plan is complete and ready for approval. Upon approval, Phase 2 implementation can proceed following this verification plan and the Stage 3 strategy.
