## Vesper C API Reference

### Overview
The Vesper C API provides a stable, minimal C interface for integrating Vesper’s vector search capabilities into C and other languages via FFI (Python ctypes/cffi, Rust bindgen, Go cgo, etc.).

Design principles:
- Opaque handles hide C++ internals (`vesper_index_t*`).
- Plain C types and structs only; no exceptions cross the boundary.
- Status-code returns (`vesper_status_t`) + thread‑local last error string.
- Explicit ownership: caller owns input buffers; library owns internal state and returned handles.
- Thread safety: training and add are not thread-safe on the same index; search is safe to call concurrently.

### Getting Started
- CMake
  - Link against the shared library target `vesper_c`.
  - Include directory: `${PROJECT_SOURCE_DIR}/include` (public header: `vesper/c/vesper.h`).

Example CMake snippet
```cmake
add_executable(my_app app.c)
target_include_directories(my_app PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(my_app PRIVATE vesper_c)
```
Windows DLL note: ensure `vesper_c.dll` is on PATH or copied next to your executable. The repository’s example target copies the DLL post-build.

### Error Handling
- All functions return a `vesper_status_t`:
  - `VESPER_OK`, `VESPER_ERROR_INVALID_PARAM`, `VESPER_ERROR_NOT_TRAINED`, `VESPER_ERROR_IO`, `VESPER_ERROR_INTERNAL`, `VESPER_ERROR_UNKNOWN`.
- Use `vesper_get_last_error()` to retrieve a thread‑local UTF‑8 message after a non‑OK status.
- Functions only write to output parameters on success unless documented otherwise (metadata size‑query writes size even with null buffer).

### Data Structures
- `vesper_index_t` (opaque): index handle.
- `vesper_ivfpq_train_params_t`:
  - `nlist` (uint32): number of coarse clusters
  - `m` (uint32): number of PQ sub‑quantizers; must divide dimension
  - `nbits` (uint32): bits per sub‑quantizer (typically 8)
- `vesper_ivfpq_search_params_t`:
  - `k` (uint32): top‑k
  - `nprobe` (uint32): number of lists to probe
- `vesper_ivfpq_stats_t`:
  - `n_vectors` (size_t), `n_lists` (size_t), `m` (size_t), `memory_bytes` (size_t), `avg_list_size` (float)

### API Reference
- Version and errors
  - `const char* vesper_version(void)`
  - `const char* vesper_get_last_error(void)` (thread‑local)

- IVF‑PQ lifecycle
  - `vesper_status_t vesper_ivfpq_create(vesper_index_t** out_index)`
    - Allocates a new empty index. `out_index` must be non‑NULL.
  - `vesper_status_t vesper_ivfpq_destroy(vesper_index_t* index)`
    - Destroys the handle; safe to pass NULL (no‑op) in future versions.

- Training and ingest
  - `vesper_status_t vesper_ivfpq_train(vesper_index_t* index, const float* base_vectors, size_t dim, size_t n, const vesper_ivfpq_train_params_t* params)`
    - Preconditions: `index`, `base_vectors`, `params` non‑NULL; `dim>0`, `n>0`; `params->m` divides `dim`.
  - `vesper_status_t vesper_ivfpq_add(vesper_index_t* index, const uint64_t* ids, const float* base_vectors, size_t n)`
    - Adds `n` vectors (row‑major). `ids` and `base_vectors` must be non‑NULL. Index must be trained.

- Search
  - `vesper_status_t vesper_ivfpq_search(const vesper_index_t* index, const float* query, const vesper_ivfpq_search_params_t* params, uint64_t* out_ids, float* out_distances)`
    - Single query. Output buffers sized to `k`.
  - `vesper_status_t vesper_ivfpq_search_batch(const vesper_index_t* index, const float* queries, size_t nq, const vesper_ivfpq_search_params_t* params, uint64_t* out_ids, float* out_distances)`
    - Batch queries (row‑major, `nq x dim`). Output buffers sized to `nq*k`.
    - If fewer than `k` neighbors are available, results are padded to `k`.

- Diagnostics and state
  - `vesper_status_t vesper_ivfpq_is_trained(const vesper_index_t* index, int* out_is_trained)`
  - `vesper_status_t vesper_ivfpq_get_dimension(const vesper_index_t* index, size_t* out_dim)`
  - `vesper_status_t vesper_ivfpq_get_stats(const vesper_index_t* index, vesper_ivfpq_stats_t* out_stats)`

- JSON metadata
  - `vesper_status_t vesper_ivfpq_set_metadata_json(vesper_index_t* index, const char* json_str)`
    - Stores a user metadata blob; validated on save/load.
  - `vesper_status_t vesper_ivfpq_get_metadata_json(const vesper_index_t* index, char* out_buffer, size_t buffer_size, size_t* out_required_size)`
    - Two‑call pattern: call with `out_buffer=NULL, buffer_size=0` to get `out_required_size`, then allocate and call again to copy. The copied string is NUL‑terminated.

- Persistence
  - `vesper_status_t vesper_ivfpq_save(const vesper_index_t* index, const char* file_path)`
  - `vesper_status_t vesper_ivfpq_load(vesper_index_t* index, const char* file_path)`

### Usage Notes and Examples
- Metadata size query
```c
size_t need=0; vesper_ivfpq_get_metadata_json(idx, NULL, 0, &need);
char* buf = malloc(need);
vesper_ivfpq_get_metadata_json(idx, buf, need, NULL);
```
- Typical workflow
```c
vesper_index_t* idx=NULL; vesper_ivfpq_create(&idx);
vesper_ivfpq_train(idx, data, dim, n, &tp);
vesper_ivfpq_add(idx, ids, data, n);
vesper_ivfpq_search(idx, q, &sp, out_ids, out_dists);
vesper_ivfpq_save(idx, "index.ivfpq");
```

### Thread Safety
- `search` is safe to call concurrently on the same index instance.
- `train`, `add`, and `save/load` must not be called concurrently with each other or with `search` on the same instance.

### Platform Notes
- Windows: ensure `vesper_c.dll` is deployed next to your executable or on PATH. MSVC runtime must be available.
- Linux/macOS: link with `-lvesper_c` and ensure the runtime loader can locate the shared object (RPATH/LD_LIBRARY_PATH/DYLD_LIBRARY_PATH as applicable).

### Complete Example
See `examples/c/vesper_c_example.c` for an end‑to‑end program demonstrating all APIs, including error handling, single and batch search, stats, metadata (two‑call pattern), and persistence round‑trip.

### Migration Guide (C++ → C)
- Replace direct C++ index types with `vesper_index_t*` handles.
- Convert exceptions to status‑code checks and query `vesper_get_last_error()` for diagnostics.
- Replace C++ strings/containers with caller‑allocated C buffers.
- Map training/search parameters to the corresponding C structs.



### IndexManager C API

The IndexManager provides a unified facade over multiple index types (HNSW, IVF‑PQ, DiskANN) with build‑time selection strategies and query‑time routing. The C API follows the same conventions as the IVF‑PQ C API: opaque handle, status codes, and thread‑local last error.

Data structures
- vesper_index_type_t: VESPER_INDEX_HNSW, VESPER_INDEX_IVF_PQ, VESPER_INDEX_DISKANN
- vesper_selection_strategy_t: VESPER_SELECT_AUTO, VESPER_SELECT_MANUAL, VESPER_SELECT_HYBRID
- vesper_manager_build_config_t
  - type: preferred index type when MANUAL; used as a candidate in HYBRID
  - strategy: selection strategy (AUTO, MANUAL, HYBRID)
  - ivf: IVF‑PQ training params used when building IVF‑PQ
  - hnsw: HNSW build params (M, ef_construction, seed)
  - vamana: DiskANN/Vamana build params (degree, L, alpha)
- vesper_query_config_t
  - k, nprobe, ef_search, l_search, epsilon
  - use_exact_rerank, rerank_k, rerank_alpha, rerank_cand_ceiling
  - use_query_planner: enable/disable query planner routing
  - has_preferred_index + preferred_index: force an index type at query time
  - has_filter + filter_json: JSON filter expression applied at query-time (copied per call)
    - Note: JSON parsing may be disabled in this build; if unavailable, calls will return an error such as "JSON parsing not enabled".
- vesper_index_stats_t
  - type (vesper_index_type_t)
  - num_vectors, memory_usage_bytes, disk_usage_bytes
  - avg_query_time_ms, measured_recall, query_count

Function reference
- Lifecycle
  - vesper_status_t vesper_mgr_create(size_t dim, vesper_manager_t** out_mgr)
  - vesper_status_t vesper_mgr_destroy(vesper_manager_t* mgr)
- Build and ingest
  - vesper_status_t vesper_mgr_build(vesper_manager_t* mgr, const float* base_vectors, size_t n, const vesper_manager_build_config_t* cfg)
  - vesper_status_t vesper_mgr_add_batch(vesper_manager_t* mgr, const uint64_t* ids, const float* base_vectors, size_t n)
  - Update/Remove (Phase 2):
    - vesper_status_t vesper_mgr_update(vesper_manager_t* mgr, uint64_t id, const float* vector)
    - vesper_status_t vesper_mgr_update_batch(vesper_manager_t* mgr, const uint64_t* ids, const float* vectors, size_t n)
    - vesper_status_t vesper_mgr_remove(vesper_manager_t* mgr, uint64_t id)
    - vesper_status_t vesper_mgr_remove_batch(vesper_manager_t* mgr, const uint64_t* ids, size_t n)
- Search
  - vesper_status_t vesper_mgr_search(const vesper_manager_t* mgr, const float* query, const vesper_query_config_t* qc, uint64_t* out_ids, float* out_dists)
  - vesper_status_t vesper_mgr_search_batch(const vesper_manager_t* mgr, const float* queries, size_t nq, const vesper_query_config_t* qc, uint64_t* out_ids, float* out_dists)
- Persistence
  - vesper_status_t vesper_mgr_save(const vesper_manager_t* mgr, const char* dir)
  - vesper_status_t vesper_mgr_load(vesper_manager_t* mgr, const char* dir)
- Diagnostics
  - vesper_status_t vesper_mgr_memory_usage(const vesper_manager_t* mgr, size_t* out_bytes)
  - vesper_status_t vesper_mgr_disk_usage(const vesper_manager_t* mgr, size_t* out_bytes)
  - vesper_status_t vesper_mgr_get_stats(const vesper_manager_t* mgr, vesper_index_stats_t* out_stats, size_t* inout_capacity, size_t* out_count)
- Memory budget control (Phase 2)
  - vesper_status_t vesper_mgr_set_memory_budget_mb(vesper_manager_t* mgr, uint32_t mb)
  - vesper_status_t vesper_mgr_get_memory_budget_mb(const vesper_manager_t* mgr, uint32_t* out_mb)

Two‑call stats pattern
- Call vesper_mgr_get_stats(mgr, NULL, NULL, &count) to fetch count
- Allocate stats[count]
- Call vesper_mgr_get_stats(mgr, stats, &capacity=count, &count) to copy entries

Preferred index at query time
- To force a specific index (e.g., IVF‑PQ) at query time, set:
  - qc.use_query_planner = 0
  - qc.has_preferred_index = 1; qc.preferred_index = VESPER_INDEX_IVF_PQ
- If both planner and preferred_index are unset, IndexManager uses a heuristic or built‑in planner (when enabled) to select an index.

Thread safety
- Build, add_batch, save, and load are not thread‑safe when called concurrently with each other or with search on the same manager instance.
- search and search_batch are safe to call concurrently.

Complete workflow example
- See examples/c/vesper_manager_example.c for an end‑to‑end program:
  - create → build (Manual+IVF‑PQ) → add_batch → search (single+batch) → stats (two‑call) → save → destroy → create → load → search



### Language bindings

The C API is designed to be consumed from other languages via FFI.
- Python (ctypes): see examples/python/vesper_ctypes_example.py
- Rust (bindgen): see examples/rust/vesper_bindgen_example.rs

Notes
- Ensure the shared library (vesper_c.dll / libvesper_c.so / libvesper_c.dylib) is discoverable by the OS loader (PATH/LD_LIBRARY_PATH/DYLD_LIBRARY_PATH or by placing the library next to the executable/script).
- The examples demonstrate both IVF-PQ and IndexManager flows, including error handling and two-call patterns.
- The examples are developer-run and not wired into CI.
