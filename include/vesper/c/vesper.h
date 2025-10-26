#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Symbol visibility
#if defined(_WIN32)
  #if defined(VESPER_C_API_EXPORTS)
    #define VESPER_C_API __declspec(dllexport)
  #else
    #define VESPER_C_API __declspec(dllimport)
  #endif
#else
  #define VESPER_C_API __attribute__((visibility("default")))
#endif

#include <stddef.h>
#include <stdint.h>

// Opaque handles
typedef struct vesper_index_t vesper_index_t;

typedef enum vesper_status_e {
  VESPER_OK = 0,
  VESPER_ERROR_UNKNOWN = 1,
  VESPER_ERROR_INVALID_PARAM = 2,
  VESPER_ERROR_NOT_TRAINED = 3,
  VESPER_ERROR_IO = 4,
  VESPER_ERROR_INTERNAL = 5
} vesper_status_t;

// Thread-local last error string
VESPER_C_API const char* vesper_get_last_error(void);

// Version info (semantic version string, e.g. "1.1.0")
VESPER_C_API const char* vesper_version(void);

// -------------------------
// IVF-PQ specific API
// -------------------------

typedef struct vesper_ivfpq_train_params_s {
  uint32_t nlist;     // number of coarse clusters
  uint32_t m;         // sub-quantizers; must divide dim
  uint32_t nbits;     // bits per sub-quantizer code (e.g., 8)
} vesper_ivfpq_train_params_t;

typedef struct vesper_ivfpq_search_params_s {
  uint32_t k;         // top-k
  uint32_t nprobe;    // number of lists to probe
} vesper_ivfpq_search_params_t;

// Diagnostics / stats
typedef struct vesper_ivfpq_stats_s {
  size_t n_vectors;
  size_t n_lists;
  size_t m;
  size_t memory_bytes;
  float  avg_list_size;
} vesper_ivfpq_stats_t;

// Lifecycle
VESPER_C_API vesper_status_t vesper_ivfpq_create(vesper_index_t** out_index);
VESPER_C_API vesper_status_t vesper_ivfpq_destroy(vesper_index_t* index);

// Train the index (required before add/search)
VESPER_C_API vesper_status_t vesper_ivfpq_train(
  vesper_index_t* index,
  const float* base_vectors, // size: n * dim
  size_t dim,
  size_t n,
  const vesper_ivfpq_train_params_t* params);

// Add vectors with user-provided 64-bit IDs
VESPER_C_API vesper_status_t vesper_ivfpq_add(
  vesper_index_t* index,
  const uint64_t* ids,       // size: n
  const float* base_vectors, // size: n * dim
  size_t n);

// Search a single query vector
VESPER_C_API vesper_status_t vesper_ivfpq_search(
  const vesper_index_t* index,
  const float* query,             // size: dim
  const vesper_ivfpq_search_params_t* params,
  uint64_t* out_ids,              // size: k
  float* out_distances);          // size: k

// Search a batch of queries (row-major queries: nq x dim)
VESPER_C_API vesper_status_t vesper_ivfpq_search_batch(
  const vesper_index_t* index,
  const float* queries,           // size: nq * dim
  size_t nq,
  const vesper_ivfpq_search_params_t* params,
  uint64_t* out_ids,              // size: nq * k
  float* out_distances);          // size: nq * k

// Diagnostics and state accessors
VESPER_C_API vesper_status_t vesper_ivfpq_is_trained(const vesper_index_t* index, int* out_is_trained);
VESPER_C_API vesper_status_t vesper_ivfpq_get_dimension(const vesper_index_t* index, size_t* out_dim);
VESPER_C_API vesper_status_t vesper_ivfpq_get_stats(const vesper_index_t* index, vesper_ivfpq_stats_t* out_stats);

// JSON Metadata
VESPER_C_API vesper_status_t vesper_ivfpq_set_metadata_json(vesper_index_t* index, const char* json_str);
VESPER_C_API vesper_status_t vesper_ivfpq_get_metadata_json(const vesper_index_t* index,
                                                            char* out_buffer, size_t buffer_size,
                                                            size_t* out_required_size);

// Persistence
VESPER_C_API vesper_status_t vesper_ivfpq_save(
  const vesper_index_t* index,
  const char* file_path);

VESPER_C_API vesper_status_t vesper_ivfpq_load(
  vesper_index_t* index,
  const char* file_path);

#ifdef __cplusplus
} // extern "C"
#endif

