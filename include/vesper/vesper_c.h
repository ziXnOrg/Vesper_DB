#ifndef VESPER_C_H
#define VESPER_C_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

// Versioning and stability
// - VESPER_C_ABI_VERSION increments on incompatible changes.
// - All structs are POD with explicit sizes; callers manage memory.
#define VESPER_C_ABI_VERSION 1

// Error codes (subset of C++ error_code)
typedef enum {
  VESPER_OK = 0,
  VESPER_E_IO_FAILED = 1001,
  VESPER_E_CONFIG_INVALID = 2001,
  VESPER_E_DATA_INTEGRITY = 3001,
  VESPER_E_PRECONDITION_FAILED = 4001,
  VESPER_E_RESOURCE_EXHAUSTED = 5001,
  VESPER_E_NOT_FOUND = 6001,
  VESPER_E_UNAVAILABLE = 7001,
  VESPER_E_CANCELLED = 8001,
  VESPER_E_INTERNAL = 9001
} vesper_status_t;

// Opaque handles
// - Ownership: returned pointers belong to the caller; use close() to release.
typedef struct vesper_collection_t_ vesper_collection_t;

// Search params and results are caller-owned POD structures.
typedef struct {
  const char* metric; // "l2" | "ip" | "cosine"; pointer must remain valid during call
  uint32_t k;
  float target_recall;
  uint32_t nprobe;
  uint32_t ef_search;
  uint32_t rerank;
} vesper_search_params_t;

typedef struct {
  uint64_t id;
  float score;
} vesper_search_result_t;

// API ownership & lifetime rules
// - All input buffers (e.g., vectors, query) are borrowed and must remain valid
//   for the duration of the call.
// - Output arrays are provided by the caller (out/out_cap); the function writes up to out_cap
//   results and sets out_size. If out_cap is insufficient, return VESPER_E_PRECONDITION_FAILED.
// - Functions are thread-safe for reads; mutating functions require a single writer per collection.

// API
vesper_status_t vesper_open_collection(const char* path, vesper_collection_t** out);
vesper_status_t vesper_close_collection(vesper_collection_t* c);

vesper_status_t vesper_insert(vesper_collection_t* c, uint64_t id,
                              const float* vec, size_t dim);
vesper_status_t vesper_remove(vesper_collection_t* c, uint64_t id);

vesper_status_t vesper_search(vesper_collection_t* c, const float* query, size_t dim,
                              const vesper_search_params_t* p,
                              vesper_search_result_t* out, size_t out_cap,
                              size_t* out_size);

vesper_status_t vesper_seal_segment(vesper_collection_t* c);
vesper_status_t vesper_compact(vesper_collection_t* c);
vesper_status_t vesper_snapshot(vesper_collection_t* c);
vesper_status_t vesper_recover(vesper_collection_t* c);

#ifdef __cplusplus
}
#endif

#endif // VESPER_C_H

