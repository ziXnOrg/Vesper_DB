#pragma once

#include <stddef.h>
#include <stdint.h>

#include "vesper/c/vesper.h"  /* visibility macros, status, ivfpq param structs */

#ifdef __cplusplus
extern "C" {
#endif

/* ---------- Enums ---------- */
typedef enum vesper_index_type_e {
  VESPER_INDEX_HNSW = 0,
  VESPER_INDEX_IVF_PQ = 1,
  VESPER_INDEX_DISKANN = 2
} vesper_index_type_t;

typedef enum vesper_selection_strategy_e {
  VESPER_SELECT_AUTO = 0,
  VESPER_SELECT_MANUAL = 1,
  VESPER_SELECT_HYBRID = 2
} vesper_selection_strategy_t;

/* ---------- Opaque handle ---------- */
typedef struct vesper_manager_t vesper_manager_t;

/* ---------- Config structs ---------- */
typedef struct vesper_hnsw_build_params_s {
  uint32_t M;                /* graph degree */
  uint32_t ef_construction;  /* construction beam */
  uint32_t seed;             /* RNG seed */
} vesper_hnsw_build_params_t;

typedef struct vesper_vamana_build_params_s {
  uint32_t degree;  /* out-degree */
  uint32_t L;       /* candidate set size */
  float    alpha;   /* pruning parameter */
} vesper_vamana_build_params_t;

typedef struct vesper_manager_build_config_s {
  vesper_index_type_t          type;      /* default: VESPER_INDEX_IVF_PQ */
  vesper_selection_strategy_t  strategy;  /* default: VESPER_SELECT_MANUAL */
  vesper_ivfpq_train_params_t  ivf;       /* IVF-PQ training params */
  vesper_hnsw_build_params_t   hnsw;      /* HNSW build params */
  vesper_vamana_build_params_t vamana;    /* DiskANN/Vamana build params */
} vesper_manager_build_config_t;

typedef struct vesper_query_config_s {
  uint32_t k;               /* default 10 */
  float    epsilon;         /* default 0.0f */
  uint32_t ef_search;       /* HNSW */
  uint32_t nprobe;          /* IVF-PQ */
  uint32_t l_search;        /* DiskANN */
  int      use_exact_rerank;        /* boolean */
  uint32_t rerank_k;
  float    rerank_alpha;            /* default 2.0 */
  uint32_t rerank_cand_ceiling;     /* default 2000 */
  int      use_query_planner;       /* boolean; default 1 */
  int      has_preferred_index;     /* boolean */
  vesper_index_type_t preferred_index; /* valid if has_preferred_index==1 */
  /* Metadata filtering via JSON expression (copied per-call; pointer not retained) */
  int      has_filter;              /* boolean */
  const char* filter_json;          /* JSON string; UTF-8; nullable if has_filter==0 */
} vesper_query_config_t;

typedef struct vesper_index_stats_s {
  vesper_index_type_t type;
  size_t  num_vectors;
  size_t  memory_usage_bytes;
  size_t  disk_usage_bytes;
  float   build_time_seconds;
  float   avg_query_time_ms;
  float   measured_recall;
  uint64_t query_count;
} vesper_index_stats_t;

/* ---------- API functions ---------- */
VESPER_C_API vesper_status_t vesper_mgr_create(size_t dim, vesper_manager_t** out_mgr);
VESPER_C_API vesper_status_t vesper_mgr_destroy(vesper_manager_t* mgr);

VESPER_C_API vesper_status_t vesper_mgr_build(vesper_manager_t* mgr,
  const float* base_vectors, size_t n, const vesper_manager_build_config_t* cfg);

VESPER_C_API vesper_status_t vesper_mgr_add_batch(vesper_manager_t* mgr,
  const uint64_t* ids, const float* vectors, size_t n);

/* Phase 2: update/remove */
VESPER_C_API vesper_status_t vesper_mgr_update(vesper_manager_t* mgr, uint64_t id, const float* vector);
VESPER_C_API vesper_status_t vesper_mgr_update_batch(vesper_manager_t* mgr, const uint64_t* ids, const float* vectors, size_t n);
VESPER_C_API vesper_status_t vesper_mgr_remove(vesper_manager_t* mgr, uint64_t id);
VESPER_C_API vesper_status_t vesper_mgr_remove_batch(vesper_manager_t* mgr, const uint64_t* ids, size_t n);

/* Search */
VESPER_C_API vesper_status_t vesper_mgr_search(const vesper_manager_t* mgr,
  const float* query, const vesper_query_config_t* qc,
  uint64_t* out_ids, float* out_dists);

VESPER_C_API vesper_status_t vesper_mgr_search_batch(const vesper_manager_t* mgr,
  const float* queries, size_t nq, const vesper_query_config_t* qc,
  uint64_t* out_ids, float* out_dists);

/* Persistence */
VESPER_C_API vesper_status_t vesper_mgr_save(const vesper_manager_t* mgr, const char* dir_path);
VESPER_C_API vesper_status_t vesper_mgr_load(vesper_manager_t* mgr, const char* dir_path);

/* Resource usage */
VESPER_C_API vesper_status_t vesper_mgr_memory_usage(const vesper_manager_t* mgr, size_t* out_bytes);
VESPER_C_API vesper_status_t vesper_mgr_disk_usage(const vesper_manager_t* mgr, size_t* out_bytes);

/* Memory budget control */
VESPER_C_API vesper_status_t vesper_mgr_set_memory_budget_mb(vesper_manager_t* mgr, uint32_t mb);
VESPER_C_API vesper_status_t vesper_mgr_get_memory_budget_mb(const vesper_manager_t* mgr, uint32_t* out_mb);

/* Two-call pattern: if out_stats==NULL or *inout_capacity==0, returns count via out_count */
VESPER_C_API vesper_status_t vesper_mgr_get_stats(const vesper_manager_t* mgr,
  vesper_index_stats_t* out_stats, size_t* inout_capacity, size_t* out_count);

#ifdef __cplusplus
} /* extern "C" */
#endif

