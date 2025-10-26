#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#include "vesper/c/vesper.h"
#include "vesper/c/vesper_manager.h"

static void die_on_error(vesper_status_t st, const char* where) {
  if (st != VESPER_OK) {
    const char* err = vesper_get_last_error();
    fprintf(stderr, "[ERROR] %s failed: status=%d, last_error=\"%s\"\n", where, (int)st, err ? err : "");
    exit(1);
  }
}

int main(void) {
  printf("Vesper C API version: %s\n", vesper_version());

  const size_t dim = 32;
  const size_t n = 100;
  float* base = (float*)malloc(sizeof(float) * n * dim);
  uint64_t* ids = (uint64_t*)malloc(sizeof(uint64_t) * n);
  if (!base || !ids) {
    fprintf(stderr, "allocation failure\n");
    return 1;
  }
  for (size_t i = 0; i < n; ++i) {
    ids[i] = (uint64_t)i;
    for (size_t d = 0; d < dim; ++d) {
      base[i*dim + d] = (float)((i + 1) * (d + 1)) / (float)(dim);
    }
  }

  vesper_manager_t* mgr = NULL;
  die_on_error(vesper_mgr_create(dim, &mgr), "vesper_mgr_create");

  vesper_manager_build_config_t cfg;
  cfg.type = VESPER_INDEX_IVF_PQ;
  cfg.strategy = VESPER_SELECT_MANUAL;
  cfg.ivf.nlist = 16; cfg.ivf.m = 8; cfg.ivf.nbits = 8;
  /* optional other index params (unused here) */
  cfg.hnsw.M = 16; cfg.hnsw.ef_construction = 200; cfg.hnsw.seed = 42;
  cfg.vamana.degree = 64; cfg.vamana.L = 128; cfg.vamana.alpha = 1.2f;
  die_on_error(vesper_mgr_build(mgr, base, n, &cfg), "vesper_mgr_build");

  die_on_error(vesper_mgr_add_batch(mgr, ids, base, n), "vesper_mgr_add_batch");

  vesper_query_config_t qc;
  memset(&qc, 0, sizeof(qc));
  qc.k = 5; qc.nprobe = 4; qc.ef_search = 100; qc.l_search = 128; qc.epsilon = 0.0f;
  qc.use_exact_rerank = 0; qc.rerank_k = 0; qc.rerank_alpha = 2.0f; qc.rerank_cand_ceiling = 2000;
  qc.use_query_planner = 0; qc.has_preferred_index = 1; qc.preferred_index = VESPER_INDEX_IVF_PQ;

  uint64_t ids_out[5]; float dists_out[5];
  memset(ids_out, 0, sizeof(ids_out)); memset(dists_out, 0, sizeof(dists_out));
  die_on_error(vesper_mgr_search(mgr, &base[0], &qc, ids_out, dists_out), "vesper_mgr_search");
  printf("IndexManager single search (k=5) results:\n");
  for (int i = 0; i < 5; ++i) {
    printf("  #%d: id=%" PRIu64 " dist=%f\n", i, ids_out[i], dists_out[i]);
  }

  // Demonstrate filter_json usage (may be unavailable if JSON parsing is disabled)
  vesper_query_config_t qc_filter = qc;
  qc_filter.has_filter = 1;
  qc_filter.filter_json = "{\"term\":{\"field\":\"category\",\"value\":\"A\"}}";
  memset(ids_out, 0, sizeof(ids_out)); memset(dists_out, 0, sizeof(dists_out));
  vesper_status_t st_filter = vesper_mgr_search(mgr, &base[0], &qc_filter, ids_out, dists_out);
  if (st_filter == VESPER_OK) {
    printf("Filtered search succeeded (k=5): first id=%" PRIu64 "\n", ids_out[0]);
  } else {
    printf("Filtered search not available in this build: %s\n", vesper_get_last_error());
  }

  // Batch search first 10
  const size_t nq = 10;
  uint64_t* ids_b = (uint64_t*)malloc(sizeof(uint64_t) * nq * qc.k);
  float* dists_b = (float*)malloc(sizeof(float) * nq * qc.k);
  if (!ids_b || !dists_b) { fprintf(stderr, "allocation failure\n"); return 1; }
  memset(ids_b, 0, sizeof(uint64_t) * nq * qc.k);
  memset(dists_b, 0, sizeof(float) * nq * qc.k);
  die_on_error(vesper_mgr_search_batch(mgr, base, nq, &qc, ids_b, dists_b), "vesper_mgr_search_batch");
  printf("IndexManager batch search (nq=10, k=5) first row:\n");
  for (int i = 0; i < 5; ++i) {
    printf("  #%d: id=%" PRIu64 " dist=%f\n", i, ids_b[i], dists_b[i]);
  }

  // Stats two-call pattern
  size_t count = 0;
  die_on_error(vesper_mgr_get_stats(mgr, NULL, NULL, &count), "vesper_mgr_get_stats(size)");
  vesper_index_stats_t* stats = (vesper_index_stats_t*)malloc(sizeof(vesper_index_stats_t) * count);
  if (!stats) { fprintf(stderr, "allocation failure\n"); return 1; }
  size_t cap = count;
  die_on_error(vesper_mgr_get_stats(mgr, stats, &cap, &count), "vesper_mgr_get_stats(copy)");
  printf("IndexManager stats (count=%zu):\n", count);
  for (size_t i = 0; i < count; ++i) {
    printf("  type=%d num_vectors=%zu mem=%zu disk=%zu avg_q_ms=%.3f recall=%.3f queries=%" PRIu64 "\n",
      (int)stats[i].type, stats[i].num_vectors, stats[i].memory_usage_bytes,
      stats[i].disk_usage_bytes, stats[i].avg_query_time_ms, stats[i].measured_recall,
      stats[i].query_count);
  }

  // Save, destroy, reload, search again
  const char* dir = "vesper_mgr_example";
  die_on_error(vesper_mgr_save(mgr, dir), "vesper_mgr_save");
  die_on_error(vesper_mgr_destroy(mgr), "vesper_mgr_destroy");
  mgr = NULL;
  die_on_error(vesper_mgr_create(dim, &mgr), "vesper_mgr_create(reload)");
  die_on_error(vesper_mgr_load(mgr, dir), "vesper_mgr_load");
  memset(ids_out, 0, sizeof(ids_out)); memset(dists_out, 0, sizeof(dists_out));
  die_on_error(vesper_mgr_search(mgr, &base[0], &qc, ids_out, dists_out), "vesper_mgr_search(after_load)");
  printf("after load (k=5):\n");
  for (int i = 0; i < 5; ++i) {
    printf("  #%d: id=%" PRIu64 " dist=%f\n", i, ids_out[i], dists_out[i]);
  }

  // Demonstrate update/remove
  float updated_vec[32];
  for (size_t d = 0; d < dim; ++d) updated_vec[d] = 0.5f; /* constant vector */
  die_on_error(vesper_mgr_update(mgr, 0, updated_vec), "vesper_mgr_update(id=0)");
  die_on_error(vesper_mgr_remove(mgr, 1), "vesper_mgr_remove(id=1)");
  memset(ids_out, 0, sizeof(ids_out)); memset(dists_out, 0, sizeof(dists_out));
  die_on_error(vesper_mgr_search(mgr, &base[0], &qc, ids_out, dists_out), "vesper_mgr_search(after_update_remove)");
  printf("after update/remove (k=5): first id=%" PRIu64 "\n", ids_out[0]);

  // Memory budget control (set and get)
  die_on_error(vesper_mgr_set_memory_budget_mb(mgr, 64), "vesper_mgr_set_memory_budget_mb(64)");
  uint32_t budget_mb = 0;
  die_on_error(vesper_mgr_get_memory_budget_mb(mgr, &budget_mb), "vesper_mgr_get_memory_budget_mb");
  printf("memory budget now: %u MB\n", budget_mb);

  // Memory/disk usage
  size_t mem=0, disk=0;
  die_on_error(vesper_mgr_memory_usage(mgr, &mem), "vesper_mgr_memory_usage");
  die_on_error(vesper_mgr_disk_usage(mgr, &disk), "vesper_mgr_disk_usage");
  printf("resource usage: mem=%zu bytes disk=%zu bytes\n", mem, disk);

  // Cleanup
  die_on_error(vesper_mgr_destroy(mgr), "vesper_mgr_destroy(final)");
  free(stats);
  free(ids_b);
  free(dists_b);
  free(base);
  free(ids);

  printf("IndexManager C API example completed successfully.\n");
  return 0;
}

