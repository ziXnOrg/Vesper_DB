#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#include "vesper/c/vesper.h"

static void die_on_error(vesper_status_t st, const char* where) {
  if (st != VESPER_OK) {
    const char* err = vesper_get_last_error();
    fprintf(stderr, "[ERROR] %s failed: status=%d, last_error=\"%s\"\n", where, (int)st, err ? err : "");
    exit(1);
  }
}

int main(void) {
  printf("Vesper C API version: %s\n", vesper_version());

  // Synthetic data: 100 vectors, dim=32
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
      // Simple pattern with some variation
      base[i*dim + d] = (float)((i + 1) * (d + 1)) / (float)(dim);
    }
  }

  // Create IVF-PQ index
  vesper_index_t* index = NULL;
  die_on_error(vesper_ivfpq_create(&index), "ivfpq_create");

  // Diagnostics: is_trained (should be 0)
  int is_trained = -1;
  die_on_error(vesper_ivfpq_is_trained(index, &is_trained), "ivfpq_is_trained");
  printf("is_trained (before): %d\n", is_trained);

  // Train parameters: nlist=16, m=8 (divides 32), nbits=8
  vesper_ivfpq_train_params_t tp; tp.nlist = 16; tp.m = 8; tp.nbits = 8;
  die_on_error(vesper_ivfpq_train(index, base, dim, n, &tp), "ivfpq_train");

  // Diagnostics: dimension, is_trained (should be 1)
  size_t out_dim = 0;
  die_on_error(vesper_ivfpq_get_dimension(index, &out_dim), "ivfpq_get_dimension");
  die_on_error(vesper_ivfpq_is_trained(index, &is_trained), "ivfpq_is_trained");
  printf("dimension=%zu, is_trained (after): %d\n", out_dim, is_trained);

  // Add all vectors
  die_on_error(vesper_ivfpq_add(index, ids, base, n), "ivfpq_add");

  // Stats
  vesper_ivfpq_stats_t stats;
  memset(&stats, 0, sizeof(stats));
  die_on_error(vesper_ivfpq_get_stats(index, &stats), "ivfpq_get_stats");
  printf("stats: n_vectors=%zu n_lists=%zu m=%zu memory_bytes=%zu avg_list_size=%.3f\n",
         stats.n_vectors, stats.n_lists, stats.m, stats.memory_bytes, stats.avg_list_size);

  // Single search: query = base[0]
  vesper_ivfpq_search_params_t sp; sp.k = 5; sp.nprobe = 4;
  uint64_t out_ids[5]; float out_dists[5];
  memset(out_ids, 0, sizeof(out_ids)); memset(out_dists, 0, sizeof(out_dists));
  die_on_error(vesper_ivfpq_search(index, &base[0], &sp, out_ids, out_dists), "ivfpq_search");
  printf("single search (k=5) results:\n");
  for (int i = 0; i < 5; ++i) {
    printf("  #%d: id=%" PRIu64 " dist=%f\n", i, out_ids[i], out_dists[i]);
  }

  // Batch search: queries = base[0..9]
  const size_t nq = 10; // first 10 as queries
  uint64_t* out_ids_b = (uint64_t*)malloc(sizeof(uint64_t) * nq * sp.k);
  float* out_dists_b = (float*)malloc(sizeof(float) * nq * sp.k);
  if (!out_ids_b || !out_dists_b) { fprintf(stderr, "allocation failure\n"); return 1; }
  memset(out_ids_b, 0, sizeof(uint64_t) * nq * sp.k);
  memset(out_dists_b, 0, sizeof(float) * nq * sp.k);
  die_on_error(vesper_ivfpq_search_batch(index, base, nq, &sp, out_ids_b, out_dists_b), "ivfpq_search_batch");
  printf("batch search (nq=10, k=5) first row:\n");
  for (int i = 0; i < 5; ++i) {
    printf("  #%d: id=%" PRIu64 " dist=%f\n", i, out_ids_b[i], out_dists_b[i]);
  }

  // Metadata: set and get with two-call pattern
  const char* meta_json = "{\"source\":\"vesper_c_example\",\"note\":\"hello\"}";
  die_on_error(vesper_ivfpq_set_metadata_json(index, meta_json), "ivfpq_set_metadata_json");
  size_t needed = 0;
  die_on_error(vesper_ivfpq_get_metadata_json(index, NULL, 0, &needed), "ivfpq_get_metadata_json(size)");
  char* buffer = (char*)malloc(needed);
  if (!buffer) { fprintf(stderr, "allocation failure\n"); return 1; }
  die_on_error(vesper_ivfpq_get_metadata_json(index, buffer, needed, NULL), "ivfpq_get_metadata_json(copy)");
  printf("metadata json (len=%zu): %s\n", needed ? (needed - 1) : 0, buffer);

  // Persist, destroy, reload, search again
  const char* path = "vesper_c_example.ivfpq";
  die_on_error(vesper_ivfpq_save(index, path), "ivfpq_save");
  die_on_error(vesper_ivfpq_destroy(index), "ivfpq_destroy");
  index = NULL;
  die_on_error(vesper_ivfpq_create(&index), "ivfpq_create(reload)");
  die_on_error(vesper_ivfpq_load(index, path), "ivfpq_load");
  memset(out_ids, 0, sizeof(out_ids)); memset(out_dists, 0, sizeof(out_dists));
  die_on_error(vesper_ivfpq_search(index, &base[0], &sp, out_ids, out_dists), "ivfpq_search(after_load)");
  printf("single search after load:\n");
  for (int i = 0; i < 5; ++i) {
    printf("  #%d: id=%" PRIu64 " dist=%f\n", i, out_ids[i], out_dists[i]);
  }

  // Cleanup
  die_on_error(vesper_ivfpq_destroy(index), "ivfpq_destroy(final)");
  free(buffer);
  free(out_ids_b);
  free(out_dists_b);
  free(base);
  free(ids);

  printf("C API example completed successfully.\n");
  return 0;
}

