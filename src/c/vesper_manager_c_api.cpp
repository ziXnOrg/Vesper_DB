#include "vesper/c/vesper_manager.h"

#include <new>
#include <memory>
#include <vector>
#include <cstring>

#include "vesper/index/index_manager.hpp"
#include "vesper/metadata/metadata_store.hpp"
#include "vesper_c_error.hpp"

using vesper_c::set_error;
using vesper_c::clear_error;

using vesper::index::IndexManager;
using vesper::index::IndexBuildConfig;
using vesper::index::QueryConfig;
using vesper::index::IndexType;
using vesper::index::SelectionStrategy;

struct vesper_manager_t {
  std::unique_ptr<IndexManager> mgr;
};

extern "C" {

VESPER_C_API vesper_status_t vesper_mgr_create(size_t dim, vesper_manager_t** out_mgr) {
  if (!out_mgr || dim == 0) return VESPER_ERROR_INVALID_PARAM;
  clear_error();
  try {
    auto h = std::make_unique<vesper_manager_t>();
    h->mgr = std::make_unique<IndexManager>(dim);
    *out_mgr = h.release();
    return VESPER_OK;
  } catch (const std::bad_alloc&) {
    set_error("allocation failure");
    return VESPER_ERROR_INTERNAL;
  } catch (const std::exception& e) {
    set_error(e.what());
    return VESPER_ERROR_INTERNAL;
  } catch (...) {
    set_error("unknown error in vesper_mgr_create");
    return VESPER_ERROR_UNKNOWN;
  }
}

VESPER_C_API vesper_status_t vesper_mgr_destroy(vesper_manager_t* mgr) {
  clear_error();
  try {
    delete mgr;
    return VESPER_OK;
  } catch (...) {
    set_error("unknown error in vesper_mgr_destroy");
    return VESPER_ERROR_UNKNOWN;
  }
}

static inline IndexType to_index_type(vesper_index_type_t t) {
  switch (t) {
    case VESPER_INDEX_HNSW: return IndexType::HNSW;
    case VESPER_INDEX_IVF_PQ: return IndexType::IVF_PQ;
    case VESPER_INDEX_DISKANN: return IndexType::DiskANN;
    default: return IndexType::IVF_PQ;
  }
}

static inline SelectionStrategy to_strategy(vesper_selection_strategy_t s) {
  switch (s) {
    case VESPER_SELECT_AUTO: return SelectionStrategy::Auto;
    case VESPER_SELECT_MANUAL: return SelectionStrategy::Manual;
    case VESPER_SELECT_HYBRID: return SelectionStrategy::Hybrid;
    default: return SelectionStrategy::Manual;
  }
}

VESPER_C_API vesper_status_t vesper_mgr_build(vesper_manager_t* mgr,
  const float* base_vectors, size_t n, const vesper_manager_build_config_t* cfg)
{
  if (!mgr || !mgr->mgr || !base_vectors || n == 0 || !cfg) return VESPER_ERROR_INVALID_PARAM;
  clear_error();
  try {
    IndexBuildConfig bc;
    bc.type = to_index_type(cfg->type);
    bc.strategy = to_strategy(cfg->strategy);
    // IVF-PQ mapping
    bc.ivf_params.nlist = cfg->ivf.nlist;
    bc.ivf_params.m = cfg->ivf.m;
    bc.ivf_params.nbits = cfg->ivf.nbits;
    // HNSW mapping
    bc.hnsw_params.M = cfg->hnsw.M;
    bc.hnsw_params.efConstruction = cfg->hnsw.ef_construction;
    bc.hnsw_params.seed = cfg->hnsw.seed;
    // Vamana (DiskANN) mapping
    bc.vamana_params.degree = cfg->vamana.degree;
    bc.vamana_params.L = cfg->vamana.L;
    bc.vamana_params.alpha = cfg->vamana.alpha;
    auto r = mgr->mgr->build(base_vectors, n, bc);
    if (!r) {
      set_error(r.error().message);
      return VESPER_ERROR_INVALID_PARAM;
    }
    return VESPER_OK;
  } catch (const std::exception& e) {
    set_error(e.what());
    return VESPER_ERROR_INTERNAL;
  } catch (...) {
    set_error("unknown error in vesper_mgr_build");
    return VESPER_ERROR_UNKNOWN;
  }
}

VESPER_C_API vesper_status_t vesper_mgr_add_batch(vesper_manager_t* mgr,
  const uint64_t* ids, const float* vectors, size_t n)
{
  if (!mgr || !mgr->mgr || !ids || !vectors || n == 0) return VESPER_ERROR_INVALID_PARAM;
  clear_error();
  try {
    auto r = mgr->mgr->add_batch(ids, vectors, n);
    if (!r) {
      set_error(r.error().message);
      return VESPER_ERROR_INVALID_PARAM;
    }
    return VESPER_OK;
  } catch (const std::exception& e) {
    set_error(e.what());
    return VESPER_ERROR_INTERNAL;
  } catch (...) {
    set_error("unknown error in vesper_mgr_add_batch");
    return VESPER_ERROR_UNKNOWN;
  }
}

VESPER_C_API vesper_status_t vesper_mgr_update(vesper_manager_t* mgr, uint64_t id, const float* vector) {
  if (!mgr || !mgr->mgr || !vector) return VESPER_ERROR_INVALID_PARAM;
  clear_error();
  try {
    auto r = mgr->mgr->update(id, vector);
    if (!r) { set_error(r.error().message); return VESPER_ERROR_INVALID_PARAM; }
    return VESPER_OK;
  } catch (const std::exception& e) { set_error(e.what()); return VESPER_ERROR_INTERNAL; }
  catch (...) { set_error("unknown error in vesper_mgr_update"); return VESPER_ERROR_UNKNOWN; }
}

VESPER_C_API vesper_status_t vesper_mgr_update_batch(vesper_manager_t* mgr, const uint64_t* ids, const float* vectors, size_t n) {
  if (!mgr || !mgr->mgr || !ids || !vectors || n == 0) return VESPER_ERROR_INVALID_PARAM;
  clear_error();
  try {
    auto r = mgr->mgr->update_batch(ids, vectors, n);
    if (!r) { set_error(r.error().message); return VESPER_ERROR_INVALID_PARAM; }
    return VESPER_OK;
  } catch (const std::exception& e) { set_error(e.what()); return VESPER_ERROR_INTERNAL; }
  catch (...) { set_error("unknown error in vesper_mgr_update_batch"); return VESPER_ERROR_UNKNOWN; }
}

VESPER_C_API vesper_status_t vesper_mgr_remove(vesper_manager_t* mgr, uint64_t id) {
  if (!mgr || !mgr->mgr) return VESPER_ERROR_INVALID_PARAM;
  clear_error();
  try {
    auto r = mgr->mgr->remove(id);
    if (!r) { set_error(r.error().message); return VESPER_ERROR_INVALID_PARAM; }
    return VESPER_OK;
  } catch (const std::exception& e) { set_error(e.what()); return VESPER_ERROR_INTERNAL; }
  catch (...) { set_error("unknown error in vesper_mgr_remove"); return VESPER_ERROR_UNKNOWN; }
}

VESPER_C_API vesper_status_t vesper_mgr_remove_batch(vesper_manager_t* mgr, const uint64_t* ids, size_t n) {
  if (!mgr || !mgr->mgr || !ids || n == 0) return VESPER_ERROR_INVALID_PARAM;
  clear_error();
  try {
    for (size_t i = 0; i < n; ++i) {
      auto r = mgr->mgr->remove(ids[i]);
      if (!r) { set_error(r.error().message); return VESPER_ERROR_INVALID_PARAM; }
    }
    return VESPER_OK;
  } catch (const std::exception& e) { set_error(e.what()); return VESPER_ERROR_INTERNAL; }
  catch (...) { set_error("unknown error in vesper_mgr_remove_batch"); return VESPER_ERROR_UNKNOWN; }
}

static inline bool to_query_config(const vesper_query_config_t* qc, QueryConfig& c) {
  c = QueryConfig{};
  c.k = qc->k;
  c.epsilon = qc->epsilon;
  c.ef_search = qc->ef_search;
  c.nprobe = qc->nprobe;
  c.l_search = qc->l_search;
  c.use_exact_rerank = qc->use_exact_rerank != 0;
  c.rerank_k = qc->rerank_k;
  c.rerank_alpha = qc->rerank_alpha;
  c.rerank_cand_ceiling = qc->rerank_cand_ceiling;
  c.use_query_planner = qc->use_query_planner != 0;
  if (qc->has_preferred_index) {
    c.preferred_index = to_index_type(qc->preferred_index);
  } else {
    c.preferred_index.reset();
  }
  // Metadata filter via JSON (parsed per-call; string not retained)
  if (qc->has_filter) {
    if (!qc->filter_json) {
      set_error("has_filter=1 but filter_json is NULL");
      return false;
    }
    auto parsed = vesper::metadata::utils::parse_filter_json(std::string(qc->filter_json));
    if (!parsed) {
      set_error(parsed.error().message);
      return false;
    }
    c.filter = std::move(*parsed);
  }
  return true;
}

VESPER_C_API vesper_status_t vesper_mgr_search(const vesper_manager_t* mgr,
  const float* query, const vesper_query_config_t* qc,
  uint64_t* out_ids, float* out_dists)
{
  if (!mgr || !mgr->mgr || !query || !qc || !out_ids || !out_dists) return VESPER_ERROR_INVALID_PARAM;
  clear_error();
  try {
    QueryConfig c;
    if (!to_query_config(qc, c)) {
      return VESPER_ERROR_INVALID_PARAM;
    }
    auto r = mgr->mgr->search(query, c);
    if (!r) {
      set_error(r.error().message);
      return VESPER_ERROR_INVALID_PARAM;
    }
    size_t k = qc->k;
    size_t take = r->size() < k ? r->size() : k;
    for (size_t i = 0; i < take; ++i) {
      out_ids[i] = (*r)[i].first;
      out_dists[i] = (*r)[i].second;
    }
    for (size_t i = take; i < k; ++i) {
      out_ids[i] = 0;
      out_dists[i] = (take > 0) ? (*r)[take-1].second : 0.0f;
    }
    return VESPER_OK;
  } catch (const std::exception& e) {
    set_error(e.what());
    return VESPER_ERROR_INTERNAL;
  } catch (...) {
    set_error("unknown error in vesper_mgr_search");
    return VESPER_ERROR_UNKNOWN;
  }
}

VESPER_C_API vesper_status_t vesper_mgr_search_batch(const vesper_manager_t* mgr,
  const float* queries, size_t nq, const vesper_query_config_t* qc,
  uint64_t* out_ids, float* out_dists)
{
  if (!mgr || !mgr->mgr || !queries || nq == 0 || !qc || !out_ids || !out_dists)
    return VESPER_ERROR_INVALID_PARAM;
  clear_error();
  try {
    QueryConfig c;
    if (!to_query_config(qc, c)) {
      return VESPER_ERROR_INVALID_PARAM;
    }
    auto r = mgr->mgr->search_batch(queries, nq, c);
    if (!r) {
      set_error(r.error().message);
      return VESPER_ERROR_INVALID_PARAM;
    }
    size_t k = qc->k;
    for (size_t qi = 0; qi < nq; ++qi) {
      const auto& row = (*r)[qi];
      size_t take = row.size() < k ? row.size() : k;
      for (size_t i = 0; i < take; ++i) {
        out_ids[qi*k + i] = row[i].first;
        out_dists[qi*k + i] = row[i].second;
      }
      for (size_t i = take; i < k; ++i) {
        out_ids[qi*k + i] = 0;
        out_dists[qi*k + i] = (take > 0) ? row[take-1].second : 0.0f;
      }
    }
    return VESPER_OK;
  } catch (const std::exception& e) {
    set_error(e.what());
    return VESPER_ERROR_INTERNAL;
  } catch (...) {
    set_error("unknown error in vesper_mgr_search_batch");
    return VESPER_ERROR_UNKNOWN;
  }
}

VESPER_C_API vesper_status_t vesper_mgr_save(const vesper_manager_t* mgr, const char* dir_path) {
  if (!mgr || !mgr->mgr || !dir_path) return VESPER_ERROR_INVALID_PARAM;
  clear_error();
  try {
    auto r = mgr->mgr->save(std::string(dir_path));
    if (!r) { set_error(r.error().message); return VESPER_ERROR_IO; }
    return VESPER_OK;
  } catch (const std::exception& e) {
    set_error(e.what());
    return VESPER_ERROR_INTERNAL;
  } catch (...) {
    set_error("unknown error in vesper_mgr_save");
    return VESPER_ERROR_UNKNOWN;
  }
}

VESPER_C_API vesper_status_t vesper_mgr_load(vesper_manager_t* mgr, const char* dir_path) {
  if (!mgr || !mgr->mgr || !dir_path) return VESPER_ERROR_INVALID_PARAM;
  clear_error();
  try {
    auto r = mgr->mgr->load(std::string(dir_path));
    if (!r) { set_error(r.error().message); return VESPER_ERROR_IO; }
    return VESPER_OK;
  } catch (const std::exception& e) {
    set_error(e.what());
    return VESPER_ERROR_INTERNAL;
  } catch (...) {
    set_error("unknown error in vesper_mgr_load");
    return VESPER_ERROR_UNKNOWN;
  }
}

VESPER_C_API vesper_status_t vesper_mgr_memory_usage(const vesper_manager_t* mgr, size_t* out_bytes) {
  if (!mgr || !mgr->mgr || !out_bytes) return VESPER_ERROR_INVALID_PARAM;
  clear_error();
  try {
    *out_bytes = mgr->mgr->memory_usage();
    return VESPER_OK;
  } catch (const std::exception& e) {
    set_error(e.what());
    return VESPER_ERROR_INTERNAL;
  } catch (...) {
    set_error("unknown error in vesper_mgr_memory_usage");
    return VESPER_ERROR_UNKNOWN;
  }
}

VESPER_C_API vesper_status_t vesper_mgr_disk_usage(const vesper_manager_t* mgr, size_t* out_bytes) {
  if (!mgr || !mgr->mgr || !out_bytes) return VESPER_ERROR_INVALID_PARAM;
  clear_error();
  try {
    *out_bytes = mgr->mgr->disk_usage();
    return VESPER_OK;
  } catch (const std::exception& e) {
    set_error(e.what());
    return VESPER_ERROR_INTERNAL;
  } catch (...) {
    set_error("unknown error in vesper_mgr_disk_usage");
    return VESPER_ERROR_UNKNOWN;
  }
}

VESPER_C_API vesper_status_t vesper_mgr_set_memory_budget_mb(vesper_manager_t* mgr, uint32_t mb) {
  if (!mgr || !mgr->mgr) return VESPER_ERROR_INVALID_PARAM;
  clear_error();
  try {
    auto r = mgr->mgr->set_memory_budget(static_cast<std::size_t>(mb));
    if (!r) { set_error(r.error().message); return VESPER_ERROR_INVALID_PARAM; }
    return VESPER_OK;
  } catch (const std::exception& e) { set_error(e.what()); return VESPER_ERROR_INTERNAL; }
  catch (...) { set_error("unknown error in vesper_mgr_set_memory_budget_mb"); return VESPER_ERROR_UNKNOWN; }
}

VESPER_C_API vesper_status_t vesper_mgr_get_memory_budget_mb(const vesper_manager_t* mgr, uint32_t* out_mb) {
  if (!mgr || !mgr->mgr || !out_mb) return VESPER_ERROR_INVALID_PARAM;
  clear_error();
  try {
    *out_mb = static_cast<uint32_t>(mgr->mgr->get_memory_budget());
    return VESPER_OK;
  } catch (const std::exception& e) { set_error(e.what()); return VESPER_ERROR_INTERNAL; }
  catch (...) { set_error("unknown error in vesper_mgr_get_memory_budget_mb"); return VESPER_ERROR_UNKNOWN; }
}

VESPER_C_API vesper_status_t vesper_mgr_get_stats(const vesper_manager_t* mgr,
  vesper_index_stats_t* out_stats, size_t* inout_capacity, size_t* out_count)
{
  if (!mgr || !mgr->mgr || !out_count) return VESPER_ERROR_INVALID_PARAM;
  clear_error();
  try {
    auto v = mgr->mgr->get_stats();
    *out_count = v.size();
    if (!out_stats) {
      return VESPER_OK; // size query only
    }
    if (!inout_capacity) {
      set_error("inout_capacity must be provided when out_stats is non-NULL");
      return VESPER_ERROR_INVALID_PARAM;
    }
    size_t cap = *inout_capacity;
    size_t ncopy = (cap < v.size()) ? cap : v.size();
    for (size_t i = 0; i < ncopy; ++i) {
      out_stats[i].type = (v[i].type == IndexType::HNSW) ? VESPER_INDEX_HNSW
                        : (v[i].type == IndexType::IVF_PQ) ? VESPER_INDEX_IVF_PQ
                        : VESPER_INDEX_DISKANN;
      out_stats[i].num_vectors = v[i].num_vectors;
      out_stats[i].memory_usage_bytes = v[i].memory_usage_bytes;
      out_stats[i].disk_usage_bytes = v[i].disk_usage_bytes;
      out_stats[i].build_time_seconds = v[i].build_time_seconds;
      out_stats[i].avg_query_time_ms = v[i].avg_query_time_ms;
      out_stats[i].measured_recall = v[i].measured_recall;
      out_stats[i].query_count = v[i].query_count;
    }
    return VESPER_OK;
  } catch (const std::exception& e) {
    set_error(e.what());
    return VESPER_ERROR_INTERNAL;
  } catch (...) {
    set_error("unknown error in vesper_mgr_get_stats");
    return VESPER_ERROR_UNKNOWN;
  }
}

} // extern "C"

