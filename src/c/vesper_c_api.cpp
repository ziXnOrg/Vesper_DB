#include "vesper/c/vesper.h"

#include <string>
#include <string_view>
#include <new>
#include <memory>
#include <mutex>

#include "vesper/index/ivf_pq.hpp"

using vesper::index::IvfPqIndex;
using vesper::index::IvfPqTrainParams;
using vesper::index::IvfPqSearchParams;

struct vesper_index_t {
  std::unique_ptr<IvfPqIndex> ivfpq;
};

#include "vesper_c_error.hpp"

thread_local std::string vesper_c::g_last_error;
using vesper_c::set_error;
using vesper_c::clear_error;

extern "C" {

VESPER_C_API const char* vesper_get_last_error(void) {
  return vesper_c::g_last_error.empty() ? "" : vesper_c::g_last_error.c_str();
}

VESPER_C_API const char* vesper_version(void) {
  return "dev";
}

// Phase 1: Diagnostics and state accessors
vesper_status_t vesper_ivfpq_is_trained(const vesper_index_t* index, int* out_is_trained) {
  if (!index || !index->ivfpq || !out_is_trained) return VESPER_ERROR_INVALID_PARAM;
  clear_error();
  try {
    *out_is_trained = index->ivfpq->is_trained() ? 1 : 0;
    return VESPER_OK;
  } catch (const std::exception& e) {
    set_error(e.what());
    return VESPER_ERROR_INTERNAL;
  } catch (...) {
    set_error("unknown error in is_trained");
    return VESPER_ERROR_UNKNOWN;
  }
}

vesper_status_t vesper_ivfpq_get_dimension(const vesper_index_t* index, size_t* out_dim) {
  if (!index || !index->ivfpq || !out_dim) return VESPER_ERROR_INVALID_PARAM;
  clear_error();
  try {
    *out_dim = index->ivfpq->dimension();
    return VESPER_OK;
  } catch (const std::exception& e) {
    set_error(e.what());
    return VESPER_ERROR_INTERNAL;
  } catch (...) {
    set_error("unknown error in get_dimension");
    return VESPER_ERROR_UNKNOWN;
  }
}

vesper_status_t vesper_ivfpq_get_stats(const vesper_index_t* index, vesper_ivfpq_stats_t* out_stats) {
  if (!index || !index->ivfpq || !out_stats) return VESPER_ERROR_INVALID_PARAM;
  clear_error();
  try {
    auto st = index->ivfpq->get_stats();
    out_stats->n_vectors = st.n_vectors;
    out_stats->n_lists = st.n_lists;
    out_stats->m = st.m;
    out_stats->memory_bytes = st.memory_bytes;
    out_stats->avg_list_size = st.avg_list_size;
    return VESPER_OK;
  } catch (const std::exception& e) {
    set_error(e.what());
    return VESPER_ERROR_INTERNAL;
  } catch (...) {
    set_error("unknown error in get_stats");
    return VESPER_ERROR_UNKNOWN;
  }
}

// Phase 2: JSON metadata
vesper_status_t vesper_ivfpq_set_metadata_json(vesper_index_t* index, const char* json_str) {
  if (!index || !index->ivfpq || !json_str) return VESPER_ERROR_INVALID_PARAM;
  clear_error();
  try {
    index->ivfpq->set_metadata_json(std::string_view(json_str));
    return VESPER_OK;
  } catch (const std::exception& e) {
    set_error(e.what());
    return VESPER_ERROR_INTERNAL;
  } catch (...) {
    set_error("unknown error in set_metadata_json");
    return VESPER_ERROR_UNKNOWN;
  }
}

vesper_status_t vesper_ivfpq_get_metadata_json(
  const vesper_index_t* index,
  char* out_buffer,
  size_t buffer_size,
  size_t* out_required_size) {
  if (!index || !index->ivfpq) return VESPER_ERROR_INVALID_PARAM;
  clear_error();
  try {
    std::string s = index->ivfpq->get_metadata_json();
    size_t required = s.size() + 1; // include NUL
    if (out_required_size) *out_required_size = required;

    if (!out_buffer || buffer_size == 0) {
      // Size query only
      return VESPER_OK;
    }
    if (buffer_size < required) {
      set_error("buffer too small for metadata json");
      return VESPER_ERROR_INVALID_PARAM;
    }
    // Copy including NUL terminator
    std::memcpy(out_buffer, s.c_str(), required);
    return VESPER_OK;
  } catch (const std::exception& e) {
    set_error(e.what());
    return VESPER_ERROR_INTERNAL;
  } catch (...) {
    set_error("unknown error in get_metadata_json");
    return VESPER_ERROR_UNKNOWN;
  }
}


// -------------------------
// IVF-PQ
// -------------------------

VESPER_C_API vesper_status_t vesper_ivfpq_create(vesper_index_t** out_index) {
  if (!out_index) return VESPER_ERROR_INVALID_PARAM;
  clear_error();
  try {
    auto h = std::make_unique<vesper_index_t>();
    h->ivfpq = std::make_unique<IvfPqIndex>();
    *out_index = h.release();
    return VESPER_OK;
  } catch (const std::bad_alloc&) {
    set_error("allocation failure");
    return VESPER_ERROR_INTERNAL;
  } catch (const std::exception& e) {
    set_error(e.what());
    return VESPER_ERROR_INTERNAL;
  } catch (...) {
    set_error("unknown error in create");
    return VESPER_ERROR_UNKNOWN;
  }
}

VESPER_C_API vesper_status_t vesper_ivfpq_destroy(vesper_index_t* index) {
  clear_error();
  try {
    delete index;
    return VESPER_OK;
  } catch (...) {
    set_error("unknown error in destroy");
    return VESPER_ERROR_UNKNOWN;
  }
}

VESPER_C_API vesper_status_t vesper_ivfpq_train(
  vesper_index_t* index,
  const float* base_vectors,
  size_t dim,
  size_t n,
  const vesper_ivfpq_train_params_t* params)
{
  if (!index || !index->ivfpq || !base_vectors || !params || dim == 0 || n == 0)
    return VESPER_ERROR_INVALID_PARAM;
  clear_error();
  try {
    IvfPqTrainParams tp;
    tp.nlist = params->nlist;
    tp.m = params->m;
    tp.nbits = params->nbits;
    tp.timings_enabled = false;
    auto r = index->ivfpq->train(base_vectors, dim, n, tp);
    if (!r) {
      set_error(r.error().message);
      return VESPER_ERROR_INVALID_PARAM;
    }
    return VESPER_OK;
  } catch (const std::exception& e) {
    set_error(e.what());
    return VESPER_ERROR_INTERNAL;
  } catch (...) {
    set_error("unknown error in train");
    return VESPER_ERROR_UNKNOWN;
  }
}

VESPER_C_API vesper_status_t vesper_ivfpq_add(
  vesper_index_t* index,
  const uint64_t* ids,
  const float* base_vectors,
  size_t n)
{
  if (!index || !index->ivfpq || !ids || !base_vectors || n == 0)
    return VESPER_ERROR_INVALID_PARAM;
  clear_error();
  try {
    auto r = index->ivfpq->add(ids, base_vectors, n);
    if (!r) {
      set_error(r.error().message);
      return VESPER_ERROR_INVALID_PARAM;
    }
    return VESPER_OK;
  } catch (const std::exception& e) {
    set_error(e.what());
    return VESPER_ERROR_INTERNAL;
  } catch (...) {
    set_error("unknown error in add");
    return VESPER_ERROR_UNKNOWN;
  }
}

VESPER_C_API vesper_status_t vesper_ivfpq_search(
  const vesper_index_t* index,
  const float* query,
  const vesper_ivfpq_search_params_t* params,
  uint64_t* out_ids,
  float* out_distances)
{
  if (!index || !index->ivfpq || !query || !params || !out_ids || !out_distances)
    return VESPER_ERROR_INVALID_PARAM;
  clear_error();
  try {
    IvfPqSearchParams sp; sp.k = params->k; sp.nprobe = params->nprobe;
    auto r = index->ivfpq->search(query, sp);
    if (!r) {
      set_error(r.error().message);
      return VESPER_ERROR_INVALID_PARAM;
    }
    // Fill outputs (pad if fewer than k)
    size_t k = params->k;
    size_t take = r->size() < k ? r->size() : k;
    for (size_t i = 0; i < take; ++i) {
      out_ids[i] = (*r)[i].first;
      out_distances[i] = (*r)[i].second;
    }
    for (size_t i = take; i < k; ++i) {
      out_ids[i] = 0;
      out_distances[i] = (take > 0) ? (*r)[take - 1].second : 0.0f;
    }
    return VESPER_OK;
  } catch (const std::exception& e) {
    set_error(e.what());
    return VESPER_ERROR_INTERNAL;
  } catch (...) {
    set_error("unknown error in search");
    return VESPER_ERROR_UNKNOWN;
  }
}

VESPER_C_API vesper_status_t vesper_ivfpq_search_batch(
  const vesper_index_t* index,
  const float* queries,
  size_t nq,
  const vesper_ivfpq_search_params_t* params,
  uint64_t* out_ids,
  float* out_distances)
{
  if (!index || !index->ivfpq || !queries || !params || !out_ids || !out_distances || nq == 0)
    return VESPER_ERROR_INVALID_PARAM;
  clear_error();
  try {
    IvfPqSearchParams sp; sp.k = params->k; sp.nprobe = params->nprobe;
    auto r = index->ivfpq->search_batch(queries, nq, sp);
    if (!r) {
      set_error(r.error().message);
      return VESPER_ERROR_INVALID_PARAM;
    }
    size_t k = params->k;
    for (size_t qi = 0; qi < nq; ++qi) {
      const auto& vec = (*r)[qi];
      size_t take = vec.size() < k ? vec.size() : k;
      for (size_t i = 0; i < take; ++i) {
        out_ids[qi*k + i] = vec[i].first;
        out_distances[qi*k + i] = vec[i].second;
      }
      for (size_t i = take; i < k; ++i) {
        out_ids[qi*k + i] = 0;
        out_distances[qi*k + i] = (take > 0) ? vec[take - 1].second : 0.0f;
      }
    }
    return VESPER_OK;
  } catch (const std::exception& e) {
    set_error(e.what());
    return VESPER_ERROR_INTERNAL;
  } catch (...) {
    set_error("unknown error in search_batch");
    return VESPER_ERROR_UNKNOWN;
  }
}

VESPER_C_API vesper_status_t vesper_ivfpq_save(
  const vesper_index_t* index,
  const char* file_path)
{
  if (!index || !index->ivfpq || !file_path)
    return VESPER_ERROR_INVALID_PARAM;
  clear_error();
  try {
    auto r = index->ivfpq->save(std::string(file_path));
    if (!r) {
      set_error(r.error().message);
      return VESPER_ERROR_IO;
    }
    return VESPER_OK;
  } catch (const std::exception& e) {
    set_error(e.what());
    return VESPER_ERROR_INTERNAL;
  } catch (...) {
    set_error("unknown error in save");
    return VESPER_ERROR_UNKNOWN;
  }
}

VESPER_C_API vesper_status_t vesper_ivfpq_load(
  vesper_index_t* index,
  const char* file_path)
{
  if (!index || !file_path)
    return VESPER_ERROR_INVALID_PARAM;
  clear_error();
  try {
    auto r = IvfPqIndex::load(std::string(file_path));
    if (!r) {
      set_error(r.error().message);
      return VESPER_ERROR_IO;
    }
    index->ivfpq = std::make_unique<IvfPqIndex>(std::move(*r));
    return VESPER_OK;
  } catch (const std::exception& e) {
    set_error(e.what());
    return VESPER_ERROR_INTERNAL;
  } catch (...) {
    set_error("unknown error in load");
    return VESPER_ERROR_UNKNOWN;
  }
}

} // extern "C"

