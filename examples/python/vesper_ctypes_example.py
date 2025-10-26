# Python ctypes example for Vesper C API (IVF-PQ and IndexManager)
#
# How to run (Windows):
#   1) Build the repo (produces build/Release/vesper_c.dll)
#   2) Ensure vesper_c.dll is on PATH or set VESPER_C_DLL to its full path
#   3) python examples/python/vesper_ctypes_example.py
#
# How to run (Linux/macOS):
#   1) Build the repo (libvesper_c.so / libvesper_c.dylib)
#   2) export LD_LIBRARY_PATH/DYLD_LIBRARY_PATH accordingly or set VESPER_C_DLL
#   3) python examples/python/vesper_ctypes_example.py

import os
import sys
import ctypes as C
from ctypes import c_uint32, c_uint64, c_size_t, c_float, c_char_p, c_int, c_void_p

# Load shared library
LIB_ENV = os.environ.get("VESPER_C_DLL")
if LIB_ENV and os.path.exists(LIB_ENV):
    lib = C.CDLL(LIB_ENV)
else:
    # Try common names; rely on loader path/working dir
    names = [
        "vesper_c.dll",           # Windows
        "libvesper_c.so",         # Linux
        "libvesper_c.dylib",      # macOS
    ]
    last_err = None
    lib = None
    for n in names:
        try:
            lib = C.CDLL(n)
            break
        except OSError as e:
            last_err = e
    if lib is None:
        raise OSError(f"Failed to load Vesper C library. Set VESPER_C_DLL. Last error: {last_err}")

# Status codes
VESPER_OK = 0

# Common
lib.vesper_version.restype = c_char_p
lib.vesper_get_last_error.restype = c_char_p

def die_on_error(status: int, where: str):
    if status != VESPER_OK:
        msg = lib.vesper_get_last_error()
        raise RuntimeError(f"{where} failed: status={status}, last_error={msg.decode('utf-8') if msg else ''}")

# ---------------- IVF-PQ API ----------------
class vesper_index_t(C.Structure):
    pass  # opaque

class VesperIvfPqTrainParams(C.Structure):
    _fields_ = [("nlist", c_uint32), ("m", c_uint32), ("nbits", c_uint32)]

class VesperIvfPqSearchParams(C.Structure):
    _fields_ = [("k", c_uint32), ("nprobe", c_uint32)]

class VesperIvfPqStats(C.Structure):
    _fields_ = [
        ("n_vectors", c_size_t), ("n_lists", c_size_t), ("m", c_size_t),
        ("memory_bytes", c_size_t), ("avg_list_size", c_float),
    ]

# Prototypes
lib.vesper_ivfpq_create.argtypes = [C.POINTER(C.POINTER(vesper_index_t))]
lib.vesper_ivfpq_create.restype = c_int
lib.vesper_ivfpq_destroy.argtypes = [C.POINTER(vesper_index_t)]
lib.vesper_ivfpq_destroy.restype = c_int
lib.vesper_ivfpq_train.argtypes = [C.POINTER(vesper_index_t), C.POINTER(c_float), c_size_t, c_size_t, C.POINTER(VesperIvfPqTrainParams)]
lib.vesper_ivfpq_train.restype = c_int
lib.vesper_ivfpq_add.argtypes = [C.POINTER(vesper_index_t), C.POINTER(c_uint64), C.POINTER(c_float), c_size_t]
lib.vesper_ivfpq_add.restype = c_int
lib.vesper_ivfpq_search.argtypes = [C.POINTER(vesper_index_t), C.POINTER(c_float), C.POINTER(VesperIvfPqSearchParams), C.POINTER(c_uint64), C.POINTER(c_float)]
lib.vesper_ivfpq_search.restype = c_int
lib.vesper_ivfpq_search_batch.argtypes = [C.POINTER(vesper_index_t), C.POINTER(c_float), c_size_t, C.POINTER(VesperIvfPqSearchParams), C.POINTER(c_uint64), C.POINTER(c_float)]
lib.vesper_ivfpq_search_batch.restype = c_int
lib.vesper_ivfpq_get_stats.argtypes = [C.POINTER(vesper_index_t), C.POINTER(VesperIvfPqStats)]
lib.vesper_ivfpq_get_stats.restype = c_int
lib.vesper_ivfpq_set_metadata_json.argtypes = [C.POINTER(vesper_index_t), c_char_p]
lib.vesper_ivfpq_set_metadata_json.restype = c_int
lib.vesper_ivfpq_get_metadata_json.argtypes = [C.POINTER(vesper_index_t), C.c_char_p, c_size_t, C.POINTER(c_size_t)]
lib.vesper_ivfpq_get_metadata_json.restype = c_int
lib.vesper_ivfpq_save.argtypes = [C.POINTER(vesper_index_t), c_char_p]
lib.vesper_ivfpq_save.restype = c_int
lib.vesper_ivfpq_load.argtypes = [C.POINTER(vesper_index_t), c_char_p]
lib.vesper_ivfpq_load.restype = c_int

# ---------------- IndexManager API ----------------
# Phase 2 additions (not fully demonstrated here):
# - Update/remove operations: vesper_mgr_update, vesper_mgr_update_batch, vesper_mgr_remove, vesper_mgr_remove_batch
# - Query-time filtering via qc.has_filter + qc.filter_json (may be unavailable if JSON parsing is disabled)
# - Memory budget control: vesper_mgr_set_memory_budget_mb / vesper_mgr_get_memory_budget_mb
class vesper_manager_t(C.Structure):
    pass  # opaque

# enums
VESPER_INDEX_HNSW = 0
VESPER_INDEX_IVF_PQ = 1
VESPER_INDEX_DISKANN = 2
VESPER_SELECT_AUTO = 0
VESPER_SELECT_MANUAL = 1
VESPER_SELECT_HYBRID = 2

class VesperManagerBuildConfig(C.Structure):
    _fields_ = [
        ("type", c_int), ("strategy", c_int), ("ivf", VesperIvfPqTrainParams)
    ]

class VesperQueryConfig(C.Structure):
    _fields_ = [
        ("k", c_uint32), ("nprobe", c_uint32), ("ef_search", c_uint32), ("l_search", c_uint32), ("epsilon", c_float),
        ("use_exact_rerank", c_int), ("rerank_k", c_uint32), ("rerank_alpha", c_float), ("rerank_cand_ceiling", c_uint32),
        ("use_query_planner", c_int), ("has_preferred_index", c_int), ("preferred_index", c_int)
    ]

class VesperIndexStats(C.Structure):
    _fields_ = [
        ("type", c_int), ("num_vectors", c_size_t), ("memory_usage_bytes", c_size_t), ("disk_usage_bytes", c_size_t),
        ("avg_query_time_ms", c_float), ("measured_recall", c_float), ("query_count", c_uint64)
    ]

lib.vesper_mgr_create.argtypes = [c_size_t, C.POINTER(C.POINTER(vesper_manager_t))]
lib.vesper_mgr_create.restype = c_int
lib.vesper_mgr_destroy.argtypes = [C.POINTER(vesper_manager_t)]
lib.vesper_mgr_destroy.restype = c_int
lib.vesper_mgr_build.argtypes = [C.POINTER(vesper_manager_t), C.POINTER(c_float), c_size_t, C.POINTER(VesperManagerBuildConfig)]
lib.vesper_mgr_build.restype = c_int
lib.vesper_mgr_add_batch.argtypes = [C.POINTER(vesper_manager_t), C.POINTER(c_uint64), C.POINTER(c_float), c_size_t]
lib.vesper_mgr_add_batch.restype = c_int
lib.vesper_mgr_search.argtypes = [C.POINTER(vesper_manager_t), C.POINTER(c_float), C.POINTER(VesperQueryConfig), C.POINTER(c_uint64), C.POINTER(c_float)]
lib.vesper_mgr_search.restype = c_int
lib.vesper_mgr_search_batch.argtypes = [C.POINTER(vesper_manager_t), C.POINTER(c_float), c_size_t, C.POINTER(VesperQueryConfig), C.POINTER(c_uint64), C.POINTER(c_float)]
lib.vesper_mgr_search_batch.restype = c_int
lib.vesper_mgr_save.argtypes = [C.POINTER(vesper_manager_t), c_char_p]
lib.vesper_mgr_save.restype = c_int
lib.vesper_mgr_load.argtypes = [C.POINTER(vesper_manager_t), c_char_p]
lib.vesper_mgr_load.restype = c_int
lib.vesper_mgr_get_stats.argtypes = [C.POINTER(vesper_manager_t), C.POINTER(VesperIndexStats), C.POINTER(c_size_t), C.POINTER(c_size_t)]
lib.vesper_mgr_get_stats.restype = c_int
lib.vesper_mgr_memory_usage.argtypes = [C.POINTER(vesper_manager_t), C.POINTER(c_size_t)]
lib.vesper_mgr_memory_usage.restype = c_int
lib.vesper_mgr_disk_usage.argtypes = [C.POINTER(vesper_manager_t), C.POINTER(c_size_t)]
lib.vesper_mgr_disk_usage.restype = c_int


def make_synth_data(n: int, dim: int):
    import array
    base = array.array('f', [0.0] * (n * dim))
    ids = array.array('Q', [0] * n)
    for i in range(n):
        ids[i] = i
        for d in range(dim):
            base[i * dim + d] = float((i + 1) * (d + 1)) / float(dim)
    return base, ids


def run_ivfpq_demo():
    print("\n=== IVF-PQ via ctypes ===")
    dim = 32
    n = 100
    base, ids = make_synth_data(n, dim)
    base_ptr = (c_float * (n * dim))(*base)
    ids_ptr = (c_uint64 * n)(*ids)

    idx = C.POINTER(vesper_index_t)()
    die_on_error(lib.vesper_ivfpq_create(C.byref(idx)), "ivfpq_create")

    tp = VesperIvfPqTrainParams(16, 8, 8)
    die_on_error(lib.vesper_ivfpq_train(idx, base_ptr, c_size_t(dim), c_size_t(n), C.byref(tp)), "ivfpq_train")
    die_on_error(lib.vesper_ivfpq_add(idx, ids_ptr, base_ptr, c_size_t(n)), "ivfpq_add")

    sp = VesperIvfPqSearchParams(5, 4)
    out_ids = (c_uint64 * sp.k)()
    out_dists = (c_float * sp.k)()
    die_on_error(lib.vesper_ivfpq_search(idx, C.byref(base_ptr, 0), C.byref(sp), out_ids, out_dists), "ivfpq_search")
    print("top-5:")
    for i in range(sp.k):
        print(f"  {i}: id={out_ids[i]} dist={out_dists[i]:.6f}")

    # Metadata two-call
    meta_json = b'{"demo":"python-ctypes"}'
    die_on_error(lib.vesper_ivfpq_set_metadata_json(idx, meta_json), "ivfpq_set_metadata_json")
    need = c_size_t(0)
    die_on_error(lib.vesper_ivfpq_get_metadata_json(idx, None, 0, C.byref(need)), "ivfpq_get_metadata_json(size)")
    buf = (C.c_char * need.value)()
    die_on_error(lib.vesper_ivfpq_get_metadata_json(idx, buf, need.value, None), "ivfpq_get_metadata_json(copy)")
    print("metadata:", C.string_at(buf).decode('utf-8'))

    # Save/load round-trip
    path = b"ivfpq_py_demo.ivfpq"
    die_on_error(lib.vesper_ivfpq_save(idx, path), "ivfpq_save")
    die_on_error(lib.vesper_ivfpq_load(idx, path), "ivfpq_load")
    die_on_error(lib.vesper_ivfpq_search(idx, C.byref(base_ptr, 0), C.byref(sp), out_ids, out_dists), "ivfpq_search(after_load)")

    # Stats
    stats = VesperIvfPqStats()
    die_on_error(lib.vesper_ivfpq_get_stats(idx, C.byref(stats)), "ivfpq_get_stats")
    print(f"n={stats.n_vectors} lists={stats.n_lists} m={stats.m} mem={stats.memory_bytes}")

    die_on_error(lib.vesper_ivfpq_destroy(idx), "ivfpq_destroy")


def run_manager_demo():
    print("\n=== IndexManager via ctypes ===")
    dim = 32
    n = 100
    base, ids = make_synth_data(n, dim)
    base_ptr = (c_float * (n * dim))(*base)
    ids_ptr = (c_uint64 * n)(*ids)

    mgr = C.POINTER(vesper_manager_t)()
    die_on_error(lib.vesper_mgr_create(c_size_t(dim), C.byref(mgr)), "mgr_create")

    cfg = VesperManagerBuildConfig(type=VESPER_INDEX_IVF_PQ, strategy=VESPER_SELECT_MANUAL, ivf=VesperIvfPqTrainParams(16,8,8))
    die_on_error(lib.vesper_mgr_build(mgr, base_ptr, c_size_t(n), C.byref(cfg)), "mgr_build")
    die_on_error(lib.vesper_mgr_add_batch(mgr, ids_ptr, base_ptr, c_size_t(n)), "mgr_add_batch")

    qc = VesperQueryConfig()
    qc.k = 5; qc.nprobe = 4; qc.ef_search = 100; qc.l_search = 128; qc.epsilon = 0.0
    qc.use_exact_rerank = 0; qc.rerank_k = 0; qc.rerank_alpha = 2.0; qc.rerank_cand_ceiling = 2000
    qc.use_query_planner = 0; qc.has_preferred_index = 1; qc.preferred_index = VESPER_INDEX_IVF_PQ

    out_ids = (c_uint64 * qc.k)()
    out_dists = (c_float * qc.k)()
    die_on_error(lib.vesper_mgr_search(mgr, C.byref(base_ptr, 0), C.byref(qc), out_ids, out_dists), "mgr_search")
    print("top-5:")
    for i in range(qc.k):
        print(f"  {i}: id={out_ids[i]} dist={out_dists[i]:.6f}")

    # Stats two-call
    count = c_size_t(0)
    die_on_error(lib.vesper_mgr_get_stats(mgr, None, None, C.byref(count)), "mgr_get_stats(size)")
    stats_arr = (VesperIndexStats * count.value)()
    cap = c_size_t(count.value)
    die_on_error(lib.vesper_mgr_get_stats(mgr, stats_arr, C.byref(cap), C.byref(count)), "mgr_get_stats(copy)")
    print("stats count=", count.value)
    for i in range(count.value):
        s = stats_arr[i]
        print(f"  type={s.type} n={s.num_vectors} mem={s.memory_usage_bytes} disk={s.disk_usage_bytes}")

    # Round-trip
    d = b"vesper_mgr_py_demo"
    die_on_error(lib.vesper_mgr_save(mgr, d), "mgr_save")
    die_on_error(lib.vesper_mgr_load(mgr, d), "mgr_load")
    die_on_error(lib.vesper_mgr_search(mgr, C.byref(base_ptr, 0), C.byref(qc), out_ids, out_dists), "mgr_search(after_load)")

    mem = c_size_t(0); disk = c_size_t(0)
    die_on_error(lib.vesper_mgr_memory_usage(mgr, C.byref(mem)), "mgr_memory_usage")
    die_on_error(lib.vesper_mgr_disk_usage(mgr, C.byref(disk)), "mgr_disk_usage")
    print(f"usage: mem={mem.value} disk={disk.value}")

    die_on_error(lib.vesper_mgr_destroy(mgr), "mgr_destroy")


def main():
    print("Vesper version:", lib.vesper_version().decode("utf-8"))
    run_ivfpq_demo()
    run_manager_demo()


if __name__ == "__main__":
    main()

