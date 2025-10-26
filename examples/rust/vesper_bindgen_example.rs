// Rust bindgen example for Vesper C API (IVF-PQ and IndexManager)
//
// Quick setup (Cargo project):
//   cargo new vesper_bindings_demo && cd vesper_bindings_demo
//   # Add to Cargo.toml:
//   # [dependencies]
//   # libc = "0.2"
//   #
//   # [build-dependencies]
//   # bindgen = "0.69"
//   #
//   # build = "build.rs"
//   // Create build.rs with:
//   /*
//   use std::env;
//   use std::path::PathBuf;
//   fn main() {
//       println!("cargo:rustc-link-lib=vesper_c");
//       // Add library search path if needed, e.g. build/Release on Windows
//       if let Ok(p) = env::var("VESPER_LIB_DIR") { println!("cargo:rustc-link-search=native={}", p); }
//       let bindings = bindgen::Builder::default()
//           .header("../include/vesper/c/vesper.h")
//           .header("../include/vesper/c/vesper_manager.h")
//           .allowlist_item("vesper_.*")
//           .clang_arg(format!("-I{}", "../include"))
//           .generate()
//           .expect("bindgen failed");
//       let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
//       bindings
//           .write_to_file(out_path.join("vesper_bindings.rs"))
//           .expect("Couldn't write bindings!");
//   }
//   */
//
// Runtime notes:
//   - Ensure vesper_c shared library is discoverable by the OS loader:
//     * Windows: copy vesper_c.dll next to target exe or set PATH / VESPER_LIB_DIR
//     * Linux: set LD_LIBRARY_PATH or install to system lib dir
//     * macOS: set DYLD_LIBRARY_PATH
//
// IndexManager Phase 2 additions (not fully demonstrated here):
//   - Update/remove: vesper_mgr_update, vesper_mgr_update_batch, vesper_mgr_remove, vesper_mgr_remove_batch
//   - Query-time filtering: set qc.has_filter=1 and qc.filter_json to a filter expression
//     Note: JSON parsing may be disabled in this build, returning an error.
//   - Memory budget control: vesper_mgr_set_memory_budget_mb / vesper_mgr_get_memory_budget_mb

use libc::{c_char, c_float, c_int, c_uint, c_ulonglong, size_t};
use std::ffi::CString;
use std::ptr;

// Include bindgen output
include!(concat!(env!("OUT_DIR"), "/vesper_bindings.rs"));

unsafe fn die_on_error(st: vesper_status_t, where_: &str) {
    if st != VESPER_status_t_VESPER_OK {
        let msg = vesper_get_last_error();
        let s = if msg.is_null() { "".to_string() } else { std::ffi::CStr::from_ptr(msg).to_string_lossy().into_owned() };
        panic!("{} failed: {:?} => {}", where_, st, s);
    }
}

fn make_data(n: usize, dim: usize) -> (Vec<f32>, Vec<u64>) {
    let mut base = vec![0.0f32; n * dim];
    let mut ids = vec![0u64; n];
    for i in 0..n { ids[i] = i as u64; for d in 0..dim { base[i*dim + d] = ((i+1)*(d+1)) as f32 / (dim as f32); } }
    (base, ids)
}

fn main() {
    println!("Vesper version: {}", unsafe { std::ffi::CStr::from_ptr(unsafe { vesper_version() }).to_string_lossy() });
    unsafe {
        // === IVF-PQ ===
        let dim: usize = 32; let n: usize = 100;
        let (base, ids) = make_data(n, dim);
        let mut idx: *mut vesper_index_t = ptr::null_mut();
        die_on_error(vesper_ivfpq_create(&mut idx), "ivfpq_create");
        let mut tp = vesper_ivfpq_train_params_t { nlist: 16, m: 8, nbits: 8 };
        die_on_error(vesper_ivfpq_train(idx, base.as_ptr(), dim as size_t, n as size_t, &mut tp), "ivfpq_train");
        die_on_error(vesper_ivfpq_add(idx, ids.as_ptr(), base.as_ptr(), n as size_t), "ivfpq_add");
        let mut sp = vesper_ivfpq_search_params_t { k: 5, nprobe: 4 };
        let mut out_ids = vec![0u64; sp.k as usize];
        let mut out_d = vec![0f32; sp.k as usize];
        die_on_error(vesper_ivfpq_search(idx, base.as_ptr(), &mut sp, out_ids.as_mut_ptr(), out_d.as_mut_ptr()), "ivfpq_search");
        println!("IVFPQ k=5 first: id={} dist={}", out_ids[0], out_d[0]);
        // Round-trip
        let path = CString::new("ivfpq_rust_demo.ivfpq").unwrap();
        die_on_error(vesper_ivfpq_save(idx, path.as_ptr()), "ivfpq_save");
        die_on_error(vesper_ivfpq_load(idx, path.as_ptr()), "ivfpq_load");
        die_on_error(vesper_ivfpq_search(idx, base.as_ptr(), &mut sp, out_ids.as_mut_ptr(), out_d.as_mut_ptr()), "ivfpq_search(after_load)");
        die_on_error(vesper_ivfpq_destroy(idx), "ivfpq_destroy");

        // === IndexManager ===
        let mut mgr: *mut vesper_manager_t = ptr::null_mut();
        die_on_error(vesper_mgr_create(dim as size_t, &mut mgr), "mgr_create");
        let mut cfg = vesper_manager_build_config_t {
            type_: vesper_index_type_t_VESPER_INDEX_IVF_PQ,
            strategy: vesper_selection_strategy_t_VESPER_SELECT_MANUAL,
            ivf: vesper_ivfpq_train_params_t { nlist: 16, m: 8, nbits: 8 },
        };
        die_on_error(vesper_mgr_build(mgr, base.as_ptr(), n as size_t, &mut cfg), "mgr_build");
        die_on_error(vesper_mgr_add_batch(mgr, ids.as_ptr(), base.as_ptr(), n as size_t), "mgr_add_batch");
        let mut qc = vesper_query_config_t {
            k: 5, nprobe: 4, ef_search: 100, l_search: 128, epsilon: 0.0,
            use_exact_rerank: 0, rerank_k: 0, rerank_alpha: 2.0, rerank_cand_ceiling: 2000,
            use_query_planner: 0, has_preferred_index: 1, preferred_index: vesper_index_type_t_VESPER_INDEX_IVF_PQ,
        };
        let mut ids_out = vec![0u64; qc.k as usize];
        let mut d_out = vec![0f32; qc.k as usize];
        die_on_error(vesper_mgr_search(mgr, base.as_ptr(), &mut qc, ids_out.as_mut_ptr(), d_out.as_mut_ptr()), "mgr_search");
        println!("Mgr k=5 first: id={} dist={}", ids_out[0], d_out[0]);
        // Stats two-call
        let mut count: size_t = 0;
        die_on_error(vesper_mgr_get_stats(mgr, ptr::null_mut(), ptr::null_mut(), &mut count), "mgr_get_stats(size)");
        let mut stats = vec![vesper_index_stats_t { type_: 0, num_vectors: 0, memory_usage_bytes: 0, disk_usage_bytes: 0, avg_query_time_ms: 0.0, measured_recall: 0.0, query_count: 0 }; count as usize];
        let mut cap = count;
        die_on_error(vesper_mgr_get_stats(mgr, stats.as_mut_ptr(), &mut cap, &mut count), "mgr_get_stats(copy)");
        println!("stats_count={} first_type={}", count, if count>0 { stats[0].type_ } else { -1 });
        // Round-trip
        let dir = CString::new("vesper_mgr_rust_demo").unwrap();
        die_on_error(vesper_mgr_save(mgr, dir.as_ptr()), "mgr_save");
        die_on_error(vesper_mgr_load(mgr, dir.as_ptr()), "mgr_load");
        die_on_error(vesper_mgr_search(mgr, base.as_ptr(), &mut qc, ids_out.as_mut_ptr(), d_out.as_mut_ptr()), "mgr_search(after_load)");
        die_on_error(vesper_mgr_destroy(mgr), "mgr_destroy");
    }
}

