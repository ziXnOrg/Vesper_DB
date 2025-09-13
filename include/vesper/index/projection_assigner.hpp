#pragma once

#include <cstddef>
#include <cstdint>

namespace vesper {
namespace index {

struct ProjScreenInputs {
    const float* qproj;               // [n x p]
    const float* qnorm;               // [n]
    std::size_t n;                    // number of queries
    std::size_t p;                    // projection dim
    const float* centroids_rm;        // [C x p], row-major projected centroids
    const float* centroid_norms;      // [C]
    const float* centroids_pack8;     // optional packed panels for AVX microkernel (may be null)
    std::size_t C;                    // number of centroids (nlist)
    std::uint32_t L;                  // shortlist size per query
};

struct ProjScreenOutputs {
    std::uint32_t* cand_idx;          // [n x L]
    float* cand_dist;                 // [n x L]
};

// Projection-based screening with fused selection.
// Backends:
//  - If VESPER_HAS_CBLAS: blocked SGEMM + selection
//  - Else: scalar fallback (p-agnostic)
// Notes:
//  - AVX2 microkernel backend (p==16) will be added next; scalar fallback is used for now when CBLAS is unavailable.
//  - Profiling: if env VESPER_PROJ_PROF=1, prints timing breakdown to stderr.
void projection_screen_select(const ProjScreenInputs& in, ProjScreenOutputs& out);

} // namespace index
} // namespace vesper

