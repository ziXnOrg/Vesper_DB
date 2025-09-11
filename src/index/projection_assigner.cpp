#include "vesper/index/projection_assigner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <vector>

#ifdef VESPER_HAS_CBLAS
#include <cblas.h>
#endif

namespace vesper {
namespace index {

namespace {
struct Cand { float dist; std::uint32_t idx; };
static inline bool cand_cmp(const Cand& a, const Cand& b) { return a.dist < b.dist; }
}

void projection_screen_select(const ProjScreenInputs& in, ProjScreenOutputs& out) {
    using clock = std::chrono::steady_clock;
    const bool do_prof = [](){ const char* v = std::getenv("VESPER_PROJ_PROF"); return v && v[0]=='1'; }();
    auto t0 = clock::now();

    const std::size_t n = in.n;
    const std::size_t C = in.C;
    const std::size_t p = in.p;
    const std::uint32_t L = std::max<std::uint32_t>(1u, std::min<std::uint32_t>(in.L, static_cast<std::uint32_t>(C)));

    // Per-row heap (max-heap) state implemented over output buffers
    std::vector<std::size_t> heap_size(n, 0);
    auto row_base = [&](std::size_t i){ return i * static_cast<std::size_t>(L); };

    auto heap_begin = [&](std::size_t i){ return reinterpret_cast<Cand*>(out.cand_dist) + row_base(i); };
    auto heap_end   = [&](std::size_t i){ return reinterpret_cast<Cand*>(out.cand_dist) + row_base(i) + heap_size[i]; };

    // We maintain parallel arrays by writing Cand into cand_dist buffer then copying idx/dist to outputs.
    // Simpler: keep a separate temporary heap storage and at the end scatter.
    std::vector<Cand> heap_storage(n * static_cast<std::size_t>(L), Cand{INFINITY, 0});

#ifdef VESPER_HAS_CBLAS
    auto t_gemm0 = clock::now();
    const std::size_t QB = 256, CB = 256;
    std::vector<float> dots(QB * CB);
    for (std::size_t j0 = 0; j0 < C; j0 += CB) {
        const std::size_t jb = std::min<std::size_t>(CB, C - j0);
        for (std::size_t i0 = 0; i0 < n; i0 += QB) {
            const std::size_t qb = std::min<std::size_t>(QB, n - i0);
            // C = A * B^T where A = qproj[i0: i0+qb, :], B = centroids_rm[j0: j0+jb, :]
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        static_cast<int>(qb), static_cast<int>(jb), static_cast<int>(p),
                        1.0f,
                        in.qproj + i0 * p, static_cast<int>(p),
                        in.centroids_rm + j0 * p, static_cast<int>(p),
                        0.0f,
                        dots.data(), static_cast<int>(jb));
            // Selection epilogue per row
            for (std::size_t r = 0; r < qb; ++r) {
                const std::size_t i = i0 + r;
                std::size_t& hs = heap_size[i];
                const float qi = in.qnorm[i];
                auto* base = heap_storage.data() + row_base(i);
                for (std::size_t jj = 0; jj < jb; ++jj) {
                    const std::size_t cj = j0 + jj;
                    const float dot = dots[r * jb + jj];
                    const float distp = qi + in.centroid_norms[cj] - 2.0f * dot;
                    if (hs < L) {
                        base[hs++] = Cand{distp, static_cast<std::uint32_t>(cj)};
                        if (hs == L) std::make_heap(base, base + hs, cand_cmp);
                    } else if (distp < base[0].dist) {
                        std::pop_heap(base, base + hs, cand_cmp);
                        base[hs - 1] = Cand{distp, static_cast<std::uint32_t>(cj)};
                        std::push_heap(base, base + hs, cand_cmp);
                    }
                }
            }
        }
    }
    auto t_gemm1 = clock::now();
    auto gemm_ms = std::chrono::duration<double, std::milli>(t_gemm1 - t_gemm0).count();
#else
    auto t_sel0 = clock::now();
    // Scalar fallback
    for (std::size_t j0 = 0; j0 < C; ++j0) {
        const float* yc = in.centroids_rm + j0 * p;
        const float cn = in.centroid_norms[j0];
        for (std::size_t i = 0; i < n; ++i) {
            const float* qp = in.qproj + i * p;
            float dot = 0.0f; for (std::size_t k = 0; k < p; ++k) dot += qp[k] * yc[k];
            const float distp = in.qnorm[i] + cn - 2.0f * dot;
            std::size_t& hs = heap_size[i];
            auto* base = heap_storage.data() + row_base(i);
            if (hs < L) {
                base[hs++] = Cand{distp, static_cast<std::uint32_t>(j0)};
                if (hs == L) std::make_heap(base, base + hs, cand_cmp);
            } else if (distp < base[0].dist) {
                std::pop_heap(base, base + hs, cand_cmp);
                base[hs - 1] = Cand{distp, static_cast<std::uint32_t>(j0)};
                std::push_heap(base, base + hs, cand_cmp);
            }
        }
    }
    auto t_sel1 = clock::now();
    auto gemm_ms = std::chrono::duration<double, std::milli>(t_sel1 - t_sel0).count();
#endif

    // Scatter to output arrays in arbitrary order (heap content is not sorted ascending)
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t base = row_base(i);
        const std::size_t hs = heap_size[i];
        for (std::size_t t = 0; t < L; ++t) {
            const std::size_t src = (t < hs) ? (base + t) : base; // pad if needed
            out.cand_idx[base + t]  = heap_storage[src].idx;
            out.cand_dist[base + t] = heap_storage[src].dist;
        }
    }

    auto t1 = clock::now();
    if (do_prof) {
        auto total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        // Note: gemm_ms equals total selection time in scalar path
        fprintf(stderr, "[proj-prof] select_total_ms=%.3f gemm_or_scalar_ms=%.3f n=%zu C=%zu p=%zu L=%u\n",
                total_ms, gemm_ms, n, C, p, L);
    }
}

} // namespace index
} // namespace vesper

