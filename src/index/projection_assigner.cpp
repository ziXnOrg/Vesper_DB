#include "vesper/index/projection_assigner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include "vesper/core/platform_utils.hpp"
#include <numeric>
#include <limits>
#include <vector>

#ifdef VESPER_HAS_CBLAS
#include <cblas.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace vesper {
namespace index {

namespace {
struct Cand { float dist; std::uint32_t idx; };
static inline bool cand_cmp(const Cand& a, const Cand& b) { return a.dist < b.dist; }
}

// Fast argmax over a float array; returns index and value. Scalar fallback; can be SIMD-optimized later.
static inline void argmax_f32(const float* x, std::size_t len, std::size_t& out_idx, float& out_val) {
    std::size_t idx = 0; float val = x[0];
    for (std::size_t i = 1; i < len; ++i) { const float v = x[i]; if (v > val) { val = v; idx = i; } }
    out_idx = idx; out_val = val;
}


void projection_screen_select(const ProjScreenInputs& in, ProjScreenOutputs& out) {
    using clock = std::chrono::steady_clock;
    const bool do_prof = [](){ auto v = vesper::core::safe_getenv("VESPER_PROJ_PROF"); return v && !v->empty() && ((*v)[0]=='1'); }();
    auto t0 = clock::now();

    const std::size_t n = in.n;
    const std::size_t C = in.C;
    const std::size_t p = in.p;
    const std::uint32_t L = std::max<std::uint32_t>(1u, std::min<std::uint32_t>(in.L, static_cast<std::uint32_t>(C)));

    auto row_base = [&](std::size_t i){ return i * static_cast<std::size_t>(L); };

    // Maintain per-row current top-L across processed blocks
    std::vector<std::uint32_t> curr_idx(n * static_cast<std::size_t>(L), 0);
    std::vector<float> curr_dist(n * static_cast<std::size_t>(L), INFINITY);
    std::vector<std::size_t> curr_size(n, 0);
    // Cached worst tracking for running top-L per row
    std::vector<std::size_t> curr_worst_pos(n, 0);
    std::vector<float> curr_worst_val(n, -std::numeric_limits<float>::infinity());

    double compute_ms_acc = 0.0, select_ms_acc = 0.0;
    std::size_t total_block_candidates = 0;

#ifdef VESPER_HAS_CBLAS
    const std::size_t QB = 256, CB = 256;
    std::vector<float> dots(QB * CB);
    for (std::size_t j0 = 0; j0 < C; j0 += CB) {
        const std::size_t jb = std::min<std::size_t>(CB, C - j0);
        for (std::size_t i0 = 0; i0 < n; i0 += QB) {
            const std::size_t qb = std::min<std::size_t>(QB, n - i0);
            auto t_c0 = clock::now();
            // C = A * B^T where A = qproj[i0: i0+qb, :], B = centroids_rm[j0: j0+jb, :]
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        static_cast<int>(qb), static_cast<int>(jb), static_cast<int>(p),
                        1.0f,
                        in.qproj + i0 * p, static_cast<int>(p),
                        in.centroids_rm + j0 * p, static_cast<int>(p),
                        0.0f,
                        dots.data(), static_cast<int>(jb));
            auto t_c1 = clock::now();
            compute_ms_acc += std::chrono::duration<double, std::milli>(t_c1 - t_c0).count();

            auto t_s0 = clock::now();
            // For each row in this block: build jb distances, select top-T in-block (T=L/2), then merge with curr top-L
            const std::uint32_t T = static_cast<std::uint32_t>(std::min<std::size_t>(L, jb));
            for (std::size_t r = 0; r < qb; ++r) {
                const std::size_t i = i0 + r;
                const float qi = in.qnorm[i];
                // Fixed-size block top-L selection (no nth_element)
                std::uint32_t blk_idx[256];
                float blk_dist[256];
                std::size_t bsz = 0; int argmax = -1; float maxd = -std::numeric_limits<float>::infinity();
                for (std::size_t jj = 0; jj < jb; ++jj) {
                    const std::size_t cj = j0 + jj;
                    const float dot = dots[r * jb + jj];
                    const float d = qi + in.centroid_norms[cj] - 2.0f * dot;
                    if (bsz < T) {
                        blk_idx[bsz] = static_cast<std::uint32_t>(cj);
                        blk_dist[bsz] = d;
                        if (argmax < 0 || d > maxd) { argmax = static_cast<int>(bsz); maxd = d; }
                        ++bsz;
                    } else if (d < maxd) {
                        blk_idx[argmax] = static_cast<std::uint32_t>(cj);
                        blk_dist[argmax] = d;
                        // recompute argmax
                        argmax = 0; maxd = blk_dist[0];
                        for (std::size_t t = 1; t < bsz; ++t) { if (blk_dist[t] > maxd) { maxd = blk_dist[t]; argmax = static_cast<int>(t); } }
                    }
                }
                // Merge block results into current top-L (cached worst tracking)
                const std::size_t base = row_base(i);
                for (std::size_t t = 0; t < bsz; ++t) {
                    const float d = blk_dist[t];
                    if (curr_size[i] < L) {
                        const std::size_t pos = base + curr_size[i];
                        curr_idx[pos] = blk_idx[t];
                        curr_dist[pos] = d;
                        if (curr_size[i] == 0 || d > curr_worst_val[i]) { curr_worst_pos[i] = curr_size[i]; curr_worst_val[i] = d; }
                        ++curr_size[i];
                    } else if (d < curr_worst_val[i]) {
                        const std::size_t wpos = base + curr_worst_pos[i];
                        curr_idx[wpos] = blk_idx[t];
                        curr_dist[wpos] = d;
                        std::size_t w = 0; float wdist = 0.0f;
                        argmax_f32(&curr_dist[base], L, w, wdist);
                        curr_worst_pos[i] = w; curr_worst_val[i] = wdist;
                    }
                }
                total_block_candidates += bsz;
            }
            auto t_s1 = clock::now();
            select_ms_acc += std::chrono::duration<double, std::milli>(t_s1 - t_s0).count();
            if (do_prof) {
                fprintf(stderr, "[proj-prof] block j0=%zu jb=%zu i0=%zu qb=%zu gemm_ms=%.3f sel_ms=%.3f\n", j0, jb, i0, qb,
                        std::chrono::duration<double, std::milli>(t_c1 - t_c0).count(),
                        std::chrono::duration<double, std::milli>(t_s1 - t_s0).count());
            }
        }
    }
#elif defined(__AVX2__)
    if (p == 16 && in.centroids_pack8 != nullptr) {
        const std::size_t CB = 256;
        for (std::size_t j0 = 0; j0 < C; j0 += CB) {
            const std::size_t jb = std::min<std::size_t>(CB, C - j0);
            auto t_c0 = clock::now();
            // We'll process queries in tiles of 16 rows
            const std::size_t ntiles = (n / 16) * 16;
            // Temporary buffers reused per tile
            // Selection buffers
            // Use per-row distance buffer sized [16 x jb]
            std::vector<float> dist_buf16(jb * 16);
            for (std::size_t tile = 0; tile < ntiles; tile += 16) {
                const std::size_t i0 = tile;
                // Pack A-panel: Apack[k*16 + r] = qproj[(i0+r)*p + k]
                alignas(32) float Apack[16*16];
                for (int k = 0; k < 16; ++k) {
                    for (int r = 0; r < 16; ++r) {
                        Apack[k*16 + r] = in.qproj[(static_cast<std::size_t>(i0 + r) * p) + static_cast<std::size_t>(k)];
                    }
                }
                // We'll accumulate distances for this block per row
                // Initialize dist_buf per row on demand inside row loop
                const std::size_t cb_start = (j0 / 8) * 8;
                const std::size_t cb_end = j0 + jb;
                // Precompute qnorm into small array
                float qnorm16[16]; for (int r = 0; r < 16; ++r) qnorm16[r] = in.qnorm[i0 + static_cast<std::size_t>(r)];
                // For each centroid 8-pack in [cb_start, cb_end)
                for (std::size_t cjblk = cb_start; cjblk < cb_end; cjblk += 8) {
                    const std::size_t block_id = cjblk / 8;
                    const float* Bpack = in.centroids_pack8 + block_id * (16 * 8);
                    __m256 Ctop[8]; __m256 Cbot[8];
                    for (int j = 0; j < 8; ++j) { Ctop[j] = _mm256_setzero_ps(); Cbot[j] = _mm256_setzero_ps(); }
                    for (int k = 0; k < 16; ++k) {
                        __m256 a_top = _mm256_loadu_ps(Apack + k*16 + 0);
                        __m256 a_bot = _mm256_loadu_ps(Apack + k*16 + 8);
                        const float* yk = Bpack + k*8;
                        for (int j = 0; j < 8; ++j) {
                            __m256 b = _mm256_broadcast_ss(yk + j);
                            Ctop[j] = _mm256_fmadd_ps(a_top, b, Ctop[j]);
                            Cbot[j] = _mm256_fmadd_ps(a_bot, b, Cbot[j]);
                        }
                    }
                    // Store dots to temporaries
                    float dots_top[8][8]; float dots_bot[8][8];
                    for (int j = 0; j < 8; ++j) { _mm256_storeu_ps(dots_top[j], Ctop[j]); _mm256_storeu_ps(dots_bot[j], Cbot[j]); }
                    // Convert to distances and stash into per-row buffer dist_buf16[r*jb + (cj - j0)]
                    for (int r = 0; r < 16; ++r) {
                        const std::size_t i = static_cast<std::size_t>(i0 + r);
                        if (i >= n) break;
                        for (int lane = 0; lane < 8; ++lane) {
                            const std::size_t cj = cjblk + static_cast<std::size_t>(lane);
                            if (cj < j0 || cj >= (j0 + jb) || cj >= C) continue;
                            const float dot = (r < 8 ? dots_top[lane][r] : dots_bot[lane][r - 8]);
                            dist_buf16[static_cast<std::size_t>(r) * jb + (cj - j0)] = qnorm16[r] + in.centroid_norms[cj] - 2.0f * dot;
                        }
                    }
                }
                // For each row in this tile, select top-T (T=L/2) for this block using fixed-size buffers, and merge with cached-worst
                const std::uint32_t T = static_cast<std::uint32_t>(std::min<std::size_t>(L, jb));
                for (int r = 0; r < 16; ++r) {
                    const std::size_t i = static_cast<std::size_t>(i0 + r);
                    if (i >= n) break;
                    std::uint32_t blk_idx[256];
                    float blk_dist[256];
                    std::size_t bsz = 0; int argmax = -1; float maxd = -std::numeric_limits<float>::infinity();
                    for (std::size_t jj = 0; jj < jb; ++jj) {
                        const float d = dist_buf16[static_cast<std::size_t>(r) * jb + jj];
                        if (bsz < T) {
                            blk_idx[bsz] = static_cast<std::uint32_t>(j0 + jj);
                            blk_dist[bsz] = d;
                            if (argmax < 0 || d > maxd) { argmax = static_cast<int>(bsz); maxd = d; }
                            ++bsz;
                        } else if (d < maxd) {
                            blk_idx[argmax] = static_cast<std::uint32_t>(j0 + jj);
                            blk_dist[argmax] = d;
                            argmax = 0; maxd = blk_dist[0];
                            for (std::size_t t = 1; t < bsz; ++t) { if (blk_dist[t] > maxd) { maxd = blk_dist[t]; argmax = static_cast<int>(t); } }
                        }
                    }
                    const std::size_t base = row_base(i);
                    for (std::size_t t = 0; t < bsz; ++t) {
                        const float d = blk_dist[t];
                        if (curr_size[i] < L) {
                            const std::size_t pos = base + curr_size[i];
                            curr_idx[pos] = blk_idx[t];
                            curr_dist[pos] = d;
                            if (curr_size[i] == 0 || d > curr_worst_val[i]) { curr_worst_pos[i] = curr_size[i]; curr_worst_val[i] = d; }
                            ++curr_size[i];
                        } else if (d < curr_worst_val[i]) {
                            const std::size_t wpos = base + curr_worst_pos[i];
                            curr_idx[wpos] = blk_idx[t];
                            curr_dist[wpos] = d;
                            std::size_t w = 0; float wdist = 0.0f;
                            argmax_f32(&curr_dist[base], L, w, wdist);
                            curr_worst_pos[i] = w; curr_worst_val[i] = wdist;
                        }
                    }
                    total_block_candidates += bsz;
                }
            }
                // Handle remainder rows not covered by 16-row tiles
                if (ntiles < n) {
                    const std::uint32_t T = static_cast<std::uint32_t>(std::min<std::size_t>(L, jb));
                    for (std::size_t i = ntiles; i < n; ++i) {
                        // Compute distances for this block j0..j0+jb for row i (scalar dot over p=16)
                        const float qi = in.qnorm[i];
                        std::vector<float> dist1(jb);
                        for (std::size_t jj = 0; jj < jb; ++jj) {
                            const std::size_t cj = j0 + jj;
                            const float* yc = in.centroids_rm + cj * p;
                            const float* qp = in.qproj + i * p;
                            float dot = 0.0f; for (int k = 0; k < 16; ++k) dot += qp[k] * yc[k];
                            dist1[jj] = qi + in.centroid_norms[cj] - 2.0f * dot;
                        }
                        // Select top-T for this block and merge into running top-L
                        std::uint32_t blk_idx[256];
                        float blk_dist[256];
                        std::size_t bsz = 0; int argmax = -1; float maxd = -std::numeric_limits<float>::infinity();
                        for (std::size_t jj = 0; jj < jb; ++jj) {
                            const float d = dist1[jj];
                            if (bsz < T) {
                                blk_idx[bsz] = static_cast<std::uint32_t>(j0 + jj);
                                blk_dist[bsz] = d;
                                if (argmax < 0 || d > maxd) { argmax = static_cast<int>(bsz); maxd = d; }
                                ++bsz;
                            } else if (d < maxd) {
                                blk_idx[argmax] = static_cast<std::uint32_t>(j0 + jj);
                                blk_dist[argmax] = d;
                                argmax = 0; maxd = blk_dist[0];
                                for (std::size_t t = 1; t < bsz; ++t) { if (blk_dist[t] > maxd) { maxd = blk_dist[t]; argmax = static_cast<int>(t); } }
                            }
                        }
                        const std::size_t base = row_base(i);
                        for (std::size_t t = 0; t < bsz; ++t) {
                            const float d = blk_dist[t];
                            if (curr_size[i] < L) {
                                const std::size_t pos = base + curr_size[i];
                                curr_idx[pos] = blk_idx[t];
                                curr_dist[pos] = d;
                                if (curr_size[i] == 0 || d > curr_worst_val[i]) { curr_worst_pos[i] = curr_size[i]; curr_worst_val[i] = d; }
                                ++curr_size[i];
                            } else if (d < curr_worst_val[i]) {
                                const std::size_t wpos = base + curr_worst_pos[i];
                                curr_idx[wpos] = blk_idx[t];
                                curr_dist[wpos] = d;
                                std::size_t w = 0; float wdist = 0.0f;
                                argmax_f32(&curr_dist[base], L, w, wdist);
                                curr_worst_pos[i] = w; curr_worst_val[i] = wdist;
                            }
                        }
                        total_block_candidates += bsz;
                    }
                }

            auto t_c1 = clock::now();
            compute_ms_acc += std::chrono::duration<double, std::milli>(t_c1 - t_c0).count();
            if (do_prof) {
                fprintf(stderr, "[proj-prof] block j0=%zu jb=%zu avx2_ms=%.3f\n", j0, jb,
                        std::chrono::duration<double, std::milli>(t_c1 - t_c0).count());
            }
        }
    } else
#endif
#if !defined(VESPER_HAS_CBLAS)
    {
        // Scalar fallback with blockwise selection
        const std::size_t CB = 256;
        for (std::size_t j0 = 0; j0 < C; j0 += CB) {
            const std::size_t jb = std::min<std::size_t>(CB, C - j0);
            auto t_c0 = clock::now();
            (void)t_c0; // compute is fused with selection here
            auto t_s0 = clock::now();
            // Scalar fallback: compute distances, select top-T (T=L/2) per block using fixed-size buffers, then merge with cached-worst
            std::vector<float> dist_buf(jb);
            const std::uint32_t T = static_cast<std::uint32_t>(std::min<std::size_t>(L, jb));
            for (std::size_t i = 0; i < n; ++i) {
                const float* qp = in.qproj + i * p;
                const float qi = in.qnorm[i];
                for (std::size_t jj = 0; jj < jb; ++jj) {
                    const std::size_t cj = j0 + jj;
                    const float* yc = in.centroids_rm + cj * p;
                    float dot = 0.0f; for (std::size_t k = 0; k < p; ++k) dot += qp[k] * yc[k];
                    dist_buf[jj] = qi + in.centroid_norms[cj] - 2.0f * dot;
                }
                std::uint32_t blk_idx[256];
                float blk_dist[256];
                std::size_t bsz = 0; int argmax = -1; float maxd = -std::numeric_limits<float>::infinity();
                for (std::size_t jj = 0; jj < jb; ++jj) {
                    const float d = dist_buf[jj];
                    if (bsz < T) {
                        blk_idx[bsz] = static_cast<std::uint32_t>(j0 + jj);
                        blk_dist[bsz] = d;
                        if (argmax < 0 || d > maxd) { argmax = static_cast<int>(bsz); maxd = d; }
                        ++bsz;
                    } else if (d < maxd) {
                        blk_idx[argmax] = static_cast<std::uint32_t>(j0 + jj);
                        blk_dist[argmax] = d;
                        argmax = 0; maxd = blk_dist[0];
                        for (std::size_t t = 1; t < bsz; ++t) { if (blk_dist[t] > maxd) { maxd = blk_dist[t]; argmax = static_cast<int>(t); } }
                    }
                }
                const std::size_t base = row_base(i);
                for (std::size_t t = 0; t < bsz; ++t) {
                    const float d = blk_dist[t];
                    if (curr_size[i] < L) {
                        const std::size_t pos = base + curr_size[i];
                        curr_idx[pos] = blk_idx[t];
                        curr_dist[pos] = d;
                        if (curr_size[i] == 0 || d > curr_worst_val[i]) { curr_worst_pos[i] = curr_size[i]; curr_worst_val[i] = d; }
                        ++curr_size[i];
                    } else if (d < curr_worst_val[i]) {
                        const std::size_t wpos = base + curr_worst_pos[i];
                        curr_idx[wpos] = blk_idx[t];
                        curr_dist[wpos] = d;
                        std::size_t w = 0; float wdist = 0.0f;
                        argmax_f32(&curr_dist[base], L, w, wdist);
                        curr_worst_pos[i] = w; curr_worst_val[i] = wdist;
                    }
                }
                total_block_candidates += bsz;
            }
            auto t_s1 = clock::now();
            select_ms_acc += std::chrono::duration<double, std::milli>(t_s1 - t_s0).count();
            if (do_prof) {
                fprintf(stderr, "[proj-prof] block j0=%zu jb=%zu scalar_sel_ms=%.3f\n", j0, jb,
                        std::chrono::duration<double, std::milli>(t_s1 - t_s0).count());
            }
        }
    }
#endif

    // Scatter to output arrays in arbitrary order (not guaranteed sorted)
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t base = row_base(i);
        const std::size_t hs = curr_size[i];
        for (std::size_t t = 0; t < L; ++t) {
            const std::size_t src = (t < hs) ? (base + t) : base; // pad if needed
            out.cand_idx[base + t]  = curr_idx[src];
            out.cand_dist[base + t] = curr_dist[src];
        }
    }

    auto t1 = clock::now();
    if (do_prof) {
        auto total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        fprintf(stderr, "[proj-prof] total_ms=%.3f compute_ms=%.3f select_ms=%.3f n=%zu C=%zu p=%zu L=%u total_block_cands=%zu\n",
                total_ms, compute_ms_acc, select_ms_acc, n, C, p, L, total_block_candidates);
    }
}

} // namespace index
} // namespace vesper

