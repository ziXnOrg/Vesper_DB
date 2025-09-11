#include "vesper/index/projection_assigner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <numeric>
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

void projection_screen_select(const ProjScreenInputs& in, ProjScreenOutputs& out) {
    using clock = std::chrono::steady_clock;
    const bool do_prof = [](){ const char* v = std::getenv("VESPER_PROJ_PROF"); return v && v[0]=='1'; }();
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
            // For each row in this block: build jb distances, select top-L in-block, then merge with curr top-L
            std::vector<int> idx_buf(jb);
            std::vector<float> dist_buf(jb);
            const std::uint32_t lb = static_cast<std::uint32_t>(std::min<std::size_t>(L, jb));
            for (std::size_t r = 0; r < qb; ++r) {
                const std::size_t i = i0 + r;
                const float qi = in.qnorm[i];
                // Build distances for this row
                for (std::size_t jj = 0; jj < jb; ++jj) {
                    const std::size_t cj = j0 + jj;
                    const float dot = dots[r * jb + jj];
                    dist_buf[jj] = qi + in.centroid_norms[cj] - 2.0f * dot;
                }
                std::iota(idx_buf.begin(), idx_buf.end(), 0);
                if (jb > lb) {
                    std::nth_element(idx_buf.begin(), idx_buf.begin() + lb, idx_buf.end(),
                        [&](int a, int b){ return dist_buf[a] < dist_buf[b]; });
                }
                // Merge block top-lb with curr top-L
                const std::size_t base = row_base(i);
                const std::size_t s1 = curr_size[i];
                const std::size_t s2 = lb;
                std::uint32_t tmp_idx[512];
                float tmp_dist[512];
                std::size_t tcount = 0;
                for (std::size_t t = 0; t < s1; ++t) { tmp_idx[tcount] = curr_idx[base + t]; tmp_dist[tcount++] = curr_dist[base + t]; }
                for (std::size_t t = 0; t < s2; ++t) { tmp_idx[tcount] = static_cast<std::uint32_t>(j0 + idx_buf[t]); tmp_dist[tcount++] = dist_buf[idx_buf[t]]; }
                const std::size_t keep = std::min<std::size_t>(L, tcount);
                std::vector<int> ord(tcount); std::iota(ord.begin(), ord.end(), 0);
                std::nth_element(ord.begin(), ord.begin() + static_cast<long long>(keep), ord.end(),
                    [&](int a, int b){ return tmp_dist[a] < tmp_dist[b]; });
                for (std::size_t t = 0; t < keep; ++t) { curr_idx[base + t] = tmp_idx[ord[t]]; curr_dist[base + t] = tmp_dist[ord[t]]; }
                curr_size[i] = keep;
                total_block_candidates += s2;
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
            std::vector<int> idx_buf(jb);
            std::vector<float> dist_buf(jb);
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
                        __m256 yk = _mm256_loadu_ps(Bpack + k*8);
                        for (int j = 0; j < 8; ++j) {
                            __m256 b = _mm256_set1_ps(reinterpret_cast<const float*>(&yk)[j]);
                            Ctop[j] = _mm256_fmadd_ps(a_top, b, Ctop[j]);
                            Cbot[j] = _mm256_fmadd_ps(a_bot, b, Cbot[j]);
                        }
                    }
                    // Store dots to temporaries
                    float dots_top[8][8]; float dots_bot[8][8];
                    for (int j = 0; j < 8; ++j) { _mm256_storeu_ps(dots_top[j], Ctop[j]); _mm256_storeu_ps(dots_bot[j], Cbot[j]); }
                    // Convert to distances and stash into per-row dist_buf
                    for (int r = 0; r < 16; ++r) {
                        const std::size_t i = static_cast<std::size_t>(i0 + r);
                        if (i >= n) break;
                        if (dist_buf.size() != jb) dist_buf.resize(jb);
                        for (int lane = 0; lane < 8; ++lane) {
                            const std::size_t cj = cjblk + static_cast<std::size_t>(lane);
                            if (cj < j0 || cj >= (j0 + jb) || cj >= C) continue;
                            const float dot = (r < 8 ? dots_top[lane][r] : dots_bot[lane][r - 8]);
                            dist_buf[cj - j0] = qnorm16[r] + in.centroid_norms[cj] - 2.0f * dot;
                        }
                    }
                }
                // For each row in this tile, select top-L for this block and merge
                const std::uint32_t lb = static_cast<std::uint32_t>(std::min<std::size_t>(L, jb));
                for (int r = 0; r < 16; ++r) {
                    const std::size_t i = static_cast<std::size_t>(i0 + r);
                    if (i >= n) break;
                    std::iota(idx_buf.begin(), idx_buf.end(), 0);
                    if (jb > lb) {
                        std::nth_element(idx_buf.begin(), idx_buf.begin() + lb, idx_buf.end(),
                            [&](int a, int b){ return dist_buf[a] < dist_buf[b]; });
                    }
                    const std::size_t base = row_base(i);
                    const std::size_t s1 = curr_size[i];
                    const std::size_t s2 = lb;
                    std::uint32_t tmp_idx[512];
                    float tmp_dist[512];
                    std::size_t tcount = 0;
                    for (std::size_t t = 0; t < s1; ++t) { tmp_idx[tcount] = curr_idx[base + t]; tmp_dist[tcount++] = curr_dist[base + t]; }
                    for (std::size_t t = 0; t < s2; ++t) { tmp_idx[tcount] = static_cast<std::uint32_t>(j0 + idx_buf[t]); tmp_dist[tcount++] = dist_buf[idx_buf[t]]; }
                    const std::size_t keep = std::min<std::size_t>(L, tcount);
                    std::vector<int> ord(tcount); std::iota(ord.begin(), ord.end(), 0);
                    std::nth_element(ord.begin(), ord.begin() + static_cast<long long>(keep), ord.end(),
                        [&](int a, int b){ return tmp_dist[a] < tmp_dist[b]; });
                    for (std::size_t t = 0; t < keep; ++t) { curr_idx[base + t] = tmp_idx[ord[t]]; curr_dist[base + t] = tmp_dist[ord[t]]; }
                    curr_size[i] = keep;
                    total_block_candidates += s2;
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
    {
        // Scalar fallback with blockwise selection
        const std::size_t CB = 256;
        for (std::size_t j0 = 0; j0 < C; j0 += CB) {
            const std::size_t jb = std::min<std::size_t>(CB, C - j0);
            auto t_c0 = clock::now();
            (void)t_c0; // compute is fused with selection here
            auto t_s0 = clock::now();
            std::vector<int> idx_buf(jb);
            std::vector<float> dist_buf(jb);
            const std::uint32_t lb = static_cast<std::uint32_t>(std::min<std::size_t>(L, jb));
            for (std::size_t i = 0; i < n; ++i) {
                const float* qp = in.qproj + i * p;
                const float qi = in.qnorm[i];
                for (std::size_t jj = 0; jj < jb; ++jj) {
                    const std::size_t cj = j0 + jj;
                    const float* yc = in.centroids_rm + cj * p;
                    float dot = 0.0f; for (std::size_t k = 0; k < p; ++k) dot += qp[k] * yc[k];
                    dist_buf[jj] = qi + in.centroid_norms[cj] - 2.0f * dot;
                }
                std::iota(idx_buf.begin(), idx_buf.end(), 0);
                if (jb > lb) {
                    std::nth_element(idx_buf.begin(), idx_buf.begin() + lb, idx_buf.end(),
                        [&](int a, int b){ return dist_buf[a] < dist_buf[b]; });
                }
                const std::size_t base = row_base(i);
                const std::size_t s1 = curr_size[i];
                const std::size_t s2 = lb;
                std::uint32_t tmp_idx[512];
                float tmp_dist[512];
                std::size_t tcount = 0;
                for (std::size_t t = 0; t < s1; ++t) { tmp_idx[tcount] = curr_idx[base + t]; tmp_dist[tcount++] = curr_dist[base + t]; }
                for (std::size_t t = 0; t < s2; ++t) { tmp_idx[tcount] = static_cast<std::uint32_t>(j0 + idx_buf[t]); tmp_dist[tcount++] = dist_buf[idx_buf[t]]; }
                const std::size_t keep = std::min<std::size_t>(L, tcount);
                std::vector<int> ord(tcount); std::iota(ord.begin(), ord.end(), 0);
                std::nth_element(ord.begin(), ord.begin() + static_cast<long long>(keep), ord.end(),
                    [&](int a, int b){ return tmp_dist[a] < tmp_dist[b]; });
                for (std::size_t t = 0; t < keep; ++t) { curr_idx[base + t] = tmp_idx[ord[t]]; curr_dist[base + t] = tmp_dist[ord[t]]; }
                curr_size[i] = keep;
                total_block_candidates += s2;
            }
            auto t_s1 = clock::now();
            select_ms_acc += std::chrono::duration<double, std::milli>(t_s1 - t_s0).count();
            if (do_prof) {
                fprintf(stderr, "[proj-prof] block j0=%zu jb=%zu scalar_sel_ms=%.3f\n", j0, jb,
                        std::chrono::duration<double, std::milli>(t_s1 - t_s0).count());
            }
        }
    }

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

