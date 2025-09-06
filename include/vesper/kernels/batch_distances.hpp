#pragma once

/** \file batch_distances.hpp
 *  \brief SIMD-optimized batch distance computations.
 *
 * Provides high-performance batch distance operations for:
 * - Distance matrix computation
 * - Multi-query search
 * - Centroid-to-centroid distances
 *
 * Optimizations:
 * - AVX2/AVX-512 vectorization
 * - Cache blocking for large matrices
 * - Parallel execution with OpenMP
 */

#include <cstdint>
#include <vesper/span_polyfill.hpp>
#include <vector>
#include <algorithm>
#include <numeric>
#ifdef __x86_64__
#include <immintrin.h>
#endif
#include <queue>
#include <limits>




#include "vesper/kernels/dispatch.hpp"
#include "vesper/index/aligned_buffer.hpp"

namespace vesper::kernels {

// Forward declarations for unified APIs used below
/** \brief Operation selector for batch distance/score computations.
 *
 * Semantics:
 * - L2               => squared L2 distance (smaller is closer)
 * - InnerProduct     => dot-product similarity (larger is closer)
 * - CosineDistance   => cosine distance 1 - cos(a,b) (smaller is closer)
 * - CosineSimilarity => cosine similarity cos(a,b) (larger is closer)
 *
 * Ordering:
 * - Distance ops sort ascending; similarity ops sort descending.
 *
 * Determinism: All kernels are deterministic for identical inputs across runs
 * (parallel, but without races on outputs). Numeric differences across
 * backends are bounded and covered by unit test tolerances.
 */

enum class DistanceOp { L2, InnerProduct, CosineDistance, CosineSimilarity };
inline auto distance_matrix(
    DistanceOp op,
    const float* a_data, std::size_t n_a,
    const float* b_data, std::size_t n_b,
    std::size_t dim,
    float* out) -> void;
inline auto distance_matrix(
    DistanceOp op,
    const index::AlignedCentroidBuffer& A,
    const index::AlignedCentroidBuffer& B,
    float* out) -> void;


// Helper predicates clarifying semantics per operation
inline constexpr bool is_similarity_op(DistanceOp op) noexcept {
  return op == DistanceOp::InnerProduct || op == DistanceOp::CosineSimilarity;
}
inline constexpr bool is_distance_op(DistanceOp op) noexcept {
  return op == DistanceOp::L2 || op == DistanceOp::CosineDistance;
}
// Sorting direction per op: distances ascending; similarities descending
inline constexpr bool sort_ascending(DistanceOp op) noexcept { return is_distance_op(op); }

inline auto distance_matrix(
    DistanceOp op,
    const float* queries, std::size_t n_queries,
    const index::AlignedCentroidBuffer& centroids,
    std::size_t dim,
    float* out) -> void;





/** \brief Compute distance matrix between two sets of vectors.
 *
 * Computes all pairwise distances between vectors in sets A and B.
 * Uses cache blocking and SIMD for optimal performance.
 *
 * \param a_data First set of vectors [n_a x dim]
 * \param b_data Second set of vectors [n_b x dim]
 * \param n_a Number of vectors in set A
 * \param n_b Number of vectors in set B
 * \param dim Vector dimensionality
 * \param[out] distances Output matrix [n_a x n_b]
 *
 * Complexity: O(n_a * n_b * dim) with SIMD acceleration
 */
inline auto compute_distance_matrix_l2(
    const float* a_data, const float* b_data,
    std::size_t n_a, std::size_t n_b, std::size_t dim,
    float* distances) -> void {
  // Forward to unified API for consistency and maintenance
  distance_matrix(DistanceOp::L2, a_data, n_a, b_data, n_b, dim, distances);
}

/** \brief Compute distance matrix with AVX2 specialization.
 *
 * Optimized for aligned memory and dimension divisible by 8.
 */
#ifdef __AVX2__
inline auto compute_distance_matrix_l2_avx2(
    const index::AlignedCentroidBuffer& a_buffer,
    const index::AlignedCentroidBuffer& b_buffer,
    index::AlignedDistanceMatrix& distances) -> void {

    const std::uint32_t n_a = a_buffer.size();
    const std::uint32_t n_b = b_buffer.size();
    const std::size_t dim = a_buffer.dimension();

    // Ensure dimension is multiple of 8 for AVX2
    const std::size_t vec_dim = dim & ~7ULL;
    const std::size_t remainder = dim - vec_dim;

    #pragma omp parallel for schedule(dynamic, 4)
    for (std::uint32_t i = 0; i < n_a; ++i) {
        const float* a_vec = a_buffer[i];

        // Prefetch next row
        if (i + 1 < n_a) {
            a_buffer.prefetch_read(i + 1);
        }

        for (std::uint32_t j = 0; j < n_b; ++j) {
            const float* b_vec = b_buffer[j];

            __m256 sum = _mm256_setzero_ps();

            // Main vectorized loop
            for (std::size_t d = 0; d < vec_dim; d += 8) {
                const __m256 a = _mm256_load_ps(a_vec + d);
                const __m256 b = _mm256_load_ps(b_vec + d);
                const __m256 diff = _mm256_sub_ps(a, b);
                sum = _mm256_fmadd_ps(diff, diff, sum);
            }

            // Horizontal sum
            const __m128 sum_high = _mm256_extractf128_ps(sum, 1);
            const __m128 sum_low = _mm256_castps256_ps128(sum);
            const __m128 sum128 = _mm_add_ps(sum_low, sum_high);
            const __m128 sum64 = _mm_hadd_ps(sum128, sum128);
            const __m128 sum32 = _mm_hadd_ps(sum64, sum64);
            float result = _mm_cvtss_f32(sum32);

            // Handle remainder
            for (std::size_t d = vec_dim; d < dim; ++d) {
                const float diff = a_vec[d] - b_vec[d];
                result += diff * diff;
            }

            distances.set(i, j, result);
        }
    }
}
#endif

/** \brief Compute distances from multiple queries to centroids.
 *
 * Optimized for search operations where we need distances from
 * many queries to a fixed set of centroids.
 *
 * \param queries Query vectors [n_queries x dim]
 * \param centroids Centroid buffer
 * \param n_queries Number of queries
 * \param[out] distances Output [n_queries x k]
 */
inline auto compute_query_centroid_distances(
    const float* queries,
    const index::AlignedCentroidBuffer& centroids,
    std::size_t n_queries,
    float* distances) -> void {
  // Forward to unified rectangular API (queries x centroids)
  const std::size_t dim = centroids.dimension();
  distance_matrix(DistanceOp::L2, queries, n_queries, centroids, dim, distances);
}

/** \brief Find k nearest centroids for multiple queries.
 *
 * \param queries Query vectors [n_queries x dim]
 * \param centroids Centroid buffer
 * \param n_queries Number of queries
 * \param k_nearest Number of nearest centroids to find
 * \param[out] indices Nearest centroid indices [n_queries x k_nearest]
 * \param[out] distances Distances to nearest centroids [n_queries x k_nearest]
 */
/** \deprecated Materializes all query-to-centroid distances before selecting top-k.
 * Prefer find_nearest_centroids_batch_fused(...) which avoids materialization and
 * typically reduces memory bandwidth and allocations. This function is kept for
 * backward compatibility and may be deprecated in a future minor release.
 *
 * Ordering: ascending by distance; ties broken by smaller centroid index (stable).
 */

inline auto find_nearest_centroids_batch(
    const float* queries,
    const index::AlignedCentroidBuffer& centroids,
    std::size_t n_queries,
    std::uint32_t k_nearest,
    std::uint32_t* indices,
    float* distances) -> void {

    const std::uint32_t n_centroids = centroids.size();
    // dim not needed here; compute_query_centroid_distances handles dimension

    // Temporary buffer for all distances
    std::vector<float> all_distances(n_queries * n_centroids);
    compute_query_centroid_distances(queries, centroids, n_queries, all_distances.data());

    // Find k nearest for each query
    #pragma omp parallel for
    for (int q = 0; q < static_cast<int>(n_queries); ++q) {
        const float* query_dists = all_distances.data() + q * n_centroids;
        std::uint32_t* query_indices = indices + q * k_nearest;
        float* query_nearest = distances + q * k_nearest;

        // Create index array
        std::vector<std::uint32_t> idx(n_centroids);
        std::iota(idx.begin(), idx.end(), 0);

        // Partial sort to find k nearest
        std::partial_sort(idx.begin(), idx.begin() + k_nearest, idx.end(),
            [query_dists](std::uint32_t i, std::uint32_t j) {
                const float di = query_dists[i];
                const float dj = query_dists[j];
                if (di < dj) return true;
                if (di > dj) return false;
                return i < j; // stable tie-break by smaller index
            });

        // Copy results
        for (std::uint32_t i = 0; i < k_nearest; ++i) {
            query_indices[i] = idx[i];
            query_nearest[i] = query_dists[idx[i]];
        }
    }
}

/** \brief Compute symmetric distance matrix with optimizations.
 *
 * Exploits symmetry to compute only upper triangle.
 */
inline auto compute_symmetric_distance_matrix(
    const index::AlignedCentroidBuffer& centroids,
    index::AlignedDistanceMatrix& distances) -> void {

    const std::uint32_t k = centroids.size();
    const std::size_t dim = centroids.dimension();
    (void)dim; // may be unused on some backends

    #ifdef __AVX2__
    if (dim >= 8 && dim % 8 == 0) {
        // Use optimized AVX2 version
        #pragma omp parallel for schedule(dynamic, 4)
        for (int i = 0; i < static_cast<int>(k); ++i) {
            const float* a_vec = centroids[i];

            // Diagonal is always 0
            distances.set(i, i, 0.0f);

            // Compute upper triangle
            for (std::uint32_t j = i + 1; j < k; ++j) {
                const float* b_vec = centroids[j];

                __m256 sum = _mm256_setzero_ps();

                for (std::size_t d = 0; d < dim; d += 8) {
                    const __m256 a = _mm256_load_ps(a_vec + d);
                    const __m256 b = _mm256_load_ps(b_vec + d);
                    const __m256 diff = _mm256_sub_ps(a, b);
                    sum = _mm256_fmadd_ps(diff, diff, sum);
                }

                // Horizontal sum
                const __m128 sum_high = _mm256_extractf128_ps(sum, 1);
                const __m128 sum_low = _mm256_castps256_ps128(sum);
                const __m128 sum128 = _mm_add_ps(sum_low, sum_high);
                const __m128 sum64 = _mm_hadd_ps(sum128, sum128);
                const __m128 sum32 = _mm_hadd_ps(sum64, sum64);
                const float dist = _mm_cvtss_f32(sum32);

                distances.set_symmetric(i, j, dist);
            }
        }
    } else
    #endif
    {
        // Fallback to standard computation
        const auto& ops = select_backend_auto();
        #pragma omp parallel for schedule(dynamic, 4)
        for (int i = 0; i < static_cast<int>(k); ++i) {
            distances.set(i, i, 0.0f);

            for (std::uint32_t j = i + 1; j < k; ++j) {
                const float dist = ops.l2_sq(
                    centroids.get_centroid(i),
                    centroids.get_centroid(j)
                );
                distances.set_symmetric(i, j, dist);
            }

        }
    }
  }


  /** \brief Fused multi-query top-k over centroids without materializing full distance rows.
   *
   * Computes, for each query, the top-k nearest centroids according to the selected operation.
   * No intermediate [n_centroids] distance array is materialized; selection is performed on-the-fly.
   *
   * Semantics by op:
   * - DistanceOp::L2, DistanceOp::CosineDistance  => treat values as distances; per-query results sorted ascending
   * - DistanceOp::InnerProduct, DistanceOp::CosineSimilarity => treat values as similarities; per-query results sorted descending
   * - Ties are broken by smaller centroid index (stable, deterministic ordering)
   *
   * Contracts:
   * - queries: row-major [n_queries x dim]
   * - indices_out: [n_queries x k], distances_out: [n_queries x k]
   * - Effective k is min(k, centroids.size())
   * - Inputs must be finite; for cosine ops, vector norms should be > 0
   * - Deterministic for identical inputs
   *
   * Complexity: O(n_queries * centroids.size() * dim) with per-query O(k) memory; parallelized across queries.
   */
  inline auto find_nearest_centroids_batch_fused(
      DistanceOp op,
      const float* queries,
      std::size_t n_queries,
      const index::AlignedCentroidBuffer& centroids,
      std::uint32_t k,
      std::uint32_t* indices_out,
      float* distances_out) -> void {

    const auto& ops = select_backend_auto();
    const std::uint32_t n_centroids = centroids.size();
    if (n_centroids == 0 || n_queries == 0 || k == 0) return;
    const std::uint32_t k_eff = std::min<std::uint32_t>(k, n_centroids);
    const std::size_t dim = centroids.dimension();

    const bool is_similarity = (op == DistanceOp::InnerProduct || op == DistanceOp::CosineSimilarity);

    #pragma omp parallel for schedule(dynamic, 32)
    for (std::size_t q = 0; q < n_queries; ++q) {
      const float* query = queries + q * dim;

      // Heap of (score, index). For distances we keep a max-heap (largest on top),
      // for similarities we keep a min-heap (smallest on top), both of size <= k.
      struct Pair { float score; std::uint32_t idx; };
      auto dist_cmp = [](const Pair& a, const Pair& b){
        if (a.score != b.score) return a.score < b.score; // max-heap: larger score is worse
        return a.idx < b.idx; // tie-break: larger index is worse
      };
      auto sim_cmp  = [](const Pair& a, const Pair& b){
        if (a.score != b.score) return a.score > b.score; // min-heap: smaller score is worse
        return a.idx < b.idx; // tie-break: larger index is worse
      };

      std::vector<Pair> heap;
      heap.reserve(k_eff + 1);

      for (std::uint32_t c = 0; c < n_centroids; ++c) {
        float s;
        switch (op) {
          case DistanceOp::L2:
            s = ops.l2_sq(std::span(query, dim), centroids.get_centroid(c));
            break;
          case DistanceOp::InnerProduct:
            s = ops.inner_product(std::span(query, dim), centroids.get_centroid(c));
            break;
          case DistanceOp::CosineDistance:
            s = ops.cosine_distance(std::span(query, dim), centroids.get_centroid(c));
            break;
          case DistanceOp::CosineSimilarity:
            s = ops.cosine_similarity(std::span(query, dim), centroids.get_centroid(c));
            break;


        }

        if (!is_similarity) {
          // distances: keep max-heap of size k_eff
          if (heap.size() < k_eff) {
            heap.push_back({s, c});
            std::push_heap(heap.begin(), heap.end(), dist_cmp);
          } else if (s < heap.front().score || (s == heap.front().score && c < heap.front().idx)) {
            std::pop_heap(heap.begin(), heap.end(), dist_cmp);
            heap.back() = {s, c};
            std::push_heap(heap.begin(), heap.end(), dist_cmp);
          }
        } else {
          // similarities: keep min-heap of size k_eff
          if (heap.size() < k_eff) {
            heap.push_back({s, c});
            std::push_heap(heap.begin(), heap.end(), sim_cmp);
          } else if (s > heap.front().score || (s == heap.front().score && c < heap.front().idx)) {
            std::pop_heap(heap.begin(), heap.end(), sim_cmp);
            heap.back() = {s, c};
            std::push_heap(heap.begin(), heap.end(), sim_cmp);
          }
        }
      }

      // Extract in the requested order: distances ascending; similarities descending
      if (!is_similarity) {
        std::sort_heap(heap.begin(), heap.end(), dist_cmp); // ascending by score
        for (std::uint32_t i = 0; i < k_eff; ++i) {
/** \brief Rectangular distance/score matrix (raw pointers A x B).
 *
 * Computes all pairwise values between A [n_a x dim] and B [n_b x dim] in row-major order.
 * Interpretation depends on op:
 * - L2, CosineDistance => distances (smaller is closer)
 * - InnerProduct, CosineSimilarity => similarity scores (larger is closer)
 *
 * \param op Operation selector (see DistanceOp)
 * \param a_data Row-major A data [n_a x dim]
 * \param n_a Number of rows in A
 * \param b_data Row-major B data [n_b x dim]
 * \param n_b Number of rows in B
 * \param dim Vector dimensionality
 * \param[out] out Row-major [n_a x n_b]
 *
 * Deterministic given identical inputs. Parallelized with OpenMP where available.
 */

          indices_out[q * k + i]  = heap[i].idx;
          distances_out[q * k + i] = heap[i].score;
        }
        // If k > k_eff, leave the remainder undefined
      } else {
        std::sort_heap(heap.begin(), heap.end(), sim_cmp); // descending by score
        for (std::uint32_t i = 0; i < k_eff; ++i) {
/** \brief Rectangular distance/score matrix for aligned centroid buffers (A x B).
 * Equivalent semantics to the raw-pointer overload but uses AlignedCentroidBuffer
 * for both inputs. Output is row-major [A.size() x B.size()].
 * Deterministic; parallelized where available.
 */

          indices_out[q * k + i]  = heap[i].idx;
          distances_out[q * k + i] = heap[i].score;
        }
      }
    }
  }

  // Unified rectangular distance/score matrix (raw pointers)
  inline auto distance_matrix(
      DistanceOp op,
      const float* a_data, std::size_t n_a,
      const float* b_data, std::size_t n_b,
      std::size_t dim,
      float* out) -> void {
    const auto& ops = select_backend_auto();
    #pragma omp parallel for collapse(2) schedule(dynamic, 1)
    for (std::size_t i = 0; i < n_a; ++i) {
      for (std::size_t j = 0; j < n_b; ++j) {
        const float* av = a_data + i * dim;
        const float* bv = b_data + j * dim;
        float v = 0.0f;
        switch (op) {
          case DistanceOp::L2:              v = ops.l2_sq(std::span(av, dim), std::span(bv, dim)); break;
          case DistanceOp::InnerProduct:    v = ops.inner_product(std::span(av, dim), std::span(bv, dim)); break;
          case DistanceOp::CosineDistance:  v = ops.cosine_distance(std::span(av, dim), std::span(bv, dim)); break;
          case DistanceOp::CosineSimilarity:v = ops.cosine_similarity(std::span(av, dim), std::span(bv, dim)); break;
        }
        out[i * n_b + j] = v;
      }
    }
  }

  // Unified rectangular distance/score matrix (aligned buffers)
  inline auto distance_matrix(
      DistanceOp op,
      const index::AlignedCentroidBuffer& A,
      const index::AlignedCentroidBuffer& B,
      float* out) -> void {
    const auto& ops = select_backend_auto();
    const std::uint32_t n_a = A.size();
    const std::uint32_t n_b = B.size();
    const std::size_t dim = A.dimension();
    (void)dim;
    #pragma omp parallel for schedule(dynamic, 4)
    for (std::uint32_t i = 0; i < n_a; ++i) {
      const auto a = A.get_centroid(i);
      for (std::uint32_t j = 0; j < n_b; ++j) {
        float v = 0.0f;
        switch (op) {
          case DistanceOp::L2:              v = ops.l2_sq(a, B.get_centroid(j)); break;
          case DistanceOp::InnerProduct:    v = ops.inner_product(a, B.get_centroid(j)); break;
          case DistanceOp::CosineDistance:  v = ops.cosine_distance(a, B.get_centroid(j)); break;
          case DistanceOp::CosineSimilarity:v = ops.cosine_similarity(a, B.get_centroid(j)); break;
        }
        out[static_cast<std::size_t>(i) * n_b + j] = v;
      }
    }
  }

  // Mixed rectangular distance/score matrix: queries (raw) x centroids (aligned)
  inline auto distance_matrix(
      DistanceOp op,
      const float* queries, std::size_t n_queries,
      const index::AlignedCentroidBuffer& centroids,
      std::size_t dim,
      float* out) -> void {
    const auto& ops = select_backend_auto();
    const std::uint32_t n_c = centroids.size();
    #pragma omp parallel for collapse(2) schedule(dynamic, 1)
    for (std::size_t q = 0; q < n_queries; ++q) {
      for (std::uint32_t c = 0; c < n_c; ++c) {
        const float* qv = queries + q * dim;
        float v = 0.0f;
        switch (op) {
          case DistanceOp::L2:               v = ops.l2_sq(std::span(qv, dim), centroids.get_centroid(c)); break;
          case DistanceOp::InnerProduct:     v = ops.inner_product(std::span(qv, dim), centroids.get_centroid(c)); break;
          case DistanceOp::CosineDistance:   v = ops.cosine_distance(std::span(qv, dim), centroids.get_centroid(c)); break;
          case DistanceOp::CosineSimilarity: v = ops.cosine_similarity(std::span(qv, dim), centroids.get_centroid(c)); break;
        }
        out[q * n_c + c] = v;
      }
    }
  }



} // namespace vesper::kernels