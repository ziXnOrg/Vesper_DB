/**
 * Copyright (c) 2025 Colin Macritchie / Ripple Group, LLC
 */

#include "vesper/index/capq_dist_avx2.hpp"
#include "vesper/index/capq_q4.hpp"

#include <immintrin.h>

namespace vesper::index {

bool capq_avx2_available() noexcept {
#if defined(__AVX2__) || (defined(_MSC_VER) && defined(__AVX2__)) || defined(__AVX2__)
  return true;
#else
  return false;
#endif
}

float distance_q8_avx2(const std::int8_t* q8_query,
                       const std::int8_t* q8_db,
                       const CapqQ8Params& params) noexcept {
#if defined(__AVX2__) || (defined(_MSC_VER) && defined(__AVX2__)) || defined(__AVX2__)
  // Process 32 bytes at a time: unpack to 16-bit, subtract, square, accumulate, then scale per-dim
  // Note: We still apply per-dimension scale^2, so we accumulate in float across lanes.
  float sum = 0.0f;
  int d = 0;
  for (; d + 31 < 64; d += 32) {
    __m256i qa = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8_query + d));
    __m256i qb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8_db + d));

    __m256i qa_lo = _mm256_srai_epi16(_mm256_unpacklo_epi8(qa, _mm256_cmpgt_epi8(_mm256_setzero_si256(), qa)), 0);
    __m256i qa_hi = _mm256_srai_epi16(_mm256_unpackhi_epi8(qa, _mm256_cmpgt_epi8(_mm256_setzero_si256(), qa)), 0);
    __m256i qb_lo = _mm256_srai_epi16(_mm256_unpacklo_epi8(qb, _mm256_cmpgt_epi8(_mm256_setzero_si256(), qb)), 0);
    __m256i qb_hi = _mm256_srai_epi16(_mm256_unpackhi_epi8(qb, _mm256_cmpgt_epi8(_mm256_setzero_si256(), qb)), 0);

    __m256i diff_lo = _mm256_sub_epi16(qa_lo, qb_lo);
    __m256i diff_hi = _mm256_sub_epi16(qa_hi, qb_hi);

    // Convert to 32-bit for squaring
    __m256i dlo_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(diff_lo));
    __m256i dlo_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(diff_lo, 1));
    __m256i dhi_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(diff_hi));
    __m256i dhi_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(diff_hi, 1));

    __m256 flo_lo = _mm256_cvtepi32_ps(_mm256_mullo_epi32(dlo_lo, dlo_lo));
    __m256 flo_hi = _mm256_cvtepi32_ps(_mm256_mullo_epi32(dlo_hi, dlo_hi));
    __m256 fhi_lo = _mm256_cvtepi32_ps(_mm256_mullo_epi32(dhi_lo, dhi_lo));
    __m256 fhi_hi = _mm256_cvtepi32_ps(_mm256_mullo_epi32(dhi_hi, dhi_hi));

    // Apply per-dimension scales^2
    alignas(32) float s2[32];
    for (int i = 0; i < 32; ++i) { const float s = params.scale[d + i]; s2[i] = s * s; }
    __m256 ms0 = _mm256_load_ps(s2 + 0);
    __m256 ms1 = _mm256_load_ps(s2 + 8);
    __m256 ms2 = _mm256_load_ps(s2 + 16);
    __m256 ms3 = _mm256_load_ps(s2 + 24);

    flo_lo = _mm256_mul_ps(flo_lo, ms0);
    flo_hi = _mm256_mul_ps(flo_hi, ms1);
    fhi_lo = _mm256_mul_ps(fhi_lo, ms2);
    fhi_hi = _mm256_mul_ps(fhi_hi, ms3);

    __m256 acc = _mm256_add_ps(_mm256_add_ps(flo_lo, flo_hi), _mm256_add_ps(fhi_lo, fhi_hi));
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, acc);
    for (int i = 0; i < 8; ++i) sum += tmp[i];
  }
  // Tail
  for (; d < 64; ++d) {
    const int diff = static_cast<int>(q8_query[d]) - static_cast<int>(q8_db[d]);
    const float s = params.scale[d];
    sum += static_cast<float>(diff * diff) * (s * s);
  }
  return sum;
#else
  // Fallback to scalar if AVX2 not available
  float sum = 0.0f;
  for (int d = 0; d < 64; ++d) {
    const int diff = static_cast<int>(q8_query[d]) - static_cast<int>(q8_db[d]);
    const float s = params.scale[d];
    sum += static_cast<float>(diff * diff) * (s * s);
  }
  return sum;
#endif
}

std::uint32_t hamming_distance_avx2(const std::uint64_t* a,
                                    const std::uint64_t* b,
                                    std::size_t words) noexcept {
#if defined(__AVX2__) || (defined(_MSC_VER) && defined(__AVX2__)) || defined(__AVX2__)
  std::uint32_t dist = 0;
  std::size_t i = 0;
  // Process 256 bits per loop (4x u64)
  for (; i + 3 < words; i += 4) {
    __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
    __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
    __m256i vx = _mm256_xor_si256(va, vb);
    // popcount 64-bit lanes using lookup via 8-bit chunks
    // Split into 4x 64-bit values
    alignas(32) std::uint64_t lanes[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), vx);
    dist += static_cast<std::uint32_t>(__popcnt64(lanes[0]) + __popcnt64(lanes[1]) + __popcnt64(lanes[2]) + __popcnt64(lanes[3]));
  }
  // Tail
  for (; i < words; ++i) {
    dist += static_cast<std::uint32_t>(__popcnt64(a[i] ^ b[i]));
  }
  return dist;
#else
  std::uint32_t dist = 0;
  for (std::size_t i = 0; i < words; ++i) {
    std::uint64_t x = a[i] ^ b[i];
#if defined(__GNUG__) || defined(__clang__)
    dist += static_cast<std::uint32_t>(__builtin_popcountll(x));
#else
    dist += static_cast<std::uint32_t>(__popcnt64(x));
#endif
  }
  return dist;
#endif
}

float distance_q4_avx2(const std::uint8_t* q4_query,
                       const std::uint8_t* q4_db,
                       const CapqQ8Params& params) noexcept {
#if defined(__AVX2__) || (defined(_MSC_VER) && defined(__AVX2__)) || defined(__AVX2__)
  // Precompute (16*scale)^2
  alignas(32) float s2_64[64];
  for (int i = 0; i < 64; ++i) { const float s = 16.0f * params.scale[i]; s2_64[i] = s * s; }

  float sum = 0.0f;
  const __m128i mask0f128 = _mm_set1_epi8(0x0F);
  for (int b = 0; b < 32; b += 16) {
    // Load 16 bytes chunk safely
    __m128i qchunk = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q4_query + b));
    __m128i dchunk = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q4_db    + b));

    // Extract low/high nibbles and widen to 16-bit
    __m128i q_lo8 = _mm_and_si128(qchunk, mask0f128);
    __m128i d_lo8 = _mm_and_si128(dchunk, mask0f128);
    __m128i q_hi8 = _mm_and_si128(_mm_srli_epi16(qchunk, 4), mask0f128);
    __m128i d_hi8 = _mm_and_si128(_mm_srli_epi16(dchunk, 4), mask0f128);

    __m256i q_lo_w = _mm256_cvtepu8_epi16(q_lo8);
    __m256i d_lo_w = _mm256_cvtepu8_epi16(d_lo8);
    __m256i q_hi_w = _mm256_cvtepu8_epi16(q_hi8);
    __m256i d_hi_w = _mm256_cvtepu8_epi16(d_hi8);

    // Interleave to get [lo0,hi0, lo1,hi1, ... lo7,hi7] for first 8 bytes (16 dims)
    __m256i q_pairs0 = _mm256_unpacklo_epi16(q_lo_w, q_hi_w);
    __m256i d_pairs0 = _mm256_unpacklo_epi16(d_lo_w, d_hi_w);
    __m256i diff0 = _mm256_sub_epi16(q_pairs0, d_pairs0);
    __m256i sq0 = _mm256_mullo_epi16(diff0, diff0);

    int base0 = 2 * b; // starting dim index for these 16 dims
    __m256 f0_lo = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(sq0)));
    __m256 f0_hi = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(sq0, 1)));
    __m256 ms00 = _mm256_load_ps(s2_64 + base0 + 0);
    __m256 ms01 = _mm256_load_ps(s2_64 + base0 + 8);
    f0_lo = _mm256_mul_ps(f0_lo, ms00);
    f0_hi = _mm256_mul_ps(f0_hi, ms01);

    // Interleave to get remaining 8 bytes (16 dims)
    __m256i q_pairs1 = _mm256_unpackhi_epi16(q_lo_w, q_hi_w);
    __m256i d_pairs1 = _mm256_unpackhi_epi16(d_lo_w, d_hi_w);
    __m256i diff1 = _mm256_sub_epi16(q_pairs1, d_pairs1);
    __m256i sq1 = _mm256_mullo_epi16(diff1, diff1);

    int base1 = base0 + 16;
    __m256 f1_lo = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(sq1)));
    __m256 f1_hi = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(sq1, 1)));
    __m256 ms10 = _mm256_load_ps(s2_64 + base1 + 0);
    __m256 ms11 = _mm256_load_ps(s2_64 + base1 + 8);
    f1_lo = _mm256_mul_ps(f1_lo, ms10);
    f1_hi = _mm256_mul_ps(f1_hi, ms11);

    __m256 acc0 = _mm256_add_ps(f0_lo, f0_hi);
    __m256 acc1 = _mm256_add_ps(f1_lo, f1_hi);
    alignas(32) float tmp0[8];
    alignas(32) float tmp1[8];
    _mm256_store_ps(tmp0, acc0);
    _mm256_store_ps(tmp1, acc1);
    for (int t = 0; t < 8; ++t) sum += tmp0[t] + tmp1[t];
  }

  return sum;
#else
  float sum = 0.0f;
  for (int i = 0; i < 32; ++i) {
    const std::uint8_t qq = q4_query[i];
    const std::uint8_t dd = q4_db[i];
    const int q_lo = static_cast<int>(qq & 0x0F);
    const int q_hi = static_cast<int>((qq >> 4) & 0x0F);
    const int d_lo = static_cast<int>(dd & 0x0F);
    const int d_hi = static_cast<int>((dd >> 4) & 0x0F);
    const int d0 = 2 * i;
    const int d1 = 2 * i + 1;
    const float s0 = params.scale[d0];
    const float s1 = params.scale[d1];
    const int diff0 = q_lo - d_lo;
    const int diff1 = q_hi - d_hi;
    sum += static_cast<float>(diff0 * diff0) * (256.0f * s0 * s0);
    sum += static_cast<float>(diff1 * diff1) * (256.0f * s1 * s1);
  }
  return sum;
#endif
}

float distance_q4_ade_avx2(const float zq[64],
                           const std::uint8_t* q4_db,
                           const CapqQ4Codebooks& cb) noexcept {
#if defined(__AVX2__) || (defined(_MSC_VER) && defined(__AVX2__)) || defined(__AVX2__)
  // Process 16 dims at a time. We need mu per dim (selected by code) and weights.
  float sum = 0.0f;
  for (int i = 0; i < 32; ++i) {
    const std::uint8_t dd = q4_db[i];
    const int c0 = static_cast<int>(dd & 0x0F);
    const int c1 = static_cast<int>((dd >> 4) & 0x0F);
    const int d0 = 2 * i;
    const int d1 = 2 * i + 1;
    const float mu0 = cb.centers[d0 * 16 + c0];
    const float mu1 = cb.centers[d1 * 16 + c1];
    const float e0 = zq[d0] - mu0;
    const float e1 = zq[d1] - mu1;
    sum += cb.weights[d0] * (e0 * e0) + cb.weights[d1] * (e1 * e1);
  }
  return sum;
#else
  return distance_q4_ade(zq, q4_db, cb);
#endif
}

} // namespace vesper::index


