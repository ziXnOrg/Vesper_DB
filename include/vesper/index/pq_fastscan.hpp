#pragma once

/** \file pq_fastscan.hpp
 *  \brief FastScan Product Quantization with SIMD-friendly layout.
 *
 * Based on Faiss's FastScan PQ, reorganizes PQ codes in blocks for
 * efficient SIMD distance computation using lookup tables.
 *
 * Key innovations:
 * - Block-based code organization (16 or 32 codes per block)
 * - Transposed storage for coalesced memory access
 * - SIMD lookup table accumulation
 * - Prefetching and cache optimization
 *
 * Performance: 2-4x faster than standard PQ on modern CPUs.
 */

#include <cstdint>
#include <memory>
#include <vesper/span_polyfill.hpp>
#include <vector>
#include <algorithm>
#include <expected>
#ifdef __x86_64__
#include <immintrin.h>
#endif

#include "vesper/error.hpp"
#include "vesper/index/aligned_buffer.hpp"

namespace vesper::index {

/** \brief FastScan PQ configuration. */
struct FastScanPqConfig {
    std::uint32_t m{8};                  /**< Number of subquantizers */
    std::uint32_t nbits{8};              /**< Bits per subquantizer */
    std::uint32_t block_size{32};        /**< Codes per block (16 or 32) */
    bool use_avx512{false};              /**< Enable AVX-512 if available */
};

/** \brief Block of PQ codes with transposed layout.
 *
 * Stores codes in transposed format for SIMD processing:
 * - Standard: codes[vector_id][subquantizer_id]
 * - Transposed: codes[subquantizer_id][vector_id_in_block]
 */
class PqCodeBlock {
public:
    static constexpr std::size_t MAX_BLOCK_SIZE = 32;
    static constexpr std::size_t CACHE_LINE = 64;

    PqCodeBlock(std::uint32_t m, std::uint32_t block_size)
        : m_(m)
        , block_size_(block_size)
        , n_codes_(0)
        , codes_(m * MAX_BLOCK_SIZE) {
        clear();
    }

    /** \brief Add a PQ code to the block. */
    auto add_code(const std::uint8_t* code) -> bool {
        if (n_codes_ >= block_size_) return false;

        for (std::uint32_t sub = 0; sub < m_; ++sub) {
            codes_[sub * MAX_BLOCK_SIZE + n_codes_] = code[sub];
        }
        n_codes_++;
        return true;
    }

    /** \brief Get pointer to codes for subquantizer. */
    [[nodiscard]] auto get_subquantizer_codes(std::uint32_t sub) const noexcept
        -> const std::uint8_t* {
        return codes_.data() + sub * MAX_BLOCK_SIZE;
    }

    /** \brief Check if block is full. */
    [[nodiscard]] auto is_full() const noexcept -> bool {
        return n_codes_ >= block_size_;
    }

    /** \brief Get number of codes in block. */
    [[nodiscard]] auto size() const noexcept -> std::uint32_t {
        return n_codes_;
    }

    /** \brief Clear the block. */
    auto clear() noexcept -> void {
        n_codes_ = 0;
        std::fill(codes_.begin(), codes_.end(), 0);
    }

private:
    std::uint32_t m_;
    std::uint32_t block_size_;
    std::uint32_t n_codes_;
    std::vector<std::uint8_t, AlignedAllocator<std::uint8_t, CACHE_LINE>> codes_;
};

/** \brief FastScan Product Quantizer.
 *
 * Implements block-based PQ with SIMD-optimized distance computation.
 */
class FastScanPq {
public:
    FastScanPq(const FastScanPqConfig& config)
        : config_(config)
        , ksub_(1U << config.nbits)
        , trained_(false) {
    }

    /** \brief Train PQ codebooks on sample data.
     *
     * \param data Training vectors [n x dim]
     * \param n Number of training vectors
     * \param dim Vector dimensionality
     * \return Success or error
     */
    auto train(const float* data, std::size_t n, std::size_t dim)
        -> std::expected<void, core::error>;

    /** \brief Encode vectors into PQ codes.
     *
     * \param data Vectors to encode [n x dim]
     * \param n Number of vectors
     * \param[out] codes Output codes [n x m]
     */
    auto encode(const float* data, std::size_t n, std::uint8_t* codes) const -> void;

    /** \brief Encode vectors into blocks.
     *
     * \param data Vectors to encode [n x dim]
     * \param n Number of vectors
     * \return Vector of code blocks
     */
    auto encode_blocks(const float* data, std::size_t n)
        -> std::vector<PqCodeBlock>;

    /** \brief Decode PQ codes back to vectors.
     *
     * \param codes PQ codes [n x m]
     * \param n Number of codes
     * \param[out] data Output vectors [n x dim]
     */
    auto decode(const std::uint8_t* codes, std::size_t n, float* data) const -> void;

    /** \brief Compute distances using lookup tables (ADC).
     *
     * \param query Query vector [dim]
     * \param blocks Code blocks
     * \param[out] distances Output distances laid out with per-block stride = config.block_size.
     * \details Output layout: total_codes = blocks.size() * config.block_size. For each block b,
     *          only the first blocks[b].size() entries are valid; the remainder of the block's
     *          stride are padding. Callers must not consume padding entries.
     */
    auto compute_distances(const float* query,
                          const std::vector<PqCodeBlock>& blocks,
                          float* distances) const -> void;

    /** \brief Compute distances with AVX2 acceleration. */
    #ifdef __AVX2__
    auto compute_distances_avx2(const float* query,
                               const std::vector<PqCodeBlock>& blocks,
                               float* distances) const -> void;
    #endif

    /** \brief Compute distances with AVX-512 acceleration. */
    #ifdef __AVX512F__
    auto compute_distances_avx512(const float* query,
                                 const std::vector<PqCodeBlock>& blocks,
                                 float* distances) const -> void;
    #endif

    /** \brief Precompute lookup tables for a query.
     *
     * \param query Query vector [dim]
     * \return Lookup tables [m x ksub]
     */
    auto compute_lookup_tables(const float* query) const
        -> AlignedCentroidBuffer;

    /** \brief Get configuration. */
    [[nodiscard]] auto config() const noexcept -> const FastScanPqConfig& {
        return config_;
    }

    /** \brief Check if trained. */
    [[nodiscard]] auto is_trained() const noexcept -> bool {
        return trained_;
    }

    /** \brief Get dimension per subquantizer. */
    [[nodiscard]] auto dsub() const noexcept -> std::size_t {
        return dsub_;
    }

    /** \brief Get total dimension. */
    [[nodiscard]] auto dimension() const noexcept -> std::size_t {
        return config_.m * dsub_;
    }

    /** \brief Export codebooks as a dense row-major array [m*ksub x dsub]. */
    auto export_codebooks(std::vector<float>& out) const -> void;
    /** \brief Import pre-trained codebooks and mark as trained. */
    auto import_pretrained(std::size_t dsub, std::span<const float> data) -> void;
    /** \brief Get ksub (codes per subquantizer). */
    [[nodiscard]] auto ksub() const noexcept -> std::uint32_t { return ksub_; }

private:
    FastScanPqConfig config_;
    std::uint32_t ksub_;                        /**< Codes per subquantizer */
    std::size_t dsub_{0};                       /**< Dimension per subquantizer */
    bool trained_;

    /** Codebooks [m][ksub][dsub] in aligned buffer */
    std::unique_ptr<AlignedCentroidBuffer> codebooks_;

    /** \brief Train a single subquantizer. */
    auto train_subquantizer(const float* data, std::size_t n,
                           std::uint32_t sub_idx) -> void;

    /** \brief Find nearest codebook entry. */
    auto find_nearest_code(const float* vec, std::uint32_t sub_idx) const
        -> std::uint8_t;
};

/** \brief SIMD-optimized distance computation kernel. */
#ifdef __AVX2__
inline auto accumulate_distances_avx2(
    const std::uint8_t* codes,
    const float* lut,
    std::uint32_t n_codes,
    float* distances) -> void {

    // Process 8 codes at a time
    std::uint32_t i = 0;
    for (; i + 8 <= n_codes; i += 8) {
        // Load 8 code bytes
        const __m128i code_bytes = _mm_loadl_epi64(
            reinterpret_cast<const __m128i*>(codes + i));

        // Convert to 32-bit integers for gather
        const __m256i indices = _mm256_cvtepu8_epi32(code_bytes);

        // Gather distances from lookup table
        const __m256 dists = _mm256_i32gather_ps(lut, indices, 4);

        // Load current accumulated distances
        __m256 acc = _mm256_load_ps(distances + i);

        // Accumulate
        acc = _mm256_add_ps(acc, dists);

        // Store back
        _mm256_store_ps(distances + i, acc);
    }

    // Handle remainder
    for (; i < n_codes; ++i) {
        distances[i] += lut[codes[i]];
    }
}
#endif

/** \brief Compute ADC distances for a batch of queries.
 *
 * \param pq Trained FastScan PQ
 * \param queries Query vectors [n_queries x dim]
 * \param n_queries Number of queries
 * \param blocks Code blocks
 * \param[out] distances Output [n_queries x total_codes], where total_codes = blocks.size() * pq.config().block_size
 * \details Output layout matches compute_distances(): fixed per-block stride = config.block_size; only the first
 *          blocks[b].size() entries per block are valid; remainder of each block's stride is padding.
 */
inline auto compute_batch_distances(
    const FastScanPq& pq,
    const float* queries,
    std::size_t n_queries,
    const std::vector<PqCodeBlock>& blocks,
    float* distances) -> void {

    // Early exit on empty inputs to avoid dereferencing blocks[0] and to define no-op semantics
    if (n_queries == 0 || blocks.empty()) {
        return;
    }
    const std::size_t total_codes = blocks.size() * pq.config().block_size;

    #pragma omp parallel for
    for (int q = 0; q < static_cast<int>(n_queries); ++q) {
        const float* query = queries + q * pq.dimension();
        float* query_dists = distances + static_cast<std::size_t>(q) * total_codes;
        pq.compute_distances(query, blocks, query_dists);
    }
}

} // namespace vesper::index
