#pragma once

/** \file pq_simple.hpp
 *  \brief Simple, portable Product Quantization implementation.
 *
 * Platform-agnostic PQ implementation without SIMD optimizations.
 * Provides the same interface as FastScanPq but focuses on portability.
 */

#include <cstdint>
#include <memory>
#include <vesper/span_polyfill.hpp>
#include <vector>
#include <expected>

#include "vesper/error.hpp"
#include "vesper/index/aligned_buffer.hpp"

namespace vesper::index {

/** \brief FastScan PQ configuration (duplicated for compatibility). */
struct FastScanPqConfig {
    std::uint32_t m{8};                  /**< Number of subquantizers */
    std::uint32_t nbits{8};              /**< Bits per subquantizer */
    std::uint32_t block_size{32};        /**< Codes per block (ignored in simple impl) */
    bool use_avx512{false};              /**< Enable AVX-512 (ignored in simple impl) */
};

/** \brief Simple Product Quantization implementation.
 *
 * Portable implementation that works on all platforms including ARM.
 * Trades some performance for compatibility.
 */
class SimplePq {
public:
    SimplePq(const FastScanPqConfig& config);
    ~SimplePq();
    
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
    
    /** \brief Compute lookup tables for asymmetric distance computation.
     *
     * \param query Query vector [dim]
     * \return Lookup tables [m x ksub]
     */
    auto compute_lookup_tables(const float* query) const -> AlignedCentroidBuffer;
    
    /** \brief Compute asymmetric distance using lookup tables.
     *
     * \param codes PQ codes [m]
     * \param tables Precomputed lookup tables [m x ksub]
     * \return Approximated distance
     */
    auto compute_distance(const std::uint8_t* codes, const float* tables) const -> float;
    
    /** \brief Check if PQ is trained. */
    auto is_trained() const noexcept -> bool { return trained_; }
    
    /** \brief Get number of subquantizers. */
    auto m() const noexcept -> std::uint32_t { return m_; }
    
    /** \brief Get codebook size. */
    auto ksub() const noexcept -> std::uint32_t { return ksub_; }
    
private:
    std::uint32_t m_;           /**< Number of subquantizers */
    std::uint32_t nbits_;       /**< Bits per subquantizer */
    std::uint32_t ksub_;        /**< Codebook size (2^nbits) */
    std::size_t dsub_;          /**< Dimension per subquantizer */
    bool trained_;              /**< Training status */
    
    /** \brief Codebooks [m x ksub x dsub] */
    std::unique_ptr<AlignedCentroidBuffer> codebooks_;
};

// Define FastScanPq as SimplePq for ARM platforms
#ifndef __x86_64__
using FastScanPq = SimplePq;
#endif

} // namespace vesper::index