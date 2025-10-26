/** \file rabitq_quantizer.hpp
 *  \brief RaBitQ-inspired binary quantization for extreme memory efficiency
 *
 * Implements rotation-based binary quantization achieving 32x compression
 * with minimal accuracy loss. Based on the RaBitQ paper (SIGMOD 2024/2025).
 * 
 * Features:
 * - Random rotation for better quantization boundaries
 * - 1-bit, 4-bit, and 8-bit quantization modes
 * - Asymmetric distance computation (ADC)
 * - SIMD-optimized Hamming distance
 */

#pragma once

#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <vector>
#include <optional>

#include "vesper/error.hpp"

namespace vesper::index {

/** \brief Quantization bit width options */
enum class QuantizationBits : std::uint8_t {
    BIT_1 = 1,   /**< 1-bit binary quantization (32x compression) */
    BIT_4 = 4,   /**< 4-bit quantization (8x compression) */
    BIT_8 = 8    /**< 8-bit quantization (4x compression) */
};

/** \brief RaBitQ training parameters */
struct RaBitQTrainParams {
    QuantizationBits bits{QuantizationBits::BIT_1};  /**< Quantization bit width */
    std::uint32_t seed{42};                          /**< Random seed for rotation matrix */
    bool use_rotation{true};                         /**< Apply random rotation */
    std::uint32_t num_iterations{10};                /**< Training iterations for threshold */
    float balance_factor{0.5f};                      /**< Balance factor for bit distribution */
    bool normalize{true};                            /**< Normalize vectors before quantization */
};

/** \brief Quantized vector representation */
struct QuantizedVector {
    std::vector<std::uint8_t> codes;  /**< Packed binary/scalar codes */
    float scale{1.0f};                 /**< Scale factor for reconstruction */
    float offset{0.0f};                /**< Offset for reconstruction */
    std::uint32_t dimension{0};        /**< Original dimension */
    QuantizationBits bits{QuantizationBits::BIT_1};
};

/** \brief Statistics for quantization quality */
struct QuantizationStats {
    float mean_squared_error{0.0f};      /**< MSE between original and reconstructed */
    float max_error{0.0f};               /**< Maximum reconstruction error */
    float bit_balance{0.5f};             /**< Balance of 0s and 1s (for binary) */
    float compression_ratio{1.0f};       /**< Actual compression achieved */
    std::size_t memory_bytes{0};         /**< Memory usage after quantization */
};

/** \brief RaBitQ-inspired quantizer for vector compression
 *
 * Implements rotation-based quantization with theoretical error bounds.
 * Supports multiple bit widths and optimized distance computation.
 */
class RaBitQuantizer {
public:
    RaBitQuantizer();
    ~RaBitQuantizer();
    RaBitQuantizer(RaBitQuantizer&&) noexcept;
    RaBitQuantizer& operator=(RaBitQuantizer&&) noexcept;
    RaBitQuantizer(const RaBitQuantizer&) = delete;
    RaBitQuantizer& operator=(const RaBitQuantizer&) = delete;
    
    /** \brief Train quantizer on data
     *
     * \param data Training vectors [n x dim]
     * \param n Number of vectors
     * \param dim Vector dimension
     * \param params Training parameters
     * \return Success or error
     *
     * Learns rotation matrix and quantization thresholds.
     */
    auto train(const float* data, std::size_t n, std::size_t dim,
               const RaBitQTrainParams& params = {})
        -> std::expected<void, core::error>;
    
    /** \brief Quantize a single vector
     *
     * \param vec Input vector [dim]
     * \return Quantized representation
     *
     * Applies rotation and quantization based on trained parameters.
     */
    auto quantize(std::span<const float> vec) const
        -> std::expected<QuantizedVector, core::error>;
    
    /** \brief Batch quantize vectors
     *
     * \param data Vectors to quantize [n x dim]
     * \param n Number of vectors
     * \return Quantized vectors
     */
    auto quantize_batch(const float* data, std::size_t n) const
        -> std::expected<std::vector<QuantizedVector>, core::error>;
    
    /** \brief Reconstruct vector from quantized form
     *
     * \param qvec Quantized vector
     * \return Reconstructed vector
     */
    auto reconstruct(const QuantizedVector& qvec) const
        -> std::expected<std::vector<float>, core::error>;
    
    /** \brief Compute distance between quantized vectors
     *
     * \param a First quantized vector
     * \param b Second quantized vector
     * \return Approximate distance
     *
     * Uses optimized Hamming distance for binary quantization.
     */
    auto distance(const QuantizedVector& a, const QuantizedVector& b) const
        -> float;
    
    /** \brief Asymmetric distance: float query vs quantized vector
     *
     * \param query Float query vector [dim]
     * \param qvec Quantized database vector
     * \return Approximate distance
     *
     * More accurate than symmetric quantized distance.
     */
    auto asymmetric_distance(std::span<const float> query,
                            const QuantizedVector& qvec) const
        -> float;
    
    /** \brief Batch asymmetric distance computation
     *
     * \param query Query vector [dim]
     * \param qvecs Quantized vectors
     * \param k Number of nearest to return
     * \return Indices and distances of k-nearest
     */
    auto search_batch(std::span<const float> query,
                     const std::vector<QuantizedVector>& qvecs,
                     std::size_t k) const
        -> std::vector<std::pair<std::size_t, float>>;
    
    /** \brief Get quantization statistics
     *
     * \param original Original vectors [n x dim]
     * \param quantized Quantized vectors
     * \param n Number of vectors
     * \return Quality statistics
     */
    auto compute_stats(const float* original,
                      const std::vector<QuantizedVector>& quantized,
                      std::size_t n) const
        -> QuantizationStats;
    
    /** \brief Check if quantizer is trained */
    auto is_trained() const noexcept -> bool;
    
    /** \brief Get vector dimension */
    auto dimension() const noexcept -> std::size_t;
    
    /** \brief Get quantization bit width */
    auto bit_width() const noexcept -> QuantizationBits;
    
    /** \brief Save quantizer to file
     *
     * \param path Output file path
     * \return Success or error
     */
    auto save(const std::string& path) const
        -> std::expected<void, core::error>;
    
    /** \brief Load quantizer from file
     *
     * \param path Input file path
     * \return Loaded quantizer or error
     */
    static auto load(const std::string& path)
        -> std::expected<RaBitQuantizer, core::error>;
    
    /** \brief Get memory usage for n vectors */
    auto estimate_memory(std::size_t n) const noexcept -> std::size_t;
    
    /** \brief Get theoretical error bound */
    auto error_bound() const noexcept -> float;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/** \brief Optimized binary quantization operations */
namespace binary_ops {
    /** \brief Compute Hamming distance between binary vectors */
    auto hamming_distance(std::span<const std::uint8_t> a,
                         std::span<const std::uint8_t> b) noexcept -> std::uint32_t;
    
    /** \brief Population count (number of 1 bits) */
    auto popcount(std::span<const std::uint8_t> bits) noexcept -> std::uint32_t;
    
    /** \brief Pack float vector to binary */
    auto pack_binary(std::span<const float> vec, float threshold = 0.0f)
        -> std::vector<std::uint8_t>;
    
    /** \brief Unpack binary to float approximation */
    auto unpack_binary(std::span<const std::uint8_t> bits, std::size_t dim)
        -> std::vector<float>;
}

} // namespace vesper::index