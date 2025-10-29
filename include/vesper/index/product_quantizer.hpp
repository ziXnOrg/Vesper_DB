#pragma once

/** \file product_quantizer.hpp
 *  \brief Product Quantization for compact vector encoding.
 *
 * Decomposes high-dimensional vectors into subspaces and quantizes each
 * independently. Enables fast approximate distance computation via lookup tables.
 *
 * Features:
 * - Configurable subspace partitioning (m subquantizers)
 * - 8-bit codes for memory efficiency
 * - Asymmetric Distance Computation (ADC) with precomputed tables
 * - Optional OPQ rotation for improved quantization
 *
 * Thread-safety: Training is single-threaded; encoding/decoding are thread-safe.
 * Memory: O(m * ksub * dsub) for codebooks, O(m) bytes per encoded vector.
 */
/**
 * ABI note: The public C++ API in this header uses STL types (e.g., std::string)
 * and std::expected in function signatures. These are not guaranteed ABI-stable
 * across compilers, standard libraries, or differing CRT/ABI versions. As a result,
 * the C++ API is intended for in-process use with the same toolchain. For cross-DSO
 * or FFI/plugin boundaries, prefer the stable C API under include/vesper/c/.
 */

#include <cstdint>
#include <expected>
#include <vesper/span_polyfill.hpp>
#include <vector>
#include <memory>
#include <optional>

#include <filesystem>
#include "vesper/error.hpp"

namespace vesper::index {

/** \brief Product Quantizer training parameters. */
struct PqTrainParams {
    std::uint32_t m{8};                  /**< Number of subquantizers */
    std::uint32_t nbits{8};              /**< Bits per subquantizer (typically 8) */
    std::uint32_t max_iter{25};          /**< K-means iterations per subspace */
    float epsilon{1e-4f};                 /**< Convergence threshold */
    std::uint32_t seed{42};              /**< Random seed */
    bool verbose{false};                  /**< Training progress */
};

/** \brief OPQ (Optimized Product Quantization) parameters. */
struct OpqParams {
    std::uint32_t iter{10};               /**< Alternating optimization iterations */
    bool init_rotation{true};             /**< Initialize with PCA rotation */
    float reg{0.01f};                     /**< Regularization for rotation */
};

/** \brief Product Quantizer for vector compression.
 *
 * Divides d-dimensional vectors into m subspaces of dimension d/m,
 * learning a codebook for each subspace via k-means.
 */
class ProductQuantizer {
public:
    ProductQuantizer();
    ~ProductQuantizer();
    ProductQuantizer(ProductQuantizer&&) noexcept;
    ProductQuantizer& operator=(ProductQuantizer&&) noexcept;
    ProductQuantizer(const ProductQuantizer&) = delete;
    ProductQuantizer& operator=(const ProductQuantizer&) = delete;

    /** \brief Train product quantizer on data.
     *
     * Learns codebooks for each subspace using k-means.
     *
     * \param data Training vectors [n x dim]
     * \param n Number of training vectors
     * \param dim Vector dimensionality
     * \param params Training parameters
     * \return Success or error
     *
     * Preconditions: dim divisible by m; n >= ksub (2^nbits)
     * Complexity: O(n * m * ksub * dsub * iterations)
     */
    auto train(const float* data, std::size_t n, std::size_t dim,
               const PqTrainParams& params)
        -> std::expected<void, core::error>;

    /** \brief Train with OPQ rotation optimization.
     *
     * Jointly learns rotation matrix and codebooks to minimize quantization error.
     *
     * \param data Training vectors [n x dim]
     * \param n Number of training vectors
     * \param dim Vector dimensionality
     * \param pq_params PQ training parameters
     * \param opq_params OPQ parameters
     * \return Success or error
     *
     * Complexity: O(opq_iter * (n * dim^2 + PQ training))
     */
    auto train_opq(const float* data, std::size_t n, std::size_t dim,
                   const PqTrainParams& pq_params, const OpqParams& opq_params)
        -> std::expected<void, core::error>;

    /** \brief Encode vectors to PQ codes.
     *
     * \param data Vectors to encode [n x dim]
     * \param n Number of vectors
     * \param codes Output codes [n x m]
     * \return Success or error
     *
     * Preconditions: Quantizer is trained
     * Complexity: O(n * m * ksub * dsub)
     * Thread-safety: Safe for concurrent calls
     */
    auto encode(const float* data, std::size_t n, std::uint8_t* codes) const
        -> std::expected<void, core::error>;

    /** \brief Encode single vector.
     *
     * \param vec Vector to encode [dim]
     * \param code Output code [m]
     * \return Success or error
     */
    auto encode_one(const float* vec, std::uint8_t* code) const
        -> std::expected<void, core::error>;

    /** \brief Decode PQ codes to vectors.
     *
     * Reconstructs approximate vectors from codes.
     *
     * \param codes PQ codes [n x m]
     * \param n Number of codes
     * \param data Output vectors [n x dim]
     * \return Success or error
     *
     * Complexity: O(n * m * dsub)
     */
    auto decode(const std::uint8_t* codes, std::size_t n, float* data) const
        -> std::expected<void, core::error>;

    /** \brief Compute distance table for query.
     *
     * Precomputes distances from query to all codebook entries
     * for Asymmetric Distance Computation (ADC).
     *
     * \param query Query vector [dim]
     * \param table Output distance table [m x ksub]
     * \return Success or error
     *
     * Complexity: O(m * ksub * dsub)
     * Thread-safety: Safe for concurrent calls
     */
    auto compute_distance_table(const float* query, float* table) const
        -> std::expected<void, core::error>;

    /** \brief Compute distances using ADC.
     *
     * Fast distance computation using precomputed query table.
     *
     * \param table Distance table from compute_distance_table [m x ksub]
     * \param codes PQ codes [n x m]
     * \param n Number of codes
     * \param distances Output distances [n]
     * \return Success or error
     *
     * Complexity: O(n * m)
     * Thread-safety: Safe for concurrent calls
     */
    auto compute_distances_adc(const float* table, const std::uint8_t* codes,
                               std::size_t n, float* distances) const
        -> std::expected<void, core::error>;

    /** \brief Symmetric distance between PQ codes.
     *
     * Computes approximate distance between two encoded vectors.
     *
     * \param code1 First PQ code [m]
     * \param code2 Second PQ code [m]
     * \return Approximate L2 distance
     *
     * Complexity: O(m * dsub)
     */
    auto compute_distance_symmetric(const std::uint8_t* code1,
                                    const std::uint8_t* code2) const -> float;

    /** \brief Get quantizer parameters. */
    struct Info {
        std::uint32_t m{0};              /**< Number of subquantizers */
        std::uint32_t ksub{0};           /**< Codebook size per subquantizer */
        std::uint32_t dsub{0};           /**< Subspace dimension */
        std::size_t dim{0};              /**< Total dimension */
        bool has_rotation{false};        /**< OPQ rotation applied */
    };

    auto get_info() const noexcept -> Info;

    /** \brief Check if quantizer is trained. */
    auto is_trained() const noexcept -> bool;

    /** \brief Get code size in bytes. */
    auto code_size() const noexcept -> std::size_t;

    /** \brief Compute quantization error on data.
     *
     * \param data Vectors [n x dim]
     * \param n Number of vectors
     * \return Mean squared quantization error
     */
    auto compute_quantization_error(const float* data, std::size_t n) const -> float;

    /** \brief Save quantizer to file. */
    auto save(const std::string& path) const -> std::expected<void, core::error>;

    /** \brief Load quantizer from file. */
    static auto load(const std::string& path)
        -> std::expected<ProductQuantizer, core::error>;

    /** \brief Save quantizer to file (filesystem path overload).
     *
     * Provided for ergonomics. ABI guidance above still applies.
     */
    auto save(const std::filesystem::path& path) const -> std::expected<void, core::error>;

    /** \brief Load quantizer from file (filesystem path overload). */
    static auto load(const std::filesystem::path& path)
        -> std::expected<ProductQuantizer, core::error>;


private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/** \brief Compute recall of PQ approximation.
 *
 * Measures how well PQ preserves nearest neighbor relationships.
 *
 * \param pq Trained product quantizer
 * \param data Vectors [n x dim]
 * \param n Number of vectors
 * \param queries Query vectors [nq x dim]
 * \param nq Number of queries
 * \param k Number of neighbors
 * \return Recall@k in [0, 1]
 */
auto compute_pq_recall(const ProductQuantizer& pq,
                      const float* data, std::size_t n,
                      const float* queries, std::size_t nq,
                      std::size_t k) -> float;

} // namespace vesper::index
