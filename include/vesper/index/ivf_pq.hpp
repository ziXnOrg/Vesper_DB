#pragma once

/** \file ivf_pq.hpp
 *  \brief Inverted File with Product Quantization index for scalable vector search.
 *
 * IVF-PQ provides space-efficient indexing with Asymmetric Distance Computation (ADC).
 * Features:
 * - Coarse quantization via k-means clustering
 * - Product Quantization for compact vector encoding (8-32 bytes/vector)
 * - Optional OPQ rotation for improved quantization
 * - Fast ADC using precomputed lookup tables
 *
 * Thread-safety: Training is single-threaded; search operations are thread-safe.
 * Memory: O(nlist*d + m*ksub*dsub + N*m) where N is number of vectors.
 */

#include <cstdint>
#include <expected>
#include <memory>
#include <vesper/span_polyfill.hpp>
#include <vector>
#include <optional>

#include "vesper/error.hpp"

namespace vesper::index {

/** \brief Training parameters for IVF-PQ index. */
struct IvfPqTrainParams {
    std::uint32_t nlist{256};           /**< Number of coarse centroids */
    std::uint32_t m{8};                 /**< Number of subquantizers */
    std::uint32_t nbits{8};              /**< Bits per subquantizer (typically 8) */
    std::uint32_t max_iter{25};         /**< Max k-means iterations */
    float epsilon{1e-4f};                /**< Convergence threshold */
    bool verbose{false};                 /**< Training progress output */
    bool use_opq{false};                 /**< Enable OPQ rotation */
    std::uint32_t opq_iter{10};          /**< OPQ optimization iterations */
    std::uint32_t seed{42};              /**< Random seed for reproducibility */
};

/** \brief Search parameters for IVF-PQ index. */
struct IvfPqSearchParams {
    std::uint32_t nprobe{8};             /**< Number of cells to search */
    std::uint32_t k{10};                 /**< Number of neighbors to return */
    bool use_exact_rerank{false};       /**< Exact distance on shortlist */
    std::uint32_t rerank_k{0};          /**< Size of rerank shortlist (0=auto) */
};

/** \brief Statistics from training. */
struct IvfPqTrainStats {
    std::uint32_t iterations{0};         /**< Actual iterations performed */
    float final_error{0.0f};             /**< Final quantization error */
    float train_time_sec{0.0f};          /**< Training wall time */
    std::vector<float> errors;           /**< Error per iteration */
};

/** \brief Compact representation of a vector using PQ codes. */
struct PqCode {
    std::vector<std::uint8_t> codes;     /**< Subquantizer assignments [m] */
    
    auto size() const noexcept -> std::size_t { return codes.size(); }
    auto data() const noexcept -> const std::uint8_t* { return codes.data(); }
};

/** \brief IVF-PQ index for approximate nearest neighbor search.
 *
 * Combines inverted file structure with product quantization for
 * memory-efficient indexing of large vector collections.
 */
class IvfPqIndex {
public:
    IvfPqIndex();
    ~IvfPqIndex();
    IvfPqIndex(IvfPqIndex&&) noexcept;
    IvfPqIndex& operator=(IvfPqIndex&&) noexcept;
    IvfPqIndex(const IvfPqIndex&) = delete;
    IvfPqIndex& operator=(const IvfPqIndex&) = delete;

    /** \brief Train index on a sample of vectors.
     *
     * Learns coarse centroids and PQ codebooks from training data.
     * 
     * \param data Training vectors [n_train x dim]
     * \param dim Vector dimensionality
     * \param n Number of training vectors
     * \param params Training parameters
     * \return Training statistics or error
     *
     * Preconditions: n >= params.nlist; dim divisible by params.m
     * Complexity: O(nlist * n * dim * iterations)
     */
    auto train(const float* data, std::size_t dim, std::size_t n, 
               const IvfPqTrainParams& params) 
        -> std::expected<IvfPqTrainStats, core::error>;

    /** \brief Add vectors to the index.
     *
     * Assigns vectors to coarse centroids and encodes with PQ.
     *
     * \param ids Vector identifiers [n]
     * \param data Vectors to add [n x dim]
     * \param n Number of vectors
     * \return Success or error
     *
     * Preconditions: Index is trained; vectors have same dim as training
     * Complexity: O(n * (nlist * dim + m * ksub))
     */
    auto add(const std::uint64_t* ids, const float* data, std::size_t n)
        -> std::expected<void, core::error>;

    /** \brief Search for k nearest neighbors.
     *
     * Performs multi-probe search using ADC for fast distance computation.
     *
     * \param query Query vector [dim]
     * \param params Search parameters
     * \return Vector IDs and distances of k nearest neighbors
     *
     * Preconditions: Index is trained and non-empty
     * Complexity: O(nprobe * (dim + avg_list_size * m))
     * Thread-safety: Safe for concurrent calls
     */
    auto search(const float* query, const IvfPqSearchParams& params) const
        -> std::expected<std::vector<std::pair<std::uint64_t, float>>, core::error>;

    /** \brief Batch search for multiple queries.
     *
     * \param queries Query vectors [n_queries x dim]
     * \param n_queries Number of queries
     * \param params Search parameters
     * \return Results for each query
     *
     * Thread-safety: Internally parallelized
     */
    auto search_batch(const float* queries, std::size_t n_queries,
                      const IvfPqSearchParams& params) const
        -> std::expected<std::vector<std::vector<std::pair<std::uint64_t, float>>>, core::error>;

    /** \brief Get index statistics. */
    struct Stats {
        std::size_t n_vectors{0};        /**< Total indexed vectors */
        std::size_t n_lists{0};          /**< Number of inverted lists */
        std::size_t m{0};                /**< Number of subquantizers */
        std::size_t code_size{0};        /**< Bytes per PQ code */
        std::size_t memory_bytes{0};     /**< Total memory usage */
        float avg_list_size{0.0f};       /**< Average vectors per list */
    };
    
    auto get_stats() const noexcept -> Stats;

    /** \brief Check if index is trained. */
    auto is_trained() const noexcept -> bool;

    /** \brief Get vector dimensionality. */
    auto dimension() const noexcept -> std::size_t;

    /** \brief Clear all indexed vectors (keeps training). */
    auto clear() -> void;

    /** \brief Reset index completely (requires retraining). */
    auto reset() -> void;

    /** \brief Serialize index to file.
     *
     * \param path Output file path
     * \return Success or error
     */
    auto save(const std::string& path) const -> std::expected<void, core::error>;

    /** \brief Load index from file.
     *
     * \param path Input file path
     * \return Loaded index or error
     */
    static auto load(const std::string& path) -> std::expected<IvfPqIndex, core::error>;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/** \brief Compute recall of index against ground truth.
 *
 * \param index Trained and populated index
 * \param queries Test queries [n_queries x dim]
 * \param n_queries Number of queries
 * \param ground_truth True nearest neighbors [n_queries x k]
 * \param k Number of neighbors to evaluate
 * \param params Search parameters
 * \return Recall@k in [0, 1]
 */
auto compute_recall(const IvfPqIndex& index,
                   const float* queries, std::size_t n_queries,
                   const std::uint64_t* ground_truth, std::size_t k,
                   const IvfPqSearchParams& params) -> float;

} // namespace vesper::index