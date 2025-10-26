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

#include <string>
#include <string_view>
#include <functional>

#include "vesper/error.hpp"
#ifdef VESPER_NO_ROARING
struct roaring_bitmap_t; // forward-declare to avoid hard dependency during fuzz builds
#else
#include <roaring/roaring.h>
#endif

#include "vesper/index/kmeans_elkan.hpp"

namespace vesper::index {

// Serialization format versions
inline constexpr std::uint16_t IVFPQ_VER_MAJOR = 1;
inline constexpr std::uint16_t IVFPQ_VER_MINOR_V10 = 0; // legacy monolithic layout
inline constexpr std::uint16_t IVFPQ_VER_MINOR_V11 = 1; // sectioned layout (optional zstd + mmap)

// Optional load mode is selected via environment variable VESPER_IVFPQ_LOAD_MMAP ("1" -> mmap when possible)
// Save v1.1 is gated by VESPER_IVFPQ_SAVE_V11 ("1" -> write sectioned format); default remains v1.0.


/** \brief OPQ initialization policy. */
enum class OpqInit { Identity, PCA };

/** \brief Coarse assigner selection for centroid assignment. */
enum class CoarseAssigner { Brute, HNSW, KDTree, Projection };

    /** \brief KD-tree split heuristic for centroid KD builder. */
    enum class KdSplitHeuristic { Variance, BBoxExtent };


/** \brief Training parameters for IVF-PQ index. */
struct IvfPqTrainParams {
    std::uint32_t nlist{256};           /**< Number of coarse centroids */
    std::uint32_t m{8};                 /**< Number of subquantizers */
    std::uint32_t nbits{8};             /**< Bits per subquantizer (typically 8) */
    std::uint32_t max_iter{25};         /**< Max k-means iterations */
    float epsilon{1e-4f};               /**< Convergence threshold */
    bool verbose{false};                /**< Training progress output */
    bool use_opq{true};                 /**< Enable OPQ rotation (default ON for better recall) */
    std::uint32_t opq_iter{10};         /**< OPQ optimization iterations */
    std::size_t opq_sample_n{20000};    /**< Max samples for OPQ alternations (0=auto) */
    OpqInit opq_init{OpqInit::PCA};     /**< OPQ init: Identity or PCA (default PCA) */
    std::uint32_t seed{42};             /**< Random seed for reproducibility */

    // K-means initialization for coarse quantizer (Elkan)
    // Defaults preserve prior behavior (k-means++). k-means|| is recommended for large n (>10k)
    vesper::index::KmeansElkan::Config::InitMethod kmeans_init_method{vesper::index::KmeansElkan::Config::InitMethod::KMeansPlusPlus};
    std::uint32_t kmeans_parallel_rounds{5};          /**< k-means|| sampling rounds (Bahmani et al., 2012) */
    std::uint32_t kmeans_parallel_oversampling{0};    /**< Oversampling factor l (0 => 2*k) */

    // RaBitQ integration
    bool use_rabitq{false};             /**< Use RaBitQ instead of standard PQ */
    std::uint8_t rabitq_bits{1};        /**< Bits for RaBitQ (1, 4, or 8) */
    bool rabitq_rotation{true};         /**< Apply rotation in RaBitQ */

    // Coarse assignment selection (default KDTree for fast exact assignment)
    CoarseAssigner coarse_assigner{CoarseAssigner::KDTree};

    // ANN-based coarse assignment over centroids (when coarse_assigner==HNSW)
    bool use_centroid_ann{true};               /**< Use alternative coarse assigner (ann mode). Auto-disabled for nlist < 1024 */
    std::uint32_t centroid_ann_ef_search{96};  /**< HNSW efSearch during assignment (default 96) */
    std::uint32_t centroid_ann_ef_construction{200}; /**< HNSW efConstruction during build (default 200) */
    std::uint32_t centroid_ann_M{16};          /**< HNSW connectivity (M) during build (default 16) */
    bool validate_ann_assignment{false};       /**< Sampled correctness checks vs brute-force */
    float validate_ann_sample_rate{0.0f};      /**< Fraction [0,1] of points to validate during add() */
    std::uint32_t centroid_ann_refine_k{96};   /**< Top-L candidates to refine with exact L2 (default 96) */

    // Projection-based coarse assignment parameters (when coarse_assigner==Projection)
    std::uint32_t projection_dim{16};          /**< Dimensionality of projection (e.g., 16 for d=128) */

	    // KD-tree coarse assignment tuning (when coarse_assigner==KDTree)
	    std::uint32_t kd_leaf_size{256};           /**< Target max leaf size during KD build (env VESPER_KD_LEAF_SIZE overrides) */
	    bool kd_batch_assign{true};                /**< Use batched KD assignment path by default (env VESPER_KD_BATCH overrides if set) */
	    KdSplitHeuristic kd_split{KdSplitHeuristic::Variance}; /**< Split heuristic: Variance (default) or BBoxExtent */


    // Optional instrumentation
    bool timings_enabled{false};               /**< If true (or if verbose), record add() timings in Stats */
};

/** \brief Search parameters for IVF-PQ index. */
struct IvfPqSearchParams {
    std::uint32_t nprobe{8};             /**< Number of cells to search */
    std::uint32_t k{10};                 /**< Number of neighbors to return */
    // Candidate pool size produced by IVF-PQ before final selection; 0 => use k
    std::uint32_t cand_k{0};
    // Optional exact rerank on a shortlist (computed after IVF-PQ candidate collection)
    bool use_exact_rerank{false};
    std::uint32_t rerank_k{0};           /**< Size of rerank shortlist (0=auto) */
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

    // Optional user metadata (JSON blob) serialized with the index
    // Unchecked setter (legacy): assigns as-is; save/load enforce size/structural limits.
    void set_metadata_json(std::string_view json);
    auto get_metadata_json() const -> std::string;

    // Checked setter: enforces 64 KiB size, UTF-8, and structural caps (depth<=64, keys<=4096).
    // Applies optional schema validator if installed via set_metadata_validator().
    using MetadataValidator = std::function<std::expected<void, core::error>(std::string_view)>;
    auto set_metadata_json_checked(std::string_view json) -> std::expected<void, core::error>;
    void set_metadata_validator(MetadataValidator validator);

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
        // ANN coarse-assignment telemetry
        bool ann_enabled{false};
        std::uint64_t ann_assignments{0};
        std::uint64_t ann_validated{0};
        std::uint64_t ann_mismatches{0};
        // Optional timing telemetry (populated if verbose or timings_enabled)
        bool timings_enabled{false};
        double t_assign_ms{0.0};         /**< Time spent in coarse assignment (ms) */
        double t_encode_ms{0.0};         /**< Time spent in PQ encode() (ms) */
        double t_lists_ms{0.0};          /**< Time spent appending to inverted lists (ms) */
        // KD-tree traversal instrumentation (V1)
        std::uint64_t kd_nodes_pushed{0};
        std::uint64_t kd_nodes_popped{0};
        std::uint64_t kd_leaves_scanned{0};
        // KD timing breakdown (only populated when timings_enabled or VESPER_KD_STATS)
        double kd_traversal_ms{0.0};
        double kd_leaf_ms{0.0};
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

    /** \brief Reconstruct vectors from a specific cluster.
     *
     * Decodes PQ codes back to full vectors for a given cluster.
     *
     * \param cluster_id Cluster index
     * \param[out] ids Vector IDs in the cluster
     * \param[out] vectors Reconstructed vectors (flattened)
     * \return Success or error
     */
    auto reconstruct_cluster(std::uint32_t cluster_id,
                            std::vector<std::uint64_t>& ids,
                            std::vector<float>& vectors) const
        -> std::expected<void, core::error>;

    /** \brief Reconstruct a single vector by ID.
     *
     * \param id Vector ID
     * \return Reconstructed vector or error
     */
    auto reconstruct(std::uint64_t id) const
        -> std::expected<std::vector<float>, core::error>;

    /** \brief Get vector by ID (alias for reconstruct).
     *
     * \param id Vector ID
     * \return Reconstructed vector or error
     */
    auto get_vector(std::uint64_t id) const
        -> std::expected<std::vector<float>, core::error> {
        return reconstruct(id);
    }

    /** \brief Get number of clusters in the index.
     * \return Number of clusters or error
     */
    auto get_num_clusters() const -> std::expected<std::uint32_t, core::error>;

    /** \brief Get vector dimension.
     * \return Dimension or error
     */
    auto get_dimension() const -> std::expected<std::size_t, core::error>;

    /** \brief Get cluster assignment for a vector.
     * \param id Vector ID
     * \return Cluster ID or error
     */
    auto get_cluster_assignment(std::uint64_t id) const
        -> std::expected<std::uint32_t, core::error>;

    /** \brief Get cluster centroid.
     * \param cluster_id Cluster ID
     * \return Centroid vector or error
     */
    auto get_cluster_centroid(std::uint32_t cluster_id) const
        -> std::expected<std::vector<float>, core::error>;

    /** \brief Update cluster centroid.
     * \param cluster_id Cluster ID
     * \param centroid New centroid data
     * \return Success or error
     */
    auto update_cluster_centroid(std::uint32_t cluster_id, const float* centroid)
        -> std::expected<void, core::error>;

    /** \brief Reassign vector to different cluster.
     * \param id Vector ID
     * \param new_cluster New cluster ID
     * \return Success or error
     */
    auto reassign_vector(std::uint64_t id, std::uint32_t new_cluster)
        -> std::expected<void, core::error>;

    /** \brief Compact inverted list by removing deleted entries.
     * \param cluster_id Cluster ID
     * \param deleted_ids Set of deleted IDs
     * \return Success or error
     */
#ifndef VESPER_NO_ROARING
    auto compact_inverted_list(std::uint32_t cluster_id, const roaring_bitmap_t* deleted_ids)
        -> std::expected<void, core::error>;
#endif

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


    /** \brief Debug-only: compute ADC global rank and distance for a specific ID.
     *
     * Requires VESPER_IVFPQ_DEBUG=1 at runtime. Probes all lists and computes the
     * ADC distance for every code to determine the global rank of target_id.
     * Intended for diagnostics and tests only.
     *
     * \param query Query vector [dim]
     * \param target_id ID whose ADC rank to compute
     * \return pair<rank (1-based), adc_distance> or error
     */
    auto debug_explain_adc_rank(const float* query, std::uint64_t target_id) const
        -> std::expected<std::pair<std::size_t, float>, core::error>;

    /** \brief Debug-only: compute centroid rank (1-based) of the assigned centroid
     * for target_id among all coarse centroids for this query.
     *
     * Requires VESPER_IVFPQ_DEBUG=1 at runtime. Computes L2 distances from the
     * query to all nlist centroids, ranks the centroid that holds target_id,
     * and returns (centroid_rank_1based, assigned_centroid_id).
     */
    auto debug_explain_centroid_rank(const float* query, std::uint64_t target_id) const
        -> std::expected<std::pair<std::size_t, std::uint32_t>, core::error>;


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
