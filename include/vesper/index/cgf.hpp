#pragma once

/** \file cgf.hpp
 *  \brief Cascaded Geometric Filtering - Novel hybrid index for high-recall vector search.
 *
 * CGF (Cascaded Geometric Filtering) achieves HNSW-level recall (90%+) with
 * IVF-PQ-level memory efficiency through a multi-stage filtering pipeline:
 *
 * 1. Ultra-coarse geometric filtering (99% elimination)
 * 2. Smart IVF with learned probing patterns
 * 3. Hybrid storage (PQ for filtering + 8-bit quantization for accuracy)
 * 4. Mini-HNSW graphs per cluster for final refinement
 *
 * Key innovation: Use compression for elimination, not approximation.
 */

#include <cstdint>
#include <expected>
#include <memory>
#include <vector>
#include <array>
#include <unordered_map>
#include <queue>

#include "vesper/error.hpp"
#include "vesper/index/hnsw.hpp"
#include "vesper/index/ivf_pq.hpp"
#include "vesper/index/cgf_capq_bridge.hpp"

namespace vesper::index {

/** \brief CGF build parameters. */
struct CgfBuildParams {
    // Coarse filtering
    std::uint32_t n_super_clusters{0};      /**< Number of super-clusters (0 = auto: N^0.25) */
    std::uint32_t n_projections{8};         /**< Random projections for skyline signatures */

    // IVF parameters
    std::uint32_t nlist{1024};              /**< Number of IVF clusters */
    bool use_learned_probing{true};         /**< Use ML for cluster selection */
    float confidence_threshold{0.95f};       /**< Cumulative probability for probing */

    // Hybrid storage
    std::uint32_t pq_m{16};                 /**< PQ subquantizers for filtering */
    std::uint32_t pq_nbits{8};              /**< Bits per PQ code */
    bool use_8bit_storage{true};            /**< Store 8-bit quantized vectors */

    // Mini-HNSW parameters
    std::uint32_t mini_hnsw_M{8};           /**< Connections in mini graphs */
    std::uint32_t mini_hnsw_ef{32};         /**< Construction ef for mini graphs */
    std::uint32_t cluster_graph_threshold{100}; /**< Min vectors to build graph */

    // Resource limits
    std::size_t memory_budget_mb{0};        /**< Memory budget (0 = unlimited) */
    std::uint32_t num_threads{0};           /**< Threads for construction (0 = auto) */

    // Training parameters
    std::size_t training_sample_size{100000}; /**< Samples for training */
    std::uint32_t predictor_epochs{10};     /**< Epochs for probing predictor */
};

/** \brief CGF search parameters. */
struct CgfSearchParams {
    std::uint32_t k{10};                    /**< Number of results */

    // Phase controls
    bool use_coarse_filter{true};           /**< Enable geometric pre-filtering */
    bool use_learned_probing{true};         /**< Use ML cluster selection */
    bool use_mini_hnsw{true};                /**< Use per-cluster graphs */
    bool use_exact_rerank{true};            /**< Final exact reranking */

    // Probing parameters
    std::uint32_t max_probe{0};             /**< Max clusters to probe (0 = auto) */
    float confidence_threshold{0.95f};       /**< Confidence for adaptive probing */

    // Distance thresholds
    float pq_filter_threshold{0.0f};        /**< PQ distance cutoff (0 = auto) */
    std::uint32_t rerank_pool_size{0};      /**< Final rerank pool (0 = auto: 3*k) */

    // Performance hints
    bool prefetch_clusters{true};           /**< Prefetch cluster data */
    bool parallel_cluster_search{true};     /**< Search clusters in parallel */
};

/** \brief Skyline signature for geometric filtering. */
struct SkylineSignature {
    std::vector<float> min_projections;     /**< Min projections on random axes */
    std::vector<float> max_projections;     /**< Max projections on random axes */
    std::vector<std::uint32_t> member_clusters; /**< Child IVF clusters */
    std::array<float, 128> centroid;        /**< Super-cluster centroid */
    float radius;                            /**< Bounding radius */
};

/** \brief Compressed vector storage. */
struct CompressedVector {
    std::uint64_t id;                       /**< Vector ID */
    std::vector<std::uint8_t> pq_codes;     /**< PQ codes for filtering */
    std::vector<std::int8_t> quantized;     /**< 8-bit quantized vector */
    float scale;                             /**< Quantization scale */
    float offset;                            /**< Quantization offset */
};

/** \brief Learned cluster probe predictor. */
class ClusterPredictor {
public:
    ClusterPredictor(std::size_t dim, std::uint32_t n_clusters);

    /** Train predictor on query patterns. */
    auto train(const std::vector<float>& queries,
               const std::vector<std::vector<std::uint32_t>>& relevant_clusters,
               std::uint32_t epochs)
        -> std::expected<void, core::error>;

    /** Predict cluster relevance for query. */
    auto predict_relevance(const float* query, std::uint32_t cluster_id) const
        -> float;

    /** Get ranked clusters for query. */
    auto get_probe_order(const float* query, std::uint32_t n_clusters) const
        -> std::vector<std::pair<float, std::uint32_t>>;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/** \brief Mini-HNSW graph for a single cluster. */
class MiniHnsw {
public:
    MiniHnsw(std::size_t dim, std::uint32_t M, std::uint32_t ef_construction);

    /** Build graph from cluster vectors. */
    auto build(const std::vector<CompressedVector>& vectors)
        -> std::expected<void, core::error>;

    /** Search within cluster. */
    auto search(const float* query, std::uint32_t k, std::uint32_t ef_search) const
        -> std::vector<std::pair<std::uint64_t, float>>;

    /** Get graph statistics. */
    auto stats() const -> HnswStats;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/** \brief CGF Index Statistics. */
struct CgfStats {
    // Index structure
    std::size_t n_vectors{0};               /**< Total vectors indexed */
    std::uint32_t n_super_clusters{0};      /**< Number of super-clusters */
    std::uint32_t n_clusters{0};            /**< Number of IVF clusters */
    std::uint32_t n_mini_graphs{0};         /**< Number of mini-HNSW graphs */

    // Memory usage
    std::size_t memory_skyline_mb{0};       /**< Skyline signatures memory */
    std::size_t memory_pq_mb{0};            /**< PQ codes memory */
    std::size_t memory_quantized_mb{0};     /**< Quantized vectors memory */
    std::size_t memory_graphs_mb{0};        /**< Mini-HNSW graphs memory */
    std::size_t memory_total_mb{0};         /**< Total memory usage */

    // Performance metrics
    float avg_cluster_size{0.0f};           /**< Average vectors per cluster */
    float cluster_imbalance{0.0f};          /**< Cluster size std dev / mean */
    float predictor_accuracy{0.0f};         /**< Probe predictor accuracy */

    // Build statistics
    float build_time_sec{0.0f};             /**< Total build time */
    float train_time_sec{0.0f};             /**< Training time */
    float index_time_sec{0.0f};             /**< Indexing time */

    // CAPQ prefilter metrics (if enabled)
    bool capq_enabled{false};
    std::size_t capq_last_s1_keep{0};
    std::size_t capq_last_s2_keep{0};
    double capq_last_s1_ms{0.0};
    double capq_last_s2_ms{0.0};
    double capq_last_s3_ms{0.0};
    double capq_last_s1_drop{0.0};
    double capq_last_s2_drop{0.0};
};

/** \brief Cascaded Geometric Filtering index.
 *
 * Novel hybrid index achieving high recall with low memory footprint
 * through intelligent cascaded filtering and hybrid storage.
 */
class CgfIndex {
public:
    CgfIndex();
    ~CgfIndex();
    CgfIndex(CgfIndex&&) noexcept;
    CgfIndex& operator=(CgfIndex&&) noexcept;
    CgfIndex(const CgfIndex&) = delete;
    CgfIndex& operator=(const CgfIndex&) = delete;

    /** \brief Train index on sample data.
     *
     * \param data Training vectors [n * dim]
     * \param dim Vector dimensionality
     * \param n Number of training vectors
     * \param params Build parameters
     * \return Success or error
     *
     * Trains all components: skyline signatures, IVF centroids, PQ codebooks,
     * quantization parameters, and cluster predictor.
     */
    auto train(const float* data, std::size_t dim, std::size_t n,
               const CgfBuildParams& params = {})
        -> std::expected<void, core::error>;

    /** \brief Add vectors to index.
     *
     * \param ids Vector identifiers [n]
     * \param data Vector data [n * dim]
     * \param n Number of vectors
     * \return Success or error
     *
     * Assigns vectors to clusters, compresses them, and builds mini-HNSW graphs.
     */
    auto add(const std::uint64_t* ids, const float* data, std::size_t n)
        -> std::expected<void, core::error>;

    /** \brief Search for nearest neighbors.
     *
     * \param query Query vector [dim]
     * \param params Search parameters
     * \return Vector IDs and distances
     *
     * Executes cascaded filtering pipeline:
     * 1. Geometric pre-filtering
     * 2. Smart cluster selection
     * 3. Hybrid distance computation
     * 4. Mini-HNSW traversal
     * 5. Exact reranking
     */
    auto search(const float* query, const CgfSearchParams& params = {}) const
        -> std::expected<std::vector<std::pair<std::uint64_t, float>>, core::error>;

    /** \brief Batch search for multiple queries.
     *
     * \param queries Query vectors [n_queries * dim]
     * \param n_queries Number of queries
     * \param params Search parameters
     * \return Results for each query
     */
    auto search_batch(const float* queries, std::size_t n_queries,
                      const CgfSearchParams& params = {}) const
        -> std::expected<std::vector<std::vector<std::pair<std::uint64_t, float>>>,
                        core::error>;

    /** \brief Save index to file.
     *
     * \param path File path
     * \return Success or error
     */
    auto save(const std::string& path) const
        -> std::expected<void, core::error>;

    /** \brief Load index from file.
     *
     * \param path File path
     * \return Success or error
     */
    auto load(const std::string& path)
        -> std::expected<void, core::error>;

    /** \brief Get index statistics. */
    auto stats() const -> CgfStats;

    /** \brief Optimize index for better performance.
     *
     * Rebalances clusters, rebuilds mini-graphs, and retrains predictor.
     */
    auto optimize()
        -> std::expected<void, core::error>;

    /** Enable CAPQ prefilter stage. The view and models must outlive the index or be refreshed. */
    auto enable_capq(const CapqSoAView& view,
                     const CapqWhiteningModel& wm,
                     const CapqQ8Params& q8p,
                     const std::array<std::uint64_t, 6>& seeds,
                     CapqHammingBits hbits)
        -> std::expected<void, core::error>;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vesper::index
