#pragma once

/**
 * \file dataset_loader.hpp
 * \brief Dataset loading utilities for integration testing
 * 
 * Supports loading vector datasets in various formats:
 * - HDF5 (ANN benchmarks format)
 * - FVECS/IVECS (TEXMEX format)
 * - BVECS (binary format)
 */

#include <vector>
#include <string>
#include <fstream>
#include <memory>
#include <expected>
#include <span>
#include <cstdint>
#include <filesystem>

#include "vesper/error.hpp"

namespace vesper::test {

/** \brief Dataset format types */
enum class DatasetFormat {
    HDF5,      /**< HDF5 format (requires HDF5 library) */
    FVECS,     /**< Float vectors format */
    IVECS,     /**< Integer vectors format */
    BVECS,     /**< Binary vectors format */
    AUTO       /**< Auto-detect from file extension */
};

/** \brief Distance metric type */
enum class DistanceMetric {
    L2,        /**< Euclidean distance */
    IP,        /**< Inner product */
    COSINE,    /**< Cosine similarity */
    ANGULAR    /**< Angular distance */
};

/** \brief Dataset information */
struct DatasetInfo {
    std::string name;
    std::size_t num_vectors{0};
    std::size_t dimension{0};
    std::size_t num_queries{0};
    DistanceMetric metric{DistanceMetric::L2};
    bool has_groundtruth{false};
};

/** \brief Loaded dataset */
struct Dataset {
    DatasetInfo info;
    std::vector<float> base_vectors;      /**< Training/base vectors [n x d] */
    std::vector<float> query_vectors;     /**< Query vectors [q x d] */
    std::vector<std::uint32_t> groundtruth; /**< Ground truth neighbors [q x k] */
    std::vector<float> distances;         /**< Ground truth distances [q x k] */
    std::uint32_t k{100};                 /**< Number of neighbors in ground truth */
};

/** \brief Dataset loader interface */
class DatasetLoader {
public:
    /**
     * \brief Load dataset from file
     * \param filepath Path to dataset file
     * \param format Dataset format (auto-detect if AUTO)
     * \return Loaded dataset or error
     */
    static auto load(const std::filesystem::path& filepath, 
                    DatasetFormat format = DatasetFormat::AUTO)
        -> std::expected<Dataset, core::error>;
    
    /**
     * \brief Load FVECS format file
     * \param filepath Path to .fvecs file
     * \return Vectors and dimension
     */
    static auto load_fvecs(const std::filesystem::path& filepath)
        -> std::expected<std::pair<std::vector<float>, std::size_t>, core::error>;
    
    /**
     * \brief Load IVECS format file
     * \param filepath Path to .ivecs file
     * \return Vectors and dimension
     */
    static auto load_ivecs(const std::filesystem::path& filepath)
        -> std::expected<std::pair<std::vector<std::uint32_t>, std::size_t>, core::error>;
    
    /**
     * \brief Load BVECS format file
     * \param filepath Path to .bvecs file
     * \return Vectors and dimension
     */
    static auto load_bvecs(const std::filesystem::path& filepath)
        -> std::expected<std::pair<std::vector<std::uint8_t>, std::size_t>, core::error>;
    
    /**
     * \brief Load standard benchmark dataset by name
     * \param name Dataset name (e.g., "sift-1m", "fashion-mnist", "glove-100")
     * \param data_dir Root data directory
     * \return Loaded dataset or error
     */
    static auto load_benchmark(const std::string& name,
                              const std::filesystem::path& data_dir = "data")
        -> std::expected<Dataset, core::error>;

private:
    /**
     * \brief Detect format from file extension
     */
    static DatasetFormat detect_format(const std::filesystem::path& filepath);
    
    /**
     * \brief Load HDF5 format (if HDF5 support is available)
     */
    static auto load_hdf5(const std::filesystem::path& filepath)
        -> std::expected<Dataset, core::error>;
    
    /**
     * \brief Load TEXMEX format dataset (base + query + groundtruth)
     */
    static auto load_texmex(const std::filesystem::path& base_path)
        -> std::expected<Dataset, core::error>;
};

/** \brief Helper class for computing search metrics */
class SearchMetrics {
public:
    /**
     * \brief Compute recall@k
     * \param results Search results [q x k] 
     * \param groundtruth Ground truth [q x k_gt]
     * \param k Number of results to consider
     * \return Recall value [0, 1]
     */
    static float compute_recall(
        const std::vector<std::uint32_t>& results,
        const std::vector<std::uint32_t>& groundtruth,
        std::size_t num_queries,
        std::size_t k,
        std::size_t k_gt);
    
    /**
     * \brief Compute mean reciprocal rank (MRR)
     */
    static float compute_mrr(
        const std::vector<std::uint32_t>& results,
        const std::vector<std::uint32_t>& groundtruth,
        std::size_t num_queries,
        std::size_t k);
    
    /**
     * \brief Compute precision@k
     */
    static float compute_precision(
        const std::vector<std::uint32_t>& results,
        const std::vector<std::uint32_t>& groundtruth,
        std::size_t num_queries,
        std::size_t k);
};

/** \brief Performance metrics collector */
class PerformanceMetrics {
public:
    struct Stats {
        double mean{0};
        double median{0};
        double p50{0};
        double p95{0};
        double p99{0};
        double min{0};
        double max{0};
        double stddev{0};
    };
    
    /**
     * \brief Record a latency measurement
     */
    void record_latency(double microseconds);
    
    /**
     * \brief Record throughput
     */
    void record_throughput(double ops_per_second);
    
    /**
     * \brief Get latency statistics
     */
    Stats get_latency_stats() const;
    
    /**
     * \brief Get throughput statistics
     */
    Stats get_throughput_stats() const;
    
    /**
     * \brief Reset all metrics
     */
    void reset();
    
    /**
     * \brief Print summary report
     */
    void print_summary(const std::string& name = "") const;

private:
    std::vector<double> latencies_;
    std::vector<double> throughputs_;
    
    static Stats compute_stats(std::vector<double> values);
};

} // namespace vesper::test