#include "dataset_loader.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <unordered_set>
#include <iostream>
#include <iomanip>

namespace vesper::test {

namespace fs = std::filesystem;
namespace {
// Normalize vectors to unit L2 norm in-place (handles zero vectors safely)
inline void normalize_in_place(std::vector<float>& data, std::size_t dim) {
    if (dim == 0 || data.empty()) return;
    const float eps = 1e-12f;
    const std::size_t n = data.size() / dim;
    for (std::size_t i = 0; i < n; ++i) {
        float* v = data.data() + i * dim;
        float norm2 = 0.0f;
        for (std::size_t d = 0; d < dim; ++d) norm2 += v[d] * v[d];
        float denom = std::sqrt(std::max(norm2, eps));
        for (std::size_t d = 0; d < dim; ++d) v[d] /= denom;
    }
}

inline DistanceMetric infer_metric_from_name(const std::string& name_lower) {
    if (name_lower.find("angular") != std::string::npos || name_lower.find("cosine") != std::string::npos)
        return DistanceMetric::ANGULAR;
    if (name_lower.find("ip") != std::string::npos || name_lower.find("inner") != std::string::npos)
        return DistanceMetric::IP;
    return DistanceMetric::L2;
}
} // anonymous namespace


// Implementation of DatasetLoader

DatasetFormat DatasetLoader::detect_format(const fs::path& filepath) {
    auto ext = filepath.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (ext == ".hdf5" || ext == ".h5") return DatasetFormat::HDF5;
    if (ext == ".fvecs") return DatasetFormat::FVECS;
    if (ext == ".ivecs") return DatasetFormat::IVECS;
    if (ext == ".bvecs") return DatasetFormat::BVECS;

    return DatasetFormat::FVECS; // Default
}

auto DatasetLoader::load(const fs::path& filepath, DatasetFormat format)
    -> std::expected<Dataset, core::error> {

    if (!fs::exists(filepath)) {
        return std::vesper_unexpected(core::error{
            core::error_code::not_found,
            "Dataset file not found: " + filepath.string(),
            "dataset_loader"
        });
    }

    if (format == DatasetFormat::AUTO) {
        format = detect_format(filepath);
    }

    switch (format) {
        case DatasetFormat::HDF5:
            return load_hdf5(filepath);

        case DatasetFormat::FVECS:
        case DatasetFormat::IVECS:
        case DatasetFormat::BVECS:
            return load_texmex(filepath);

        default:
            return std::vesper_unexpected(core::error{
                core::error_code::invalid_argument,
                "Unsupported dataset format",
                "dataset_loader"
            });
    }
}

auto DatasetLoader::load_fvecs(const fs::path& filepath)
    -> std::expected<std::pair<std::vector<float>, std::size_t>, core::error> {

    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        return std::vesper_unexpected(core::error{
            core::error_code::io_failed,
            "Failed to open file: " + filepath.string(),
            "dataset_loader"
        });
    }

    std::vector<float> data;
    std::size_t dimension = 0;

    while (file.good()) {
        std::uint32_t dim;
        file.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        if (!file.good()) break;

        if (dimension == 0) {
            dimension = dim;
        } else if (dimension != dim) {
            return std::vesper_unexpected(core::error{
                core::error_code::data_integrity,
                "Inconsistent dimensions in FVECS file",
                "dataset_loader"
            });
        }

        std::vector<float> vec(dim);
        file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(float));
        if (!file.good() && !file.eof()) {
            return std::vesper_unexpected(core::error{
                core::error_code::io_failed,
                "Failed to read vector data",
                "dataset_loader"
            });
        }

        data.insert(data.end(), vec.begin(), vec.end());
    }

    return std::make_pair(std::move(data), dimension);
}

auto DatasetLoader::load_ivecs(const fs::path& filepath)
    -> std::expected<std::pair<std::vector<std::uint32_t>, std::size_t>, core::error> {

    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        return std::vesper_unexpected(core::error{
            core::error_code::io_failed,
            "Failed to open file: " + filepath.string(),
            "dataset_loader"
        });
    }

    std::vector<std::uint32_t> data;
    std::size_t dimension = 0;

    while (file.good()) {
        std::uint32_t dim;
        file.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        if (!file.good()) break;

        if (dimension == 0) {
            dimension = dim;
        } else if (dimension != dim) {
            return std::vesper_unexpected(core::error{
                core::error_code::data_integrity,
                "Inconsistent dimensions in IVECS file",
                "dataset_loader"
            });
        }

        std::vector<std::uint32_t> vec(dim);
        file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(std::uint32_t));
        if (!file.good() && !file.eof()) {
            return std::vesper_unexpected(core::error{
                core::error_code::io_failed,
                "Failed to read vector data",
                "dataset_loader"
            });
        }

        data.insert(data.end(), vec.begin(), vec.end());
    }

    return std::make_pair(std::move(data), dimension);
}

auto DatasetLoader::load_bvecs(const fs::path& filepath)
    -> std::expected<std::pair<std::vector<std::uint8_t>, std::size_t>, core::error> {

    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        return std::vesper_unexpected(core::error{
            core::error_code::io_failed,
            "Failed to open file: " + filepath.string(),
            "dataset_loader"
        });
    }

    std::vector<std::uint8_t> data;
    std::size_t dimension = 0;

    while (file.good()) {
        std::uint32_t dim;
        file.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        if (!file.good()) break;

        if (dimension == 0) {
            dimension = dim;
        } else if (dimension != dim) {
            return std::vesper_unexpected(core::error{
                core::error_code::data_integrity,
                "Inconsistent dimensions in BVECS file",
                "dataset_loader"
            });
        }

        std::vector<std::uint8_t> vec(dim);
        file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(std::uint8_t));
        if (!file.good() && !file.eof()) {
            return std::vesper_unexpected(core::error{
                core::error_code::io_failed,
                "Failed to read vector data",
                "dataset_loader"
            });
        }

        data.insert(data.end(), vec.begin(), vec.end());
    }

    return std::make_pair(std::move(data), dimension);
}

auto DatasetLoader::load_texmex(const fs::path& base_path)
    -> std::expected<Dataset, core::error> {

    Dataset dataset;

    // Determine base name and directory
    fs::path dir = base_path.parent_path();
    std::string base_name = base_path.stem().string();

    // Remove suffixes like "_base", "_query", etc.
    if (base_name.ends_with("_base")) {
        base_name = base_name.substr(0, base_name.length() - 5);
    } else if (base_name.ends_with("_query")) {
        base_name = base_name.substr(0, base_name.length() - 6);
    }

    // Load base vectors
    fs::path base_file = dir / (base_name + "_base.fvecs");
    if (!fs::exists(base_file)) {
        base_file = dir / (base_name + ".fvecs");
    }

    if (fs::exists(base_file)) {
        auto result = load_fvecs(base_file);
        if (!result) return std::vesper_unexpected(result.error());

        auto [data, dim] = result.value();
        dataset.base_vectors = std::move(data);
        dataset.info.dimension = dim;
        dataset.info.num_vectors = dataset.base_vectors.size() / dim;
    }

    // Load query vectors
    fs::path query_file = dir / (base_name + "_query.fvecs");
    if (fs::exists(query_file)) {
        auto result = load_fvecs(query_file);
        if (!result) return std::vesper_unexpected(result.error());

        auto [data, dim] = result.value();
        if (dataset.info.dimension != 0 && dataset.info.dimension != dim) {
            return std::vesper_unexpected(core::error{
                core::error_code::data_integrity,
                "Query dimension doesn't match base dimension",
                "dataset_loader"
            });
        }
        dataset.query_vectors = std::move(data);
        dataset.info.num_queries = dataset.query_vectors.size() / dim;
    }

    // Load ground truth
    fs::path gt_file = dir / (base_name + "_groundtruth.ivecs");
    if (fs::exists(gt_file)) {
        auto result = load_ivecs(gt_file);
        if (!result) return std::vesper_unexpected(result.error());

        auto [data, k] = result.value();
        dataset.groundtruth = std::move(data);
        dataset.k = static_cast<std::uint32_t>(k);
        dataset.info.has_groundtruth = true;
    }

    // Load distances if available
    fs::path dist_file = dir / (base_name + "_distances.fvecs");
    if (fs::exists(dist_file)) {
        auto result = load_fvecs(dist_file);
        if (result) {
            dataset.distances = std::move(result.value().first);
        }
    }

    // Set dataset info and infer metric from dataset name
    dataset.info.name = base_name;
    std::string name_lower = base_name;
    std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);
    dataset.info.metric = infer_metric_from_name(name_lower);

    // If cosine/angular metric, normalize base and query to unit length so L2 ranks by cosine
    if ((dataset.info.metric == DistanceMetric::ANGULAR || dataset.info.metric == DistanceMetric::COSINE)
        && dataset.info.dimension > 0) {
        if (!dataset.base_vectors.empty()) {
            normalize_in_place(dataset.base_vectors, dataset.info.dimension);
        }
        if (!dataset.query_vectors.empty()) {
            normalize_in_place(dataset.query_vectors, dataset.info.dimension);
        }
    }

    return dataset;
}

auto DatasetLoader::load_hdf5(const fs::path& filepath)
    -> std::expected<Dataset, core::error> {
    // HDF5 support requires linking with HDF5 library
    // For now, return an error
    return std::vesper_unexpected(core::error{
        core::error_code::precondition_failed,
        "HDF5 support not compiled in. Use FVECS format instead.",
        "dataset_loader"
    });
}

auto DatasetLoader::load_benchmark(const std::string& name,
                                  const fs::path& data_dir)
    -> std::expected<Dataset, core::error> {

    // Map common names to file patterns
    std::string normalized_name = name;
    std::transform(normalized_name.begin(), normalized_name.end(),
                  normalized_name.begin(), ::tolower);

    // Try FVECS format first
    fs::path fvecs_dir = data_dir / "fvecs";
    if (fs::exists(fvecs_dir)) {
        fs::path dataset_path = fvecs_dir / (normalized_name + "_base.fvecs");
        if (fs::exists(dataset_path)) {
            return load_texmex(dataset_path);
        }
    }

    // Try HDF5 format
    fs::path hdf5_dir = data_dir / "hdf5";
    if (fs::exists(hdf5_dir)) {
        fs::path dataset_path = hdf5_dir / (normalized_name + ".hdf5");
        if (fs::exists(dataset_path)) {
            return load_hdf5(dataset_path);
        }
    }

    return std::vesper_unexpected(core::error{
        core::error_code::not_found,
        "Benchmark dataset not found: " + name,
        "dataset_loader"
    });
}

// Implementation of SearchMetrics

float SearchMetrics::compute_recall(
    const std::vector<std::uint32_t>& results,
    const std::vector<std::uint32_t>& groundtruth,
    std::size_t num_queries,
    std::size_t k,
    std::size_t k_gt) {

    if (results.empty() || groundtruth.empty() || num_queries == 0) return 0.0f;

    const std::size_t k_ref = std::max<std::size_t>(1, std::min(k, k_gt));
    float total = 0.0f;

    for (std::size_t q = 0; q < num_queries; ++q) {
        // Build set of ground truth top-k_ref neighbors
        std::unordered_set<std::uint32_t> gt_set;
        for (std::size_t i = 0; i < k_ref; ++i) {
            gt_set.insert(groundtruth[q * k_gt + i]);
        }

        // Count matches in top-k results
        std::size_t matches = 0;
        for (std::size_t i = 0; i < k; ++i) {
            if (gt_set.count(results[q * k + i]) > 0) {
                matches++;
            }
        }

        total += static_cast<float>(matches) / static_cast<float>(k_ref);
    }

    return total / static_cast<float>(num_queries);
}

float SearchMetrics::compute_mrr(
    const std::vector<std::uint32_t>& results,
    const std::vector<std::uint32_t>& groundtruth,
    std::size_t num_queries,
    std::size_t k) {

    if (results.empty() || groundtruth.empty()) return 0.0f;

    float total_mrr = 0.0f;
    std::size_t k_gt = groundtruth.size() / num_queries;

    for (std::size_t q = 0; q < num_queries; ++q) {
        // Find first relevant result
        std::uint32_t first_relevant = groundtruth[q * k_gt];

        for (std::size_t i = 0; i < k; ++i) {
            if (results[q * k + i] == first_relevant) {
                total_mrr += 1.0f / (i + 1.0f);
                break;
            }
        }
    }

    return total_mrr / static_cast<float>(num_queries);
}

float SearchMetrics::compute_precision(
    const std::vector<std::uint32_t>& results,
    const std::vector<std::uint32_t>& groundtruth,
    std::size_t num_queries,
    std::size_t k) {

    if (results.empty() || groundtruth.empty()) return 0.0f;

    std::size_t k_gt = groundtruth.size() / num_queries;
    float total_precision = 0.0f;

    for (std::size_t q = 0; q < num_queries; ++q) {
        // Build set of ground truth neighbors
        std::unordered_set<std::uint32_t> gt_set;
        for (std::size_t i = 0; i < k_gt; ++i) {
            gt_set.insert(groundtruth[q * k_gt + i]);
        }

        // Count matches in results
        std::size_t matches = 0;
        for (std::size_t i = 0; i < k; ++i) {
            if (gt_set.count(results[q * k + i]) > 0) {
                matches++;
            }
        }

        total_precision += static_cast<float>(matches) / static_cast<float>(k);
    }

    return total_precision / static_cast<float>(num_queries);
}

// Implementation of PerformanceMetrics

void PerformanceMetrics::record_latency(double microseconds) {
    latencies_.push_back(microseconds);
}

void PerformanceMetrics::record_throughput(double ops_per_second) {
    throughputs_.push_back(ops_per_second);
}

PerformanceMetrics::Stats PerformanceMetrics::compute_stats(std::vector<double> values) {
    if (values.empty()) return {};

    std::sort(values.begin(), values.end());

    Stats stats;
    stats.min = values.front();
    stats.max = values.back();

    // Mean
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    stats.mean = sum / values.size();

    // Percentiles
    auto percentile = [&values](double p) {
        std::size_t idx = static_cast<std::size_t>(p * (values.size() - 1));
        return values[idx];
    };

    stats.p50 = stats.median = percentile(0.50);
    stats.p95 = percentile(0.95);
    stats.p99 = percentile(0.99);

    // Standard deviation
    double sq_sum = 0.0;
    for (double v : values) {
        sq_sum += (v - stats.mean) * (v - stats.mean);
    }
    stats.stddev = std::sqrt(sq_sum / values.size());

    return stats;
}

PerformanceMetrics::Stats PerformanceMetrics::get_latency_stats() const {
    return compute_stats(latencies_);
}

PerformanceMetrics::Stats PerformanceMetrics::get_throughput_stats() const {
    return compute_stats(throughputs_);
}

void PerformanceMetrics::reset() {
    latencies_.clear();
    throughputs_.clear();
}

void PerformanceMetrics::print_summary(const std::string& name) const {
    if (!name.empty()) {
        std::cout << "=== " << name << " ===" << std::endl;
    }

    if (!latencies_.empty()) {
        auto stats = get_latency_stats();
        std::cout << "Latency (μs):" << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  Mean:   " << stats.mean << std::endl;
        std::cout << "  Median: " << stats.median << std::endl;
        std::cout << "  P95:    " << stats.p95 << std::endl;
        std::cout << "  P99:    " << stats.p99 << std::endl;
        std::cout << "  Min:    " << stats.min << std::endl;
        std::cout << "  Max:    " << stats.max << std::endl;
        std::cout << "  StdDev: " << stats.stddev << std::endl;
    }

    if (!throughputs_.empty()) {
        auto stats = get_throughput_stats();
        std::cout << "Throughput (ops/s):" << std::endl;
        std::cout << std::fixed << std::setprecision(0);
        std::cout << "  Mean:   " << stats.mean << std::endl;
        std::cout << "  Median: " << stats.median << std::endl;
        std::cout << "  Max:    " << stats.max << std::endl;
    }
}

} // namespace vesper::test