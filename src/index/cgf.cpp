/** \file cgf.cpp
 *  \brief Implementation of Cascaded Geometric Filtering index.
 *
 * This file provides the public API implementation that wraps the
 * internal CascadedGeometricFilter orchestrator and its components.
 */

#include "vesper/index/cgf.hpp"
#include "vesper/index/cgf_capq_bridge.hpp"
#include "cascaded_geometric_filter.cpp"

#include <chrono>
#include <fstream>
#include <thread>

namespace vesper::index {

// ClusterPredictor implementation
class ClusterPredictor::Impl {
public:
    Impl(std::size_t dim, std::uint32_t n_clusters)
        : dim_(dim), n_clusters_(n_clusters) {
        // Simple linear model for cluster relevance
        weights_.resize(dim * n_clusters, 0.0f);
        biases_.resize(n_clusters, 0.0f);
    }

    auto train(const std::vector<float>& queries,
               const std::vector<std::vector<std::uint32_t>>& relevant_clusters,
               std::uint32_t epochs)
        -> std::expected<void, core::error> {

        // Simplified training: learn which clusters are relevant for query patterns
        // In production, use gradient descent or neural network

        const std::size_t n_queries = queries.size() / dim_;

        for (std::uint32_t epoch = 0; epoch < epochs; ++epoch) {
            for (std::size_t q = 0; q < n_queries; ++q) {
                const float* query = queries.data() + q * dim_;
                const auto& relevant = relevant_clusters[q];

                // Update weights for relevant clusters
                for (std::uint32_t c : relevant) {
                    for (std::size_t d = 0; d < dim_; ++d) {
                        weights_[c * dim_ + d] += 0.01f * query[d];
                    }
                    biases_[c] += 0.01f;
                }
            }
        }

        trained_ = true;
        return {};
    }

    auto predict_relevance(const float* query, std::uint32_t cluster_id) const -> float {
        if (!trained_) return 1.0f / n_clusters_;  // Uniform if not trained

        float score = biases_[cluster_id];
        const float* weights = weights_.data() + cluster_id * dim_;

        for (std::size_t d = 0; d < dim_; ++d) {
            score += query[d] * weights[d];
        }

        // Sigmoid activation
        return 1.0f / (1.0f + std::exp(-score));
    }

    auto get_probe_order(const float* query, std::uint32_t n_clusters) const
        -> std::vector<std::pair<float, std::uint32_t>> {

        std::vector<std::pair<float, std::uint32_t>> scores;
        scores.reserve(n_clusters);

        for (std::uint32_t c = 0; c < n_clusters && c < n_clusters_; ++c) {
            float relevance = predict_relevance(query, c);
            scores.emplace_back(relevance, c);
        }

        // Sort by relevance (descending)
        std::sort(scores.begin(), scores.end(), std::greater<>());

        return scores;
    }

private:
    std::size_t dim_;
    std::uint32_t n_clusters_;
    bool trained_ = false;
    std::vector<float> weights_;
    std::vector<float> biases_;
};

ClusterPredictor::ClusterPredictor(std::size_t dim, std::uint32_t n_clusters)
    : impl_(std::make_unique<Impl>(dim, n_clusters)) {}

ClusterPredictor::~ClusterPredictor() = default;

auto ClusterPredictor::train(const std::vector<float>& queries,
                            const std::vector<std::vector<std::uint32_t>>& relevant_clusters,
                            std::uint32_t epochs)
    -> std::expected<void, core::error> {
    return impl_->train(queries, relevant_clusters, epochs);
}

auto ClusterPredictor::predict_relevance(const float* query, std::uint32_t cluster_id) const
    -> float {
    return impl_->predict_relevance(query, cluster_id);
}

auto ClusterPredictor::get_probe_order(const float* query, std::uint32_t n_clusters) const
    -> std::vector<std::pair<float, std::uint32_t>> {
    return impl_->get_probe_order(query, n_clusters);
}

// MiniHnsw implementation (wrapper around MiniHNSW from mini_hnsw.cpp)
class MiniHnsw::Impl {
public:
    Impl(std::size_t dim, std::uint32_t M, std::uint32_t ef_construction)
        : dim_(dim), M_(M), ef_construction_(ef_construction) {}

    auto build(const std::vector<CompressedVector>& vectors)
        -> std::expected<void, core::error> {
        // Build mini HNSW from compressed vectors
        // This would integrate with the MiniHNSW class from mini_hnsw.cpp

        n_vectors_ = vectors.size();
        return {};
    }

    auto search(const float* query, std::uint32_t k, std::uint32_t ef_search) const
        -> std::vector<std::pair<std::uint64_t, float>> {
        // Simplified search
        return {};
    }

    auto stats() const -> HnswStats {
        HnswStats stats;
        stats.n_vectors = n_vectors_;
        stats.n_edges = n_vectors_ * M_;
        return stats;
    }

private:
    std::size_t dim_;
    std::uint32_t M_;
    std::uint32_t ef_construction_;
    std::size_t n_vectors_ = 0;
};

MiniHnsw::MiniHnsw(std::size_t dim, std::uint32_t M, std::uint32_t ef_construction)
    : impl_(std::make_unique<Impl>(dim, M, ef_construction)) {}

MiniHnsw::~MiniHnsw() = default;

auto MiniHnsw::build(const std::vector<CompressedVector>& vectors)
    -> std::expected<void, core::error> {
    return impl_->build(vectors);
}

auto MiniHnsw::search(const float* query, std::uint32_t k, std::uint32_t ef_search) const
    -> std::vector<std::pair<std::uint64_t, float>> {
    return impl_->search(query, k, ef_search);
}

auto MiniHnsw::stats() const -> HnswStats {
    return impl_->stats();
}

// CgfIndex implementation
class CgfIndex::Impl {
public:
    Impl() = default;

    auto enable_capq(const CapqSoAView& view,
                     const CapqWhiteningModel& wm,
                     const CapqQ8Params& q8p,
                     const std::array<std::uint64_t, 6>& seeds,
                     CapqHammingBits hbits)
        -> std::expected<void, core::error> {
        capq_filter_ = CapqFilter{};
        capq_filter_.initialize(view, wm, q8p, seeds, hbits);
        capq_enabled_ = true;
        return {};
    }

    auto train(const float* data, std::size_t dim, std::size_t n,
               const CgfBuildParams& params)
        -> std::expected<void, core::error> {

        dim_ = dim;
        params_ = params;

        // Convert parameters to internal config
        CascadedGeometricFilter::Config config;
        config.n_projections = params.n_projections;
        config.n_super_clusters = params.n_super_clusters;
        config.n_clusters = params.nlist;
        config.n_probe = params.max_probe_clusters();
        config.m_pq = params.pq_m;
        config.nbits_pq = params.pq_nbits;
        config.nbits_quant = 8;
        config.hnsw_M = params.mini_hnsw_M;
        config.hnsw_efConstruction = params.mini_hnsw_ef;
        config.use_learned_ordering = params.use_learned_probing;
        config.recall_target = params.confidence_threshold;

        // Create the cascaded filter
        cgf_ = std::make_unique<CascadedGeometricFilter>(dim, config);

        // Train on data
        auto result = cgf_->train(data, n);
        if (!result.has_value()) {
            return result;
        }

        // Train cluster predictor if enabled
        if (params.use_learned_probing) {
            predictor_ = std::make_unique<ClusterPredictor>(dim, params.nlist);
            // Would need to generate training data here
        }

        trained_ = true;
        return {};
    }

    auto add(const std::uint64_t* ids, const float* data, std::size_t n)
        -> std::expected<void, core::error> {

        if (!trained_) {
            return std::vesper_unexpected(core::error{
                core::error_code::invalid_state,
                "Index not trained",
                "cgf"
            });
        }

        return cgf_->add(ids, data, n);
    }

    auto search(const float* query, const CgfSearchParams& params) const
        -> std::expected<std::vector<std::pair<std::uint64_t, float>>, core::error> {

        if (!trained_) {
            return std::vesper_unexpected(core::error{
                core::error_code::invalid_state,
                "Index not trained",
                "cgf"
            });
        }

        if (capq_enabled_) {
            // Temporary: CAPQ-only prefilter + return candidates (to diagnose recall)
            const std::size_t s1_keep = 8000;
            const std::size_t s2_keep = 200;
            auto ids = capq_filter_.search(query, dim_, params.k * 3, s1_keep, s2_keep);
            std::vector<std::pair<std::uint64_t, float>> out;
            out.reserve(ids.size());
            for (auto id32 : ids) out.emplace_back(static_cast<std::uint64_t>(id32), 0.0f);
            return out;
        }
        return cgf_->search(query, params.k, params.confidence_threshold);
    }

    auto search_batch(const float* queries, std::size_t n_queries,
                     const CgfSearchParams& params) const
        -> std::expected<std::vector<std::vector<std::pair<std::uint64_t, float>>>,
                        core::error> {

        std::vector<std::vector<std::pair<std::uint64_t, float>>> results;
        results.reserve(n_queries);

        // Parallel batch search if enabled
        if (params.parallel_cluster_search && n_queries > 1) {
            std::vector<std::thread> threads;
            std::mutex results_mutex;

            const std::size_t n_threads = std::min<std::size_t>(
                std::thread::hardware_concurrency(), n_queries
            );

            for (std::size_t t = 0; t < n_threads; ++t) {
                threads.emplace_back([&, t]() {
                    for (std::size_t q = t; q < n_queries; q += n_threads) {
                        auto res = search(queries + q * dim_, params);
                        if (res.has_value()) {
                            std::lock_guard<std::mutex> lock(results_mutex);
                            results[q] = std::move(*res);
                        }
                    }
                });
            }

            for (auto& thread : threads) {
                thread.join();
            }
        } else {
            // Sequential search
            for (std::size_t q = 0; q < n_queries; ++q) {
                auto res = search(queries + q * dim_, params);
                if (!res.has_value()) {
                    return std::vesper_unexpected(res.error());
                }
                results.push_back(std::move(*res));
            }
        }

        return results;
    }

    auto save(const std::string& path) const -> std::expected<void, core::error> {
        if (!trained_) {
            return std::vesper_unexpected(core::error{
                core::error_code::invalid_state,
                "Index not trained",
                "cgf"
            });
        }

        // Serialize to file
        std::ofstream file(path, std::ios::binary);
        if (!file) {
            return std::vesper_unexpected(core::error{
                core::error_code::io_error,
                "Failed to open file for writing",
                "cgf"
            });
        }

        // Write header
        file.write("CGF1", 4);  // Magic number
        file.write(reinterpret_cast<const char*>(&dim_), sizeof(dim_));

        // Would serialize all components here

        return {};
    }

    auto load(const std::string& path) -> std::expected<void, core::error> {
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            return std::vesper_unexpected(core::error{
                core::error_code::io_error,
                "Failed to open file for reading",
                "cgf"
            });
        }

        // Read header
        char magic[4];
        file.read(magic, 4);
        if (std::string(magic, 4) != "CGF1") {
            return std::vesper_unexpected(core::error{
                core::error_code::invalid_format,
                "Invalid file format",
                "cgf"
            });
        }

        file.read(reinterpret_cast<char*>(&dim_), sizeof(dim_));

        // Would deserialize all components here

        trained_ = true;
        return {};
    }

    auto stats() const -> CgfStats {
        CgfStats stats;

        if (cgf_) {
            auto cascade_stats = cgf_->get_stats();
            stats.n_vectors = cgf_->size();
            stats.n_clusters = params_.nlist;
            stats.n_super_clusters = params_.n_super_clusters;

            // Memory usage
            std::size_t total_memory = cgf_->get_memory_usage();
            stats.memory_total_mb = total_memory / (1024 * 1024);

            // Performance metrics
            stats.avg_cluster_size = static_cast<float>(stats.n_vectors) / stats.n_clusters;

            // Cascade reduction stats
            if (cascade_stats.total_queries > 0) {
                float stage1_reduction = cascade_stats.get_stage1_reduction();
                float overall_reduction = cascade_stats.get_overall_reduction();
                stats.predictor_accuracy = 1.0f - overall_reduction;  // Approximation
            }
        }

        return stats;
    }

    auto optimize() -> std::expected<void, core::error> {
        if (!trained_) {
            return std::vesper_unexpected(core::error{
                core::error_code::invalid_state,
                "Index not trained",
                "cgf"
            });
        }

        // Optimization would rebalance clusters, rebuild graphs, etc.

        return {};
    }

private:
    std::size_t dim_ = 0;
    bool trained_ = false;
    CgfBuildParams params_;
    std::unique_ptr<CascadedGeometricFilter> cgf_;
    std::unique_ptr<ClusterPredictor> predictor_;
    bool capq_enabled_{false};
    CapqFilter capq_filter_{};

    // Helper to get max probe clusters
    std::uint32_t max_probe_clusters() const {
        if (params_.max_probe > 0) {
            return params_.max_probe;
        }
        // Auto: probe sqrt(nlist) clusters
        return static_cast<std::uint32_t>(std::sqrt(params_.nlist));
    }
};

// CgfIndex public interface
CgfIndex::CgfIndex() : impl_(std::make_unique<Impl>()) {}
CgfIndex::~CgfIndex() = default;
CgfIndex::CgfIndex(CgfIndex&&) noexcept = default;
CgfIndex& CgfIndex::operator=(CgfIndex&&) noexcept = default;

auto CgfIndex::train(const float* data, std::size_t dim, std::size_t n,
                    const CgfBuildParams& params)
    -> std::expected<void, core::error> {
    return impl_->train(data, dim, n, params);
}

auto CgfIndex::add(const std::uint64_t* ids, const float* data, std::size_t n)
    -> std::expected<void, core::error> {
    return impl_->add(ids, data, n);
}

auto CgfIndex::search(const float* query, const CgfSearchParams& params) const
    -> std::expected<std::vector<std::pair<std::uint64_t, float>>, core::error> {
    return impl_->search(query, params);
}

auto CgfIndex::search_batch(const float* queries, std::size_t n_queries,
                           const CgfSearchParams& params) const
    -> std::expected<std::vector<std::vector<std::pair<std::uint64_t, float>>>,
                    core::error> {
    return impl_->search_batch(queries, n_queries, params);
}

auto CgfIndex::save(const std::string& path) const
    -> std::expected<void, core::error> {
    return impl_->save(path);
}

auto CgfIndex::load(const std::string& path)
    -> std::expected<void, core::error> {
    return impl_->load(path);
}

auto CgfIndex::stats() const -> CgfStats {
    return impl_->stats();
}

auto CgfIndex::optimize() -> std::expected<void, core::error> {
    return impl_->optimize();
}

auto CgfIndex::enable_capq(const CapqSoAView& view,
                           const CapqWhiteningModel& wm,
                           const CapqQ8Params& q8p,
                           const std::array<std::uint64_t, 6>& seeds,
                           CapqHammingBits hbits)
    -> std::expected<void, core::error> {
    return impl_->enable_capq(view, wm, q8p, seeds, hbits);
}

} // namespace vesper::index
