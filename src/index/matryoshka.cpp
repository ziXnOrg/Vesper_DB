/** \file matryoshka.cpp
 *  \brief Implementation of Matryoshka Representation Learning
 */

#include "vesper/index/matryoshka.hpp"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <limits>

namespace vesper::index {

// Forward declare the Impl class
class MatryoshkaEmbedding::Impl {
public:
    std::uint32_t full_dimension{0};
    MatryoshkaConfig config;
    std::vector<float> dimension_scales;
    bool initialized{false};
    
    void compute_dimension_scales() {
        dimension_scales.clear();
        dimension_scales.reserve(config.dimensions.size());
        
        float scale = 1.0f;
        for (const auto& dim : config.dimensions) {
            dimension_scales.push_back(scale);
            scale *= config.dimension_weights_decay;
        }
    }
};

MatryoshkaEmbedding::MatryoshkaEmbedding()
    : impl_(std::make_unique<Impl>()) {}

MatryoshkaEmbedding::~MatryoshkaEmbedding() = default;

MatryoshkaEmbedding::MatryoshkaEmbedding(MatryoshkaEmbedding&&) noexcept = default;

MatryoshkaEmbedding& MatryoshkaEmbedding::operator=(MatryoshkaEmbedding&&) noexcept = default;

auto MatryoshkaEmbedding::init(std::uint32_t full_dimension, const MatryoshkaConfig& config)
    -> std::expected<void, core::error> {
    using core::error;
    using core::error_code;
    
    if (full_dimension == 0) {
        return std::vesper_unexpected(error{
            error_code::invalid_argument,
            "Full dimension must be greater than zero",
            "matryoshka"
        });
    }
    
    // Validate dimensions are in ascending order and less than full dimension
    for (std::size_t i = 0; i < config.dimensions.size(); ++i) {
        if (config.dimensions[i] > full_dimension) {
            return std::vesper_unexpected(error{
                error_code::invalid_argument,
                "Nested dimension exceeds full dimension",
                "matryoshka"
            });
        }
        if (i > 0 && config.dimensions[i] <= config.dimensions[i-1]) {
            return std::vesper_unexpected(error{
                error_code::invalid_argument,
                "Dimensions must be in ascending order",
                "matryoshka"
            });
        }
    }
    
    impl_->full_dimension = full_dimension;
    impl_->config = config;
    impl_->compute_dimension_scales();
    impl_->initialized = true;
    
    return {};
}

auto MatryoshkaEmbedding::analyze(const float* embeddings, std::size_t n,
                                  const std::uint64_t* ground_truth)
    -> std::expected<MatryoshkaStats, core::error> {
    using core::error;
    using core::error_code;
    
    if (!impl_->initialized) {
        return std::vesper_unexpected(error{
            error_code::precondition_failed,
            "MatryoshkaEmbedding not initialized",
            "matryoshka"
        });
    }
    
    if (!embeddings || n == 0) {
        return std::vesper_unexpected(error{
            error_code::invalid_argument,
            "Invalid embeddings data",
            "matryoshka"
        });
    }
    
    MatryoshkaStats stats;
    stats.dimension_importance.resize(impl_->config.dimensions.size());
    stats.dimension_variance.resize(impl_->config.dimensions.size());
    stats.recall_per_dimension.resize(impl_->config.dimensions.size());
    
    // Compute variance explained at each dimension level
    std::vector<float> mean(impl_->full_dimension, 0.0f);
    
    // Compute mean
    for (std::size_t i = 0; i < n; ++i) {
        for (std::uint32_t j = 0; j < impl_->full_dimension; ++j) {
            mean[j] += embeddings[i * impl_->full_dimension + j];
        }
    }
    for (auto& m : mean) {
        m /= n;
    }
    
    // Compute variance at each dimension level
    for (std::size_t d_idx = 0; d_idx < impl_->config.dimensions.size(); ++d_idx) {
        std::uint32_t dim = impl_->config.dimensions[d_idx];
        float variance = 0.0f;
        
        for (std::size_t i = 0; i < n; ++i) {
            for (std::uint32_t j = 0; j < dim; ++j) {
                float diff = embeddings[i * impl_->full_dimension + j] - mean[j];
                variance += diff * diff;
            }
        }
        
        stats.dimension_variance[d_idx] = variance / (n * dim);
        
        // Importance is proportional to variance
        stats.dimension_importance[d_idx] = stats.dimension_variance[d_idx];
        
        // Estimate recall (simplified - in practice would need actual search)
        if (ground_truth) {
            // TODO: Implement actual recall computation
            stats.recall_per_dimension[d_idx] = 0.9f + (0.05f * d_idx / impl_->config.dimensions.size());
        } else {
            // Heuristic based on dimension ratio
            float dim_ratio = static_cast<float>(dim) / impl_->full_dimension;
            stats.recall_per_dimension[d_idx] = std::min(0.99f, dim_ratio * 1.1f);
        }
    }
    
    // Find optimal dimension for target recall
    for (std::size_t i = 0; i < stats.recall_per_dimension.size(); ++i) {
        if (stats.recall_per_dimension[i] >= impl_->config.target_recall) {
            stats.optimal_dimension = impl_->config.dimensions[i];
            stats.estimated_recall = stats.recall_per_dimension[i];
            stats.compression_ratio = static_cast<float>(impl_->full_dimension) / stats.optimal_dimension;
            break;
        }
    }
    
    // If no dimension meets target recall, use the largest
    if (stats.optimal_dimension == 0 && !impl_->config.dimensions.empty()) {
        stats.optimal_dimension = impl_->config.dimensions.back();
        stats.estimated_recall = stats.recall_per_dimension.back();
        stats.compression_ratio = static_cast<float>(impl_->full_dimension) / stats.optimal_dimension;
    }
    
    return stats;
}

auto MatryoshkaEmbedding::truncate(std::span<const float> full_embedding, std::uint32_t target_dim) const
    -> std::expected<std::vector<float>, core::error> {
    using core::error;
    using core::error_code;
    
    if (!impl_->initialized) {
        return std::vesper_unexpected(error{
            error_code::precondition_failed,
            "MatryoshkaEmbedding not initialized",
            "matryoshka"
        });
    }
    
    if (full_embedding.size() < target_dim) {
        return std::vesper_unexpected(error{
            error_code::invalid_argument,
            "Target dimension exceeds embedding size",
            "matryoshka"
        });
    }
    
    // Check if target_dim is in configured dimensions
    auto it = std::find(impl_->config.dimensions.begin(), 
                        impl_->config.dimensions.end(), 
                        target_dim);
    if (it == impl_->config.dimensions.end()) {
        return std::vesper_unexpected(error{
            error_code::invalid_argument,
            "Target dimension not in configured dimensions",
            "matryoshka"
        });
    }
    
    std::vector<float> truncated(full_embedding.begin(), full_embedding.begin() + target_dim);
    
    // Normalize if configured
    if (impl_->config.normalize_per_dimension) {
        float norm = 0.0f;
        for (const auto& val : truncated) {
            norm += val * val;
        }
        norm = std::sqrt(norm);
        
        if (norm > 0.0f) {
            for (auto& val : truncated) {
                val /= norm;
            }
        }
    }
    
    return truncated;
}

auto MatryoshkaEmbedding::truncate_batch(const float* full_embeddings, std::size_t n,
                                         std::uint32_t target_dim) const
    -> std::expected<std::vector<float>, core::error> {
    using core::error;
    using core::error_code;
    
    if (!impl_->initialized) {
        return std::vesper_unexpected(error{
            error_code::precondition_failed,
            "MatryoshkaEmbedding not initialized",
            "matryoshka"
        });
    }
    
    if (!full_embeddings || n == 0) {
        return std::vesper_unexpected(error{
            error_code::invalid_argument,
            "Invalid input embeddings",
            "matryoshka"
        });
    }
    
    std::vector<float> truncated_batch;
    truncated_batch.reserve(n * target_dim);
    
    for (std::size_t i = 0; i < n; ++i) {
        auto truncated = truncate(
            std::span<const float>(full_embeddings + i * impl_->full_dimension, 
                                   impl_->full_dimension),
            target_dim
        );
        
        if (!truncated) {
            return std::vesper_unexpected(truncated.error());
        }
        
        truncated_batch.insert(truncated_batch.end(), 
                              truncated.value().begin(), 
                              truncated.value().end());
    }
    
    return truncated_batch;
}

auto MatryoshkaEmbedding::distance_at_dimension(std::span<const float> a,
                                               std::span<const float> b,
                                               std::uint32_t dim) const -> float {
    if (!impl_->initialized || dim == 0) {
        return std::numeric_limits<float>::infinity();
    }
    
    std::uint32_t min_dim = std::min({dim, 
                                      static_cast<std::uint32_t>(a.size()), 
                                      static_cast<std::uint32_t>(b.size())});
    
    float distance = 0.0f;
    
    if (impl_->config.normalize_per_dimension) {
        // Cosine distance for normalized vectors
        float dot_product = 0.0f;
        float norm_a = 0.0f;
        float norm_b = 0.0f;
        
        for (std::uint32_t i = 0; i < min_dim; ++i) {
            dot_product += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        
        norm_a = std::sqrt(norm_a);
        norm_b = std::sqrt(norm_b);
        
        if (norm_a > 0.0f && norm_b > 0.0f) {
            distance = 1.0f - (dot_product / (norm_a * norm_b));
        } else {
            distance = 1.0f;
        }
    } else {
        // L2 distance
        for (std::uint32_t i = 0; i < min_dim; ++i) {
            float diff = a[i] - b[i];
            distance += diff * diff;
        }
        distance = std::sqrt(distance);
    }
    
    return distance;
}

auto MatryoshkaEmbedding::progressive_search(std::span<const float> query,
                                            const float* database, std::size_t n,
                                            std::size_t k, std::uint32_t start_dim) const
    -> std::vector<std::pair<std::uint64_t, float>> {
    if (!impl_->initialized || !database || n == 0 || k == 0) {
        return {};
    }

    std::vector<std::pair<std::uint64_t, float>> results;
    results.reserve(std::min<std::size_t>(k * 10, n));

    // Determine starting index based on start_dim (find first >= start_dim)
    std::size_t start_idx = 0;
    if (start_dim > 0) {
        while (start_idx < impl_->config.dimensions.size() && impl_->config.dimensions[start_idx] < start_dim) {
            ++start_idx;
        }
        if (start_idx >= impl_->config.dimensions.size()) {
            // If start_dim is larger than any configured dimension, start from last
            start_idx = impl_->config.dimensions.size() - 1;
        }
    }

    // Progressive refinement through configured dimensions starting at start_idx
    for (std::size_t di = start_idx; di < impl_->config.dimensions.size(); ++di) {
        const auto dim = impl_->config.dimensions[di];
        results.clear();

        // Compute distances at current dimension
        for (std::size_t i = 0; i < n; ++i) {
            float dist = distance_at_dimension(
                query.subspan(0, dim),
                std::span<const float>(database + i * impl_->full_dimension, dim),
                dim
            );
            results.emplace_back(static_cast<std::uint64_t>(i), dist);
        }

        // Sort and keep top candidates (k*10 heuristic like before)
        std::size_t num_candidates = std::min<std::size_t>(k * 10, results.size());
        std::partial_sort(results.begin(), results.begin() + num_candidates, results.end(),
                          [](const auto& a, const auto& b) { return a.second < b.second; });
        results.resize(num_candidates);
    }

    // Final refinement with full dimension if needed
    if (!impl_->config.dimensions.empty() && impl_->config.dimensions.back() < impl_->full_dimension) {
        for (auto& kv : results) {
            auto idx = kv.first;
            kv.second = distance_at_dimension(
                query,
                std::span<const float>(database + idx * impl_->full_dimension, impl_->full_dimension),
                impl_->full_dimension
            );
        }
        std::sort(results.begin(), results.end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });
    }

    if (results.size() > k) {
        results.resize(k);
    }
    return results;
}

// Methods not declared in header removed

} // namespace vesper::index