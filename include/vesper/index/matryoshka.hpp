/** \file matryoshka.hpp
 *  \brief Matryoshka Representation Learning for flexible dimensionality
 *
 * Implements support for Matryoshka embeddings that encode information at
 * multiple granularities, allowing truncation without retraining.
 *
 * Features:
 * - Dynamic dimensionality reduction at query time
 * - Adaptive dimension selection based on recall requirements
 * - Efficient storage with dimension-aware indexing
 * - Compatible with all index types (HNSW, IVF-PQ, DiskGraph)
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <vector>
#include <optional>
#include <unordered_map>

#include "vesper/error.hpp"

namespace vesper::index {

/** \brief Matryoshka embedding configuration */
struct MatryoshkaConfig {
    std::vector<std::uint32_t> dimensions{32, 64, 128, 256, 512, 1024};  /**< Nested dimensions */
    bool normalize_per_dimension{true};      /**< Normalize at each dimension level */
    float dimension_weights_decay{0.9f};     /**< Weight decay for higher dimensions */
    bool use_adaptive_selection{true};       /**< Auto-select dimension based on data */
    float target_recall{0.95f};             /**< Target recall for adaptive selection */
};

/** \brief Statistics for Matryoshka embeddings */
struct MatryoshkaStats {
    std::vector<float> dimension_importance;    /**< Importance score per dimension level */
    std::vector<float> dimension_variance;      /**< Variance explained at each level */
    std::vector<float> recall_per_dimension;    /**< Measured recall at each dimension level */
    std::uint32_t optimal_dimension{0};         /**< Recommended dimension for target recall */
    float compression_ratio{1.0f};              /**< Achieved compression vs full dimension */
    float estimated_recall{0.0f};               /**< Estimated recall at optimal dimension */
};

/** \brief Matryoshka embedding wrapper for flexible dimensionality
 *
 * Enables using different dimensions of the same embedding without retraining.
 * Particularly useful for memory-constrained environments or adaptive search.
 */
class MatryoshkaEmbedding {
public:
    MatryoshkaEmbedding();
    ~MatryoshkaEmbedding();
    MatryoshkaEmbedding(MatryoshkaEmbedding&&) noexcept;
    MatryoshkaEmbedding& operator=(MatryoshkaEmbedding&&) noexcept;
    MatryoshkaEmbedding(const MatryoshkaEmbedding&) = delete;
    MatryoshkaEmbedding& operator=(const MatryoshkaEmbedding&) = delete;
    
    /** \brief Initialize with configuration
     *
     * \param full_dimension Maximum embedding dimension
     * \param config Matryoshka configuration
     * \return Success or error
     */
    auto init(std::uint32_t full_dimension, const MatryoshkaConfig& config = {})
        -> std::expected<void, core::error>;
    
    /** \brief Analyze embeddings to determine optimal truncation
     *
     * \param embeddings Full embeddings [n x full_dim]
     * \param n Number of embeddings
     * \param ground_truth Optional ground truth for recall estimation
     * \return Statistics and recommendations
     *
     * Analyzes variance and information content at each dimension level.
     */
    auto analyze(const float* embeddings, std::size_t n,
                 const std::uint64_t* ground_truth = nullptr)
        -> std::expected<MatryoshkaStats, core::error>;
    
    /** \brief Truncate embedding to specified dimension
     *
     * \param full_embedding Full dimension embedding
     * \param target_dim Target dimension (must be in config.dimensions)
     * \return Truncated and normalized embedding
     */
    auto truncate(std::span<const float> full_embedding, std::uint32_t target_dim) const
        -> std::expected<std::vector<float>, core::error>;
    
    /** \brief Batch truncate embeddings
     *
     * \param full_embeddings Full embeddings [n x full_dim]
     * \param n Number of embeddings
     * \param target_dim Target dimension
     * \return Truncated embeddings [n x target_dim]
     */
    auto truncate_batch(const float* full_embeddings, std::size_t n,
                       std::uint32_t target_dim) const
        -> std::expected<std::vector<float>, core::error>;
    
    /** \brief Compute distance between embeddings at specified dimension
     *
     * \param a First embedding (full or truncated)
     * \param b Second embedding (full or truncated)
     * \param dim Dimension to use for comparison
     * \return Distance (L2 or cosine based on normalization)
     */
    auto distance_at_dimension(std::span<const float> a,
                              std::span<const float> b,
                              std::uint32_t dim) const -> float;
    
    /** \brief Progressive search with dimension escalation
     *
     * \param query Query embedding (full dimension)
     * \param database Database embeddings [n x full_dim]
     * \param n Number of database vectors
     * \param k Number of neighbors to find
     * \param start_dim Starting dimension (smallest)
     * \return Neighbors found with progressive refinement
     *
     * Starts with low dimension for speed, refines with higher dimensions.
     */
    auto progressive_search(std::span<const float> query,
                          const float* database, std::size_t n,
                          std::size_t k, std::uint32_t start_dim = 0) const
        -> std::vector<std::pair<std::uint64_t, float>>;
    
    /** \brief Get valid truncation dimensions */
    auto get_dimensions() const noexcept -> const std::vector<std::uint32_t>&;
    
    /** \brief Get full embedding dimension */
    auto full_dimension() const noexcept -> std::uint32_t;
    
    /** \brief Check if dimension is valid for truncation */
    auto is_valid_dimension(std::uint32_t dim) const noexcept -> bool;
    
    /** \brief Get memory savings for dimension */
    auto memory_savings(std::uint32_t dim) const noexcept -> float;
    
    /** \brief Estimate recall loss for dimension reduction
     *
     * \param from_dim Source dimension
     * \param to_dim Target dimension
     * \return Estimated recall loss (0-1)
     */
    auto estimate_recall_loss(std::uint32_t from_dim, std::uint32_t to_dim) const
        -> float;
    
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/** \brief Index adapter for Matryoshka embeddings */
template<typename IndexType>
class MatryoshkaIndexAdapter {
public:
    MatryoshkaIndexAdapter(std::shared_ptr<IndexType> index,
                           std::shared_ptr<MatryoshkaEmbedding> matryoshka)
        : index_(index), matryoshka_(matryoshka) {}
    
    /** \brief Add vector with automatic dimension selection
     *
     * \param id Vector ID
     * \param full_embedding Full dimension embedding
     * \return Success or error
     *
     * Automatically selects optimal dimension based on index capacity.
     */
    auto add(std::uint64_t id, std::span<const float> full_embedding)
        -> std::expected<void, core::error> {
        
        // Select dimension based on index state
        auto dim = select_dimension_for_capacity();
        
        // Truncate to selected dimension
        auto truncated = matryoshka_->truncate(full_embedding, dim);
        if (!truncated) {
            return std::vesper_unexpected(truncated.error());
        }
        
        // Add to underlying index
        return index_->add(id, truncated.value().data());
    }
    
    /** \brief Search with progressive dimension refinement
     *
     * \param query Query embedding (full dimension)
     * \param k Number of neighbors
     * \param use_progressive Use progressive refinement
     * \return Search results
     */
    auto search(std::span<const float> query, std::size_t k,
               bool use_progressive = true)
        -> std::expected<std::vector<std::pair<std::uint64_t, float>>, core::error> {
        
        if (use_progressive) {
            // Start with coarse search, refine with higher dimensions
            auto dims = matryoshka_->get_dimensions();
            std::vector<std::pair<std::uint64_t, float>> results;
            
            for (auto dim : dims) {
                auto truncated_query = matryoshka_->truncate(query, dim);
                if (!truncated_query) {
                    return std::vesper_unexpected(truncated_query.error());
                }
                
                // Search at current dimension
                auto dim_results = index_->search(truncated_query.value().data(), k * 2);
                if (!dim_results) {
                    return std::vesper_unexpected(dim_results.error());
                }
                
                // Merge and refine results
                results = merge_results(results, dim_results.value());
                
                // Stop if we have high confidence
                if (results.size() >= k && estimate_confidence(results) > 0.95f) {
                    break;
                }
            }
            
            // Return top-k
            if (results.size() > k) {
                results.resize(k);
            }
            return results;
            
        } else {
            // Direct search at current index dimension
            auto current_dim = get_current_dimension();
            auto truncated = matryoshka_->truncate(query, current_dim);
            if (!truncated) {
                return std::vesper_unexpected(truncated.error());
            }
            
            return index_->search(truncated.value().data(), k);
        }
    }
    
private:
    std::shared_ptr<IndexType> index_;
    std::shared_ptr<MatryoshkaEmbedding> matryoshka_;
    std::uint32_t current_dimension_{0};
    
    auto select_dimension_for_capacity() -> std::uint32_t {
        // Logic to select dimension based on memory/performance constraints
        // This is a simplified version
        auto dims = matryoshka_->get_dimensions();
        return dims[dims.size() / 2]; // Use middle dimension by default
    }
    
    auto get_current_dimension() const -> std::uint32_t {
        return current_dimension_;
    }
    
    auto merge_results(const std::vector<std::pair<std::uint64_t, float>>& a,
                      const std::vector<std::pair<std::uint64_t, float>>& b)
        -> std::vector<std::pair<std::uint64_t, float>> {
        // Merge two result sets, keeping unique IDs with best scores
        std::unordered_map<std::uint64_t, float> merged;
        
        for (const auto& [id, dist] : a) {
            merged[id] = dist;
        }
        
        for (const auto& [id, dist] : b) {
            if (merged.find(id) == merged.end() || merged[id] > dist) {
                merged[id] = dist;
            }
        }
        
        std::vector<std::pair<std::uint64_t, float>> result;
        for (const auto& [id, dist] : merged) {
            result.emplace_back(id, dist);
        }
        
        std::sort(result.begin(), result.end(),
                 [](const auto& a, const auto& b) { return a.second < b.second; });
        
        return result;
    }
    
    auto estimate_confidence(const std::vector<std::pair<std::uint64_t, float>>& results)
        -> float {
        // Simple confidence estimation based on distance distribution
        if (results.size() < 2) return 0.0f;
        
        float min_dist = results[0].second;
        float max_dist = results.back().second;
        float range = max_dist - min_dist;
        
        // High confidence if there's good separation
        return range > 0.5f ? 0.95f : 0.8f;
    }
};

} // namespace vesper::index