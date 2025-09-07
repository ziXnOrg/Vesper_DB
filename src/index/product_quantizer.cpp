/** \file product_quantizer.cpp
 *  \brief Implementation of Product Quantization for vector compression.
 */

#include "vesper/index/product_quantizer.hpp"
#include "vesper/index/kmeans.hpp"
#include "vesper/index/aligned_buffer.hpp"
#include "vesper/kernels/dispatch.hpp"
#include <expected>

#include <algorithm>
#include <numeric>
#include <cstring>
#include <random>
#include <cmath>
#include <unordered_set>

namespace vesper::index {

/** \brief Implementation class for ProductQuantizer. */
class ProductQuantizer::Impl {
public:
    Impl() = default;
    ~Impl() = default;
    
    auto train(const float* data, std::size_t n, std::size_t dim,
               const PqTrainParams& params)
        -> std::expected<void, core::error> {
        
        using core::error;
        using core::error_code;
        
        // Validate parameters
        if (dim % params.m != 0) {
            return std::vesper_unexpected(error{
                error_code::precondition_failed,
                "Dimension must be divisible by number of subquantizers",
                "product_quantizer"
            });
        }
        
        const std::uint32_t ksub = 1U << params.nbits;
        if (n < ksub * params.m) {
            return std::vesper_unexpected(error{
                error_code::precondition_failed,
                "Need at least ksub * m training vectors",
                "product_quantizer"
            });
        }
        
        // Store parameters
        m_ = params.m;
        nbits_ = params.nbits;
        ksub_ = ksub;
        dsub_ = dim / params.m;
        dim_ = dim;
        trained_ = false;
        
        // Allocate codebooks
        codebooks_ = std::make_unique<AlignedCentroidBuffer>(m_ * ksub_, dsub_);
        
        // Train each subquantizer independently
        std::mt19937 rng(params.seed);
        
        for (std::uint32_t sq = 0; sq < m_; ++sq) {
            // Extract subvectors for this subquantizer
            std::vector<float> subvectors(n * dsub_);
            for (std::size_t i = 0; i < n; ++i) {
                std::memcpy(subvectors.data() + i * dsub_,
                           data + i * dim_ + sq * dsub_,
                           dsub_ * sizeof(float));
            }
            
            // Run k-means on subvectors
            KmeansParams kmeans_params;
            kmeans_params.k = ksub_;
            kmeans_params.max_iter = params.max_iter;
            kmeans_params.epsilon = params.epsilon;
            kmeans_params.seed = params.seed + sq;
            kmeans_params.verbose = params.verbose && (sq == 0);
            
            auto result = kmeans_cluster(
                subvectors.data(), n, dsub_,
                kmeans_params
            );
            
            if (!result) {
                return std::vesper_unexpected(result.error());
            }
            
            // Copy centroids to codebook
            for (std::uint32_t k = 0; k < ksub_; ++k) {
                auto dest = codebooks_->get_centroid(sq * ksub_ + k);
                std::memcpy(dest.data(),
                           result->centroids[k].data(),
                           dsub_ * sizeof(float));
            }
        }
        
        trained_ = true;
        return {};
    }
    
    auto train_opq(const float* data, std::size_t n, std::size_t dim,
                   const PqTrainParams& pq_params, const OpqParams& opq_params)
        -> std::expected<void, core::error> {
        
        // TODO: Implement OPQ with alternating optimization
        // For now, just do regular PQ training
        return train(data, n, dim, pq_params);
    }
    
    auto encode(const float* data, std::size_t n, std::uint8_t* codes) const
        -> std::expected<void, core::error> {
        
        using core::error;
        using core::error_code;
        
        if (!trained_) {
            return std::vesper_unexpected(error{
                error_code::precondition_failed,
                "Quantizer not trained",
                "product_quantizer"
            });
        }
        
        const auto& ops = kernels::select_backend_auto();
        
        // Encode each vector
        #pragma omp parallel for schedule(dynamic, 64)
        for (int i = 0; i < static_cast<int>(n); ++i) {
            encode_one_impl(data + static_cast<std::size_t>(i) * dim_, 
                           codes + static_cast<std::size_t>(i) * m_, 
                           ops);
        }
        
        return {};
    }
    
    auto encode_one(const float* vec, std::uint8_t* code) const
        -> std::expected<void, core::error> {
        
        using core::error;
        using core::error_code;
        
        if (!trained_) {
            return std::vesper_unexpected(error{
                error_code::precondition_failed,
                "Quantizer not trained",
                "product_quantizer"
            });
        }
        
        const auto& ops = kernels::select_backend_auto();
        encode_one_impl(vec, code, ops);
        return {};
    }
    
    auto decode(const std::uint8_t* codes, std::size_t n, float* data) const
        -> std::expected<void, core::error> {
        
        using core::error;
        using core::error_code;
        
        if (!trained_) {
            return std::vesper_unexpected(error{
                error_code::precondition_failed,
                "Quantizer not trained",
                "product_quantizer"
            });
        }
        
        // Decode each vector
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < static_cast<int>(n); ++i) {
            decode_one_impl(codes + static_cast<std::size_t>(i) * m_,
                           data + static_cast<std::size_t>(i) * dim_);
        }
        
        return {};
    }
    
    auto compute_distance_table(const float* query, float* table) const
        -> std::expected<void, core::error> {
        
        using core::error;
        using core::error_code;
        
        if (!trained_) {
            return std::vesper_unexpected(error{
                error_code::precondition_failed,
                "Quantizer not trained",
                "product_quantizer"
            });
        }
        
        const auto& ops = kernels::select_backend_auto();
        
        // Compute distances from query subvectors to all centroids
        for (std::uint32_t sq = 0; sq < m_; ++sq) {
            const float* query_sub = query + sq * dsub_;
            float* table_sub = table + sq * ksub_;
            
            for (std::uint32_t k = 0; k < ksub_; ++k) {
                const float* centroid = codebooks_->get_centroid(sq * ksub_ + k).data();
                table_sub[k] = ops.l2_sq(
                    std::span(query_sub, dsub_),
                    std::span(centroid, dsub_)
                );
            }
        }
        
        return {};
    }
    
    auto compute_distances_adc(const float* table, const std::uint8_t* codes,
                               std::size_t n, float* distances) const
        -> std::expected<void, core::error> {
        
        using core::error;
        using core::error_code;
        
        if (!trained_) {
            return std::vesper_unexpected(error{
                error_code::precondition_failed,
                "Quantizer not trained",
                "product_quantizer"
            });
        }
        
        // Fast distance computation using lookup table
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < static_cast<int>(n); ++i) {
            const std::uint8_t* code = codes + static_cast<std::size_t>(i) * m_;
            float dist = 0.0f;
            
            for (std::uint32_t sq = 0; sq < m_; ++sq) {
                dist += table[sq * ksub_ + code[sq]];
            }
            
            distances[i] = dist;
        }
        
        return {};
    }
    
    auto compute_distance_symmetric(const std::uint8_t* code1,
                                    const std::uint8_t* code2) const -> float {
        if (!trained_) return 0.0f;
        
        const auto& ops = kernels::select_backend_auto();
        float dist = 0.0f;
        
        for (std::uint32_t sq = 0; sq < m_; ++sq) {
            const float* centroid1 = codebooks_->get_centroid(sq * ksub_ + code1[sq]).data();
            const float* centroid2 = codebooks_->get_centroid(sq * ksub_ + code2[sq]).data();
            
            dist += ops.l2_sq(
                std::span(centroid1, dsub_),
                std::span(centroid2, dsub_)
            );
        }
        
        return dist;
    }
    
    auto get_info() const noexcept -> ProductQuantizer::Info {
        return {
            .m = m_,
            .ksub = ksub_,
            .dsub = dsub_,
            .dim = dim_,
            .has_rotation = has_rotation_
        };
    }
    
    auto is_trained() const noexcept -> bool {
        return trained_;
    }
    
    auto code_size() const noexcept -> std::size_t {
        return m_;
    }
    
    auto compute_quantization_error(const float* data, std::size_t n) const -> float {
        if (!trained_) return 0.0f;
        
        const auto& ops = kernels::select_backend_auto();
        double total_error = 0.0;
        
        #pragma omp parallel for reduction(+:total_error)
        for (int i = 0; i < static_cast<int>(n); ++i) {
            std::vector<std::uint8_t> code(m_);
            const auto& ops2 = kernels::select_backend_auto();
            encode_one_impl(data + static_cast<std::size_t>(i) * dim_, 
                           code.data(), ops2);
            
            std::vector<float> reconstructed(dim_);
            decode_one_impl(code.data(), reconstructed.data());
            
            float error = ops.l2_sq(
                std::span(data + static_cast<std::size_t>(i) * dim_, dim_),
                std::span(reconstructed.data(), dim_)
            );
            total_error += error;
        }
        
        return static_cast<float>(total_error / n);
    }
    
    auto save(const std::string& path) const -> std::expected<void, core::error> {
        // TODO: Implement save to file
        (void)path;
        return {};
    }
    
    static auto load(const std::string& path) 
        -> std::expected<std::unique_ptr<Impl>, core::error> {
        // TODO: Implement load from file
        (void)path;
        auto impl = std::make_unique<Impl>();
        return impl;
    }
    
private:
    void encode_one_impl(const float* vec, std::uint8_t* code,
                        const kernels::KernelOps& ops) const {
        for (std::uint32_t sq = 0; sq < m_; ++sq) {
            const float* vec_sub = vec + sq * dsub_;
            
            // Find nearest centroid
            float min_dist = std::numeric_limits<float>::max();
            std::uint8_t best_idx = 0;
            
            for (std::uint32_t k = 0; k < ksub_; ++k) {
                const float* centroid = codebooks_->get_centroid(sq * ksub_ + k).data();
                float dist = ops.l2_sq(
                    std::span(vec_sub, dsub_),
                    std::span(centroid, dsub_)
                );
                
                if (dist < min_dist) {
                    min_dist = dist;
                    best_idx = static_cast<std::uint8_t>(k);
                }
            }
            
            code[sq] = best_idx;
        }
    }
    
    void decode_one_impl(const std::uint8_t* code, float* vec) const {
        for (std::uint32_t sq = 0; sq < m_; ++sq) {
            const float* centroid = codebooks_->get_centroid(sq * ksub_ + code[sq]).data();
            std::memcpy(vec + sq * dsub_, centroid, dsub_ * sizeof(float));
        }
    }
    
private:
    std::uint32_t m_{0};                   // Number of subquantizers
    std::uint32_t nbits_{0};               // Bits per subquantizer  
    std::uint32_t ksub_{0};                // Codebook size per subquantizer
    std::uint32_t dsub_{0};                // Subspace dimension
    std::size_t dim_{0};                   // Total dimension
    bool trained_{false};                  // Is quantizer trained
    bool has_rotation_{false};             // OPQ rotation applied
    
    std::unique_ptr<AlignedCentroidBuffer> codebooks_;  // Codebooks [m * ksub x dsub]
    std::vector<float> rotation_matrix_;                // OPQ rotation [dim x dim]
};

// ProductQuantizer public interface implementation

ProductQuantizer::ProductQuantizer() : impl_(nullptr) {}

ProductQuantizer::~ProductQuantizer() = default;

ProductQuantizer::ProductQuantizer(ProductQuantizer&&) noexcept = default;

ProductQuantizer& ProductQuantizer::operator=(ProductQuantizer&&) noexcept = default;

auto ProductQuantizer::train(const float* data, std::size_t n, std::size_t dim,
                             const PqTrainParams& params)
    -> std::expected<void, core::error> {
    if (!impl_) {
        impl_ = std::make_unique<Impl>();
    }
    return impl_->train(data, n, dim, params);
}

auto ProductQuantizer::train_opq(const float* data, std::size_t n, std::size_t dim,
                                 const PqTrainParams& pq_params, const OpqParams& opq_params)
    -> std::expected<void, core::error> {
    if (!impl_) {
        impl_ = std::make_unique<Impl>();
    }
    return impl_->train_opq(data, n, dim, pq_params, opq_params);
}

auto ProductQuantizer::encode(const float* data, std::size_t n, std::uint8_t* codes) const
    -> std::expected<void, core::error> {
    if (!impl_) {
        return std::vesper_unexpected(core::error{
            core::error_code::precondition_failed,
            "Quantizer not initialized",
            "product_quantizer"
        });
    }
    return impl_->encode(data, n, codes);
}

auto ProductQuantizer::encode_one(const float* vec, std::uint8_t* code) const
    -> std::expected<void, core::error> {
    if (!impl_) {
        return std::vesper_unexpected(core::error{
            core::error_code::precondition_failed,
            "Quantizer not initialized",
            "product_quantizer"
        });
    }
    return impl_->encode_one(vec, code);
}

auto ProductQuantizer::decode(const std::uint8_t* codes, std::size_t n, float* data) const
    -> std::expected<void, core::error> {
    if (!impl_) {
        return std::vesper_unexpected(core::error{
            core::error_code::precondition_failed,
            "Quantizer not initialized",
            "product_quantizer"
        });
    }
    return impl_->decode(codes, n, data);
}

auto ProductQuantizer::compute_distance_table(const float* query, float* table) const
    -> std::expected<void, core::error> {
    if (!impl_) {
        return std::vesper_unexpected(core::error{
            core::error_code::precondition_failed,
            "Quantizer not initialized",
            "product_quantizer"
        });
    }
    return impl_->compute_distance_table(query, table);
}

auto ProductQuantizer::compute_distances_adc(const float* table, const std::uint8_t* codes,
                                             std::size_t n, float* distances) const
    -> std::expected<void, core::error> {
    if (!impl_) {
        return std::vesper_unexpected(core::error{
            core::error_code::precondition_failed,
            "Quantizer not initialized",
            "product_quantizer"
        });
    }
    return impl_->compute_distances_adc(table, codes, n, distances);
}

auto ProductQuantizer::compute_distance_symmetric(const std::uint8_t* code1,
                                                  const std::uint8_t* code2) const -> float {
    if (!impl_) return 0.0f;
    return impl_->compute_distance_symmetric(code1, code2);
}

auto ProductQuantizer::get_info() const noexcept -> Info {
    if (!impl_) return {};
    return impl_->get_info();
}

auto ProductQuantizer::is_trained() const noexcept -> bool {
    if (!impl_) return false;
    return impl_->is_trained();
}

auto ProductQuantizer::code_size() const noexcept -> std::size_t {
    if (!impl_) return 0;
    return impl_->code_size();
}

auto ProductQuantizer::compute_quantization_error(const float* data, std::size_t n) const -> float {
    if (!impl_) return 0.0f;
    return impl_->compute_quantization_error(data, n);
}

auto ProductQuantizer::save(const std::string& path) const -> std::expected<void, core::error> {
    if (!impl_) {
        return std::vesper_unexpected(core::error{
            core::error_code::precondition_failed,
            "Quantizer not initialized",
            "product_quantizer"
        });
    }
    return impl_->save(path);
}

auto ProductQuantizer::load(const std::string& path) 
    -> std::expected<ProductQuantizer, core::error> {
    auto impl_result = Impl::load(path);
    if (!impl_result) {
        return std::vesper_unexpected(impl_result.error());
    }
    
    ProductQuantizer pq;
    pq.impl_ = std::move(*impl_result);
    return std::move(pq);
}

// Helper function implementation
auto compute_pq_recall(const ProductQuantizer& pq,
                      const float* data, std::size_t n,
                      const float* queries, std::size_t nq,
                      std::size_t k) -> float {
    if (!pq.is_trained() || n == 0 || nq == 0 || k == 0) {
        return 0.0f;
    }
    
    const auto& ops = kernels::select_backend_auto();
    const auto info = pq.get_info();
    
    // Encode all data
    std::vector<std::uint8_t> codes(n * pq.code_size());
    auto encode_result = pq.encode(data, n, codes.data());
    if (!encode_result) {
        return 0.0f;
    }
    
    float total_recall = 0.0f;
    
    #pragma omp parallel for reduction(+:total_recall)
    for (int q = 0; q < static_cast<int>(nq); ++q) {
        const float* query = queries + static_cast<std::size_t>(q) * info.dim;
        
        // Compute exact neighbors
        std::vector<std::pair<float, std::size_t>> exact_dists;
        exact_dists.reserve(n);
        
        for (std::size_t i = 0; i < n; ++i) {
            float dist = ops.l2_sq(
                std::span(query, info.dim),
                std::span(data + i * info.dim, info.dim)
            );
            exact_dists.emplace_back(dist, i);
        }
        
        std::partial_sort(exact_dists.begin(), 
                         exact_dists.begin() + std::min(k, n),
                         exact_dists.end());
        
        std::unordered_set<std::size_t> exact_set;
        for (std::size_t i = 0; i < std::min(k, n); ++i) {
            exact_set.insert(exact_dists[i].second);
        }
        
        // Compute PQ neighbors
        std::vector<float> distance_table(pq.code_size() * (1U << 8));
        pq.compute_distance_table(query, distance_table.data());
        
        std::vector<float> pq_distances(n);
        pq.compute_distances_adc(distance_table.data(), codes.data(), n, pq_distances.data());
        
        std::vector<std::pair<float, std::size_t>> pq_dists;
        pq_dists.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            pq_dists.emplace_back(pq_distances[i], i);
        }
        
        std::partial_sort(pq_dists.begin(),
                         pq_dists.begin() + std::min(k, n),
                         pq_dists.end());
        
        // Compute recall
        std::size_t hits = 0;
        for (std::size_t i = 0; i < std::min(k, n); ++i) {
            if (exact_set.count(pq_dists[i].second)) {
                hits++;
            }
        }
        
        total_recall += static_cast<float>(hits) / std::min(k, n);
    }
    
    return total_recall / nq;
}

} // namespace vesper::index