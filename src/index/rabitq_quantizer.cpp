/** \file rabitq_quantizer.cpp
 *  \brief Implementation of RaBitQ-inspired binary quantization
 */

#include "vesper/index/rabitq_quantizer.hpp"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <random>
#include <limits>

#ifdef _MSC_VER
#include <intrin.h>
#define __builtin_popcount __popcnt
#endif

namespace vesper::index {

// Forward declare the Impl class
class RaBitQuantizer::Impl {
public:
    RaBitQTrainParams params;
    std::vector<float> rotation_matrix;
    std::vector<float> thresholds;
    std::uint32_t dimension{0};
    bool trained{false};
};

RaBitQuantizer::RaBitQuantizer() 
    : impl_(std::make_unique<Impl>()) {}

RaBitQuantizer::~RaBitQuantizer() = default;

RaBitQuantizer::RaBitQuantizer(RaBitQuantizer&&) noexcept = default;

RaBitQuantizer& RaBitQuantizer::operator=(RaBitQuantizer&&) noexcept = default;

auto RaBitQuantizer::train(const float* data, std::size_t n, std::size_t dim,
                           const RaBitQTrainParams& params)
    -> std::expected<void, core::error> {
    using core::error;
    using core::error_code;
    
    if (!data || n == 0 || dim == 0) {
        return std::vesper_unexpected(error{
            error_code::invalid_argument,
            "Invalid training data",
            "rabitq"
        });
    }
    
    impl_->params = params;
    impl_->dimension = static_cast<std::uint32_t>(dim);
    
    // Initialize rotation matrix if needed
    if (params.use_rotation) {
        std::mt19937 rng(params.seed);
        std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt(static_cast<float>(dim)));
        
        impl_->rotation_matrix.resize(dim * dim);
        for (auto& val : impl_->rotation_matrix) {
            val = dist(rng);
        }
        
        // TODO: Orthogonalize the rotation matrix using QR decomposition
    }
    
    // Compute quantization thresholds
    impl_->thresholds.resize(dim);
    
    // Simple median-based thresholds for now
    std::vector<float> values;
    values.reserve(n);
    
    for (std::size_t d = 0; d < dim; ++d) {
        values.clear();
        for (std::size_t i = 0; i < n; ++i) {
            values.push_back(data[i * dim + d]);
        }
        std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
        impl_->thresholds[d] = values[values.size() / 2];
    }
    
    impl_->trained = true;
    return {};
}

auto RaBitQuantizer::quantize(std::span<const float> vec) const
    -> std::expected<QuantizedVector, core::error> {
    using core::error;
    using core::error_code;
    
    if (!impl_->trained) {
        return std::vesper_unexpected(error{
            error_code::precondition_failed,
            "Quantizer not trained",
            "rabitq"
        });
    }
    
    if (vec.size() != impl_->dimension) {
        return std::vesper_unexpected(error{
            error_code::invalid_argument,
            "Vector dimension mismatch",
            "rabitq"
        });
    }
    
    QuantizedVector result;
    result.dimension = impl_->dimension;
    result.bits = impl_->params.bits;
    
    // Apply rotation if needed
    std::vector<float> rotated;
    if (impl_->params.use_rotation && !impl_->rotation_matrix.empty()) {
        rotated.resize(impl_->dimension);
        // Simple matrix-vector multiplication
        for (std::uint32_t i = 0; i < impl_->dimension; ++i) {
            float sum = 0.0f;
            for (std::uint32_t j = 0; j < impl_->dimension; ++j) {
                sum += impl_->rotation_matrix[i * impl_->dimension + j] * vec[j];
            }
            rotated[i] = sum;
        }
    } else {
        rotated.assign(vec.begin(), vec.end());
    }
    
    // Quantize based on bit width
    switch (impl_->params.bits) {
    case QuantizationBits::BIT_1: {
        // Binary quantization
        std::size_t num_bytes = (impl_->dimension + 7) / 8;
        result.codes.resize(num_bytes, 0);
        
        for (std::uint32_t i = 0; i < impl_->dimension; ++i) {
            if (rotated[i] > impl_->thresholds[i]) {
                result.codes[i / 8] |= (1 << (i % 8));
            }
        }
        break;
    }
    
    case QuantizationBits::BIT_4: {
        // 4-bit quantization
        std::size_t num_bytes = (impl_->dimension + 1) / 2;
        result.codes.resize(num_bytes, 0);
        
        // Simple uniform quantization to 16 levels
        for (std::uint32_t i = 0; i < impl_->dimension; ++i) {
            float normalized = (rotated[i] - impl_->thresholds[i] + 1.0f) / 2.0f;
            normalized = std::clamp(normalized, 0.0f, 1.0f);
            std::uint8_t level = static_cast<std::uint8_t>(normalized * 15);
            
            if (i % 2 == 0) {
                result.codes[i / 2] = level;
            } else {
                result.codes[i / 2] |= (level << 4);
            }
        }
        break;
    }
    
    case QuantizationBits::BIT_8: {
        // 8-bit quantization
        result.codes.resize(impl_->dimension);
        
        // Simple uniform quantization to 256 levels
        for (std::uint32_t i = 0; i < impl_->dimension; ++i) {
            float normalized = (rotated[i] - impl_->thresholds[i] + 1.0f) / 2.0f;
            normalized = std::clamp(normalized, 0.0f, 1.0f);
            result.codes[i] = static_cast<std::uint8_t>(normalized * 255);
        }
        break;
    }
    }
    
    return result;
}

auto RaBitQuantizer::quantize_batch(const float* vecs, std::size_t n) const
    -> std::expected<std::vector<QuantizedVector>, core::error> {
    using core::error;
    using core::error_code;
    
    if (!impl_->trained) {
        return std::vesper_unexpected(error{
            error_code::precondition_failed,
            "Quantizer not trained",
            "rabitq"
        });
    }
    
    std::vector<QuantizedVector> results;
    results.reserve(n);
    
    for (std::size_t i = 0; i < n; ++i) {
        auto qvec = quantize(std::span<const float>(vecs + i * impl_->dimension, impl_->dimension));
        if (!qvec) {
            return std::vesper_unexpected(qvec.error());
        }
        results.push_back(std::move(qvec.value()));
    }
    
    return results;
}

auto RaBitQuantizer::reconstruct(const QuantizedVector& qvec) const
    -> std::expected<std::vector<float>, core::error> {
    using core::error;
    using core::error_code;
    
    if (!impl_->trained) {
        return std::vesper_unexpected(error{
            error_code::precondition_failed,
            "Quantizer not trained",
            "rabitq"
        });
    }
    
    std::vector<float> result(qvec.dimension);
    
    // Dequantize based on bit width
    switch (qvec.bits) {
    case QuantizationBits::BIT_1: {
        for (std::uint32_t i = 0; i < qvec.dimension; ++i) {
            bool bit = (qvec.codes[i / 8] >> (i % 8)) & 1;
            result[i] = bit ? impl_->thresholds[i] + 1.0f : impl_->thresholds[i] - 1.0f;
        }
        break;
    }
    
    case QuantizationBits::BIT_4: {
        for (std::uint32_t i = 0; i < qvec.dimension; ++i) {
            std::uint8_t level = (i % 2 == 0) 
                ? (qvec.codes[i / 2] & 0x0F)
                : ((qvec.codes[i / 2] >> 4) & 0x0F);
            float normalized = static_cast<float>(level) / 15.0f;
            result[i] = impl_->thresholds[i] + (normalized * 2.0f - 1.0f);
        }
        break;
    }
    
    case QuantizationBits::BIT_8: {
        for (std::uint32_t i = 0; i < qvec.dimension; ++i) {
            float normalized = static_cast<float>(qvec.codes[i]) / 255.0f;
            result[i] = impl_->thresholds[i] + (normalized * 2.0f - 1.0f);
        }
        break;
    }
    }
    
    // Apply inverse rotation if needed
    if (impl_->params.use_rotation && !impl_->rotation_matrix.empty()) {
        std::vector<float> temp = result;
        // Transpose multiplication for inverse (assuming orthogonal matrix)
        for (std::uint32_t i = 0; i < impl_->dimension; ++i) {
            float sum = 0.0f;
            for (std::uint32_t j = 0; j < impl_->dimension; ++j) {
                sum += impl_->rotation_matrix[j * impl_->dimension + i] * temp[j];
            }
            result[i] = sum;
        }
    }
    
    return result;
}

auto RaBitQuantizer::distance(const QuantizedVector& a, const QuantizedVector& b) const
    -> float {
    using core::error;
    using core::error_code;
    
    if (a.dimension != b.dimension || a.bits != b.bits) {
        return std::numeric_limits<float>::infinity();
    }
    
    float distance = 0.0f;
    
    switch (a.bits) {
    case QuantizationBits::BIT_1: {
        // Hamming distance for binary vectors
        std::uint32_t hamming = 0;
        for (std::size_t i = 0; i < a.codes.size(); ++i) {
            std::uint8_t xor_val = a.codes[i] ^ b.codes[i];
            // Count set bits (popcount)
            hamming += __builtin_popcount(xor_val);
        }
        distance = static_cast<float>(hamming);
        break;
    }
    
    case QuantizationBits::BIT_4:
    case QuantizationBits::BIT_8: {
        // L2 distance for scalar quantized vectors
        auto avec = reconstruct(a);
        auto bvec = reconstruct(b);
        if (!avec || !bvec) {
            return std::numeric_limits<float>::infinity();
        }
        
        for (std::uint32_t i = 0; i < a.dimension; ++i) {
            float diff = avec.value()[i] - bvec.value()[i];
            distance += diff * diff;
        }
        distance = std::sqrt(distance);
        break;
    }
    }
    
    return distance;
}

auto RaBitQuantizer::compute_stats(const float* original,
                                   const std::vector<QuantizedVector>& quantized,
                                   std::size_t n) const
    -> QuantizationStats {
    using core::error;
    using core::error_code;
    
    QuantizationStats stats;
    
    if (!original || quantized.empty() || n == 0) {
        return stats;
    }
    
    float total_mse = 0.0f;
    std::size_t total_memory = 0;
    
    for (std::size_t i = 0; i < n && i < quantized.size(); ++i) {
        auto reconstructed = reconstruct(quantized[i]);
        if (!reconstructed) continue;
        
        float sum_squared_error = 0.0f;
        for (std::uint32_t j = 0; j < quantized[i].dimension; ++j) {
            float error = original[i * quantized[i].dimension + j] - reconstructed.value()[j];
            float squared_error = error * error;
            sum_squared_error += squared_error;
            stats.max_error = std::max(stats.max_error, std::abs(error));
        }
        total_mse += sum_squared_error / quantized[i].dimension;
        total_memory += quantized[i].codes.size();
    }
    
    stats.mean_squared_error = total_mse / n;
    
    // Compute bit balance for binary quantization
    if (!quantized.empty() && quantized[0].bits == QuantizationBits::BIT_1) {
        std::uint32_t total_ones = 0;
        std::uint32_t total_bits = 0;
        for (const auto& qvec : quantized) {
            for (const auto& byte : qvec.codes) {
                total_ones += __builtin_popcount(byte);
            }
            total_bits += qvec.dimension;
        }
        stats.bit_balance = static_cast<float>(total_ones) / total_bits;
    }
    
    // Compute compression ratio
    std::size_t original_bytes = n * impl_->dimension * sizeof(float);
    stats.memory_bytes = total_memory;
    stats.compression_ratio = static_cast<float>(original_bytes) / total_memory;
    
    return stats;
}

auto RaBitQuantizer::is_trained() const noexcept -> bool {
    return impl_->trained;
}

auto RaBitQuantizer::dimension() const noexcept -> std::size_t {
    return static_cast<std::size_t>(impl_->dimension);
}

auto RaBitQuantizer::bit_width() const noexcept -> QuantizationBits {
    return impl_->params.bits;
}

// Additional methods not in header but useful internally


// Stub implementations for remaining methods
auto RaBitQuantizer::asymmetric_distance(std::span<const float> query,
                                        const QuantizedVector& qvec) const
    -> float {
    // For now, reconstruct and compute L2 distance
    auto reconstructed = reconstruct(qvec);
    if (!reconstructed) {
        return std::numeric_limits<float>::infinity();
    }
    
    float distance = 0.0f;
    for (std::size_t i = 0; i < query.size() && i < reconstructed.value().size(); ++i) {
        float diff = query[i] - reconstructed.value()[i];
        distance += diff * diff;
    }
    return std::sqrt(distance);
}

auto RaBitQuantizer::search_batch(std::span<const float> query,
                                 const std::vector<QuantizedVector>& qvecs,
                                 std::size_t k) const
    -> std::vector<std::pair<std::size_t, float>> {
    std::vector<std::pair<std::size_t, float>> results;
    results.reserve(qvecs.size());
    
    for (std::size_t i = 0; i < qvecs.size(); ++i) {
        float dist = asymmetric_distance(query, qvecs[i]);
        results.emplace_back(i, dist);
    }
    
    // Partial sort to get top-k
    if (k < results.size()) {
        std::partial_sort(results.begin(), results.begin() + k, results.end(),
                         [](const auto& a, const auto& b) { return a.second < b.second; });
        results.resize(k);
    } else {
        std::sort(results.begin(), results.end(),
                 [](const auto& a, const auto& b) { return a.second < b.second; });
    }
    
    return results;
}

auto RaBitQuantizer::save(const std::string& path) const
    -> std::expected<void, core::error> {
    // TODO: Implement serialization
    return {};
}

auto RaBitQuantizer::load(const std::string& path)
    -> std::expected<RaBitQuantizer, core::error> {
    // TODO: Implement deserialization
    RaBitQuantizer quantizer;
    // Load data from file and populate quantizer.impl_
    return quantizer;
}

} // namespace vesper::index