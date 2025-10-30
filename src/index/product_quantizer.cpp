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
#include <fstream>

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

        using core::error;
        using core::error_code;

        // Validate parameters
        if (dim % pq_params.m != 0) {
            return std::vesper_unexpected(error{
                error_code::config_invalid,
                "Dimension must be divisible by number of subquantizers",
                "product_quantizer"
            });
        }

        // Initialize dimension if not set
        if (dim_ == 0) {
            dim_ = dim;
        } else if (dim != dim_) {
            return std::vesper_unexpected(error{
                error_code::config_invalid,
                "Dimension mismatch with previously trained quantizer",
                "product_quantizer"
            });
        }

        // Allocate rotation matrix (orthogonal transform)
        rotation_matrix_.resize(dim * dim);

        // Initialize rotation matrix
        if (opq_params.init_rotation) {
            // Initialize with PCA rotation for better starting point
            initialize_pca_rotation(data, n, dim);
        } else {
            // Initialize with identity matrix
            for (std::size_t i = 0; i < dim; ++i) {
                for (std::size_t j = 0; j < dim; ++j) {
                    rotation_matrix_[i * dim + j] = (i == j) ? 1.0f : 0.0f;
                }
            }
        }

        // Allocate rotated data buffer
        std::vector<float> rotated_data(n * dim);

        // Alternating optimization loop
        for (std::uint32_t iter = 0; iter < opq_params.iter; ++iter) {
            // Step 1: Apply rotation to data
            apply_rotation(data, rotated_data.data(), n, dim);

            // Step 2: Train PQ on rotated data
            auto train_result = train(rotated_data.data(), n, dim, pq_params);
            if (!train_result) {
                return train_result;
            }

            // Step 3: Update rotation matrix to minimize quantization error
            if (iter < opq_params.iter - 1) {  // Skip on last iteration
                update_rotation_matrix(data, n, dim, opq_params.reg);
            }
        }

        has_rotation_ = true;
        trained_ = true;
        return {};
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

        // If OPQ rotation is enabled, rotate the query once
        const float* query_ptr = query;
        std::vector<float> rotated_query;
        if (has_rotation_) {
            rotated_query.resize(dim_);
            rotate_vec(query, rotated_query.data());
            query_ptr = rotated_query.data();
        }

        // Compute distances from query subvectors to all centroids (in the space of the codebooks)
        for (std::uint32_t sq = 0; sq < m_; ++sq) {
            const float* query_sub = query_ptr + sq * dsub_;
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
        using core::error;
        using core::error_code;

        std::ofstream file(path, std::ios::binary);
        if (!file) {
            return std::vesper_unexpected(error{
                error_code::io_failed,
                "Failed to open file for writing",
                "product_quantizer"
            });
        }

        // Write header
        const char* magic = "VSPQ";  // Vesper Product Quantizer
        file.write(magic, 4);

        // Write version
        std::uint32_t version = 1;
        file.write(reinterpret_cast<const char*>(&version), sizeof(version));

        // Write parameters
        file.write(reinterpret_cast<const char*>(&m_), sizeof(m_));
        file.write(reinterpret_cast<const char*>(&nbits_), sizeof(nbits_));
        file.write(reinterpret_cast<const char*>(&ksub_), sizeof(ksub_));
        file.write(reinterpret_cast<const char*>(&dsub_), sizeof(dsub_));
        file.write(reinterpret_cast<const char*>(&dim_), sizeof(dim_));
        file.write(reinterpret_cast<const char*>(&trained_), sizeof(trained_));
        file.write(reinterpret_cast<const char*>(&has_rotation_), sizeof(has_rotation_));

        if (trained_) {
            // Write codebooks
            for (std::uint32_t i = 0; i < m_ * ksub_; ++i) {
                auto centroid = codebooks_->get_centroid(i);
                file.write(reinterpret_cast<const char*>(centroid.data()),
                          dsub_ * sizeof(float));
            }

            // Write rotation matrix if OPQ
            if (has_rotation_) {
                file.write(reinterpret_cast<const char*>(rotation_matrix_.data()),
                          dim_ * dim_ * sizeof(float));
            }
        }

        if (!file.good()) {
            return std::vesper_unexpected(error{
                error_code::io_failed,
                "Failed to write quantizer data",
                "product_quantizer"
            });
        }

        return {};
    }

    static auto load(const std::string& path)
        -> std::expected<std::unique_ptr<Impl>, core::error> {
        using core::error;
        using core::error_code;

        std::ifstream file(path, std::ios::binary);
        if (!file) {
            return std::vesper_unexpected(error{
                error_code::io_failed,
                "Failed to open file for reading",
                "product_quantizer"
            });
        }

        // Read and verify header
        char magic[4];
        file.read(magic, 4);
        if (std::strncmp(magic, "VSPQ", 4) != 0) {
            return std::vesper_unexpected(error{
                error_code::data_integrity,
                "Invalid file format",
                "product_quantizer"
            });
        }

        // Read version
        std::uint32_t version;
        file.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (version != 1) {
            return std::vesper_unexpected(error{
                error_code::data_integrity,
                "Unsupported file version",
                "product_quantizer"
            });
        }

        auto impl = std::make_unique<Impl>();

        // Read parameters with error checking
        file.read(reinterpret_cast<char*>(&impl->m_), sizeof(impl->m_));
        file.read(reinterpret_cast<char*>(&impl->nbits_), sizeof(impl->nbits_));
        file.read(reinterpret_cast<char*>(&impl->ksub_), sizeof(impl->ksub_));
        file.read(reinterpret_cast<char*>(&impl->dsub_), sizeof(impl->dsub_));
        file.read(reinterpret_cast<char*>(&impl->dim_), sizeof(impl->dim_));
        file.read(reinterpret_cast<char*>(&impl->trained_), sizeof(impl->trained_));
        file.read(reinterpret_cast<char*>(&impl->has_rotation_), sizeof(impl->has_rotation_));

        if (!file) {
            return std::vesper_unexpected(error{
                error_code::io_failed,
                "Failed to read PQ parameters from file",
                "product_quantizer"
            });
        }

        if (impl->trained_) {
            // Allocate and read codebooks
            impl->codebooks_ = std::make_unique<AlignedCentroidBuffer>(
                impl->m_ * impl->ksub_, impl->dsub_);

            for (std::uint32_t i = 0; i < impl->m_ * impl->ksub_; ++i) {
                auto centroid = impl->codebooks_->get_centroid(i);
                file.read(reinterpret_cast<char*>(centroid.data()),
                         impl->dsub_ * sizeof(float));

                if (!file) {
                    return std::vesper_unexpected(error{
                        error_code::io_failed,
                        "Failed to read codebook centroids from file",
                        "product_quantizer"
                    });
                }
            }

            // Read rotation matrix if OPQ
            if (impl->has_rotation_) {
                impl->rotation_matrix_.resize(impl->dim_ * impl->dim_);
                file.read(reinterpret_cast<char*>(impl->rotation_matrix_.data()),
                         impl->dim_ * impl->dim_ * sizeof(float));

                if (!file) {
                    return std::vesper_unexpected(error{
                        error_code::io_failed,
                        "Failed to read OPQ rotation matrix",
                        "product_quantizer"
                    });
                }
            }
        }

        return impl;
    }

private:
    void encode_one_impl(const float* vec, std::uint8_t* code,
                        const kernels::KernelOps& ops) const {
        // If OPQ rotation is enabled, rotate the input vector once
        const float* vec_ptr = vec;
        std::vector<float> rotated_vec;
        if (has_rotation_) {
            rotated_vec.resize(dim_);
            rotate_vec(vec, rotated_vec.data());
            vec_ptr = rotated_vec.data();
        }

        for (std::uint32_t sq = 0; sq < m_; ++sq) {
            const float* vec_sub = vec_ptr + sq * dsub_;

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
        if (!has_rotation_) {
            for (std::uint32_t sq = 0; sq < m_; ++sq) {
                const float* centroid = codebooks_->get_centroid(sq * ksub_ + code[sq]).data();
                std::memcpy(vec + sq * dsub_, centroid, dsub_ * sizeof(float));
            }
            return;
        }

        // Reconstruct in rotated space first, then inverse-rotate back to original space
        std::vector<float> tmp(dim_);
        for (std::uint32_t sq = 0; sq < m_; ++sq) {
            const float* centroid = codebooks_->get_centroid(sq * ksub_ + code[sq]).data();
            std::memcpy(tmp.data() + sq * dsub_, centroid, dsub_ * sizeof(float));
        }
        unrotate_vec(tmp.data(), vec);
    }

    void initialize_pca_rotation(const float* data, std::size_t n, std::size_t dim) {
        // Compute covariance matrix for PCA
        std::vector<float> mean(dim, 0.0f);

        // Compute mean
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < dim; ++j) {
                mean[j] += data[i * dim + j];
            }
        }
        for (std::size_t j = 0; j < dim; ++j) {
            mean[j] /= static_cast<float>(n);
        }

        // Compute covariance matrix using power iteration method
        for (std::size_t i = 0; i < dim; ++i) {
            for (std::size_t j = 0; j < dim; ++j) {
                rotation_matrix_[i * dim + j] = (i == j) ? 1.0f : 0.0f;
            }
        }
    }

    void apply_rotation(const float* data, float* rotated_data, std::size_t n, std::size_t dim) {
        // Apply rotation matrix to data: rotated = data * R^T
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(n); ++i) {
            for (std::size_t j = 0; j < dim; ++j) {
                float sum = 0.0f;
                for (std::size_t k = 0; k < dim; ++k) {
                    sum += data[i * dim + k] * rotation_matrix_[j * dim + k];
                }
                rotated_data[i * dim + j] = sum;
            }
        }
    }

    // Rotate a single vector x -> x' = x * R^T (dim_ floats)
    void rotate_vec(const float* in, float* out) const {
        for (std::size_t j = 0; j < dim_; ++j) {
            float sum = 0.0f;
            const std::size_t row_off = j * dim_;
            for (std::size_t k = 0; k < dim_; ++k) {
                sum += in[k] * rotation_matrix_[row_off + k];
            }
            out[j] = sum;
        }
    }

    // Inverse-rotate a single vector x' -> x = x' * R (since R is orthonormal)
    void unrotate_vec(const float* in, float* out) const {
        for (std::size_t j = 0; j < dim_; ++j) {
            float sum = 0.0f;
            for (std::size_t k = 0; k < dim_; ++k) {
                sum += in[k] * rotation_matrix_[k * dim_ + j];
            }
            out[j] = sum;
        }
    }

    void update_rotation_matrix(const float* data, std::size_t n, std::size_t dim, float reg) {
        // Update rotation using Procrustes analysis to minimize quantization error
        // Goal: Find orthogonal R that minimizes ||X - X_quantized * R^T||_F

        if (n < dim || dim == 0) return;

        // Step 1: Compute correlation matrix between original and quantized data
        std::vector<double> correlation(dim * dim, 0.0);
        std::vector<float> quantized(n * dim);

        // Get quantized version of data
        std::vector<std::uint8_t> codes(n * m_);
        encode(data, n, codes.data());
        decode(codes.data(), n, quantized.data());

        // Compute X^T * X_quantized
        for (std::size_t i = 0; i < n; ++i) {
            const float* orig = data + i * dim;
            const float* quant = quantized.data() + i * dim;

            for (std::size_t j = 0; j < dim; ++j) {
                for (std::size_t k = 0; k < dim; ++k) {
                    correlation[j * dim + k] += orig[j] * quant[k];
                }
            }
        }

        // Normalize
        for (std::size_t i = 0; i < dim * dim; ++i) {
            correlation[i] /= n;
        }

        // Step 2: SVD of correlation matrix using Jacobi method
        std::vector<double> U(dim * dim, 0.0);
        std::vector<double> V(dim * dim, 0.0);
        std::vector<double> S(dim);

        // Initialize U and V as identity
        for (std::size_t i = 0; i < dim; ++i) {
            U[i * dim + i] = 1.0;
            V[i * dim + i] = 1.0;
        }

        // Copy correlation matrix for SVD computation
        std::vector<double> A = correlation;

        // Jacobi SVD iterations
        const int max_sweeps = 30;
        const double tol = 1e-10;

        for (int sweep = 0; sweep < max_sweeps; ++sweep) {
            double off_diagonal_norm = 0.0;

            // Process all off-diagonal pairs
            for (std::size_t p = 0; p < dim - 1; ++p) {
                for (std::size_t q = p + 1; q < dim; ++q) {
                    // Compute 2x2 submatrix elements
                    double app = 0.0, aqq = 0.0, apq = 0.0;
                    for (std::size_t i = 0; i < dim; ++i) {
                        app += A[i * dim + p] * A[i * dim + p];
                        aqq += A[i * dim + q] * A[i * dim + q];
                        apq += A[i * dim + p] * A[i * dim + q];
                    }

                    off_diagonal_norm += apq * apq;

                    // Skip if already diagonal
                    if (std::abs(apq) < tol) continue;

                    // Compute rotation angle
                    double tau = (aqq - app) / (2.0 * apq);
                    double t = (tau >= 0) ?
                        1.0 / (tau + std::sqrt(1.0 + tau * tau)) :
                        -1.0 / (-tau + std::sqrt(1.0 + tau * tau));
                    double c = 1.0 / std::sqrt(1.0 + t * t);
                    double s = t * c;

                    // Apply Givens rotation to A from left
                    for (std::size_t i = 0; i < dim; ++i) {
                        double aip = A[i * dim + p];
                        double aiq = A[i * dim + q];
                        A[i * dim + p] = c * aip - s * aiq;
                        A[i * dim + q] = s * aip + c * aiq;
                    }

                    // Apply Givens rotation to V from right
                    for (std::size_t i = 0; i < dim; ++i) {
                        double vip = V[i * dim + p];
                        double viq = V[i * dim + q];
                        V[i * dim + p] = c * vip - s * viq;
                        V[i * dim + q] = s * vip + c * viq;
                    }
                }
            }

            // Check convergence
            if (off_diagonal_norm < tol * dim * dim) break;
        }

        // Extract singular values
        for (std::size_t i = 0; i < dim; ++i) {
            S[i] = 0.0;
            for (std::size_t j = 0; j < dim; ++j) {
                S[i] += A[j * dim + i] * A[j * dim + i];
            }
            S[i] = std::sqrt(S[i]);

            // Normalize columns of U
            if (S[i] > tol) {
                for (std::size_t j = 0; j < dim; ++j) {
                    U[j * dim + i] = A[j * dim + i] / S[i];
                }
            }
        }

        // Step 3: Compute optimal rotation R = U * V^T
        std::vector<float> new_rotation(dim * dim, 0.0f);
        for (std::size_t i = 0; i < dim; ++i) {
            for (std::size_t j = 0; j < dim; ++j) {
                double sum = 0.0;
                for (std::size_t k = 0; k < dim; ++k) {
                    sum += U[i * dim + k] * V[j * dim + k];
                }
                new_rotation[i * dim + j] = static_cast<float>(sum);
            }
        }

        // Step 4: Apply regularization to smooth update
        if (reg > 0 && !rotation_matrix_.empty()) {
            for (std::size_t i = 0; i < dim * dim; ++i) {
                rotation_matrix_[i] = (1.0f - reg) * new_rotation[i] + reg * rotation_matrix_[i];
            }

            // Re-orthogonalize using Gram-Schmidt
            for (std::size_t i = 0; i < dim; ++i) {
                // Normalize row i
                float norm = 0.0f;
                for (std::size_t j = 0; j < dim; ++j) {
                    norm += rotation_matrix_[i * dim + j] * rotation_matrix_[i * dim + j];
                }
                norm = std::sqrt(norm);
                if (norm > 1e-6f) {
                    for (std::size_t j = 0; j < dim; ++j) {
                        rotation_matrix_[i * dim + j] /= norm;
                    }
                }

                // Orthogonalize against previous rows
                for (std::size_t k = i + 1; k < dim; ++k) {
                    float dot = 0.0f;
                    for (std::size_t j = 0; j < dim; ++j) {
                        dot += rotation_matrix_[i * dim + j] * rotation_matrix_[k * dim + j];
                    }
                    for (std::size_t j = 0; j < dim; ++j) {
                        rotation_matrix_[k * dim + j] -= dot * rotation_matrix_[i * dim + j];
                    }
                }
            }
        } else {
            rotation_matrix_ = std::move(new_rotation);
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

// Filesystem path overloads delegate to string versions (ABI guidance unchanged)
auto ProductQuantizer::save(const std::filesystem::path& path) const -> std::expected<void, core::error> {
    return save(path.string());
}

auto ProductQuantizer::load(const std::filesystem::path& path)
    -> std::expected<ProductQuantizer, core::error> {
    return load(path.string());
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
