#include "vesper/index/pq_fastscan.hpp"
#include "vesper/index/kmeans_elkan.hpp"
#include "vesper/kernels/dispatch.hpp"
#include "vesper/kernels/batch_distances.hpp"

#include <algorithm>
#include <numeric>
#include <cmath>

namespace vesper::index {

auto FastScanPq::train(const float* data, std::size_t n, std::size_t dim)
    -> std::expected<void, core::error> {
    using core::error;
    using core::error_code;

    if (dim % config_.m != 0) {
        return std::vesper_unexpected(error{
            error_code::precondition_failed,
            "Dimension must be divisible by number of subquantizers",
            "pq_fastscan"
        });
    }

    dsub_ = dim / config_.m;

    // Allocate codebooks
    codebooks_ = std::make_unique<AlignedCentroidBuffer>(
        config_.m * ksub_, dsub_);

    // Train each subquantizer independently
    #pragma omp parallel for schedule(dynamic)
    for (int sub = 0; sub < static_cast<int>(config_.m); ++sub) {
        train_subquantizer(data, n, sub);
    }

    trained_ = true;
    return {};
}

auto FastScanPq::train_subquantizer(const float* data, std::size_t n,
                                    std::uint32_t sub_idx) -> void {
    // Extract subvector data
    std::vector<float> sub_data(n * dsub_);

    for (std::size_t i = 0; i < n; ++i) {
        const float* src = data + i * config_.m * dsub_ + sub_idx * dsub_;
        float* dst = sub_data.data() + i * dsub_;
        std::copy(src, src + dsub_, dst);
    }

    // Train k-means on subvector
    KmeansElkan elkan;
    KmeansElkan::Config kmeans_config{
        .k = ksub_,
        .max_iter = 25,
        .epsilon = 1e-4f,
        .seed = 42 + sub_idx
    };

    auto result = elkan.cluster(sub_data.data(), n, dsub_, kmeans_config);

    if (result.has_value()) {
        // Copy centroids to codebook
        for (std::uint32_t code = 0; code < ksub_; ++code) {
            const auto& centroid = result->centroids[code];
            const std::uint32_t idx = sub_idx * ksub_ + code;
            codebooks_->set_centroid(idx, centroid);
        }
    }
}

auto FastScanPq::find_nearest_code(const float* vec, std::uint32_t sub_idx) const
    -> std::uint8_t {
    float min_dist = std::numeric_limits<float>::max();
    std::uint8_t best_code = 0;

    const auto& ops = kernels::select_backend_auto();

    for (std::uint32_t code = 0; code < ksub_; ++code) {
        const std::uint32_t idx = sub_idx * ksub_ + code;
        const float dist = ops.l2_sq(
            std::span(vec, dsub_),
            codebooks_->get_centroid(idx)
        );

        if (dist < min_dist) {
            min_dist = dist;
            best_code = static_cast<std::uint8_t>(code);
        }
    }

    return best_code;
}

auto FastScanPq::encode(const float* data, std::size_t n,
                       std::uint8_t* codes) const -> void {
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(n); ++i) {
        const float* vec = data + i * config_.m * dsub_;
        std::uint8_t* code = codes + i * config_.m;

        for (std::uint32_t sub = 0; sub < config_.m; ++sub) {
            const float* sub_vec = vec + sub * dsub_;
            code[sub] = find_nearest_code(sub_vec, sub);
        }
    }
}

auto FastScanPq::encode_blocks(const float* data, std::size_t n)
    -> std::vector<PqCodeBlock> {
    // First encode all vectors
    std::vector<std::uint8_t> all_codes(n * config_.m);
    encode(data, n, all_codes.data());

    // Organize into blocks
    const std::size_t n_blocks = (n + config_.block_size - 1) / config_.block_size;
    std::vector<PqCodeBlock> blocks;
    blocks.reserve(n_blocks);

    for (std::size_t i = 0; i < n; i += config_.block_size) {
        PqCodeBlock block(config_.m, config_.block_size);

        const std::size_t block_end = std::min(i + config_.block_size, n);
        for (std::size_t j = i; j < block_end; ++j) {
            block.add_code(all_codes.data() + j * config_.m);
        }

        blocks.push_back(std::move(block));
    }

    return blocks;
}

auto FastScanPq::decode(const std::uint8_t* codes, std::size_t n, float* data) const -> void {
    if (!trained_ || !codebooks_) {
        // Not trained, return zeros
        std::fill(data, data + n * config_.m * dsub_, 0.0f);
        return;
    }

    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(n); ++i) {
        const std::uint8_t* code = codes + i * config_.m;
        float* vec = data + i * config_.m * dsub_;

        // Reconstruct vector from codebook entries
        for (std::uint32_t sub = 0; sub < config_.m; ++sub) {
            const std::uint8_t code_idx = code[sub];
            const std::uint32_t codebook_idx = sub * ksub_ + code_idx;
            const float* centroid = (*codebooks_)[codebook_idx];

            // Copy centroid to output vector
            std::memcpy(vec + sub * dsub_, centroid, dsub_ * sizeof(float));
        }
    }
}

auto FastScanPq::compute_lookup_tables(const float* query) const
    -> AlignedCentroidBuffer {
    AlignedCentroidBuffer luts(config_.m, ksub_);

    const auto& ops = kernels::select_backend_auto();

    #pragma omp parallel for
    for (int sub = 0; sub < static_cast<int>(config_.m); ++sub) {
        const float* query_sub = query + sub * dsub_;
        float* lut = luts[sub];

        for (std::uint32_t code = 0; code < ksub_; ++code) {
            const std::uint32_t idx = sub * ksub_ + code;
            lut[code] = ops.l2_sq(
                std::span(query_sub, dsub_),
                codebooks_->get_centroid(idx)
            );
        }
    }

    return luts;
}

auto FastScanPq::compute_distances(const float* query,
                                  const std::vector<PqCodeBlock>& blocks,
                                  float* distances) const -> void {
    // Compute lookup tables
    auto luts = compute_lookup_tables(query);

    // Initialize distances to zero
    const std::size_t total_codes = blocks.size() * config_.block_size;
    std::fill(distances, distances + total_codes, 0.0f);

    // Process each subquantizer
    for (std::uint32_t sub = 0; sub < config_.m; ++sub) {
        const float* lut = luts[sub];
        std::size_t offset = 0;

        for (const auto& block : blocks) {
            const std::uint8_t* codes = block.get_subquantizer_codes(sub);
            const std::uint32_t n_codes = block.size();

            // Accumulate distances from lookup table
            for (std::uint32_t i = 0; i < n_codes; ++i) {
                distances[offset + i] += lut[codes[i]];
            }

            offset += config_.block_size;
        }
    }
}

#ifdef __AVX2__
auto FastScanPq::compute_distances_avx2(const float* query,
                                       const std::vector<PqCodeBlock>& blocks,
                                       float* distances) const -> void {
    // Compute lookup tables
    auto luts = compute_lookup_tables(query);

    // Initialize distances
    const std::size_t total_codes = blocks.size() * config_.block_size;
    std::fill(distances, distances + total_codes, 0.0f);

    // Ensure distances are aligned
    std::vector<float, AlignedAllocator<float, 32>> aligned_dists(total_codes);

    // Process each subquantizer with SIMD
    for (std::uint32_t sub = 0; sub < config_.m; ++sub) {
        const float* lut = luts[sub];
        std::size_t offset = 0;

        for (const auto& block : blocks) {
            const std::uint8_t* codes = block.get_subquantizer_codes(sub);
            const std::uint32_t n_codes = block.size();

            // Use SIMD accumulation
            accumulate_distances_avx2(codes, lut, n_codes,
                                     aligned_dists.data() + offset);

            offset += config_.block_size;
        }
    }

    // Copy back to output
    std::copy(aligned_dists.begin(), aligned_dists.end(), distances);
}
#endif

#ifdef __AVX512F__
auto FastScanPq::compute_distances_avx512(const float* query,
                                         const std::vector<PqCodeBlock>& blocks,
                                         float* distances) const -> void {
    // Compute lookup tables
    auto luts = compute_lookup_tables(query);

    const std::size_t total_codes = blocks.size() * config_.block_size;
    std::fill(distances, distances + total_codes, 0.0f);

    // Process with AVX-512 16-wide operations
    for (std::uint32_t sub = 0; sub < config_.m; ++sub) {
        const float* lut = luts[sub];
        std::size_t offset = 0;

        for (const auto& block : blocks) {
            const std::uint8_t* codes = block.get_subquantizer_codes(sub);
            const std::uint32_t n_codes = block.size();

            // Process 16 codes at a time with AVX-512
            std::uint32_t i = 0;
            for (; i + 16 <= n_codes; i += 16) {
                // Load 16 code bytes
                const __m128i code_bytes = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(codes + i));

                // Convert to 32-bit integers
                const __m512i indices = _mm512_cvtepu8_epi32(code_bytes);

                // Gather distances from lookup table
                const __m512 dists = _mm512_i32gather_ps(indices, lut, 4);

                // Load current accumulated distances
                __m512 acc = _mm512_loadu_ps(distances + offset + i);

                // Accumulate
                acc = _mm512_add_ps(acc, dists);

                // Store back
                _mm512_storeu_ps(distances + offset + i, acc);
            }

            // Handle remainder with scalar
            for (; i < n_codes; ++i) {
                distances[offset + i] += lut[codes[i]];
            }

            offset += config_.block_size;
        }
    }
}
#endif

} // namespace vesper::index

namespace vesper::index {

auto FastScanPq::export_codebooks(std::vector<float>& out) const -> void {
    out.clear();
    const std::size_t k = static_cast<std::size_t>(config_.m) * static_cast<std::size_t>(ksub_);
    out.resize(k * dsub_);
    if (!codebooks_) return;
    for (std::size_t i = 0; i < k; ++i) {
        const float* src = (*codebooks_)[static_cast<std::uint32_t>(i)];
        std::memcpy(out.data() + i * dsub_, src, dsub_ * sizeof(float));
    }
}

auto FastScanPq::import_pretrained(std::size_t dsub, std::span<const float> data) -> void {
    dsub_ = dsub;
    const std::size_t k = static_cast<std::size_t>(config_.m) * static_cast<std::size_t>(ksub_);
    codebooks_ = std::make_unique<AlignedCentroidBuffer>(static_cast<std::uint32_t>(k), dsub_);
    for (std::size_t i = 0; i < k; ++i) {
        (*codebooks_).set_centroid(static_cast<std::uint32_t>(i),
                                   std::span<const float>(data.data() + i * dsub_, dsub_));
    }
    trained_ = true;
}

} // namespace vesper::index
