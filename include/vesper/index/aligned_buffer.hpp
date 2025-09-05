#pragma once

/** \file aligned_buffer.hpp
 *  \brief Cache-aligned memory buffer for SIMD-friendly data structures.
 *
 * Provides contiguous, cache-aligned memory for vector data to maximize
 * SIMD performance and minimize cache misses.
 *
 * Key features:
 * - 64-byte alignment for cache lines
 * - Contiguous memory layout for centroids
 * - Zero-copy views and slices
 * - NUMA-aware allocation support
 */

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vesper/span_polyfill.hpp>
#include <vector>
#include <new>
#include <utility>
#include <limits>

namespace vesper::index {

/** \brief Aligned memory allocator for SIMD operations. */
template<typename T, std::size_t Alignment = 64>
class AlignedAllocator {
public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using const_pointer = const T*;
    
    template<typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };
    
    AlignedAllocator() noexcept = default;
    
    template<typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}
    
    [[nodiscard]] auto allocate(size_type n) -> T* {
        if (n > std::numeric_limits<size_type>::max() / sizeof(T)) {
            throw std::bad_array_new_length();
        }
        
        const size_type bytes = n * sizeof(T);
        void* ptr = std::aligned_alloc(Alignment, align_up(bytes, Alignment));
        
        if (!ptr) {
            throw std::bad_alloc();
        }
        
        return static_cast<T*>(ptr);
    }
    
    auto deallocate(T* ptr, size_type) noexcept -> void {
        std::free(ptr);
    }
    
    template<typename U>
    auto operator==(const AlignedAllocator<U, Alignment>&) const noexcept -> bool {
        return true;
    }
    
    template<typename U>
    auto operator!=(const AlignedAllocator<U, Alignment>&) const noexcept -> bool {
        return false;
    }
    
private:
    static constexpr auto align_up(size_type n, size_type alignment) noexcept -> size_type {
        return (n + alignment - 1) & ~(alignment - 1);
    }
};

/** \brief Contiguous aligned buffer for centroid storage.
 *
 * Stores multiple vectors in a single contiguous, cache-aligned buffer
 * for optimal memory access patterns during k-means and search.
 */
class AlignedCentroidBuffer {
public:
    /** \brief Construct buffer for k centroids of dimension dim. */
    AlignedCentroidBuffer(std::uint32_t k, std::size_t dim)
        : k_(k)
        , dim_(dim)
        , stride_(align_to_cache_line(dim))
        , data_(k_ * stride_) {
        std::fill(data_.begin(), data_.end(), 0.0f);
    }
    
    /** \brief Get pointer to centroid i. */
    [[nodiscard]] auto operator[](std::uint32_t i) noexcept -> float* {
        return data_.data() + i * stride_;
    }
    
    [[nodiscard]] auto operator[](std::uint32_t i) const noexcept -> const float* {
        return data_.data() + i * stride_;
    }
    
    /** \brief Get centroid as span. */
    [[nodiscard]] auto get_centroid(std::uint32_t i) noexcept -> std::span<float> {
        return std::span((*this)[i], dim_);
    }
    
    [[nodiscard]] auto get_centroid(std::uint32_t i) const noexcept -> std::span<const float> {
        return std::span((*this)[i], dim_);
    }
    
    /** \brief Copy centroid data from vector. */
    auto set_centroid(std::uint32_t i, const std::vector<float>& data) -> void {
        auto dest = get_centroid(i);
        std::copy(data.begin(), data.end(), dest.begin());
    }
    
    /** \brief Copy centroid data from span. */
    auto set_centroid(std::uint32_t i, std::span<const float> data) -> void {
        auto dest = get_centroid(i);
        std::copy(data.begin(), data.end(), dest.begin());
    }
    
    /** \brief Get all centroids as vector of vectors (for compatibility). */
    [[nodiscard]] auto to_vectors() const -> std::vector<std::vector<float>> {
        std::vector<std::vector<float>> result(k_);
        for (std::uint32_t i = 0; i < k_; ++i) {
            const auto centroid = get_centroid(i);
            result[i].assign(centroid.begin(), centroid.end());
        }
        return result;
    }
    
    /** \brief Load centroids from vector of vectors. */
    auto from_vectors(const std::vector<std::vector<float>>& centroids) -> void {
        for (std::uint32_t i = 0; i < std::min(k_, static_cast<std::uint32_t>(centroids.size())); ++i) {
            set_centroid(i, centroids[i]);
        }
    }
    
    /** \brief Clear all centroids to zero. */
    auto clear() noexcept -> void {
        std::fill(data_.begin(), data_.end(), 0.0f);
    }
    
    /** \brief Get number of centroids. */
    [[nodiscard]] auto size() const noexcept -> std::uint32_t { return k_; }
    
    /** \brief Get dimension of centroids. */
    [[nodiscard]] auto dimension() const noexcept -> std::size_t { return dim_; }
    
    /** \brief Get stride between centroids. */
    [[nodiscard]] auto stride() const noexcept -> std::size_t { return stride_; }
    
    /** \brief Get raw data pointer. */
    [[nodiscard]] auto data() noexcept -> float* { return data_.data(); }
    [[nodiscard]] auto data() const noexcept -> const float* { return data_.data(); }
    
    /** \brief Prefetch centroid for reading. */
    auto prefetch_read(std::uint32_t i) const noexcept -> void {
        const float* ptr = (*this)[i];
        __builtin_prefetch(ptr, 0, 3);
        __builtin_prefetch(ptr + 16, 0, 3);
    }
    
    /** \brief Prefetch centroid for writing. */
    auto prefetch_write(std::uint32_t i) noexcept -> void {
        float* ptr = (*this)[i];
        __builtin_prefetch(ptr, 1, 3);
        __builtin_prefetch(ptr + 16, 1, 3);
    }
    
private:
    static constexpr std::size_t CACHE_LINE = 64;
    
    static constexpr auto align_to_cache_line(std::size_t dim) noexcept -> std::size_t {
        constexpr std::size_t floats_per_line = CACHE_LINE / sizeof(float);
        return ((dim + floats_per_line - 1) / floats_per_line) * floats_per_line;
    }
    
    std::uint32_t k_;
    std::size_t dim_;
    std::size_t stride_;
    std::vector<float, AlignedAllocator<float, CACHE_LINE>> data_;
};

/** \brief Aligned distance matrix for inter-centroid distances.
 *
 * Stores symmetric distance matrix in cache-friendly layout.
 */
class AlignedDistanceMatrix {
public:
    AlignedDistanceMatrix(std::uint32_t k)
        : k_(k)
        , stride_(align_to_cache_line(k))
        , data_(k_ * stride_) {
        std::fill(data_.begin(), data_.end(), 0.0f);
    }
    
    /** \brief Get distance between centroids i and j. */
    [[nodiscard]] auto operator()(std::uint32_t i, std::uint32_t j) const noexcept -> float {
        return data_[i * stride_ + j];
    }
    
    /** \brief Set distance between centroids i and j. */
    auto set(std::uint32_t i, std::uint32_t j, float dist) noexcept -> void {
        data_[i * stride_ + j] = dist;
    }
    
    /** \brief Set symmetric distance. */
    auto set_symmetric(std::uint32_t i, std::uint32_t j, float dist) noexcept -> void {
        set(i, j, dist);
        set(j, i, dist);
    }
    
    /** \brief Get row i as span. */
    [[nodiscard]] auto row(std::uint32_t i) const noexcept -> std::span<const float> {
        return std::span(data_.data() + i * stride_, k_);
    }
    
    /** \brief Convert to vector of vectors (for compatibility). */
    [[nodiscard]] auto to_vectors() const -> std::vector<std::vector<float>> {
        std::vector<std::vector<float>> result(k_, std::vector<float>(k_));
        for (std::uint32_t i = 0; i < k_; ++i) {
            for (std::uint32_t j = 0; j < k_; ++j) {
                result[i][j] = (*this)(i, j);
            }
        }
        return result;
    }
    
    /** \brief Load from vector of vectors. */
    auto from_vectors(const std::vector<std::vector<float>>& matrix) -> void {
        for (std::uint32_t i = 0; i < std::min(k_, static_cast<std::uint32_t>(matrix.size())); ++i) {
            for (std::uint32_t j = 0; j < std::min(k_, static_cast<std::uint32_t>(matrix[i].size())); ++j) {
                set(i, j, matrix[i][j]);
            }
        }
    }
    
    [[nodiscard]] auto size() const noexcept -> std::uint32_t { return k_; }
    
private:
    static constexpr std::size_t CACHE_LINE = 64;
    
    static constexpr auto align_to_cache_line(std::size_t n) noexcept -> std::size_t {
        constexpr std::size_t floats_per_line = CACHE_LINE / sizeof(float);
        return ((n + floats_per_line - 1) / floats_per_line) * floats_per_line;
    }
    
    std::uint32_t k_;
    std::size_t stride_;
    std::vector<float, AlignedAllocator<float, CACHE_LINE>> data_;
};

} // namespace vesper::index