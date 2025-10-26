#pragma once

/** \file memory.hpp
 *  \brief Cross-platform memory management abstractions.
 *
 * Provides portable aligned memory allocation and deallocation functions
 * that work consistently across Windows, Linux, and macOS.
 *
 * Key features:
 * - Aligned memory allocation with proper error handling
 * - Safe deallocation that matches the allocation method
 * - Compile-time and runtime alignment validation
 * - Zero-overhead abstraction
 */

#include <cstddef>
#include <cstdlib>
#include <new>
#include <type_traits>

#ifdef _MSC_VER
#include <malloc.h>  // For _aligned_malloc/_aligned_free
#endif

namespace vesper::platform {

/** \brief Check if a value is a power of two.
 *
 * \param n Value to check
 * \return true if n is a power of two
 */
constexpr auto is_power_of_two(std::size_t n) noexcept -> bool {
    return n > 0 && (n & (n - 1)) == 0;
}

/** \brief Round up size to alignment boundary.
 *
 * \param size Size to align
 * \param alignment Alignment requirement (must be power of 2)
 * \return Size rounded up to alignment boundary
 */
constexpr auto align_up(std::size_t size, std::size_t alignment) noexcept -> std::size_t {
    return (size + alignment - 1) & ~(alignment - 1);
}

/** \brief Allocate aligned memory.
 *
 * Allocates memory with the specified alignment requirement.
 * The alignment must be a power of two and at least sizeof(void*).
 *
 * \param size Number of bytes to allocate
 * \param alignment Alignment requirement in bytes
 * \return Pointer to allocated memory or nullptr on failure
 *
 * \note The returned pointer must be freed with aligned_deallocate()
 * \note On Windows, uses _aligned_malloc
 * \note On POSIX systems, uses std::aligned_alloc
 */
[[nodiscard]] inline auto aligned_allocate(std::size_t size, std::size_t alignment) noexcept -> void* {
    // Validate alignment
    if (!is_power_of_two(alignment) || alignment < sizeof(void*)) {
        return nullptr;
    }
    
    // Ensure size is a multiple of alignment for std::aligned_alloc
    const std::size_t aligned_size = align_up(size, alignment);
    
#ifdef _MSC_VER
    // Windows: _aligned_malloc doesn't require size to be multiple of alignment
    return _aligned_malloc(aligned_size, alignment);
#else
    // POSIX: std::aligned_alloc requires size to be multiple of alignment
    return std::aligned_alloc(alignment, aligned_size);
#endif
}

/** \brief Allocate aligned memory with type safety.
 *
 * Template version that allocates memory for a specific type.
 *
 * \tparam T Type to allocate
 * \tparam Alignment Alignment requirement (default: alignof(T))
 * \param count Number of elements to allocate
 * \return Pointer to allocated memory or nullptr on failure
 */
template<typename T, std::size_t Alignment = alignof(T)>
[[nodiscard]] auto aligned_allocate_typed(std::size_t count) noexcept -> T* {
    static_assert(is_power_of_two(Alignment), "Alignment must be power of two");
    static_assert(Alignment >= alignof(T), "Alignment must be at least alignof(T)");
    
    const std::size_t bytes = count * sizeof(T);
    return static_cast<T*>(aligned_allocate(bytes, Alignment));
}

/** \brief Deallocate aligned memory.
 *
 * Frees memory allocated with aligned_allocate().
 * Safe to call with nullptr.
 *
 * \param ptr Pointer to memory to free (may be nullptr)
 *
 * \note On Windows, uses _aligned_free
 * \note On POSIX systems, uses std::free
 */
inline auto aligned_deallocate(void* ptr) noexcept -> void {
    if (!ptr) return;
    
#ifdef _MSC_VER
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

/** \brief RAII wrapper for aligned memory.
 *
 * Automatically deallocates memory on destruction.
 * Non-copyable but movable.
 */
template<typename T, std::size_t Alignment = alignof(T)>
class aligned_unique_ptr {
public:
    using pointer = T*;
    using element_type = T;
    
    aligned_unique_ptr() noexcept = default;
    
    explicit aligned_unique_ptr(std::size_t count) 
        : ptr_(aligned_allocate_typed<T, Alignment>(count))
        , count_(ptr_ ? count : 0) {
        if (!ptr_ && count > 0) {
            throw std::bad_alloc();
        }
    }
    
    ~aligned_unique_ptr() {
        reset();
    }
    
    // Move operations
    aligned_unique_ptr(aligned_unique_ptr&& other) noexcept
        : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }
    
    auto operator=(aligned_unique_ptr&& other) noexcept -> aligned_unique_ptr& {
        if (this != &other) {
            reset();
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }
    
    // Delete copy operations
    aligned_unique_ptr(const aligned_unique_ptr&) = delete;
    auto operator=(const aligned_unique_ptr&) -> aligned_unique_ptr& = delete;
    
    // Accessors
    [[nodiscard]] auto get() noexcept -> T* { return ptr_; }
    [[nodiscard]] auto get() const noexcept -> const T* { return ptr_; }
    [[nodiscard]] auto size() const noexcept -> std::size_t { return count_; }
    [[nodiscard]] explicit operator bool() const noexcept { return ptr_ != nullptr; }
    
    auto operator*() -> T& { return *ptr_; }
    auto operator*() const -> const T& { return *ptr_; }
    auto operator->() noexcept -> T* { return ptr_; }
    auto operator->() const noexcept -> const T* { return ptr_; }
    
    auto operator[](std::size_t idx) -> T& { return ptr_[idx]; }
    auto operator[](std::size_t idx) const -> const T& { return ptr_[idx]; }
    
    // Reset
    auto reset() noexcept -> void {
        if (ptr_) {
            // Call destructors for non-trivial types
            if constexpr (!std::is_trivially_destructible_v<T>) {
                for (std::size_t i = 0; i < count_; ++i) {
                    ptr_[i].~T();
                }
            }
            aligned_deallocate(ptr_);
            ptr_ = nullptr;
            count_ = 0;
        }
    }
    
    auto release() noexcept -> T* {
        T* tmp = ptr_;
        ptr_ = nullptr;
        count_ = 0;
        return tmp;
    }
    
private:
    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};

/** \brief Create aligned unique pointer with constructor arguments.
 *
 * \tparam T Type to create
 * \tparam Alignment Alignment requirement
 * \tparam Args Constructor argument types
 * \param args Constructor arguments
 * \return Aligned unique pointer to constructed object
 */
template<typename T, std::size_t Alignment = alignof(T), typename... Args>
[[nodiscard]] auto make_aligned_unique(Args&&... args) -> aligned_unique_ptr<T, Alignment> {
    aligned_unique_ptr<T, Alignment> ptr(1);
    if (ptr) {
        new (ptr.get()) T(std::forward<Args>(args)...);
    }
    return ptr;
}

} // namespace vesper::platform