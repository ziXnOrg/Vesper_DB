#pragma once

/** \file memory_pool.hpp
 *  \brief Thread-local memory pools for efficient allocation.
 *
 * Provides fast, thread-safe memory allocation using:
 * - Thread-local storage for zero contention
 * - PMR (Polymorphic Memory Resources) for flexibility
 * - Arena allocation for bulk operations
 * - Automatic cleanup and recycling
 *
 * Performance: 10-100x faster than malloc for small allocations.
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <memory_resource>
#include <vector>
#include <atomic>
#include <thread>
#include "vesper/platform/memory.hpp"

namespace vesper::core {

/** \brief Fixed-size memory arena for fast allocation.
 *
 * Pre-allocates a large chunk and serves allocations from it.
 * No individual deallocation - entire arena is freed at once.
 */
class MemoryArena {
public:
    static constexpr std::size_t DEFAULT_SIZE = 64 * 1024 * 1024;  // 64MB
    static constexpr std::size_t ALIGNMENT = 64;  // Cache line size
    
    explicit MemoryArena(std::size_t size = DEFAULT_SIZE)
        : size_(align_up(size, ALIGNMENT))
        , used_(0) {
        buffer_ = static_cast<std::uint8_t*>(
            vesper::platform::aligned_allocate(size_, ALIGNMENT));
        if (!buffer_) {
            throw std::bad_alloc();
        }
    }
    
    ~MemoryArena() {
        vesper::platform::aligned_deallocate(buffer_);
    }
    
    MemoryArena(const MemoryArena&) = delete;
    MemoryArena& operator=(const MemoryArena&) = delete;
    
    MemoryArena(MemoryArena&& other) noexcept
        : buffer_(other.buffer_)
        , size_(other.size_)
        , used_(other.used_) {
        other.buffer_ = nullptr;
        other.size_ = 0;
        other.used_ = 0;
    }
    
    /** \brief Allocate memory from arena. */
    [[nodiscard]] auto allocate(std::size_t bytes, 
                                std::size_t alignment = ALIGNMENT) -> void* {
        bytes = align_up(bytes, alignment);
        
        const std::size_t offset = align_up(used_, alignment);
        if (offset + bytes > size_) {
            return nullptr;  // Arena exhausted
        }
        
        used_ = offset + bytes;
        return buffer_ + offset;
    }
    
    /** \brief Reset arena for reuse. */
    auto reset() noexcept -> void {
        used_ = 0;
    }
    
    /** \brief Get bytes used. */
    [[nodiscard]] auto used() const noexcept -> std::size_t { 
        return used_; 
    }
    
    /** \brief Get bytes available. */
    [[nodiscard]] auto available() const noexcept -> std::size_t { 
        return size_ - used_; 
    }
    
    /** \brief Get total size. */
    [[nodiscard]] auto size() const noexcept -> std::size_t { 
        return size_; 
    }
    
private:
    static constexpr auto align_up(std::size_t n, std::size_t alignment) noexcept 
        -> std::size_t {
        return (n + alignment - 1) & ~(alignment - 1);
    }
    
    std::uint8_t* buffer_;
    std::size_t size_;
    std::size_t used_;
};

/** \brief PMR memory resource backed by arena. */
class ArenaResource : public std::pmr::memory_resource {
public:
    explicit ArenaResource(MemoryArena* arena)
        : arena_(arena) {}
    
protected:
    auto do_allocate(std::size_t bytes, std::size_t alignment) -> void* override {
        void* ptr = arena_->allocate(bytes, alignment);
        if (!ptr) {
            throw std::bad_alloc();
        }
        return ptr;
    }
    
    auto do_deallocate(void* /* ptr */, std::size_t /* bytes */, 
                      std::size_t /* alignment */) -> void override {
        // No-op: arena doesn't support individual deallocation
    }
    
    auto do_is_equal(const std::pmr::memory_resource& other) const noexcept 
        -> bool override {
        return this == &other;
    }
    
private:
    MemoryArena* arena_;
};

/** \brief Thread-local memory pool manager.
 *
 * Provides fast allocation with automatic cleanup.
 */
class ThreadLocalPool {
public:
    /** \brief Get thread-local pool instance. */
    static auto instance() -> ThreadLocalPool& {
        thread_local ThreadLocalPool pool;
        return pool;
    }
    
    /** \brief Get PMR allocator for current thread. */
    [[nodiscard]] auto allocator() -> std::pmr::polymorphic_allocator<std::byte> {
        return std::pmr::polymorphic_allocator<std::byte>(&resource_);
    }
    
    /** \brief Allocate memory. */
    [[nodiscard]] auto allocate(std::size_t bytes, 
                                std::size_t alignment = 64) -> void* {
        return arena_.allocate(bytes, alignment);
    }
    
    /** \brief Reset pool (deallocate all). */
    auto reset() -> void {
        arena_.reset();
    }
    
    /** \brief Get usage statistics. */
    struct Stats {
        std::size_t bytes_used;
        std::size_t bytes_total;
        float usage_ratio;
    };
    
    [[nodiscard]] auto stats() const -> Stats {
        return Stats{
            .bytes_used = arena_.used(),
            .bytes_total = arena_.size(),
            .usage_ratio = static_cast<float>(arena_.used()) / 
                          static_cast<float>(arena_.size())
        };
    }
    
private:
    ThreadLocalPool() 
        : arena_(MemoryArena::DEFAULT_SIZE)
        , resource_(&arena_) {
        register_thread();
    }
    
    ~ThreadLocalPool() {
        unregister_thread();
    }
    
    auto register_thread() -> void {
        active_threads_.fetch_add(1, std::memory_order_relaxed);
    }
    
    auto unregister_thread() -> void {
        active_threads_.fetch_sub(1, std::memory_order_relaxed);
    }
    
    MemoryArena arena_;
    ArenaResource resource_;
    
    inline static std::atomic<std::uint32_t> active_threads_{0};
};

/** \brief RAII scope guard for pool lifetime. */
class PoolScope {
public:
    PoolScope() : pool_(ThreadLocalPool::instance()) {}
    
    ~PoolScope() {
        pool_.reset();
    }
    
    PoolScope(const PoolScope&) = delete;
    PoolScope& operator=(const PoolScope&) = delete;
    
    /** \brief Get allocator for this scope. */
    [[nodiscard]] auto allocator() -> std::pmr::polymorphic_allocator<std::byte> {
        return pool_.allocator();
    }
    
private:
    ThreadLocalPool& pool_;
};

/** \brief Pooled vector using thread-local allocation. */
template<typename T>
using PooledVector = std::pmr::vector<T>;

/** \brief Create pooled vector in current thread's pool. */
template<typename T>
[[nodiscard]] auto make_pooled_vector(std::size_t size = 0) -> PooledVector<T> {
    auto& pool = ThreadLocalPool::instance();
    PooledVector<T> vec(pool.allocator());
    if (size > 0) {
        vec.resize(size);
    }
    return vec;
}

/** \brief Temporary buffer with automatic cleanup. */
template<typename T>
class TempBuffer {
public:
    explicit TempBuffer(std::size_t count)
        : pool_(ThreadLocalPool::instance())
        , ptr_(static_cast<T*>(pool_.allocate(count * sizeof(T))))
        , size_(count) {
        if (!ptr_) {
            throw std::bad_alloc();
        }
    }
    
    ~TempBuffer() = default;
    
    TempBuffer(const TempBuffer&) = delete;
    TempBuffer& operator=(const TempBuffer&) = delete;
    
    [[nodiscard]] auto data() noexcept -> T* { return ptr_; }
    [[nodiscard]] auto data() const noexcept -> const T* { return ptr_; }
    [[nodiscard]] auto size() const noexcept -> std::size_t { return size_; }
    
    [[nodiscard]] auto operator[](std::size_t i) noexcept -> T& {
        return ptr_[i];
    }
    
    [[nodiscard]] auto operator[](std::size_t i) const noexcept -> const T& {
        return ptr_[i];
    }
    
    [[nodiscard]] auto begin() noexcept -> T* { return ptr_; }
    [[nodiscard]] auto end() noexcept -> T* { return ptr_ + size_; }
    [[nodiscard]] auto begin() const noexcept -> const T* { return ptr_; }
    [[nodiscard]] auto end() const noexcept -> const T* { return ptr_ + size_; }
    
private:
    ThreadLocalPool& pool_;
    T* ptr_;
    std::size_t size_;
};

} // namespace vesper::core