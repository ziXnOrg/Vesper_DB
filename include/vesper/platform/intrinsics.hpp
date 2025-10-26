#pragma once

/** \file intrinsics.hpp
 *  \brief Cross-platform CPU intrinsics abstractions.
 *
 * Provides portable CPU intrinsics for performance optimizations
 * including prefetching, memory barriers, and SIMD operations.
 *
 * Key features:
 * - Cache prefetch hints with multiple levels
 * - Memory ordering and barrier operations
 * - CPU feature detection
 * - No-op fallbacks for unsupported platforms
 */

#include <cstddef>
#include <cstdint>

// Platform-specific headers
#ifdef _MSC_VER
#define NOMINMAX  // Prevent Windows.h from defining min/max macros
#include <intrin.h>
#include <windows.h>  // For MemoryBarrier
#pragma intrinsic(_mm_prefetch)
#pragma intrinsic(_ReadWriteBarrier)
#elif defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace vesper::platform {

/** \brief Prefetch hint levels.
 *
 * Maps to architecture-specific prefetch hints.
 */
enum class prefetch_hint {
    /// Prefetch to all cache levels (T0)
    all_levels = 0,
    /// Prefetch to L2 and L3 cache (T1)
    l2_l3 = 1,
    /// Prefetch to L3 cache only (T2)
    l3_only = 2,
    /// Non-temporal prefetch (NTA) - bypass cache
    non_temporal = 3
};

/** \brief Prefetch memory for reading.
 *
 * Hints to the CPU to prefetch a cache line for reading.
 * This is a performance hint and may be ignored by the CPU.
 *
 * \param addr Address to prefetch (does not need to be aligned)
 * \param hint Cache level hint
 *
 * \note No-op on platforms without prefetch support
 */
inline auto prefetch_read(const void* addr, prefetch_hint hint = prefetch_hint::all_levels) noexcept -> void {
    if (!addr) return;
    
#ifdef _MSC_VER
    // Windows/MSVC intrinsics - hint must be a constant
    switch (hint) {
        case prefetch_hint::all_levels: 
            _mm_prefetch(static_cast<const char*>(addr), _MM_HINT_T0);
            break;
        case prefetch_hint::l2_l3: 
            _mm_prefetch(static_cast<const char*>(addr), _MM_HINT_T1);
            break;
        case prefetch_hint::l3_only: 
            _mm_prefetch(static_cast<const char*>(addr), _MM_HINT_T2);
            break;
        case prefetch_hint::non_temporal: 
            _mm_prefetch(static_cast<const char*>(addr), _MM_HINT_NTA);
            break;
    }
    
#elif defined(__GNUC__) || defined(__clang__)
    // GCC/Clang builtin
    int locality = 3;  // 0=no locality, 3=high locality
    switch (hint) {
        case prefetch_hint::all_levels: locality = 3; break;
        case prefetch_hint::l2_l3: locality = 2; break;
        case prefetch_hint::l3_only: locality = 1; break;
        case prefetch_hint::non_temporal: locality = 0; break;
    }
    __builtin_prefetch(addr, 0, locality);  // 0=read
    
#else
    // No-op for unsupported platforms
    (void)addr;
    (void)hint;
#endif
}

/** \brief Prefetch memory for writing.
 *
 * Hints to the CPU to prefetch a cache line for writing.
 * This is a performance hint and may be ignored by the CPU.
 *
 * \param addr Address to prefetch (does not need to be aligned)
 * \param hint Cache level hint
 *
 * \note No-op on platforms without prefetch support
 */
inline auto prefetch_write(void* addr, prefetch_hint hint = prefetch_hint::all_levels) noexcept -> void {
    if (!addr) return;
    
#ifdef _MSC_VER
    // Windows/MSVC: same intrinsic but processor knows the intent
    switch (hint) {
        case prefetch_hint::all_levels: 
            _mm_prefetch(static_cast<const char*>(addr), _MM_HINT_T0);
            break;
        case prefetch_hint::l2_l3: 
            _mm_prefetch(static_cast<const char*>(addr), _MM_HINT_T1);
            break;
        case prefetch_hint::l3_only: 
            _mm_prefetch(static_cast<const char*>(addr), _MM_HINT_T2);
            break;
        case prefetch_hint::non_temporal: 
            _mm_prefetch(static_cast<const char*>(addr), _MM_HINT_NTA);
            break;
    }
    
#elif defined(__GNUC__) || defined(__clang__)
    // GCC/Clang builtin
    int locality = 3;
    switch (hint) {
        case prefetch_hint::all_levels: locality = 3; break;
        case prefetch_hint::l2_l3: locality = 2; break;
        case prefetch_hint::l3_only: locality = 1; break;
        case prefetch_hint::non_temporal: locality = 0; break;
    }
    __builtin_prefetch(addr, 1, locality);  // 1=write
    
#else
    // No-op for unsupported platforms
    (void)addr;
    (void)hint;
#endif
}

/** \brief Prefetch multiple cache lines.
 *
 * Prefetches multiple consecutive cache lines starting from addr.
 * Useful for prefetching arrays or large structures.
 *
 * \param addr Starting address
 * \param bytes Number of bytes to prefetch
 * \param hint Cache level hint
 * \param for_write true if prefetching for write, false for read
 */
inline auto prefetch_range(const void* addr, std::size_t bytes, 
                           prefetch_hint hint = prefetch_hint::all_levels,
                           bool for_write = false) noexcept -> void {
    if (!addr || bytes == 0) return;
    
    constexpr std::size_t CACHE_LINE_SIZE = 64;  // Common cache line size
    const auto* ptr = static_cast<const char*>(addr);
    const auto* end = ptr + bytes;
    
    while (ptr < end) {
        if (for_write) {
            prefetch_write(const_cast<char*>(ptr), hint);
        } else {
            prefetch_read(ptr, hint);
        }
        ptr += CACHE_LINE_SIZE;
    }
}

/** \brief Compiler memory barrier.
 *
 * Prevents the compiler from reordering memory operations
 * across this barrier. Does not emit CPU fence instructions.
 */
inline auto compiler_barrier() noexcept -> void {
#ifdef _MSC_VER
    _ReadWriteBarrier();
#elif defined(__GNUC__) || defined(__clang__)
    __asm__ __volatile__("" : : : "memory");
#else
    // Fallback: volatile operations can't be reordered
    std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

/** \brief CPU memory fence.
 *
 * Full memory barrier that prevents both compiler and CPU
 * from reordering memory operations across this point.
 */
inline auto memory_fence() noexcept -> void {
#ifdef _MSC_VER
    MemoryBarrier();
#elif defined(__GNUC__) || defined(__clang__)
    __sync_synchronize();
#else
    std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
}

/** \brief Pause instruction for spin-wait loops.
 *
 * Hints to the CPU that we're in a spin-wait loop.
 * Improves performance and reduces power consumption.
 */
inline auto cpu_pause() noexcept -> void {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    _mm_pause();
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    __asm__ __volatile__("yield");
#else
    // No-op for unsupported architectures
#endif
}

/** \brief Get CPU timestamp counter.
 *
 * Returns the current value of the processor's time-stamp counter.
 * Useful for fine-grained performance measurements.
 *
 * \return Current timestamp counter value
 *
 * \note Not available on all platforms
 * \note May not be synchronized across CPU cores
 */
[[nodiscard]] inline auto read_timestamp_counter() noexcept -> std::uint64_t {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    return __rdtsc();
#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
    std::uint32_t lo, hi;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return (static_cast<std::uint64_t>(hi) << 32) | lo;
#elif defined(__aarch64__)
    std::uint64_t val;
    __asm__ __volatile__("mrs %0, cntvct_el0" : "=r"(val));
    return val;
#else
    // Fallback to zero
    return 0;
#endif
}

/** \brief Count leading zeros.
 *
 * Returns the number of leading zero bits in the value.
 *
 * \param value Value to count leading zeros
 * \return Number of leading zero bits (0-31 for 32-bit)
 */
[[nodiscard]] inline auto count_leading_zeros(std::uint32_t value) noexcept -> int {
    if (value == 0) return 32;
    
#ifdef _MSC_VER
    unsigned long index;
    _BitScanReverse(&index, value);
    return 31 - static_cast<int>(index);
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_clz(value);
#else
    // Fallback implementation
    int count = 0;
    std::uint32_t mask = 0x80000000u;
    while ((value & mask) == 0 && mask != 0) {
        count++;
        mask >>= 1;
    }
    return count;
#endif
}

/** \brief Count trailing zeros.
 *
 * Returns the number of trailing zero bits in the value.
 *
 * \param value Value to count trailing zeros
 * \return Number of trailing zero bits (0-31 for 32-bit)
 */
[[nodiscard]] inline auto count_trailing_zeros(std::uint32_t value) noexcept -> int {
    if (value == 0) return 32;
    
#ifdef _MSC_VER
    unsigned long index;
    _BitScanForward(&index, value);
    return static_cast<int>(index);
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_ctz(value);
#else
    // Fallback implementation
    int count = 0;
    while ((value & 1) == 0) {
        count++;
        value >>= 1;
    }
    return count;
#endif
}

/** \brief Population count (number of set bits).
 *
 * Returns the number of bits set to 1 in the value.
 *
 * \param value Value to count bits
 * \return Number of set bits
 */
[[nodiscard]] inline auto popcount(std::uint32_t value) noexcept -> int {
#ifdef _MSC_VER
    return static_cast<int>(__popcnt(value));
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_popcount(value);
#else
    // Fallback implementation (Brian Kernighan's algorithm)
    int count = 0;
    while (value) {
        value &= value - 1;
        count++;
    }
    return count;
#endif
}

} // namespace vesper::platform