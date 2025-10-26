#pragma once

/** \file compiler.hpp
 *  \brief Cross-platform compiler attributes and hints.
 *
 * Provides portable macros for compiler-specific attributes,
 * optimizations, and diagnostics.
 *
 * Key features:
 * - Function attributes (inline, noinline, hot, cold)
 * - Branch prediction hints
 * - Alignment attributes
 * - Warning suppression
 */

// Compiler detection
#if defined(_MSC_VER)
    #define VESPER_COMPILER_MSVC 1
    #define VESPER_COMPILER_VERSION _MSC_VER
#elif defined(__clang__)
    #define VESPER_COMPILER_CLANG 1
    #define VESPER_COMPILER_VERSION (__clang_major__ * 100 + __clang_minor__)
#elif defined(__GNUC__)
    #define VESPER_COMPILER_GCC 1
    #define VESPER_COMPILER_VERSION (__GNUC__ * 100 + __GNUC_MINOR__)
#else
    #define VESPER_COMPILER_UNKNOWN 1
#endif

// Function inlining hints
#ifdef VESPER_COMPILER_MSVC
    #define VESPER_ALWAYS_INLINE __forceinline
    #define VESPER_NEVER_INLINE __declspec(noinline)
#elif defined(VESPER_COMPILER_GCC) || defined(VESPER_COMPILER_CLANG)
    #define VESPER_ALWAYS_INLINE __attribute__((always_inline)) inline
    #define VESPER_NEVER_INLINE __attribute__((noinline))
#else
    #define VESPER_ALWAYS_INLINE inline
    #define VESPER_NEVER_INLINE
#endif

// Hot/cold path hints
#ifdef VESPER_COMPILER_MSVC
    #define VESPER_HOT
    #define VESPER_COLD
#elif defined(VESPER_COMPILER_GCC) || defined(VESPER_COMPILER_CLANG)
    #define VESPER_HOT __attribute__((hot))
    #define VESPER_COLD __attribute__((cold))
#else
    #define VESPER_HOT
    #define VESPER_COLD
#endif

// Branch prediction hints
#ifdef VESPER_COMPILER_GCC
    #define VESPER_LIKELY(x) __builtin_expect(!!(x), 1)
    #define VESPER_UNLIKELY(x) __builtin_expect(!!(x), 0)
#elif defined(VESPER_COMPILER_CLANG)
    #define VESPER_LIKELY(x) __builtin_expect(!!(x), 1)
    #define VESPER_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
    #define VESPER_LIKELY(x) (x)
    #define VESPER_UNLIKELY(x) (x)
#endif

// C++20 [[likely]]/[[unlikely]] fallback
#if __cplusplus >= 202002L
    #define VESPER_LIKELY_ATTR [[likely]]
    #define VESPER_UNLIKELY_ATTR [[unlikely]]
#else
    #define VESPER_LIKELY_ATTR
    #define VESPER_UNLIKELY_ATTR
#endif

// Alignment attributes
#ifdef VESPER_COMPILER_MSVC
    #define VESPER_ALIGN(n) __declspec(align(n))
    #define VESPER_ASSUME_ALIGNED(ptr, n) ptr
#elif defined(VESPER_COMPILER_GCC) || defined(VESPER_COMPILER_CLANG)
    #define VESPER_ALIGN(n) __attribute__((aligned(n)))
    #define VESPER_ASSUME_ALIGNED(ptr, n) __builtin_assume_aligned(ptr, n)
#else
    #define VESPER_ALIGN(n) alignas(n)
    #define VESPER_ASSUME_ALIGNED(ptr, n) ptr
#endif

// Cache line size (typical)
#ifndef VESPER_CACHE_LINE_SIZE
    #define VESPER_CACHE_LINE_SIZE 64
#endif

#define VESPER_CACHE_ALIGNED VESPER_ALIGN(VESPER_CACHE_LINE_SIZE)

// Restrict pointer aliasing
#ifdef VESPER_COMPILER_MSVC
    #define VESPER_RESTRICT __restrict
#elif defined(VESPER_COMPILER_GCC) || defined(VESPER_COMPILER_CLANG)
    #define VESPER_RESTRICT __restrict__
#else
    #define VESPER_RESTRICT
#endif

// Pure/const function attributes
#ifdef VESPER_COMPILER_MSVC
    #define VESPER_PURE
    #define VESPER_CONST
#elif defined(VESPER_COMPILER_GCC) || defined(VESPER_COMPILER_CLANG)
    #define VESPER_PURE __attribute__((pure))
    #define VESPER_CONST __attribute__((const))
#else
    #define VESPER_PURE
    #define VESPER_CONST
#endif

// Unused parameter/variable
#define VESPER_UNUSED(x) ((void)(x))

// Warning suppression
#ifdef VESPER_COMPILER_MSVC
    #define VESPER_PUSH_WARNING __pragma(warning(push))
    #define VESPER_POP_WARNING __pragma(warning(pop))
    #define VESPER_DISABLE_WARNING_MSVC(num) __pragma(warning(disable: num))
    #define VESPER_DISABLE_WARNING_GCC(name)
    #define VESPER_DISABLE_WARNING_CLANG(name)
#elif defined(VESPER_COMPILER_CLANG)
    #define VESPER_PUSH_WARNING _Pragma("clang diagnostic push")
    #define VESPER_POP_WARNING _Pragma("clang diagnostic pop")
    #define VESPER_DISABLE_WARNING_MSVC(num)
    #define VESPER_DISABLE_WARNING_GCC(name)
    #define VESPER_DISABLE_WARNING_CLANG(name) \
        _Pragma("clang diagnostic ignored \"-W" #name "\"")
#elif defined(VESPER_COMPILER_GCC)
    #define VESPER_PUSH_WARNING _Pragma("GCC diagnostic push")
    #define VESPER_POP_WARNING _Pragma("GCC diagnostic pop")
    #define VESPER_DISABLE_WARNING_MSVC(num)
    #define VESPER_DISABLE_WARNING_GCC(name) \
        _Pragma("GCC diagnostic ignored \"-W" #name "\"")
    #define VESPER_DISABLE_WARNING_CLANG(name)
#else
    #define VESPER_PUSH_WARNING
    #define VESPER_POP_WARNING
    #define VESPER_DISABLE_WARNING_MSVC(num)
    #define VESPER_DISABLE_WARNING_GCC(name)
    #define VESPER_DISABLE_WARNING_CLANG(name)
#endif

// Common warning suppressions
#define VESPER_DISABLE_PADDING_WARNING \
    VESPER_DISABLE_WARNING_MSVC(4820) \
    VESPER_DISABLE_WARNING_CLANG(padded) \
    VESPER_DISABLE_WARNING_GCC(padded)

// Deprecated attribute
#ifdef VESPER_COMPILER_MSVC
    #define VESPER_DEPRECATED(msg) __declspec(deprecated(msg))
#elif defined(VESPER_COMPILER_GCC) || defined(VESPER_COMPILER_CLANG)
    #define VESPER_DEPRECATED(msg) __attribute__((deprecated(msg)))
#else
    #define VESPER_DEPRECATED(msg) [[deprecated(msg)]]
#endif

// Export/import for shared libraries (future use)
#ifdef VESPER_COMPILER_MSVC
    #define VESPER_EXPORT __declspec(dllexport)
    #define VESPER_IMPORT __declspec(dllimport)
#elif defined(VESPER_COMPILER_GCC) || defined(VESPER_COMPILER_CLANG)
    #define VESPER_EXPORT __attribute__((visibility("default")))
    #define VESPER_IMPORT __attribute__((visibility("default")))
#else
    #define VESPER_EXPORT
    #define VESPER_IMPORT
#endif

// API macro (for future shared library support)
#ifdef VESPER_BUILDING_SHARED
    #define VESPER_API VESPER_EXPORT
#elif defined(VESPER_USING_SHARED)
    #define VESPER_API VESPER_IMPORT
#else
    #define VESPER_API
#endif

// Prefetch hints (moved from intrinsics for convenience)
#ifdef VESPER_COMPILER_MSVC
    #include <intrin.h>
    #define VESPER_PREFETCH(addr, rw, locality) \
        _mm_prefetch(static_cast<const char*>(addr), _MM_HINT_T0)
#elif defined(VESPER_COMPILER_GCC) || defined(VESPER_COMPILER_CLANG)
    #define VESPER_PREFETCH(addr, rw, locality) \
        __builtin_prefetch(addr, rw, locality)
#else
    #define VESPER_PREFETCH(addr, rw, locality) ((void)0)
#endif

// Debug assertions
#ifdef NDEBUG
    #define VESPER_ASSERT(cond) ((void)0)
    #define VESPER_ASSERT_MSG(cond, msg) ((void)0)
#else
    #include <cassert>
    #define VESPER_ASSERT(cond) assert(cond)
    #define VESPER_ASSERT_MSG(cond, msg) assert((cond) && (msg))
#endif

// Unreachable code hint
#ifdef VESPER_COMPILER_MSVC
    #define VESPER_UNREACHABLE() __assume(0)
#elif defined(VESPER_COMPILER_GCC) || defined(VESPER_COMPILER_CLANG)
    #define VESPER_UNREACHABLE() __builtin_unreachable()
#else
    #define VESPER_UNREACHABLE() ((void)0)
#endif

// Thread-local storage
#ifdef VESPER_COMPILER_MSVC
    #define VESPER_THREAD_LOCAL __declspec(thread)
#else
    #define VESPER_THREAD_LOCAL thread_local
#endif

// Packed structures
#ifdef VESPER_COMPILER_MSVC
    #define VESPER_PACKED_BEGIN __pragma(pack(push, 1))
    #define VESPER_PACKED_END __pragma(pack(pop))
    #define VESPER_PACKED_STRUCT(name) VESPER_PACKED_BEGIN struct name
#elif defined(VESPER_COMPILER_GCC) || defined(VESPER_COMPILER_CLANG)
    #define VESPER_PACKED_BEGIN
    #define VESPER_PACKED_END
    #define VESPER_PACKED_STRUCT(name) struct __attribute__((packed)) name
#else
    #define VESPER_PACKED_BEGIN
    #define VESPER_PACKED_END
    #define VESPER_PACKED_STRUCT(name) struct name
#endif

// Vectorization hints
#ifdef VESPER_COMPILER_MSVC
    #define VESPER_VECTORIZE __pragma(loop(ivdep))
#elif defined(VESPER_COMPILER_CLANG)
    #define VESPER_VECTORIZE _Pragma("clang loop vectorize(enable)")
#elif defined(VESPER_COMPILER_GCC)
    #define VESPER_VECTORIZE _Pragma("GCC ivdep")
#else
    #define VESPER_VECTORIZE
#endif

// Platform detection helpers
#ifdef _WIN32
    #define VESPER_PLATFORM_WINDOWS 1
#elif defined(__linux__)
    #define VESPER_PLATFORM_LINUX 1
#elif defined(__APPLE__)
    #define VESPER_PLATFORM_MACOS 1
#else
    #define VESPER_PLATFORM_UNKNOWN 1
#endif

// Architecture detection
#if defined(_M_X64) || defined(__x86_64__)
    #define VESPER_ARCH_X64 1
#elif defined(_M_IX86) || defined(__i386__)
    #define VESPER_ARCH_X86 1
#elif defined(_M_ARM64) || defined(__aarch64__)
    #define VESPER_ARCH_ARM64 1
#elif defined(_M_ARM) || defined(__arm__)
    #define VESPER_ARCH_ARM 1
#else
    #define VESPER_ARCH_UNKNOWN 1
#endif