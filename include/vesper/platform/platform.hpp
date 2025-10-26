#pragma once

/** \file platform.hpp
 *  \brief Central platform abstraction header.
 *
 * Single include for all platform abstractions.
 * This provides a clean interface for cross-platform development.
 */

// Include all platform abstractions
#include "vesper/platform/compiler.hpp"
#include "vesper/platform/memory.hpp"
#include "vesper/platform/intrinsics.hpp"
#include "vesper/platform/filesystem.hpp"
#include "vesper/platform/parallel.hpp"

namespace vesper::platform {

/** \brief Platform information structure. */
struct PlatformInfo {
    const char* os_name;
    const char* arch_name;
    const char* compiler_name;
    int compiler_version;
    bool has_openmp;
    bool has_avx2;
    bool has_avx512;
    int cache_line_size;
    int num_cores;
};

/** \brief Get current platform information.
 *
 * \return Platform details
 */
[[nodiscard]] inline auto get_platform_info() noexcept -> PlatformInfo {
    PlatformInfo info{};
    
    // OS detection
#ifdef VESPER_PLATFORM_WINDOWS
    info.os_name = "Windows";
#elif defined(VESPER_PLATFORM_LINUX)
    info.os_name = "Linux";
#elif defined(VESPER_PLATFORM_MACOS)
    info.os_name = "macOS";
#else
    info.os_name = "Unknown";
#endif
    
    // Architecture detection
#ifdef VESPER_ARCH_X64
    info.arch_name = "x86_64";
#elif defined(VESPER_ARCH_X86)
    info.arch_name = "x86";
#elif defined(VESPER_ARCH_ARM64)
    info.arch_name = "ARM64";
#elif defined(VESPER_ARCH_ARM)
    info.arch_name = "ARM";
#else
    info.arch_name = "Unknown";
#endif
    
    // Compiler detection
#ifdef VESPER_COMPILER_MSVC
    info.compiler_name = "MSVC";
    info.compiler_version = VESPER_COMPILER_VERSION;
#elif defined(VESPER_COMPILER_CLANG)
    info.compiler_name = "Clang";
    info.compiler_version = VESPER_COMPILER_VERSION;
#elif defined(VESPER_COMPILER_GCC)
    info.compiler_name = "GCC";
    info.compiler_version = VESPER_COMPILER_VERSION;
#else
    info.compiler_name = "Unknown";
    info.compiler_version = 0;
#endif
    
    // Feature detection
#ifdef _OPENMP
    info.has_openmp = true;
#else
    info.has_openmp = false;
#endif
    
    // SIMD detection (basic check, real detection should use CPUID)
#ifdef __AVX2__
    info.has_avx2 = true;
#else
    info.has_avx2 = false;
#endif
    
#ifdef __AVX512F__
    info.has_avx512 = true;
#else
    info.has_avx512 = false;
#endif
    
    info.cache_line_size = VESPER_CACHE_LINE_SIZE;
    info.num_cores = get_num_threads();
    
    return info;
}

/** \brief Print platform information to stdout. */
inline auto print_platform_info() -> void {
    auto info = get_platform_info();
    
    std::printf("Platform Information:\n");
    std::printf("  OS: %s\n", info.os_name);
    std::printf("  Architecture: %s\n", info.arch_name);
    std::printf("  Compiler: %s v%d\n", info.compiler_name, info.compiler_version);
    std::printf("  OpenMP: %s\n", info.has_openmp ? "Yes" : "No");
    std::printf("  AVX2: %s\n", info.has_avx2 ? "Yes" : "No");
    std::printf("  AVX512: %s\n", info.has_avx512 ? "Yes" : "No");
    std::printf("  Cache Line: %d bytes\n", info.cache_line_size);
    std::printf("  CPU Cores: %d\n", info.num_cores);
}

} // namespace vesper::platform