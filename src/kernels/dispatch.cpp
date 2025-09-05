#include "vesper/kernels/dispatch.hpp"
#include "vesper/kernels/backends/scalar.hpp"

#ifdef __x86_64__
#include <cpuid.h>
#include "vesper/kernels/backends/avx2.hpp"
// Only include AVX-512 if explicitly enabled
#ifdef __AVX512F__
#include "vesper/kernels/backends/avx512.hpp"
#endif
#endif

#ifdef __aarch64__
// ARM NEON support to be added
#endif

#include <atomic>
#include <cstdint>
#include <cstdlib>

namespace vesper::kernels {

namespace detail {

/** \brief CPU feature detection flags. */
struct CpuFeatures {
    bool has_avx2{false};
    bool has_avx512f{false};
    bool has_fma{false};
    bool has_neon{false};
};

/** \brief Detect CPU features at runtime using CPUID.
 *  
 * Thread-safe through atomic initialization.
 * Results cached for process lifetime.
 */
[[gnu::cold]] inline auto detect_cpu_features() noexcept -> CpuFeatures {
    CpuFeatures features{};
    
#ifdef __x86_64__
    // Check for AVX2 and FMA support
    unsigned int eax, ebx, ecx, edx;
    
    // Check maximum supported CPUID level
    if (__get_cpuid(0, &eax, &ebx, &ecx, &edx)) {
        const unsigned int max_level = eax;
        
        // Check for AVX2 (CPUID.07H:EBX.AVX2[bit 5])
        if (max_level >= 7) {
            __cpuid_count(7, 0, eax, ebx, ecx, edx);
            features.has_avx2 = (ebx & (1u << 5)) != 0;
            
            // Check for AVX-512F (CPUID.07H:EBX.AVX512F[bit 16])
            features.has_avx512f = (ebx & (1u << 16)) != 0;
        }
        
        // Check for FMA (CPUID.01H:ECX.FMA[bit 12])
        if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
            features.has_fma = (ecx & (1u << 12)) != 0;
        }
    }
#endif

#ifdef __aarch64__
    // ARM NEON is mandatory on AArch64
    features.has_neon = true;
#endif
    
    return features;
}

/** \brief Get CPU features singleton.
 *
 * Thread-safe initialization via call_once semantics.
 */
inline const CpuFeatures& get_cpu_features() noexcept {
    static const CpuFeatures features = detect_cpu_features();
    return features;
}

/** \brief Environment variable override for kernel selection.
 *
 * Checks VESPER_SIMD_MASK to force specific backend.
 * Useful for testing and debugging.
 */
inline auto get_simd_override() noexcept -> std::string_view {
    static const char* env = std::getenv("VESPER_SIMD_MASK");
    if (env && env[0]) {
        // Parse mask: 0=force scalar, 1=allow AVX2, 2=allow AVX512
        const int mask = std::atoi(env);
        if (mask == 0) return "scalar";
        if (mask == 1) return "avx2";
        if (mask == 2) return "avx512";
    }
    return "";
}

} // namespace detail

const KernelOps& select_backend(std::string_view name) noexcept {
    // Check for explicit backend selection
    if (name == "scalar") {
        return get_scalar_ops();
    }
    
#ifdef __x86_64__
    if (name == "avx2") {
        const auto& features = detail::get_cpu_features();
        if (features.has_avx2 && features.has_fma) {
            return get_avx2_ops();
        }
        // Fall back to scalar if AVX2 not available
        return get_scalar_ops();
    }
    
    if (name == "avx512") {
        const auto& features = detail::get_cpu_features();
#ifdef __AVX512F__
        if (features.has_avx512f) {
            return get_avx512_ops();
        }
#endif
        // Fall back to AVX2 if available
        if (features.has_avx2 && features.has_fma) {
            return get_avx2_ops();
        }
        // Fall back to scalar
        return get_scalar_ops();
    }
#endif

#ifdef __aarch64__
    if (name == "neon") {
        // NEON backend not yet implemented, return scalar
        return get_scalar_ops();
    }
#endif
    
    // Unknown backend, return scalar
    return get_scalar_ops();
}

const KernelOps& select_backend_auto() noexcept {
    // Check for environment override
    const auto override = detail::get_simd_override();
    if (!override.empty()) {
        return select_backend(override);
    }
    
    // Auto-detect best available backend
    const auto& features = detail::get_cpu_features();
    
#ifdef __x86_64__
#ifdef __AVX512F__
    // Prefer AVX-512 if available
    if (features.has_avx512f) {
        return get_avx512_ops();
    }
#endif
    
    // Fall back to AVX2 if available
    if (features.has_avx2 && features.has_fma) {
        return get_avx2_ops();
    }
#endif

#ifdef __aarch64__
    if (features.has_neon) {
        // NEON backend not yet implemented, return scalar
        return get_scalar_ops();
    }
#endif
    
    // Default to scalar implementation
    return get_scalar_ops();
}

} // namespace vesper::kernels