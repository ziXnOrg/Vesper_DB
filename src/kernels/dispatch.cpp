#include "vesper/kernels/dispatch.hpp"
#include "vesper/kernels/backends/scalar.hpp"

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64)
#ifdef _MSC_VER
#include <intrin.h>  // For __cpuid on MSVC
#else
#include <cpuid.h>  // For __get_cpuid on GCC/Clang
#endif
#include "vesper/kernels/backends/avx2.hpp"
// Only include AVX-512 if explicitly enabled
#ifdef __AVX512F__
#include "vesper/kernels/backends/avx512.hpp"
#endif
#endif

#ifdef __aarch64__
// ARM NEON support to be added
#endif

#if defined(VESPER_ENABLE_ACCELERATE) && VESPER_ENABLE_ACCELERATE && defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif

#include <atomic>
namespace {
#if defined(VESPER_ENABLE_ACCELERATE) && VESPER_ENABLE_ACCELERATE && defined(__APPLE__)
inline float l2_sq_accel(std::span<const float> a, std::span<const float> b) noexcept {
    const float* ap = a.data();
    const float* bp = b.data();
    const vDSP_Length n = static_cast<vDSP_Length>(a.size());
    // dist^2 = ||a||^2 + ||b||^2 - 2 a·b, using vDSP primitives
    float aa = 0.0f, bb = 0.0f, ab = 0.0f;
    vDSP_svesq(ap, 1, &aa, n);
    vDSP_svesq(bp, 1, &bb, n);
    vDSP_dotpr(ap, 1, bp, 1, &ab, n);
    float d2 = aa + bb - 2.0f * ab;
    return d2 >= 0.0f ? d2 : 0.0f; // numeric guard
}
#endif
}

#include <cstdint>
#include <cstdlib>

namespace vesper::kernels {

// Forward declare scalar ops to satisfy name lookup in this TU
const KernelOps& get_scalar_ops() noexcept;


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
#ifdef _MSC_VER
[[maybe_unused]]
#else
[[gnu::cold]]
#endif
inline auto detect_cpu_features() noexcept -> CpuFeatures {
    CpuFeatures features{};

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64)
    // Check for AVX2 and FMA support
#ifdef _MSC_VER
    int cpu_info[4];
    __cpuid(cpu_info, 0);
    const unsigned int max_level = cpu_info[0];
    
    // Check for AVX2 (CPUID.07H:EBX.AVX2[bit 5])
    if (max_level >= 7) {
        __cpuidex(cpu_info, 7, 0);
        features.has_avx2 = (cpu_info[1] & (1u << 5)) != 0;
        // Check for AVX-512F (CPUID.07H:EBX.AVX512F[bit 16])
        features.has_avx512f = (cpu_info[1] & (1u << 16)) != 0;
    }
    
    // Check for FMA (CPUID.01H:ECX.FMA[bit 12])
    __cpuid(cpu_info, 1);
    features.has_fma = (cpu_info[2] & (1u << 12)) != 0;
#else
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
#endif  // defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64)

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
#if defined(VESPER_ENABLE_ACCELERATE) && VESPER_ENABLE_ACCELERATE && defined(__APPLE__)
static inline const KernelOps& get_accelerate_ops() noexcept {
    static const KernelOps ops = []{
        KernelOps k = get_scalar_ops();
        k.l2_sq = &l2_sq_accel;
        return k;
    }();
    return ops;
}
#endif


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

/** \brief New environment override for backend selection (RFC).
 *
 * VESPER_KERNEL_BACKEND takes precedence over legacy VESPER_SIMD_MASK.
 * Accepted values (case-insensitive): scalar, avx2, avx512, accelerate, neon, auto.
 * - "auto" or unknown values are ignored (no override).
 */
inline auto get_backend_name_override() noexcept -> std::string_view {
    static const char* env = std::getenv("VESPER_KERNEL_BACKEND");
    if (!env || !env[0]) return "";
    auto eq_ci = [](std::string_view a, std::string_view b) {
        if (a.size() != b.size()) return false;
        for (std::size_t i = 0; i < a.size(); ++i) {
            char ca = a[i]; char cb = b[i];
            if (ca >= 'A' && ca <= 'Z') ca = static_cast<char>(ca - 'A' + 'a');
            if (cb >= 'A' && cb <= 'Z') cb = static_cast<char>(cb - 'A' + 'a');
            if (ca != cb) return false;
        }
        return true;
    };
    const std::string_view s{env};
    if (eq_ci(s, "scalar"))      return "scalar";
    if (eq_ci(s, "avx2"))        return "avx2";
    if (eq_ci(s, "avx512"))      return "avx512";
    if (eq_ci(s, "accelerate"))  return "accelerate";
    if (eq_ci(s, "neon"))        return "neon";
    if (eq_ci(s, "auto"))        return ""; // no override
    return ""; // unknown -> ignore
}

} // namespace detail

const KernelOps& select_backend(std::string_view name) noexcept {
    // Check for explicit backend selection
    if (name == "scalar") {
        return get_scalar_ops();
    }

#if defined(VESPER_ENABLE_ACCELERATE) && VESPER_ENABLE_ACCELERATE && defined(__APPLE__)
    if (name == "accelerate") {
        return detail::get_accelerate_ops();
    }
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64)
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
    // Check for environment override: name has precedence over legacy mask
    const auto name_override = detail::get_backend_name_override();
    if (!name_override.empty()) {
        return select_backend(name_override);
    }
    const auto mask_override = detail::get_simd_override();
    if (!mask_override.empty()) {
        return select_backend(mask_override);
    }

#if defined(VESPER_ENABLE_ACCELERATE) && VESPER_ENABLE_ACCELERATE && defined(__APPLE__)
    // Prefer Accelerate when enabled on Apple platforms
    return detail::get_accelerate_ops();
#endif

    // Auto-detect best available backend
    const auto& features = detail::get_cpu_features();

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64)
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