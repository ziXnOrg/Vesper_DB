#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <vector>
#include <cmath>
#include <random>

#include "vesper/kernels/dispatch.hpp"
#include "vesper/kernels/backends/scalar.hpp"

#ifdef __x86_64__
#include "vesper/kernels/backends/avx2.hpp"
#include "vesper/kernels/backends/avx512.hpp"
#endif

using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

namespace {

/** \brief Generate random vector with given seed for reproducibility. */
auto generate_random_vector(std::size_t dim, std::uint32_t seed) -> std::vector<float> {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    std::vector<float> v(dim);
    for (auto& x : v) {
        x = dist(gen);
    }
    return v;
}

/** \brief Normalize vector to unit length. */
void normalize_vector(std::vector<float>& v) {
    float norm = 0.0f;
    for (float x : v) {
        norm += x * x;
    }
    norm = std::sqrt(norm);
    if (norm > 0.0f) {
        for (float& x : v) {
            x /= norm;
        }
    }
}

/** \brief Test kernel consistency across different dimensions. */
template<typename KernelFunc>
void test_kernel_dimensions(KernelFunc scalar_func, KernelFunc simd_func, const char* name) {
    // Test various dimensions including non-aligned sizes
    const std::vector<std::size_t> dimensions = {
        1, 3, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 
        384, 512, 768, 1024, 1536, 2048
    };
    
    for (std::size_t dim : dimensions) {
        INFO("Testing " << name << " with dimension " << dim);
        
        auto a = generate_random_vector(dim, 42);
        auto b = generate_random_vector(dim, 137);
        
        const float scalar_result = scalar_func(a, b);
        const float simd_result = simd_func(a, b);
        
        // Allow small relative error due to different accumulation order
        REQUIRE_THAT(simd_result, WithinRel(scalar_result, 1e-5f) || WithinAbs(scalar_result, 1e-6f));
    }
}

/** \brief Test kernel with special values. */
template<typename KernelFunc>
void test_kernel_special_values(KernelFunc func, const char* name) {
    INFO("Testing " << name << " with special values");
    
    // Test zero vectors
    {
        std::vector<float> zeros(16, 0.0f);
        const float result = func(zeros, zeros);
        REQUIRE(std::isfinite(result));
    }
    
    // Test identical vectors
    {
        std::vector<float> ones(16, 1.0f);
        const float result = func(ones, ones);
        REQUIRE(std::isfinite(result));
    }
    
    // Test orthogonal vectors (for inner product)
    {
        std::vector<float> a(16, 0.0f);
        std::vector<float> b(16, 0.0f);
        a[0] = 1.0f;
        b[1] = 1.0f;
        const float result = func(a, b);
        REQUIRE(std::isfinite(result));
    }
}

} // anonymous namespace

TEST_CASE("SIMD kernels correctness", "[kernels][simd]") {
    const auto& scalar_ops = vesper::kernels::get_scalar_ops();
    
    SECTION("L2 squared distance") {
        auto scalar_l2 = [&](const std::vector<float>& a, const std::vector<float>& b) {
            return scalar_ops.l2_sq(a, b);
        };
        
#ifdef __x86_64__
        SECTION("AVX2 consistency") {
            auto avx2_l2 = [](const std::vector<float>& a, const std::vector<float>& b) {
                return vesper::kernels::avx2_l2_sq(a, b);
            };
            test_kernel_dimensions(scalar_l2, avx2_l2, "AVX2 L2");
            test_kernel_special_values(avx2_l2, "AVX2 L2");
        }
        
        SECTION("AVX-512 consistency") {
            auto avx512_l2 = [](const std::vector<float>& a, const std::vector<float>& b) {
                return vesper::kernels::avx512_l2_sq(a, b);
            };
            test_kernel_dimensions(scalar_l2, avx512_l2, "AVX-512 L2");
            test_kernel_special_values(avx512_l2, "AVX-512 L2");
        }
#endif
    }
    
    SECTION("Inner product") {
        auto scalar_ip = [&](const std::vector<float>& a, const std::vector<float>& b) {
            return scalar_ops.inner_product(a, b);
        };
        
#ifdef __x86_64__
        SECTION("AVX2 consistency") {
            auto avx2_ip = [](const std::vector<float>& a, const std::vector<float>& b) {
                return vesper::kernels::avx2_inner_product(a, b);
            };
            test_kernel_dimensions(scalar_ip, avx2_ip, "AVX2 IP");
            test_kernel_special_values(avx2_ip, "AVX2 IP");
        }
        
        SECTION("AVX-512 consistency") {
            auto avx512_ip = [](const std::vector<float>& a, const std::vector<float>& b) {
                return vesper::kernels::avx512_inner_product(a, b);
            };
            test_kernel_dimensions(scalar_ip, avx512_ip, "AVX-512 IP");
            test_kernel_special_values(avx512_ip, "AVX-512 IP");
        }
#endif
    }
    
    SECTION("Cosine similarity") {
        auto scalar_cos = [&](const std::vector<float>& a, const std::vector<float>& b) {
            return scalar_ops.cosine_similarity(a, b);
        };
        
#ifdef __x86_64__
        SECTION("AVX2 consistency") {
            auto avx2_cos = [](const std::vector<float>& a, const std::vector<float>& b) {
                return vesper::kernels::avx2_cosine_similarity(a, b);
            };
            
            // Test with normalized vectors for better precision
            for (std::size_t dim : {8, 16, 32, 64, 128, 256, 384, 512}) {
                INFO("Testing AVX2 cosine with dimension " << dim);
                
                auto a = generate_random_vector(dim, 42);
                auto b = generate_random_vector(dim, 137);
                normalize_vector(a);
                normalize_vector(b);
                
                const float scalar_result = scalar_cos(a, b);
                const float simd_result = avx2_cos(a, b);
                
                REQUIRE_THAT(simd_result, WithinRel(scalar_result, 1e-5f));
            }
        }
        
        SECTION("AVX-512 consistency") {
            auto avx512_cos = [](const std::vector<float>& a, const std::vector<float>& b) {
                return vesper::kernels::avx512_cosine_similarity(a, b);
            };
            
            // Test with normalized vectors for better precision
            for (std::size_t dim : {16, 32, 64, 128, 256, 384, 512}) {
                INFO("Testing AVX-512 cosine with dimension " << dim);
                
                auto a = generate_random_vector(dim, 42);
                auto b = generate_random_vector(dim, 137);
                normalize_vector(a);
                normalize_vector(b);
                
                const float scalar_result = scalar_cos(a, b);
                const float simd_result = avx512_cos(a, b);
                
                REQUIRE_THAT(simd_result, WithinRel(scalar_result, 1e-5f));
            }
        }
#endif
    }
    
    SECTION("Backend auto-selection") {
        const auto& auto_ops = vesper::kernels::select_backend_auto();
        
        // Verify that auto-selected backend produces correct results
        std::vector<float> a = generate_random_vector(128, 42);
        std::vector<float> b = generate_random_vector(128, 137);
        
        const float scalar_l2 = scalar_ops.l2_sq(a, b);
        const float auto_l2 = auto_ops.l2_sq(a, b);
        
        REQUIRE_THAT(auto_l2, WithinRel(scalar_l2, 1e-5f));
        
        const float scalar_ip = scalar_ops.inner_product(a, b);
        const float auto_ip = auto_ops.inner_product(a, b);
        
        REQUIRE_THAT(auto_ip, WithinRel(scalar_ip, 1e-5f));
    }
}