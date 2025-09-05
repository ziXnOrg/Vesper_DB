#include <gtest/gtest.h>
#include "vesper/kernels/dispatch.hpp"
#include "vesper/kernels/backends/scalar.hpp"
#include "vesper/kernels/backends/avx2.hpp"
#include "vesper/kernels/backends/avx512.hpp"

#include <random>
#include <numeric>
#include <cmath>
#include <limits>

namespace vesper::kernels {
namespace {

/** \brief Test fixture for SIMD kernel validation. */
class SimdKernelTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Generate test vectors with various properties
        gen_.seed(42);
        
        // Normal distribution vectors
        GenerateNormalVectors(normal_a_, normal_b_, 1024);
        
        // Edge case vectors
        GenerateEdgeCaseVectors();
        
        // Aligned vectors
        GenerateAlignedVectors();
    }
    
    void GenerateNormalVectors(std::vector<float>& a, std::vector<float>& b, 
                               std::size_t dim) {
        a.resize(dim);
        b.resize(dim);
        
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (std::size_t i = 0; i < dim; ++i) {
            a[i] = dist(gen_);
            b[i] = dist(gen_);
        }
    }
    
    void GenerateEdgeCaseVectors() {
        // Zero vectors
        zeros_a_.resize(128, 0.0f);
        zeros_b_.resize(128, 0.0f);
        
        // Ones vectors
        ones_a_.resize(128, 1.0f);
        ones_b_.resize(128, 1.0f);
        
        // Very small values (denormals)
        denormals_a_.resize(128);
        denormals_b_.resize(128);
        for (std::size_t i = 0; i < 128; ++i) {
            denormals_a_[i] = std::numeric_limits<float>::denorm_min() * (i + 1);
            denormals_b_[i] = std::numeric_limits<float>::denorm_min() * (i + 1);
        }
        
        // Large values
        large_a_.resize(128);
        large_b_.resize(128);
        for (std::size_t i = 0; i < 128; ++i) {
            large_a_[i] = 1e6f * (i + 1);
            large_b_[i] = 1e6f * (i + 1);
        }
    }
    
    void GenerateAlignedVectors() {
        // 64-byte aligned vectors for optimal SIMD
        aligned_a_.resize(256);
        aligned_b_.resize(256);
        
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (std::size_t i = 0; i < 256; ++i) {
            aligned_a_[i] = dist(gen_);
            aligned_b_[i] = dist(gen_);
        }
    }
    
    /** \brief Compare results with ULP tolerance. */
    bool CompareWithULP(float a, float b, int max_ulp = 4) {
        // Handle special cases
        if (std::isnan(a) || std::isnan(b)) return false;
        if (std::isinf(a) || std::isinf(b)) return a == b;
        
        // ULP comparison
        std::int32_t ia, ib;
        std::memcpy(&ia, &a, sizeof(float));
        std::memcpy(&ib, &b, sizeof(float));
        
        // Handle sign differences
        if ((ia < 0) != (ib < 0)) {
            return a == b;  // Only exactly equal if signs differ
        }
        
        return std::abs(ia - ib) <= max_ulp;
    }
    
    std::mt19937 gen_;
    std::vector<float> normal_a_, normal_b_;
    std::vector<float> zeros_a_, zeros_b_;
    std::vector<float> ones_a_, ones_b_;
    std::vector<float> denormals_a_, denormals_b_;
    std::vector<float> large_a_, large_b_;
    std::vector<float> aligned_a_, aligned_b_;
};

// L2 Distance Tests

TEST_F(SimdKernelTest, L2Distance_ScalarVsAVX2_Equivalence) {
    if (!cpu_has_avx2()) {
        GTEST_SKIP() << "AVX2 not available";
    }
    
    const auto& scalar_ops = backends::ScalarBackend{};
    const auto& avx2_ops = backends::Avx2Backend{};
    
    // Test various dimensions
    for (std::size_t dim : {7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 1024}) {
        std::vector<float> a(dim), b(dim);
        GenerateNormalVectors(a, b, dim);
        
        float scalar_result = scalar_ops.l2_sq(a, b);
        float avx2_result = avx2_ops.l2_sq(a, b);
        
        EXPECT_TRUE(CompareWithULP(scalar_result, avx2_result, 8))
            << "Dimension: " << dim 
            << ", Scalar: " << scalar_result 
            << ", AVX2: " << avx2_result;
    }
}

TEST_F(SimdKernelTest, L2Distance_EdgeCases) {
    const auto& ops = select_backend_auto();
    
    // Zero vectors
    float zero_dist = ops.l2_sq(zeros_a_, zeros_b_);
    EXPECT_FLOAT_EQ(zero_dist, 0.0f);
    
    // Identical vectors
    float same_dist = ops.l2_sq(ones_a_, ones_a_);
    EXPECT_FLOAT_EQ(same_dist, 0.0f);
    
    // Orthogonal unit vectors
    std::vector<float> unit_x{1.0f, 0.0f, 0.0f};
    std::vector<float> unit_y{0.0f, 1.0f, 0.0f};
    float ortho_dist = ops.l2_sq(unit_x, unit_y);
    EXPECT_FLOAT_EQ(ortho_dist, 2.0f);
}

TEST_F(SimdKernelTest, L2Distance_NumericalStability) {
    const auto& ops = select_backend_auto();
    
    // Test with denormals
    float denorm_dist = ops.l2_sq(denormals_a_, denormals_b_);
    EXPECT_FALSE(std::isnan(denorm_dist));
    EXPECT_FALSE(std::isinf(denorm_dist));
    
    // Test with large values (potential overflow)
    float large_dist = ops.l2_sq(large_a_, large_b_);
    EXPECT_FALSE(std::isnan(large_dist));
    
    // Test cancellation
    std::vector<float> a{1e10f, 1.0f};
    std::vector<float> b{1e10f, 2.0f};
    float cancel_dist = ops.l2_sq(a, b);
    EXPECT_NEAR(cancel_dist, 1.0f, 1e-5f);
}

// Inner Product Tests

TEST_F(SimdKernelTest, InnerProduct_ScalarVsAVX2_Equivalence) {
    if (!cpu_has_avx2()) {
        GTEST_SKIP() << "AVX2 not available";
    }
    
    const auto& scalar_ops = backends::ScalarBackend{};
    const auto& avx2_ops = backends::Avx2Backend{};
    
    for (std::size_t dim : {8, 16, 32, 64, 128, 256}) {
        std::vector<float> a(dim), b(dim);
        GenerateNormalVectors(a, b, dim);
        
        float scalar_result = scalar_ops.inner_product(a, b);
        float avx2_result = avx2_ops.inner_product(a, b);
        
        EXPECT_TRUE(CompareWithULP(scalar_result, avx2_result, 8))
            << "Dimension: " << dim;
    }
}

TEST_F(SimdKernelTest, InnerProduct_Properties) {
    const auto& ops = select_backend_auto();
    
    // Commutative
    float ab = ops.inner_product(normal_a_, normal_b_);
    float ba = ops.inner_product(normal_b_, normal_a_);
    EXPECT_FLOAT_EQ(ab, ba);
    
    // Distributive
    std::vector<float> c(normal_a_.size());
    for (std::size_t i = 0; i < c.size(); ++i) {
        c[i] = normal_a_[i] + normal_b_[i];
    }
    
    float a_c = ops.inner_product(normal_a_, c);
    float a_a = ops.inner_product(normal_a_, normal_a_);
    float a_b = ops.inner_product(normal_a_, normal_b_);
    
    EXPECT_NEAR(a_c, a_a + a_b, 1e-5f);
}

// Cosine Similarity Tests

TEST_F(SimdKernelTest, CosineSimilarity_Bounds) {
    const auto& ops = select_backend_auto();
    
    // Normalize vectors
    auto normalize = [](std::vector<float>& v) {
        float norm = 0.0f;
        for (float x : v) norm += x * x;
        norm = std::sqrt(norm);
        for (float& x : v) x /= norm;
    };
    
    normalize(normal_a_);
    normalize(normal_b_);
    
    float cosine = ops.cosine_similarity(normal_a_, normal_b_);
    
    EXPECT_GE(cosine, -1.0f);
    EXPECT_LE(cosine, 1.0f);
    
    // Same vector -> cosine = 1
    float self_cosine = ops.cosine_similarity(normal_a_, normal_a_);
    EXPECT_NEAR(self_cosine, 1.0f, 1e-6f);
}

// Batch Distance Tests

TEST_F(SimdKernelTest, BatchDistance_Correctness) {
    const std::size_t n = 100;
    const std::size_t dim = 128;
    
    // Generate dataset
    std::vector<float> data(n * dim);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : data) {
        v = dist(gen_);
    }
    
    // Compute distances to first vector
    const float* query = data.data();
    std::vector<float> distances(n);
    
    const auto& ops = select_backend_auto();
    
    for (std::size_t i = 0; i < n; ++i) {
        distances[i] = ops.l2_sq(
            std::span(query, dim),
            std::span(data.data() + i * dim, dim)
        );
    }
    
    // First vector to itself should be 0
    EXPECT_FLOAT_EQ(distances[0], 0.0f);
    
    // All other distances should be positive
    for (std::size_t i = 1; i < n; ++i) {
        EXPECT_GT(distances[i], 0.0f);
    }
}

// Performance Characteristics Tests

TEST_F(SimdKernelTest, Performance_AlignmentImpact) {
    const auto& ops = select_backend_auto();
    
    // Test aligned vs unaligned performance
    // (This is more of a smoke test - real perf testing in benchmarks)
    
    // Aligned vectors (address % 64 == 0)
    float aligned_dist = ops.l2_sq(aligned_a_, aligned_b_);
    
    // Unaligned vectors (offset by 1 float)
    std::span<const float> unaligned_a(aligned_a_.data() + 1, aligned_a_.size() - 1);
    std::span<const float> unaligned_b(aligned_b_.data() + 1, aligned_b_.size() - 1);
    float unaligned_dist = ops.l2_sq(unaligned_a, unaligned_b);
    
    // Both should compute correctly
    EXPECT_FALSE(std::isnan(aligned_dist));
    EXPECT_FALSE(std::isnan(unaligned_dist));
}

// Backend Selection Tests

TEST_F(SimdKernelTest, BackendSelection_Correctness) {
    // Test that we select appropriate backends
    const auto& auto_backend = select_backend_auto();
    
    #ifdef __AVX512F__
    if (cpu_has_avx512()) {
        // Should select AVX-512
        const auto& avx512_backend = backends::Avx512Backend{};
        float auto_result = auto_backend.l2_sq(normal_a_, normal_b_);
        float avx512_result = avx512_backend.l2_sq(normal_a_, normal_b_);
        EXPECT_FLOAT_EQ(auto_result, avx512_result);
    }
    #endif
    
    #ifdef __AVX2__
    if (cpu_has_avx2() && !cpu_has_avx512()) {
        // Should select AVX2
        const auto& avx2_backend = backends::Avx2Backend{};
        float auto_result = auto_backend.l2_sq(normal_a_, normal_b_);
        float avx2_result = avx2_backend.l2_sq(normal_a_, normal_b_);
        EXPECT_FLOAT_EQ(auto_result, avx2_result);
    }
    #endif
}

// Cross-validation Tests

TEST_F(SimdKernelTest, CrossValidation_AllBackends) {
    const auto& scalar = backends::ScalarBackend{};
    
    // Generate test vector
    const std::size_t dim = 128;
    std::vector<float> a(dim), b(dim);
    GenerateNormalVectors(a, b, dim);
    
    float scalar_l2 = scalar.l2_sq(a, b);
    float scalar_ip = scalar.inner_product(a, b);
    float scalar_cos = scalar.cosine_similarity(a, b);
    
    #ifdef __AVX2__
    if (cpu_has_avx2()) {
        const auto& avx2 = backends::Avx2Backend{};
        EXPECT_TRUE(CompareWithULP(scalar_l2, avx2.l2_sq(a, b), 8));
        EXPECT_TRUE(CompareWithULP(scalar_ip, avx2.inner_product(a, b), 8));
        EXPECT_TRUE(CompareWithULP(scalar_cos, avx2.cosine_similarity(a, b), 8));
    }
    #endif
    
    #ifdef __AVX512F__
    if (cpu_has_avx512()) {
        const auto& avx512 = backends::Avx512Backend{};
        EXPECT_TRUE(CompareWithULP(scalar_l2, avx512.l2_sq(a, b), 8));
        EXPECT_TRUE(CompareWithULP(scalar_ip, avx512.inner_product(a, b), 8));
        EXPECT_TRUE(CompareWithULP(scalar_cos, avx512.cosine_similarity(a, b), 8));
    }
    #endif
}

} // namespace
} // namespace vesper::kernels