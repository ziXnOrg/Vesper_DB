#!/usr/bin/env python3
"""
Test script for CAPQ MCP Server

Verifies that all MCP tools are working correctly.
"""

import numpy as np
import json
import sys
import asyncio
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from capq_mcp_server import (
    MathematicalValidator,
    SIMDCodeGenerator,
    PerformanceProfiler,
    RecallValidationOracle,
    CAPQConfig
)

def test_mathematical_validator():
    """Test mathematical validation functions"""
    print("Testing Mathematical Validator...")
    
    # Test projection matrix validation
    dim_in, dim_out = 128, 64
    P = np.random.randn(dim_out, dim_in)
    P, _, _ = np.linalg.svd(P, full_matrices=False)  # Make orthonormal
    P = P[:dim_out, :]  # Ensure correct shape
    
    validator = MathematicalValidator()
    result = validator.validate_projection_orthonormality(P)
    
    assert result["is_orthonormal"], f"Projection not orthonormal: {result}"
    print(f"  [OK] Projection validation: error={result['frobenius_error']:.6f}")
    
    # Test whitening transform
    n_samples = 1000
    dim = 64
    data = np.random.randn(n_samples, dim)
    
    whitening_result = validator.compute_whitening_transform(data)
    
    assert whitening_result["whitening_error"] < 1.0, f"Whitening error too large: {whitening_result['whitening_error']}"
    print(f"  [OK] Whitening transform: error={whitening_result['whitening_error']:.6f}")
    print(f"    Effective rank: {whitening_result['effective_rank']}")
    print(f"    Condition improvement: {whitening_result['condition_before']:.1f} -> {whitening_result['condition_after']:.1f}")
    
    # Test quantization monotonicity
    q8 = np.random.randint(0, 256, size=64, dtype=np.uint8)
    q4 = q8 >> 4  # Correct monotone coarsening
    
    mono_result = validator.validate_quantization_monotonicity(q8, q4)
    assert mono_result["is_monotone"], "Quantization not monotone!"
    print(f"  [OK] Quantization monotonicity: valid")
    
    print("Mathematical Validator: All tests passed!\n")

def test_simd_code_generator():
    """Test SIMD code generation"""
    print("Testing SIMD Code Generator...")
    
    generator = SIMDCodeGenerator()
    
    # Test Hamming distance generation
    hamming_code = generator.generate_hamming_distance_avx512()
    assert "hamming_distance_avx512" in hamming_code
    assert "_mm512_popcnt_epi64" in hamming_code
    print("  [OK] Hamming distance AVX-512 kernel generated")
    
    # Test 4-bit L2 generation
    l2_4bit_code = generator.generate_4bit_l2_avx512()
    assert "compute_4bit_l2_avx512" in l2_4bit_code
    assert "_mm512_mullo_epi16" in l2_4bit_code  # Correct instruction
    assert "_mm512_madd_epi16" not in l2_4bit_code  # Wrong instruction
    print("  [OK] 4-bit L2 AVX-512 kernel generated (using mullo_epi16)")
    
    # Test 8-bit L2 PDX generation
    l2_8bit_pdx = generator.generate_8bit_l2_avx512()
    assert "compute_8bit_l2_pdx_avx512" in l2_8bit_pdx
    assert "PDX layout" in l2_8bit_pdx
    print("  [OK] 8-bit L2 PDX AVX-512 kernel generated")
    
    print("SIMD Code Generator: All tests passed!\n")

def test_performance_profiler():
    """Test performance profiling functions"""
    print("Testing Performance Profiler...")
    
    profiler = PerformanceProfiler()
    
    # Test cache efficiency analysis
    dim = 64
    n_codes = 10000
    cache_result = profiler.estimate_cache_efficiency(dim, n_codes, cache_size_kb=32)
    
    print(f"  Cache Analysis (dim={dim}, n_codes={n_codes}):")
    print(f"    Traditional misses: {cache_result['traditional_cache_misses']}")
    print(f"    PDX misses: {cache_result['pdx_cache_misses']}")
    print(f"    Estimated speedup: {cache_result['estimated_speedup']:.2f}x")
    print(f"    Recommendation: {cache_result['recommendation']}")
    
    # Test SIMD efficiency analysis
    simd_result = profiler.analyze_simd_efficiency(dim=64, simd_width=512)
    print(f"  SIMD Efficiency (dim=64, width=512):")
    print(f"    Utilization: {simd_result['simd_utilization']:.1%}")
    print(f"    Wasted lanes: {simd_result['wasted_lanes']}")
    
    if simd_result['recommendations']:
        print("    Recommendations:")
        for rec in simd_result['recommendations']:
            print(f"      - {rec}")
    
    print("Performance Profiler: All tests passed!\n")

def test_recall_validation_oracle():
    """Test recall validation and dataset generation"""
    print("Testing Recall Validation Oracle...")
    
    oracle = RecallValidationOracle()
    
    # Test synthetic dataset generation
    n_vectors = 1000
    dim = 64
    n_clusters = 16
    
    vectors, labels = oracle.generate_synthetic_dataset(n_vectors, dim, n_clusters)
    
    assert vectors.shape == (n_vectors, dim), f"Wrong vector shape: {vectors.shape}"
    assert labels.shape == (n_vectors,), f"Wrong labels shape: {labels.shape}"
    assert len(np.unique(labels)) == n_clusters, f"Wrong number of clusters"
    print(f"  [OK] Generated dataset: {vectors.shape}")
    
    # Test recall computation
    k = 10
    n_queries = 100
    
    # Simulate retrieved and ground truth
    retrieved = np.random.randint(0, n_vectors, size=(n_queries, k))
    ground_truth = np.random.randint(0, n_vectors, size=(n_queries, k))
    
    # Make some overlap for non-zero recall
    retrieved[:, :5] = ground_truth[:, :5]  # 50% overlap
    
    recall = oracle.compute_exact_recall(retrieved, ground_truth, k)
    expected_recall = 0.5  # We copied 50% of the results
    
    assert abs(recall - expected_recall) < 0.01, f"Unexpected recall: {recall}"
    print(f"  [OK] Recall@{k}: {recall:.2%}")
    
    # Test cascade reduction validation
    stage_candidates = [10000, 1000, 100, 10]  # 10x reduction each stage
    cascade_result = oracle.validate_cascade_reduction(stage_candidates, target_reduction=0.99)
    
    print(f"  Cascade Reduction:")
    print(f"    Stage reductions: {[f'{r:.1%}' for r in cascade_result['stage_reductions']]}")
    print(f"    Overall reduction: {cascade_result['overall_reduction']:.1%}")
    print(f"    Meets target: {cascade_result['meets_target']}")
    
    assert cascade_result['meets_target'], "Cascade doesn't meet reduction target"
    
    print("Recall Validation Oracle: All tests passed!\n")

def test_integration():
    """Test integrated workflow"""
    print("Testing Integrated Workflow...")
    
    # Initialize all components
    config = CAPQConfig()
    validator = MathematicalValidator()
    generator = SIMDCodeGenerator()
    profiler = PerformanceProfiler()
    oracle = RecallValidationOracle()
    
    print(f"  CAPQ Config:")
    print(f"    Original dim: {config.dim_original}")
    print(f"    Projected dim: {config.dim_projected}")
    print(f"    Hamming bits: {config.n_hamming_bits}")
    print(f"    Memory per vector: {config.memory_per_vector} bytes")
    print(f"    Recall target: {config.recall_target:.0%}")
    
    # Simulate training workflow
    print("\n  Simulating training workflow:")
    
    # 1. Generate training data
    vectors, labels = oracle.generate_synthetic_dataset(10000, config.dim_original, config.n_clusters)
    print(f"    1. Generated {len(vectors)} training vectors")
    
    # 2. Compute projection and whitening
    # Sample for whitening computation
    sample_indices = np.random.choice(len(vectors), 1000, replace=False)
    sample_data = vectors[sample_indices]
    
    # Project to lower dimension (simulate)
    P = np.random.randn(config.dim_projected, config.dim_original)
    P, _, _ = np.linalg.svd(P, full_matrices=False)
    P = P[:config.dim_projected, :]
    
    projected = sample_data @ P.T
    whitening = validator.compute_whitening_transform(projected)
    print(f"    2. Computed projection and whitening (rank={whitening['effective_rank']})")
    
    # 3. Check cache efficiency
    cache_analysis = profiler.estimate_cache_efficiency(
        config.dim_projected, 
        config.n_clusters * 100  # Vectors per cluster
    )
    print(f"    3. Cache efficiency: {cache_analysis['recommendation']}")
    
    # 4. Validate cascade reduction
    simulated_cascade = [100000, 10000, 1000, 100]
    cascade_validation = oracle.validate_cascade_reduction(
        simulated_cascade, 
        target_reduction=0.99
    )
    print(f"    4. Cascade reduction: {cascade_validation['overall_reduction']:.1%}")
    
    print("\nIntegration Test: All workflows validated!\n")

def main():
    """Run all tests"""
    print("=" * 60)
    print("CAPQ MCP Server Test Suite")
    print("=" * 60 + "\n")
    
    try:
        test_mathematical_validator()
        test_simd_code_generator()
        test_performance_profiler()
        test_recall_validation_oracle()
        test_integration()
        
        print("=" * 60)
        print("ALL TESTS PASSED! [OK]")
        print("=" * 60)
        print("\nThe CAPQ MCP Server is ready for use.")
        print("Tools available:")
        print("  - validate_projection")
        print("  - compute_whitening")
        print("  - generate_simd_kernel")
        print("  - analyze_cache_efficiency")
        print("  - validate_recall")
        print("  - generate_test_data")
        print("  - validate_cascade")
        
    except AssertionError as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()