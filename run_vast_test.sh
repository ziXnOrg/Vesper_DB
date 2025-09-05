#!/bin/bash

# Comprehensive Vast.ai test script for Vesper
set -e

echo "=== Vesper Vast.ai Performance Test ==="
echo "Date: $(date)"
echo ""

# Configuration
MAX_PRICE=0.15  # Max price per hour
MIN_CORES=8      # Minimum CPU cores
MIN_RAM=16000    # Minimum RAM in MB

# Function to check if vastai CLI is available
check_vastai() {
    if ! command -v ~/.local/bin/vastai &> /dev/null; then
        echo "ERROR: vastai CLI not found at ~/.local/bin/vastai"
        echo "Install with: pip install --user vastai-cli"
        exit 1
    fi
}

# Function to find or create instance
get_or_create_instance() {
    echo "Checking for existing instances..."
    
    # Check for running instances
    INSTANCE_INFO=$(~/.local/bin/vastai show instances --raw 2>/dev/null | \
                    jq -r '.[] | select(.actual_status == "running") | "\(.id):\(.ssh_host):\(.ssh_port)"' | head -1)
    
    if [ -n "$INSTANCE_INFO" ]; then
        echo "Found existing instance: $INSTANCE_INFO"
        IFS=':' read -r INSTANCE_ID SSH_HOST SSH_PORT <<< "$INSTANCE_INFO"
        return 0
    fi
    
    echo "No running instances found. Creating new one..."
    
    # Search for suitable offers with AVX2 support
    echo "Searching for x86_64 instances with AVX2 support..."
    OFFER_ID=$(~/.local/bin/vastai search offers \
        "dph<=$MAX_PRICE cpu_cores>=$MIN_CORES cpu_ram>=$MIN_RAM inet_down>=100 reliability>0.95" \
        --raw 2>/dev/null | \
        jq -r '.[] | select(.cpu_flags_detected and (.cpu_flags_detected | contains("avx2"))) | .id' | head -1)
    
    if [ -z "$OFFER_ID" ]; then
        echo "ERROR: No suitable instances available within budget"
        echo "Try increasing MAX_PRICE or reducing requirements"
        exit 1
    fi
    
    echo "Creating instance with offer ID: $OFFER_ID"
    CONTRACT_ID=$(~/.local/bin/vastai create instance "$OFFER_ID" \
        --image pytorch/pytorch --disk 20 --ssh --raw | jq -r '.new_contract')
    
    echo "Contract ID: $CONTRACT_ID"
    
    # Wait for instance to be ready
    echo "Waiting for instance to start (up to 5 minutes)..."
    for i in {1..30}; do
        STATUS=$(~/.local/bin/vastai show instances --raw 2>/dev/null | \
                 jq -r ".[] | select(.id == $CONTRACT_ID) | .actual_status" || echo "null")
        
        if [ "$STATUS" = "running" ]; then
            echo "Instance is running!"
            INSTANCE_INFO=$(~/.local/bin/vastai show instances --raw 2>/dev/null | \
                          jq -r ".[] | select(.id == $CONTRACT_ID) | \"\(.id):\(.ssh_host):\(.ssh_port)\"")
            IFS=':' read -r INSTANCE_ID SSH_HOST SSH_PORT <<< "$INSTANCE_INFO"
            return 0
        fi
        
        echo "Status: $STATUS (attempt $i/30)"
        sleep 10
    done
    
    echo "ERROR: Instance failed to start in time"
    exit 1
}

# Function to deploy code
deploy_code() {
    echo ""
    echo "Deploying code to $SSH_HOST:$SSH_PORT..."
    
    # Wait for SSH to be ready
    echo "Waiting for SSH..."
    for i in {1..20}; do
        if ssh -p "$SSH_PORT" -o ConnectTimeout=5 -o StrictHostKeyChecking=no \
               -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
               root@"$SSH_HOST" "echo 'SSH ready'" 2>/dev/null; then
            break
        fi
        echo "  Attempt $i/20..."
        sleep 5
    done
    
    # Upload code
    echo "Uploading Vesper code..."
    rsync -avz --exclude='.git' --exclude='build*' --exclude='*.o' --exclude='*.a' \
        --exclude='experimental' --exclude='.bak' \
        -e "ssh -p $SSH_PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR" \
        /Users/shaiiko/Vesper/ root@"$SSH_HOST":/root/vesper/
}

# Function to build and test
run_tests() {
    echo ""
    echo "Building and testing on remote instance..."
    
    ssh -p "$SSH_PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
        root@"$SSH_HOST" << 'REMOTE_SCRIPT'
set -e

echo "=== Remote Build and Test ==="
cd /root/vesper

# Install dependencies if needed
if ! command -v cmake &>/dev/null; then
    echo "Installing build dependencies..."
    apt-get update && apt-get install -y cmake build-essential git libomp-dev
fi

# Build
echo ""
echo "Building Vesper..."
rm -rf build
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DVESPER_ENABLE_TESTS=ON \
      -DVESPER_ENABLE_BENCH=ON \
      -DCMAKE_CXX_FLAGS="-march=native" \
      ..
cmake --build . -j$(nproc)

# Run tests
echo ""
echo "=== Running Tests ==="

# Check CPU capabilities
echo "CPU Info:"
lscpu | grep -E "Model name|CPU MHz|CPU\(s\)|Thread|Core|Socket|Flags" | head -10
echo ""

# Run index performance tests
if [ -f tests/unit/vesper_tests ]; then
    echo "Running HNSW tests..."
    ./tests/unit/vesper_tests --gtest_filter="Hnsw*" || true
fi

# Run benchmarks if available
if [ -f bench/vesper_bench ]; then
    echo ""
    echo "Running performance benchmarks..."
    ./bench/vesper_bench --benchmark_filter="Hnsw" \
                        --benchmark_report_aggregates_only=true || true
fi

# Custom performance test
echo ""
echo "=== HNSW Performance Test ==="
cat > test_hnsw_perf.cpp << 'EOF'
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "vesper/index/hnsw.hpp"

int main() {
    const std::size_t n_vectors = 50000;
    const std::size_t dim = 128;
    
    // Generate random data
    std::vector<float> data(n_vectors * dim);
    std::vector<std::uint64_t> ids(n_vectors);
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (std::size_t i = 0; i < n_vectors * dim; ++i) {
        data[i] = dist(gen);
    }
    for (std::size_t i = 0; i < n_vectors; ++i) {
        ids[i] = i;
    }
    
    // Build HNSW
    vesper::index::HnswIndex hnsw;
    vesper::index::HnswBuildParams params{
        .M = 16,
        .efConstruction = 200,
        .seed = 42
    };
    
    auto init_result = hnsw.init(dim, params, n_vectors);
    if (!init_result.has_value()) {
        std::cerr << "Failed to init HNSW" << std::endl;
        return 1;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    auto add_result = hnsw.add_batch(ids.data(), data.data(), n_vectors);
    auto end = std::chrono::high_resolution_clock::now();
    
    if (!add_result.has_value()) {
        std::cerr << "Failed to add vectors" << std::endl;
        return 1;
    }
    
    auto build_sec = std::chrono::duration<double>(end - start).count();
    double build_rate = n_vectors / build_sec;
    
    // Test search
    vesper::index::HnswSearchParams search_params{
        .efSearch = 100,
        .k = 10
    };
    
    const std::size_t n_queries = 1000;
    start = std::chrono::high_resolution_clock::now();
    for (std::size_t q = 0; q < n_queries; ++q) {
        auto results = hnsw.search(data.data() + (q % n_vectors) * dim, search_params);
    }
    end = std::chrono::high_resolution_clock::now();
    
    auto search_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_latency = search_ms / n_queries;
    
    std::cout << "========================================" << std::endl;
    std::cout << "HNSW Performance Results" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Vectors: " << n_vectors << std::endl;
    std::cout << "Dimension: " << dim << std::endl;
    std::cout << "Build time: " << build_sec << " sec" << std::endl;
    std::cout << "Build rate: " << static_cast<int>(build_rate) << " vec/sec" << std::endl;
    std::cout << "Avg search latency: " << avg_latency << " ms" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    
    bool build_pass = build_rate >= 50000;
    bool search_pass = avg_latency <= 3.0;
    
    std::cout << "Build rate target (50k vec/sec): " << (build_pass ? "✓ PASS" : "✗ FAIL") << std::endl;
    std::cout << "Search latency target (≤3ms): " << (search_pass ? "✓ PASS" : "✗ FAIL") << std::endl;
    std::cout << "========================================" << std::endl;
    
    return (build_pass && search_pass) ? 0 : 1;
}
EOF

g++ -std=c++20 -O3 -march=native test_hnsw_perf.cpp \
    -I../include -L. -lvesper_core -pthread -o test_hnsw_perf

./test_hnsw_perf

echo ""
echo "=== Test Complete ==="
REMOTE_SCRIPT
}

# Function to cleanup
cleanup_instance() {
    if [ "$1" = "destroy" ]; then
        echo ""
        echo "Destroying instance $INSTANCE_ID..."
        ~/.local/bin/vastai destroy instance "$INSTANCE_ID" 2>/dev/null || true
        echo "Instance destroyed"
    else
        echo ""
        echo "Instance $INSTANCE_ID left running for further testing"
        echo "To destroy manually: vastai destroy instance $INSTANCE_ID"
    fi
}

# Main execution
main() {
    check_vastai
    get_or_create_instance
    deploy_code
    run_tests
    
    # Ask user if they want to destroy the instance
    echo ""
    read -p "Destroy instance? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cleanup_instance destroy
    else
        cleanup_instance keep
    fi
    
    echo ""
    echo "=== Vast.ai test complete ==="
}

# Run main function
main "$@"