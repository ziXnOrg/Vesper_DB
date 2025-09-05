#!/bin/bash
set -e

echo "Installing dependencies..."
apt-get update -y
apt-get install -y build-essential cmake git g++

echo "CPU Info:"
lscpu | head -10
echo ""
grep -o 'avx[^ ]*' /proc/cpuinfo | sort -u || echo "No AVX"

echo ""
echo "Extracting code..."
cd /root
tar xzf vesper.tar.gz
cd Vesper

echo "Building..."
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -march=native"
make vesper_core -j$(nproc)

echo "Building test..."
g++ -std=c++20 -O3 -march=native -I../include -o perf_test \
    ../tests/unit/actual_index_performance_test.cpp \
    libvesper_core.a -pthread

echo "Running performance test..."
./perf_test

echo "Complete!"
