#!/usr/bin/env bash
set -euo pipefail

# Build and run tests with CTest
BUILD_DIR=${BUILD_DIR:-build}
CONFIG=${CONFIG:-Release}

cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=$CONFIG
cmake --build "$BUILD_DIR" --config $CONFIG -j
ctest --test-dir "$BUILD_DIR" --output-on-failure

