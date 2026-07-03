#!/bin/bash
#
# ESPectre - C++ Test Runner
#
# Configures, builds, and runs the full host-side C++ test suite.
#
# Usage:
#   ./run_all_tests.sh
#   ./run_all_tests.sh -R test_ml_detector
#   ./run_all_tests.sh --build-dir build-debug -- -VV

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

CTEST_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-dir)
            if [[ $# -lt 2 ]]; then
                echo "error: --build-dir requires a value" >&2
                exit 1
            fi
            BUILD_DIR="$SCRIPT_DIR/$2"
            shift 2
            ;;
        --)
            shift
            CTEST_ARGS+=("$@")
            break
            ;;
        *)
            CTEST_ARGS+=("$1")
            shift
            ;;
    esac
done

echo "==> Configuring CMake in $BUILD_DIR"
cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR"

echo "==> Building tests"
cmake --build "$BUILD_DIR"

echo "==> Running CTest"
if [[ ${#CTEST_ARGS[@]} -gt 0 ]]; then
    ctest --test-dir "$BUILD_DIR" --output-on-failure "${CTEST_ARGS[@]}"
else
    ctest --test-dir "$BUILD_DIR" --output-on-failure
fi
