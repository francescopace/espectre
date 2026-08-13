#!/bin/bash
#
# ESPectre - C++ Test Runner
#
# Configures, builds, and runs the full host-side C++ test suite.
#
# Usage:
#   ./run_all_tests.sh
#   ./run_all_tests.sh -R test_high_accuracy_detector
#   ./run_all_tests.sh --build-dir build-debug -- -VV
#   CTEST_PARALLEL_LEVEL=2 ./run_all_tests.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
BUILD_TYPE="RelWithDebInfo"

CTEST_ARGS=()

detect_parallel_jobs() {
    local jobs="${CTEST_PARALLEL_LEVEL:-}"
    if [[ -n "$jobs" ]] && [[ ! "$jobs" =~ ^[1-9][0-9]*$ ]]; then
        echo "error: CTEST_PARALLEL_LEVEL must be a positive integer" >&2
        return 1
    fi
    if [[ -z "$jobs" ]] && command -v getconf >/dev/null 2>&1; then
        jobs="$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)"
    fi
    if [[ -z "$jobs" ]] && command -v nproc >/dev/null 2>&1; then
        jobs="$(nproc 2>/dev/null || true)"
    fi
    if [[ -z "$jobs" ]] && command -v sysctl >/dev/null 2>&1; then
        jobs="$(sysctl -n hw.logicalcpu 2>/dev/null || true)"
    fi
    if [[ ! "$jobs" =~ ^[1-9][0-9]*$ ]]; then
        jobs=1
    fi
    printf '%s\n' "$jobs"
}

PARALLEL_JOBS="$(detect_parallel_jobs)"

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
cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"

echo "==> Building $BUILD_TYPE tests"
cmake --build "$BUILD_DIR" --config "$BUILD_TYPE" --parallel "$PARALLEL_JOBS"

echo "==> Running CTest with $PARALLEL_JOBS parallel jobs"
if [[ ${#CTEST_ARGS[@]} -gt 0 ]]; then
    ctest --test-dir "$BUILD_DIR" --parallel "$PARALLEL_JOBS" \
        --build-config "$BUILD_TYPE" --output-on-failure "${CTEST_ARGS[@]}"
else
    ctest --test-dir "$BUILD_DIR" --parallel "$PARALLEL_JOBS" \
        --build-config "$BUILD_TYPE" --output-on-failure
fi
