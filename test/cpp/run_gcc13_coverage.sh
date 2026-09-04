#!/bin/bash
#
# ESPectre - Canonical GCC 13 Coverage Runner
#
# Builds a Linux/amd64 Ubuntu 24.04 image and runs the coverage workflow with
# the GCC, gcov, and gcovr versions used by CI.
#
# Usage:
#   ./run_gcc13_coverage.sh
#   ./run_gcc13_coverage.sh --ci
#   CTEST_PARALLEL_LEVEL=2 ./run_gcc13_coverage.sh --ci

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DOCKERFILE="$SCRIPT_DIR/gcc13-coverage.Dockerfile"
IMAGE_NAME="espectre-gcc13-coverage:ubuntu-24.04"
PLATFORM="linux/amd64"

if ! command -v docker >/dev/null 2>&1; then
    echo "error: Docker is required to run canonical GCC 13 coverage" >&2
    exit 1
fi

echo "==> Building canonical GCC 13 coverage image"
docker build \
    --platform "$PLATFORM" \
    --tag "$IMAGE_NAME" \
    - < "$DOCKERFILE"

DOCKER_RUN_ARGS=(
    --rm
    --platform "$PLATFORM"
    --user "$(id -u):$(id -g)"
    --env HOME=/tmp
    --env GIT_CONFIG_COUNT=1
    --env GIT_CONFIG_KEY_0=safe.directory
    --env GIT_CONFIG_VALUE_0=/workspace
    --volume "$WORKSPACE_ROOT:/workspace"
    --workdir /workspace
)
if [[ -n "${CTEST_PARALLEL_LEVEL:-}" ]]; then
    DOCKER_RUN_ARGS+=(--env "CTEST_PARALLEL_LEVEL=$CTEST_PARALLEL_LEVEL")
fi

echo "==> Running canonical GCC 13 coverage on $PLATFORM"
docker run "${DOCKER_RUN_ARGS[@]}" \
    "$IMAGE_NAME" \
    "$@"
