#!/usr/bin/env bash
set -euo pipefail

: "${MATTER_TARGET:?MATTER_TARGET is required}"
: "${MATTER_OUTPUT:?MATTER_OUTPUT is required}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
APP_DIR="${REPO_ROOT}/src/cpp/frontend/matter/app"
BUILD_DIR="build-${MATTER_TARGET}"
DOCKER_IMAGE="${MATTER_DOCKER_IMAGE:-espressif/idf:release-v5.5}"
MATTER_HOME="${REPO_ROOT}/.github/.cache/matter-home"
OUTPUT_DIR="$(dirname "${MATTER_OUTPUT}")"

mkdir -p "${MATTER_HOME}" "${OUTPUT_DIR}"

docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HOME="/work/.github/.cache/matter-home" \
  -v "${REPO_ROOT}:/work" \
  -w "/work/src/cpp/frontend/matter/app" \
  "${DOCKER_IMAGE}" \
  bash -lc "
    set -euo pipefail
    idf.py -B ${BUILD_DIR} set-target ${MATTER_TARGET}
    idf.py -B ${BUILD_DIR} build
    python -m esptool --chip ${MATTER_TARGET} merge-bin --pad-to-size 4MB -o /work/${MATTER_OUTPUT#${REPO_ROOT}/} @${BUILD_DIR}/flash_args
  "
