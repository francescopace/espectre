#!/usr/bin/env bash
set -euo pipefail

: "${MATTER_TARGET:?MATTER_TARGET is required}"
: "${MATTER_OUTPUT:?MATTER_OUTPUT is required}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="build-${MATTER_TARGET}"
DOCKER_IMAGE="${MATTER_DOCKER_IMAGE:-espressif/idf:release-v5.5}"
MATTER_HOME="${REPO_ROOT}/.github/.cache/matter-home"
OUTPUT_DIR="$(dirname "${MATTER_OUTPUT}")"
MATTER_OUTPUT_IN_WORK="/work/${MATTER_OUTPUT#"${REPO_ROOT}"/}"
MATTER_SDKCONFIG_DEFAULTS="${MATTER_SDKCONFIG_DEFAULTS:-}"

mkdir -p "${MATTER_HOME}" "${OUTPUT_DIR}"

docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HOME="/work/.github/.cache/matter-home" \
  -e SDKCONFIG_DEFAULTS="${MATTER_SDKCONFIG_DEFAULTS}" \
  -e MATTER_OUTPUT="${MATTER_OUTPUT_IN_WORK}" \
  -v "${REPO_ROOT}:/work" \
  -w "/work/src/cpp/frontend/matter/app" \
  "${DOCKER_IMAGE}" \
  bash -lc "
    set -euo pipefail
    if [ -n \"\${SDKCONFIG_DEFAULTS:-}\" ]; then
      export SDKCONFIG_DEFAULTS
    fi
    idf.py -B ${BUILD_DIR} set-target ${MATTER_TARGET}
    idf.py -B ${BUILD_DIR} build
    if python -m esptool merge-bin -h >/dev/null 2>&1; then
      python -m esptool --chip ${MATTER_TARGET} merge-bin --pad-to-size 4MB -o \"\${MATTER_OUTPUT}\" @${BUILD_DIR}/flash_args
    else
      python -m esptool --chip ${MATTER_TARGET} merge_bin --fill-flash-size 4MB -o \"\${MATTER_OUTPUT}\" @${BUILD_DIR}/flash_args
    fi
  "
