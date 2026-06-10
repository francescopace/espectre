#!/usr/bin/env bash
set -euo pipefail

: "${BLE_TARGET:?BLE_TARGET is required}"
: "${BLE_OUTPUT:?BLE_OUTPUT is required}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
APP_DIR="${REPO_ROOT}/src/cpp/frontend/ble/app"
BUILD_DIR="build-${BLE_TARGET}"
DOCKER_IMAGE="${BLE_DOCKER_IMAGE:-espressif/idf:release-v5.5}"
OUTPUT_DIR="$(dirname "${BLE_OUTPUT}")"
BLE_SDKCONFIG_DEFAULTS="${BLE_SDKCONFIG_DEFAULTS:-}"
BLE_HOME="${REPO_ROOT}/.github/.cache/ble-home"

mkdir -p "${BLE_HOME}" "${OUTPUT_DIR}"
rm -rf "${APP_DIR}/${BUILD_DIR}"

docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HOME="/work/.github/.cache/ble-home" \
  -e SDKCONFIG_DEFAULTS="${BLE_SDKCONFIG_DEFAULTS}" \
  -v "${REPO_ROOT}:/work" \
  -w "/work/src/cpp/frontend/ble/app" \
  "${DOCKER_IMAGE}" \
  bash -lc "
    set -euo pipefail
    if [ -n \"\${SDKCONFIG_DEFAULTS:-}\" ]; then
      export SDKCONFIG_DEFAULTS
    fi
    idf.py -B ${BUILD_DIR} set-target ${BLE_TARGET}
    idf.py -B ${BUILD_DIR} build
    cd ${BUILD_DIR}
    if python -m esptool merge-bin -h >/dev/null 2>&1; then
      python -m esptool --chip ${BLE_TARGET} merge-bin --pad-to-size 4MB -o /work/${BLE_OUTPUT#${REPO_ROOT}/} @flash_args
    else
      python -m esptool --chip ${BLE_TARGET} merge_bin --fill-flash-size 4MB -o /work/${BLE_OUTPUT#${REPO_ROOT}/} @flash_args
    fi
  "
