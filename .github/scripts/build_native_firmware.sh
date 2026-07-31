#!/usr/bin/env bash
set -euo pipefail

: "${NATIVE_TARGET:?NATIVE_TARGET is required}"
: "${NATIVE_OUTPUT:?NATIVE_OUTPUT is required}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
APP_DIR="${REPO_ROOT}/src/cpp/frontend/native/app"
BUILD_DIR="build-${NATIVE_TARGET}"
DOCKER_IMAGE="${NATIVE_DOCKER_IMAGE:-espressif/idf:release-v5.5}"
OUTPUT_DIR="$(dirname "${NATIVE_OUTPUT}")"
NATIVE_OUTPUT_IN_WORK="/work/${NATIVE_OUTPUT#"${REPO_ROOT}"/}"
NATIVE_OTA_OUTPUT_IN_WORK=""
NATIVE_SDKCONFIG_DEFAULTS="${NATIVE_SDKCONFIG_DEFAULTS:-}"
NATIVE_OTA_SNAPSHOT="${NATIVE_OTA_SNAPSHOT:-0}"
NATIVE_HOME="${REPO_ROOT}/.github/.cache/native-home"
NATIVE_ROOT_MANAGED_COMPONENTS="${NATIVE_HOME}/root_managed_components"

if [ -n "${NATIVE_OTA_OUTPUT:-}" ]; then
  NATIVE_OTA_OUTPUT_IN_WORK="/work/${NATIVE_OTA_OUTPUT#"${REPO_ROOT}"/}"
fi

mkdir -p "${NATIVE_HOME}" "${NATIVE_ROOT_MANAGED_COMPONENTS}" "${OUTPUT_DIR}"

docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HOME="/work/.github/.cache/native-home" \
  -e SDKCONFIG_DEFAULTS="${NATIVE_SDKCONFIG_DEFAULTS}" \
  -e NATIVE_OTA_SNAPSHOT="${NATIVE_OTA_SNAPSHOT}" \
  -e NATIVE_OUTPUT="${NATIVE_OUTPUT_IN_WORK}" \
  -e NATIVE_OTA_OUTPUT="${NATIVE_OTA_OUTPUT_IN_WORK}" \
  -v "${NATIVE_ROOT_MANAGED_COMPONENTS}:/opt/esp/root_managed_components" \
  -v "${REPO_ROOT}:/work" \
  -w "/work/src/cpp/frontend/native/app" \
  "${DOCKER_IMAGE}" \
  bash -lc "
    set -euo pipefail
    case \"${NATIVE_TARGET}\" in
      esp32) NATIVE_CHIP=esp32 ;;
      esp32c3) NATIVE_CHIP=c3 ;;
      esp32c5) NATIVE_CHIP=c5 ;;
      esp32c6) NATIVE_CHIP=c6 ;;
      esp32s3) NATIVE_CHIP=s3 ;;
      *) echo \"Unsupported native target: ${NATIVE_TARGET}\" >&2; exit 1 ;;
    esac
    # ESP-IDF activates a venv where --user installs are rejected; install into HOME instead.
    SITE_PACKAGES=\"\${HOME}/.local/lib/python/site-packages\"
    mkdir -p \"\${SITE_PACKAGES}\"
    export PYTHONPATH=\"\${SITE_PACKAGES}\${PYTHONPATH:+:\${PYTHONPATH}}\"
    if ! python /work/espectre --help >/dev/null 2>&1; then
      python -m pip install --target \"\${SITE_PACKAGES}\" -r /work/requirements.txt
    fi
    if [ -n \"\${SDKCONFIG_DEFAULTS:-}\" ]; then
      export SDKCONFIG_DEFAULTS
    fi
    export ESPECTRE_IDF_BUILD_DIR=${BUILD_DIR}
    cd /work
    python /work/espectre native build --chip \"\${NATIVE_CHIP}\" --clean
    cd /work/src/cpp/frontend/native/app/${BUILD_DIR}
    if python -m esptool merge-bin -h >/dev/null 2>&1; then
      python -m esptool --chip ${NATIVE_TARGET} merge-bin --pad-to-size 4MB -o \"\${NATIVE_OUTPUT}\" @flash_args
    else
      python -m esptool --chip ${NATIVE_TARGET} merge_bin --fill-flash-size 4MB -o \"\${NATIVE_OUTPUT}\" @flash_args
    fi
    if [ -n \"\${NATIVE_OTA_OUTPUT:-}\" ]; then
      cp espectre-native.bin \"\${NATIVE_OTA_OUTPUT}\"
    fi
  "
