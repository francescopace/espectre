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
    case \"${MATTER_TARGET}\" in
      esp32) MATTER_CHIP=esp32 ;;
      esp32c3) MATTER_CHIP=c3 ;;
      esp32c5) MATTER_CHIP=c5 ;;
      esp32c6) MATTER_CHIP=c6 ;;
      esp32s3) MATTER_CHIP=s3 ;;
      *) echo \"Unsupported Matter target: ${MATTER_TARGET}\" >&2; exit 1 ;;
    esac
    if ! python /work/espectre --help >/dev/null 2>&1; then
      python -m pip install --user -r /work/requirements.txt
    fi
    if [ -n \"\${SDKCONFIG_DEFAULTS:-}\" ]; then
      export SDKCONFIG_DEFAULTS
    fi
    export ESPECTRE_IDF_BUILD_DIR=${BUILD_DIR}
    cd /work
    python /work/espectre matter build --chip \"\${MATTER_CHIP}\" --clean
    cd ${BUILD_DIR}
    if python -m esptool merge-bin -h >/dev/null 2>&1; then
      python -m esptool --chip ${MATTER_TARGET} merge-bin --pad-to-size 4MB -o \"\${MATTER_OUTPUT}\" @flash_args
    else
      python -m esptool --chip ${MATTER_TARGET} merge_bin --fill-flash-size 4MB -o \"\${MATTER_OUTPUT}\" @flash_args
    fi
  "
