#!/usr/bin/env bash
set -euo pipefail

: "${STREAMER_TARGET:?STREAMER_TARGET is required}"
: "${STREAMER_OUTPUT:?STREAMER_OUTPUT is required}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
APP_DIR="${REPO_ROOT}/src/cpp/frontend/streamer/app"
BUILD_DIR="build-${STREAMER_TARGET}"
DOCKER_IMAGE="${STREAMER_DOCKER_IMAGE:-espressif/idf:release-v5.5}"
OUTPUT_DIR="$(dirname "${STREAMER_OUTPUT}")"
STREAMER_OUTPUT_IN_WORK="/work/${STREAMER_OUTPUT#"${REPO_ROOT}"/}"
STREAMER_SDKCONFIG_DEFAULTS="${STREAMER_SDKCONFIG_DEFAULTS:-}"
STREAMER_HOME="${REPO_ROOT}/.github/.cache/streamer-home"

mkdir -p "${STREAMER_HOME}" "${OUTPUT_DIR}"

docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HOME="/work/.github/.cache/streamer-home" \
  -e SDKCONFIG_DEFAULTS="${STREAMER_SDKCONFIG_DEFAULTS}" \
  -e STREAMER_OUTPUT="${STREAMER_OUTPUT_IN_WORK}" \
  -v "${REPO_ROOT}:/work" \
  -w "/work/src/cpp/frontend/streamer/app" \
  "${DOCKER_IMAGE}" \
  bash -lc "
    set -euo pipefail
    case \"${STREAMER_TARGET}\" in
      esp32) STREAMER_CHIP=esp32 ;;
      esp32c3) STREAMER_CHIP=c3 ;;
      esp32c5) STREAMER_CHIP=c5 ;;
      esp32c6) STREAMER_CHIP=c6 ;;
      esp32s3) STREAMER_CHIP=s3 ;;
      *) echo \"Unsupported streamer target: ${STREAMER_TARGET}\" >&2; exit 1 ;;
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
    python /work/espectre streamer build --chip \"\${STREAMER_CHIP}\" --clean
    cd /work/src/cpp/frontend/streamer/app/${BUILD_DIR}
    if python -m esptool merge-bin -h >/dev/null 2>&1; then
      python -m esptool --chip ${STREAMER_TARGET} merge-bin --pad-to-size 4MB -o \"\${STREAMER_OUTPUT}\" @flash_args
    else
      python -m esptool --chip ${STREAMER_TARGET} merge_bin --fill-flash-size 4MB -o \"\${STREAMER_OUTPUT}\" @flash_args
    fi
  "
