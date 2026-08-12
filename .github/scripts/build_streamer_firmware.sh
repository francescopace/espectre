#!/usr/bin/env bash
set -euo pipefail

: "${STREAMER_TARGET:?STREAMER_TARGET is required}"
: "${STREAMER_OUTPUT:?STREAMER_OUTPUT is required}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="build-${STREAMER_TARGET}"
DOCKER_IMAGE="${STREAMER_DOCKER_IMAGE:-espressif/idf:v5.5.5@sha256:a9231d0697ab8f7517cc072e93b7c83e04907bfbfba80b6440d7dbbf90665cf2}"
OUTPUT_DIR="$(dirname "${STREAMER_OUTPUT}")"
STREAMER_OUTPUT_IN_WORK="/work/${STREAMER_OUTPUT#"${REPO_ROOT}"/}"
STREAMER_SDKCONFIG_DEFAULTS="${STREAMER_SDKCONFIG_DEFAULTS:-}"
STREAMER_HOME="${REPO_ROOT}/.github/.cache/streamer-home"
STREAMER_ROOT_MANAGED_COMPONENTS="${STREAMER_HOME}/root_managed_components"

mkdir -p "${STREAMER_HOME}" "${STREAMER_ROOT_MANAGED_COMPONENTS}" "${OUTPUT_DIR}"

docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HOME="/work/.github/.cache/streamer-home" \
  -e SDKCONFIG_DEFAULTS="${STREAMER_SDKCONFIG_DEFAULTS}" \
  -e STREAMER_OUTPUT="${STREAMER_OUTPUT_IN_WORK}" \
  -v "${STREAMER_ROOT_MANAGED_COMPONENTS}:/opt/esp/root_managed_components" \
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
    REQUIREMENTS_HASH=\"\$(sha256sum /work/requirements.txt | cut -d ' ' -f 1)\"
    REQUIREMENTS_MARKER=\"\${HOME}/.espectre-requirements-\${REQUIREMENTS_HASH}\"
    export PYTHONPATH=\"\${SITE_PACKAGES}\${PYTHONPATH:+:\${PYTHONPATH}}\"
    if [ ! -f \"\${REQUIREMENTS_MARKER}\" ]; then
      rm -rf \"\${SITE_PACKAGES}\"
      mkdir -p \"\${SITE_PACKAGES}\"
      python -m pip install --target \"\${SITE_PACKAGES}\" -r /work/requirements.txt
      rm -f \"\${HOME}\"/.espectre-requirements-*
      touch \"\${REQUIREMENTS_MARKER}\"
    fi
    if [ -n \"\${SDKCONFIG_DEFAULTS:-}\" ]; then
      export SDKCONFIG_DEFAULTS
    fi
    export ESPECTRE_IDF_BUILD_DIR=${BUILD_DIR}
    cd /work
    python /work/espectre streamer build --chip \"\${STREAMER_CHIP}\" --backend local --clean
    cd /work/src/cpp/frontend/streamer/app/${BUILD_DIR}
    if python -m esptool merge-bin -h >/dev/null 2>&1; then
      python -m esptool --chip ${STREAMER_TARGET} merge-bin --pad-to-size 4MB -o \"\${STREAMER_OUTPUT}\" @flash_args
    else
      python -m esptool --chip ${STREAMER_TARGET} merge_bin --fill-flash-size 4MB -o \"\${STREAMER_OUTPUT}\" @flash_args
    fi
    python /work/.github/scripts/build_firmware_compliance.py \
      --frontend streamer \
      --project-description /work/src/cpp/frontend/streamer/app/${BUILD_DIR}/project_description.json \
      --firmware \"\${STREAMER_OUTPUT}\" \
      --output-dir \"\$(dirname \"\${STREAMER_OUTPUT}\")\"
  "
