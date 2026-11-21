#!/usr/bin/env bash
set -euo pipefail

: "${NATIVE_TARGET:?NATIVE_TARGET is required}"
: "${NATIVE_OUTPUT:?NATIVE_OUTPUT is required}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="${NATIVE_BUILD_DIR:-build-container-${NATIVE_TARGET}}"
DOCKER_IMAGE="${NATIVE_DOCKER_IMAGE:-espressif/idf:v6.1.1@sha256:2b4f966c37eaedc19542b0637a5f850b2eff9478d8e858192aa0073d92c2dc91}"
OUTPUT_DIR="$(dirname "${NATIVE_OUTPUT}")"
NATIVE_OUTPUT_IN_WORK="/work/${NATIVE_OUTPUT#"${REPO_ROOT}"/}"
NATIVE_OTA_OUTPUT_IN_WORK=""
NATIVE_SDKCONFIG_DEFAULTS="${NATIVE_SDKCONFIG_DEFAULTS:-}"
NATIVE_OTA_CHANNEL="${NATIVE_OTA_CHANNEL:-release}"
ESPECTRE_GIT_VERSION="${ESPECTRE_GIT_VERSION:-$(python3 "${REPO_ROOT}/.github/scripts/detect_git_version.py")}"
NATIVE_HOME="${REPO_ROOT}/.cache/build/native-home"
NATIVE_ROOT_MANAGED_COMPONENTS="${NATIVE_HOME}/root_managed_components"

if [ -n "${NATIVE_OTA_OUTPUT:-}" ]; then
  NATIVE_OTA_OUTPUT_IN_WORK="/work/${NATIVE_OTA_OUTPUT#"${REPO_ROOT}"/}"
fi

mkdir -p "${NATIVE_HOME}" "${NATIVE_ROOT_MANAGED_COMPONENTS}" "${OUTPUT_DIR}"

docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HOME="/work/.cache/build/native-home" \
  -e ESPECTRE_GIT_VERSION="${ESPECTRE_GIT_VERSION}" \
  -e SDKCONFIG_DEFAULTS="${NATIVE_SDKCONFIG_DEFAULTS}" \
  -e NATIVE_OTA_CHANNEL="${NATIVE_OTA_CHANNEL}" \
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
      esp32s2) NATIVE_CHIP=s2 ;;
      *) echo \"Unsupported native target: ${NATIVE_TARGET}\" >&2; exit 1 ;;
    esac
    # Run the host CLI and SBOM tools in their own environment. IDF subprocesses
    # keep the image's interpreter, PATH, and Espressif-constrained packages.
    TOOLS_ENV=\"\${HOME}/cli-venv\"
    REQUIREMENTS_HASH=\"\$(cat /work/requirements.txt /work/.github/scripts/build_native_firmware.sh | sha256sum | cut -d ' ' -f 1)\"
    REQUIREMENTS_MARKER=\"\${HOME}/.espectre-requirements-\${REQUIREMENTS_HASH}\"
    if [ ! -f \"\${REQUIREMENTS_MARKER}\" ]; then
      python -m venv --clear \"\${TOOLS_ENV}\"
      awk '/^(colorama|python-dotenv|paho-mqtt|prompt_toolkit|pyserial|PyYAML|license-expression|esp-idf-sbom)==/' /work/requirements.txt > \"\${TOOLS_ENV}/requirements.txt\"
      \"\${TOOLS_ENV}/bin/python\" -m pip install -r \"\${TOOLS_ENV}/requirements.txt\"
      python \"\${IDF_PATH}/tools/idf_tools.py\" check-python-dependencies
      rm -f \"\${HOME}\"/.espectre-requirements-*
      touch \"\${REQUIREMENTS_MARKER}\"
    fi
    export ESPECTRE_SKIP_VENV_REEXEC=1
    if [ -n \"\${SDKCONFIG_DEFAULTS:-}\" ]; then
      export SDKCONFIG_DEFAULTS
    fi
    export ESPECTRE_IDF_BUILD_DIR=${BUILD_DIR}
    cd /work
    \"\${TOOLS_ENV}/bin/python\" /work/espectre native build --chip \"\${NATIVE_CHIP}\" --backend local --clean
    cd /work/src/cpp/frontend/native/app/${BUILD_DIR}
    if python -m esptool merge-bin -h >/dev/null 2>&1; then
      python -m esptool --chip ${NATIVE_TARGET} merge-bin --pad-to-size 4MB -o \"\${NATIVE_OUTPUT}\" @flash_args
    else
      python -m esptool --chip ${NATIVE_TARGET} merge_bin --fill-flash-size 4MB -o \"\${NATIVE_OUTPUT}\" @flash_args
    fi
    if [ -n \"\${NATIVE_OTA_OUTPUT:-}\" ]; then
      cp espectre-native.bin \"\${NATIVE_OTA_OUTPUT}\"
    fi
    \"\${TOOLS_ENV}/bin/python\" /work/.github/scripts/build_firmware_compliance.py \
      --frontend native \
      --project-description /work/src/cpp/frontend/native/app/${BUILD_DIR}/project_description.json \
      --firmware \"\${NATIVE_OUTPUT}\" \
      --output-dir \"\$(dirname \"\${NATIVE_OUTPUT}\")\"
    if [ -n \"\${NATIVE_OTA_OUTPUT:-}\" ]; then
      \"\${TOOLS_ENV}/bin/python\" /work/.github/scripts/build_firmware_compliance.py \
        --frontend native \
        --project-description /work/src/cpp/frontend/native/app/${BUILD_DIR}/project_description.json \
        --firmware \"\${NATIVE_OTA_OUTPUT}\" \
        --output-dir \"\$(dirname \"\${NATIVE_OTA_OUTPUT}\")\"
    fi
  "
