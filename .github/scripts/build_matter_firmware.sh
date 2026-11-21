#!/usr/bin/env bash
set -euo pipefail

: "${MATTER_TARGET:?MATTER_TARGET is required}"
: "${MATTER_OUTPUT:?MATTER_OUTPUT is required}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="build-container-${MATTER_TARGET}"
DOCKER_IMAGE="${MATTER_DOCKER_IMAGE:-espressif/idf:v6.1.1@sha256:2b4f966c37eaedc19542b0637a5f850b2eff9478d8e858192aa0073d92c2dc91}"
MATTER_HOME="${REPO_ROOT}/.cache/build/matter-home"
MATTER_ROOT_MANAGED_COMPONENTS="${MATTER_HOME}/root_managed_components"
MATTER_CCACHE="${MATTER_HOME}/ccache-${MATTER_TARGET}"
OUTPUT_DIR="$(dirname "${MATTER_OUTPUT}")"
MATTER_OUTPUT_IN_WORK="/work/${MATTER_OUTPUT#"${REPO_ROOT}"/}"
MATTER_SDKCONFIG_DEFAULTS="${MATTER_SDKCONFIG_DEFAULTS:-}"
ESPECTRE_GIT_VERSION="${ESPECTRE_GIT_VERSION:-$(python3 "${REPO_ROOT}/.github/scripts/detect_git_version.py")}"

mkdir -p "${MATTER_HOME}" "${MATTER_ROOT_MANAGED_COMPONENTS}" "${MATTER_CCACHE}" "${OUTPUT_DIR}"

docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HOME="/work/.cache/build/matter-home" \
  -e ESPECTRE_GIT_VERSION="${ESPECTRE_GIT_VERSION}" \
  -e IDF_CCACHE_ENABLE=1 \
  -e CCACHE_DIR="/work/.cache/build/matter-home/ccache-${MATTER_TARGET}" \
  -e CCACHE_MAXSIZE=750M \
  -e SDKCONFIG_DEFAULTS="${MATTER_SDKCONFIG_DEFAULTS}" \
  -e MATTER_OUTPUT="${MATTER_OUTPUT_IN_WORK}" \
  -v "${MATTER_ROOT_MANAGED_COMPONENTS}:/opt/esp/root_managed_components" \
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
    # Run the host CLI and SBOM tools in their own environment. IDF subprocesses
    # keep the image's interpreter, PATH, and Espressif-constrained packages.
    TOOLS_ENV=\"\${HOME}/cli-venv\"
    REQUIREMENTS_HASH=\"\$(cat /work/requirements.txt /work/.github/scripts/build_matter_firmware.sh | sha256sum | cut -d ' ' -f 1)\"
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
    \"\${TOOLS_ENV}/bin/python\" /work/espectre matter build --chip \"\${MATTER_CHIP}\" --backend local
    cd /work/src/cpp/frontend/matter/app/${BUILD_DIR}
    if python -m esptool merge-bin -h >/dev/null 2>&1; then
      python -m esptool --chip ${MATTER_TARGET} merge-bin --pad-to-size 4MB -o \"\${MATTER_OUTPUT}\" @flash_args
    else
      python -m esptool --chip ${MATTER_TARGET} merge_bin --fill-flash-size 4MB -o \"\${MATTER_OUTPUT}\" @flash_args
    fi
    \"\${TOOLS_ENV}/bin/python\" /work/.github/scripts/build_firmware_compliance.py \
      --frontend matter \
      --project-description /work/src/cpp/frontend/matter/app/${BUILD_DIR}/project_description.json \
      --firmware \"\${MATTER_OUTPUT}\" \
      --output-dir \"\$(dirname \"\${MATTER_OUTPUT}\")\"
  "
