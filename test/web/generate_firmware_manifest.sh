#!/bin/bash
#
# ESPectre - Stage local website firmware
#
# Merge locally built C3, S3, and C5 factory images into the release web
# flasher catalog.
#
# Usage:
#   ./generate_firmware_manifest.sh
#   ./generate_firmware_manifest.sh --dry-run
#   ./generate_firmware_manifest.sh --replace
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PYTHON="${REPO_ROOT}/.venv/bin/python"
else
    PYTHON="python3"
fi

VERSION="$("${PYTHON}" "${REPO_ROOT}/.github/scripts/detect_git_version.py")"

exec "${PYTHON}" "${REPO_ROOT}/.github/scripts/stage_web_firmware.py" \
    --from-local-builds \
    --channel release \
    --version "${VERSION}" \
    --output-dir "${REPO_ROOT}/docs/web/artifacts/firmware/release" \
    --url-prefix /artifacts/firmware/release \
    --chip c3 \
    --chip s3 \
    --chip c5 \
    "$@"
