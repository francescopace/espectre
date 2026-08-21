#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - Generate SDK API Reference

Stamp the Doxygen PROJECT_NUMBER from the same git-describe identity used by
SDK bundles, then generate the browsable reference. The committed Doxyfile is
left unchanged; packaging stamps the copy that ships in each bundle.

Usage, from the repository root:

    python3 .github/scripts/generate_sdk_api.py

Author: Francesco Pace <francesco.pace@gmail.com>
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from build_sdk_package import stamp_doxyfile_project_number
from detect_git_version import detect_git_version, parse_version_core

REPO_ROOT = Path(__file__).resolve().parents[2]
DOXYFILE = REPO_ROOT / "src" / "cpp" / "Doxyfile"
API_OUTPUT_DIR = REPO_ROOT / "docs" / "web" / "artifacts" / "sdk"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the ESPectre SDK API reference with the current SDK identity."
    )
    parser.add_argument(
        "--version",
        help="Override git describe. Must start with numeric MAJOR.MINOR.PATCH.",
    )
    return parser.parse_args()


def generate_sdk_api(version: str | None = None) -> str:
    sdk_version = version or detect_git_version()
    parse_version_core(sdk_version)
    API_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="espectre-doxy-") as tmp_dir:
        stamped = Path(tmp_dir) / "Doxyfile"
        shutil.copy2(DOXYFILE, stamped)
        stamp_doxyfile_project_number(stamped, sdk_version)
        try:
            subprocess.run(["doxygen", str(stamped)], cwd=REPO_ROOT, check=True)
        except FileNotFoundError as error:
            raise FileNotFoundError("doxygen is not installed or not on PATH") from error

    return sdk_version


def main() -> int:
    sdk_version = generate_sdk_api(parse_args().version)
    print(f"Generated SDK API reference for {sdk_version}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
