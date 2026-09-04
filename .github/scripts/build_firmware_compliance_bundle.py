#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Build the consolidated firmware compliance archive published with releases."""

from __future__ import annotations

import argparse
import stat
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
COMPLIANCE_SUFFIXES = (
    "-sbom.spdx.json",
    "-THIRD_PARTY_NOTICES.txt",
    "-third-party-licenses.zip",
)
TOP_LEVEL_LEGAL_PATHS = (
    REPO_ROOT / "LICENSE",
    REPO_ROOT / "LICENSING.md",
    REPO_ROOT / "THIRD_PARTY_NOTICES.md",
)
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a firmware compliance release bundle.")
    parser.add_argument("--firmware-dir", required=True, help="Directory containing firmware compliance files")
    parser.add_argument("--output-dir", required=True, help="Directory where the bundle is written")
    parser.add_argument(
        "--channel",
        choices=("release", "preview", "develop"),
        required=True,
        help="Release channel used to name the bundle",
    )
    parser.add_argument("--version", required=True, help="Release version used for the release-channel filename")
    return parser.parse_args()


def bundle_filename(channel: str, version: str) -> str:
    suffix = version if channel == "release" else channel
    return f"firmware-compliance-{suffix}.zip"


def compliance_paths(firmware_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in firmware_dir.iterdir()
        if path.is_file() and path.name.endswith(COMPLIANCE_SUFFIXES)
    )


def write_archive_entry(archive: zipfile.ZipFile, path: Path) -> None:
    info = zipfile.ZipInfo(path.name, date_time=ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def build_bundle(
    firmware_dir: Path,
    output_path: Path,
    *,
    legal_paths: tuple[Path, ...] = TOP_LEVEL_LEGAL_PATHS,
) -> Path:
    compliance = compliance_paths(firmware_dir)
    if not compliance:
        raise ValueError(f"No firmware compliance files found in {firmware_dir}")

    missing_legal = [str(path) for path in legal_paths if not path.is_file()]
    if missing_legal:
        raise FileNotFoundError(f"Missing top-level legal files: {missing_legal}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w") as archive:
        for path in sorted((*legal_paths, *compliance), key=lambda candidate: candidate.name):
            write_archive_entry(archive, path)
    return output_path


def main() -> int:
    args = parse_args()
    output_path = Path(args.output_dir) / bundle_filename(args.channel, args.version)
    build_bundle(Path(args.firmware_dir), output_path)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
