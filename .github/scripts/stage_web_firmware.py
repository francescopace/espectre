#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - Stage Web Firmware

Stage same-origin firmware assets for the web flasher.

Author: Francesco Pace <francesco.pace@gmail.com>
"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

from build_firmware_manifest import build_manifest
from build_firmware_compliance_bundle import COMPLIANCE_SUFFIXES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage same-origin firmware assets for the web flasher.")
    parser.add_argument("--firmware-dir", required=True, help="Directory containing built firmware assets")
    parser.add_argument("--output-dir", required=True, help="Directory where staged firmware assets should be written")
    parser.add_argument("--channel", choices=("release", "preview", "develop"), required=True, help="Release channel exposed to the web UI")
    parser.add_argument("--version", required=True, help="Human-readable version label")
    parser.add_argument("--release-tag", required=True, help="Release tag used for metadata")
    parser.add_argument("--url-prefix", required=True, help="Same-origin URL prefix used by the staged manifest")
    parser.add_argument("--commit", help="Optional source commit SHA for snapshot builds")
    return parser.parse_args()


def clean_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.iterdir():
        if path.is_file() and (
            path.suffix in (".bin", ".zip")
            or path.name.endswith("-sbom.spdx.json")
            or path.name.endswith("-THIRD_PARTY_NOTICES.txt")
            or path.name.startswith("firmware-manifest-")
        ):
            path.unlink()


def referenced_filenames(manifest: dict) -> set[str]:
    filenames: set[str] = set()
    for frontend in manifest["frontends"].values():
        for artifact in frontend["artifacts"]:
            if artifact["build_type"] == "factory":
                filenames.add(artifact["filename"])
    return filenames


def materialize_compliance_bundle(firmware_dir: Path) -> None:
    bundles = sorted(firmware_dir.glob("firmware-compliance-*.zip"))
    if not bundles:
        return
    if len(bundles) != 1:
        raise ValueError(f"Expected one firmware compliance bundle, found: {bundles}")

    with zipfile.ZipFile(bundles[0]) as archive:
        for info in archive.infolist():
            if info.is_dir() or not info.filename.endswith(COMPLIANCE_SUFFIXES):
                continue
            if Path(info.filename).name != info.filename:
                raise ValueError(f"Invalid firmware compliance bundle entry: {info.filename}")
            destination = firmware_dir / info.filename
            contents = archive.read(info)
            if destination.is_file() and destination.read_bytes() != contents:
                raise ValueError(f"Conflicting firmware compliance artifact: {info.filename}")
            destination.write_bytes(contents)


def stage_web_firmware(args: argparse.Namespace) -> Path:
    firmware_dir = Path(args.firmware_dir)
    output_dir = Path(args.output_dir)
    manifest_path = output_dir / f"firmware-manifest-{args.channel}.json"

    materialize_compliance_bundle(firmware_dir)
    clean_output_dir(output_dir)

    manifest = build_manifest(
        argparse.Namespace(
            firmware_dir=str(firmware_dir),
            output=str(manifest_path),
            channel=args.channel,
            version=args.version,
            release_tag=args.release_tag,
            commit=args.commit,
            url_prefix=args.url_prefix,
        )
    )

    for frontend in manifest["frontends"].values():
        frontend["artifacts"] = [
            artifact for artifact in frontend["artifacts"] if artifact["build_type"] == "factory"
        ]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    for filename in sorted(referenced_filenames(manifest)):
        shutil.copy2(firmware_dir / filename, output_dir / filename)
        firmware_stem = Path(filename).stem
        for suffix in ("-sbom.spdx.json", "-THIRD_PARTY_NOTICES.txt", "-third-party-licenses.zip"):
            companion = firmware_dir / f"{firmware_stem}{suffix}"
            if companion.is_file():
                shutil.copy2(companion, output_dir / companion.name)

    return manifest_path


def main() -> int:
    args = parse_args()
    stage_web_firmware(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
