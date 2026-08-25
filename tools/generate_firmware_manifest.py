#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - Generate Firmware Manifest

Collect locally built Native, Matter, and ESPHome factory images, merge the
ESP-IDF flash layouts into web-flasher binaries, and write
``docs/web/artifacts/firmware/<channel>/firmware-manifest-<channel>.json``.

Author: Francesco Pace <francesco.pace@gmail.com>
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
GITHUB_SCRIPTS = REPO_ROOT / ".github" / "scripts"
if str(GITHUB_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(GITHUB_SCRIPTS))

from build_firmware_manifest import CHIP_METADATA, FRONTEND_CHIPS  # noqa: E402
from detect_git_version import detect_git_version  # noqa: E402
from stage_web_firmware import stage_web_firmware  # noqa: E402

IDF_APP_BIN = {
    "native": "espectre-native.bin",
    "matter": "espectre-matter.bin",
}
IDF_APP_DIRS = {
    "native": REPO_ROOT / "src" / "cpp" / "frontend" / "native" / "app",
    "matter": REPO_ROOT / "src" / "cpp" / "frontend" / "matter" / "app",
}
ESPHOME_BUILD_ROOT = (
    REPO_ROOT / "src" / "cpp" / "frontend" / "esphome" / "examples" / ".esphome" / "build"
)
CHIP_ALIASES = {
    "esp32": "esp32",
    "s2": "esp32s2",
    "esp32s2": "esp32s2",
    "s3": "esp32s3",
    "esp32s3": "esp32s3",
    "c3": "esp32c3",
    "esp32c3": "esp32c3",
    "c5": "esp32c5",
    "esp32c5": "esp32c5",
    "c6": "esp32c6",
    "esp32c6": "esp32c6",
}
CANONICAL_BUILD_DIR = {chip: f"build-{chip}" for chip in CHIP_METADATA}


@dataclass(frozen=True)
class LocalImage:
    frontend: str
    chip: str
    source: Path
    kind: str
    project_version: str | None = None


def parse_chip(value: str) -> str:
    """Normalize a CLI chip name to an IDF target."""
    key = value.strip().lower().replace("-", "")
    chip = CHIP_ALIASES.get(key)
    if chip is None:
        choices = ", ".join(sorted(CHIP_METADATA))
        raise argparse.ArgumentTypeError(f"Unknown chip {value!r}. Choose from: {choices}")
    return chip


def published_filename(channel: str, frontend: str, version: str, chip: str) -> str:
    """Return the web-catalog filename for one factory image."""
    if channel == "release":
        prefix = f"espectre-{frontend}-{version}"
    elif channel == "preview":
        prefix = f"espectre-{frontend}-preview"
    else:
        prefix = f"espectre-{frontend}-develop"
    return f"{prefix}-{chip}.bin"


def read_json(path: Path) -> dict:
    """Load a JSON object from disk."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def project_description(build_dir: Path) -> dict | None:
    """Return ``project_description.json`` when present."""
    path = build_dir / "project_description.json"
    if not path.is_file():
        return None
    try:
        return read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def discover_idf_images(frontends: Sequence[str], chips: Sequence[str] | None) -> list[LocalImage]:
    """Find canonical local Native and Matter app images."""
    images: list[LocalImage] = []
    selected = set(chips) if chips else None
    for frontend in frontends:
        app_dir = IDF_APP_DIRS.get(frontend)
        if app_dir is None:
            continue
        for chip in sorted(FRONTEND_CHIPS[frontend]):
            if selected is not None and chip not in selected:
                continue
            build_dir = app_dir / CANONICAL_BUILD_DIR[chip]
            app_bin = build_dir / IDF_APP_BIN[frontend]
            flash_args = build_dir / "flash_args"
            if not app_bin.is_file() or not flash_args.is_file():
                continue
            description = project_description(build_dir) or {}
            target = description.get("target")
            if target not in (None, chip):
                print(
                    f"Skipping {frontend} {chip}: build directory target is {target!r}.",
                    file=sys.stderr,
                )
                continue
            version = description.get("project_version")
            images.append(
                LocalImage(
                    frontend=frontend,
                    chip=chip,
                    source=build_dir,
                    kind="idf-merge",
                    project_version=version if isinstance(version, str) and version else None,
                )
            )
    return images


def discover_esphome_images(chips: Sequence[str] | None) -> list[LocalImage]:
    """Find local ESPHome factory images, keeping the newest file per chip."""
    if not ESPHOME_BUILD_ROOT.is_dir():
        return []
    selected = set(chips) if chips else None
    newest: dict[str, LocalImage] = {}
    for factory in ESPHOME_BUILD_ROOT.glob("*/build/firmware.factory.bin"):
        description = project_description(factory.parent) or {}
        chip = description.get("target")
        if chip not in FRONTEND_CHIPS["esphome"]:
            continue
        if selected is not None and chip not in selected:
            continue
        current = newest.get(chip)
        if current is not None and current.source.stat().st_mtime >= factory.stat().st_mtime:
            continue
        newest[chip] = LocalImage(
            frontend="esphome",
            chip=chip,
            source=factory,
            kind="esphome-factory",
        )
    return [newest[chip] for chip in sorted(newest)]


def resolve_version(images: Sequence[LocalImage], requested: str | None) -> str:
    """Prefer an explicit version, then a shared IDF project version, then git describe."""
    if requested:
        return requested
    idf_versions = sorted(
        {
            image.project_version
            for image in images
            if image.kind == "idf-merge" and image.project_version
        }
    )
    if len(idf_versions) == 1:
        return idf_versions[0]
    if len(idf_versions) > 1:
        raise ValueError(
            "Local IDF builds report mixed project versions "
            f"{', '.join(idf_versions)}. Pass --version to choose one."
        )
    return detect_git_version(REPO_ROOT)


def existing_manifest(output_dir: Path, channel: str) -> dict | None:
    """Return the current channel manifest when it is valid JSON."""
    path = output_dir / f"firmware-manifest-{channel}.json"
    if not path.is_file():
        return None
    try:
        return read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def merge_idf_factory(image: LocalImage, output: Path, *, verbose: bool) -> None:
    """Merge one ESP-IDF flash layout into a 4 MB factory image."""
    command = [
        sys.executable,
        "-m",
        "esptool",
        "--chip",
        image.chip,
        "merge-bin",
        "--pad-to-size",
        "4MB",
        "-o",
        str(output),
        "@flash_args",
    ]
    result = subprocess.run(
        command,
        cwd=image.source,
        check=False,
        capture_output=not verbose,
        text=True,
    )
    if result.returncode == 0:
        return
    detail = (result.stderr or result.stdout or f"exit {result.returncode}").strip()
    raise RuntimeError(f"esptool merge-bin failed for {image.frontend} {image.chip}: {detail}")


def prepare_factory_images(
    images: Sequence[LocalImage],
    stage_dir: Path,
    *,
    channel: str,
    version: str,
    verbose: bool,
) -> dict[str, LocalImage]:
    """Write merged or copied factory images into the staging directory."""
    staged: dict[str, LocalImage] = {}
    for image in images:
        filename = published_filename(channel, image.frontend, version, image.chip)
        destination = stage_dir / filename
        print(f"Staging {filename} from {image.source}")
        if image.kind == "idf-merge":
            merge_idf_factory(image, destination, verbose=verbose)
        else:
            shutil.copy2(image.source, destination)
        staged[filename] = image
    return staged


def keep_existing_images(
    output_dir: Path,
    stage_dir: Path,
    *,
    staged_names: set[str],
) -> list[str]:
    """Copy previously staged factory images that this run did not rebuild."""
    kept: list[str] = []
    if not output_dir.is_dir():
        return kept
    for path in sorted(output_dir.glob("espectre-*.bin")):
        if path.name in staged_names:
            continue
        shutil.copy2(path, stage_dir / path.name)
        kept.append(path.name)
        print(f"Keeping {path.name}")
    return kept


def summarize(manifest: dict, staged: dict[str, LocalImage], kept: Sequence[str]) -> None:
    """Print the restaged catalog."""
    print(f"Version: {manifest['version']}")
    print(f"Channel: {manifest['channel']}")
    print(f"Release tag: {manifest['release_tag']}")
    for frontend, metadata in manifest["frontends"].items():
        for artifact in metadata["artifacts"]:
            filename = artifact["filename"]
            origin = "local build" if filename in staged else "kept existing"
            print(f"  {frontend:8} {artifact['chip']:8} {origin}")
    if not any(metadata["artifacts"] for metadata in manifest["frontends"].values()):
        print("  (no factory images)")
    extra = [name for name in kept if name not in staged]
    if extra:
        print(f"Kept {len(extra)} previously staged factory image(s).")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the local firmware-manifest arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate the website firmware manifest from locally built "
            "Native, Matter, and ESPHome factory images."
        ),
    )
    parser.add_argument(
        "--channel",
        choices=("release", "preview", "develop"),
        default="release",
        help="Website firmware channel to restage. Default: release.",
    )
    parser.add_argument(
        "--chip",
        action="append",
        type=parse_chip,
        dest="chips",
        metavar="CHIP",
        help="Limit staging to this chip; repeat as needed. Default: every discovered chip.",
    )
    parser.add_argument(
        "--frontend",
        action="append",
        choices=("esphome", "matter", "native"),
        dest="frontends",
        metavar="FRONTEND",
        help="Limit staging to this frontend; repeat as needed. Default: all three.",
    )
    parser.add_argument("--version", help="Catalog version label. Default: the shared local IDF project version, otherwise git describe.")
    parser.add_argument("--release-tag", help="Manifest release tag. Default: the current channel tag, otherwise local.")
    parser.add_argument("--commit", help="Optional source commit recorded in the manifest.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override the staged firmware directory. Default: docs/web/artifacts/firmware/<channel>.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace the channel catalog instead of keeping factory images that were not rebuilt.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the local images that would be staged without writing the catalog.",
    )
    parser.add_argument("--verbose", action="store_true", help="Show esptool merge-bin output.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Discover local factory images and write the website firmware manifest."""
    args = parse_args(argv)
    frontends = tuple(args.frontends or ("esphome", "matter", "native"))
    chips = tuple(dict.fromkeys(args.chips)) if args.chips else None
    images = []
    if "native" in frontends or "matter" in frontends:
        images.extend(discover_idf_images(frontends, chips))
    if "esphome" in frontends:
        images.extend(discover_esphome_images(chips))
    images = sorted(images, key=lambda image: (image.frontend, image.chip))

    if not images:
        print("No local factory images found for the selected frontends and chips.", file=sys.stderr)
        return 1

    version = resolve_version(images, args.version)
    output_dir = args.output_dir or (REPO_ROOT / "docs" / "web" / "artifacts" / "firmware" / args.channel)
    manifest = existing_manifest(output_dir, args.channel)
    release_tag = args.release_tag or (manifest or {}).get("release_tag") or "local"
    commit = args.commit if args.commit is not None else (manifest or {}).get("commit")

    print(f"Discovered {len(images)} local factory image(s) for {version}:")
    for image in images:
        print(f"  {image.frontend:8} {image.chip:8} {image.source}")
    if args.dry_run:
        return 0

    with tempfile.TemporaryDirectory(prefix="espectre-web-firmware-") as temp_dir:
        stage_dir = Path(temp_dir)
        staged = prepare_factory_images(
            images,
            stage_dir,
            channel=args.channel,
            version=version,
            verbose=args.verbose,
        )
        kept: list[str] = []
        if not args.replace:
            kept = keep_existing_images(output_dir, stage_dir, staged_names=set(staged))
        manifest_path = stage_web_firmware(
            argparse.Namespace(
                firmware_dir=str(stage_dir),
                output_dir=str(output_dir),
                channel=args.channel,
                version=version,
                release_tag=release_tag,
                url_prefix=f"/artifacts/firmware/{args.channel}",
                commit=commit,
            )
        )

    written = read_json(manifest_path)
    print(f"Wrote {manifest_path}")
    summarize(written, staged, kept)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
