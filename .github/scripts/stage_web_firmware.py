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
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from build_firmware_compliance_bundle import COMPLIANCE_SUFFIXES
from build_firmware_manifest import (
    CHIP_METADATA,
    FRONTEND_CHIPS,
    build_manifest,
    published_factory_filename,
)
from detect_git_version import detect_git_version

REPO_ROOT = Path(__file__).resolve().parents[2]
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
CI_REQUIRED_FLAGS = (
    ("--firmware-dir", "firmware_dir"),
    ("--output-dir", "output_dir"),
    ("--channel", "channel"),
    ("--version", "version"),
    ("--release-tag", "release_tag"),
    ("--url-prefix", "url_prefix"),
)
LOCAL_ONLY_FLAGS = ("--chip", "--frontend", "--replace", "--dry-run")


@dataclass(frozen=True)
class LocalImage:
    frontend: str
    chip: str
    source: Path
    kind: str
    project_version: str | None = None


def parse_chip(value: str) -> str:
    key = value.strip().lower().replace("-", "")
    chip = CHIP_ALIASES.get(key)
    if chip is None:
        choices = ", ".join(sorted(CHIP_METADATA))
        raise argparse.ArgumentTypeError(f"Unknown chip {value!r}. Choose from: {choices}")
    return chip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage same-origin firmware assets for the web flasher.")
    parser.add_argument("--firmware-dir", help="Directory containing built firmware assets")
    parser.add_argument("--output-dir", help="Directory where staged firmware assets should be written")
    parser.add_argument(
        "--channel",
        choices=("release", "preview", "develop"),
        help="Release channel exposed to the web UI",
    )
    parser.add_argument("--version", help="Human-readable version label")
    parser.add_argument("--release-tag", help="Release tag used for metadata")
    parser.add_argument("--url-prefix", help="Same-origin URL prefix used by the staged manifest")
    parser.add_argument("--commit", help="Optional source commit SHA for snapshot builds")
    parser.add_argument(
        "--from-local-builds",
        action="store_true",
        help="Discover canonical local Native, Matter, and ESPHome factory images instead of using --firmware-dir",
    )
    parser.add_argument(
        "--chip",
        action="append",
        type=parse_chip,
        dest="chips",
        metavar="CHIP",
        help="With --from-local-builds, limit staging to this chip; repeat as needed",
    )
    parser.add_argument(
        "--frontend",
        action="append",
        choices=("esphome", "matter", "native"),
        dest="frontends",
        metavar="FRONTEND",
        help="With --from-local-builds, limit staging to this frontend; repeat as needed",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="With --from-local-builds, replace the channel catalog instead of keeping factory images that were not rebuilt",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --from-local-builds, list local images without writing the catalog",
    )
    parser.add_argument("--verbose", action="store_true", help="Show esptool merge-bin output")
    args = parser.parse_args()
    if args.from_local_builds:
        args.channel = args.channel or "release"
        return args
    missing = [flag for flag, attr in CI_REQUIRED_FLAGS if not getattr(args, attr)]
    if missing:
        parser.error("the following arguments are required: " + ", ".join(missing))
    if args.chips or args.frontends or args.replace or args.dry_run:
        parser.error("the following arguments require --from-local-builds: " + ", ".join(LOCAL_ONLY_FLAGS))
    return args


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
    with tempfile.TemporaryDirectory(prefix="espectre-web-manifest-") as temp_dir:
        staged_manifest_path = Path(temp_dir) / manifest_path.name
        manifest = build_manifest(
            argparse.Namespace(
                firmware_dir=str(firmware_dir),
                output=str(staged_manifest_path),
                channel=args.channel,
                version=args.version,
                release_tag=args.release_tag,
                commit=args.commit,
                url_prefix=args.url_prefix,
            )
        )

        for frontend in manifest["frontends"].values():
            frontend["artifacts"] = [
                artifact
                for artifact in frontend["artifacts"]
                if artifact["build_type"] == "factory"
            ]
        staged_manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        clean_output_dir(output_dir)
        for filename in sorted(referenced_filenames(manifest)):
            shutil.copy2(firmware_dir / filename, output_dir / filename)
            firmware_stem = Path(filename).stem
            for suffix in (
                "-sbom.spdx.json",
                "-THIRD_PARTY_NOTICES.txt",
                "-third-party-licenses.zip",
            ):
                companion = firmware_dir / f"{firmware_stem}{suffix}"
                if companion.is_file():
                    shutil.copy2(companion, output_dir / companion.name)
        shutil.copy2(staged_manifest_path, manifest_path)

    return manifest_path


def read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def project_description(build_dir: Path) -> dict | None:
    path = build_dir / "project_description.json"
    if not path.is_file():
        return None
    try:
        return read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def discover_idf_images(frontends: Sequence[str], chips: Sequence[str] | None) -> list[LocalImage]:
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
    if not ESPHOME_BUILD_ROOT.is_dir():
        return []
    selected = set(chips) if chips else None
    newest: dict[str, LocalImage] = {}
    for factory in ESPHOME_BUILD_ROOT.rglob("firmware.factory.bin"):
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


def resolve_local_version(images: Sequence[LocalImage], requested: str | None) -> str:
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
    path = output_dir / f"firmware-manifest-{channel}.json"
    if not path.is_file():
        return None
    try:
        return read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def resolve_local_release_tag(
    channel: str,
    version: str,
    requested: str | None,
    manifest: dict | None,
) -> str:
    if requested:
        return requested
    if channel == "release":
        return version
    return (manifest or {}).get("release_tag") or "local"


def merge_idf_factory(image: LocalImage, output: Path, *, verbose: bool) -> None:
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
    staged: dict[str, LocalImage] = {}
    for image in images:
        filename = published_factory_filename(channel, image.frontend, version, image.chip)
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


def stage_from_local_builds(args: argparse.Namespace) -> int:
    frontends = tuple(args.frontends or ("esphome", "matter", "native"))
    chips = tuple(dict.fromkeys(args.chips)) if args.chips else None
    images: list[LocalImage] = []
    if "native" in frontends or "matter" in frontends:
        images.extend(discover_idf_images(frontends, chips))
    if "esphome" in frontends:
        images.extend(discover_esphome_images(chips))
    images = sorted(images, key=lambda image: (image.frontend, image.chip))

    if not images:
        print("No local factory images found for the selected frontends and chips.", file=sys.stderr)
        return 1

    version = resolve_local_version(images, args.version)
    output_dir = Path(args.output_dir) if args.output_dir else (
        REPO_ROOT / "docs" / "web" / "artifacts" / "firmware" / args.channel
    )
    manifest = existing_manifest(output_dir, args.channel)
    release_tag = resolve_local_release_tag(args.channel, version, args.release_tag, manifest)
    commit = args.commit if args.commit is not None else (manifest or {}).get("commit")
    url_prefix = args.url_prefix or f"/artifacts/firmware/{args.channel}"

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
                url_prefix=url_prefix,
                commit=commit,
            )
        )

    written = read_json(manifest_path)
    print(f"Wrote {manifest_path}")
    summarize(written, staged, kept)
    return 0


def main() -> int:
    args = parse_args()
    if args.from_local_builds:
        return stage_from_local_builds(args)
    stage_web_firmware(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
