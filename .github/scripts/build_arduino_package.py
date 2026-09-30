#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - Arduino Library Builder

Build the ESPectre Arduino library: the sensing SDK sources and the `ESPectre.h`
entry header, laid out for Arduino CLI and the IDE's Add .ZIP Library.

Author: Francesco Pace <francesco.pace@gmail.com>
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
from pathlib import Path

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from build_sdk_package import resolve_source_date_epoch, stamp_sdk_version_header, write_zipfile

REPO_ROOT = Path(__file__).resolve().parents[2]
CPP_ROOT = REPO_ROOT / "src" / "cpp"
ARDUINO_ROOT = CPP_ROOT / "frontend" / "arduino"
SOURCES_CMAKE = CPP_ROOT / "espectre_sources.cmake"
LIBRARY_NAME = "ESPectre"
# Arduino compiles every source under src/, so only these groups ship. The
# optional groups need linker options or components the sketch does not own.
SOURCE_GROUPS = (
    "ESPECTRE_CORE_SOURCES",
    "ESPECTRE_RUNTIME_COMMON_SOURCES",
    "ESPECTRE_RUNTIME_ESP_IDF_TRAFFIC_SOURCES",
    "ESPECTRE_RUNTIME_ESP_IDF_PLATFORM_SOURCES",
)
SDK_FACADES = ("espectre_sdk.h", "espectre_core_sdk.h")
HEADER_SUFFIXES = {".h", ".hpp"}
LEGAL_FILES = ("LICENSE", "LICENSING.md", "THIRD_PARTY_NOTICES.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the ESPectre Arduino library.")
    parser.add_argument("--version", required=True, help="Library version, for example 3.2.0.")
    parser.add_argument("--output-dir", required=True, help="Directory where the library ZIP is written.")
    parser.add_argument(
        "--source-date-epoch",
        type=int,
        help="Reproducible archive timestamp; defaults to SOURCE_DATE_EPOCH or the checkout commit time.",
    )
    return parser.parse_args()


def cmake_source_groups(path: Path = SOURCES_CMAKE) -> dict[str, list[str]]:
    """Source groups from espectre_sources.cmake, relative to the SDK root."""
    groups: dict[str, list[str]] = {}
    for name, body in re.findall(r"set\((ESPECTRE_\w+_SOURCES)\s+(.*?)\)", path.read_text(encoding="utf-8"), re.DOTALL):
        entries: list[str] = []
        for token in re.findall(r"\$\{(ESPECTRE_\w+)\}(/[^\"\s]+)?", body):
            reference, suffix = token
            if suffix:
                entries.append(suffix.lstrip("/"))
            else:
                entries.extend(groups[reference])
        groups[name] = entries
    return groups


def collect_sdk_files() -> list[Path]:
    """SDK files the library ships, relative to the SDK root."""
    groups = cmake_source_groups()
    sources = {Path(source) for group in SOURCE_GROUPS for source in groups[group]}
    headers = {
        path.relative_to(CPP_ROOT)
        for root in (CPP_ROOT / "core", CPP_ROOT / "runtime")
        for path in root.rglob("*")
        if path.suffix in HEADER_SUFFIXES
    }
    return sorted({Path(facade) for facade in SDK_FACADES} | headers | sources)


def stamp_library_properties(path: Path, version: str) -> None:
    source, count = re.subn(r"(?m)^version=.*$", f"version={version}", path.read_text(encoding="utf-8"), count=1)
    if count != 1:
        raise ValueError(f"Unable to stamp the library version in {path}")
    path.write_text(source, encoding="utf-8")


def stage_library(destination: Path, version: str) -> None:
    destination.mkdir(parents=True)
    shutil.copy2(ARDUINO_ROOT / "library.properties", destination / "library.properties")
    stamp_library_properties(destination / "library.properties", version)
    shutil.copy2(ARDUINO_ROOT / "README.md", destination / "README.md")
    shutil.copytree(ARDUINO_ROOT / "examples", destination / "examples")
    shutil.copytree(ARDUINO_ROOT / "src", destination / "src")
    for name in LEGAL_FILES:
        shutil.copy2(REPO_ROOT / name, destination / name)
    for relative in collect_sdk_files():
        target = destination / "src" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(CPP_ROOT / relative, target)
    stamp_sdk_version_header(destination / "src" / "runtime" / "espectre_sdk_version.h", version)


def build_arduino_package(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"espectre-arduino-{args.version}.zip"
    epoch = resolve_source_date_epoch(args.source_date_epoch)
    with tempfile.TemporaryDirectory() as staging:
        library = Path(staging) / LIBRARY_NAME
        stage_library(library, args.version)
        write_zipfile(library, output, LIBRARY_NAME, epoch)
    return output


def main() -> int:
    print(build_arduino_package(parse_args()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
