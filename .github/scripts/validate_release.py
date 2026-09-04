#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Validate release metadata before any firmware build or publication."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from detect_git_version import detect_git_version

REPO_ROOT = Path(__file__).resolve().parents[2]
CHANGELOG = REPO_ROOT / "docs" / "CHANGELOG.md"
SEMVER_PATTERN = re.compile(
    r"^(?P<core>(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*))"
    r"(?:-(?P<prerelease>"
    r"(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*)"
    r"(?:\.(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*))*"
    r"))?$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an ESPectre release tag.")
    parser.add_argument("--tag", required=True)
    return parser.parse_args()


def changelog_header(tag: str) -> str:
    prefix = f"## [{tag}]"
    matches = [
        line
        for line in CHANGELOG.read_text(encoding="utf-8").splitlines()
        if line == prefix or line.startswith(f"{prefix} ")
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one CHANGELOG section for {tag}, found {len(matches)}")
    return matches[0]


def validate(tag: str) -> None:
    match = SEMVER_PATTERN.fullmatch(tag)
    if not match:
        raise ValueError(
            "Release tag must be semantic versioning without a leading v, "
            "for example 3.0.0 or 3.0.0-rc1"
        )
    described = detect_git_version(exact_match=True)
    if described != tag:
        raise ValueError(f"git describe is {described!r}, expected the release tag {tag!r}")
    header = changelog_header(tag)
    if "unreleased" in header.casefold() or "in progress" in header.casefold():
        raise ValueError(f"CHANGELOG section is not finalized: {header}")


def main() -> int:
    args = parse_args()
    validate(args.tag)
    print(f"Release metadata validated for {args.tag}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
