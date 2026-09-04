#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - Website Asset Versions

First-party CSS, JS, and brand assets use a content hash as the cache-busting
`?v=` query string. Unchanged files keep their hash; an asset edit changes only
that file's query string.

Stamp committed HTML after asset edits:

    python3 .github/scripts/web_asset_versions.py

CI and local tests use `--check-current` so a stale hash fails closed.

Author: Francesco Pace <francesco.pace@gmail.com>
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = REPO_ROOT / "docs" / "web"
HASH_LENGTH = 12
STAMPED_HTML = ("index.html", "404.html")
ASSET_URL_RE = re.compile(
    r'(?P<prefix>(?:href|src|data-script-src)=")'
    r'(?P<path>(?:/assets/(?:css|js)/[^"?]+|/assets/images/brand/espectre-logo\.svg))'
    r'(?:(?P<query>\?v=[^"]*))?'
    r'(?P<suffix>")'
)


def asset_version(relative: str) -> str:
    path = WEB_ROOT / relative.lstrip("/")
    return hashlib.sha256(path.read_bytes()).hexdigest()[:HASH_LENGTH]


def _web_path(url_path: str) -> str:
    return url_path.lstrip("/")


def stamp_text(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        url_path = match.group("path")
        return f'{match.group("prefix")}{url_path}?v={asset_version(_web_path(url_path))}{match.group("suffix")}'

    return ASSET_URL_RE.sub(replace, text)


def stamped_html_paths() -> tuple[Path, ...]:
    return tuple(WEB_ROOT / name for name in STAMPED_HTML)


def stamp() -> list[Path]:
    written: list[Path] = []
    for path in stamped_html_paths():
        original = path.read_text(encoding="utf-8")
        updated = stamp_text(original)
        if updated == original:
            continue
        path.write_text(updated, encoding="utf-8")
        written.append(path)
    return written


def check_current() -> list[str]:
    mismatches: list[str] = []
    for path in stamped_html_paths():
        original = path.read_text(encoding="utf-8")
        updated = stamp_text(original)
        if updated != original:
            mismatches.append(str(path.relative_to(REPO_ROOT)))
    return mismatches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stamp first-party website cache-busting hashes from file contents."
    )
    parser.add_argument(
        "--check-current",
        action="store_true",
        help="Exit non-zero when committed HTML hashes do not match file contents.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.check_current:
        mismatches = check_current()
        if mismatches:
            joined = ", ".join(mismatches)
            print(
                f"error: stale website asset hashes in {joined}; "
                "run python3 .github/scripts/web_asset_versions.py",
                file=sys.stderr,
            )
            return 1
        return 0

    written = stamp()
    if not written:
        print("website asset hashes are current")
        return 0
    for path in written:
        print(f"wrote {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
