#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Verify the complete generated GitHub Pages tree before upload."""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from detect_git_version import detect_git_version


REPO_ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = REPO_ROOT / "docs" / "web"
EXPECTED_CHIPS = {"esp32", "esp32c3", "esp32c5", "esp32c6", "esp32s3"}
EXPECTED_FRONTENDS = {"esphome", "matter", "native"}
SITEMAP_NAMESPACE = "http://www.sitemaps.org/schemas/sitemap/0.9"
SITE_HOST = "espectre.dev"
SPA_ROUTE_NAME_RE = re.compile(r"\{\s*name:\s*'([^']+)'")
SPA_PAGE_ROUTE_RE = re.compile(r'<main\b[^>]*\bdata-page="([^"]+)"')
SPA_STATIC_PATH_RE = re.compile(r"staticPath:\s*'(/[^']+)'")
EXPECTED_SITEMAP_PATHS = {
    "/",
    "/guides/",
    "/guides/hardware/",
    "/guides/setup/",
    "/guides/home-assistant/",
    "/guides/placement/",
    "/guides/detection/",
    "/guides/detectors/",
    "/guides/micropython/",
    "/guides/future-wifi-sensing/",
    "/sdk/",
    "/sdk/detectors/",
    "/sdk/api/",
    "/sdk/examples/",
    "/sdk/architecture/",
    "/artifacts/sdk/api/",
    "/artifacts/sdk/release/",
    "/artifacts/sdk/preview/",
    "/artifacts/sdk/develop/",
    "/media/",
    "/roadmap/",
    "/privacy/",
    "/terms/",
    "/legal/",
    "/security/",
    "/licensing/",
    "/contact/",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the generated ESPectre Pages tree.")
    parser.add_argument("--require-preview", action="store_true")
    parser.add_argument("--require-release", action="store_true")
    parser.add_argument("--require-develop", action="store_true")
    return parser.parse_args()


def require_file(relative_path: str) -> Path:
    path = (WEB_ROOT / relative_path).resolve()
    try:
        path.relative_to(WEB_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"Website path escapes the generated tree: {relative_path}") from error
    if not path.is_file():
        raise FileNotFoundError(f"Missing generated website file: {relative_path}")
    return path


def registered_spa_routes() -> list[str]:
    registry = require_file("assets/js/route-registry.js").read_text(encoding="utf-8")
    routes = SPA_ROUTE_NAME_RE.findall(registry)
    if not routes:
        raise ValueError("Route registry contains no SPA routes")
    return routes


def registered_static_paths() -> list[str]:
    registry = require_file("assets/js/route-registry.js").read_text(encoding="utf-8")
    paths = SPA_STATIC_PATH_RE.findall(registry)
    if not paths:
        raise ValueError("Route registry contains no static paths")
    return paths


def verify_spa_routes() -> None:
    index = require_file("index.html").read_text(encoding="utf-8")
    expected = registered_spa_routes()
    found = SPA_PAGE_ROUTE_RE.findall(index)
    missing = sorted(set(expected) - set(found))
    unexpected = sorted(set(found) - set(expected))
    if missing or unexpected or sorted(expected) != sorted(found):
        raise ValueError(
            "Invalid SPA route inventory: "
            f"missing={missing}, unexpected={unexpected}"
        )


def verify_generated_pages() -> None:
    for static_path in registered_static_paths():
        relative_path = static_path.strip("/")
        require_file(f"{relative_path}/index.html")


def verify_sitemap(*, require_preview: bool, require_release: bool, require_develop: bool) -> None:
    sitemap_path = require_file("sitemap.xml")
    root = ET.parse(sitemap_path).getroot()
    expected_root = f"{{{SITEMAP_NAMESPACE}}}urlset"
    if root.tag != expected_root:
        raise ValueError(f"Unexpected sitemap root: {root.tag}")

    paths: set[str] = set()
    missing_lastmod: list[str] = []
    today = datetime.now(timezone.utc).date()
    for entry in root.findall(f"{{{SITEMAP_NAMESPACE}}}url"):
        location = entry.find(f"{{{SITEMAP_NAMESPACE}}}loc")
        if location is None or not (location.text or "").strip():
            raise ValueError("Sitemap entry has no loc")
        url = (location.text or "").strip()
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname != SITE_HOST:
            raise ValueError(f"Sitemap URL must use https://{SITE_HOST}: {url}")
        if parsed.path in paths:
            raise ValueError(f"Duplicate sitemap path: {parsed.path}")
        paths.add(parsed.path)

        if entry.find(f"{{{SITEMAP_NAMESPACE}}}changefreq") is not None:
            raise ValueError(f"Sitemap must not contain changefreq: {url}")
        lastmod = entry.find(f"{{{SITEMAP_NAMESPACE}}}lastmod")
        if lastmod is None or not (lastmod.text or "").strip():
            missing_lastmod.append(parsed.path)
            continue
        value = (lastmod.text or "").strip()
        try:
            parsed_date = datetime.strptime(value, "%Y-%m-%d").date()
        except ValueError as error:
            raise ValueError(f"Invalid sitemap lastmod for {url}: {value!r}") from error
        if parsed_date > today:
            raise ValueError(f"Sitemap lastmod is in the future for {url}: {value}")

    if paths != EXPECTED_SITEMAP_PATHS:
        raise ValueError(
            "Invalid sitemap URL inventory: "
            f"missing={sorted(EXPECTED_SITEMAP_PATHS - paths)}, "
            f"unexpected={sorted(paths - EXPECTED_SITEMAP_PATHS)}"
        )
    allowed_missing = set()
    if not require_preview:
        allowed_missing.add("/artifacts/sdk/preview/")
    if not require_release:
        allowed_missing.add("/artifacts/sdk/release/")
    if not require_develop:
        allowed_missing.add("/artifacts/sdk/develop/")
    unexpected_missing = sorted(set(missing_lastmod) - allowed_missing)
    if unexpected_missing:
        raise ValueError(f"Sitemap entries are missing lastmod: {unexpected_missing}")


def verify_firmware_channel(channel: str) -> None:
    channel_dir = WEB_ROOT / "artifacts" / "firmware" / channel
    manifest_path = require_file(f"artifacts/firmware/{channel}/firmware-manifest-{channel}.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("channel") != channel:
        raise ValueError(
            f"Firmware manifest channel mismatch: expected {channel!r}, "
            f"found {manifest.get('channel')!r}"
        )

    frontends = manifest.get("frontends", {})
    if set(frontends) != EXPECTED_FRONTENDS:
        raise ValueError(
            f"Invalid {channel} website frontends: "
            f"expected={sorted(EXPECTED_FRONTENDS)}, found={sorted(frontends)}"
        )

    seen = set()
    artifact_count = 0
    for frontend, metadata in frontends.items():
        for artifact in metadata.get("artifacts", []):
            artifact_count += 1
            if artifact.get("build_type") != "factory":
                raise ValueError(f"Website manifest contains non-factory firmware: {artifact}")
            key = (frontend, artifact.get("chip"))
            if key in seen:
                raise ValueError(f"Website manifest contains duplicate firmware: {key}")
            seen.add(key)
            filename = artifact.get("filename", "")
            if not filename or Path(filename).name != filename:
                raise ValueError(f"Invalid firmware artifact filename: {filename!r}")
            require_file(f"artifacts/firmware/{channel}/{filename}")

    expected = {(frontend, chip) for frontend in EXPECTED_FRONTENDS for chip in EXPECTED_CHIPS}
    if seen != expected:
        raise ValueError(
            f"Invalid {channel} website firmware matrix: "
            f"missing={sorted(expected - seen)}, unexpected={sorted(seen - expected)}"
        )
    binaries = sorted(channel_dir.glob("*.bin"))
    if artifact_count != len(expected) or len(binaries) != len(expected):
        raise ValueError(
            f"Expected {len(expected)} {channel} firmware artifacts and images, "
            f"found {artifact_count} manifest entries and {len(binaries)} images"
        )


def verify_sdk_api_version() -> None:
    version = detect_git_version()
    html = require_file("artifacts/sdk/api/index.html").read_text(encoding="utf-8")
    if 'id="projectnumber"' not in html or version not in html:
        raise ValueError(
            f"Generated SDK API reference does not show version {version!r}"
        )


def verify_sdk_channel(channel: str) -> None:
    require_file(f"artifacts/sdk/{channel}/index.html")
    manifest_path = require_file(f"artifacts/sdk/{channel}/sdk-manifest-{channel}.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("channel") != channel:
        raise ValueError(
            f"SDK manifest channel mismatch: expected {channel!r}, found {manifest.get('channel')!r}"
        )
    artifacts = manifest.get("artifacts", [])
    if {artifact.get("format") for artifact in artifacts} != {"tar.gz", "zip"}:
        raise ValueError(f"Invalid {channel} SDK artifact formats")
    for artifact in artifacts:
        if not re.fullmatch(r"[0-9a-f]{64}", artifact.get("sha256", "")):
            raise ValueError(f"Invalid SDK SHA-256 metadata: {artifact}")


def verify(args: argparse.Namespace) -> None:
    for path in (
        "sitemap.xml",
        "index.html",
        "404.html",
        "assets/js/app.js",
        "assets/js/route-registry.js",
        "assets/js/espectre-ble.js",
        "assets/js/espectre-mqtt.js",
        "assets/css/styles.css",
        "vendor/qrcodejs-1.0.0/qrcode.min.js",
        "vendor/esp-web-tools-10.4.0/install-button.js",
        "vendor/mqtt-5.3.0/mqtt.min.js",
        "artifacts/sdk/api/index.html",
    ):
        require_file(path)
    verify_spa_routes()
    verify_generated_pages()
    verify_sdk_api_version()
    verify_sitemap(
        require_preview=args.require_preview,
        require_release=args.require_release,
        require_develop=args.require_develop,
    )
    if args.require_preview:
        verify_firmware_channel("preview")
        verify_sdk_channel("preview")
    if args.require_release:
        verify_firmware_channel("release")
        verify_sdk_channel("release")
    if args.require_develop:
        verify_firmware_channel("develop")
        verify_sdk_channel("develop")


def main() -> int:
    verify(parse_args())
    print("Website build verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
