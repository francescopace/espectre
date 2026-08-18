#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Verify the complete generated GitHub Pages tree before upload."""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = REPO_ROOT / "docs" / "web"
EXPECTED_CHIPS = {"esp32", "esp32c3", "esp32c5", "esp32c6", "esp32s3"}
EXPECTED_FRONTENDS = {"esphome", "matter", "native"}
SITEMAP_NAMESPACE = "http://www.sitemaps.org/schemas/sitemap/0.9"
SITE_HOST = "espectre.dev"
SPA_ROUTE_NAME_RE = re.compile(r"\{\s*name:\s*'([^']+)'")
SPA_PAGE_ROUTE_RE = re.compile(r'<main\b[^>]*\bdata-page="([^"]+)"')
EXPECTED_SITEMAP_PATHS = {
    "/",
    "/guides/",
    "/guides/hardware/",
    "/guides/setup/",
    "/guides/placement/",
    "/guides/detection/",
    "/guides/detectors/",
    "/guides/custom-firmware/",
    "/docs/",
    "/docs/api/",
    "/docs/examples/",
    "/docs/architecture/",
    "/artifacts/sdk/api/",
    "/artifacts/sdk/stable/",
    "/artifacts/sdk/main/",
    "/media/",
    "/roadmap/",
    "/privacy/",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the generated ESPectre Pages tree.")
    parser.add_argument("--require-main", action="store_true")
    parser.add_argument("--require-stable", action="store_true")
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
    for guide in ("hardware", "setup", "placement", "detection", "custom-firmware"):
        require_file(f"guides/{guide}/index.html")
    for page in (
        "docs/index.html",
        "docs/api/index.html",
        "docs/examples/index.html",
        "docs/architecture/index.html",
        "media/index.html",
        "roadmap/index.html",
        "privacy/index.html",
    ):
        require_file(page)

    invalid_links = []
    for path in sorted(WEB_ROOT.rglob("*.html")):
        text = path.read_text(encoding="utf-8")
        if re.search(r'href="/sdk/api(?:/|\")', text):
            invalid_links.append(str(path.relative_to(WEB_ROOT)))
    if invalid_links:
        raise ValueError(
            "SDK API links must use /artifacts/sdk/api/: " + ", ".join(invalid_links)
        )


def verify_sitemap(*, require_main: bool, require_stable: bool) -> None:
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
    if not require_main:
        allowed_missing.add("/artifacts/sdk/main/")
    if not require_stable:
        allowed_missing.add("/artifacts/sdk/stable/")
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
        "assets/js/LICENSES/Apache-2.0.txt",
        "assets/css/styles.css",
        "vendor/qrcodejs-1.0.0/qrcode.min.js",
        "vendor/esp-web-tools-10.4.0/install-button.js",
        "vendor/mqtt-5.3.0/mqtt.min.js",
        "artifacts/sdk/api/index.html",
    ):
        require_file(path)
    verify_spa_routes()
    verify_generated_pages()
    verify_sitemap(require_main=args.require_main, require_stable=args.require_stable)
    if args.require_main:
        verify_firmware_channel("main")
        verify_sdk_channel("main")
    if args.require_stable:
        verify_firmware_channel("stable")
        verify_sdk_channel("stable")


def main() -> int:
    verify(parse_args())
    print("Website build verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
