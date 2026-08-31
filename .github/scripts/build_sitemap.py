#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Generate the deployable sitemap from the canonical route manifest."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from web_routes import (
    ROUTES_PATH,
    SITEMAP_NAMESPACE,
    content_path,
    load_manifest,
    staged_sdk_channels,
    static_routes,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = REPO_ROOT / "docs" / "web"
DEFAULT_SITEMAP_OUTPUT = WEB_ROOT / "sitemap.xml"
ROUTE_MANIFEST = load_manifest()
SITE_ORIGIN = ROUTE_MANIFEST["siteOrigin"]
SITE_HOST = urlparse(SITE_ORIGIN).hostname or ""
ROUTE_MANIFEST_SOURCE = ROUTES_PATH.relative_to(REPO_ROOT)
STATIC_PAGE_BUILDER = Path(".github/scripts/build_static_pages.py")
SDK_PAGE_BUILDER = Path(".github/scripts/stage_web_sdk.py")
WEB_PAGE_SHELL = Path(".github/scripts/web_page_shell.py")
WEB_ASSET_VERSIONS = Path(".github/scripts/web_asset_versions.py")
SDK_API_BUILDER = Path(".github/scripts/generate_sdk_api.py")
MCSS_TEMPLATES = Path(".github/mcss/templates")
SDK_API_INPUTS = (SDK_API_BUILDER, MCSS_TEMPLATES)
DOXYFILE = Path("src/cpp/Doxyfile")
SHARED_STATIC_INPUTS = (
    ROUTE_MANIFEST_SOURCE,
    STATIC_PAGE_BUILDER,
    WEB_PAGE_SHELL,
    WEB_ASSET_VERSIONS,
    Path("docs/web/assets/css/styles.css"),
    Path("docs/web/assets/js/route-registry.js"),
    Path("docs/web/assets/js/navigation.js"),
    Path("docs/web/assets/js/analytics.js"),
    Path("docs/web/assets/images/brand/espectre-logo.svg"),
)
SDK_CHANNEL_PAGE_INPUTS = (
    ROUTE_MANIFEST_SOURCE,
    SDK_PAGE_BUILDER,
    WEB_PAGE_SHELL,
    WEB_ASSET_VERSIONS,
    *SHARED_STATIC_INPUTS[4:],
)

ROUTE_SOURCES = {
    "/": (Path("docs/web/index.html"), ROUTE_MANIFEST_SOURCE),
    **{
        route["staticPath"]: (
            Path("docs/web") / content_path(route),
            *SHARED_STATIC_INPUTS,
        )
        for route in static_routes(ROUTE_MANIFEST)
    },
}
SDK_CHANNELS_BY_PATH = {
    sdk_channel["path"]: sdk_channel["sdkChannel"]
    for sdk_channel in ROUTE_MANIFEST["sdkChannels"]
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate evidence-based sitemap lastmod values.")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_SITEMAP_OUTPUT),
        help="Generated sitemap output path.",
    )
    return parser.parse_args()


def normalized_date(value: str) -> str:
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).date().isoformat()


def latest_git_date(paths: tuple[Path, ...]) -> str:
    result = subprocess.run(
        ["git", "log", "-1", "--format=%cI", "--", *(str(path) for path in paths)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    timestamp = result.stdout.strip()
    if not timestamp:
        raise ValueError(
            "Unable to determine lastmod from Git history for: "
            + ", ".join(str(path) for path in paths)
        )
    return normalized_date(timestamp)


def doxygen_sources() -> tuple[Path, ...]:
    source = (REPO_ROOT / DOXYFILE).read_text(encoding="utf-8")
    input_block = re.search(r"^INPUT\s*=(.*?)(?=^[A-Z_]+\s*=)", source, re.MULTILINE | re.DOTALL)
    if input_block is None:
        raise ValueError("Doxyfile has no INPUT block")
    inputs = tuple(Path(path) for path in re.findall(r"(?:src/cpp|docs)/[^\s\\]+", input_block.group(1)))
    if not inputs:
        raise ValueError("Doxyfile INPUT block contains no paths")
    return (DOXYFILE, *inputs)


def sdk_channel_date(channel: str) -> str | None:
    manifest_path = WEB_ROOT / "artifacts" / "sdk" / channel / f"sdk-manifest-{channel}.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("channel") != channel:
        raise ValueError(
            f"SDK manifest channel mismatch while building sitemap: expected {channel!r}, "
            f"found {manifest.get('channel')!r}"
        )
    generated_at = manifest.get("generated_at")
    if not generated_at:
        raise ValueError(f"SDK manifest has no generated_at timestamp: {manifest_path}")
    return max(normalized_date(str(generated_at)), latest_git_date(SDK_CHANNEL_PAGE_INPUTS))


def lastmod_for_url(url: str) -> str | None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != SITE_HOST:
        raise ValueError(f"Sitemap URL must use https://{SITE_HOST}: {url}")
    if parsed.path == "/sdk/api/":
        return latest_git_date(
            (*ROUTE_SOURCES[parsed.path], *SDK_API_INPUTS, *doxygen_sources())
        )
    if parsed.path in ROUTE_SOURCES:
        return latest_git_date(ROUTE_SOURCES[parsed.path])
    channel = SDK_CHANNELS_BY_PATH.get(parsed.path)
    if channel:
        return sdk_channel_date(channel)
    raise ValueError(f"Sitemap URL has no lastmod ownership mapping: {url}")


def public_urls() -> tuple[str, ...]:
    paths = (
        *(route["staticPath"] for route in ROUTE_MANIFEST["routes"]),
        *(sdk_channel["path"] for sdk_channel in staged_sdk_channels(WEB_ROOT, ROUTE_MANIFEST)),
    )
    return tuple(f"{SITE_ORIGIN}{path}" for path in paths)


def build_sitemap(output_path: Path) -> None:
    root = ET.Element(f"{{{SITEMAP_NAMESPACE}}}urlset")
    for url in public_urls():
        entry = ET.SubElement(root, f"{{{SITEMAP_NAMESPACE}}}url")
        ET.SubElement(entry, f"{{{SITEMAP_NAMESPACE}}}loc").text = url
        lastmod = lastmod_for_url(url)
        if lastmod is not None:
            ET.SubElement(entry, f"{{{SITEMAP_NAMESPACE}}}lastmod").text = lastmod

    ET.register_namespace("", SITEMAP_NAMESPACE)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True, short_empty_elements=False)


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    build_sitemap(output_path)
    print(f"Sitemap lastmod values generated in {output_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
