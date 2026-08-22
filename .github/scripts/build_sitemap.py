#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Generate the deployable sitemap from the canonical URL inventory."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = REPO_ROOT / "docs" / "web"
DEFAULT_SITEMAP_TEMPLATE = Path(__file__).resolve().with_name("sitemap.template.xml")
DEFAULT_SITEMAP_OUTPUT = WEB_ROOT / "sitemap.xml"
SITEMAP_NAMESPACE = "http://www.sitemaps.org/schemas/sitemap/0.9"
SITE_HOST = "espectre.dev"
STATIC_PAGE_BUILDER = Path(".github/scripts/build_static_pages.py")
SDK_PAGE_BUILDER = Path(".github/scripts/stage_web_sdk.py")
DOXYFILE = Path("src/cpp/Doxyfile")

ROUTE_SOURCES = {
    "/": (Path("docs/web/index.html"),),
    "/guides/": (Path("docs/web/content/guides.html"), STATIC_PAGE_BUILDER),
    "/guides/hardware/": (Path("docs/web/content/guides/hardware.html"), STATIC_PAGE_BUILDER),
    "/guides/setup/": (Path("docs/web/content/guides/setup.html"), STATIC_PAGE_BUILDER),
    "/guides/home-assistant/": (Path("docs/web/content/guides/home-assistant.html"), STATIC_PAGE_BUILDER),
    "/guides/placement/": (Path("docs/web/content/guides/placement.html"), STATIC_PAGE_BUILDER),
    "/guides/detection/": (Path("docs/web/content/guides/detection.html"), STATIC_PAGE_BUILDER),
    "/guides/detectors/": (Path("docs/web/content/guides/detectors.html"), STATIC_PAGE_BUILDER),
    "/guides/micropython/": (Path("docs/web/content/guides/micropython.html"), STATIC_PAGE_BUILDER),
    "/guides/future-wifi-sensing/": (Path("docs/web/content/guides/future-wifi-sensing.html"), STATIC_PAGE_BUILDER),
    "/sdk/": (Path("docs/web/content/sdk.html"), STATIC_PAGE_BUILDER),
    "/sdk/detectors/": (Path("docs/web/content/sdk/detectors.html"), STATIC_PAGE_BUILDER),
    "/sdk/api/": (Path("docs/web/content/sdk/api.html"), STATIC_PAGE_BUILDER),
    "/sdk/examples/": (Path("docs/web/content/sdk/examples.html"), STATIC_PAGE_BUILDER),
    "/sdk/architecture/": (Path("docs/web/content/sdk/architecture.html"), STATIC_PAGE_BUILDER),
    "/media/": (Path("docs/web/content/media.html"), STATIC_PAGE_BUILDER),
    "/roadmap/": (Path("docs/web/content/roadmap.html"), STATIC_PAGE_BUILDER),
    "/privacy/": (Path("docs/web/content/privacy.html"), STATIC_PAGE_BUILDER),
    "/terms/": (Path("docs/web/content/terms.html"), STATIC_PAGE_BUILDER),
    "/legal/": (Path("docs/web/content/legal.html"), STATIC_PAGE_BUILDER),
    "/security/": (Path("docs/web/content/security.html"), STATIC_PAGE_BUILDER),
    "/licensing/": (Path("docs/web/content/licensing.html"), STATIC_PAGE_BUILDER),
    "/contact/": (Path("docs/web/content/contact.html"), STATIC_PAGE_BUILDER),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate evidence-based sitemap lastmod values.")
    parser.add_argument(
        "--template",
        "--sitemap",
        default=str(DEFAULT_SITEMAP_TEMPLATE),
        help="Canonical sitemap URL inventory to enrich.",
    )
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
    return max(normalized_date(str(generated_at)), latest_git_date((SDK_PAGE_BUILDER,)))


def lastmod_for_url(url: str) -> str | None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != SITE_HOST:
        raise ValueError(f"Sitemap URL must use https://{SITE_HOST}: {url}")
    if parsed.path in ROUTE_SOURCES:
        return latest_git_date(ROUTE_SOURCES[parsed.path])
    if parsed.path == "/artifacts/sdk/api/":
        return latest_git_date(doxygen_sources())
    if parsed.path == "/artifacts/sdk/release/":
        return sdk_channel_date("release")
    if parsed.path == "/artifacts/sdk/preview/":
        return sdk_channel_date("preview")
    if parsed.path == "/artifacts/sdk/develop/":
        return sdk_channel_date("develop")
    raise ValueError(f"Sitemap URL has no lastmod ownership mapping: {url}")


def build_sitemap(sitemap_path: Path, output_path: Path) -> None:
    if sitemap_path.resolve() == output_path.resolve():
        raise ValueError("Sitemap template and generated output must use separate paths")
    tree = ET.parse(sitemap_path)
    root = tree.getroot()
    expected_root = f"{{{SITEMAP_NAMESPACE}}}urlset"
    if root.tag != expected_root:
        raise ValueError(f"Unexpected sitemap root: {root.tag}")

    seen: set[str] = set()
    for entry in root.findall(f"{{{SITEMAP_NAMESPACE}}}url"):
        location = entry.find(f"{{{SITEMAP_NAMESPACE}}}loc")
        if location is None or not (location.text or "").strip():
            raise ValueError("Sitemap entry has no loc")
        url = (location.text or "").strip()
        if url in seen:
            raise ValueError(f"Duplicate sitemap URL: {url}")
        seen.add(url)

        for tag in ("changefreq", "lastmod"):
            existing = entry.find(f"{{{SITEMAP_NAMESPACE}}}{tag}")
            if existing is not None:
                entry.remove(existing)
        lastmod = lastmod_for_url(url)
        if lastmod is not None:
            ET.SubElement(entry, f"{{{SITEMAP_NAMESPACE}}}lastmod").text = lastmod

    ET.register_namespace("", SITEMAP_NAMESPACE)
    ET.indent(tree, space="  ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True, short_empty_elements=False)


def main() -> int:
    args = parse_args()
    sitemap_path = Path(args.template)
    output_path = Path(args.output)
    build_sitemap(sitemap_path, output_path)
    print(f"Sitemap lastmod values generated in {output_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
