#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - Stage Web SDK

Stage same-origin SDK metadata pages that point to GitHub release assets.

Author: Francesco Pace <francesco.pace@gmail.com>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from web_asset_versions import asset_version

WEB_ROOT = Path(__file__).resolve().parents[2] / "docs" / "web"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage web SDK metadata pages.")
    parser.add_argument("--sdk-dir", required=True, help="Directory containing SDK release assets and manifest.")
    parser.add_argument("--output-dir", required=True, help="Directory where staged web SDK files should be written.")
    parser.add_argument("--channel", choices=("release", "preview", "develop"), required=True, help="Website SDK channel.")
    return parser.parse_args()


def clean_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.iterdir():
        if path.is_file() and (path.suffix == ".json" or path.name == "index.html"):
            path.unlink()


def load_sdk_manifest(sdk_dir: Path) -> dict:
    matches = sorted(sdk_dir.glob("sdk-manifest-*.json"))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one SDK manifest in {sdk_dir}, found {len(matches)}")
    return json.loads(matches[0].read_text(encoding="utf-8"))


def channel_copy(manifest: dict, channel: str) -> tuple[str, str]:
    if channel == "release":
        return (
            "Latest SDK Release",
            "Official SDK bundle for open-source and commercial embedding workflows.",
        )
    if channel == "preview":
        return (
            "SDK Release Preview",
            "Rolling SDK preview built from main. Use it to validate upcoming source changes before the next release.",
        )
    return (
        "SDK Development",
        "Rolling SDK bundle built from develop. Use it to validate in-progress source changes before they reach main.",
    )


def channel_note(channel: str) -> str:
    if channel == "develop":
        return (
            '<div class="note">This is a rolling development bundle from <code>develop</code>, not a production SDK. '
            'Use <a href="/artifacts/sdk/release/">release</a> for production integrations, or '
            '<a href="/artifacts/sdk/preview/">preview</a> to validate <code>main</code>.</div>'
        )
    if channel == "preview":
        return (
            '<div class="note">Rolling preview from <code>main</code>. '
            'Use <a href="/artifacts/sdk/release/">release</a> for production, or '
            '<a href="/artifacts/sdk/develop/">develop</a> for pre-main validation.</div>'
        )
    return (
        '<div class="note">Looking for a rolling bundle? See '
        '<a href="/artifacts/sdk/preview/">preview</a> from <code>main</code>, or '
        '<a href="/artifacts/sdk/develop/">develop</a> from <code>develop</code>.</div>'
    )


def render_page(manifest: dict, channel: str) -> str:
    title, description = channel_copy(manifest, channel)
    commit = manifest.get("commit") or "n/a"
    artifact_links = "\n".join(
        f'      <li><a href="{artifact["url"]}" data-sdk-channel="{channel}" '
        f'data-sdk-format="{artifact["format"]}"><code>{artifact["filename"]}</code></a> '
        f'(<span>{artifact["format"]}</span>)</li>'
        for artifact in manifest["artifacts"]
    )
    optional_groups = ", ".join(manifest["install_surfaces"]["cmake"]["optional_source_groups"])
    styles_version = asset_version("assets/css/styles.css")
    route_registry_version = asset_version("assets/js/route-registry.js")
    navigation_version = asset_version("assets/js/navigation.js")
    analytics_version = asset_version("assets/js/analytics.js")
    logo_version = asset_version("assets/images/brand/espectre-logo.svg")
    return f"""<!DOCTYPE html>
<html lang="en" data-theme="light" data-static-page data-site-section="documentation">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} | ESPectre</title>
<meta name="description" content="{description}">
<link rel="canonical" href="https://espectre.dev/artifacts/sdk/{channel}/">
<link rel="icon" type="image/png" href="/assets/images/brand/favicon.png">
<link rel="stylesheet" href="/assets/css/styles.css?v={styles_version}">
<script src="/assets/js/route-registry.js?v={route_registry_version}" defer></script>
<script src="/assets/js/navigation.js?v={navigation_version}" defer></script>
<script src="/assets/js/analytics.js?v={analytics_version}" defer></script>
</head>
<body>
<a class="skip-link" href="#main-content">Skip to content</a>
<header class="site-header">
  <div class="site-header-inner">
    <a href="/" class="brand">
      <img src="/assets/images/brand/espectre-logo.svg?v={logo_version}" alt="" width="30" height="30" aria-hidden="true">
      ESPectre
    </a>
    <button class="nav-toggle" type="button" aria-expanded="false" aria-controls="main-navigation">
      <span aria-hidden="true">☰</span><span class="sr-only">Open navigation</span>
    </button>
    <nav class="main-nav" id="main-navigation" aria-label="Main">
      <a href="/" class="nav-link">Home</a>
      <a href="/#tools" class="nav-link">Tools</a>
      <a href="/guides/" class="nav-link">Guides</a>
      <a href="/media/" class="nav-link">Media</a>
      <a href="/sdk/" class="nav-link active" aria-current="page">SDK</a>
      <a href="/roadmap/" class="nav-link">Roadmap</a>
      <a href="https://github.com/francescopace/espectre" target="_blank" rel="noopener" class="nav-link">GitHub ↗</a>
    </nav>
  </div>
</header>
<main class="page-narrow page-article" id="main-content" tabindex="-1">
  <div class="breadcrumb"><a href="/sdk/">SDK</a> <span class="crumb-sep">/</span> <span class="crumb-here">{title}</span></div>
  <article class="article">
    <span class="guide-tag">SDK · DOWNLOADS</span>
    <h1>{title}</h1>
    <p class="article-lead">{description}</p>

    <div class="table-wrap"><table>
      <thead><tr><th>Field</th><th>Value</th></tr></thead>
      <tbody>
        <tr><td>Channel</td><td><code>{manifest["channel"]}</code></td></tr>
        <tr><td>Version label</td><td><code>{manifest["version"]}</code></td></tr>
        <tr><td>Package version</td><td><code>{manifest["package_version"]}</code></td></tr>
        <tr><td>Release tag</td><td><code>{manifest["release_tag"]}</code></td></tr>
        <tr><td>Protocol version</td><td><code>{manifest["protocol_version"]}</code></td></tr>
        <tr><td>ESP-IDF baseline</td><td><code>{manifest["supported_esp_idf"]}</code></td></tr>
        <tr><td>Commit</td><td><code>{commit}</code></td></tr>
      </tbody>
    </table></div>

    <h2>Downloads</h2>
    <ul class="checklist">
{artifact_links}
    </ul>

    <h2>Install surfaces</h2>
    <div class="table-wrap"><table>
      <thead><tr><th>Surface</th><th>Bundle anchor</th></tr></thead>
      <tbody>
        <tr><td>CMake / ESP-IDF</td><td><code>{manifest["install_surfaces"]["cmake"]["entrypoint"]}</code> plus optional groups <code>{optional_groups}</code></td></tr>
        <tr><td>ESP-IDF component layout</td><td><code>{manifest["install_surfaces"]["esp_idf_component"]["component_root"]}</code>, <code>{manifest["install_surfaces"]["esp_idf_component"]["cmake"]}</code>, and <code>{manifest["install_surfaces"]["esp_idf_component"]["kconfig"]}</code></td></tr>
      </tbody>
    </table></div>

    {channel_note(channel)}
  </article>
</main>
<footer class="site-footer">
  <div class="site-footer-inner">
    <div class="footer-brand">
      <img src="/assets/images/brand/espectre-logo.svg?v={logo_version}" alt="" width="23" height="23" aria-hidden="true">
      ESPectre © 2026 · Open source Wi-Fi sensing platform
    </div>
    <div class="footer-links">
      <a href="/privacy/">Privacy</a>
      <a href="/privacy/#cookie-settings" class="js-cookie-settings">Cookie settings</a>
      <a href="/terms/">Terms</a>
      <a href="/legal/">Legal</a>
      <a href="/security/">Security</a>
      <a href="/licensing/">Licensing</a>
      <a href="/contact/">Contact</a>
    </div>
  </div>
</footer>
<aside class="consent-banner js-consent-banner" role="dialog" aria-labelledby="consent-title" hidden>
  <div>
    <strong id="consent-title">Optional analytics</strong>
    <p>Help improve ESPectre with privacy-conscious usage analytics. Browser-tool credentials and device identifiers are never included.</p>
    <a href="/privacy/">Read the privacy notice</a>
  </div>
  <div class="consent-actions">
    <button class="btn-ghost js-consent-reject" type="button">Reject</button>
    <button class="btn-primary btn-sm js-consent-accept" type="button">Accept analytics</button>
  </div>
</aside>
</body>
</html>
"""


def stage_web_sdk(args: argparse.Namespace) -> Path:
    sdk_dir = Path(args.sdk_dir)
    output_dir = Path(args.output_dir)
    manifest = load_sdk_manifest(sdk_dir)
    if manifest.get("channel") != args.channel:
        raise ValueError(
            f"SDK manifest channel mismatch: expected {args.channel!r}, "
            f"found {manifest.get('channel')!r}"
        )

    clean_output_dir(output_dir)

    manifest_path = output_dir / f"sdk-manifest-{args.channel}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    page = render_page(manifest, args.channel)
    (output_dir / "index.html").write_text(page, encoding="utf-8")
    return manifest_path


def main() -> int:
    stage_web_sdk(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
