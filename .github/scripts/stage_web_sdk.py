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
import re
from pathlib import Path

WEB_ROOT = Path(__file__).resolve().parents[2] / "docs" / "web"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage web SDK metadata pages.")
    parser.add_argument("--sdk-dir", required=True, help="Directory containing SDK release assets and manifest.")
    parser.add_argument("--output-dir", required=True, help="Directory where staged web SDK files should be written.")
    parser.add_argument("--channel", choices=("stable", "main"), required=True, help="Website SDK channel.")
    return parser.parse_args()


def styles_version() -> str:
    index = (WEB_ROOT / "index.html").read_text(encoding="utf-8")
    match = re.search(r'href="/assets/css/styles\.css\?v=([0-9.]+)"', index)
    if not match:
        raise ValueError("styles.css version not found in docs/web/index.html")
    return match.group(1)


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
    if channel == "stable":
        return (
            "Latest SDK Release",
            "Official SDK bundle for open-source and commercial embedding workflows.",
        )
    return (
        "SDK Release Preview",
        "Rolling SDK preview built from main. Use it to validate upcoming source changes before the next stable release.",
    )


def render_page(manifest: dict, channel: str, styles_css_version: str) -> str:
    title, description = channel_copy(manifest, channel)
    commit = manifest.get("commit") or "n/a"
    artifact_links = "\n".join(
        f'      <li><a href="{artifact["url"]}"><code>{artifact["filename"]}</code></a> '
        f'(<span>{artifact["format"]}</span>)</li>'
        for artifact in manifest["artifacts"]
    )
    optional_groups = ", ".join(manifest["install_surfaces"]["cmake"]["optional_source_groups"])
    return f"""<!DOCTYPE html>
<html lang="en" data-theme="light" data-static-page data-site-section="documentation">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} | ESPectre</title>
<meta name="description" content="{description}">
<link rel="canonical" href="https://espectre.dev/artifacts/sdk/{channel}/">
<link rel="icon" type="image/png" href="/assets/images/brand/favicon.png">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&amp;family=Instrument+Sans:wght@400;500;600&amp;family=JetBrains+Mono:wght@400;600&amp;display=swap" rel="stylesheet">
<link rel="stylesheet" href="/assets/css/styles.css?v={styles_css_version}">
<script src="/assets/js/navigation.js?v={styles_css_version}" defer></script>
<script src="/assets/js/analytics.js?v={styles_css_version}" defer></script>
</head>
<body>
<header class="site-header">
  <div class="site-header-inner">
    <a href="/" class="brand">
      <svg width="22" height="22" viewBox="0 0 32 32" aria-hidden="true"><path d="M16 3c-6.6 0-11 4.9-11 11.5V27l3.7-2.4 3.6 2.4 3.7-2.4 3.7 2.4 3.6-2.4L27 27V14.5C27 7.9 22.6 3 16 3z" fill="var(--accent)"/><circle cx="12.2" cy="13.5" r="1.9" fill="var(--bg)"/><circle cx="19.8" cy="13.5" r="1.9" fill="var(--bg)"/></svg>
      ESPectre
    </a>
    <button class="nav-toggle" type="button" aria-expanded="false" aria-controls="main-navigation">
      <span aria-hidden="true">☰</span><span class="sr-only">Open navigation</span>
    </button>
    <nav class="main-nav" id="main-navigation" aria-label="Main">
      <a href="/" class="nav-link">Home</a>
      <a href="/#tools" class="nav-link">Tools</a>
      <a href="/guides/" class="nav-link">Guides</a>
      <a href="/docs/" class="nav-link active">Docs</a>
      <a href="/media/" class="nav-link">Media</a>
      <a href="/roadmap/" class="nav-link">Roadmap</a>
      <a href="https://github.com/francescopace/espectre" target="_blank" rel="noopener" class="nav-link">GitHub ↗</a>
    </nav>
  </div>
</header>
<main class="page-narrow page-article">
  <div class="breadcrumb"><a href="/docs/">Docs</a> <span class="crumb-sep">/</span> <span class="crumb-here">{title}</span></div>
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

    <div class="note">Developer preview SDK bundles from <code>develop</code> are intentionally GitHub-only and do not get a public website page.</div>
  </article>
</main>
<footer class="site-footer">
  <div class="site-footer-inner">
    <div class="footer-brand">
      <svg width="16" height="16" viewBox="0 0 32 32" aria-hidden="true"><path d="M16 3c-6.6 0-11 4.9-11 11.5V27l3.7-2.4 3.6 2.4 3.7-2.4 3.7 2.4 3.6-2.4L27 27V14.5C27 7.9 22.6 3 16 3z" fill="var(--dim)"/></svg>
      ESPectre © 2026 · Open source Wi-Fi sensing platform · GPLv3 + commercial licensing
    </div>
    <div class="footer-links">
      <a href="/#privacy">Privacy</a>
      <button class="footer-link-button js-cookie-settings" type="button">Cookie settings</button>
      <a href="mailto:contact@espectre.dev">Contact/Commercial Licensing</a>
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
    page = render_page(manifest, args.channel, styles_version())
    (output_dir / "index.html").write_text(page, encoding="utf-8")
    return manifest_path


def main() -> int:
    stage_web_sdk(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
