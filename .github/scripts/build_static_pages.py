#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - Build Static Pages

Generates static, indexable pages from the HTML fragments shared with the SPA.
Each fragment is the single source of truth: the SPA fetches it on demand for
its hash route, and this script wraps the same markup into the corresponding
canonical path, so the content is written once and indexed at one URL.

The output is not committed (see docs/web/.gitignore): CI runs this script
before every site verification and deploy, so the published pages always
match the fragments. To preview the static pages locally, run:

    python3 .github/scripts/build_static_pages.py

The pages reuse the site assets (version read from index.html so cache busting
stays in lockstep) with a lightweight static header, and default to the light
theme like the app. Runtime JavaScript is limited to responsive navigation and
consent-gated analytics.

Author: Francesco Pace <francesco.pace@gmail.com>
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

WEB_ROOT = Path(__file__).resolve().parents[2] / "docs" / "web"
SITE_ORIGIN = "https://espectre.dev"

PAGES = (
    {
        "source": "content/guides.html",
        "output": "guides",
        "title": "Guides | ESPectre",
        "description": (
            "Guides for ESP32 board choice, flashing, detection behavior, "
            "and embedding ESPectre into product firmware."
        ),
        "active_nav": "guides",
        "main_class": "page-narrow",
        "og_type": "website",
    },
    {
        "source": "content/guides/hardware.html",
        "output": "guides/hardware",
        "title": "Choosing an ESP32 board | ESPectre",
        "description": (
            "Which ESP32 board to buy for Wi-Fi motion sensing, what actually "
            "affects detection quality, and how products can embed ESPectre."
        ),
        "parent_href": "/guides/",
        "parent_label": "Guides",
        "active_nav": "guides",
    },
    {
        "source": "content/guides/setup.html",
        "output": "guides/setup",
        "title": "Flash & Wi-Fi setup | ESPectre",
        "description": (
            "From a blank ESP32 board to a working Wi-Fi motion sensor, "
            "entirely from the browser: flashing, provisioning, and calibration."
        ),
        "parent_href": "/guides/",
        "parent_label": "Guides",
        "active_nav": "guides",
    },
    {
        "source": "content/guides/placement.html",
        "output": "guides/placement",
        "title": "Sensor placement guide | ESPectre",
        "description": (
            "Place an ESPectre Wi-Fi motion sensor using practical distance, "
            "RSSI, antenna orientation, obstacle, and room-testing guidance."
        ),
        "parent_href": "/guides/",
        "parent_label": "Guides",
        "active_nav": "guides",
    },
    {
        "source": "content/guides/detection.html",
        "output": "guides/detection",
        "title": "How detection works | ESPectre",
        "description": (
            "What the ESPectre movement score measures, how CSI-based motion "
            "detection works, and how to place and tune the sensor."
        ),
        "parent_href": "/guides/",
        "parent_label": "Guides",
        "active_nav": "guides",
    },
    {
        "source": "content/guides/detectors.html",
        "output": "guides/detectors",
        "title": "Choose a detection profile | ESPectre",
        "description": (
            "Compare ESPectre Lightweight and High-Accuracy Detection by CPU, "
            "memory, startup behavior, detection quality, and product fit."
        ),
        "parent_href": "/guides/",
        "parent_label": "Guides",
        "active_nav": "guides",
    },
    {
        "source": "content/guides/firmware.html",
        "output": "guides/custom-firmware",
        "title": "Build custom firmware | ESPectre",
        "description": (
            "Build an ESPectre frontend from source, or embed the C++ sensing "
            "layers into your own ESP-IDF firmware."
        ),
        "parent_href": "/guides/",
        "parent_label": "Guides",
        "active_nav": "guides",
    },
    {
        "source": "content/docs.html",
        "output": "docs",
        "title": "ESPectre SDK quick guide | ESPectre",
        "description": (
            "Embed the ESPectre C++ sensing runtime in an ESP-IDF product with "
            "a concise setup path, integration examples, and architecture overview."
        ),
        "active_nav": "docs",
        "main_class": "page-narrow",
        "og_type": "website",
    },
    {
        "source": "content/docs/api.html",
        "output": "docs/api",
        "title": "API orientation | ESPectre",
        "description": (
            "Find the main ESPectre SDK types, understand the runtime lifecycle, "
            "and continue to the generated Doxygen API reference."
        ),
        "parent_href": "/docs/",
        "parent_label": "Docs",
        "active_nav": "docs",
    },
    {
        "source": "content/docs/examples.html",
        "output": "docs/examples",
        "title": "Examples | ESPectre",
        "description": (
            "Choose among the maintained ESPHome, Native, Matter, and Streamer "
            "frontends when embedding ESPectre in a product."
        ),
        "parent_href": "/docs/",
        "parent_label": "Docs",
        "active_nav": "docs",
    },
    {
        "source": "content/docs/architecture.html",
        "output": "docs/architecture",
        "title": "Architecture | ESPectre",
        "description": (
            "How ESPectre splits reusable sensing code across core, runtime, "
            "and frontend layers, including ports to new platforms."
        ),
        "parent_href": "/docs/",
        "parent_label": "Docs",
        "active_nav": "docs",
    },
    {
        "source": "content/media.html",
        "output": "media",
        "title": "Media | ESPectre",
        "description": (
            "Articles, tutorials, independent coverage, podcasts, and "
            "community conversations about the ESPectre Wi-Fi sensing platform."
        ),
        "active_nav": "media",
        "content_group": "media",
        "main_class": "page-narrow",
    },
    {
        "source": "content/roadmap.html",
        "output": "roadmap",
        "title": "Roadmap | ESPectre",
        "description": (
            "ESPectre product direction toward v3.0 and beyond: easier adoption, optional "
            "multi-device orchestration, sensing research, and future "
            "standards-backed Wi-Fi Sensing hardware."
        ),
        "active_nav": "roadmap",
        "main_class": "page-narrow",
        "og_type": "website",
    },
    {
        "source": "content/privacy.html",
        "output": "privacy",
        "title": "Website privacy and analytics | ESPectre",
        "description": (
            "How the ESPectre website handles analytics consent, cookies, "
            "browser-tool data, retention, and privacy choices."
        ),
        "active_nav": "privacy",
        "content_group": "privacy",
        "main_class": "page-narrow",
        "og_type": "website",
    },
)

PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en" data-theme="light" data-static-page data-site-section="{content_group}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{description}">
<link rel="canonical" href="{canonical}">
<meta property="og:type" content="{og_type}">
<meta property="og:url" content="{canonical}">
<meta property="og:title" content="{title}">
<meta property="og:description" content="{description}">
<meta property="og:image" content="{origin}/assets/images/brand/espectre-og.jpg">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{title}">
<meta name="twitter:description" content="{description}">
<meta name="twitter:image" content="{origin}/assets/images/brand/espectre-og.jpg">
<link rel="icon" type="image/png" href="/assets/images/brand/favicon.png">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&amp;family=Instrument+Sans:wght@400;500;600&amp;family=JetBrains+Mono:wght@400;600&amp;display=swap" rel="stylesheet">
<link rel="stylesheet" href="/assets/css/styles.css?v={styles_version}">
<script src="/assets/js/route-registry.js?v={styles_version}"></script>
<script src="/assets/js/navigation.js?v={styles_version}" defer></script>
<script src="/assets/js/analytics.js?v={styles_version}" defer></script>
</head>
<body>

<a class="skip-link" href="#main-content">Skip to content</a>

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
      <a href="/guides/" class="nav-link{guides_active}"{guides_current}>Guides</a>
      <a href="/media/" class="nav-link{media_active}"{media_current}>Media</a>
      <a href="/roadmap/" class="nav-link{roadmap_active}"{roadmap_current}>Roadmap</a>
      <a href="/docs/" class="nav-link{docs_active}"{docs_current}>Docs</a>
      <a href="https://github.com/francescopace/espectre" target="_blank" rel="noopener" class="nav-link">GitHub ↗</a>
    </nav>
  </div>
</header>

<main class="{main_class}" id="main-content" tabindex="-1">
{breadcrumb}
{content}
</main>

<footer class="site-footer">
  <div class="site-footer-inner">
    <div class="footer-brand">
      <svg width="16" height="16" viewBox="0 0 32 32" aria-hidden="true"><path d="M16 3c-6.6 0-11 4.9-11 11.5V27l3.7-2.4 3.6 2.4 3.7-2.4 3.7 2.4 3.6-2.4L27 27V14.5C27 7.9 22.6 3 16 3z" fill="var(--dim)"/></svg>
      ESPectre © 2026 · Open source Wi-Fi sensing platform · GPLv3 + commercial licensing
    </div>
    <div class="footer-links">
      <a href="/privacy/">Privacy</a>
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

def styles_version() -> str:
    """Reads the cache-busting version index.html uses for styles.css."""
    index = (WEB_ROOT / "index.html").read_text()
    match = re.search(r'href="/assets/css/styles\.css\?v=([0-9.]+)"', index)
    if not match:
        sys.exit("error: styles.css version not found in index.html")
    return match.group(1)


def crumb_from_title(title: str) -> str:
    return title.split(" | ")[0]


def breadcrumb(spec: dict[str, str]) -> str:
    if "parent_href" not in spec:
        return ""

    return (
        f'  <div class="breadcrumb"><a href="{spec["parent_href"]}">'
        f'{spec["parent_label"]}</a> <span class="crumb-sep">/</span> '
        f'<span class="crumb-here">{crumb_from_title(spec["title"])}</span></div>'
    )


def build() -> None:
    version = styles_version()

    for spec in PAGES:
        fragment_path = WEB_ROOT / spec["source"]
        content = fragment_path.read_text().rstrip("\n")

        canonical = f"{SITE_ORIGIN}/{spec['output']}/"
        page = PAGE_TEMPLATE.format(
            title=spec["title"],
            description=spec["description"],
            canonical=canonical,
            origin=SITE_ORIGIN,
            styles_version=version,
            og_type=spec.get("og_type", "article"),
            breadcrumb=breadcrumb(spec),
            guides_active=" active" if spec["active_nav"] == "guides" else "",
            docs_active=" active" if spec["active_nav"] == "docs" else "",
            media_active=" active" if spec["active_nav"] == "media" else "",
            roadmap_active=" active" if spec["active_nav"] == "roadmap" else "",
            guides_current=' aria-current="page"' if spec["active_nav"] == "guides" else "",
            docs_current=' aria-current="page"' if spec["active_nav"] == "docs" else "",
            media_current=' aria-current="page"' if spec["active_nav"] == "media" else "",
            roadmap_current=' aria-current="page"' if spec["active_nav"] == "roadmap" else "",
            content_group=spec.get("content_group", "documentation"),
            main_class=spec.get("main_class", "page-narrow page-article"),
            content=content,
        )
        out_dir = WEB_ROOT / spec["output"]
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "index.html").write_text(page)
        print(f"wrote {spec['output']}/index.html")


if __name__ == "__main__":
    build()
