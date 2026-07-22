#!/usr/bin/env python3
"""
ESPectre - Build Guide Pages

Generates the static, indexable guide pages and the sitemap from the shared
guide fragments in docs/web/guides/content/. Each fragment is the single
source of truth for one guide: the SPA fetches it on demand for the
`#guide-*` routes, and this script wraps the same markup into a standalone
page served at /guides/<name>/, so the content exists at exactly one
indexable URL and never has to be written twice.

The output is not committed (see docs/web/.gitignore): CI runs this script
before every site verification and deploy, so the published pages always
match the fragments. To preview the static pages locally, run:

    python3 .github/scripts/build_guide_pages.py

The pages reuse the site stylesheet (version read from index.html so cache
busting stays in lockstep) with a lightweight static header, and default to
the light theme like the app; there is no runtime JS except analytics.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

WEB_ROOT = Path(__file__).resolve().parents[2] / "docs" / "web"
SITE_ORIGIN = "https://espectre.dev"

# fragment name -> (url directory, page title, meta description)
GUIDES = {
    "hardware": (
        "hardware",
        "Choosing an ESP32 board | ESPectre",
        "Which ESP32 board to buy for Wi-Fi motion sensing, what actually "
        "affects detection quality, and how products can embed ESPectre.",
    ),
    "setup": (
        "setup",
        "Flash & Wi-Fi setup | ESPectre",
        "From a blank ESP32 board to a working Wi-Fi motion sensor, entirely "
        "from the browser: flashing, provisioning, and calibration.",
    ),
    "detection": (
        "detection",
        "How detection works | ESPectre",
        "What the ESPectre movement score measures, how CSI-based motion "
        "detection works, and how to place and tune the sensor.",
    ),
    "firmware": (
        "custom-firmware",
        "Build custom firmware | ESPectre",
        "Build an ESPectre frontend from source, or embed the C++ sensing "
        "layers into your own ESP-IDF firmware.",
    ),
}

PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en" data-theme="light" data-static-page>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{description}">
<link rel="canonical" href="{canonical}">
<meta property="og:type" content="article">
<meta property="og:url" content="{canonical}">
<meta property="og:title" content="{title}">
<meta property="og:description" content="{description}">
<meta property="og:image" content="{origin}/espectre.png">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{title}">
<meta name="twitter:description" content="{description}">
<meta name="twitter:image" content="{origin}/espectre.png">
<link rel="icon" type="image/png" href="/favicon.png">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Instrument+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="/styles.css?v={styles_version}">
<script src="/analytics.js?v={styles_version}" defer></script>
</head>
<body>

<header class="site-header">
  <div class="site-header-inner">
    <a href="/" class="brand">
      <svg width="22" height="22" viewBox="0 0 32 32" aria-hidden="true"><path d="M16 3c-6.6 0-11 4.9-11 11.5V27l3.7-2.4 3.6 2.4 3.7-2.4 3.7 2.4 3.6-2.4L27 27V14.5C27 7.9 22.6 3 16 3z" fill="var(--accent)"/><circle cx="12.2" cy="13.5" r="1.9" fill="var(--bg)"/><circle cx="19.8" cy="13.5" r="1.9" fill="var(--bg)"/></svg>
      ESPectre
    </a>
    <nav class="main-nav" aria-label="Main">
      <a href="/" class="nav-link">Home</a>
      <a href="/#tools" class="nav-link">Tools</a>
      <a href="/#guides" class="nav-link active">Guides</a>
      <a href="/#docs" class="nav-link">Docs</a>
      <a href="https://github.com/francescopace/espectre" target="_blank" rel="noopener" class="nav-link">GitHub ↗</a>
    </nav>
  </div>
</header>

<main class="page-narrow page-article">
  <div class="breadcrumb"><a href="/#guides">Guides</a> <span class="crumb-sep">/</span> <span class="crumb-here">{crumb}</span></div>
{article}
</main>

<footer class="site-footer">
  <div class="site-footer-inner">
    <div class="footer-brand">
      <svg width="16" height="16" viewBox="0 0 32 32" aria-hidden="true"><path d="M16 3c-6.6 0-11 4.9-11 11.5V27l3.7-2.4 3.6 2.4 3.7-2.4 3.7 2.4 3.6-2.4L27 27V14.5C27 7.9 22.6 3 16 3z" fill="var(--dim)"/></svg>
      ESPectre © 2026 · Open source Wi-Fi sensing platform · GPLv3 + commercial licensing
    </div>
    <div class="footer-links">
      <a href="mailto:contact@espectre.dev">Contact</a>
      <a href="mailto:security@espectre.dev">Security</a>
    </div>
  </div>
</footer>

</body>
</html>
"""

SITEMAP_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
{urls}
</urlset>
"""


def styles_version() -> str:
    """Reads the cache-busting version index.html uses for styles.css."""
    index = (WEB_ROOT / "index.html").read_text()
    match = re.search(r'href="styles\.css\?v=([0-9.]+)"', index)
    if not match:
        sys.exit("error: styles.css version not found in index.html")
    return match.group(1)


def crumb_from_title(title: str) -> str:
    return title.split(" | ")[0]


def build() -> None:
    version = styles_version()
    urls = [f"{SITE_ORIGIN}/"]

    for fragment_name, (url_dir, title, description) in GUIDES.items():
        fragment_path = WEB_ROOT / "guides" / "content" / f"{fragment_name}.html"
        article = fragment_path.read_text().rstrip("\n")

        canonical = f"{SITE_ORIGIN}/guides/{url_dir}/"
        page = PAGE_TEMPLATE.format(
            title=title,
            description=description,
            canonical=canonical,
            origin=SITE_ORIGIN,
            styles_version=version,
            crumb=crumb_from_title(title),
            article=article,
        )
        out_dir = WEB_ROOT / "guides" / url_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "index.html").write_text(page)
        urls.append(canonical)
        print(f"wrote guides/{url_dir}/index.html")

    entries = "\n".join(
        f"  <url><loc>{url}</loc><changefreq>weekly</changefreq></url>"
        for url in urls
    )
    (WEB_ROOT / "sitemap.xml").write_text(SITEMAP_TEMPLATE.format(urls=entries))
    print(f"wrote sitemap.xml ({len(urls)} URLs)")


if __name__ == "__main__":
    build()
