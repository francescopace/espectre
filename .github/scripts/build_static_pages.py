#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - Build Static Pages

Generates static, indexable pages from the HTML fragments shared with the SPA.
Route paths, titles, descriptions, hierarchy, and Analytics grouping come from
the canonical website route manifest.

The output is not committed (see docs/web/.gitignore): CI runs this script
before every site verification and deploy. To preview static pages locally, run:

    python3 .github/scripts/build_static_pages.py

Author: Francesco Pace <francesco.pace@gmail.com>
"""

from __future__ import annotations

import sys
from html import escape
from pathlib import Path

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from web_asset_versions import asset_version
from web_page_shell import render_consent_banner, render_footer, render_header
from web_routes import (
    active_navigation,
    content_group,
    content_path,
    load_manifest,
    main_class,
    og_type,
    output_path,
    static_routes,
)


WEB_ROOT = Path(__file__).resolve().parents[2] / "docs" / "web"
ROUTE_MANIFEST = load_manifest()
SITE_ORIGIN = ROUTE_MANIFEST["siteOrigin"]


def page_spec(route: dict[str, str]) -> dict[str, str]:
    source = content_path(route)
    output = output_path(route)
    if source is None or output is None:
        raise ValueError(f"Route {route['name']} is not a generated static page")
    spec = {
        "name": route["name"],
        "source": source,
        "output": output,
        "title": route["title"],
        "description": route["description"],
        "active_nav": active_navigation(route),
        "content_group": content_group(route),
        "main_class": main_class(route, static=True),
        "og_type": og_type(route),
    }
    group = route.get("group")
    if group:
        parent = ROUTE_MANIFEST["by_name"][group]
        spec.update(
            parent_href=parent["staticPath"],
            parent_label=parent["title"].split(" | ")[0],
        )
    return spec


PAGES = tuple(page_spec(route) for route in static_routes(ROUTE_MANIFEST))


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en" data-theme="light" data-static-page data-spa-route="{name}" data-site-section="{content_group}">
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
<link rel="stylesheet" href="/assets/css/styles.css?v={styles_version}">
<script src="/assets/js/route-bootstrap.js?v={route_bootstrap_version}"></script>
<script src="/assets/js/route-registry.js?v={route_registry_version}" defer></script>
<script src="/assets/js/navigation.js?v={navigation_version}" defer></script>
<script src="/assets/js/analytics.js?v={analytics_version}" defer></script>
</head>
<body>

<a class="skip-link" href="#main-content">Skip to content</a>

{header}

<main class="{main_class}" id="main-content" tabindex="-1">
{breadcrumb}
{content}
</main>

{footer}

{consent_banner}

</body>
</html>
"""


def crumb_from_title(title: str) -> str:
    return title.split(" | ")[0]


def breadcrumb(spec: dict[str, str]) -> str:
    if "parent_href" not in spec:
        return ""
    return (
        f'  <div class="breadcrumb"><a href="{escape(spec["parent_href"], quote=True)}">'
        f'{escape(spec["parent_label"])}</a> <span class="crumb-sep">/</span> '
        f'<span class="crumb-here">{escape(crumb_from_title(spec["title"]))}</span></div>'
    )


def build() -> None:
    styles_version = asset_version("assets/css/styles.css")
    route_bootstrap_version = asset_version("assets/js/route-bootstrap.js")
    route_registry_version = asset_version("assets/js/route-registry.js")
    navigation_version = asset_version("assets/js/navigation.js")
    analytics_version = asset_version("assets/js/analytics.js")
    logo_version = asset_version("assets/images/brand/espectre-logo.svg")

    for spec in PAGES:
        content = (WEB_ROOT / spec["source"]).read_text().rstrip("\n")
        canonical = f"{SITE_ORIGIN}/{spec['output']}/"
        page = PAGE_TEMPLATE.format(
            name=escape(spec["name"], quote=True),
            title=escape(spec["title"], quote=True),
            description=escape(spec["description"], quote=True),
            canonical=escape(canonical, quote=True),
            origin=escape(SITE_ORIGIN, quote=True),
            styles_version=styles_version,
            route_bootstrap_version=route_bootstrap_version,
            route_registry_version=route_registry_version,
            navigation_version=navigation_version,
            analytics_version=analytics_version,
            logo_version=logo_version,
            header=render_header(
                ROUTE_MANIFEST,
                logo_version=logo_version,
                active=spec["active_nav"],
            ),
            footer=render_footer(ROUTE_MANIFEST, logo_version=logo_version),
            consent_banner=render_consent_banner(),
            og_type=spec["og_type"],
            breadcrumb=breadcrumb(spec),
            content_group=spec["content_group"],
            main_class=spec["main_class"],
            content=content,
        )
        out_dir = WEB_ROOT / spec["output"]
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "index.html").write_text(page)
        print(f"wrote {spec['output']}/index.html")


if __name__ == "__main__":
    build()
