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
from html.parser import HTMLParser
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


class PageNavigationParser(HTMLParser):
    """Locate the authored article and its sections without rewriting its HTML."""

    VOID_TAGS = {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr'}

    def __init__(self, source: str) -> None:
        super().__init__()
        # HTMLParser counts lines on newlines only, unlike str.splitlines().
        self.line_starts = [0]
        self.line_starts.extend(index + 1 for index, char in enumerate(source) if char == '\n')
        self.stack: list[tuple[str, int]] = []
        self.root_depth = 0
        self.group = ''
        self.has_toc = False
        self.has_path = False
        self.end = 0
        self.headings: list[dict] = []
        self.heading: dict | None = None
        self.feed(source)

    def source_offset(self) -> int:
        line, column = self.getpos()
        return self.line_starts[line - 1] + column

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        offset = self.source_offset()
        classes = (values.get('class') or '').split()
        self.has_toc |= 'page-toc' in classes
        self.has_path |= 'page-path' in classes
        if 'data-page-toc' in values:
            if self.root_depth:
                raise ValueError('Nested data-page-toc containers are not supported')
            self.root_depth = len(self.stack) + 1
            self.group = values.get('data-page-path') or ''
        if self.root_depth and tag == 'h2' and 'data-toc-exclude' not in values:
            self.heading = {
                'id': values.get('id'), 'label': values.get('data-toc-label'), 'text': [],
                'insert': self.stack[self.root_depth][1] if len(self.stack) > self.root_depth else offset,
            }
            self.headings.append(self.heading)
        if tag not in self.VOID_TAGS:
            self.stack.append((tag, offset))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in self.VOID_TAGS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag == 'h2':
            self.heading = None
        if self.root_depth and len(self.stack) == self.root_depth:
            self.end = self.source_offset()
            self.root_depth = 0
        if self.stack and self.stack[-1][0] == tag:
            self.stack.pop()

    def handle_data(self, data: str) -> None:
        if self.heading is not None:
            self.heading['text'].append(data)


def render_page_navigation(content: str, spec: dict[str, str]) -> str:
    """Ship crawlable navigation before first paint; JS enhances it in place."""
    parser = PageNavigationParser(content)
    edits: list[tuple[int, str]] = []
    if len(parser.headings) >= 2 and not parser.has_toc:
        links = []
        for heading in parser.headings:
            text = ' '.join(''.join(heading['text']).split())
            # Authored IDs keep inbound links identical in static pages and the SPA.
            if not heading['id']:
                raise ValueError(f"Section heading needs an explicit id in {spec['source']}: {text}")
            label = heading['label'] or text.rstrip('.:')
            links.append(f'<a href="#{escape(heading["id"], quote=True)}">{escape(label)}</a>')
        toc = '<details class="page-toc" data-toc-generated="true"><summary>Sections</summary><nav aria-label="Page sections"><strong>On this page</strong>' + ''.join(links) + '</nav></details>\n'
        edits.append((parser.headings[0]['insert'], toc))
    if parser.group and parser.end and not parser.has_path:
        entries = [r for r in ROUTE_MANIFEST['routes'] if r.get('group') == parser.group and r.get('pathLabel')]
        parent = ROUTE_MANIFEST['by_name'].get(parser.group)
        if parent and parent.get('pathLabel'):
            entries.insert(0, parent)
        if len(entries) >= 2 and any(r['name'] == spec['name'] for r in entries):
            label = {'guides': 'Guides', 'sdk': 'SDK guide', 'tools': 'Tools'}.get(parser.group, parser.group)
            links = []
            for entry in entries:
                current = ' aria-current="page"' if entry['name'] == spec['name'] else ''
                links.append(f'<a href="{escape(entry["staticPath"], quote=True)}"{current}>{escape(entry["pathLabel"])}</a>')
            path = f'<details class="page-path" data-path-generated="true"><summary>{escape(label)}</summary><nav aria-label="{escape(label, quote=True)}"><strong>{escape(label)}</strong>' + ''.join(links) + '</nav></details>\n'
            edits.append((parser.end, path))
    for offset, markup in sorted(edits, reverse=True):
        content = content[:offset] + markup + content[offset:]
    return content


def build() -> None:
    styles_version = asset_version("assets/css/styles.css")
    route_bootstrap_version = asset_version("assets/js/route-bootstrap.js")
    route_registry_version = asset_version("assets/js/route-registry.js")
    navigation_version = asset_version("assets/js/navigation.js")
    analytics_version = asset_version("assets/js/analytics.js")
    logo_version = asset_version("assets/images/brand/espectre-logo.svg")

    for spec in PAGES:
        content = (WEB_ROOT / spec["source"]).read_text().rstrip("\n")
        content = render_page_navigation(content, spec)
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
