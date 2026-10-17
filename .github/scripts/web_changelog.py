#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Render the website changelog from the repository's release notes."""

from __future__ import annotations

import re
from html import escape
from pathlib import Path
from urllib.parse import urljoin, urlsplit

from markdown import Markdown
from markdown.extensions.toc import slugify
from markdown.treeprocessors import Treeprocessor


CHANGELOG_PATH = Path(__file__).resolve().parents[2] / "docs" / "CHANGELOG.md"
RELEASE_HEADING = re.compile(r"^## \[([^\]]+)\] - (.+)$", re.MULTILINE)


class ReleaseLinks(Treeprocessor):
    """Resolve relative documentation links against the corresponding release."""

    def __init__(self, md: Markdown, ref: str) -> None:
        super().__init__(md)
        self.base = f"https://github.com/francescopace/espectre/blob/{ref}/docs/CHANGELOG.md"

    def run(self, root):
        for link in root.iter("a"):
            href = link.get("href", "")
            if href and not urlsplit(href).scheme and not href.startswith(("/", "#")):
                link.set("href", urljoin(self.base, href))


def render_changelog() -> str:
    source = CHANGELOG_PATH.read_text(encoding="utf-8")
    releases = list(RELEASE_HEADING.finditer(source))
    if not releases:
        raise ValueError("Changelog contains no release headings")
    parts = [
        "<!-- Generated from docs/CHANGELOG.md; do not edit. -->",
        '<article class="article changelog-article" data-page-toc>',
        "  <h1>Changelog</h1>",
        '  <p class="article-lead">Release notes, fixes, and upgrade instructions. '
        "Unreleased entries describe work in progress.</p>",
    ]
    seen_versions = set()
    for index, release in enumerate(releases):
        version, metadata = release.groups()
        if not re.fullmatch(r"\d+\.\d+\.\d+(?:-[a-z0-9.-]+)?", version) or version in seen_versions:
            raise ValueError(f"Invalid or duplicate changelog version: {version}")
        seen_versions.add(version)
        release_id = f"release-{version.replace('.', '-')}"
        unreleased = metadata.lower() in {"work in progress", "unreleased"}
        label = f"{version} (Unreleased)" if unreleased else version
        date, _, title = metadata.partition(" - ")
        heading = [f'<span class="changelog-version">{escape(version)}</span>']
        if unreleased:
            heading.append('<span class="changelog-meta">Unreleased</span>')
        else:
            heading.append(f'<time class="changelog-meta" datetime="{escape(date, quote=True)}">({escape(date)})</time>')
        if title:
            heading.append(f'<span class="changelog-title">{escape(title)}</span>')
        parts.append(f'<h2 id="{release_id}" data-toc-label="{escape(label)}">{" ".join(heading)}</h2>')
        end = releases[index + 1].start() if index + 1 < len(releases) else len(source)
        body = source[release.end():end].strip()
        md = Markdown(extensions=["fenced_code", "tables", "toc"], extension_configs={
            "toc": {"slugify": lambda value, separator, prefix=release_id: f"{prefix}-{slugify(value, separator)}"},
        })
        md.treeprocessors.register(ReleaseLinks(md, "develop" if unreleased else version), "release_links", 5)
        html = md.convert(body)
        # Release headings supply the boundary; keep any separators inside the notes.
        html = re.sub(r"\s*<hr\s*/?>\s*\Z", "", html)
        html = html.replace("<pre>", '<pre class="code">')
        html = html.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")
        parts.append(html)
    parts.append("</article>\n")
    return "\n".join(parts)
