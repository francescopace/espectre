#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Security helpers for HTML generated into the public website."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from urllib.parse import urlsplit


SAFE_API_TAGS = frozenset(
    {
        "a",
        "article",
        "aside",
        "blockquote",
        "br",
        "code",
        "dd",
        "details",
        "div",
        "dl",
        "dt",
        "em",
        "h1",
        "h2",
        "h3",
        "h4",
        "hr",
        "kbd",
        "li",
        "ol",
        "p",
        "pre",
        "samp",
        "section",
        "small",
        "span",
        "strong",
        "sub",
        "summary",
        "sup",
        "table",
        "tbody",
        "td",
        "tfoot",
        "th",
        "thead",
        "tr",
        "ul",
        "var",
        "wbr",
    }
)
SAFE_API_ATTRIBUTES = frozenset(
    {
        "class",
        "colspan",
        "data-api-reference-fragment",
        "data-api-reference-member",
        "data-api-reference-ref",
        "href",
        "id",
        "name",
        "role",
        "rowspan",
        "scope",
        "style",
        "title",
    }
)
SAFE_HREF_SCHEMES = frozenset({"http", "https", "mailto", "tel"})
SAFE_TABLE_STYLE_RE = re.compile(r"^width:\s*1%\s*;?$", re.IGNORECASE)


class PassiveApiFragmentParser(HTMLParser):
    """Reject markup that can execute code or load active embedded content."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)

    @staticmethod
    def validate_attribute(tag: str, name: str, value: str | None) -> None:
        normalized = name.lower()
        if normalized.startswith("on"):
            raise ValueError(f"API reference contains an event handler: {name}")
        if normalized.startswith("aria-"):
            return
        if normalized not in SAFE_API_ATTRIBUTES:
            raise ValueError(f"API reference contains an unsafe {tag} attribute: {name}")
        if normalized == "style" and not SAFE_TABLE_STYLE_RE.fullmatch(value or ""):
            raise ValueError(f"API reference contains an unsafe inline style: {value!r}")
        if normalized != "href" or not value:
            return
        if any(character in value for character in "\r\n\0"):
            raise ValueError("API reference contains control characters in a link")
        if value.startswith(("#", "./", "../")) or (
            value.startswith("/") and not value.startswith("//")
        ):
            return
        scheme = urlsplit(value).scheme.lower()
        if scheme not in SAFE_HREF_SCHEMES:
            raise ValueError(f"API reference contains an unsafe link: {value!r}")

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized = tag.lower()
        if normalized not in SAFE_API_TAGS:
            raise ValueError(f"API reference contains an unsafe HTML tag: {tag}")
        for name, value in attrs:
            self.validate_attribute(normalized, name, value)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_decl(self, decl: str) -> None:
        raise ValueError(f"API reference contains an unsafe declaration: {decl}")

    def handle_pi(self, data: str) -> None:
        raise ValueError("API reference contains an unsafe processing instruction")


SCRIPT_BLOCK_RE = re.compile(r"\s*<script\b[^>]*>.*?</script\s*>", re.IGNORECASE | re.DOTALL)
EVENT_HANDLER_RE = re.compile(
    r"\s+on[a-z][a-z0-9_:-]*\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s>]+)",
    re.IGNORECASE,
)


def validate_passive_api_fragment(fragment: str) -> None:
    parser = PassiveApiFragmentParser()
    parser.feed(fragment)
    parser.close()


def passivize_api_fragment(fragment: str) -> str:
    """Remove renderer-provided executable markup, then validate the result."""
    passive = SCRIPT_BLOCK_RE.sub("", fragment)
    passive = EVENT_HANDLER_RE.sub("", passive)
    validate_passive_api_fragment(passive)
    return passive
