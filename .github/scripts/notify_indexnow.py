#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Notify IndexNow after a successful GitHub Pages deployment."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse


DEFAULT_SITEMAP = Path(__file__).resolve().with_name("sitemap.template.xml")
INDEXNOW_ENDPOINT = "https://api.indexnow.org/indexnow"
SITE_HOST = "espectre.dev"
INDEXNOW_KEY = "1a2e73ccf9558a06830546c288699e0c"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Notify IndexNow from the ESPectre sitemap.")
    parser.add_argument("--sitemap", default=str(DEFAULT_SITEMAP))
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def sitemap_urls(path: Path) -> list[str]:
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    urls = [
        (element.text or "").strip()
        for element in root.findall("{http://www.sitemaps.org/schemas/sitemap/0.9}url/{http://www.sitemaps.org/schemas/sitemap/0.9}loc")
    ]
    urls = [url for url in urls if url]
    if not urls:
        raise ValueError(f"No URLs found in sitemap: {path}")
    invalid = [
        url
        for url in urls
        if (parsed := urlparse(url)).scheme != "https" or parsed.hostname != SITE_HOST
    ]
    if invalid:
        raise ValueError(f"Sitemap contains URLs outside https://{SITE_HOST}: {invalid}")
    return urls


def notify(
    urls: list[str],
    *,
    attempts: int = 3,
    timeout: float = 30.0,
    request_fn: Callable[..., object] = urllib.request.urlopen,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    payload = json.dumps(
        {
            "host": SITE_HOST,
            "key": INDEXNOW_KEY,
            "keyLocation": f"https://{SITE_HOST}/{INDEXNOW_KEY}.txt",
            "urlList": urls,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        INDEXNOW_ENDPOINT,
        data=payload,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = request_fn(request, timeout=timeout)
            with response:
                status = getattr(response, "status", 200)
                if not 200 <= status < 300:
                    raise RuntimeError(f"IndexNow returned HTTP {status}")
            return
        except (OSError, RuntimeError, urllib.error.URLError) as error:
            last_error = error
            if attempt < attempts:
                sleep_fn(float(2 ** (attempt - 1)))
    raise RuntimeError(f"IndexNow notification failed after {attempts} attempts") from last_error


def main() -> int:
    args = parse_args()
    urls = sitemap_urls(Path(args.sitemap))
    notify(urls, attempts=args.attempts, timeout=args.timeout)
    print(f"Notified IndexNow with {len(urls)} URLs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
