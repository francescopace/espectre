#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Check that Pages serves the files from the verified deployment archive."""

from __future__ import annotations

import argparse
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath
from urllib.parse import quote, urlparse
from uuid import uuid4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--path", required=True, action="append", dest="paths")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--interval", type=float, default=10)
    args = parser.parse_args()
    if args.timeout <= 0 or args.interval <= 0:
        parser.error("timeout and interval must be positive")
    base = urlparse(args.base_url)
    if base.scheme not in ("http", "https") or not base.netloc or base.query or base.fragment:
        parser.error("base-url must be an HTTP(S) URL without a query or fragment")
    return args


def archive_files(archive_path: Path, paths: list[str]) -> dict[str, bytes]:
    files = {}
    with tarfile.open(archive_path) as archive:
        for path in paths:
            if PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts:
                raise ValueError(f"Expected a relative website path: {path}")
            member = archive.getmember(f"./{path}")
            if not member.isfile() or member.size > 2 * 1024 * 1024:
                raise ValueError(f"Expected a regular metadata file of at most 2 MiB: {path}")
            with archive.extractfile(member) as source:
                files[path] = source.read()
    return files


def verify(base_url: str, files: dict[str, bytes], *, timeout: float, interval: float) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            for path, expected in files.items():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Deployment verification deadline reached")
                url = f"{base_url.rstrip('/')}/{quote(path)}?deployment-check={uuid4().hex}"
                request = urllib.request.Request(url, headers={"Cache-Control": "no-cache"})
                with urllib.request.urlopen(request, timeout=min(15, remaining)) as response:
                    actual = response.read(len(expected) + 1)
                if actual != expected:
                    raise ValueError(f"The deployed file differs from the verified archive: {path}")
            print(f"Verified {len(files)} deployed files against the Pages archive.", flush=True)
            return
        except (urllib.error.URLError, TimeoutError, ValueError) as error:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(f"Pages is not serving the verified deployment: {error}") from error
            print(f"Waiting for the deployed files: {error}", flush=True)
            time.sleep(min(interval, remaining))


def main() -> None:
    args = parse_args()
    verify(args.base_url, archive_files(args.archive, args.paths),
           timeout=args.timeout, interval=args.interval)


if __name__ == "__main__":
    main()
