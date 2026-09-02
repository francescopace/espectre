#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Load and validate the canonical ESPectre website route manifest."""

from __future__ import annotations

import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = REPO_ROOT / "docs" / "web"
ROUTES_PATH = WEB_ROOT / "routes.json"
SITEMAP_NAMESPACE = "http://www.sitemaps.org/schemas/sitemap/0.9"


def content_path(route: dict[str, str]) -> str | None:
    path = route["staticPath"]
    if path == "/":
        return None
    return f"content{path.rstrip('/')}.html"


def output_path(route: dict[str, str]) -> str | None:
    path = route["staticPath"].strip("/")
    return path or None


def content_group(route: dict[str, str]) -> str:
    return route["contentGroup"]


def active_navigation(route: dict[str, str]) -> str:
    return route.get("group", route["name"])


def main_class(route: dict[str, str], *, static: bool) -> str:
    group = route.get("group")
    if group == "tools":
        suffix = " page-tool-static" if static else ""
        return f"page-narrow page-tool{suffix}"
    if group in {"guides", "sdk"}:
        return "page-narrow page-article"
    return "page-narrow"


def og_type(route: dict[str, str]) -> str:
    if route.get("group") == "tools":
        return "website"
    if not route.get("group") and route["name"] != "media":
        return "website"
    return "article"


def _validate_route(route: object, index: int) -> dict[str, str]:
    if not isinstance(route, dict):
        raise ValueError(f"Route {index} must be an object")
    required = ("name", "title", "description", "staticPath")
    for field in required:
        if not isinstance(route.get(field), str) or not route[field]:
            raise ValueError(f"Route {index} has no valid {field}")
    path = route["staticPath"]
    if not path.startswith("/") or not path.endswith("/"):
        raise ValueError(f"Route {route['name']} must use a canonical trailing-slash path")
    return route


def load_manifest(path: Path = ROUTES_PATH) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Route manifest must be an object")
    origin = manifest.get("siteOrigin")
    if not isinstance(origin, str) or not origin.startswith("https://") or origin.endswith("/"):
        raise ValueError("Route manifest siteOrigin must be an HTTPS origin without a trailing slash")

    routes = tuple(_validate_route(route, index) for index, route in enumerate(manifest.get("routes", ())))
    if not routes:
        raise ValueError("Route manifest contains no routes")
    names = [route["name"] for route in routes]
    paths = [route["staticPath"] for route in routes]
    if len(names) != len(set(names)):
        raise ValueError("Route manifest contains duplicate route names")
    if len(paths) != len(set(paths)):
        raise ValueError("Route manifest contains duplicate static paths")
    if names[0] != "home" or paths[0] != "/":
        raise ValueError("Route manifest must begin with the home route at /")

    by_name = {route["name"]: route for route in routes}
    for route in routes:
        group = route.get("group")
        if group and group not in by_name:
            raise ValueError(f"Route {route['name']} has unknown group {group}")

    navigation = manifest.get("navigation")
    if not isinstance(navigation, dict):
        raise ValueError("Route manifest navigation must be an object")
    for section in ("main", "footer"):
        members = navigation.get(section)
        if not isinstance(members, list) or not all(member in by_name for member in members):
            raise ValueError(f"Route manifest navigation.{section} contains an unknown route")

    content_groups = manifest.get("contentGroups")
    if not isinstance(content_groups, dict) or set(content_groups) != set(names):
        raise ValueError("Route manifest contentGroups must define every route exactly once")
    if not all(isinstance(group, str) and group for group in content_groups.values()):
        raise ValueError("Route manifest contentGroups values must be non-empty strings")
    routes = tuple({**route, "contentGroup": content_groups[route["name"]]} for route in routes)
    by_name = {route["name"]: route for route in routes}

    sdk_channels = manifest.get("sdkChannels")
    if not isinstance(sdk_channels, list):
        raise ValueError("Route manifest sdkChannels must be an array")
    sdk_channel_paths: set[str] = set()
    sdk_channel_names: set[str] = set()
    for sdk_channel in sdk_channels:
        if not isinstance(sdk_channel, dict):
            raise ValueError("Route manifest sdkChannels entries must be objects")
        sdk_channel_path = sdk_channel.get("path")
        if not isinstance(sdk_channel_path, str) or not sdk_channel_path.startswith("/") or not sdk_channel_path.endswith("/"):
            raise ValueError("Route manifest sdkChannels paths must be canonical trailing-slash paths")
        if sdk_channel_path in sdk_channel_paths or sdk_channel_path in paths:
            raise ValueError(f"Route manifest contains duplicate public path {sdk_channel_path}")
        sdk_channel_paths.add(sdk_channel_path)
        sdk_channel_name = sdk_channel.get("sdkChannel")
        if not isinstance(sdk_channel_name, str) or not re.fullmatch(r"[a-z0-9-]+", sdk_channel_name):
            raise ValueError(f"SDK channel {sdk_channel_path} has no valid channel name")
        if sdk_channel_name in sdk_channel_names:
            raise ValueError(f"Route manifest contains duplicate SDK channel {sdk_channel_name}")
        sdk_channel_names.add(sdk_channel_name)
        if not isinstance(sdk_channel.get("analyticsName"), str) or not sdk_channel["analyticsName"]:
            raise ValueError(f"SDK channel {sdk_channel_path} has no Analytics name")

    return {
        **manifest,
        "routes": routes,
        "by_name": by_name,
        "sdkChannels": tuple(sdk_channels),
    }


def static_routes(manifest: dict | None = None) -> tuple[dict[str, str], ...]:
    data = manifest or load_manifest()
    return tuple(route for route in data["routes"] if content_path(route) is not None)


def staged_sdk_channels(
    web_root: Path = WEB_ROOT,
    manifest: dict | None = None,
) -> tuple[dict[str, str], ...]:
    data = manifest or load_manifest()
    staged: list[dict[str, str]] = []
    for sdk_channel in data["sdkChannels"]:
        channel = sdk_channel["sdkChannel"]
        channel_dir = web_root / "artifacts" / "sdk" / channel
        manifest_path = channel_dir / f"sdk-manifest-{channel}.json"
        index_path = channel_dir / "index.html"
        present = (manifest_path.is_file(), index_path.is_file())
        if any(present) and not all(present):
            raise ValueError(
                f"Incomplete staged SDK channel {channel}: "
                f"manifest={present[0]}, index={present[1]}"
            )
        if all(present):
            staged.append(sdk_channel)
    return tuple(staged)
