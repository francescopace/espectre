# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Enforce the C++ dependency direction at source level."""

from __future__ import annotations

import os
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CPP_ROOT = REPO_ROOT / "src" / "cpp"
INCLUDE_PATTERN = re.compile(r'^\s*#include\s+"([^"]+)"', re.MULTILINE)
SOURCE_SUFFIXES = {".cpp", ".h"}
IGNORED_PARTS = {".esphome", "managed_components"}


def is_first_party_source(path: Path) -> bool:
    """Exclude generated build trees and managed dependencies."""
    relative = path.relative_to(CPP_ROOT)
    return not any(
        part in IGNORED_PARTS or part == "build" or part.startswith("build-")
        for part in relative.parts
    )


def first_party_sources() -> list[Path]:
    """Return maintained sources from the three architectural layers."""
    sources: list[Path] = []
    for layer_root in (CPP_ROOT / "core", CPP_ROOT / "runtime", CPP_ROOT / "frontend"):
        for current_root, directories, filenames in os.walk(layer_root):
            directories[:] = [
                directory
                for directory in directories
                if directory not in IGNORED_PARTS
                and directory != "build"
                and not directory.startswith("build-")
            ]
            sources.extend(
                Path(current_root) / filename
                for filename in filenames
                if Path(filename).suffix in SOURCE_SUFFIXES
            )
    return sorted(sources)


def layer(path: Path) -> int:
    """Order layers from lowest-level domain code to concrete frontends."""
    parts = path.relative_to(CPP_ROOT).parts
    if parts[0] == "core":
        return 0
    if parts[0] == "runtime":
        return 1
    if parts[0] == "frontend":
        return 2
    raise AssertionError(f"unclassified C++ source: {path}")


def frontend_name(path: Path) -> str | None:
    """Return the concrete frontend name."""
    parts = path.relative_to(CPP_ROOT).parts
    if len(parts) >= 2 and parts[0] == "frontend":
        return parts[1]
    return None


def test_cpp_dependencies_only_point_to_same_or_lower_layers() -> None:
    """Reject Core -> Runtime/Frontend and Runtime -> Frontend dependencies."""
    sources = first_party_sources()
    headers_by_name: dict[str, list[Path]] = {}
    for header in (path for path in sources if path.suffix == ".h"):
        headers_by_name.setdefault(header.name, []).append(header)

    violations: list[str] = []
    for source in sources:
        for include in INCLUDE_PATTERN.findall(source.read_text(encoding="utf-8")):
            candidates = [source.parent / include, CPP_ROOT / include]
            if "/" not in include:
                candidates.extend(headers_by_name.get(include, []))
            target = next(
                (candidate.resolve() for candidate in candidates if candidate.is_file()),
                None,
            )
            if (
                target is None
                or not target.is_relative_to(CPP_ROOT)
                or not is_first_party_source(target)
            ):
                continue

            source_frontend = frontend_name(source)
            target_frontend = frontend_name(target)
            crosses_frontends = (
                source_frontend is not None
                and target_frontend is not None
                and source_frontend != target_frontend
            )
            if layer(target) > layer(source) or crosses_frontends:
                violations.append(
                    f"{source.relative_to(CPP_ROOT)} includes {target.relative_to(CPP_ROOT)}"
                )

    assert not violations, (
        "C++ dependencies point upward or across frontends:\n" + "\n".join(violations)
    )
