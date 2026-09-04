# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Machine-readable firmware build artifact metadata."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


BUILD_METADATA_SCHEMA_VERSION = 1


def build_artifact_metadata(
    *,
    frontend: str,
    chip: str | None,
    artifact: Path,
) -> dict[str, object]:
    """Return stable metadata for one completed firmware build."""
    resolved = artifact.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"firmware build artifact not found: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as firmware:
        for chunk in iter(lambda: firmware.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "artifact": str(resolved),
        "chip": chip,
        "command": "build",
        "firmware_sha256": digest.hexdigest(),
        "firmware_size_bytes": resolved.stat().st_size,
        "frontend": frontend,
        "schema_version": BUILD_METADATA_SCHEMA_VERSION,
    }


def print_build_artifact_metadata(
    *,
    frontend: str,
    chip: str | None,
    artifact: Path,
) -> None:
    """Print one final JSON object for a successful build."""
    print(
        json.dumps(
            build_artifact_metadata(
                frontend=frontend,
                chip=chip,
                artifact=artifact,
            ),
            sort_keys=True,
        )
    )
