# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Machine-readable firmware build artifact metadata."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from .esptool_runner import read_idf_flash_metadata

BUILD_METADATA_SCHEMA_VERSION = 1


def publish_idf_flash_artifacts(build_dir: Path, destination: Path) -> None:
    """Publish a complete flash image without replacing files used by an active flash."""
    metadata = read_idf_flash_metadata(build_dir)
    if not metadata["flash_files"]:
        raise ValueError(f"No flash files in {build_dir / 'flasher_args.json'}")
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".publish-", dir=destination) as temporary:
        staging = Path(temporary)
        flash_files = {}
        for index, (address, filename) in enumerate(metadata["flash_files"].items()):
            int(address, 0)
            digest = hashlib.sha256()
            staged_file = staging / str(index)
            with (build_dir / filename).open("rb") as source, staged_file.open("wb") as output:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
                    output.write(chunk)
            published_name = f"{digest.hexdigest()}.bin"
            os.replace(staged_file, staging / published_name)
            flash_files[address] = published_name

        # Keep older content-addressed files so a flash that already read the
        # previous metadata can finish. Explicit --clean-all removes them.
        for filename in set(flash_files.values()):
            published_file = destination / filename
            if not published_file.exists():
                os.replace(staging / filename, published_file)
        staged_metadata = staging / "flasher_args.json"
        staged_metadata.write_text(
            json.dumps({"write_flash_args": metadata["write_flash_args"], "flash_files": flash_files}),
            encoding="utf-8",
        )
        # The manifest is the commit point: readers see either complete image.
        os.replace(staged_metadata, destination / staged_metadata.name)


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
