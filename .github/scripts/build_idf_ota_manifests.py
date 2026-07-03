#!/usr/bin/env python3
"""
ESPectre - IDF OTA manifest builder.

Generates simple HTTPS OTA manifests for Native and Streamer OTA payloads. Each
manifest is a small JSON document consumed by the shared HTTPS OTA service.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build OTA manifests for Native and Streamer artifacts.")
    parser.add_argument("--firmware-dir", required=True, help="Directory containing built firmware assets")
    parser.add_argument("--release-tag", required=True, help="GitHub release tag used to download the assets")
    parser.add_argument("--version", required=True, help="Version string published in the OTA manifest")
    return parser.parse_args()


def parse_ota_asset(filename: str) -> tuple[str, str] | None:
    for frontend in ("native", "streamer"):
        prefix = f"espectre-{frontend}-"
        if filename.startswith(prefix) and filename.endswith("-ota.bin"):
            chip = filename.removeprefix(prefix).removesuffix("-ota.bin").split("-")[-1]
            return frontend, chip
    return None


def build_manifests(args: argparse.Namespace) -> list[Path]:
    firmware_dir = Path(args.firmware_dir)
    created: list[Path] = []

    for asset_path in sorted(firmware_dir.glob("espectre-*-ota.bin")):
        parsed = parse_ota_asset(asset_path.name)
        if parsed is None:
            continue
        frontend, chip = parsed
        manifest = {
            "schema_version": 1,
            "frontend": frontend,
            "chip": chip,
            "version": args.version,
            "image_url": f"https://github.com/francescopace/espectre/releases/download/{args.release_tag}/{asset_path.name}",
        }
        output_path = asset_path.with_suffix(".json")
        output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        created.append(output_path)
    return created


def main() -> int:
    args = parse_args()
    build_manifests(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
