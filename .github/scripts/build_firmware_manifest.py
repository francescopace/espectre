#!/usr/bin/env python3
"""
ESPectre - Firmware Manifest Builder

Builds a JSON manifest for published firmware assets so the static web flasher
can resolve the correct binary per frontend, channel, chip, and algorithm.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


CHIP_METADATA = {
    "esp32": {"label": "ESP32", "family": "ESP32"},
    "esp32s2": {"label": "ESP32-S2", "family": "ESP32-S2"},
    "esp32s3": {"label": "ESP32-S3", "family": "ESP32-S3"},
    "esp32c3": {"label": "ESP32-C3", "family": "ESP32-C3"},
    "esp32c5": {"label": "ESP32-C5", "family": "ESP32-C5"},
    "esp32c6": {"label": "ESP32-C6", "family": "ESP32-C6"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an ESPectre firmware manifest.")
    parser.add_argument("--firmware-dir", required=True, help="Directory containing built firmware assets")
    parser.add_argument("--output", required=True, help="Output manifest path")
    parser.add_argument("--channel", choices=("stable", "main"), required=True, help="Release channel exposed to the web UI")
    parser.add_argument("--version", required=True, help="Human-readable version label")
    parser.add_argument("--release-tag", required=True, help="GitHub release tag used to download the assets")
    parser.add_argument("--commit", help="Optional source commit SHA for snapshot builds")
    return parser.parse_args()


def parse_esphome_asset(filename: str, version_prefix: str) -> dict | None:
    if not filename.startswith(version_prefix) or not filename.endswith(".bin"):
        return None
    suffix = filename.removeprefix(version_prefix).removesuffix(".bin")
    parts = suffix.split("-")
    if not parts:
        return None

    chip = parts[0]
    algorithm = "mvs"
    build_type = "factory"

    if len(parts) >= 2 and parts[1] == "ml":
        algorithm = "ml"
        if len(parts) >= 3 and parts[2] == "ota":
            build_type = "ota"
    elif len(parts) >= 2 and parts[1] == "ota":
        build_type = "ota"

    return {
        "frontend": "esphome",
        "chip": chip,
        "algorithm": algorithm,
        "build_type": build_type,
    }


def parse_matter_asset(filename: str, version_prefix: str) -> dict | None:
    if not filename.startswith(version_prefix) or not filename.endswith(".bin"):
        return None
    chip = filename.removeprefix(version_prefix).removesuffix(".bin")
    return {
        "frontend": "matter",
        "chip": chip,
        "algorithm": None,
        "build_type": "factory",
    }


def parse_ble_asset(filename: str, version_prefix: str) -> dict | None:
    if not filename.startswith(version_prefix) or not filename.endswith(".bin"):
        return None
    chip = filename.removeprefix(version_prefix).removesuffix(".bin")
    return {
        "frontend": "ble",
        "chip": chip,
        "algorithm": None,
        "build_type": "factory",
    }


def build_manifest(args: argparse.Namespace) -> dict:
    firmware_dir = Path(args.firmware_dir)
    output_path = Path(args.output)

    if args.channel == "stable":
        esphome_prefix = f"espectre-{args.version}-"
        ble_prefix = f"espectre-ble-{args.version}-"
        matter_prefix = f"espectre-matter-{args.version}-"
    else:
        esphome_prefix = "espectre-snapshot-"
        ble_prefix = "espectre-ble-snapshot-"
        matter_prefix = "espectre-matter-snapshot-"

    manifest = {
        "schema_version": 1,
        "channel": args.channel,
        "version": args.version,
        "release_tag": args.release_tag,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "commit": args.commit,
        "frontends": {
            "esphome": {
                "label": "ESPHome",
                "post_flash": "Configure Wi-Fi via ESPHome/Home Assistant, then wait for native API discovery.",
                "artifacts": [],
            },
            "matter": {
                "label": "Matter",
                "post_flash": "Commission the device with a Matter controller after reboot.",
                "notes": [
                    "ESP32-S2 is not supported by the current Matter frontend because the implementation relies on BLE commissioning."
                ],
                "artifacts": [],
            },
            "ble": {
                "label": "BLE",
                "post_flash": "This firmware is a standalone generic BLE frontend. Configure Wi-Fi credentials in sdkconfig before building or use a preconfigured binary, then connect from your custom BLE client or web integration.",
                "notes": [
                    "The BLE frontend preserves the current custom GATT protocol, but it is not limited to any single client implementation."
                ],
                "artifacts": [],
            },
        },
    }

    for asset_path in sorted(firmware_dir.glob("*.bin")):
        filename = asset_path.name
        parsed = parse_matter_asset(filename, matter_prefix)
        if parsed is None:
            parsed = parse_ble_asset(filename, ble_prefix)
        if parsed is None:
            parsed = parse_esphome_asset(filename, esphome_prefix)
        if parsed is None:
            continue

        chip_meta = CHIP_METADATA.get(parsed["chip"])
        if chip_meta is None:
            raise ValueError(f"Unknown chip in firmware filename: {filename}")

        artifact = {
            "chip": parsed["chip"],
            "chip_label": chip_meta["label"],
            "chip_family": chip_meta["family"],
            "algorithm": parsed["algorithm"],
            "build_type": parsed["build_type"],
            "filename": filename,
            "url": f"https://github.com/francescopace/espectre/releases/download/{args.release_tag}/{filename}",
        }
        manifest["frontends"][parsed["frontend"]]["artifacts"].append(artifact)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    args = parse_args()
    build_manifest(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
