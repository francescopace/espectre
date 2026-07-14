#!/usr/bin/env python3
"""
ESPectre - Firmware Manifest Builder

Builds a JSON manifest for published firmware assets so the static web flasher
can resolve the single published image per frontend, channel, and chip.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


CHIP_METADATA = {
    "esp32": {"label": "ESP32", "family": "ESP32"},
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
    parser.add_argument("--url-prefix", help="Optional URL prefix used instead of GitHub Releases for web firmware assets")
    return parser.parse_args()


def parse_esphome_asset(filename: str, version_prefix: str) -> dict | None:
    if not filename.startswith(version_prefix) or not filename.endswith(".bin"):
        return None
    suffix = filename.removeprefix(version_prefix).removesuffix(".bin")
    if not suffix or "-" in suffix:
        return None

    return {
        "frontend": "esphome",
        "chip": suffix,
        "algorithm": "classic",
        "build_type": "factory",
    }


def parse_matter_asset(filename: str, version_prefix: str) -> dict | None:
    if not filename.startswith(version_prefix) or not filename.endswith(".bin"):
        return None
    suffix = filename.removeprefix(version_prefix).removesuffix(".bin")
    if not suffix or "-" in suffix:
        return None
    return {
        "frontend": "matter",
        "chip": suffix,
        "algorithm": None,
        "build_type": "factory",
    }


def parse_native_asset(filename: str, version_prefix: str) -> dict | None:
    if not filename.startswith(version_prefix) or not filename.endswith(".bin"):
        return None
    suffix = filename.removeprefix(version_prefix).removesuffix(".bin")
    if not suffix or "-" in suffix:
        return None
    return {
        "frontend": "native",
        "chip": suffix,
        "algorithm": None,
        "build_type": "factory",
    }


def build_artifact_url(filename: str, release_tag: str, url_prefix: str | None) -> str:
    if url_prefix:
        return f"{url_prefix.rstrip('/')}/{filename}"
    return f"https://github.com/francescopace/espectre/releases/download/{release_tag}/{filename}"


def build_manifest(args: argparse.Namespace) -> dict:
    firmware_dir = Path(args.firmware_dir)
    output_path = Path(args.output)

    if args.channel == "stable":
        esphome_prefix = f"espectre-{args.version}-"
        native_prefix = f"espectre-native-{args.version}-"
        matter_prefix = f"espectre-matter-{args.version}-"
    else:
        esphome_prefix = "espectre-snapshot-"
        native_prefix = "espectre-native-snapshot-"
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
                "artifacts": [],
            },
            "native": {
                "label": "Native",
                "post_flash": "Provision Wi-Fi and MQTT over BLE, then connect through the native BLE or MQTT control surface.",
                "notes": [
                    "The native frontend preserves the current custom GATT protocol, but it is not limited to any single client implementation."
                ],
                "artifacts": [],
            },
        },
    }

    for asset_path in sorted(firmware_dir.glob("*.bin")):
        filename = asset_path.name
        parsed = parse_matter_asset(filename, matter_prefix)
        if parsed is None:
            parsed = parse_native_asset(filename, native_prefix)
        if parsed is None:
            parsed = parse_esphome_asset(filename, esphome_prefix)
        if parsed is None:
            continue

        chip_meta = CHIP_METADATA.get(parsed["chip"])
        if chip_meta is None:
            if parsed["chip"] == "esp32s2":
                continue
            raise ValueError(f"Unknown chip in firmware filename: {filename}")

        artifact = {
            "chip": parsed["chip"],
            "chip_label": chip_meta["label"],
            "chip_family": chip_meta["family"],
            "algorithm": parsed["algorithm"],
            "build_type": parsed["build_type"],
            "filename": filename,
            "url": build_artifact_url(filename, args.release_tag, args.url_prefix),
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
