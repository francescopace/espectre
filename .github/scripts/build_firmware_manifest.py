#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - Firmware Manifest Builder

Build the published firmware manifest for web flashing.

Author: Francesco Pace <francesco.pace@gmail.com>
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
    parser.add_argument(
        "--channel",
        choices=("stable", "main", "develop"),
        required=True,
        help="Release channel exposed to the web UI or snapshot release metadata",
    )
    parser.add_argument("--version", required=True, help="Human-readable version label")
    parser.add_argument("--release-tag", required=True, help="GitHub release tag used to download the assets")
    parser.add_argument("--commit", help="Optional source commit SHA for snapshot builds")
    parser.add_argument("--url-prefix", help="Optional URL prefix used instead of GitHub Releases for web firmware assets")
    parser.add_argument(
        "--native-ota-manifest-dir",
        help="Optional directory where per-chip native OTA manifests are written",
    )
    parser.add_argument(
        "--require-complete-matrix",
        action="store_true",
        help="Fail unless all 15 factory and 5 Native OTA images are present",
    )
    return parser.parse_args()


def parse_published_asset(
    filename: str,
    version_prefix: str,
    *,
    frontend: str,
    algorithm: str | None,
    supports_ota: bool,
) -> dict | None:
    if not filename.startswith(version_prefix) or not filename.endswith(".bin"):
        return None
    suffix = filename.removeprefix(version_prefix).removesuffix(".bin")
    if not suffix:
        return None

    parts = suffix.split("-")
    if len(parts) == 1:
        chip = parts[0]
        build_type = "factory"
    elif supports_ota and len(parts) == 2 and parts[1] == "ota":
        chip = parts[0]
        build_type = "ota"
    else:
        return None

    return {
        "frontend": frontend,
        "chip": chip,
        "algorithm": algorithm,
        "build_type": build_type,
    }


def parse_esphome_asset(filename: str, version_prefix: str) -> dict | None:
    return parse_published_asset(
        filename,
        version_prefix,
        frontend="esphome",
        algorithm="classic",
        supports_ota=False,
    )


def parse_matter_asset(filename: str, version_prefix: str) -> dict | None:
    return parse_published_asset(
        filename,
        version_prefix,
        frontend="matter",
        algorithm=None,
        supports_ota=False,
    )


def parse_native_asset(filename: str, version_prefix: str) -> dict | None:
    return parse_published_asset(
        filename,
        version_prefix,
        frontend="native",
        algorithm=None,
        supports_ota=True,
    )


def build_artifact_url(filename: str, release_tag: str, url_prefix: str | None) -> str:
    if url_prefix:
        return f"{url_prefix.rstrip('/')}/{filename}"
    return f"https://github.com/francescopace/espectre/releases/download/{release_tag}/{filename}"


def compliance_artifacts(asset_path: Path, release_tag: str, url_prefix: str | None) -> list[dict]:
    companions = (
        ("spdx-sbom", "-sbom.spdx.json"),
        ("notices", "-THIRD_PARTY_NOTICES.txt"),
        ("license-archive", "-third-party-licenses.zip"),
    )
    artifacts = []
    for kind, suffix in companions:
        path = asset_path.with_name(f"{asset_path.stem}{suffix}")
        if not path.is_file():
            continue
        artifacts.append(
            {
                "kind": kind,
                "filename": path.name,
                "url": build_artifact_url(path.name, release_tag, url_prefix),
            }
        )
    return artifacts


def validate_complete_matrix(manifest: dict) -> None:
    expected = set()
    for chip in CHIP_METADATA:
        expected.add(("matter", chip, "factory"))
        expected.add(("esphome", chip, "factory"))
        expected.add(("native", chip, "factory"))
        expected.add(("native", chip, "ota"))

    entries = [
        (frontend, artifact["chip"], artifact["build_type"])
        for frontend, metadata in manifest["frontends"].items()
        for artifact in metadata["artifacts"]
    ]
    actual = set(entries)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected or len(entries) != len(actual):
        raise ValueError(
            f"Invalid firmware matrix: missing={missing}, unexpected={unexpected}, "
            f"duplicates={len(entries) - len(actual)}"
        )

    required_compliance = {"spdx-sbom", "notices", "license-archive"}
    incomplete_compliance = []
    for frontend, metadata in manifest["frontends"].items():
        for artifact in metadata["artifacts"]:
            kinds = {entry["kind"] for entry in artifact["compliance"]}
            if kinds != required_compliance:
                incomplete_compliance.append(
                    (frontend, artifact["filename"], sorted(required_compliance - kinds))
                )
    if incomplete_compliance:
        raise ValueError(f"Missing firmware compliance artifacts: {incomplete_compliance}")


def build_manifest(args: argparse.Namespace) -> dict:
    firmware_dir = Path(args.firmware_dir)
    output_path = Path(args.output)

    if args.channel == "stable":
        esphome_prefix = f"espectre-{args.version}-"
        native_prefix = f"espectre-native-{args.version}-"
        matter_prefix = f"espectre-matter-{args.version}-"
    elif args.channel == "main":
        esphome_prefix = "espectre-snapshot-"
        native_prefix = "espectre-native-snapshot-"
        matter_prefix = "espectre-matter-snapshot-"
    else:
        esphome_prefix = "espectre-snapshot-dev-"
        native_prefix = "espectre-native-snapshot-dev-"
        matter_prefix = "espectre-matter-snapshot-dev-"

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
            "compliance": compliance_artifacts(asset_path, args.release_tag, args.url_prefix),
        }
        manifest["frontends"][parsed["frontend"]]["artifacts"].append(artifact)

    if getattr(args, "require_complete_matrix", False):
        validate_complete_matrix(manifest)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def build_native_ota_manifests(manifest: dict, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for artifact in manifest["frontends"]["native"]["artifacts"]:
        if artifact["build_type"] != "ota":
            continue
        payload = {
            "schema_version": 1,
            "frontend": "native",
            "chip": artifact["chip"],
            "version": manifest["version"],
            "release_tag": manifest["release_tag"],
            "image_url": artifact["url"],
        }
        path = output_dir / f"espectre-native-ota-{artifact['chip']}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def main() -> int:
    args = parse_args()
    manifest = build_manifest(args)
    if args.native_ota_manifest_dir:
        build_native_ota_manifests(manifest, Path(args.native_ota_manifest_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
