#!/usr/bin/env python3
"""Stage pinned browser dependencies into the generated website tree."""

from __future__ import annotations

import shutil
from pathlib import Path


WEB_ROOT = Path(__file__).resolve().parents[2] / "docs" / "web"
NODE_MODULES = WEB_ROOT / "node_modules"
VENDOR_ROOT = WEB_ROOT / "vendor"


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing npm dependency asset: {path}")
    return path


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(require(source), destination)


def stage_vendor() -> None:
    if VENDOR_ROOT.exists():
        shutil.rmtree(VENDOR_ROOT)

    esp_source = require(NODE_MODULES / "esp-web-tools" / "dist" / "web")
    esp_destination = VENDOR_ROOT / "esp-web-tools-10.4.0"
    shutil.copytree(esp_source, esp_destination)
    copy_file(NODE_MODULES / "esp-web-tools" / "LICENSE", esp_destination / "LICENSE")

    mqtt_destination = VENDOR_ROOT / "mqtt-5.3.0"
    copy_file(NODE_MODULES / "mqtt" / "dist" / "mqtt.min.js", mqtt_destination / "mqtt.min.js")
    copy_file(NODE_MODULES / "mqtt" / "LICENSE.md", mqtt_destination / "LICENSE.md")

    qrcode_destination = VENDOR_ROOT / "qrcodejs-1.0.0"
    copy_file(NODE_MODULES / "qrcodejs" / "qrcode.min.js", qrcode_destination / "qrcode.min.js")
    copy_file(NODE_MODULES / "qrcodejs" / "LICENSE", qrcode_destination / "LICENSE")


def main() -> int:
    stage_vendor()
    print(f"staged browser dependencies under {VENDOR_ROOT.relative_to(WEB_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
