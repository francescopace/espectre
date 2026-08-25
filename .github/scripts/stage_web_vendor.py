#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Stage pinned browser dependencies into the generated website tree."""

from __future__ import annotations

import shutil
from pathlib import Path


WEB_ROOT = Path(__file__).resolve().parents[2] / "docs" / "web"
NODE_MODULES = WEB_ROOT / "node_modules"
VENDOR_ROOT = WEB_ROOT / "vendor"
ESP_WEB_TOOLS_DEVICE_LINK_LABEL = (
    '${this._manifest.name==="ESPectre Native"?"Configure Device":"Visit Device"}'
)


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing npm dependency asset: {path}")
    return path


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(require(source), destination)


def customize_esp_web_tools(destination: Path) -> None:
    """Use the portal's Configure route wording for the Native device link."""
    install_dialog = next(destination.glob("install-dialog-*.js"), None)
    if install_dialog is None:
        raise FileNotFoundError("Missing ESP Web Tools install dialog bundle")
    source = install_dialog.read_text(encoding="utf-8")
    expected_occurrences = 2
    source_label = ">Visit Device</div>"
    if source.count(source_label) != expected_occurrences:
        raise ValueError("Unexpected ESP Web Tools device-link label count")
    install_dialog.write_text(
        source.replace(source_label, f">{ESP_WEB_TOOLS_DEVICE_LINK_LABEL}</div>"),
        encoding="utf-8",
    )


def stage_vendor() -> None:
    if VENDOR_ROOT.exists():
        shutil.rmtree(VENDOR_ROOT)

    esp_source = require(NODE_MODULES / "esp-web-tools" / "dist" / "web")
    esp_destination = VENDOR_ROOT / "esp-web-tools-10.4.0"
    shutil.copytree(esp_source, esp_destination)
    copy_file(NODE_MODULES / "esp-web-tools" / "LICENSE", esp_destination / "LICENSE")
    customize_esp_web_tools(esp_destination)

    qrcode_destination = VENDOR_ROOT / "qrcodejs-1.0.0"
    copy_file(NODE_MODULES / "qrcodejs" / "qrcode.min.js", qrcode_destination / "qrcode.min.js")
    copy_file(NODE_MODULES / "qrcodejs" / "LICENSE", qrcode_destination / "LICENSE")


def main() -> int:
    stage_vendor()
    print(f"staged browser dependencies under {VENDOR_ROOT.relative_to(WEB_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
