# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - CLI Targets

Static target mappings for the ESPectre repository CLI.

Author: Francesco Pace <francesco.pace@gmail.com>
"""

from __future__ import annotations

from pathlib import Path

from .common import REPO_ROOT

FRONTEND_DIR = REPO_ROOT / "src" / "cpp" / "frontend"
ESPHOME_EXAMPLES_DIR = FRONTEND_DIR / "esphome" / "examples"

ESPHOME_CONFIGS = {
    "esp32": ESPHOME_EXAMPLES_DIR / "espectre-esp32.yaml",
    "c3": ESPHOME_EXAMPLES_DIR / "espectre-c3.yaml",
    "c5": ESPHOME_EXAMPLES_DIR / "espectre-c5.yaml",
    "c6": ESPHOME_EXAMPLES_DIR / "espectre-c6.yaml",
    "s3": ESPHOME_EXAMPLES_DIR / "espectre-s3.yaml",
    "s2": ESPHOME_EXAMPLES_DIR / "espectre-s2.yaml",
}

IDF_FRONTENDS = {
    "native": {
        "app_dir": FRONTEND_DIR / "native" / "app",
        "targets": {
            "esp32": "esp32",
            "c3": "esp32c3",
            "c5": "esp32c5",
            "c6": "esp32c6",
            "s3": "esp32s3",
            "s2": "esp32s2",
        },
    },
    "matter": {
        "app_dir": FRONTEND_DIR / "matter" / "app",
        "targets": {
            "esp32": "esp32",
            "c3": "esp32c3",
            "c5": "esp32c5",
            "c6": "esp32c6",
            "s3": "esp32s3",
        },
    },
}


def resolve_esphome_config(chip: str | None, config: str | None) -> Path:
    """Resolve the ESPHome config file for a chip or explicit override."""
    if config:
        path = Path(config)
        if not path.is_absolute():
            path = REPO_ROOT / path
        return path
    if not chip:
        raise ValueError("--chip is required unless --config is provided")
    try:
        return ESPHOME_CONFIGS[chip]
    except KeyError as exc:
        raise ValueError(f"Unsupported ESPHome chip: {chip}") from exc


def resolve_idf_target(frontend: str, chip: str) -> tuple[Path, str]:
    """Return (app_dir, idf_target) for a supported frontend/chip pair."""
    try:
        cfg = IDF_FRONTENDS[frontend]
        return cfg["app_dir"], cfg["targets"][chip]
    except KeyError as exc:
        raise ValueError(f"Unsupported {frontend} target: {chip}") from exc
