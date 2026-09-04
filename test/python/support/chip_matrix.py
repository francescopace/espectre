# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Canonical chip matrices derived from the production CLI registries."""

from __future__ import annotations

from src.python.espectre_cli.common import CHIP_CHOICES, MICRO_CHIP_CHOICES
from src.python.espectre_cli.targets import ESPHOME_CONFIGS, IDF_FRONTENDS


def chip_label(chip: str) -> str:
    """Return the short chip label used by dataset metadata and test IDs."""
    normalized = str(chip).strip().lower().removeprefix("esp32-")
    return "ESP32" if normalized == "esp32" else normalized.upper()


DETECTION_CHIPS = tuple(chip_label(chip) for chip in CHIP_CHOICES)
MICROPYTHON_CHIPS = tuple(chip_label(chip) for chip in MICRO_CHIP_CHOICES)
ESPHOME_CHIPS = tuple(chip_label(chip) for chip in ESPHOME_CONFIGS)
NATIVE_CHIPS = tuple(chip_label(chip) for chip in IDF_FRONTENDS["native"]["targets"])
MATTER_CHIPS = tuple(chip_label(chip) for chip in IDF_FRONTENDS["matter"]["targets"])


def chip_key(chip: str) -> str:
    """Return the production CLI key for one short chip label."""
    label = chip_label(chip)
    return "esp32" if label == "ESP32" else label.lower()
