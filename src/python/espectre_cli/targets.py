"""Static target mappings for the ESPectre repository CLI."""

from __future__ import annotations

from pathlib import Path

from .common import REPO_ROOT

ESPHOME_CONFIGS = {
    "esp32": {
        "dev": REPO_ROOT / "examples" / "espectre-esp32-dev.yaml",
        "release": REPO_ROOT / "examples" / "espectre-esp32.yaml",
    },
    "c3": {
        "dev": REPO_ROOT / "examples" / "espectre-c3-dev.yaml",
        "release": REPO_ROOT / "examples" / "espectre-c3.yaml",
    },
    "c5": {
        "dev": REPO_ROOT / "examples" / "espectre-c5-dev.yaml",
        "release": REPO_ROOT / "examples" / "espectre-c5.yaml",
    },
    "c6": {
        "dev": REPO_ROOT / "examples" / "espectre-c6-dev.yaml",
        "release": REPO_ROOT / "examples" / "espectre-c6.yaml",
    },
    "s3": {
        "dev": REPO_ROOT / "examples" / "espectre-s3-dev.yaml",
        "release": REPO_ROOT / "examples" / "espectre-s3.yaml",
    },
}

IDF_FRONTENDS = {
    "native": {
        "app_dir": REPO_ROOT / "src" / "cpp" / "frontend" / "native" / "app",
        "targets": {
            "esp32": "esp32",
            "c3": "esp32c3",
            "c5": "esp32c5",
            "c6": "esp32c6",
            "s3": "esp32s3",
        },
    },
    "matter": {
        "app_dir": REPO_ROOT / "src" / "cpp" / "frontend" / "matter" / "app",
        "targets": {
            "esp32": "esp32",
            "c3": "esp32c3",
            "c5": "esp32c5",
            "c6": "esp32c6",
            "s3": "esp32s3",
        },
    },
    "streamer": {
        "app_dir": REPO_ROOT / "src" / "cpp" / "frontend" / "streamer" / "app",
        "targets": {
            "esp32": "esp32",
            "c3": "esp32c3",
            "c5": "esp32c5",
            "c6": "esp32c6",
            "s3": "esp32s3",
        },
    },
}


def resolve_esphome_config(chip: str | None, dev: bool, config: str | None) -> Path:
    """Resolve the ESPHome config file for a chip or explicit override."""
    if config:
        path = Path(config)
        if not path.is_absolute():
            path = REPO_ROOT / path
        return path
    if not chip:
        raise ValueError("--chip is required unless --config is provided")
    try:
        key = "dev" if dev else "release"
        return ESPHOME_CONFIGS[chip][key]
    except KeyError as exc:
        raise ValueError(f"Unsupported ESPHome chip: {chip}") from exc


def resolve_idf_target(frontend: str, chip: str) -> tuple[Path, str]:
    """Return (app_dir, idf_target) for a supported frontend/chip pair."""
    try:
        cfg = IDF_FRONTENDS[frontend]
        return cfg["app_dir"], cfg["targets"][chip]
    except KeyError as exc:
        raise ValueError(f"Unsupported {frontend} target: {chip}") from exc
