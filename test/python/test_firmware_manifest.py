"""Tests for the published firmware manifest builder."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / ".github" / "scripts" / "build_firmware_manifest.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_firmware_manifest", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parsers_accept_only_published_base_images():
    module = _load_module()

    assert module.parse_esphome_asset("espectre-3.0.0-esp32c3.bin", "espectre-3.0.0-") is not None
    assert module.parse_esphome_asset("espectre-3.0.0-esp32c3-ml.bin", "espectre-3.0.0-") is None
    assert module.parse_esphome_asset("espectre-3.0.0-esp32c3-ota.bin", "espectre-3.0.0-") is None

    assert module.parse_matter_asset("espectre-matter-3.0.0-esp32c3.bin", "espectre-matter-3.0.0-") is not None
    assert module.parse_matter_asset("espectre-matter-3.0.0-esp32c3-ota.bin", "espectre-matter-3.0.0-") is None

    assert module.parse_native_asset("espectre-native-3.0.0-esp32c3.bin", "espectre-native-3.0.0-") is not None
    assert module.parse_native_asset("espectre-native-3.0.0-esp32c3-ota.bin", "espectre-native-3.0.0-") is None


def test_manifest_contains_only_fifteen_supported_images(tmp_path):
    module = _load_module()
    chips = ("esp32", "esp32s3", "esp32c3", "esp32c5", "esp32c6")

    for chip in chips:
        (tmp_path / f"espectre-3.0.0-{chip}.bin").write_bytes(b"esphome")
        (tmp_path / f"espectre-matter-3.0.0-{chip}.bin").write_bytes(b"matter")
        (tmp_path / f"espectre-native-3.0.0-{chip}.bin").write_bytes(b"native")

    (tmp_path / "espectre-3.0.0-esp32c3-ml.bin").write_bytes(b"unused")
    (tmp_path / "espectre-native-3.0.0-esp32c3-ota.bin").write_bytes(b"unused")
    (tmp_path / "espectre-3.0.0-esp32s2.bin").write_bytes(b"legacy")

    output = tmp_path / "firmware-manifest-3.0.0.json"
    manifest = module.build_manifest(
        argparse.Namespace(
            firmware_dir=str(tmp_path),
            output=str(output),
            channel="stable",
            version="3.0.0",
            release_tag="3.0.0",
            commit=None,
            url_prefix=None,
        )
    )

    artifacts = [artifact for frontend in manifest["frontends"].values() for artifact in frontend["artifacts"]]
    assert len(artifacts) == 15
    assert {artifact["chip"] for artifact in artifacts} == set(chips)
    assert all(artifact["build_type"] == "factory" for artifact in artifacts)
    assert output.is_file()
