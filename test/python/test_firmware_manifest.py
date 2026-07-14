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


def test_parsers_accept_only_published_factory_and_native_ota_images():
    module = _load_module()

    assert module.parse_esphome_asset("espectre-3.0.0-esp32c3.bin", "espectre-3.0.0-") is not None
    assert module.parse_esphome_asset("espectre-3.0.0-esp32c3-ml.bin", "espectre-3.0.0-") is None
    assert module.parse_esphome_asset("espectre-3.0.0-esp32c3-ota.bin", "espectre-3.0.0-") is None

    assert module.parse_matter_asset("espectre-matter-3.0.0-esp32c3.bin", "espectre-matter-3.0.0-") is not None
    assert module.parse_matter_asset("espectre-matter-3.0.0-esp32c3-ota.bin", "espectre-matter-3.0.0-") is None

    assert module.parse_native_asset("espectre-native-3.0.0-esp32c3.bin", "espectre-native-3.0.0-") is not None
    native_ota = module.parse_native_asset("espectre-native-3.0.0-esp32c3-ota.bin", "espectre-native-3.0.0-")
    assert native_ota is not None
    assert native_ota["build_type"] == "ota"


def test_manifest_contains_fifteen_factory_and_five_native_ota_images(tmp_path):
    module = _load_module()
    chips = ("esp32", "esp32s3", "esp32c3", "esp32c5", "esp32c6")

    for chip in chips:
        (tmp_path / f"espectre-3.0.0-{chip}.bin").write_bytes(b"esphome")
        (tmp_path / f"espectre-matter-3.0.0-{chip}.bin").write_bytes(b"matter")
        (tmp_path / f"espectre-native-3.0.0-{chip}.bin").write_bytes(b"native")
        (tmp_path / f"espectre-native-3.0.0-{chip}-ota.bin").write_bytes(b"native-ota")

    (tmp_path / "espectre-3.0.0-esp32c3-ml.bin").write_bytes(b"unused")
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
    assert len(artifacts) == 20
    assert {artifact["chip"] for artifact in artifacts} == set(chips)
    assert sum(artifact["build_type"] == "factory" for artifact in artifacts) == 15
    assert sum(artifact["build_type"] == "ota" for artifact in artifacts) == 5
    assert output.is_file()
    module.validate_complete_matrix(manifest)

    ota_manifests = module.build_native_ota_manifests(manifest, tmp_path / "ota-manifests")
    assert len(ota_manifests) == 5
    assert all(path.is_file() for path in ota_manifests)


def test_complete_matrix_validation_rejects_missing_artifact():
    module = _load_module()
    manifest = {
        "frontends": {
            "esphome": {"artifacts": []},
            "matter": {"artifacts": []},
            "native": {"artifacts": []},
        }
    }

    try:
        module.validate_complete_matrix(manifest)
    except ValueError as error:
        assert "Invalid firmware matrix" in str(error)
    else:
        raise AssertionError("Incomplete firmware matrix was accepted")
