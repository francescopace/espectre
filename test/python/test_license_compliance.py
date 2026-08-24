# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""License packaging and repository policy invariants."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / ".github" / "scripts" / "build_firmware_compliance.py"
BUNDLE_SCRIPT_PATH = REPO_ROOT / ".github" / "scripts" / "build_firmware_compliance_bundle.py"
GPL_SPDX_HEADER = "SPDX-License-Identifier: GPL-3.0-only"
COMMERCIAL_LICENSE_NOTICE = "Commercial licensing available under separate agreement; see LICENSING.md."


def load_compliance_module():
    spec = importlib.util.spec_from_file_location("build_firmware_compliance", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_bundle_module():
    spec = importlib.util.spec_from_file_location("build_firmware_compliance_bundle_test", BUNDLE_SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_firmware_compliance_collects_actual_component_licenses(tmp_path):
    module = load_compliance_module()
    idf_root = tmp_path / "esp-idf"
    idf_component = idf_root / "components" / "network"
    idf_component.mkdir(parents=True)
    (idf_root / "LICENSE").write_text("Apache License\nVersion 2.0\n", encoding="utf-8")
    (idf_component / "COPYING").write_text("network component notice\n", encoding="utf-8")

    managed_root = tmp_path / "app" / "managed_components" / "vendor__sensor"
    managed_root.mkdir(parents=True)
    (managed_root / "idf_component.yml").write_text('version: "1.2.3"\n', encoding="utf-8")
    (managed_root / "LICENSE").write_text(
        "MIT License\nPermission is hereby granted, free of charge\n",
        encoding="utf-8",
    )
    firmware = tmp_path / "espectre-matter-test.bin"
    firmware.write_bytes(b"firmware")
    description = {
        "project_path": str(tmp_path / "app"),
        "project_version": "3.0.0",
        "target": "esp32c6",
        "idf_path": str(idf_root),
        "git_revision": "5.5.5",
        "build_component_info": {
            "network": {
                "dir": str(idf_component),
                "sources": [str(idf_component / "network.c")],
                "file": str(tmp_path / "libnetwork.a"),
            },
            "sensor": {
                "dir": str(managed_root),
                "sources": [str(managed_root / "sensor.c")],
                "file": str(tmp_path / "libsensor.a"),
            },
            "espectre": {
                "dir": str(REPO_ROOT / "src" / "cpp" / "core"),
                "sources": [str(REPO_ROOT / "src" / "cpp" / "core" / "filters.cpp")],
                "file": str(tmp_path / "libespectre.a"),
            },
        },
    }
    project_description = tmp_path / "project_description.json"
    project_description.write_text(json.dumps(description), encoding="utf-8")

    sbom_path, notice_path, licenses_path = module.build_compliance(
        argparse.Namespace(
            frontend="matter",
            project_description=str(project_description),
            firmware=str(firmware),
            output_dir=str(tmp_path / "output"),
        )
    )

    sbom = json.loads(sbom_path.read_text(encoding="utf-8"))
    packages = {package["name"]: package for package in sbom["packages"]}
    assert packages["ESP-IDF"]["versionInfo"] == "5.5.5"
    assert packages["vendor/sensor"]["versionInfo"] == "1.2.3"
    assert packages["vendor/sensor"]["licenseDeclared"] == "MIT"
    firmware_package = packages[firmware.name]
    assert firmware_package["checksums"][0]["algorithm"] == "SHA256"
    assert sbom["documentDescribes"] == ["SPDXRef-Package-Firmware"]
    assert any(entry["licenseId"] == "LicenseRef-ESPectre-Commercial" for entry in sbom["hasExtractedLicensingInfos"])
    assert "Matter SDK NOTICE" in notice_path.read_text(encoding="utf-8")

    with zipfile.ZipFile(licenses_path) as archive:
        archived = set(archive.namelist())
    assert "ESP-IDF/LICENSE" in archived
    assert "ESPectre/LICENSE" in archived
    assert "ESPectre/LICENSING.md" in archived
    assert "ESP-IDF/components/network/COPYING" in archived
    assert "vendor__sensor/LICENSE" in archived
    assert "espressif__esp_matter/NOTICE" in archived


def test_repository_license_policy_covers_first_party_code_and_release_artifacts():
    licensing = (REPO_ROOT / "LICENSING.md").read_text(encoding="utf-8")
    notices = (REPO_ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    direct_client = (REPO_ROOT / "docs" / "web" / "assets" / "js" / "espectre-direct.js").read_text(
        encoding="utf-8"
    )
    mqtt_client = (REPO_ROOT / "docs" / "web" / "assets" / "js" / "espectre-mqtt.js").read_text(
        encoding="utf-8"
    )
    direct_tests = (REPO_ROOT / "test" / "web" / "test_espectre_direct.mjs").read_text(encoding="utf-8")
    ci_workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    release_workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    snapshot_workflow = (REPO_ROOT / ".github" / "workflows" / "snapshot.yml").read_text(encoding="utf-8")

    assert GPL_SPDX_HEADER in direct_client
    assert COMMERCIAL_LICENSE_NOTICE in direct_client
    assert GPL_SPDX_HEADER in mqtt_client
    assert COMMERCIAL_LICENSE_NOTICE in mqtt_client
    assert GPL_SPDX_HEADER in direct_tests
    assert COMMERCIAL_LICENSE_NOTICE in direct_tests
    assert "ESPHome C++ runtime" in licensing
    assert "test/cpp/support/LICENSE.cnpy" in licensing
    assert "build-specific SPDX SBOMs" in notices
    assert "ESP-IDF mDNS component" in notices
    assert "Improv Wi-Fi SDK for C++" in notices
    for workflow in (release_workflow, snapshot_workflow):
        assert re.search(r"(?m)^\s+firmware/\*\.bin$", workflow)
        assert re.search(r"(?m)^\s+firmware/firmware-compliance-\*\.zip$", workflow)
        assert not re.search(r"(?m)^\s+firmware/\*$", workflow)
        assert "build_firmware_compliance_bundle.py" in workflow
        assert "Remove superseded unbundled compliance assets" in workflow
        assert "--compliance-url-prefix" in workflow
    assert "--pattern 'firmware-compliance-*.zip'" in ci_workflow
    assert "build_firmware_compliance" in ci_workflow
    assert not (REPO_ROOT / "docs" / "web" / "assets" / "js" / "LICENSES").exists()
    assert (
        REPO_ROOT / "src" / "cpp" / "frontend" / "matter" / "third_party" / "esp_matter" / "NOTICE"
    ).is_file()


def test_firmware_compliance_bundle_groups_build_artifacts_and_legal_files(tmp_path):
    builder = load_bundle_module()
    firmware_dir = tmp_path / "firmware"
    legal_dir = tmp_path / "legal"
    firmware_dir.mkdir()
    legal_dir.mkdir()
    firmware = firmware_dir / "espectre-native-develop-esp32.bin"
    firmware.write_bytes(b"firmware")
    companions = []
    for suffix in builder.COMPLIANCE_SUFFIXES:
        companion = firmware.with_name(f"{firmware.stem}{suffix}")
        companion.write_bytes(suffix.encode("utf-8"))
        companions.append(companion)
    legal_paths = tuple(legal_dir / name for name in ("LICENSE", "LICENSING.md", "THIRD_PARTY_NOTICES.md"))
    for path in legal_paths:
        path.write_text(path.name, encoding="utf-8")

    output = firmware_dir / builder.bundle_filename("develop", "ignored")
    builder.build_bundle(firmware_dir, output, legal_paths=legal_paths)

    assert output.name == "firmware-compliance-develop.zip"
    assert builder.bundle_filename("release", "3.0.0") == "firmware-compliance-3.0.0.zip"
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == {path.name for path in (*legal_paths, *companions)}
        assert all(info.date_time == builder.ZIP_TIMESTAMP for info in archive.infolist())


def test_web_staging_materializes_compliance_from_release_bundle(tmp_path):
    builder = load_bundle_module()
    firmware_dir = tmp_path / "firmware"
    legal_dir = tmp_path / "legal"
    output_dir = tmp_path / "web"
    firmware_dir.mkdir()
    legal_dir.mkdir()
    firmware = firmware_dir / "espectre-esphome-preview-esp32c6.bin"
    ota_firmware = firmware_dir / "espectre-esphome-preview-esp32c6-ota.bin"
    companions = []
    ota_companions = []
    for image, image_companions in ((firmware, companions), (ota_firmware, ota_companions)):
        image.write_bytes(b"firmware")
        for suffix in builder.COMPLIANCE_SUFFIXES:
            companion = image.with_name(f"{image.stem}{suffix}")
            companion.write_bytes(suffix.encode("utf-8"))
            image_companions.append(companion)
    legal_paths = tuple(legal_dir / name for name in ("LICENSE", "LICENSING.md", "THIRD_PARTY_NOTICES.md"))
    for path in legal_paths:
        path.write_text(path.name, encoding="utf-8")
    bundle = firmware_dir / builder.bundle_filename("preview", "ignored")
    builder.build_bundle(firmware_dir, bundle, legal_paths=legal_paths)
    for companion in (*companions, *ota_companions):
        companion.unlink()

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / ".github" / "scripts" / "stage_web_firmware.py"),
            "--firmware-dir",
            str(firmware_dir),
            "--output-dir",
            str(output_dir),
            "--channel",
            "preview",
            "--version",
            "preview",
            "--release-tag",
            "snapshot",
            "--url-prefix",
            "/artifacts/firmware/preview",
        ],
        check=True,
    )

    assert (output_dir / firmware.name).is_file()
    assert all((output_dir / companion.name).is_file() for companion in companions)
    assert not (output_dir / ota_firmware.name).exists()
    assert all(not (output_dir / companion.name).exists() for companion in ota_companions)


def test_firmware_manifest_links_available_compliance_artifacts(tmp_path):
    manifest_script = REPO_ROOT / ".github" / "scripts" / "build_firmware_manifest.py"
    spec = importlib.util.spec_from_file_location("build_firmware_manifest_license_test", manifest_script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    firmware = tmp_path / "espectre-esphome-preview-esp32c6.bin"
    firmware.write_bytes(b"firmware")
    for suffix in ("-sbom.spdx.json", "-THIRD_PARTY_NOTICES.txt", "-third-party-licenses.zip"):
        firmware.with_name(f"{firmware.stem}{suffix}").write_bytes(b"compliance")
    output = tmp_path / "manifest.json"
    manifest = module.build_manifest(
        argparse.Namespace(
            firmware_dir=str(tmp_path),
            output=str(output),
            channel="preview",
            version="preview",
            release_tag="preview",
            commit="abcdef",
            url_prefix=None,
            compliance_url_prefix="https://espectre.dev/artifacts/firmware/preview",
        )
    )

    artifact = manifest["frontends"]["esphome"]["artifacts"][0]
    assert [entry["kind"] for entry in artifact["compliance"]] == [
        "spdx-sbom",
        "notices",
        "license-archive",
    ]
    assert artifact["url"].startswith("https://github.com/francescopace/espectre/releases/download/")
    assert artifact["compliance"][0]["url"].startswith(
        "https://espectre.dev/artifacts/firmware/preview/"
    )


def test_firmware_manifest_rejects_matter_esp32_s2(tmp_path):
    manifest_script = REPO_ROOT / ".github" / "scripts" / "build_firmware_manifest.py"
    spec = importlib.util.spec_from_file_location("build_firmware_manifest_s2_test", manifest_script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    (tmp_path / "espectre-matter-preview-esp32s2.bin").write_bytes(b"firmware")
    with pytest.raises(ValueError, match="Unsupported matter chip"):
        module.build_manifest(
            argparse.Namespace(
                firmware_dir=str(tmp_path),
                output=str(tmp_path / "manifest.json"),
                channel="preview",
                version="preview",
                release_tag="preview",
                commit="abcdef",
                url_prefix=None,
            )
        )


def test_complete_firmware_matrix_requires_every_compliance_companion(tmp_path):
    manifest_script = REPO_ROOT / ".github" / "scripts" / "build_firmware_manifest.py"
    spec = importlib.util.spec_from_file_location("build_firmware_manifest_matrix_test", manifest_script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    firmware_names = []
    for chip in module.FRONTEND_CHIPS["esphome"]:
        firmware_names.extend(
            (
                f"espectre-esphome-preview-{chip}.bin",
                f"espectre-esphome-preview-{chip}-ota.bin",
            )
        )
    for chip in module.FRONTEND_CHIPS["matter"]:
        firmware_names.append(f"espectre-matter-preview-{chip}.bin")
    for chip in module.FRONTEND_CHIPS["native"]:
        firmware_names.extend(
            (
                f"espectre-native-preview-{chip}.bin",
                f"espectre-native-preview-{chip}-ota.bin",
            )
        )
    companion_suffixes = (
        "-sbom.spdx.json",
        "-THIRD_PARTY_NOTICES.txt",
        "-third-party-licenses.zip",
    )
    for firmware_name in firmware_names:
        firmware = tmp_path / firmware_name
        firmware.write_bytes(b"firmware")
        for suffix in companion_suffixes:
            firmware.with_name(f"{firmware.stem}{suffix}").write_bytes(b"compliance")

    missing_notice = tmp_path / "espectre-esphome-preview-esp32-THIRD_PARTY_NOTICES.txt"
    missing_notice.unlink()
    args = argparse.Namespace(
        firmware_dir=str(tmp_path),
        output=str(tmp_path / "manifest.json"),
        channel="preview",
        version="preview",
        release_tag="preview",
        commit="abcdef",
        url_prefix=None,
        require_complete_matrix=True,
    )
    with pytest.raises(ValueError, match="Missing firmware compliance artifacts"):
        module.build_manifest(args)

    missing_notice.write_bytes(b"compliance")
    module.build_manifest(args)


def test_web_lockfile_uses_reviewed_license_families():
    lockfile = json.loads((REPO_ROOT / "docs" / "web" / "package-lock.json").read_text(encoding="utf-8"))
    allowed = {"Apache-2.0", "BSD-3-Clause", "ISC", "MIT", "0BSD", "(MIT AND Zlib)"}
    missing_license = []
    unexpected = []
    for package_path, package in lockfile["packages"].items():
        if not package_path:
            continue
        license_expression = package.get("license")
        if license_expression is None:
            missing_license.append(package_path)
        elif license_expression not in allowed:
            unexpected.append((package_path, license_expression))

    assert missing_license == ["node_modules/qrcodejs"]
    assert not unexpected
    staging = (REPO_ROOT / ".github" / "scripts" / "stage_web_vendor.py").read_text(encoding="utf-8")
    assert 'NODE_MODULES / "qrcodejs" / "LICENSE"' in staging


def test_source_files_have_consistent_license_headers():
    compiled_sources = subprocess.run(
        ["git", "ls-files", "*.c", "*.cc", "*.cpp", "*.h", "*.hpp", "*.py"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    web_sources = subprocess.run(
        ["git", "ls-files", "docs/web", "test/web/*.mjs"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    relative_paths = set(compiled_sources.stdout.splitlines())
    relative_paths.update(
        path
        for path in web_sources.stdout.splitlines()
        if Path(path).suffix in (".css", ".html", ".js", ".mjs")
    )
    license_exceptions = {
        "test/cpp/support/cnpy.cpp": "MIT",
        "test/cpp/support/cnpy.h": "MIT",
    }
    missing = []
    for relative_path in sorted(relative_paths):
        path = REPO_ROOT / relative_path
        if not path.is_file():
            continue
        header = "\n".join(path.read_text(encoding="utf-8", errors="ignore").splitlines()[:45])
        exception = license_exceptions.get(relative_path)
        if exception is not None:
            if f"SPDX-License-Identifier: {exception}" not in header:
                missing.append(f"{relative_path}: SPDX {exception} header")
            continue
        if GPL_SPDX_HEADER not in header:
            missing.append(f"{relative_path}: SPDX GPL header")
        if COMMERCIAL_LICENSE_NOTICE not in header:
            missing.append(f"{relative_path}: commercial licensing notice")

    assert not missing, "\n".join(missing)
