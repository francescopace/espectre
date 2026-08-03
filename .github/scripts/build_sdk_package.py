#!/usr/bin/env python3
"""
ESPectre - SDK Package Builder

Build source-first SDK bundles and release metadata for stable and snapshot
channels.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tarfile
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CPP_ROOT = REPO_ROOT / "src" / "cpp"
RUNTIME_PROTOCOL_HEADER = CPP_ROOT / "runtime" / "espectre_protocol.h"
SDK_VERSION_HEADER = CPP_ROOT / "runtime" / "espectre_sdk_version.h"
LIBRARY_MANIFEST = CPP_ROOT / "library.json"
IDF_COMPONENT_MANIFEST = CPP_ROOT / "idf_component.yml"
SDK_SUPPORTED_ESP_IDF = ">=5.1.0"
OPTIONAL_SOURCE_GROUPS = (
    "ESPECTRE_RUNTIME_ESP_IDF_BLE_SOURCES",
    "ESPECTRE_RUNTIME_ESP_IDF_MQTT_SOURCES",
    "ESPECTRE_RUNTIME_ESP_IDF_PROVISIONING_SOURCES",
    "ESPECTRE_RUNTIME_ESP_IDF_OTA_SOURCES",
)
SDK_REQUIRED_PATHS = (
    Path("src/cpp/CMakeLists.txt"),
    Path("src/cpp/Kconfig.projbuild"),
    Path("src/cpp/idf_component.yml"),
    Path("src/cpp/espectre_sdk.h"),
    Path("src/cpp/espectre_sources.cmake"),
    Path("src/cpp/library.json"),
    Path("src/cpp/core/ml_weights.h"),
    Path("src/cpp/runtime/espectre_sdk_version.h"),
    Path("src/cpp/runtime/espectre_protocol.h"),
    Path("src/cpp/runtime/esp_idf/runtime_sensing_kconfig.cpp"),
    Path("src/cpp/runtime/esp_idf/espectre_config/CMakeLists.txt"),
    Path("src/cpp/runtime/esp_idf/espectre_config/Kconfig.projbuild"),
    Path("src/cpp/runtime/esp_idf/espectre_config/espectre_config_stub.c"),
)
SDK_ROOTS = (
    Path("src/cpp/core"),
    Path("src/cpp/runtime"),
)
SDK_TOP_LEVEL_FILES = (
    Path("src/cpp/CMakeLists.txt"),
    Path("src/cpp/Kconfig.projbuild"),
    Path("src/cpp/idf_component.yml"),
    Path("src/cpp/espectre_sdk.h"),
    Path("src/cpp/espectre_sources.cmake"),
    Path("src/cpp/library.json"),
    Path("LICENSE"),
    Path("LICENSING.md"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ESPectre SDK bundles and manifests.")
    parser.add_argument(
        "--channel",
        choices=("stable", "main", "develop"),
        required=True,
        help="Release channel for this SDK bundle.",
    )
    parser.add_argument("--version", required=True, help="Human-readable SDK version label.")
    parser.add_argument("--release-tag", required=True, help="GitHub release tag for the published assets.")
    parser.add_argument("--output-dir", required=True, help="Directory where bundle assets are written.")
    parser.add_argument("--commit", help="Optional source commit SHA for snapshot builds.")
    parser.add_argument(
        "--url-prefix",
        help="Optional URL prefix used instead of GitHub Releases for artifact URLs.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def detect_protocol_version() -> str:
    match = re.search(
        r'ESPECTRE_PROTOCOL_VERSION\s*=\s*"([^"]+)"',
        RUNTIME_PROTOCOL_HEADER.read_text(encoding="utf-8"),
    )
    if not match:
        raise ValueError("Unable to detect ESPECTRE_PROTOCOL_VERSION")
    return match.group(1)


def detect_sdk_version() -> str:
    """Read the compile-time SDK version integrators can guard against."""
    source = SDK_VERSION_HEADER.read_text(encoding="utf-8")
    match = re.search(r'#define\s+ESPECTRE_SDK_VERSION_STRING\s+"([^"]+)"', source)
    if not match:
        raise ValueError("Unable to detect ESPECTRE_SDK_VERSION_STRING")
    version_string = match.group(1)

    components = {}
    for name in ("MAJOR", "MINOR", "PATCH"):
        component = re.search(rf"#define\s+ESPECTRE_SDK_VERSION_{name}\s+(\d+)", source)
        if not component:
            raise ValueError(f"Unable to detect ESPECTRE_SDK_VERSION_{name}")
        components[name] = component.group(1)

    expected = f"{components['MAJOR']}.{components['MINOR']}.{components['PATCH']}"
    if expected != version_string:
        raise ValueError(
            f"ESPECTRE_SDK_VERSION_STRING is {version_string!r} but the numeric macros say {expected!r}"
        )
    return version_string


def idf_component_manifest_version() -> str:
    match = re.search(
        r'^version:\s*"?([^"\s]+)"?\s*$',
        IDF_COMPONENT_MANIFEST.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    if not match:
        raise ValueError("Unable to detect the ESP-IDF component manifest version")
    return match.group(1)


def release_asset_stem(channel: str, version: str) -> str:
    if channel == "stable":
        return f"espectre-sdk-{version}"
    if channel == "main":
        return "espectre-sdk-snapshot"
    return "espectre-sdk-snapshot-dev"


def snapshot_package_version(base_version: str, suffix: str, commit: str | None) -> str:
    normalized = base_version.split("+", 1)[0]
    sha = (commit or "local")[:7]
    if "-" in normalized:
        core, prerelease = normalized.split("-", 1)
        return f"{core}-{prerelease}.{suffix}+{sha}"
    return f"{normalized}-{suffix}+{sha}"


def package_version(channel: str, version: str, commit: str | None, base_version: str) -> str:
    if channel == "stable":
        return version
    if channel == "main":
        return snapshot_package_version(base_version, "snapshot", commit)
    return snapshot_package_version(base_version, "snapshot-dev", commit)


def collect_bundle_files() -> list[Path]:
    files: list[Path] = []
    for root in SDK_ROOTS:
        for path in sorted((REPO_ROOT / root).rglob("*")):
            if path.is_file():
                files.append(path.relative_to(REPO_ROOT))
    files.extend(SDK_TOP_LEVEL_FILES)
    deduped = sorted(dict.fromkeys(files))
    return deduped


def validate_layout(bundle_files: list[Path], library_manifest: dict) -> None:
    bundle_file_set = set(bundle_files)
    missing = [str(path) for path in SDK_REQUIRED_PATHS if path not in bundle_file_set]
    if missing:
        raise ValueError(f"SDK bundle is missing required paths: {missing}")

    src_filter = library_manifest.get("build", {}).get("srcFilter", [])
    if "+<runtime/esp_idf/runtime_sensing_kconfig.cpp>" not in src_filter:
        raise ValueError("PlatformIO manifest is missing runtime_sensing_kconfig.cpp")

    build_flags = library_manifest.get("build", {}).get("flags", [])
    if "-I." not in build_flags:
        raise ValueError("PlatformIO manifest does not expose the SDK root include dir")

    # The compile-time macros integrators guard against are only useful if they
    # agree with the package metadata a dependency manager resolves.
    sdk_version = detect_sdk_version()
    manifest_versions = {
        "src/cpp/library.json": str(library_manifest["version"]),
        "src/cpp/idf_component.yml": idf_component_manifest_version(),
    }
    mismatched = {path: value for path, value in manifest_versions.items() if value != sdk_version}
    if mismatched:
        raise ValueError(
            f"ESPECTRE_SDK_VERSION_STRING is {sdk_version!r} but packaging metadata disagrees: {mismatched}"
        )


def stamp_library_manifest(path: Path, sdk_package_version: str) -> dict:
    payload = load_json(path)
    payload["version"] = sdk_package_version
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def stamp_idf_component_manifest(path: Path, sdk_package_version: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    replaced = False
    output_lines: list[str] = []
    for line in lines:
        if line.startswith("version: "):
            output_lines.append(f'version: "{sdk_package_version}"')
            replaced = True
        else:
            output_lines.append(line)
    if not replaced:
        output_lines.insert(0, f'version: "{sdk_package_version}"')
    path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")


def stage_bundle_tree(destination_root: Path, sdk_package_version: str, bundle_files: list[Path]) -> tuple[int, dict]:
    for relative_path in bundle_files:
        source = REPO_ROOT / relative_path
        target = destination_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    staged_library_manifest = stamp_library_manifest(destination_root / "src" / "cpp" / "library.json", sdk_package_version)
    stamp_idf_component_manifest(destination_root / "src" / "cpp" / "idf_component.yml", sdk_package_version)
    return len(bundle_files), staged_library_manifest


def write_tarball(source_dir: Path, output_path: Path, root_dir_name: str) -> None:
    with tarfile.open(output_path, "w:gz") as archive:
        archive.add(source_dir, arcname=root_dir_name)


def write_zipfile(source_dir: Path, output_path: Path, root_dir_name: str) -> None:
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(source_dir)
            archive.write(path, arcname=str(Path(root_dir_name) / relative))


def build_artifact_url(filename: str, release_tag: str, url_prefix: str | None) -> str:
    if url_prefix:
        return f"{url_prefix.rstrip('/')}/{filename}"
    return f"https://github.com/francescopace/espectre/releases/download/{release_tag}/{filename}"


def build_manifest(
    *,
    channel: str,
    version: str,
    sdk_package_version: str,
    release_tag: str,
    commit: str | None,
    tarball_name: str,
    zip_name: str,
    bundle_file_count: int,
    bundle_root: str,
    url_prefix: str | None,
) -> dict:
    return {
        "schema_version": 1,
        "artifact_kind": "sdk",
        "channel": channel,
        "version": version,
        "package_version": sdk_package_version,
        "release_tag": release_tag,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "commit": commit,
        "protocol_version": detect_protocol_version(),
        "sdk_version": detect_sdk_version(),
        "supported_esp_idf": SDK_SUPPORTED_ESP_IDF,
        "bundle": {
            "root_dir": bundle_root,
            "file_count": bundle_file_count,
            "required_paths": [str(path) for path in SDK_REQUIRED_PATHS],
            "source_roots": [str(path) for path in SDK_ROOTS],
            "top_level_files": [str(path) for path in SDK_TOP_LEVEL_FILES],
        },
        "artifacts": [
            {
                "format": "tar.gz",
                "filename": tarball_name,
                "url": build_artifact_url(tarball_name, release_tag, url_prefix),
            },
            {
                "format": "zip",
                "filename": zip_name,
                "url": build_artifact_url(zip_name, release_tag, url_prefix),
            },
        ],
        "install_surfaces": {
            "cmake": {
                "entrypoint": "src/cpp/espectre_sources.cmake",
                "optional_source_groups": list(OPTIONAL_SOURCE_GROUPS),
            },
            "platformio": {
                "manifest": "src/cpp/library.json",
                "package_name": "espectre-shared",
                "package_version": sdk_package_version,
            },
            "esp_idf_component": {
                "component_root": "src/cpp",
                "cmake": "src/cpp/CMakeLists.txt",
                "manifest": "src/cpp/idf_component.yml",
                "kconfig": "src/cpp/Kconfig.projbuild",
            },
        },
    }


def build_sdk_package(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tracked_library_manifest = load_json(LIBRARY_MANIFEST)
    bundle_files = collect_bundle_files()
    validate_layout(bundle_files, tracked_library_manifest)

    sdk_package_version = package_version(
        args.channel,
        args.version,
        args.commit,
        str(tracked_library_manifest["version"]),
    )
    asset_stem = release_asset_stem(args.channel, args.version)
    bundle_root = asset_stem
    tarball_name = f"{asset_stem}.tar.gz"
    zip_name = f"{asset_stem}.zip"
    manifest_name = f"sdk-manifest-{args.release_tag}.json"

    with tempfile.TemporaryDirectory(prefix="espectre-sdk-") as tmp_dir:
        staged_root = Path(tmp_dir) / bundle_root
        file_count, _ = stage_bundle_tree(staged_root, sdk_package_version, bundle_files)
        write_tarball(staged_root, output_dir / tarball_name, bundle_root)
        write_zipfile(staged_root, output_dir / zip_name, bundle_root)

    manifest = build_manifest(
        channel=args.channel,
        version=args.version,
        sdk_package_version=sdk_package_version,
        release_tag=args.release_tag,
        commit=args.commit,
        tarball_name=tarball_name,
        zip_name=zip_name,
        bundle_file_count=file_count,
        bundle_root=bundle_root,
        url_prefix=args.url_prefix,
    )
    (output_dir / manifest_name).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    build_sdk_package(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
