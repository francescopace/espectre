#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - SDK Package Builder

Build source-first SDK bundles and release metadata for stable and snapshot
channels.

Author: Francesco Pace <francesco.pace@gmail.com>
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import yaml

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from detect_git_version import parse_version_core

REPO_ROOT = Path(__file__).resolve().parents[2]
CPP_ROOT = REPO_ROOT / "src" / "cpp"
RUNTIME_PROTOCOL_HEADER = CPP_ROOT / "runtime" / "espectre_protocol.h"
SDK_VERSION_HEADER = CPP_ROOT / "runtime" / "espectre_sdk_version.h"
IDF_COMPONENT_MANIFEST = CPP_ROOT / "idf_component.yml"
COMPONENT_NAME = "francescopace/espectre"
SDK_SOURCE_SUFFIXES = {".h", ".hpp", ".c", ".cpp", ".cmake"}
OPTIONAL_SOURCE_GROUPS = (
    "ESPECTRE_RUNTIME_FRONTEND_SUPPORT_SOURCES",
    "ESPECTRE_RUNTIME_ESP_IDF_MQTT_SOURCES",
    "ESPECTRE_RUNTIME_ESP_IDF_PROVISIONING_SOURCES",
    "ESPECTRE_RUNTIME_ESP_IDF_DIRECT_SOURCES",
)
SDK_REQUIRED_PATHS = (
    Path("src/cpp/CMakeLists.txt"),
    Path("src/cpp/Kconfig.projbuild"),
    Path("src/cpp/idf_component.yml"),
    Path("src/cpp/espectre_core_sdk.h"),
    Path("src/cpp/espectre_services_sdk.h"),
    Path("src/cpp/espectre_mqtt_sdk.h"),
    Path("src/cpp/espectre_sdk.h"),
    Path("src/cpp/espectre_sources.cmake"),
    Path("src/cpp/core/ml_weights.h"),
    Path("src/cpp/runtime/espectre_sdk_version.h"),
    Path("docs/SDK.md"),
    Path("src/cpp/Doxyfile"),
    Path("src/cpp/sdk_integration.dox"),
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
    Path("src/cpp/espectre_core_sdk.h"),
    Path("src/cpp/espectre_services_sdk.h"),
    Path("src/cpp/espectre_mqtt_sdk.h"),
    Path("src/cpp/espectre_sdk.h"),
    Path("src/cpp/espectre_sources.cmake"),
    Path("src/cpp/Doxyfile"),
    Path("src/cpp/sdk_integration.dox"),
    # The integration guide travels with the sources so a bundle is
    # self-contained: `doxygen src/cpp/Doxyfile` from the bundle root rebuilds
    # the API XML offline. Packaging rewrites OUTPUT_DIRECTORY to output because
    # the repo Doxyfile targets docs/web/artifacts/sdk.
    Path("docs/SDK.md"),
    Path("LICENSE"),
    Path("LICENSING.md"),
    Path("THIRD_PARTY_NOTICES.md"),
)

# Repo Doxyfile writes under docs/web/; bundles rewrite to a single-segment path
# Doxygen can create without the website tree.
BUNDLE_DOXYFILE_OUTPUT_DIRECTORY = "output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ESPectre SDK bundles and manifests.")
    parser.add_argument(
        "--channel",
        choices=("release", "preview", "develop"),
        required=True,
        help="Release channel for this SDK bundle.",
    )
    parser.add_argument("--version", required=True, help="Human-readable SDK version label.")
    parser.add_argument("--release-tag", required=True, help="GitHub release tag for the published assets.")
    parser.add_argument("--output-dir", required=True, help="Directory where bundle assets are written.")
    parser.add_argument(
        "--component-output-dir",
        help="Also stage the registry component here and write its archive and inventory beside it.",
    )
    parser.add_argument("--commit", help="Optional source commit SHA for preview and develop builds.")
    parser.add_argument(
        "--source-date-epoch",
        type=int,
        help="Reproducible archive timestamp; defaults to SOURCE_DATE_EPOCH or the checkout commit time.",
    )
    parser.add_argument(
        "--url-prefix",
        help="Optional URL prefix used instead of GitHub Releases for artifact URLs.",
    )
    return parser.parse_args()


def detect_protocol_version() -> str:
    match = re.search(
        r'ESPECTRE_PROTOCOL_VERSION\s*=\s*"([^"]+)"',
        RUNTIME_PROTOCOL_HEADER.read_text(encoding="utf-8"),
    )
    if not match:
        raise ValueError("Unable to detect ESPECTRE_PROTOCOL_VERSION")
    return match.group(1)


def detect_sdk_version(header: Path | None = None) -> str:
    """Read the compile-time SDK version from a stamped header."""
    source = (header or SDK_VERSION_HEADER).read_text(encoding="utf-8")
    values = re.search(
        r"/\* ESPECTRE_SDK_VERSION_VALUES_BEGIN \*/(.*?)/\* ESPECTRE_SDK_VERSION_VALUES_END \*/",
        source,
        re.DOTALL,
    )
    source = values.group(1) if values else ""
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

    major, minor, patch = parse_version_core(version_string)
    expected = (int(components["MAJOR"]), int(components["MINOR"]), int(components["PATCH"]))
    if expected != (major, minor, patch):
        raise ValueError(
            f"ESPECTRE_SDK_VERSION_STRING is {version_string!r} but the numeric macros say "
            f"{components['MAJOR']}.{components['MINOR']}.{components['PATCH']}"
        )
    return version_string


def idf_component_manifest_version(manifest: Path | None = None) -> str:
    match = re.search(
        r'^version:\s*"?([^"\s]+)"?\s*$',
        (manifest or IDF_COMPONENT_MANIFEST).read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    if not match:
        raise ValueError("Unable to detect the ESP-IDF component manifest version")
    return match.group(1)


def release_asset_stem(channel: str, version: str) -> str:
    if channel == "release":
        return f"espectre-sdk-{version}"
    if channel == "preview":
        return "espectre-sdk-preview"
    return "espectre-sdk-develop"


def collect_bundle_files() -> list[Path]:
    files: list[Path] = []
    for root in SDK_ROOTS:
        for path in sorted((REPO_ROOT / root).rglob("*")):
            relative = path.relative_to(REPO_ROOT / root)
            if (path.is_file() and not path.is_symlink()
                    and not any(part.startswith((".", "build")) or part == "managed_components"
                                for part in relative.parts)
                    and (path.suffix in SDK_SOURCE_SUFFIXES
                         or path.name in {"CMakeLists.txt", "Kconfig.projbuild"})):
                files.append(path.relative_to(REPO_ROOT))
    files.extend(SDK_TOP_LEVEL_FILES)
    deduped = sorted(dict.fromkeys(files))
    return deduped


def validate_layout(bundle_files: list[Path]) -> None:
    bundle_file_set = set(bundle_files)
    missing = [str(path) for path in SDK_REQUIRED_PATHS if path not in bundle_file_set]
    if missing:
        raise ValueError(f"SDK bundle is missing required paths: {missing}")


def detect_doxyfile_project_number(path: Path) -> str:
    match = re.search(
        r'(?m)^PROJECT_NUMBER\s*=\s*"?([^"\s]+)"?\s*$',
        path.read_text(encoding="utf-8"),
    )
    if not match:
        raise ValueError(f"Unable to detect PROJECT_NUMBER in {path}")
    return match.group(1)


def stamp_doxyfile_project_number(path: Path, version: str) -> None:
    parse_version_core(version)
    text, count = re.subn(
        r"(?m)^PROJECT_NUMBER\s*=\s*.*$",
        f"PROJECT_NUMBER         = {version}",
        path.read_text(encoding="utf-8"),
        count=1,
    )
    if count != 1:
        raise ValueError(f"Unable to stamp PROJECT_NUMBER in {path}")
    path.write_text(text, encoding="utf-8")


def validate_stamped_sdk_identity(destination_root: Path, version: str) -> None:
    """Require stamped header macros, idf_component.yml, and Doxygen to match the SDK version."""
    header = destination_root / "src" / "cpp" / "runtime" / "espectre_sdk_version.h"
    manifest = destination_root / "src" / "cpp" / "idf_component.yml"
    doxyfile = destination_root / "src" / "cpp" / "Doxyfile"
    stamped = detect_sdk_version(header)
    yml_version = idf_component_manifest_version(manifest)
    project_number = detect_doxyfile_project_number(doxyfile)
    mismatched = {
        path: value
        for path, value in (
            (str(header.relative_to(destination_root)), stamped),
            (str(manifest.relative_to(destination_root)), yml_version),
            (str(doxyfile.relative_to(destination_root)), project_number),
        )
        if value != version
    }
    if mismatched:
        raise ValueError(
            f"Stamped SDK identity is {version!r} but packaging metadata disagrees: {mismatched}"
        )


def stamp_sdk_version_header(path: Path, version: str) -> None:
    major, minor, patch = parse_version_core(version)
    source = path.read_text(encoding="utf-8")
    stamped = (
        "/* ESPECTRE_SDK_VERSION_VALUES_BEGIN */\n"
        f"#define ESPECTRE_SDK_VERSION_MAJOR {major}\n"
        f"#define ESPECTRE_SDK_VERSION_MINOR {minor}\n"
        f"#define ESPECTRE_SDK_VERSION_PATCH {patch}\n"
        f'#define ESPECTRE_SDK_VERSION_STRING "{version}"\n'
        "/* ESPECTRE_SDK_VERSION_VALUES_END */"
    )
    source, count = re.subn(
        r"/\* ESPECTRE_SDK_VERSION_VALUES_BEGIN \*/.*?/\* ESPECTRE_SDK_VERSION_VALUES_END \*/",
        stamped,
        source,
        count=1,
        flags=re.DOTALL,
    )
    if count != 1:
        raise ValueError(f"Unable to stamp SDK version values in {path}")
    path.write_text(source, encoding="utf-8")


def stamp_idf_component_manifest(path: Path, version: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    replaced = False
    output_lines: list[str] = []
    for line in lines:
        if line.startswith("version: "):
            output_lines.append(f'version: "{version}"')
            replaced = True
        else:
            output_lines.append(line)
    if not replaced:
        output_lines.insert(0, f'version: "{version}"')
    path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")


def rewrite_bundle_doxyfile(path: Path, version: str) -> None:
    """Point the bundled Doxyfile at output and stamp the bundle identity."""
    text = path.read_text(encoding="utf-8")
    if not re.search(r"(?m)^OUTPUT_DIRECTORY\s*=", text):
        raise ValueError(f"Unable to rewrite OUTPUT_DIRECTORY in {path}")

    # Replace the repository usage/output preamble with bundle-oriented guidance.
    text, preamble_count = re.subn(
        r"# Usage, from the repository root.*?(?=\nPROJECT_NAME)",
        (
            "# Usage, from the unpacked SDK bundle root:\n"
            "#   doxygen src/cpp/Doxyfile\n"
            "#\n"
            "# Doxygen XML is written to output/xml/. Packaging rewrote OUTPUT_DIRECTORY\n"
            "# away from the repository website path so this works without docs/web/.\n"
        ),
        text,
        count=1,
        flags=re.DOTALL,
    )
    if preamble_count != 1:
        raise ValueError(f"Unable to rewrite Doxyfile usage preamble in {path}")

    text, output_count = re.subn(
        r"(?m)^OUTPUT_DIRECTORY\s*=\s*.*$",
        f"OUTPUT_DIRECTORY       = {BUNDLE_DOXYFILE_OUTPUT_DIRECTORY}",
        text,
        count=1,
    )
    if output_count != 1:
        raise ValueError(f"Unable to rewrite OUTPUT_DIRECTORY in {path}")

    # Replace repository-only output guidance with the bundle location.
    text, mkdir_count = re.subn(
        r"(?m)^# The repository generator replaces this with an isolated temporary directory\.\n"
        r"# Direct Doxygen runs write ignored XML under the website artifact tree\.\n",
        (
            "# The bundled configuration writes tool-neutral XML below output/.\n"
        ),
        text,
        count=1,
    )
    if mkdir_count != 1:
        raise ValueError(f"Unable to rewrite Doxyfile OUTPUT_DIRECTORY comments in {path}")

    path.write_text(text, encoding="utf-8")
    stamp_doxyfile_project_number(path, version)


def rewrite_bundle_sdk_guide(path: Path, source_ref: str) -> None:
    """Point repository-relative Markdown links at the exact packaged revision."""
    source = path.read_text(encoding="utf-8")

    def replace_link(match: re.Match[str]) -> str:
        target = match.group(1)
        anchor = match.group(2) or ""
        normalized = os.path.normpath(os.path.join("docs", target)).replace(os.sep, "/")
        if normalized == ".." or normalized.startswith("../"):
            raise ValueError(f"Bundled SDK guide link escapes the repository: {target}")
        return (
            "](https://github.com/francescopace/espectre/blob/"
            f"{source_ref}/{normalized}{anchor})"
        )

    rewritten, count = re.subn(
        r"\]\((?!https?://)(?!mailto:)(?!#)([^)#]+)(#[^)]+)?\)",
        replace_link,
        source,
    )
    if count == 0:
        raise ValueError(f"Bundled SDK guide has no relative Markdown links to rewrite: {path}")
    path.write_text(rewritten, encoding="utf-8")


def rewrite_bundle_sdk_facade(path: Path, source_ref: str) -> None:
    """Pin the generated reference's SDK guide link to the packaged revision."""
    source = path.read_text(encoding="utf-8")
    current = "https://github.com/francescopace/espectre/blob/main/docs/SDK.md"
    replacement = f"https://github.com/francescopace/espectre/blob/{source_ref}/docs/SDK.md"
    if source.count(current) != 1:
        raise ValueError(f"Unable to rewrite SDK guide link in {path}")
    path.write_text(source.replace(current, replacement), encoding="utf-8")


def rewrite_packaged_document(path: Path, original: Path, root: Path, source_ref: str) -> None:
    """Keep packaged links local, and pin omitted repository documents to Git."""
    def rewrite_link(match: re.Match[str]) -> str:
        link = match.group(1)
        if link.startswith(("https:", "http:", "mailto:", "#")):
            return match.group(0)
        local = (path.parent / link.split("#", 1)[0]).resolve()
        if local.is_relative_to(root.resolve()) and local.is_file():
            return match.group(0)
        relative = os.path.normpath(str(original.parent / link)).replace(os.sep, "/")
        if relative.startswith("../"):
            raise ValueError(f"Document link escapes the repository: {link}")
        if (root / relative.split("#", 1)[0]).is_file():
            return f"]({relative})"
        return f"](https://github.com/francescopace/espectre/blob/{source_ref}/{relative})"

    path.write_text(re.sub(r"\]\(([^)]+)\)", rewrite_link, path.read_text(encoding="utf-8")), encoding="utf-8")


def stage_bundle_tree(destination_root: Path, version: str, source_ref: str,
                      bundle_files: list[Path]) -> int:
    for relative_path in bundle_files:
        source = REPO_ROOT / relative_path
        target = destination_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    stamp_idf_component_manifest(destination_root / "src" / "cpp" / "idf_component.yml", version)
    stamp_sdk_version_header(
        destination_root / "src" / "cpp" / "runtime" / "espectre_sdk_version.h",
        version,
    )
    rewrite_bundle_doxyfile(destination_root / "src" / "cpp" / "Doxyfile", version)
    rewrite_bundle_sdk_guide(destination_root / "docs" / "SDK.md", source_ref)
    rewrite_bundle_sdk_facade(destination_root / "src" / "cpp" / "espectre_sdk.h", source_ref)
    integration = destination_root / "src/cpp/sdk_integration.dox"
    integration.write_text(integration.read_text(encoding="utf-8").replace(
        "https://github.com/francescopace/espectre/blob/main/",
        f"https://github.com/francescopace/espectre/blob/{source_ref}/",
    ), encoding="utf-8")
    for name in ("LICENSING.md", "THIRD_PARTY_NOTICES.md"):
        rewrite_packaged_document(destination_root / name, Path(name), destination_root, source_ref)
    validate_stamped_sdk_identity(destination_root, version)
    return len(bundle_files)


def resolve_source_date_epoch(explicit_epoch: int | None = None) -> int:
    if explicit_epoch is not None:
        epoch = explicit_epoch
    elif os.environ.get("SOURCE_DATE_EPOCH"):
        epoch = int(os.environ["SOURCE_DATE_EPOCH"])
    else:
        result = subprocess.run(
            ["git", "show", "-s", "--format=%ct", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        epoch = int(result.stdout.strip())
    if epoch < 0:
        raise ValueError("SOURCE_DATE_EPOCH must not be negative")
    return epoch


def normalized_mode(path: Path) -> int:
    if path.is_dir() or path.stat().st_mode & stat.S_IXUSR:
        return 0o755
    return 0o644


def normalize_tar_info(info: tarfile.TarInfo, path: Path, epoch: int) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = epoch
    info.mode = normalized_mode(path)
    return info


def write_tarball(source_dir: Path, output_path: Path, root_dir_name: str, epoch: int) -> None:
    paths = [source_dir, *sorted(source_dir.rglob("*"))]
    with output_path.open("wb") as output_file:
        with gzip.GzipFile(filename="", mode="wb", fileobj=output_file, mtime=epoch) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for path in paths:
                    relative = path.relative_to(source_dir) if path != source_dir else Path()
                    arcname = Path(root_dir_name) / relative
                    info = normalize_tar_info(archive.gettarinfo(str(path), str(arcname)), path, epoch)
                    if info.isfile():
                        with path.open("rb") as source_file:
                            archive.addfile(info, source_file)
                    else:
                        archive.addfile(info)


def write_zipfile(source_dir: Path, output_path: Path, root_dir_name: str, epoch: int) -> None:
    zip_epoch = max(epoch, 315532800)
    timestamp = datetime.fromtimestamp(zip_epoch, timezone.utc)
    date_time = (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute, timestamp.second)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(source_dir)
            info = zipfile.ZipInfo(str(Path(root_dir_name) / relative), date_time=date_time)
            info.create_system = 3
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = normalized_mode(path) << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_registry_component(bundle_root: Path, destination: Path, version: str,
                             source_ref: str, epoch: int, channel: str) -> None:
    """Package the SDK with the registry identity and Component Manager's file rules."""
    from idf_component_tools.manager import ManifestManager
    from sdk_api_markdown import generate_registry_api

    destination = destination.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"Component output directory must be empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    sdk_root = bundle_root / "src" / "cpp"
    for source in sdk_root.rglob("*"):
        if source.is_file() and source.name not in {"Doxyfile", "sdk_integration.dox"}:
            target = destination / source.relative_to(sdk_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    for name in ("LICENSE", "LICENSING.md", "THIRD_PARTY_NOTICES.md"):
        shutil.copy2(bundle_root / name, destination / name)
    shutil.copy2(bundle_root / "docs" / "SDK.md", destination / "README.md")
    generate_registry_api(bundle_root, destination, version, source_ref)
    example_root = CPP_ROOT / "examples" / "wifi_motion_detection"
    example_files = (
        "CMakeLists.txt", "README.md", "sdkconfig.defaults",
        "main/CMakeLists.txt", "main/Kconfig.projbuild", "main/idf_component.yml",
        "main/app_main.cpp", "main/optional_services.cpp",
    )
    for relative in example_files:
        target = destination / "examples" / "wifi_motion_detection" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(example_root / relative, target)
    registry_url = ("https://components.espressif.com" if channel == "release"
                    else "https://components-staging.espressif.com")
    for readme_path in (destination / "README.md", destination / "examples/wifi_motion_detection/README.md"):
        readme = readme_path.read_text(encoding="utf-8")
        for name, value in (("ESPECTRE_VERSION", version), ("ESPECTRE_REGISTRY_URL", registry_url)):
            readme, count = re.subn(rf'(?m)^{name}="[^"]+"$', f'{name}="{value}"', readme)
            if count != 1:
                raise ValueError(f"Expected one {name} setting in {readme_path}")
        readme_path.write_text(readme, encoding="utf-8")
    example_manifest = destination / "examples" / "wifi_motion_detection" / "main" / "idf_component.yml"
    example = yaml.safe_load(example_manifest.read_text())
    example["dependencies"][COMPONENT_NAME] = {"version": version, "registry_url": registry_url}
    example_manifest.write_text(yaml.safe_dump(example, sort_keys=False), encoding="utf-8")

    manifest_path = destination / "idf_component.yml"
    stamp_idf_component_manifest(manifest_path, version)
    stamp_sdk_version_header(destination / "runtime" / "espectre_sdk_version.h", version)
    metadata = {"component": COMPONENT_NAME, "version": version, "commit": source_ref}
    (destination / "sdk-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if detect_sdk_version(destination / "runtime" / "espectre_sdk_version.h") != version:
        raise ValueError("Registry component SDK identity does not match its manifest")

    # Resolve documents omitted from the component against the packaged revision.
    for name, original in (("LICENSING.md", Path("LICENSING.md")),
                           ("THIRD_PARTY_NOTICES.md", Path("THIRD_PARTY_NOTICES.md"))):
        rewrite_packaged_document(destination / name, original, destination, source_ref)

    ManifestManager(manifest_path, name="espectre").load()
    ManifestManager(example_manifest, name="main").load()
    compote = [sys.executable, "-m", "idf_component_manager"]
    with tempfile.TemporaryDirectory(prefix="espectre-component-pack-") as tmp:
        subprocess.run([*compote, "component", "pack", "--name", "espectre",
                        "--dest-dir", tmp], cwd=destination, check=True)
        archives = list(Path(tmp).glob("*.tgz"))
        if len(archives) != 1:
            raise ValueError(f"Expected one component archive, found {archives}")
        archive_path = destination.parent / f"espectre-component-{version}.tgz"
        # Preserve compote's contents, normalizing tar and gzip metadata.
        extracted = Path(tmp) / "contents"
        with tarfile.open(archives[0]) as source:
            source.extractall(extracted, filter="data")
        files = {str(path.relative_to(extracted)): sha256_file(path)
                 for path in sorted(extracted.rglob("*")) if path.is_file()}
        write_tarball(extracted, archive_path, ".", epoch)
    inventory = {"schema_version": 1, "component": COMPONENT_NAME, "version": version,
                 "commit": source_ref, "archive": archive_path.name,
                 "sha256": sha256_file(archive_path), "files": files}
    inventory_path = destination.parent / "component-inventory.json"
    inventory_path.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    print(f"Registry component: {archive_path} ({len(files)} files)")


def registry_component_version(version: str, channel: str, commit: str) -> str:
    """Keep release versions exact and identify registry snapshots by branch."""
    if channel == "release":
        return version
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Registry snapshots require a full source commit SHA")
    branch = {"preview": "main", "develop": "develop"}[channel]
    if re.fullmatch(r"\d+\.\d+\.\d+", version):
        # A branch push at a stable tag still needs a prerelease identifier.
        version = f"{version}-0-g{commit[:7]}"
    return f"{version}.{branch}"


def build_artifact_url(filename: str, release_tag: str, url_prefix: str | None) -> str:
    if url_prefix:
        return f"{url_prefix.rstrip('/')}/{filename}"
    return f"https://github.com/francescopace/espectre/releases/download/{release_tag}/{filename}"


def build_manifest(
    *,
    channel: str,
    version: str,
    release_tag: str,
    commit: str | None,
    tarball_name: str,
    zip_name: str,
    bundle_file_count: int,
    bundle_root: str,
    url_prefix: str | None,
    generated_at: str,
    tarball_sha256: str,
    zip_sha256: str,
) -> dict:
    return {
        "schema_version": 2,
        "artifact_kind": "sdk",
        "channel": channel,
        "version": version,
        "release_tag": release_tag,
        "generated_at": generated_at,
        "commit": commit,
        "protocol_version": detect_protocol_version(),
        "supported_esp_idf": yaml.safe_load(IDF_COMPONENT_MANIFEST.read_text())["dependencies"]["idf"]["version"],
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
                "sha256": tarball_sha256,
            },
            {
                "format": "zip",
                "filename": zip_name,
                "url": build_artifact_url(zip_name, release_tag, url_prefix),
                "sha256": zip_sha256,
            },
        ],
        "install_surfaces": {
            "cmake": {
                "entrypoint": "src/cpp/espectre_sources.cmake",
                "optional_source_groups": list(OPTIONAL_SOURCE_GROUPS),
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
    if args.channel == "release" and args.version != args.release_tag:
        raise ValueError(
            "Release SDK version and release tag must match: "
            f"{args.version!r} != {args.release_tag!r}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle_files = collect_bundle_files()
    validate_layout(bundle_files)

    parse_version_core(args.version)
    asset_stem = release_asset_stem(args.channel, args.version)
    bundle_root = asset_stem
    tarball_name = f"{asset_stem}.tar.gz"
    zip_name = f"{asset_stem}.zip"
    manifest_suffix = args.release_tag if args.channel == "release" else args.channel
    manifest_name = f"sdk-manifest-{manifest_suffix}.json"
    source_date_epoch = resolve_source_date_epoch(getattr(args, "source_date_epoch", None))
    generated_at = datetime.fromtimestamp(source_date_epoch, timezone.utc).isoformat()
    tarball_path = output_dir / tarball_name
    zip_path = output_dir / zip_name

    with tempfile.TemporaryDirectory(prefix="espectre-sdk-") as tmp_dir:
        staged_root = Path(tmp_dir) / bundle_root
        source_ref = args.commit or args.release_tag
        if getattr(args, "component_output_dir", None) and not args.commit:
            source_ref = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
            ).strip()
        file_count = stage_bundle_tree(staged_root, args.version, source_ref, bundle_files)
        write_tarball(staged_root, tarball_path, bundle_root, source_date_epoch)
        write_zipfile(staged_root, zip_path, bundle_root, source_date_epoch)
        if getattr(args, "component_output_dir", None):
            component_version = registry_component_version(args.version, args.channel, source_ref)
            stage_registry_component(staged_root, Path(args.component_output_dir), component_version,
                                     source_ref, source_date_epoch, args.channel)

    manifest = build_manifest(
        channel=args.channel,
        version=args.version,
        release_tag=args.release_tag,
        commit=args.commit,
        tarball_name=tarball_name,
        zip_name=zip_name,
        bundle_file_count=file_count,
        bundle_root=bundle_root,
        url_prefix=args.url_prefix,
        generated_at=generated_at,
        tarball_sha256=sha256_file(tarball_path),
        zip_sha256=sha256_file(zip_path),
    )
    (output_dir / manifest_name).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    build_sdk_package(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
