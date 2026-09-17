#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Verify registry contents and prepare an SDK consumer outside the checkout."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import time
from urllib.request import urlopen
import zipfile

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
COMPONENT_NAME = "francescopace/espectre"
SDK_SERVICE_PROFILES = ("frontend_support", "mqtt", "provisioning", "direct")


def archive_files(data: bytes) -> dict[str, bytes]:
    """Read regular files, rejecting links, duplicate names, and escaping paths."""
    files = {}
    if zipfile.is_zipfile(io.BytesIO(data)):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for member in archive.infolist():
                path = PurePosixPath(member.filename)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"Unsafe archive path: {member.filename}")
                if member.is_dir():
                    continue
                if (member.external_attr >> 16) & 0o170000 == 0o120000 or str(path) in files:
                    raise ValueError(f"Unsupported archive entry: {member.filename}")
                files[str(path)] = archive.read(member)
        return files
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe archive path: {member.name}")
            if member.isdir():
                continue
            if not member.isfile() or str(path) in files:
                raise ValueError(f"Unsupported archive entry: {member.name}")
            files[str(path)] = archive.extractfile(member).read()
    return files


def hashes(files: dict[str, bytes], *, normalize_manifest: bool = False) -> dict[str, str]:
    result = {}
    for name, data in files.items():
        if normalize_manifest and name == "idf_component.yml":
            # Component Manager rewrites YAML when packing for upload. Compare all
            # manifest values, preserving scalar types and list order, instead of layout.
            data = json.dumps(yaml.safe_load(data), sort_keys=True, allow_nan=False).encode("utf-8")
        result[name] = hashlib.sha256(data).hexdigest()
    return result


def require_same_files(actual: dict[str, str], expected: dict[str, str]) -> None:
    if actual != expected:
        changed = sorted(name for name in actual.keys() | expected.keys()
                         if actual.get(name) != expected.get(name))
        raise ValueError(f"Component contents differ: {', '.join(changed)}")


def load_package(path: Path) -> tuple[dict, dict[str, bytes]]:
    inventory = json.loads(path.read_text(encoding="utf-8"))
    if inventory["schema_version"] != 1 or inventory["component"] != COMPONENT_NAME:
        raise ValueError("Unexpected component inventory identity")
    archive = path.parent / inventory["archive"]
    if archive.parent.resolve() != path.parent.resolve():
        raise ValueError("Archive must be beside its inventory")
    data = archive.read_bytes()
    if hashlib.sha256(data).hexdigest() != inventory["sha256"]:
        raise ValueError("Component archive checksum mismatch")
    files = archive_files(data)
    require_same_files(hashes(files), inventory["files"])
    manifest = yaml.safe_load(files["idf_component.yml"])
    metadata = json.loads(files["sdk-metadata.json"])
    example = yaml.safe_load(files["examples/basic/main/idf_component.yml"])
    if (manifest["version"] != inventory["version"]
            or metadata != {"component": COMPONENT_NAME, "version": inventory["version"], "commit": inventory["commit"]}
            or example["dependencies"][COMPONENT_NAME]["version"] != inventory["version"]):
        raise ValueError("Component, metadata, example, and inventory identities differ")
    return inventory, files


def download(url: str) -> bytes:
    with urlopen(url, timeout=60) as response:
        return response.read()


def registry_files(registry: str, inventory: dict, expected_files: dict[str, bytes], wait_seconds: int,
                   allow_missing: bool = False) -> dict | None:
    # The client shares an HTTP cache across instances. Poll the live index so
    # a response fetched before publication cannot hide the version forever.
    os.environ["IDF_COMPONENT_CACHE_HTTP_REQUESTS"] = "0"
    from idf_component_tools.registry.client_errors import ComponentNotFound, VersionNotFound
    from idf_component_tools.registry.multi_storage_client import MultiStorageClient

    expected_hashes = hashes(expected_files, normalize_manifest=True)
    deadline = time.monotonic() + wait_seconds
    while True:
        client = MultiStorageClient(registry_url=registry)
        # MultiStorageClient otherwise treats an unreachable registry as an empty source.
        if client.registry_storage_client is None:
            raise RuntimeError(f"Registry storage is unavailable: {registry}")
        try:
            metadata = client.component(COMPONENT_NAME, f"=={inventory['version']}")
            files = archive_files(download(metadata["download_url"]))
            require_same_files(hashes(files, normalize_manifest=True), expected_hashes)
            print(f"Verified {COMPONENT_NAME} {inventory['version']} at {registry}", flush=True)
            return metadata
        except (ComponentNotFound, VersionNotFound):
            if time.monotonic() >= deadline:
                if allow_missing:
                    return None
                raise RuntimeError(f"Registry did not publish version {inventory['version']} in time") from None
            print("Waiting for registry processing...", flush=True)
            time.sleep(min(15, max(0, deadline - time.monotonic())))


def write_files(destination: Path, files: dict[str, bytes]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def registry_example(registry: str, inventory: dict, files: dict[str, bytes],
                     wait_seconds: int) -> dict[str, bytes]:
    metadata = registry_files(registry, inventory, files, wait_seconds)
    example_url = next(item["url"] for item in metadata["examples"] if item["name"] == "basic")
    example = archive_files(download(example_url))
    prefix = "examples/basic/"
    expected = {name.removeprefix(prefix): data for name, data in files.items() if name.startswith(prefix)}
    require_same_files(hashes(example), hashes(expected))
    return example


def prepare(args: argparse.Namespace, inventory: dict, files: dict[str, bytes]) -> None:
    destination = args.destination.resolve()
    if destination.is_relative_to(REPO_ROOT):
        raise ValueError("The verification project must be outside the checkout")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"Verification destination must be empty: {destination}")
    project = destination / "basic"
    example_prefix = "examples/basic/"
    expected_example = {name.removeprefix(example_prefix): data for name, data in files.items()
                        if name.startswith(example_prefix)}
    if args.registry_url:
        if args.example_dir:
            downloaded_example = {str(path.relative_to(args.example_dir)): path.read_bytes()
                                  for path in args.example_dir.rglob("*") if path.is_file()}
            require_same_files(hashes(downloaded_example), hashes(expected_example))
        else:
            downloaded_example = registry_example(args.registry_url, inventory, files, args.wait_seconds)
        write_files(project, downloaded_example)
    else:
        write_files(destination / "francescopace__espectre", files)
        write_files(project, expected_example)

    manifest_path = project / "main" / "idf_component.yml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    dependency = manifest["dependencies"][COMPONENT_NAME]
    if args.registry_url:
        # Scope staging to ESPectre; Espressif dependencies still use production.
        dependency["registry_url"] = args.registry_url
    else:
        dependency["override_path"] = "../../francescopace__espectre"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    defaults = project / "sdkconfig.defaults"
    profiles = () if args.profile == "minimal" else (args.profile,)
    if args.profile == "all":
        profiles = SDK_SERVICE_PROFILES
    with defaults.open("a", encoding="utf-8") as output:
        # Keep the sensing startup reachable to the linker in credential-free CI.
        output.write('\nCONFIG_ESPECTRE_EXAMPLE_WIFI_SSID="sdk-ci-placeholder"\n')
        for profile in profiles:
            output.write(f"CONFIG_ESPECTRE_SDK_ENABLE_{profile.upper()}=y\n")
        if args.target == "esp32s2" and args.profile == "all":
            output.write("CONFIG_ESPECTRE_TINYUSB_PRIMARY_CONSOLE=y\n"
                         "CONFIG_TINYUSB_CDC_ENABLED=y\n"
                         "CONFIG_ESP_CONSOLE_NONE=y\n")
    print(f"Prepared {args.target}/{args.profile}: {project}")


def verify_install(project: Path, inventory: dict, files: dict[str, bytes], target: str) -> None:
    lock = yaml.safe_load((project / "dependencies.lock").read_text(encoding="utf-8"))
    dependency = lock["dependencies"][COMPONENT_NAME]
    if str(dependency["version"]) != inventory["version"]:
        raise ValueError("The resolver selected a different SDK version")
    if dependency["source"]["type"] != "service":
        raise ValueError("The SDK was not installed from the registry")
    installed = project / "managed_components" / "francescopace__espectre"
    actual = {str(path.relative_to(installed)): path.read_bytes()
              for path in installed.rglob("*") if path.is_file()
              and str(path.relative_to(installed)) not in {".component_hash", "CHECKSUMS.json"}}
    require_same_files(hashes(actual, normalize_manifest=True), hashes(files, normalize_manifest=True))
    if lock["target"] != target:
        raise ValueError("Unexpected installed target")
    print(f"Verified clean registry installation on {target}")


def verify_dependencies(project: Path) -> None:
    """Require the isolated example to resolve only the enabled external stacks."""
    config = json.loads((project / "build/config/sdkconfig.json").read_text(encoding="utf-8"))
    lock = yaml.safe_load((project / "dependencies.lock").read_text(encoding="utf-8"))
    for line in (project / "sdkconfig.defaults").read_text(encoding="utf-8").splitlines():
        if line.startswith(("CONFIG_ESPECTRE_SDK_ENABLE_", "CONFIG_ESPECTRE_TINYUSB_PRIMARY_CONSOLE=")):
            option, value = line.removeprefix("CONFIG_").split("=", 1)
            if config.get(option, False) != (value == "y"):
                raise ValueError(f"SDK build did not apply requested option: {line}")
    expected = set()
    if config.get("ESPECTRE_SDK_ENABLE_DIRECT"):
        expected.add("espressif/mdns")
    if config.get("ESPECTRE_TINYUSB_PRIMARY_CONSOLE"):
        expected.update(("espressif/esp_tinyusb", "espressif/tinyusb"))
    actual = set(lock["dependencies"]) - {COMPONENT_NAME, "idf"}
    if actual != expected:
        raise ValueError(f"Unexpected SDK dependencies: expected {sorted(expected)}, got {sorted(actual)}")
    print(f"Verified optional dependencies: {', '.join(sorted(actual)) or 'none'}")


def build(args: argparse.Namespace, inventory: dict, files: dict[str, bytes]) -> None:
    project = args.destination.resolve() / "basic"
    if project.is_relative_to(REPO_ROOT):
        raise ValueError("The build must run outside the checkout")
    if args.docker:
        # Read only the toolchain pin; the checkout is never mounted in the container.
        sys.path.insert(0, str(REPO_ROOT / "src" / "python"))
        from espectre_cli.idf_container import IDF_DOCKER_IMAGE

        command = ["docker", "run", "--rm", "-v", f"{project.parent}:/consumer",
                   "-w", "/consumer/basic", IDF_DOCKER_IMAGE,
                   "idf.py", f"-DIDF_TARGET={args.target}", "build"]
    else:
        command = ["idf.py", f"-DIDF_TARGET={args.target}", "build"]
    subprocess.run(command, cwd=project, check=True)
    verify_dependencies(project)
    if args.registry_url:
        verify_install(project, inventory, files, args.target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "extract", "fetch-example", "prepare", "build", "verify-install"))
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--expected-commit", help="Require the inventory to match this source commit.")
    parser.add_argument("--snapshot-channel", choices=("preview", "develop"),
                        help="Require a snapshot version bound to this channel and --expected-commit.")
    parser.add_argument("--registry-url")
    parser.add_argument("--wait-seconds", type=int, default=600)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--example-dir", type=Path,
                        help="Use the registry example already downloaded and verified by this workflow.")
    parser.add_argument("--target", choices=("esp32", "esp32s2", "esp32s3", "esp32c3", "esp32c5", "esp32c6"),
                        default="esp32c3")
    parser.add_argument("--profile", choices=("minimal", *SDK_SERVICE_PROFILES, "all"),
                        default="minimal")
    parser.add_argument("--docker", action="store_true")
    args = parser.parse_args()
    if args.command == "fetch-example" and not args.registry_url:
        parser.error("fetch-example requires --registry-url")
    if args.example_dir and (args.command != "prepare" or not args.registry_url):
        parser.error("--example-dir requires prepare with --registry-url")
    inventory, files = load_package(args.inventory)
    if args.expected_commit and inventory["commit"] != args.expected_commit:
        raise ValueError("Component source commit does not match the validated CI run")
    if args.snapshot_channel:
        if not args.expected_commit:
            parser.error("--snapshot-channel requires --expected-commit")
        branch = {"preview": "main", "develop": "develop"}[args.snapshot_channel]
        suffix = f".{branch}"
        if not inventory["version"].endswith(suffix):
            raise ValueError("Component version does not match the snapshot channel")
    if args.command == "check":
        exists = False
        if args.registry_url:
            exists = registry_files(args.registry_url, inventory, files,
                                    args.wait_seconds, args.allow_missing) is not None
        if os.environ.get("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
                output.write(f"exists={str(exists).lower()}\n")
        print(f"Verified archive: {inventory['archive']} ({len(files)} files)")
    elif args.destination is None:
        parser.error("--destination is required for this command")
    elif args.command == "fetch-example":
        if args.destination.exists() and any(args.destination.iterdir()):
            raise ValueError("Example destination must be empty")
        write_files(args.destination, registry_example(args.registry_url, inventory, files, args.wait_seconds))
    elif args.command == "extract":
        if args.destination.exists() and any(args.destination.iterdir()):
            raise ValueError("Extraction destination must be empty")
        write_files(args.destination, files)
    elif args.command == "prepare":
        prepare(args, inventory, files)
    elif args.command == "build":
        build(args, inventory, files)
    else:
        verify_install(args.destination / "basic", inventory, files, args.target)


if __name__ == "__main__":
    main()
