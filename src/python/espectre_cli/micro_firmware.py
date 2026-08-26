# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Reproducible Micro-ESPectre firmware build support."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from .common import FIRMWARE_CACHE_DIR, MICROPYTHON_FIRMWARE_BUILD, REPO_ROOT
from .idf import (
    resolve_idf_build_backend,
    resolve_idf_build_dir_name,
    run_in_idf_environment,
)
from .idf_container import IDF_VERSION, run_toolchain_container


MICROPYTHON_REPOSITORY = "https://github.com/micropython/micropython.git"
MICROPYTHON_COMMIT = "1c3c201149f37fe8d81246191b3127bb198d6306"
MICROPYTHON_LIB_REPOSITORY = "https://github.com/micropython/micropython-lib.git"
MICROPYTHON_LIB_COMMIT = "ee4bb8ff139e24c42b739935fbd8ec7c4d061e02"
PROJECT_FIRMWARE_BOARDS = {
    "esp32": "ESP32_MICRO_ESPECTRE",
    "c3": "ESP32C3_MICRO_ESPECTRE",
    "c5": "ESP32C5_MICRO_ESPECTRE",
    "c6": "ESP32C6_MICRO_ESPECTRE",
    "s2": "ESP32S2_MICRO_ESPECTRE",
    "s3": "ESP32S3_MICRO_ESPECTRE",
}
PROJECT_FIRMWARE_NAMES = {
    "esp32": f"ESP32_GENERIC-{MICROPYTHON_FIRMWARE_BUILD}-espectre.bin",
    "c3": f"ESP32_GENERIC_C3-{MICROPYTHON_FIRMWARE_BUILD}-espectre.bin",
    "c5": f"ESP32_GENERIC_C5-{MICROPYTHON_FIRMWARE_BUILD}-espectre.bin",
    "c6": f"ESP32_GENERIC_C6-{MICROPYTHON_FIRMWARE_BUILD}-espectre.bin",
    "s2": f"ESP32_GENERIC_S2-{MICROPYTHON_FIRMWARE_BUILD}-espectre.bin",
    "s3": f"ESP32_GENERIC_S3-{MICROPYTHON_FIRMWARE_BUILD}-espectre.bin",
}


def _checkout_pinned_repository(url: str, commit: str, destination: Path) -> None:
    """Create or validate a cached checkout at one exact revision."""
    if destination.is_dir():
        current = subprocess.run(
            ["git", "-C", str(destination), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if current == commit:
            return
        raise RuntimeError(
            f"Cached checkout has unexpected revision: {destination} ({current})"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--filter=blob:none", "--no-checkout", url, str(destination)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(destination), "fetch", "--depth", "1", "origin", commit],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(destination), "checkout", "--detach", commit],
        check=True,
    )


def _read_idf_version(idf_path: Path) -> str:
    """Read and normalize the ESP-IDF release used by a local backend."""
    version_file = idf_path / "version.txt"
    if not version_file.is_file():
        raise RuntimeError(f"ESP-IDF version file is missing: {version_file}")
    return version_file.read_text(encoding="utf-8").strip().removeprefix("v")


def _align_idf_lockfile(micropython_dir: Path, chip: str, idf_version: str) -> None:
    """Align the cached MicroPython component lock with the ESP-IDF 5.5 patch release."""
    if re.fullmatch(r"5\.5\.\d+", idf_version) is None:
        raise RuntimeError(f"ESP-IDF 5.5 is required; found {idf_version or 'unknown'}")

    lockfile = (
        micropython_dir
        / "ports"
        / "esp32"
        / "lockfiles"
        / f"dependencies.lock.esp32{'' if chip == 'esp32' else chip}"
    )
    lines = lockfile.read_text(encoding="utf-8").splitlines(keepends=True)
    in_idf_dependency = False
    for index, line in enumerate(lines):
        if line.rstrip("\r\n") == "  idf:":
            in_idf_dependency = True
            continue
        if in_idf_dependency and line.startswith("    version: "):
            newline = "\r\n" if line.endswith("\r\n") else "\n"
            lines[index] = f"    version: {idf_version}{newline}"
            lockfile.write_text("".join(lines), encoding="utf-8")
            return
        if in_idf_dependency and not line.startswith("    "):
            break
    raise RuntimeError(f"ESP-IDF dependency entry is missing: {lockfile}")


def _configure_project_csi_capture(micropython_dir: Path) -> None:
    """Restrict the project firmware's Wi-Fi 6 CSI capture to HT20 frames."""
    source_path = micropython_dir / "ports" / "esp32" / "network_wlan_csi.c"
    source = source_path.read_text(encoding="utf-8")
    upstream_setting = ".acquire_csi_legacy = 1,"
    project_setting = ".acquire_csi_legacy = 0,"
    if upstream_setting in source:
        source_path.write_text(
            source.replace(upstream_setting, project_setting, 1),
            encoding="utf-8",
        )
        return
    if project_setting not in source:
        raise RuntimeError(
            f"MicroPython CSI legacy-capture setting is missing: {source_path}"
        )


def _configure_project_wifi_band_mode(micropython_dir: Path) -> None:
    """Expose the ESP-IDF band selector required by dual-band WLAN targets."""
    source_path = micropython_dir / "ports" / "esp32" / "network_wlan.c"
    source = source_path.read_text(encoding="utf-8")
    setter = """                    case MP_QSTR_band_mode: {
                        esp_exceptions(esp_wifi_set_band_mode(mp_obj_get_int(kwargs->table[i].value)));
                        break;
                    }
"""
    constant = """    { MP_ROM_QSTR(MP_QSTR_BAND_MODE_2G_ONLY), MP_ROM_INT(WIFI_BAND_MODE_2G_ONLY) },
"""
    if setter in source and constant in source:
        return

    bandwidth_setter = """                    case MP_QSTR_bandwidth: {
                        esp_exceptions(esp_wifi_set_bandwidth(self->if_id, mp_obj_get_int(kwargs->table[i].value)));
                        break;
                    }
"""
    bandwidth_constant = """    { MP_ROM_QSTR(MP_QSTR_BANDWIDTH_20), MP_ROM_INT(WIFI_BW20) },
"""
    if setter not in source and bandwidth_setter not in source:
        raise RuntimeError(
            f"MicroPython WLAN band-mode setter anchor is missing: {source_path}"
        )
    if constant not in source and bandwidth_constant not in source:
        raise RuntimeError(
            f"MicroPython WLAN band-mode constant anchor is missing: {source_path}"
        )

    if setter not in source:
        source = source.replace(
            bandwidth_setter,
            setter + bandwidth_setter,
            1,
        )
    if constant not in source:
        source = source.replace(
            bandwidth_constant,
            constant + bandwidth_constant,
            1,
        )
    source_path.write_text(source, encoding="utf-8")


def _write_manifest(manifest_path: Path) -> None:
    """Freeze only the ESP32 boot and filesystem helpers, never the application."""
    manifest_path.write_text(
        'freeze("$(PORT_DIR)/modules", ("_boot.py", "flashbdev.py", "inisetup.py"))\n',
        encoding="utf-8",
    )


def _stage_firmware_support(source_dir: Path, destination: Path) -> None:
    """Stage the custom board and native modules used by the project image."""
    support_dir = source_dir / "firmware"
    if not support_dir.is_dir():
        raise RuntimeError(f"Project firmware support directory is missing: {support_dir}")
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(support_dir, destination)


def build_project_firmware(
    source_dir: Path,
    *,
    chip: str = "esp32",
    clean: bool = False,
    cache_dir: Path = FIRMWARE_CACHE_DIR,
    backend: str = "auto",
    pull_policy: str = "ask",
) -> Path:
    """Build a lean project firmware used with filesystem bytecode."""
    board = PROJECT_FIRMWARE_BOARDS.get(chip)
    firmware_name = PROJECT_FIRMWARE_NAMES.get(chip)
    if board is None or firmware_name is None:
        raise ValueError(f"Unsupported project firmware chip: {chip}")

    resolved_backend = resolve_idf_build_backend(backend, pull_policy)

    cache_dir.mkdir(parents=True, exist_ok=True)
    workspace = cache_dir / "micro-esp32"
    micropython_dir = workspace / "micropython"
    micropython_lib_dir = workspace / "micropython-lib"
    build_dir_name = resolve_idf_build_dir_name(
        workspace,
        chip,
        container=resolved_backend.mode == "docker",
    )
    assert build_dir_name is not None
    build_dir = workspace / build_dir_name
    support_root = workspace / "firmware-support"
    manifest_path = workspace / "manifest.py"

    _checkout_pinned_repository(
        MICROPYTHON_REPOSITORY,
        MICROPYTHON_COMMIT,
        micropython_dir,
    )
    _checkout_pinned_repository(
        MICROPYTHON_LIB_REPOSITORY,
        MICROPYTHON_LIB_COMMIT,
        micropython_lib_dir,
    )
    _configure_project_csi_capture(micropython_dir)
    if chip == "c5":
        _configure_project_wifi_band_mode(micropython_dir)
    _stage_firmware_support(source_dir, support_root)
    _write_manifest(manifest_path)

    if clean and build_dir.exists():
        shutil.rmtree(build_dir)
    else:
        # sdkconfig defaults are copied from the repository support tree. Force
        # CMake to resolve them again so an incremental build cannot silently
        # retain an older firmware profile.
        for generated_config in ("sdkconfig", "sdkconfig.old"):
            (build_dir / generated_config).unlink(missing_ok=True)

    if resolved_backend.mode == "local":
        idf_environment = resolved_backend.idf_environment
        assert idf_environment is not None
        idf_path = idf_environment.install_dir
        if idf_path is None and idf_environment.idf_path_entry:
            idf_path = Path(idf_environment.idf_path_entry).resolve().parents[1]
        if idf_path is None:
            raise RuntimeError("The resolved local ESP-IDF path is unavailable")
        idf_version = _read_idf_version(idf_path)
    else:
        idf_version = IDF_VERSION
    _align_idf_lockfile(micropython_dir, chip, idf_version)
    jobs = str(max(1, min(8, os.cpu_count() or 1)))
    if resolved_backend.mode == "docker":
        build_root = Path("/work") / workspace.resolve().relative_to(REPO_ROOT.resolve())
    else:
        build_root = workspace.resolve()
    commands = [
        ["make", "-C", "micropython/mpy-cross", f"-j{jobs}"],
        [
            "cmake",
            "-S",
            "micropython/ports/esp32",
            "-B",
            build_dir.name,
            "-G",
            "Ninja",
            f"-DMICROPY_BOARD={board}",
            f"-DMICROPY_BOARD_DIR={build_root / 'firmware-support' / 'boards' / board}",
            f"-DMICROPY_FROZEN_MANIFEST={build_root / 'manifest.py'}",
            f"-DMICROPY_LIB_DIR={build_root / 'micropython-lib'}",
            f"-DUSER_C_MODULES={build_root / 'firmware-support' / 'native_components' / 'micropython.cmake'}",
            "-DMICROPY_PY_BTREE=0",
        ],
        [
            "cmake",
            "--build",
            build_dir.name,
            f"-j{jobs}",
        ],
        [
            "python",
            "micropython/ports/esp32/makeimg.py",
            f"{build_dir.name}/sdkconfig",
            f"{build_dir.name}/bootloader/bootloader.bin",
            f"{build_dir.name}/partition_table/partition-table.bin",
            f"{build_dir.name}/micropython.bin",
            f"{build_dir.name}/firmware.bin",
            f"{build_dir.name}/firmware.uf2",
        ],
    ]
    if resolved_backend.mode == "docker":
        run_toolchain_container(
            frontend="micro",
            workdir=workspace,
            commands=commands,
            repo_root=REPO_ROOT,
            pull_policy=pull_policy,
            docker=resolved_backend.docker,
        )
    else:
        assert resolved_backend.idf_environment is not None
        for command in commands:
            run_in_idf_environment(
                command,
                resolved_backend.idf_environment,
                cwd=workspace,
            )

    firmware_path = cache_dir / firmware_name
    shutil.copy2(build_dir / "firmware.bin", firmware_path)
    return firmware_path
