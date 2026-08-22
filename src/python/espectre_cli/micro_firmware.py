# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Reproducible Micro-ESPectre firmware build support."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from .common import FIRMWARE_CACHE_DIR, MICROPYTHON_FIRMWARE_BUILD


MICROPYTHON_REPOSITORY = "https://github.com/micropython/micropython.git"
MICROPYTHON_COMMIT = "1c3c201149f37fe8d81246191b3127bb198d6306"
MICROPYTHON_LIB_REPOSITORY = "https://github.com/micropython/micropython-lib.git"
MICROPYTHON_LIB_COMMIT = "ee4bb8ff139e24c42b739935fbd8ec7c4d061e02"
PROJECT_FIRMWARE_BOARDS = {
    "esp32": "ESP32_MICRO_ESPECTRE",
    "c3": "ESP32C3_MICRO_ESPECTRE",
    "c5": "ESP32C5_MICRO_ESPECTRE",
    "c6": "ESP32C6_MICRO_ESPECTRE",
    "s3": "ESP32S3_MICRO_ESPECTRE",
}
PROJECT_FIRMWARE_NAMES = {
    "esp32": f"ESP32_GENERIC-{MICROPYTHON_FIRMWARE_BUILD}-espectre.bin",
    "c3": f"ESP32_GENERIC_C3-{MICROPYTHON_FIRMWARE_BUILD}-espectre.bin",
    "c5": f"ESP32_GENERIC_C5-{MICROPYTHON_FIRMWARE_BUILD}-espectre.bin",
    "c6": f"ESP32_GENERIC_C6-{MICROPYTHON_FIRMWARE_BUILD}-espectre.bin",
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


def _resolve_idf_build_environment() -> dict[str, str]:
    """Resolve the ESP-IDF 5.5 host environment used by the CMake build."""
    env = os.environ.copy()
    home = Path.home()
    idf_path = Path(env["IDF_PATH"]).expanduser() if env.get("IDF_PATH") else None
    if idf_path is None or not idf_path.is_dir():
        candidates = (
            home / ".platformio" / "packages" / "framework-espidf",
            home / "esp" / "esp-idf",
        )
        idf_path = next((path for path in candidates if path.is_dir()), None)
    if idf_path is None:
        raise RuntimeError(
            "ESP-IDF was not found; activate an ESP-IDF 5.5 shell or install the repository toolchain"
        )

    python_env = (
        Path(env["IDF_PYTHON_ENV_PATH"]).expanduser()
        if env.get("IDF_PYTHON_ENV_PATH")
        else None
    )
    if python_env is None or not python_env.is_dir():
        python_envs = list(
            (home / ".espressif" / "python_env").glob("idf5.5_py*_env")
        )

        def python_version(path: Path) -> tuple[int, int]:
            match = re.search(r"_py(\d+)\.(\d+)_env$", path.name)
            return (int(match.group(1)), int(match.group(2))) if match else (0, 0)

        python_env = max(python_envs, key=python_version) if python_envs else None
    if python_env is None:
        raise RuntimeError(
            "ESP-IDF 5.5 Python environment was not found; run the ESP-IDF installer first"
        )

    tool_bins = sorted(
        (
            path
            for path in (home / ".espressif" / "tools").glob("**/bin")
            if path.is_dir()
        ),
        reverse=True,
    )
    path_entries = [str(python_env / ("Scripts" if os.name == "nt" else "bin"))]
    path_entries.extend(str(path) for path in tool_bins)
    path_entries.append(env.get("PATH", ""))
    env["IDF_PATH"] = str(idf_path)
    env["IDF_PYTHON_ENV_PATH"] = str(python_env)
    env["PATH"] = os.pathsep.join(path_entries)
    return env


def _align_idf_lockfile(micropython_dir: Path, chip: str, idf_path: Path) -> None:
    """Align the cached MicroPython component lock with the ESP-IDF 5.5 patch release."""
    version_file = idf_path / "version.txt"
    if not version_file.is_file():
        raise RuntimeError(f"ESP-IDF version file is missing: {version_file}")
    idf_version = version_file.read_text(encoding="utf-8").strip()
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
) -> Path:
    """Build a lean project firmware used with filesystem bytecode."""
    board = PROJECT_FIRMWARE_BOARDS.get(chip)
    firmware_name = PROJECT_FIRMWARE_NAMES.get(chip)
    if board is None or firmware_name is None:
        raise ValueError(f"Unsupported project firmware chip: {chip}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    workspace = cache_dir / "micro-esp32"
    micropython_dir = workspace / "micropython"
    micropython_lib_dir = workspace / "micropython-lib"
    build_dir = workspace / ("build" if chip == "esp32" else f"build-{chip}")
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

    env = _resolve_idf_build_environment()
    _align_idf_lockfile(micropython_dir, chip, Path(env["IDF_PATH"]))
    jobs = str(max(1, min(8, os.cpu_count() or 1)))
    subprocess.run(
        ["make", "-C", str(micropython_dir / "mpy-cross"), f"-j{jobs}"],
        check=True,
        env=os.environ.copy(),
    )
    subprocess.run(
        [
            "cmake",
            "-S",
            str(micropython_dir / "ports" / "esp32"),
            "-B",
            str(build_dir),
            "-G",
            "Ninja",
            f"-DMICROPY_BOARD={board}",
            "-DMICROPY_BOARD_DIR=" + str(support_root / "boards" / board),
            f"-DMICROPY_FROZEN_MANIFEST={manifest_path}",
            f"-DMICROPY_LIB_DIR={micropython_lib_dir}",
            f"-DUSER_C_MODULES={support_root / 'native_components' / 'micropython.cmake'}",
            "-DMICROPY_PY_BTREE=0",
        ],
        check=True,
        env=env,
    )
    subprocess.run(
        ["cmake", "--build", str(build_dir), f"-j{jobs}"],
        check=True,
        env=env,
    )

    python_executable = Path(env["IDF_PYTHON_ENV_PATH"]) / (
        "Scripts/python.exe" if os.name == "nt" else "bin/python"
    )
    subprocess.run(
        [
            str(python_executable),
            str(micropython_dir / "ports" / "esp32" / "makeimg.py"),
            str(build_dir / "sdkconfig"),
            str(build_dir / "bootloader" / "bootloader.bin"),
            str(build_dir / "partition_table" / "partition-table.bin"),
            str(build_dir / "micropython.bin"),
            str(build_dir / "firmware.bin"),
            str(build_dir / "firmware.uf2"),
        ],
        check=True,
        env=env,
    )

    firmware_path = cache_dir / firmware_name
    shutil.copy2(build_dir / "firmware.bin", firmware_path)
    return firmware_path
