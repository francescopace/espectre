# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Reproducible Micro-ESPectre firmware build support."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import replace
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
MICROPYTHON_PATCH_REVISION = "fixed-csi-records-v1"
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


def _prepare_micropython_patch_revision(micropython_dir: Path) -> Path:
    """Restore the pinned sources once when the project patch set changes."""
    stamp_path = micropython_dir / ".espectre-patch-revision"
    if (
        stamp_path.is_file()
        and stamp_path.read_text(encoding="utf-8").strip()
        == MICROPYTHON_PATCH_REVISION
    ):
        return stamp_path

    source_diff = subprocess.run(
        ["git", "-C", str(micropython_dir), "diff", "--binary", "--", "."],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if source_diff:
        backup_path = (
            micropython_dir.parent
            / f"micropython-before-{MICROPYTHON_PATCH_REVISION}.patch"
        )
        if not backup_path.exists():
            backup_path.write_text(source_diff, encoding="utf-8")

    subprocess.run(
        [
            "git",
            "-C",
            str(micropython_dir),
            "restore",
            "--worktree",
            "--source=HEAD",
            "--",
            ".",
        ],
        check=True,
    )
    stamp_path.unlink(missing_ok=True)
    return stamp_path


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


def _configure_project_csi_fixed_records(micropython_dir: Path) -> None:
    """Use one fixed ring stride selected by the runtime payload bound."""
    source_path = micropython_dir / "ports" / "esp32" / "network_wlan_csi.c"
    source = source_path.read_text(encoding="utf-8")
    required = (
        "static size_t wifi_csi_record_size(const csi_state_t *state)",
        "offsetof(csi_frame_t, data) + state->max_data_len",
        "MP_QSTR_max_data_len",
        "wifi_csi_record_size(state) * state->buffer_size + 1",
        "ringbuf_put_bytes(&state->ringbuffer, (uint8_t *)&frame, wifi_csi_record_size(state))",
        "ringbuf_get_bytes(&state->ringbuffer, (uint8_t *)frame, wifi_csi_record_size(state))",
        "available / wifi_csi_record_size(state)",
    )
    if all(token in source for token in required):
        return
    if "csi_frame_header_t" in source:
        raise RuntimeError(
            f"variable-record CSI source cannot be configured as fixed records: {source_path}"
        )

    native_allocation = (
        "size_t ring_size = sizeof(csi_frame_t) * state->buffer_size + 1;"
    )
    managed_allocation = (
        "ringbuf_alloc(&state->ringbuffer, sizeof(csi_frame_t) * state->buffer_size);"
    )
    if native_allocation in source:
        source = source.replace(
            native_allocation,
            "size_t ring_size = wifi_csi_record_size(state) * state->buffer_size + 1;",
            1,
        )
    elif managed_allocation in source:
        source = source.replace(
            managed_allocation,
            "ringbuf_alloc(&state->ringbuffer, "
            "wifi_csi_record_size(state) * state->buffer_size + 1);",
            1,
        )
    else:
        raise RuntimeError(
            f"MicroPython fixed-record CSI allocation anchor is missing: {source_path}"
        )

    replacements = (
        (
            '#include "modnetwork.h"\n#include <stdint.h>',
            '#include "modnetwork.h"\n#include <stddef.h>\n#include <stdint.h>',
        ),
        (
            "// ringbuf_t uses uint16_t for the byte size, so keep the Python-visible limit\n"
            "// within the maximum addressable ringbuffer capacity.\n"
            "#define CSI_MAX_BUFFER_SIZE ((UINT16_MAX - 1) / sizeof(csi_frame_t))\n\n",
            "",
        ),
        (
            "    uint16_t buffer_size;\n"
            "    volatile uint32_t dropped;",
            "    uint16_t buffer_size;\n"
            "    uint16_t max_data_len;\n"
            "    volatile uint32_t dropped;",
        ),
        (
            "} csi_state_t;\n\n"
            "static csi_state_t *wifi_csi_get_state(void) {",
            "} csi_state_t;\n\n"
            "static size_t wifi_csi_record_size(const csi_state_t *state) {\n"
            "    return offsetof(csi_frame_t, data) + state->max_data_len;\n"
            "}\n\n"
            "static csi_state_t *wifi_csi_get_state(void) {",
        ),
        (
            "        state->buffer_size = MICROPY_PY_NETWORK_WLAN_CSI_DEFAULT_BUFFER_SIZE;",
            "        state->buffer_size = MICROPY_PY_NETWORK_WLAN_CSI_DEFAULT_BUFFER_SIZE;\n"
            "        state->max_data_len = CSI_MAX_DATA_LEN;",
        ),
        (
            "frame.len = info->len > CSI_MAX_DATA_LEN ? CSI_MAX_DATA_LEN : info->len;",
            "frame.len = info->len > state->max_data_len ? state->max_data_len : info->len;",
        ),
        (
            "ringbuf_put_bytes(&state->ringbuffer, (uint8_t *)&frame, sizeof(frame))",
            "ringbuf_put_bytes(&state->ringbuffer, (uint8_t *)&frame, wifi_csi_record_size(state))",
        ),
        (
            "ringbuf_get_bytes(&state->ringbuffer, (uint8_t *)frame, sizeof(*frame))",
            "ringbuf_get_bytes(&state->ringbuffer, (uint8_t *)frame, wifi_csi_record_size(state))",
        ),
        (
            "    static const mp_arg_t allowed_args[] = {\n"
            "        { MP_QSTR_buffer_size, MP_ARG_KW_ONLY | MP_ARG_INT, {.u_int = MICROPY_PY_NETWORK_WLAN_CSI_DEFAULT_BUFFER_SIZE} },\n"
            "    };",
            "    enum { ARG_buffer_size, ARG_max_data_len };\n"
            "    static const mp_arg_t allowed_args[] = {\n"
            "        { MP_QSTR_buffer_size, MP_ARG_KW_ONLY | MP_ARG_INT, {.u_int = MICROPY_PY_NETWORK_WLAN_CSI_DEFAULT_BUFFER_SIZE} },\n"
            "        { MP_QSTR_max_data_len, MP_ARG_KW_ONLY | MP_ARG_INT, {.u_int = CSI_MAX_DATA_LEN} },\n"
            "    };",
        ),
        (
            "    mp_int_t buffer_size = parsed_args[0].u_int;\n"
            "    if (buffer_size < 1 || buffer_size > CSI_MAX_BUFFER_SIZE) {\n"
            "        mp_raise_ValueError(MP_ERROR_TEXT(\"buffer_size out of range\"));\n"
            "    }",
            "    mp_int_t max_data_len = parsed_args[ARG_max_data_len].u_int;\n"
            "    if (max_data_len < 1 || max_data_len > CSI_MAX_DATA_LEN) {\n"
            "        mp_raise_ValueError(MP_ERROR_TEXT(\"max_data_len out of range\"));\n"
            "    }\n\n"
            "    size_t record_size = offsetof(csi_frame_t, data) + max_data_len;\n"
            "    size_t max_buffer_size = (UINT16_MAX - 1) / record_size;\n"
            "    mp_int_t buffer_size = parsed_args[ARG_buffer_size].u_int;\n"
            "    if (buffer_size < 1 || (size_t)buffer_size > max_buffer_size) {\n"
            "        mp_raise_ValueError(MP_ERROR_TEXT(\"buffer_size out of range\"));\n"
            "    }",
        ),
        (
            "    state->buffer_size = buffer_size;\n"
            "    esp_exceptions(wifi_csi_enable(state));",
            "    state->buffer_size = buffer_size;\n"
            "    state->max_data_len = max_data_len;\n"
            "    esp_exceptions(wifi_csi_enable(state));",
        ),
        (
            "return MP_OBJ_NEW_SMALL_INT(available / sizeof(csi_frame_t));",
            "return MP_OBJ_NEW_SMALL_INT(available / wifi_csi_record_size(state));",
        ),
    )
    for original, replacement in replacements:
        if original not in source:
            raise RuntimeError(
                f"MicroPython fixed-record CSI anchor is missing: {source_path}"
            )
        source = source.replace(original, replacement, 1)

    if not all(token in source for token in required):
        raise RuntimeError(
            f"MicroPython fixed-record CSI layout is incomplete: {source_path}"
        )
    source_path.write_text(source, encoding="utf-8")




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
    """Stage the firmware build support used by the project image."""
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
    patch_stamp_path = _prepare_micropython_patch_revision(micropython_dir)
    _configure_project_csi_capture(micropython_dir)
    _configure_project_csi_fixed_records(micropython_dir)
    if chip == "c5":
        _configure_project_wifi_band_mode(micropython_dir)
    patch_stamp_path.write_text(MICROPYTHON_PATCH_REVISION + "\n", encoding="utf-8")
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
        core_sdk_root = Path("/work/src/cpp")
    else:
        build_root = workspace.resolve()
        core_sdk_root = (REPO_ROOT / "src" / "cpp").resolve()
    core_component_dir = (
        build_root / "firmware-support" / "components" / "espectre_core"
    )
    core_build_environment = {
        "ESPECTRE_CORE_SDK_ROOT": str(core_sdk_root),
    }
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
            f"-DESPECTRE_CORE_SDK_ROOT={core_sdk_root}",
            f"-DEXTRA_COMPONENT_DIRS={core_component_dir}",
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
            environment=core_build_environment,
        )
    else:
        assert resolved_backend.idf_environment is not None
        process_env = dict(
            resolved_backend.idf_environment.process_env or os.environ
        )
        process_env.update(core_build_environment)
        idf_environment = replace(
            resolved_backend.idf_environment,
            process_env=process_env,
        )
        for command in commands:
            run_in_idf_environment(
                command,
                idf_environment,
                cwd=workspace,
            )

    firmware_path = cache_dir / firmware_name
    shutil.copy2(build_dir / "firmware.bin", firmware_path)
    return firmware_path
