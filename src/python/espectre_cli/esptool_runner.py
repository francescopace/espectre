# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Central esptool owner for ESPectre CLI flash and run operations."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from .common import Fore, REPO_ROOT, Style, serial_console_mode


ESPTOOL_FLASH_BAUD = "460800"
ESPTOOL_ESP32_FLASH_BAUD = "115200"
IDF_FLASH_OPTION_NAMES = {
    "--flash_mode": "--flash-mode",
    "--flash_freq": "--flash-freq",
    "--flash_size": "--flash-size",
}


def esptool_before_mode(chip: str, port: str) -> str:
    """Return the supported loader-entry mode for a CLI chip alias."""
    if chip == "s2" and serial_console_mode(chip, port) == "usb_cdc":
        return "no-reset"
    return "default-reset"


def run_esptool(args: list[str], *, cwd: Path | None = None) -> None:
    """Run the pinned esptool module and stream its output unchanged."""
    display = " ".join(("esptool", *args))
    print(f"{Fore.CYAN}Command: {display}{Style.RESET_ALL}")
    subprocess.run(
        [sys.executable, "-m", "esptool", *args],
        cwd=cwd,
        check=True,
    )


def esptool_flash_command(*, chip: str, idf_target: str, port: str) -> list[str]:
    """Return the common esptool arguments for one flash session."""
    # esptool 5.3.1 loses the classic ESP32 during its stub SFDP probe at 460800.
    # macOS also raises an IOSSIOSPEED device error when an ESP32-S2 USB CDC
    # watchdog reset disconnects a port configured at 460800. Keeping those
    # sessions at 115200 preserves full-chip erase and post-flash reset support.
    use_safe_baud = idf_target == "esp32" or serial_console_mode(chip, port) == "usb_cdc"
    baud = ESPTOOL_ESP32_FLASH_BAUD if use_safe_baud else ESPTOOL_FLASH_BAUD
    after = "hard-reset" if idf_target == "esp32" else "watchdog-reset"
    args = [
        "--chip",
        idf_target,
        "--port",
        port,
        "--baud",
        baud,
        "--before",
        esptool_before_mode(chip, port),
        "--after",
        after,
        "write-flash",
    ]
    return args


def idf_flash_arguments(build_dir: Path) -> list[str]:
    """Load canonical esptool arguments from ESP-IDF build metadata."""
    metadata_path = build_dir / "flasher_args.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Flash metadata not found: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    write_args = metadata.get("write_flash_args")
    flash_files = metadata.get("flash_files")
    if not isinstance(write_args, list) or not all(isinstance(arg, str) for arg in write_args):
        raise ValueError(f"Invalid write_flash_args in {metadata_path}")
    if not isinstance(flash_files, dict) or not all(
        isinstance(address, str) and isinstance(filename, str)
        for address, filename in flash_files.items()
    ):
        raise ValueError(f"Invalid flash_files in {metadata_path}")

    args = [IDF_FLASH_OPTION_NAMES.get(arg, arg) for arg in write_args]
    for address, filename in sorted(flash_files.items(), key=lambda item: int(item[0], 0)):
        args.extend((address, filename))
    return args


def flash_build(
    build_dir: Path,
    *,
    chip: str,
    idf_target: str,
    port: str,
    erase: bool,
) -> None:
    """Flash one generated ESP-IDF build in a single esptool session."""
    args = esptool_flash_command(chip=chip, idf_target=idf_target, port=port)
    if erase:
        args.append("--erase-all")
    args.extend(idf_flash_arguments(build_dir))
    run_esptool(args, cwd=build_dir)


def flash_factory_image(
    image: Path,
    *,
    chip: str,
    idf_target: str,
    port: str,
    erase: bool,
) -> None:
    """Flash one merged factory image at offset zero."""
    if not image.is_file():
        raise FileNotFoundError(f"Factory image not found: {image}")
    args = esptool_flash_command(chip=chip, idf_target=idf_target, port=port)
    if erase:
        args.append("--erase-all")
    args.extend(("0x0", str(image)))
    run_esptool(args, cwd=REPO_ROOT)


def run_firmware(*, chip: str, idf_target: str, port: str) -> None:
    """Enter the loader and ask esptool to run the installed application."""
    run_esptool(
        [
            "--chip",
            idf_target,
            "--port",
            port,
            "--before",
            esptool_before_mode(chip, port),
            "--after",
            "no-reset",
            "run",
        ],
        cwd=REPO_ROOT,
    )
