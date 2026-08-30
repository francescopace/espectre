# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - CLI ESPHome

ESPHome frontend wrappers.

Author: Francesco Pace <francesco.pace@gmail.com>
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from .build_artifacts import print_build_artifact_metadata
from .common import Fore, REPO_ROOT, Style, resolve_serial_port, serial_console_mode
from .idf import flash_factory_image, flash_prebuilt_idf_build
from .targets import IDF_TARGET_BY_CHIP, resolve_esphome_config

ACTION_MAP = {
    "build": "compile",
    "flash": "upload",
    "config": "config",
    "monitor": "logs",
}

ESPHOME_COMMAND_PREFIX = [
    "esphome",
    "--toolchain",
    "esp-idf",
    "-s",
    "component_source",
    "local",
]


def resolve_esphome_build_artifact(config_path: Path) -> Path:
    """Return the application image produced for an ESPHome config."""
    candidates = [
        path
        for path in (config_path.parent / ".esphome" / "build").glob(
            "*/build/espectre.bin"
        )
        if path.is_file()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"ESPHome build artifact not found for {config_path}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def _is_network_device(device: str | None) -> bool:
    """Return whether --device names a hostname, IP address, or URL rather than a serial port."""
    if not device:
        return False
    if device.startswith("/") or device.lower().startswith("com"):
        return False
    return True


def run_esphome_command(args) -> None:
    """Run an ESPHome action against the resolved repository config."""
    try:
        config_path = resolve_esphome_config(args.chip, args.config)
    except ValueError as e:
        print(f"{Fore.RED}❌ {e}{Style.RESET_ALL}")
        raise SystemExit(1)

    if not config_path.exists():
        print(f"{Fore.RED}❌ ESPHome config not found: {config_path}{Style.RESET_ALL}")
        raise SystemExit(1)

    action = ACTION_MAP[args.esphome_command]
    commands: list[list[str]] = []
    if args.esphome_command == "build":
        if getattr(args, "clean_all", False):
            commands.append([*ESPHOME_COMMAND_PREFIX, "clean-all", str(config_path)])
        elif getattr(args, "clean", False):
            commands.append([*ESPHOME_COMMAND_PREFIX, "clean", str(config_path)])

    command = [*ESPHOME_COMMAND_PREFIX, action, str(config_path)]
    device = getattr(args, "device", None)
    if args.esphome_command in {"flash", "monitor"} and not _is_network_device(device):
        device = resolve_serial_port(
            device,
            chip=getattr(args, "chip", None),
            frontend="esphome",
            purpose="flash" if args.esphome_command == "flash" else "monitor",
            require_firmware_download=args.esphome_command == "flash",
        )
    if args.esphome_command == "flash" and not _is_network_device(device):
        chip = getattr(args, "chip", None)
        before = "no-reset" if serial_console_mode(chip, device) == "usb_cdc" else "default-reset"
        try:
            if getattr(args, "firmware", None):
                flash_factory_image(
                    Path(args.firmware).resolve(),
                    device,
                    IDF_TARGET_BY_CHIP.get(chip, "auto"),
                    erase=bool(getattr(args, "erase", False)),
                    before=before,
                )
            else:
                flash_prebuilt_idf_build(
                    resolve_esphome_build_artifact(config_path).parent,
                    device,
                    IDF_TARGET_BY_CHIP.get(chip, "auto"),
                    erase=bool(getattr(args, "erase", False)),
                    before=before,
                )
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"{Fore.RED}❌ Error flashing ESPHome firmware: {exc}{Style.RESET_ALL}")
            raise SystemExit(1) from exc
        return
    if device:
        command.extend(["--device", device])
    if getattr(args, "firmware", None):
        command.extend(["--file", args.firmware])
    commands.append(command)

    try:
        display_path = config_path.relative_to(REPO_ROOT)
    except ValueError:
        display_path = config_path
    print(f"{Fore.CYAN}Config: {display_path}{Style.RESET_ALL}")
    for command in commands:
        print(f"{Fore.CYAN}Command: {' '.join(command)}{Style.RESET_ALL}")
    try:
        for command in commands:
            subprocess.run(command, check=True)
    except FileNotFoundError:
        print(f"{Fore.RED}❌ esphome not found. Install it in the project environment first.{Style.RESET_ALL}")
        raise SystemExit(1)
    except subprocess.CalledProcessError as e:
        print(f"{Fore.RED}❌ ESPHome command failed with exit code {e.returncode}{Style.RESET_ALL}")
        raise SystemExit(e.returncode)
    if args.esphome_command == "build" and bool(getattr(args, "json", False)):
        try:
            artifact = resolve_esphome_build_artifact(config_path)
            print_build_artifact_metadata(
                frontend="esphome",
                chip=getattr(args, "chip", None),
                artifact=artifact,
            )
        except FileNotFoundError as exc:
            print(f"{Fore.RED}❌ {exc}{Style.RESET_ALL}")
            raise SystemExit(1) from exc
