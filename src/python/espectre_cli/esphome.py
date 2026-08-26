# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - CLI ESPHome

ESPHome frontend wrappers.

Author: Francesco Pace <francesco.pace@gmail.com>
"""

from __future__ import annotations

import subprocess

from .common import Fore, REPO_ROOT, Style
from .targets import resolve_esphome_config

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
    if getattr(args, "device", None):
        command.extend(["--device", args.device])
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
