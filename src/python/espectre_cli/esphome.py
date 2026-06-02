"""ESPHome frontend wrappers."""

from __future__ import annotations

import subprocess

from .common import Fore, REPO_ROOT, Style
from .targets import resolve_esphome_config

ACTION_MAP = {
    "build": "compile",
    "flash": "run",
    "config": "config",
    "logs": "logs",
}


def run_esphome_command(args) -> None:
    """Run an ESPHome action against the resolved repository config."""
    try:
        config_path = resolve_esphome_config(args.chip, args.dev, args.config)
    except ValueError as e:
        print(f"{Fore.RED}❌ {e}{Style.RESET_ALL}")
        raise SystemExit(1)

    if not config_path.exists():
        print(f"{Fore.RED}❌ ESPHome config not found: {config_path}{Style.RESET_ALL}")
        raise SystemExit(1)

    action = ACTION_MAP[args.esphome_command]
    command = ["esphome", action, str(config_path)]
    if getattr(args, "device", None):
        command.extend(["--device", args.device])

    try:
        display_path = config_path.relative_to(REPO_ROOT)
    except ValueError:
        display_path = config_path
    print(f"{Fore.CYAN}Config: {display_path}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Command: {' '.join(command)}{Style.RESET_ALL}")
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError:
        print(f"{Fore.RED}❌ esphome not found. Install it in the project environment first.{Style.RESET_ALL}")
        raise SystemExit(1)
    except subprocess.CalledProcessError as e:
        print(f"{Fore.RED}❌ ESPHome command failed with exit code {e.returncode}{Style.RESET_ALL}")
        raise SystemExit(e.returncode)
