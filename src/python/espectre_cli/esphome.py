"""ESPHome frontend wrappers."""

from __future__ import annotations

import shutil
import subprocess

from .common import Fore, REPO_ROOT, Style
from .targets import resolve_esphome_config

ACTION_MAP = {
    "build": "compile",
    "flash": "upload",
    "config": "config",
    "monitor": "logs",
}


def clean_esphome_build_artifacts(config_path) -> None:
    """Remove generated ESPHome build artifacts before a fresh build."""
    build_dir = config_path.parent / ".esphome"
    if build_dir.is_dir():
        shutil.rmtree(build_dir)
        try:
            display_path = build_dir.relative_to(REPO_ROOT)
        except ValueError:
            display_path = build_dir
        print(f"{Fore.CYAN}Cleaned:   {display_path}{Style.RESET_ALL}")
    else:
        print(f"{Fore.CYAN}Cleaned:   nothing to remove{Style.RESET_ALL}")


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
    if args.esphome_command == "build" and getattr(args, "clean", False):
        clean_esphome_build_artifacts(config_path)

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
