"""Thin wrappers around idf.py for ESPectre frontends."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from .common import Fore, Style, get_serial_port
from .targets import IDF_FRONTENDS, resolve_idf_target


def clean_idf_build_artifacts(app_path: Path, build_dir_name: str | None = None) -> None:
    """Remove generated ESP-IDF build artifacts before a fresh build."""
    artifact_names = [build_dir_name or "build", "sdkconfig", "sdkconfig.old", "dependencies.lock"]
    removed: list[str] = []

    for artifact_name in artifact_names:
        artifact_path = app_path / artifact_name
        if artifact_path.is_dir():
            shutil.rmtree(artifact_path)
            removed.append(artifact_name)
        elif artifact_path.exists():
            artifact_path.unlink()
            removed.append(artifact_name)

    if removed:
        print(f"{Fore.CYAN}Cleaned:   {', '.join(removed)}{Style.RESET_ALL}")
    else:
        print(f"{Fore.CYAN}Cleaned:   nothing to remove{Style.RESET_ALL}")


def resolve_sdkconfig_defaults(app_path: Path) -> str:
    """Resolve SDKCONFIG defaults from the environment or local app defaults."""
    env_defaults = os.environ.get("SDKCONFIG_DEFAULTS")
    if env_defaults:
        return env_defaults

    sdkconfig_defaults = ["sdkconfig.defaults"]
    if (app_path / "sdkconfig.wifi").exists():
        sdkconfig_defaults.append("sdkconfig.wifi")
    return ";".join(sdkconfig_defaults)


def build_idf_base_command(build_dir_name: str | None) -> list[str]:
    """Build the shared idf.py command prefix."""
    command = ["idf.py"]
    if build_dir_name:
        command.extend(["-B", build_dir_name])
    return command


def run_idf_command(frontend: str, args) -> None:
    """Run an IDF workflow for the given frontend."""
    chip = getattr(args, "chip", None)
    try:
        if args.idf_command == "build":
            app_dir, idf_target = resolve_idf_target(frontend, chip)
        else:
            app_dir = IDF_FRONTENDS[frontend]["app_dir"]
            idf_target = None
    except ValueError as e:
        print(f"{Fore.RED}❌ {e}{Style.RESET_ALL}")
        raise SystemExit(1)

    print(f"{Fore.CYAN}Frontend: {frontend}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}App dir:   {app_dir}{Style.RESET_ALL}")

    app_path = Path(app_dir)
    build_dir_name = os.environ.get("ESPECTRE_IDF_BUILD_DIR")
    if args.idf_command == "build" and getattr(args, "clean", False):
        clean_idf_build_artifacts(app_path, build_dir_name)

    defaults_arg = f"-DSDKCONFIG_DEFAULTS={resolve_sdkconfig_defaults(app_path)}"

    commands = []
    if args.idf_command == "build":
        base_command = build_idf_base_command(build_dir_name)
        commands = [
            [*base_command, defaults_arg, "set-target", idf_target],
            [*base_command, defaults_arg, "build"],
        ]
    elif args.idf_command == "flash":
        port = get_serial_port(args.port)
        base_command = build_idf_base_command(build_dir_name)
        commands = [[*base_command, "-p", port, "flash"]]

    try:
        for command in commands:
            print(f"{Fore.CYAN}Command: {' '.join(command)}{Style.RESET_ALL}")
            subprocess.run(command, cwd=app_dir, check=True)
    except FileNotFoundError:
        print(f"{Fore.RED}❌ idf.py not found. Load the ESP-IDF environment first.{Style.RESET_ALL}")
        raise SystemExit(1)
    except subprocess.CalledProcessError as e:
        print(f"{Fore.RED}❌ idf.py command failed with exit code {e.returncode}{Style.RESET_ALL}")
        raise SystemExit(e.returncode)
