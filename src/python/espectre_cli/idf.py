"""Thin wrappers around idf.py for ESPectre frontends."""

from __future__ import annotations

import subprocess
from pathlib import Path

from .common import Fore, Style, get_serial_port
from .targets import IDF_FRONTENDS, resolve_idf_target


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
    sdkconfig_defaults = ["sdkconfig.defaults"]
    if (app_path / "sdkconfig.wifi").exists():
        sdkconfig_defaults.append("sdkconfig.wifi")
    defaults_arg = f'-DSDKCONFIG_DEFAULTS={";".join(sdkconfig_defaults)}'

    commands = []
    if args.idf_command == "build":
        commands = [["idf.py", defaults_arg, "set-target", idf_target], ["idf.py", defaults_arg, "build"]]
    elif args.idf_command == "flash":
        port = get_serial_port(args.port)
        commands = [["idf.py", "-p", port, "flash"]]

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
