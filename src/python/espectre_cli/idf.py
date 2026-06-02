"""Thin wrappers around idf.py for ESPectre frontends."""

from __future__ import annotations

import subprocess

from .common import Fore, Style, get_serial_port
from .targets import resolve_idf_target


def run_idf_command(frontend: str, args) -> None:
    """Run an IDF workflow for the given frontend."""
    chip = getattr(args, "chip", None)
    try:
        app_dir, idf_target = resolve_idf_target(frontend, chip)
    except ValueError as e:
        print(f"{Fore.RED}❌ {e}{Style.RESET_ALL}")
        raise SystemExit(1)

    print(f"{Fore.CYAN}Frontend: {frontend}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}App dir:   {app_dir}{Style.RESET_ALL}")

    commands = []
    if args.idf_command == "build":
        commands = [["idf.py", "set-target", idf_target], ["idf.py", "build"]]
    elif args.idf_command == "flash":
        port = get_serial_port(args.port)
        commands = [["idf.py", "-p", port, "flash"]]
    elif args.idf_command == "monitor":
        port = get_serial_port(args.port)
        commands = [["idf.py", "-p", port, "monitor"]]

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
