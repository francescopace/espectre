"""Thin wrappers around idf.py for ESPectre frontends."""

from __future__ import annotations

import subprocess
from pathlib import Path

from .common import Fore, Style, get_serial_port
from .targets import resolve_idf_target

DEFAULT_MATTER_MONITOR_PRINT_FILTER = (
    "*:W "
    "espectre.matter:I "
    "espectre.matter.app:I "
    "espectre.runtime:I "
    "WiFiLifecycle:I "
    "GainController:I "
    "BaseDetector:I "
    "MLDetector:I "
    "MVSDetector:I "
    "CsiCapture:I "
    "TrafficGen:I "
    "ROUTE_HOOK:I"
)


def _resolve_monitor_print_filter(frontend: str, args) -> str | None:
    explicit_filter = getattr(args, "print_filter", None)
    if explicit_filter:
        return explicit_filter
    if frontend == "matter" and args.idf_command == "monitor":
        return DEFAULT_MATTER_MONITOR_PRINT_FILTER
    return None


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
    elif args.idf_command == "monitor":
        port = get_serial_port(args.port)
        monitor_command = ["idf.py", "-p", port, "monitor"]
        print_filter = _resolve_monitor_print_filter(frontend, args)
        if print_filter:
            monitor_command.append(f"--print-filter={print_filter}")
        commands = [monitor_command]

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
