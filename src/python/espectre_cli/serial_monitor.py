"""Frontend-agnostic serial monitor command."""

from __future__ import annotations

import subprocess
import sys

from .common import Fore, Style, get_serial_port


def run_serial_monitor(args) -> None:
    """Attach to a serial port and stream device logs."""
    port = get_serial_port(args.port)
    baud = str(args.baud)
    command = [sys.executable, "-m", "serial.tools.miniterm", port, baud]
    if args.raw:
        command.append("--raw")

    print(f"{Fore.CYAN}Port:    {port}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Baud:    {baud}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Command: {' '.join(command)}{Style.RESET_ALL}")

    try:
        subprocess.run(command, check=True)
    except FileNotFoundError:
        print(f"{Fore.RED}❌ Python executable not found: {sys.executable}{Style.RESET_ALL}")
        raise SystemExit(1)
    except subprocess.CalledProcessError as e:
        print(f"{Fore.RED}❌ Serial monitor exited with code {e.returncode}{Style.RESET_ALL}")
        raise SystemExit(e.returncode)
