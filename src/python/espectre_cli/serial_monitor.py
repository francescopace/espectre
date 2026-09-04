# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - CLI Serial Monitor

Frontend-agnostic serial monitor command.

Author: Francesco Pace <francesco.pace@gmail.com>
"""

from __future__ import annotations

import sys

from .common import (
    Fore,
    Style,
    resolve_serial_port,
    serial_console_mode,
)
from .esptool_runner import run_firmware
from .targets import IDF_TARGET_BY_CHIP

try:
    import serial
except ImportError:
    serial = None

def _require_pyserial() -> None:
    if serial is not None:
        return
    print(f"{Fore.RED}❌ pyserial not found. Install it with:{Style.RESET_ALL}")
    print("   pip install pyserial")
    raise SystemExit(1)


def _write_serial_output(data: bytes, *, raw: bool) -> None:
    if raw:
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
        return
    sys.stdout.write(data.decode("utf-8", errors="replace"))
    sys.stdout.flush()


def run_serial_monitor(args) -> None:
    """Attach to a serial port and stream device logs."""
    _require_pyserial()
    baud = int(args.baud)
    reset_on_open = bool(getattr(args, "reset", False))
    chip = getattr(args, "chip", None)
    frontend = getattr(args, "frontend", "native")
    port = resolve_serial_port(
        getattr(args, "port", None),
        chip=chip,
        frontend=frontend,
        purpose="monitor",
    )

    if reset_on_open and serial_console_mode(chip, port) == "usb_cdc":
        print(
            f"{Fore.RED}❌ Automatic reset is unavailable on the USB CDC console."
            f"{Style.RESET_ALL}"
        )
        print("Reset the board manually, then run monitor without --reset.")
        raise SystemExit(1)
    if reset_on_open:
        if chip is None:
            print(f"{Fore.RED}❌ --chip is required with --reset.{Style.RESET_ALL}")
            raise SystemExit(1)
        run_firmware(
            chip=chip,
            idf_target=IDF_TARGET_BY_CHIP[chip],
            port=port,
        )

    print(f"{Fore.CYAN}Port:    {port}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Baud:    {baud}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Mode:    {'raw' if args.raw else 'text'}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Reset:   {'esptool run' if reset_on_open else 'none'}{Style.RESET_ALL}")

    connection = None
    try:
        connection = serial.Serial(port, baudrate=baud, timeout=1.0)
        while True:
            pending = connection.in_waiting
            data = connection.read(pending or 1)
            if not data:
                continue
            _write_serial_output(data, raw=bool(args.raw))
    except KeyboardInterrupt:
        return
    except (OSError, serial.SerialException) as exc:
        print(f"{Fore.RED}❌ Serial monitor disconnected: {exc}{Style.RESET_ALL}")
        raise SystemExit(1) from exc
    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass
