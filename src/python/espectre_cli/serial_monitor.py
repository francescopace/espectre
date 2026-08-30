# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - CLI Serial Monitor

Frontend-agnostic serial monitor command.

Author: Francesco Pace <francesco.pace@gmail.com>
"""

from __future__ import annotations

import sys
import time

from .common import (
    Fore,
    Style,
    remember_serial_port_identity,
    resolve_serial_port,
    serial_console_mode,
)

try:
    import serial
except ImportError:
    serial = None

MAX_RECONNECT_ATTEMPTS = 3
RECONNECT_DELAY_SECONDS = 1.0
HARD_RESET_HOLD_SECONDS = 0.1


def _require_pyserial() -> None:
    if serial is not None:
        return
    print(f"{Fore.RED}❌ pyserial not found. Install it with:{Style.RESET_ALL}")
    print("   pip install pyserial")
    raise SystemExit(1)


def hard_reset_serial(connection) -> None:
    """Pulse RTS while holding DTR low to hard-reset a typical ESP USB-UART board."""
    connection.dtr = False
    connection.rts = True
    time.sleep(HARD_RESET_HOLD_SECONDS)
    connection.rts = False


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
    reconnect_attempt = 0
    chip = getattr(args, "chip", None)
    frontend = getattr(args, "frontend", "native")
    port_selector = getattr(args, "port", None)

    while True:
        try:
            port = resolve_serial_port(
                port_selector,
                chip=chip,
                frontend=frontend,
                purpose="monitor",
            )
        except SystemExit:
            if reconnect_attempt >= MAX_RECONNECT_ATTEMPTS:
                raise
            reconnect_attempt += 1
            print(
                f"{Fore.YELLOW}⚠️ Serial port unavailable, retrying "
                f"({reconnect_attempt}/{MAX_RECONNECT_ATTEMPTS})...{Style.RESET_ALL}"
            )
            time.sleep(RECONNECT_DELAY_SECONDS)
            continue
        port_selector = port
        remember_serial_port_identity(port)

        if reset_on_open and serial_console_mode(chip, port) == "usb_cdc":
            print(
                f"{Fore.RED}❌ Automatic hard reset is unavailable on the USB CDC console."
                f"{Style.RESET_ALL}"
            )
            print("Reset the board manually, then run monitor without --reset.")
            raise SystemExit(1)

        print(f"{Fore.CYAN}Port:    {port}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Baud:    {baud}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Mode:    {'raw' if args.raw else 'text'}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Reset:   {'hard' if reset_on_open else 'none'}{Style.RESET_ALL}")

        connection = None
        try:
            connection = serial.Serial(port, baudrate=baud, timeout=1.0)
            if reset_on_open:
                hard_reset_serial(connection)
            reconnect_attempt = 0
            while True:
                pending = connection.in_waiting
                data = connection.read(pending or 1)
                if not data:
                    continue
                _write_serial_output(data, raw=bool(args.raw))
        except KeyboardInterrupt:
            return
        except (OSError, serial.SerialException) as exc:
            # USB console paths can change when the device re-enumerates.
            # Re-resolve the same physical device instead of selecting globally.
            if reconnect_attempt >= MAX_RECONNECT_ATTEMPTS:
                print(f"{Fore.RED}❌ Serial monitor disconnected: {exc}{Style.RESET_ALL}")
                raise SystemExit(1)
            reconnect_attempt += 1
            print(
                f"{Fore.YELLOW}⚠️ Serial monitor disconnected, retrying "
                f"({reconnect_attempt}/{MAX_RECONNECT_ATTEMPTS})...{Style.RESET_ALL}"
            )
            time.sleep(RECONNECT_DELAY_SECONDS)
        finally:
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass
