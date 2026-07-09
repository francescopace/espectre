"""Frontend-agnostic serial monitor command."""

from __future__ import annotations

import sys
import time

from .common import Fore, Style, get_serial_port

try:
    import serial
except ImportError:
    serial = None

MAX_RECONNECT_ATTEMPTS = 3
RECONNECT_DELAY_SECONDS = 1.0


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
    reconnect_attempt = 0
    selected_port: str | None = None

    while True:
        port = args.port or selected_port
        if port is None:
            try:
                port = get_serial_port(None)
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

        print(f"{Fore.CYAN}Port:    {port}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Baud:    {baud}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Mode:    {'raw' if args.raw else 'text'}{Style.RESET_ALL}")

        connection = None
        try:
            connection = serial.Serial(port, baudrate=baud, timeout=1.0)
            selected_port = port
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
