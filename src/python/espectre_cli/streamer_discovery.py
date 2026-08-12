# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - Streamer Discovery

One-shot DNS-SD/mDNS discovery helpers for the Streamer frontend.

Author: Francesco Pace <francesco.pace@gmail.com>
"""

from __future__ import annotations

from dataclasses import dataclass
import time

from .common import Fore, Style

try:
    from zeroconf import IPVersion, ServiceBrowser, ServiceListener, Zeroconf
except ImportError:  # pragma: no cover - exercised via CLI integration tests
    IPVersion = None
    ServiceBrowser = None
    ServiceListener = object
    Zeroconf = None


STREAMER_SERVICE_TYPE = "_espectre-streamer._udp.local."
DISCOVERY_TIMEOUT_S = 2.5


@dataclass(frozen=True)
class StreamerDiscoveryRecord:
    service_name: str
    device_id: int
    device_id_text: str
    chip: str
    ip_address: str
    target_port: int
    collector_port: int | None = None


class StreamerDiscoveryError(RuntimeError):
    """Raised when discovery cannot run or yields no usable result."""


class _StreamerListener(ServiceListener):
    def __init__(self, zeroconf: Zeroconf):
        self._zeroconf = zeroconf
        self._records: dict[str, StreamerDiscoveryRecord] = {}

    @staticmethod
    def _decode_txt(properties, key: str) -> str | None:
        raw_value = properties.get(key.encode("utf-8"))
        if raw_value is None:
            return None
        if isinstance(raw_value, bytes):
            return raw_value.decode("utf-8", errors="replace").strip()
        return str(raw_value).strip()

    @staticmethod
    def _parse_device_id(value: str | None) -> tuple[int, str] | None:
        if not value:
            return None
        normalized = value.strip().lower()
        if normalized.startswith("0x"):
            normalized = normalized[2:]
        if not normalized:
            return None
        try:
            return int(normalized, 16), f"0x{normalized}"
        except ValueError:
            return None

    def _resolve_record(self, service_type: str, name: str) -> StreamerDiscoveryRecord | None:
        info = self._zeroconf.get_service_info(service_type, name, timeout=1000)
        if info is None:
            return None
        addresses = info.parsed_addresses(IPVersion.V4Only)
        if not addresses:
            return None
        parsed_device_id = self._parse_device_id(self._decode_txt(info.properties, "device_id"))
        if parsed_device_id is None:
            return None
        device_id, device_id_text = parsed_device_id
        chip = self._decode_txt(info.properties, "chip") or "unknown"
        collector_port_text = self._decode_txt(info.properties, "collector_port")
        collector_port = None
        if collector_port_text:
            try:
                collector_port = int(collector_port_text)
            except ValueError:
                collector_port = None
        return StreamerDiscoveryRecord(
            service_name=name,
            device_id=device_id,
            device_id_text=device_id_text,
            chip=chip,
            ip_address=addresses[0],
            target_port=int(info.port),
            collector_port=collector_port,
        )

    def add_service(self, zeroconf: Zeroconf, service_type: str, name: str) -> None:
        del zeroconf
        record = self._resolve_record(service_type, name)
        if record is not None:
            self._records[name] = record

    def update_service(self, zeroconf: Zeroconf, service_type: str, name: str) -> None:
        self.add_service(zeroconf, service_type, name)

    def remove_service(self, zeroconf: Zeroconf, service_type: str, name: str) -> None:
        del zeroconf, service_type
        self._records.pop(name, None)

    def snapshot(self) -> list[StreamerDiscoveryRecord]:
        return sorted(
            self._records.values(),
            key=lambda record: (record.device_id_text, record.ip_address, record.target_port, record.service_name),
        )


def _ensure_zeroconf_available() -> None:
    if Zeroconf is None or ServiceBrowser is None or IPVersion is None:
        raise StreamerDiscoveryError(
            "mDNS discovery requires the Python 'zeroconf' package. "
            "Use --target for explicit collection, or install the repository requirements."
        )


def discover_streamer_devices(timeout_s: float = DISCOVERY_TIMEOUT_S) -> list[StreamerDiscoveryRecord]:
    _ensure_zeroconf_available()
    try:
        zeroconf = Zeroconf(ip_version=IPVersion.V4Only)
    except OSError as exc:
        raise StreamerDiscoveryError(
            f"mDNS discovery could not open local network interfaces ({exc}). "
            "Use --target for explicit collection."
        ) from exc
    listener = _StreamerListener(zeroconf)
    browser = ServiceBrowser(zeroconf, STREAMER_SERVICE_TYPE, listener=listener)
    try:
        time.sleep(max(0.1, float(timeout_s)))
        return listener.snapshot()
    finally:
        cancel = getattr(browser, "cancel", None)
        if callable(cancel):
            cancel()
        zeroconf.close()


def choose_streamer_device_interactively(records: list[StreamerDiscoveryRecord]) -> StreamerDiscoveryRecord:
    if not records:
        raise StreamerDiscoveryError("No streamer devices available for selection")
    if len(records) == 1:
        return records[0]

    print(f"  {Fore.CYAN}Discovered streamer devices:{Style.RESET_ALL}")
    for idx, record in enumerate(records, start=1):
        collector_suffix = (
            f" collector_port={record.collector_port}"
            if record.collector_port is not None
            else ""
        )
        print(
            f"    {idx}. {record.device_id_text} "
            f"chip={record.chip.upper()} ip={record.ip_address} "
            f"target_port={record.target_port}{collector_suffix}"
        )
    print()
    print(f"  {Fore.YELLOW}Select one device by number, or press Ctrl+C to cancel.{Style.RESET_ALL}")

    while True:
        choice = input("  Device number: ").strip()
        if not choice:
            print(f"  {Fore.YELLOW}Enter a number between 1 and {len(records)}.{Style.RESET_ALL}")
            continue
        try:
            selected_index = int(choice)
        except ValueError:
            print(f"  {Fore.YELLOW}Enter a valid integer choice.{Style.RESET_ALL}")
            continue
        if 1 <= selected_index <= len(records):
            return records[selected_index - 1]
        print(f"  {Fore.YELLOW}Choice out of range: 1-{len(records)}.{Style.RESET_ALL}")


def print_streamer_device_list(records: list[StreamerDiscoveryRecord]) -> None:
    print()
    if not records:
        print(f"{Fore.YELLOW}No Streamer devices discovered via mDNS.{Style.RESET_ALL}")
        return
    print(f"{Fore.CYAN}Discovered Streamer devices:{Style.RESET_ALL}")
    for record in records:
        collector_suffix = (
            f" collector_port={record.collector_port}"
            if record.collector_port is not None
            else ""
        )
        print(
            f"  - {record.device_id_text} chip={record.chip.upper()} "
            f"ip={record.ip_address} target_port={record.target_port}{collector_suffix}"
        )
