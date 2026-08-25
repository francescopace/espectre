# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Shared one-shot DNS-SD discovery for ESPectre frontends."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import threading
import time

from .common import Fore, Style

try:
    from zeroconf import IPVersion, ServiceBrowser, ServiceListener, Zeroconf
except ImportError:  # pragma: no cover - exercised via CLI integration tests
    IPVersion = None
    ServiceBrowser = None
    ServiceListener = object
    Zeroconf = None


ESPECTRE_SERVICE_TYPE = "_espectre._tcp.local."
SUPPORTED_DISCOVERY_FRONTENDS = ("native", "streamer", "esphome", "matter")
DISCOVERY_TIMEOUT_S = 2.5
DISCOVERY_QUIET_WINDOW_S = 0.35
# Collect uses the same fresh PTR browse as the generic devices command.
COLLECT_DISCOVERY_QUIET_WINDOW_S = DISCOVERY_QUIET_WINDOW_S


@dataclass(frozen=True)
class DiscoveredDevice:
    service_name: str
    service_type: str
    frontend: str
    device_id: int
    device_id_text: str
    name: str
    chip: str
    ip_address: str
    port: int
    transport: str
    endpoint: str
    protocol: str
    events_endpoint: str | None = None
    firmware: str | None = None
    capabilities: tuple[str, ...] = ()
    metadata: tuple[tuple[str, str], ...] = ()

    @property
    def display_id(self) -> str:
        return self.device_id_text

    @property
    def target_port(self) -> int:
        """Return the Streamer pacing port, not the Direct HTTP port."""
        if self.frontend != "streamer":
            return self.port
        return _parse_optional_port(dict(self.metadata).get("traffic_port")) or self.port

    def as_serializable_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["device_id"] = self.device_id_text
        data.pop("device_id_text")
        data["capabilities"] = list(self.capabilities)
        data["metadata"] = dict(self.metadata)
        return data


class DeviceDiscoveryError(RuntimeError):
    """Raised when host-side device discovery cannot run."""


def _decode_txt(properties, key: str) -> str | None:
    raw_value = properties.get(key.encode("utf-8"))
    if raw_value is None:
        return None
    if isinstance(raw_value, bytes):
        return raw_value.decode("utf-8", errors="replace").strip()
    return str(raw_value).strip()


def _parse_device_id(value: str | None) -> tuple[int, str] | None:
    if not value:
        return None
    normalized = value.strip().lower().removeprefix("0x")
    try:
        device_id = int(normalized, 16)
    except ValueError:
        return None
    if device_id <= 0 or device_id > 0xFFFFFFFFFFFFFFFF:
        return None
    return device_id, f"{device_id:016x}"


def _parse_optional_port(value: str | None) -> int | None:
    if not value:
        return None
    try:
        port = int(value)
    except ValueError:
        return None
    return port if 1 <= port <= 65535 else None


def _display_name(service_name: str, properties) -> str:
    advertised_name = _decode_txt(properties, "name")
    if advertised_name:
        return advertised_name
    instance_name = service_name.split("._", 1)[0].strip()
    return instance_name or "ESPectre"


def _parse_record(service_type: str, service_name: str, info) -> DiscoveredDevice | None:
    addresses = info.parsed_addresses(IPVersion.V4Only)
    parsed_device_id = _parse_device_id(_decode_txt(info.properties, "device_id"))
    frontend = _decode_txt(info.properties, "frontend")
    transport = _decode_txt(info.properties, "transport")
    path = _decode_txt(info.properties, "path")
    events = _decode_txt(info.properties, "events")
    txtvers = _decode_txt(info.properties, "txtvers")
    protovers = _decode_txt(info.properties, "protovers")
    port = int(info.port)
    if (
        service_type != ESPECTRE_SERVICE_TYPE
        or not addresses
        or parsed_device_id is None
        or frontend not in SUPPORTED_DISCOVERY_FRONTENDS
        or transport != "http"
        or not 1 <= port <= 65535
        or path != "/espectre/v1/request"
        or events != "/espectre/v1/events"
        or txtvers != "2"
        or protovers != "1"
    ):
        return None

    device_id, device_id_text = parsed_device_id
    authority = addresses[0] if port == 80 else f"{addresses[0]}:{port}"
    traffic_port = _parse_optional_port(_decode_txt(info.properties, "traffic_port"))
    if frontend == "streamer" and traffic_port is None:
        return None
    capabilities_text = _decode_txt(info.properties, "capabilities") or ""
    capabilities = tuple(value.strip() for value in capabilities_text.split(",") if value.strip())
    metadata = tuple(
        (key, str(value))
        for key, value in (("traffic_port", traffic_port),)
        if value is not None
    )
    return DiscoveredDevice(
        service_name=service_name,
        service_type=service_type,
        frontend=frontend,
        device_id=device_id,
        device_id_text=device_id_text,
        name=_display_name(service_name, info.properties),
        chip=_decode_txt(info.properties, "chip") or "unknown",
        ip_address=addresses[0],
        port=port,
        transport=transport,
        endpoint=f"http://{authority}{path}",
        protocol=protovers,
        events_endpoint=f"http://{authority}{events}",
        firmware=_decode_txt(info.properties, "firmware"),
        capabilities=capabilities,
        metadata=metadata,
    )


class _DeviceListener(ServiceListener):
    def __init__(self, zeroconf: Zeroconf):
        self._zeroconf = zeroconf
        self._records: dict[tuple[str, str], DiscoveredDevice] = {}
        self._records_changed = threading.Condition()
        self._last_change_monotonic = 0.0

    def _resolve_record(self, service_type: str, name: str) -> DiscoveredDevice | None:
        info = self._zeroconf.get_service_info(service_type, name, timeout=1000)
        return None if info is None else _parse_record(service_type, name, info)

    def add_service(self, zeroconf: Zeroconf, service_type: str, name: str) -> None:
        del zeroconf
        record = self._resolve_record(service_type, name)
        if record is not None:
            key = (service_type, name)
            with self._records_changed:
                if self._records.get(key) != record:
                    self._records[key] = record
                    self._last_change_monotonic = time.monotonic()
                    self._records_changed.notify_all()

    def update_service(self, zeroconf: Zeroconf, service_type: str, name: str) -> None:
        self.add_service(zeroconf, service_type, name)

    def remove_service(self, zeroconf: Zeroconf, service_type: str, name: str) -> None:
        del zeroconf
        with self._records_changed:
            if self._records.pop((service_type, name), None) is not None:
                self._last_change_monotonic = time.monotonic()
                self._records_changed.notify_all()

    def snapshot(self, frontend: str | None = None) -> list[DiscoveredDevice]:
        with self._records_changed:
            records = list(self._records.values())
        if frontend is not None:
            records = [record for record in records if record.frontend == frontend]
        return sorted(
            records,
            key=lambda record: (
                record.frontend,
                record.device_id_text,
                record.ip_address,
                record.port,
                record.service_name,
            ),
        )

    def wait_for_quiet(self, timeout_s: float, quiet_window_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        with self._records_changed:
            while True:
                now = time.monotonic()
                remaining = deadline - now
                if remaining <= 0:
                    return
                if self._records:
                    quiet_remaining = self._last_change_monotonic + quiet_window_s - now
                    if quiet_remaining <= 0:
                        return
                    remaining = min(remaining, quiet_remaining)
                self._records_changed.wait(remaining)


def _ensure_zeroconf_available() -> None:
    if Zeroconf is None or ServiceBrowser is None or IPVersion is None:
        raise DeviceDiscoveryError(
            "mDNS discovery requires the Python 'zeroconf' package. "
            "Install the repository requirements, or use an explicit device address."
        )


def discover_devices(
    frontend: str | None = None,
    timeout_s: float = DISCOVERY_TIMEOUT_S,
    quiet_window_s: float = DISCOVERY_QUIET_WINDOW_S,
) -> list[DiscoveredDevice]:
    if frontend is not None and frontend not in SUPPORTED_DISCOVERY_FRONTENDS:
        raise ValueError(f"unsupported discovery frontend: {frontend}")
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("discovery timeout must be a finite value greater than zero")
    if not math.isfinite(quiet_window_s) or quiet_window_s <= 0:
        raise ValueError("discovery quiet window must be a finite value greater than zero")
    _ensure_zeroconf_available()
    try:
        zeroconf = Zeroconf(ip_version=IPVersion.V4Only)
    except OSError as exc:
        raise DeviceDiscoveryError(
            f"mDNS discovery could not open local network interfaces ({exc}). Use an explicit device address."
        ) from exc

    listener = _DeviceListener(zeroconf)
    browser = None
    try:
        browser = ServiceBrowser(zeroconf, ESPECTRE_SERVICE_TYPE, listener=listener)
        listener.wait_for_quiet(float(timeout_s), float(quiet_window_s))
    finally:
        if browser is not None:
            cancel = getattr(browser, "cancel", None)
            if callable(cancel):
                cancel()
        zeroconf.close()
    return listener.snapshot(frontend)


def choose_device_interactively(
    records: list[DiscoveredDevice],
    *,
    frontend_label: str | None = None,
) -> DiscoveredDevice:
    if not records:
        raise DeviceDiscoveryError("No devices available for selection")
    if len(records) == 1:
        return records[0]

    label = f" {frontend_label}" if frontend_label else ""
    print(f"  {Fore.CYAN}Discovered{label} devices:{Style.RESET_ALL}")
    for idx, record in enumerate(records, start=1):
        print(
            f"    {idx}. {record.device_id_text} frontend={record.frontend} "
            f"chip={record.chip.upper()} ip={record.ip_address} endpoint={record.endpoint}"
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


def print_device_list(
    records: list[DiscoveredDevice],
    *,
    json_output: bool = False,
    heading: str = "Discovered ESPectre devices",
) -> None:
    if json_output:
        print(json.dumps([record.as_serializable_dict() for record in records], indent=2, sort_keys=True))
        return

    print()
    if not records:
        print(f"{Fore.YELLOW}No ESPectre devices discovered via mDNS.{Style.RESET_ALL}")
        return
    print(f"{Fore.CYAN}{heading}:{Style.RESET_ALL}")
    for record in records:
        print(
            f"  - {record.device_id_text} frontend={record.frontend} "
            f"name={record.name!r} chip={record.chip.upper()} "
            f"ip={record.ip_address} endpoint={record.endpoint}"
        )


def run_devices_command(args) -> int:
    try:
        records = discover_devices(frontend=args.frontend, timeout_s=args.timeout)
    except (DeviceDiscoveryError, ValueError) as exc:
        print(f"{Fore.RED}Discovery failed: {exc}{Style.RESET_ALL}")
        return 1
    print_device_list(records, json_output=args.json)
    return 0
