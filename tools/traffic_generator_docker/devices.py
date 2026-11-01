# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Find ESPectre devices with mDNS and switch their traffic source through Direct HTTP."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import ipaddress
import json
import logging
import os
import threading
import time
from urllib.error import HTTPError

from espectre_cli.device_discovery import (
    ESPECTRE_DIRECT_PORT, DeviceDiscoveryError, discover_devices)
from espectre_cli.device_transport import DirectClient, DirectProtocolError, DirectRequestError

TRAFFIC_MODES = ("ping", "dns", "dns_tcp", "wifi_raw", "external")
# Direct reports raw-destination support exactly where the chip offers `wifi_raw`; see docs/API.md#capabilities.
RAW_MODE_FEATURE = "traffic_generator_target_mac"
# Firmware before that flag (3.0) offers `wifi_raw` on every chip but this one; see docs/CSI.md.
NO_RAW_CHIP = "esp32c6"
# CSI sample spacing values; devices that accept them say so in their capabilities.
SAMPLE_SPACINGS = ("half_slot", "distinct")
SPACING_FEATURE = "csi_sample_spacing"
# Asks each device for the internal source it used before external traffic, or ping.
INTERNAL = "internal"
# Panel column → Direct diagnostics field; see docs/API.md#diagnostics.
DIAGNOSTIC_FIELDS = {"generator": "generator_pps", "traffic_tx": "traffic_tx_pps",
                     "traffic_rx": "traffic_rx_pps", "accepted": "csi_accepted_pps",
                     "occupancy": "csi_occupancy", "rssi": "wifi_rssi_dbm"}
SENSING_FIELDS = ("traffic_generator_mode", "csi_target_pps", "csi_traffic_udp_port",
                  "csi_traffic_multicast_group", "csi_sample_spacing")
MAX_DEVICES = 64
# A saved address that has not answered yet has no device ID; its row uses this prefix.
UNIDENTIFIED = "address:"
MAX_MANUAL_ADDRESSES = 32
REQUEST_TIMEOUT_S = 3.0
REFRESH_WORKERS = 8


class RejectedInput(ValueError):
    """Rejected input whose message is written for the web UI."""

    def __init__(self, message):
        super().__init__(message)
        self.message = message


def parse_address(text):
    """Return (ip, port) from an IPv4 address with an optional port."""
    if not isinstance(text, str):
        raise RejectedInput("Enter an IPv4 address.")
    host, separator, port_text = text.strip().partition(":")
    try:
        address = ipaddress.IPv4Address(host)
    except ValueError:
        raise RejectedInput("Enter an IPv4 address, optionally with :port.") from None
    if address.is_unspecified or address.is_loopback or address.is_multicast or address.is_reserved:
        raise RejectedInput(f"{address} cannot be an ESPectre device.")
    port = ESPECTRE_DIRECT_PORT
    if separator:
        if not port_text.isdigit() or not 1 <= int(port_text) <= 65535:
            raise RejectedInput("The port must be in the 1-65535 range.")
        port = int(port_text)
    return str(address), port


def address_text(ip, port):
    return ip if port == ESPECTRE_DIRECT_PORT else f"{ip}:{port}"


def supported_modes(capabilities, chip):
    """Return the traffic sources a device offers, from its Direct capabilities."""
    features = capabilities.get("features")
    raw = features.get(RAW_MODE_FEATURE) if isinstance(features, dict) else None
    if not isinstance(raw, bool):
        raw = chip != NO_RAW_CHIP
    return tuple(mode for mode in TRAFFIC_MODES if mode != "wifi_raw" or raw)


def supports_spacing(capabilities):
    features = capabilities.get("features")
    return isinstance(features, dict) and features.get(SPACING_FEATURE) is True


def describe_error(error):
    if isinstance(error, DirectRequestError):
        return error.message or "The device rejected the request."
    cause = error.__cause__
    # An HTTP error status is an answer, although urllib's HTTPError is an OSError.
    if isinstance(cause, HTTPError):
        detail = str(error).partition(": ")[2].strip()
        try:
            result = json.loads(detail)
        except ValueError:
            result = None
        if isinstance(result, dict):
            # C++ frontends send a result object; the HTTP server sends plain text.
            detail = result.get("message") if isinstance(result.get("message"), str) else ""
        return detail or f"The device rejected the request (HTTP {cause.code})."
    # Network failures reach us wrapped in DirectProtocolError; URLError is an OSError.
    if isinstance(error, (OSError, TimeoutError)) or isinstance(cause, (OSError, TimeoutError)):
        return "The device did not answer."
    return str(error)


@dataclass
class Device:
    id: str
    ip_address: str
    port: int
    source: str
    name: str = ""
    chip: str = ""
    frontend: str = ""
    firmware: str = ""
    can_control: bool = False
    catalog: tuple = ()
    modes: tuple = ()
    spacing_supported: bool = False
    sensing: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    error: str | None = None

    @property
    def identified(self):
        return not self.id.startswith(UNIDENTIFIED)

    @property
    def endpoint(self):
        return f"http://{self.ip_address}:{self.port}/espectre/v1"

    def offered_modes(self):
        """Supported sources, plus the current one so the device can always show it."""
        current = self.sensing.get("traffic_generator_mode")
        return [mode for mode in TRAFFIC_MODES if mode in self.modes or mode == current]

    def as_dict(self):
        return {"id": self.id, "name": self.name, "chip": self.chip, "frontend": self.frontend,
                "firmware": self.firmware, "ip_address": self.ip_address, "port": self.port,
                "address": address_text(self.ip_address, self.port), "source": self.source,
                "can_control": self.can_control and self.error is None, "error": self.error,
                "modes": self.offered_modes(), "spacing_supported": self.spacing_supported,
                "sensing": {name: self.sensing.get(name) for name in SENSING_FIELDS},
                "fields": {column: self.diagnostics.get(name) for column, name in DIAGNOSTIC_FIELDS.items()}}


class DeviceRegistry:
    """Devices found by mDNS or added by address; all network calls run outside the lock."""

    def __init__(self, path, *, client_factory=DirectClient, discover=discover_devices):
        self.path = path
        self.client_factory = client_factory
        self.discover = discover
        self.lock = threading.Lock()
        self.devices = {}
        self.manual = self._load_manual()
        self.internal_modes = {}
        self.scanning = False
        self.scan_error = None
        self.scanned_at = None

    def _load_manual(self):
        try:
            saved = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        addresses = saved.get("manual") if isinstance(saved, dict) else None
        if not isinstance(addresses, list):
            raise ValueError(f"{self.path} must contain a manual address list")
        return [address_text(*parse_address(address)) for address in addresses[:MAX_MANUAL_ADDRESSES]]

    def _save_manual(self):
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(json.dumps({"manual": self.manual}, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def _client(self, device_or_endpoint):
        endpoint = getattr(device_or_endpoint, "endpoint", device_or_endpoint)
        return self.client_factory(endpoint, timeout=REQUEST_TIMEOUT_S)

    def probe(self, ip, port, source):
        """Identify a device and read its sensing state; raise when it is not ESPectre."""
        client = self._client(f"http://{ip}:{port}/espectre/v1")
        try:
            capabilities = client.negotiate()
            info = client.request("get", "device")
            catalog = client.request("get", "diagnostics", {"fields": []}).get("fields", [])
        finally:
            client.close()
        device_id = info.get("device_id")
        if not isinstance(device_id, str) or not device_id:
            raise DirectProtocolError("The device did not report its ID.")
        operations = {operation.get("name") for operation in capabilities["operations"]
                      if isinstance(operation, dict)}
        chip = str(info.get("chip") or "")
        device = Device(id=device_id, ip_address=ip, port=port, source=source,
                        name=str(info.get("name") or device_id), chip=chip,
                        frontend=str(info.get("frontend") or ""), firmware=str(info.get("firmware") or ""),
                        can_control="update_sensing" in operations, modes=supported_modes(capabilities, chip),
                        spacing_supported=supports_spacing(capabilities),
                        catalog=tuple(entry["name"] for entry in catalog
                                      if isinstance(entry, dict) and isinstance(entry.get("name"), str)))
        self.refresh(device)
        return device

    def refresh(self, device):
        client = self._client(device)
        try:
            device.sensing = client.request("get", "sensing")
            fields = [name for name in DIAGNOSTIC_FIELDS.values() if name in device.catalog]
            device.diagnostics = client.request("get", "diagnostics", {"fields": fields}) if fields else {}
            device.error = None
        except (DirectProtocolError, DirectRequestError, OSError, TimeoutError) as error:
            device.error = describe_error(error)
        finally:
            client.close()

    def _refresh_one(self, device):
        if device.identified:
            self.refresh(device)
        else:
            # A saved address that never answered gets identified once it does.
            self._store(self._probe_saved(device.ip_address, device.port))

    def refresh_all(self):
        with self.lock:
            devices = list(self.devices.values())
        if devices:
            with ThreadPoolExecutor(max_workers=REFRESH_WORKERS) as pool:
                list(pool.map(self._refresh_one, devices))

    def _store(self, device):
        with self.lock:
            if device.source == "manual" and address_text(device.ip_address, device.port) not in self.manual:
                # The address was removed while this probe was waiting for an answer.
                return
            same_address = [key for key, other in self.devices.items()
                            if (other.ip_address, other.port) == (device.ip_address, device.port) and key != device.id]
            if not device.identified:
                # Keep a known device that stopped answering under its own ID, with the new reason.
                known = [key for key in same_address if self.devices[key].identified]
                if known:
                    self.devices[known[0]].error = device.error
                    return
            for key in same_address:
                if not self.devices[key].identified:
                    del self.devices[key]
            existing = self.devices.get(device.id)
            if existing is not None and existing.source == "manual":
                device.source = "manual"
            if existing is None and len(self.devices) >= MAX_DEVICES:
                return
            self.devices[device.id] = device

    def scan(self):
        """Browse mDNS, then probe found and saved addresses; return how many answered."""
        with self.lock:
            if self.scanning:
                return None
            self.scanning, manual = True, list(self.manual)
        found, error = [], None
        try:
            try:
                found = [(record.ip_address, record.port, "mdns") for record in self.discover()]
            except (DeviceDiscoveryError, OSError) as failure:
                error = f"mDNS search failed: {failure}"
            found += [(*parse_address(address), "manual") for address in manual]
            with ThreadPoolExecutor(max_workers=REFRESH_WORKERS) as pool:
                probed = [device for device in pool.map(self._try_probe, found) if device is not None]
            with self.lock:
                # Keep devices that this search missed only when they were added by address.
                self.devices = {key: device for key, device in self.devices.items() if device.source == "manual"}
            for device in sorted(probed, key=lambda device: device.source == "manual"):
                self._store(device)
            return sum(device.identified for device in probed)
        finally:
            with self.lock:
                self.scanning, self.scan_error, self.scanned_at = False, error, time.time()

    def _try_probe(self, candidate):
        if candidate[2] == "manual":
            return self._probe_saved(*candidate[:2])
        try:
            return self.probe(*candidate)
        except (DirectProtocolError, DirectRequestError, OSError, TimeoutError, ValueError) as error:
            logging.info("Skipped %s: %s", address_text(*candidate[:2]), describe_error(error))
            return None

    def _probe_saved(self, ip, port):
        """Probe a saved address; when nothing answers, list it so it can be removed."""
        try:
            return self.probe(ip, port, "manual")
        except (DirectProtocolError, DirectRequestError, OSError, TimeoutError, ValueError) as error:
            text = address_text(ip, port)
            return Device(id=UNIDENTIFIED + text, ip_address=ip, port=port, source="manual",
                          name=text, error=describe_error(error))

    def add(self, address):
        ip, port = parse_address(address)
        text = address_text(ip, port)
        with self.lock:
            if text not in self.manual and len(self.manual) >= MAX_MANUAL_ADDRESSES:
                raise RejectedInput(f"Add at most {MAX_MANUAL_ADDRESSES} addresses.")
        try:
            device = self.probe(ip, port, "manual")
        except (DirectProtocolError, DirectRequestError, OSError, TimeoutError, ValueError) as error:
            # describe_error writes the same text the device list shows.
            raise RejectedInput(f"No ESPectre device answered at {text}: {describe_error(error)}") from None
        with self.lock:
            if text not in self.manual:
                self.manual.append(text)
                self._save_manual()
        device.source = "manual"
        with self.lock:
            self.devices[device.id] = device
        return device

    def remove(self, device_id):
        with self.lock:
            device = self.devices.get(device_id)
            if device is None or device.source != "manual":
                raise KeyError(device_id)
            text = address_text(device.ip_address, device.port)
            self.manual = [address for address in self.manual if address != text]
            self._save_manual()
            del self.devices[device_id]

    def set_mode(self, device_ids, mode):
        """Change each device on its own; a failure never stops the others."""
        with self.lock:
            targets = [(device_id, self.devices.get(device_id)) for device_id in device_ids]

        def change(item):
            device_id, device = item
            if device is None:
                return {"id": device_id, "status": "error", "message": "Device is no longer listed."}
            if not device.can_control:
                return {"id": device_id, "status": "error", "message": "This device cannot change its traffic source."}
            current = device.sensing.get("traffic_generator_mode")
            if current in TRAFFIC_MODES and current != "external":
                self.internal_modes[device_id] = current
            option = self.internal_modes.get(device_id, "ping") if mode == INTERNAL else mode
            result = {"id": device_id, "option": option}
            if option not in device.modes:
                return {**result, "status": "error", "message": f"This device does not offer {option}."}
            return self._apply(device, "traffic_generator_mode", option, result)

        with ThreadPoolExecutor(max_workers=REFRESH_WORKERS) as pool:
            return list(pool.map(change, targets))

    def set_spacing(self, device_ids, spacing):
        """Set the CSI sample spacing on each device that supports it."""
        with self.lock:
            targets = [(device_id, self.devices.get(device_id)) for device_id in device_ids]

        def change(item):
            device_id, device = item
            result = {"id": device_id, "option": spacing}
            if device is None:
                return {**result, "status": "error", "message": "Device is no longer listed."}
            if not device.can_control or not device.spacing_supported:
                return {**result, "status": "error", "message": "This device has no packet spacing setting."}
            return self._apply(device, "csi_sample_spacing", spacing, result)

        with ThreadPoolExecutor(max_workers=REFRESH_WORKERS) as pool:
            return list(pool.map(change, targets))

    def _apply(self, device, name, option, result):
        """Write one sensing field, then confirm it from a fresh read."""
        if device.sensing.get(name) == option and device.error is None:
            return {**result, "status": "unchanged", "message": f"Already {option}."}
        client = self._client(device)
        try:
            client.request("patch", "sensing", {name: option})
        except (DirectProtocolError, DirectRequestError, OSError, TimeoutError) as error:
            return {**result, "status": "error", "message": describe_error(error)}
        finally:
            client.close()
        self.refresh(device)
        if device.sensing.get(name) != option:
            return {**result, "status": "unconfirmed",
                    "message": "The device accepted the change but still reports another value."}
        logging.info("%s of %s set to %s", name, device.name, option)
        return {**result, "status": "confirmed", "message": f"Now {option}."}

    def snapshot(self):
        with self.lock:
            devices = sorted(self.devices.values(), key=lambda device: (device.name.lower(), device.id))
            return {"devices": [device.as_dict() for device in devices], "scanning": self.scanning,
                    "scan_error": self.scan_error, "scanned_at": self.scanned_at}
