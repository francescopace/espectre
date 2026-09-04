# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Canonical ESPectre protocol helpers used by the Micro Direct endpoint."""

THRESHOLD_MIN = 0.0
THRESHOLD_MAX = 1.0
PROTOCOL_VERSION = "1.0"
DNS_SD_TXT_SCHEMA_VERSION = "1"


def build_command_request(command_id, command, **params):
    """Build the canonical request message carried by every transport."""
    payload = {
        "command_id": command_id,
        "command": command,
    }
    payload.update(params)
    return payload


def build_command_result(device_id, command_id, command, accepted, code, message, data=None):
    """Build the canonical correlated result or error message."""
    payload = {
        "command_id": command_id,
        "command": command,
        "accepted": bool(accepted),
        "code": code,
        "message": message,
    }
    if data is not None:
        payload["data"] = data
    return payload


def build_fault_payload(device_id, message, timestamp_ms):
    """Build the canonical runtime fault event."""
    return {
        "timestamp_ms": timestamp_ms,
        "message": message,
    }


def build_status_payload(device_id, online, timestamp_ms, **state):
    """Build the canonical health resource."""
    del device_id, state
    payload = {
        "status": "ok" if online else "offline",
        "online": bool(online),
        "uptime_s": max(0, int(timestamp_ms)) // 1000,
        "timestamp_ms": timestamp_ms,
    }
    return payload


def build_motion_payload(
    device_id,
    frontend,
    timestamp_ms,
    motion_state,
    movement_score,
    threshold,
    detector,
    uptime_s,
):
    """Build one canonical motion event."""
    del device_id, frontend, threshold, detector, uptime_s
    return {
        "timestamp_ms": timestamp_ms,
        "state": motion_state,
        "score": movement_score,
    }


DIAGNOSTIC_FIELDS = (
    "free_memory_kb",
    "minimum_free_memory_kb",
    "largest_free_memory_kb",
    "cpu_frequency_mhz",
    "loop_time_ms",
    "performance_window_ready",
    "performance_window_ms",
    "runtime_load_percent",
    "loop_samples",
    "loop_avg_us",
    "loop_max_us",
    "detection_timing_supported",
    "detection_samples",
    "detection_sum_us",
    "detection_avg_us",
    "detection_min_us",
    "detection_max_us",
    "traffic_tx_pps",
    "csi_callback_pps",
    "csi_accepted_pps",
    "csi_admitted_pps",
    "csi_filtered_pps",
    "csi_missing_slots_pps",
    "csi_excess_pps",
    "csi_stale_pps",
    "csi_out_of_order_pps",
    "csi_occupancy",
    "wifi_channel",
    "wifi_rssi_dbm",
)


def build_diagnostics_payload(device_id, timestamp_ms, uptime_s, measurements=None):
    """Build a canonical diagnostics result from supported Micro measurements."""
    payload = {
        "timestamp_ms": int(timestamp_ms),
        "uptime": max(0, int(uptime_s)),
    }
    if isinstance(measurements, dict):
        for key in DIAGNOSTIC_FIELDS:
            if key in measurements:
                payload[key] = measurements[key]
    return payload


def build_protocol_catalog():
    """Return executable transport-neutral samples for the C++ parity check."""
    device_id = "0000000000000000"
    command_id = "contract-1"
    command = "update_sensing"
    return {
        "protocol_version": PROTOCOL_VERSION,
        "dns_sd": {
            "txtvers": DNS_SD_TXT_SCHEMA_VERSION,
            "protovers": PROTOCOL_VERSION,
        },
        "messages": {
            "request": build_command_request(command_id, command, threshold=0.5),
            "result": build_command_result(
                device_id, command_id, command, True, "ok", "threshold updated", {"threshold": 0.5}
            ),
            "error": build_command_result(
                device_id, command_id, command, False, "invalid_params", "threshold is invalid"
            ),
            "events": {
                "names": ["motion", "health", "device", "sensing", "ota", "fault"],
                "health": build_status_payload(device_id, True, 1000),
                "motion": build_motion_payload(
                    device_id, "micro", 1000, "motion", 0.25, 0.5, "lightweight", 1
                ),
                "fault": build_fault_payload(device_id, "runtime fault", 1000),
            },
        },
    }

_DEVICE_ID_DOMAIN = b"espectre-device-id-v1"


def _derive_device_id_from_mac(mac):
    """Derive the canonical Native-compatible device pseudonym from MAC bytes."""
    import hashlib

    mac = bytes(mac)
    if len(mac) < 6:
        return "0000000000000000"
    digest = hashlib.sha256(_DEVICE_ID_DOMAIN + mac[:6]).digest()
    return digest[:8].hex()


def derive_runtime_device_id(wlan):
    """Read the station MAC and derive its canonical runtime device pseudonym."""
    try:
        return _derive_device_id_from_mac(wlan.config("mac"))
    except Exception:
        return "0000000000000000"


def command_registry():
    """Return the command names implemented by Micro."""
    return [
        {
            "name": "read_diagnostics",
            "method": "GET",
            "path": "/espectre/v1/diagnostics",
        },
        {
            "name": "recalibrate",
            "method": "POST",
            "path": "/espectre/v1/sensing/calibrations",
        },
    ]


def build_capabilities_payload(device_id):
    """Build the exact Direct capability response exposed by Micro."""
    return {
        "protocol_version": PROTOCOL_VERSION,
        "operations": command_registry(),
        "events": ["motion"],
        "resources": ["health", "device", "capabilities", "sensing", "wifi", "diagnostics"],
        "features": {"csi": False},
    }


def _is_ascii_alnum(char):
    """Return whether one character is an ASCII letter or digit."""
    code = ord(char)
    return 48 <= code <= 57 or 65 <= code <= 90 or 97 <= code <= 122


def _normalize_chip_label(chip):
    """Normalize chip identifiers to the shared short labels used by firmware."""
    if not chip:
        return "UNK"
    normalized = "".join(ch for ch in str(chip).upper() if _is_ascii_alnum(ch))
    if normalized == "ESP32C3":
        return "C3"
    if normalized == "ESP32C5":
        return "C5"
    if normalized == "ESP32C6":
        return "C6"
    if normalized == "ESP32S2":
        return "S2"
    if normalized == "ESP32S3":
        return "S3"
    if normalized == "ESP32":
        return "ESP32"
    return normalized or "UNK"


def _protocol_device_name(device_id, chip):
    """Build the immutable ESPectre device name from chip and device_id."""
    compact_id = "".join(ch for ch in str(device_id).lower() if _is_ascii_alnum(ch))
    suffix = compact_id[-6:] if compact_id else "000000"
    return "ESPectre {} {}".format(_normalize_chip_label(chip), suffix)


def build_info_payload(
    config,
    detector_algorithm,
    wlan,
    global_state=None,
    device_id=None,
    csi_traffic_mode="internal",
    traffic_mode="ping",
    firmware_version="unknown",
    chip=None,
):
    """Build the current Micro frontend information payload."""
    import sys

    channel_primary = 0
    chip = chip or getattr(global_state, "chip_type", None) or sys.platform
    if wlan.active():
        try:
            channel_primary = wlan.config("channel")
        except Exception:  # pragma: no cover
            pass

    if device_id is None:
        device_id = derive_runtime_device_id(wlan)

    try:
        from src.device_utils import select_csi_capture_profile
    except ImportError:
        from device_utils import select_csi_capture_profile
    csi_profile = select_csi_capture_profile(chip, channel_primary)

    return {
        "device_id": device_id,
        "name": getattr(config, "DEVICE_LABEL", "") or _protocol_device_name(device_id, chip),
        "label": getattr(config, "DEVICE_LABEL", ""),
        "frontend": "micro",
        "firmware": firmware_version or "unknown",
        "chip": chip,
        "csi_profile": csi_profile,
    }
