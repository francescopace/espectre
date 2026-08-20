# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Small MQTT protocol helpers shared by commands and Home Assistant."""

THRESHOLD_MIN = 0.0
THRESHOLD_MAX = 1.0

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


def _protocol_mqtt_commands(
    supports_info=True,
    supports_stats=False,
    supports_runtime_threshold=False,
    supports_runtime_motion_hits=False,
    supports_runtime_detector=False,
    supports_manual_recalibration=False,
    supports_traffic_control=False,
    supports_ble=False,
    supports_ota=False,
):
    """Return the MQTT command names advertised by this frontend."""
    commands = ["commands"]
    if supports_info:
        commands.append("info")
    if supports_stats:
        commands.append("stats")
    if supports_runtime_threshold:
        commands.append("set_threshold")
    if supports_runtime_motion_hits:
        commands.append("set_motion_hits")
    if supports_runtime_detector:
        commands.append("set_detector")
    if supports_manual_recalibration:
        commands.append("recalibrate")
    if supports_traffic_control:
        commands.append("set_csi_traffic_mode")
        commands.append("set_traffic_generator_mode")
    if supports_ble:
        commands.append("set_ble")
    if supports_ota:
        commands.extend(["ota_status", "ota_check", "ota_start"])
    return commands


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
    runtime_policy=None,
    ha_adapter=None,
    recalibrate_callback=None,
    traffic_control_supported=False,
    device_id=None,
):
    """Build the retained frontend information payload without loading commands."""
    import sys

    channel_primary = 0
    chip = getattr(global_state, "chip_type", None) or sys.platform
    if wlan.active():
        try:
            channel_primary = wlan.config("channel")
        except Exception:  # pragma: no cover
            pass

    if device_id is None:
        device_id = derive_runtime_device_id(wlan)

    return {
        "protocol_version": "1.0",
        "device_id": device_id,
        "device_name": _protocol_device_name(device_id, chip),
        "device_label": getattr(config, "MQTT_DEVICE_LABEL", ""),
        "frontend": "micro",
        "firmware_version": "micropython",
        "chip": chip,
        "supports_info": True,
        "supports_stats": True,
        "supports_runtime_threshold": True,
        "supports_runtime_motion_hits": runtime_policy is not None,
        "supports_runtime_detector": False,
        "supports_manual_recalibration": callable(recalibrate_callback),
        "supports_traffic_control": bool(traffic_control_supported),
        "supports_ota": False,
        "supports_ble": False,
        "network": {
            "channel": {"primary": channel_primary},
        },
        "detection": {"algorithm": detector_algorithm},
        "csi_traffic_mode": getattr(ha_adapter, "_last_csi_traffic_mode", "internal"),
        "traffic_mode": getattr(ha_adapter, "_last_traffic_generator_mode", "ping"),
        "csi_target_pps": max(1, int(getattr(config, "CSI_TARGET_PPS", 100))),
        "evaluation_interval_ms": max(1, int(getattr(config, "EVALUATION_INTERVAL_MS", 250))),
        "publish_interval_ms": max(1, int(getattr(config, "PUBLISH_INTERVAL_MS", 1000))),
    }
