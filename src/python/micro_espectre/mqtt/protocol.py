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


def command_registry(
    supports_info=True,
    supports_diagnostics=False,
    supports_device_config=False,
    supports_runtime_threshold=False,
    supports_runtime_motion_hits=False,
    supports_runtime_detector=False,
    supports_manual_recalibration=False,
    supports_traffic_control=False,
    supports_ota=False,
):
    """Return the filtered canonical command registry for this frontend."""
    empty = {"additionalProperties": False}
    commands = [
        {"name": "capabilities", "kind": "query", "access": "read", "params": empty, "result": "capabilities"},
        {"name": "status", "kind": "query", "access": "read", "params": empty, "result": "status"},
        {"name": "config", "kind": "query", "access": "read", "params": empty, "result": "config"},
    ]
    if supports_info:
        commands.insert(1, {"name": "info", "kind": "query", "access": "read", "params": empty, "result": "info"})
    if supports_diagnostics:
        commands.append({"name": "diagnostics", "kind": "query", "access": "read", "params": empty, "result": "diagnostics"})
    if supports_device_config:
        commands.append({"name": "set_device_label", "kind": "mutation", "access": "device_admin", "params": {"type": "object", "properties": {"device_label": {"type": "string"}}, "required": ["device_label"], "additionalProperties": False}})
    if supports_runtime_threshold:
        commands.append({"name": "set_threshold", "kind": "mutation", "access": "control", "params": {"type": "object", "properties": {"threshold": {"type": "number", "minimum": 0, "maximum": 1}}, "required": ["threshold"], "additionalProperties": False}})
    if supports_runtime_motion_hits:
        commands.append({"name": "set_motion_hits", "kind": "mutation", "access": "control", "params": {"type": "object", "properties": {"motion_on_hits": {"type": "integer", "minimum": 1, "maximum": 20}, "motion_off_hits": {"type": "integer", "minimum": 1, "maximum": 20}}, "required": ["motion_on_hits", "motion_off_hits"], "additionalProperties": False}})
    if supports_runtime_detector:
        commands.append({"name": "set_detector", "kind": "mutation", "access": "control", "params": {"type": "object", "properties": {"detector": {"type": "string", "enum": ["lightweight", "high_accuracy"]}}, "required": ["detector"], "additionalProperties": False}})
    if supports_manual_recalibration:
        commands.append({"name": "recalibrate", "kind": "action", "access": "control", "params": empty})
    if supports_traffic_control:
        commands.append({"name": "set_csi_traffic_mode", "kind": "mutation", "access": "control", "params": {"type": "object", "properties": {"csi_traffic_mode": {"type": "string", "enum": ["internal", "external"]}}, "required": ["csi_traffic_mode"], "additionalProperties": False}})
        commands.append({"name": "set_traffic_generator_mode", "kind": "mutation", "access": "control", "params": {"type": "object", "properties": {"traffic_generator_mode": {"type": "string", "enum": ["ping", "dns"]}}, "required": ["traffic_generator_mode"], "additionalProperties": False}})
    if supports_ota:
        ota_params = {"type": "object", "properties": {"channel": {"type": "string", "enum": ["release", "preview", "develop"]}}, "required": [], "additionalProperties": False}
        commands.extend([
            {"name": "ota_status", "kind": "query", "access": "firmware_update", "params": empty, "result": "ota_status"},
            {"name": "ota_check", "kind": "action", "access": "firmware_update", "params": ota_params},
            {"name": "ota_start", "kind": "action", "access": "firmware_update", "params": ota_params},
        ])
    return commands


def build_capabilities_payload(device_id, **supports):
    commands = command_registry(**supports)
    config_sections = ["runtime"]
    if supports.get("supports_device_config"):
        config_sections.append("device")
    return {
        "protocol_version": "1.0",
        "device_id": device_id,
        "commands": commands,
        "events": ["telemetry", "status", "info", "config", "fault"],
        "config_sections": config_sections,
        "features": {"raw_csi": False},
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
