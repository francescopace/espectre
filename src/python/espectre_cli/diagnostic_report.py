# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Diagnostic report generation and privacy redaction for ESPectre devices."""

from __future__ import annotations

import json
import re
import sys
from typing import Any

from .device_discovery import discover_devices, select_discovered_device
from .device_transport import DEFAULT_DIRECT_ORIGIN, DirectClient, direct_endpoint_from_device_url

MAC_REGEX = re.compile(r"\b([0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b")
IPV4_REGEX = re.compile(r"\b((?:1\d\d|2[0-4]\d|25[0-5]|[1-9]\d|\d)\.){3}(?:1\d\d|2[0-4]\d|25[0-5]|[1-9]\d|\d)\b")


def redact_mac_address(value: str) -> tuple[str, int]:
    """Redact the last four octets of MAC addresses, preserving the OUI prefix."""
    count = 0

    def _replace_mac(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        mac = match.group(0)
        sep = ":" if ":" in mac else "-"
        parts = mac.split(sep)
        return f"{parts[0]}{sep}{parts[1]}{sep}**{sep}**{sep}**{sep}**"

    result = MAC_REGEX.sub(_replace_mac, value)
    return result, count


def redact_ipv4_address(value: str) -> tuple[str, int]:
    """Redact host octets of unicast IPv4 addresses while preserving network prefix and loopback."""
    count = 0

    def _replace_ip(match: re.Match[str]) -> str:
        nonlocal count
        ip = match.group(0)
        if ip in ("127.0.0.1", "0.0.0.0", "255.255.255.255"):
            return ip
        parts = ip.split(".")
        count += 1
        return f"{parts[0]}.{parts[1]}.x.x"

    result = IPV4_REGEX.sub(_replace_ip, value)
    return result, count


def redact_diagnostic_value(key: str, val: Any, stats: dict[str, int]) -> Any:
    """Recursively redact privacy-sensitive fields, strings, and network identifiers."""
    lower_key = key.lower()

    if "password" in lower_key or "secret" in lower_key or "token" in lower_key or "credential" in lower_key:
        if val:
            stats["secret"] = stats.get("secret", 0) + 1
            return "[REDACTED]"
        return val

    if lower_key == "ssid":
        if val and isinstance(val, str):
            stats["ssid"] = stats.get("ssid", 0) + 1
            return "<redacted_ssid>"
        return val

    if lower_key in ("bssid", "target_mac") or lower_key.endswith("_mac"):
        if val and isinstance(val, str):
            redacted, count = redact_mac_address(val)
            stats["mac"] = stats.get("mac", 0) + (count or 1)
            return redacted
        return val

    if lower_key in ("ip", "client_ip", "host_ip") or lower_key.endswith("_ip"):
        if val and isinstance(val, str):
            redacted, count = redact_ipv4_address(val)
            stats["ip"] = stats.get("ip", 0) + (count or 1)
            return redacted
        return val

    if lower_key == "device_id" and isinstance(val, str) and len(val) >= 8:
        stats["device_id"] = stats.get("device_id", 0) + 1
        return f"{val[:4]}...{val[-4:]}"

    if isinstance(val, str):
        val_mac, mac_count = redact_mac_address(val)
        val_ip, ip_count = redact_ipv4_address(val_mac)
        if mac_count:
            stats["mac"] = stats.get("mac", 0) + mac_count
        if ip_count:
            stats["ip"] = stats.get("ip", 0) + ip_count
        return val_ip

    if isinstance(val, dict):
        return {k: redact_diagnostic_value(k, v, stats) for k, v in val.items()}

    if isinstance(val, list):
        return [redact_diagnostic_value(key, item, stats) for item in val]

    return val


def redact_diagnostics(data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    """Redact an entire diagnostic payload and return the sanitized data and stats."""
    stats: dict[str, int] = {}
    redacted = redact_diagnostic_value("root", data, stats)
    return redacted, stats


def collect_diagnostic_snapshot(client: DirectClient) -> dict[str, Any]:
    """Fetch diagnostic resources from a device over Direct HTTP."""
    snapshot: dict[str, Any] = {}
    resources = [
        ("device", "device", {}),
        ("health", "health", {}),
        ("sensing", "sensing", {}),
        ("wifi", "wifi", {}),
        ("diagnostics", "diagnostics", {"fields": ["*"]}),
    ]

    for key, resource, params in resources:
        try:
            snapshot[key] = client.request("get", resource, params)
        except Exception as exc:  # noqa: BLE001
            snapshot[key] = {"error": str(exc)}

    return snapshot


def format_markdown_report(data: dict[str, Any], stats: dict[str, int]) -> str:
    """Format a diagnostic dictionary as Markdown suitable for GitHub issues."""
    lines: list[str] = [
        "# ESPectre Diagnostic Report",
        "",
        "> **Privacy Notice**: Network and personal identifiers (SSID, BSSID/MAC, IP addresses) have been automatically redacted for public sharing.",
        "",
    ]

    if stats:
        summary_items = [f"{k} ({v})" for k, v in sorted(stats.items())]
        lines.extend([
            "### Redaction Summary",
            f"- Redacted identifiers: {', '.join(summary_items)}",
            "",
        ])

    device = data.get("device", {})
    lines.extend([
        "## Device & Firmware",
        f"- **Frontend**: {device.get('frontend', 'unknown')}",
        f"- **Firmware Version**: {device.get('firmware', 'unknown')}",
        f"- **Chip Family**: {device.get('chip', 'unknown')}",
        f"- **Device ID**: `{device.get('device_id', 'unknown')}`",
        f"- **CSI Profile**: {device.get('csi_profile', 'unknown')}",
        "",
    ])

    health = data.get("health", {})
    lines.extend([
        "## Health & Runtime",
        f"- **Status**: {health.get('status', 'unknown')}",
        f"- **Online**: {health.get('online', 'unknown')}",
        f"- **Uptime**: {health.get('uptime_s', 'unknown')} s",
        "",
    ])

    sensing = data.get("sensing", {})
    lines.extend([
        "## Sensing State",
        f"- **Enabled**: {sensing.get('enabled', 'unknown')}",
        f"- **Ready**: {sensing.get('ready', 'unknown')}",
        f"- **Calibrating**: {sensing.get('calibrating', 'unknown')}",
        f"- **Mode**: {sensing.get('mode', 'unknown')}",
        f"- **Detector**: {sensing.get('detector', 'unknown')}",
        f"- **Threshold**: {sensing.get('threshold', 'unknown')}",
        f"- **Traffic Generator Mode**: {sensing.get('traffic_generator_mode', 'unknown')}",
        f"- **CSI Target PPS**: {sensing.get('csi_target_pps', 'unknown')}",
        f"- **CSI Sample Spacing**: {sensing.get('csi_sample_spacing', 'default')}",
        "",
    ])

    wifi = data.get("wifi", {})
    lines.extend([
        "## Wi-Fi & Connectivity",
        f"- **Configured**: {wifi.get('configured', 'unknown')}",
        f"- **Connected**: {wifi.get('connected', 'unknown')}",
        f"- **Band / Channel**: {wifi.get('band', 'unknown')} / {wifi.get('channel', 'unknown')}",
        f"- **RSSI**: {wifi.get('rssi_dbm', 'unknown')} dBm",
        f"- **SSID**: `{wifi.get('ssid', '<none>')}`",
        f"- **BSSID**: `{wifi.get('bssid', '<none>')}`",
        f"- **IP Address**: `{wifi.get('ip', '<none>')}`",
        "",
    ])

    diagnostics = data.get("diagnostics", {})
    lines.extend([
        "## CSI Diagnostics & Metrics",
    ])
    if isinstance(diagnostics, dict) and "error" not in diagnostics and diagnostics:
        for k, v in sorted(diagnostics.items()):
            lines.append(f"- **{k}**: {v}")
    else:
        lines.append(f"- Status: {diagnostics.get('error', 'unavailable')}")
    lines.append("")

    return "\n".join(lines).rstrip()


def format_text_report(data: dict[str, Any], stats: dict[str, int]) -> str:
    """Format a diagnostic dictionary as human-readable plain text."""
    lines: list[str] = [
        "=== ESPectre Diagnostic Report ===",
        "Privacy Notice: Network and personal identifiers have been automatically redacted.",
        "",
    ]
    if stats:
        lines.append(f"Redaction count: {', '.join(f'{k}={v}' for k, v in sorted(stats.items()))}")
        lines.append("")

    for section in ("device", "health", "sensing", "wifi", "diagnostics"):
        content = data.get(section, {})
        lines.append(f"[{section.upper()}]")
        if isinstance(content, dict):
            for k, v in sorted(content.items()):
                lines.append(f"  {k}: {v}")
        else:
            lines.append(f"  {content}")
        lines.append("")

    return "\n".join(lines).rstrip()


def _resolve_report_endpoint(args) -> str:
    """Resolve device Direct HTTP endpoint from CLI arguments or mDNS discovery."""
    target = getattr(args, "target", None) or getattr(args, "endpoint", None)
    if target:
        if getattr(args, "chip", None):
            raise ValueError("--chip requires automatic frontend discovery without --target")
        endpoint = target if "://" in target else f"http://{target}"
        return direct_endpoint_from_device_url(endpoint)

    frontend = getattr(args, "frontend", None)
    discovery_timeout = float(getattr(args, "discovery_timeout", 5.0))
    chip = getattr(args, "chip", None)
    records = discover_devices(frontend=frontend, timeout_s=discovery_timeout)
    return select_discovered_device(
        records,
        frontend_label=frontend,
        chip=chip,
    ).endpoint


def run_diagnostic_report_command(args) -> int:
    """CLI handler for the 'espectre report' command."""
    try:
        if getattr(args, "file", None):
            with open(args.file, encoding="utf-8") as f:
                raw_data = json.load(f)
        else:
            endpoint = _resolve_report_endpoint(args)
            origin = getattr(args, "origin", DEFAULT_DIRECT_ORIGIN)
            timeout = float(getattr(args, "timeout", 8.0))
            with DirectClient(endpoint, origin=origin, timeout=timeout) as client:
                client.negotiate()
                raw_data = collect_diagnostic_snapshot(client)

        redacted_data, stats = redact_diagnostics(raw_data)

        if getattr(args, "format", "markdown") == "json":
            output = json.dumps(redacted_data, indent=2, sort_keys=True)
        elif getattr(args, "format", "markdown") == "text":
            output = format_text_report(redacted_data, stats)
        else:
            output = format_markdown_report(redacted_data, stats)

        if getattr(args, "output", None):
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output + "\n")
            print(f"Diagnostic report written to {args.output}")
        else:
            print(output)
        return 0
    except (OSError, RuntimeError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        print(f"Diagnostic report generation failed: {exc}", file=sys.stderr)
        return 1


__all__ = [
    "collect_diagnostic_snapshot",
    "format_markdown_report",
    "format_text_report",
    "redact_diagnostic_value",
    "redact_diagnostics",
    "redact_ipv4_address",
    "redact_mac_address",
    "run_diagnostic_report_command",
]
