# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Unit tests for diagnostic report generation and privacy redaction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from espectre_cli.app import build_parser
from espectre_cli.diagnostic_report import (
    collect_diagnostic_snapshot,
    format_markdown_report,
    format_text_report,
    redact_diagnostic_value,
    redact_diagnostics,
    redact_ipv4_address,
    redact_mac_address,
    run_diagnostic_report_command,
)


def sample_snapshot_payload() -> dict:
    return {
        "device": {
            "frontend": "native",
            "firmware": "v0.9.1",
            "chip": "esp32c3",
            "device_id": "0123456789abcdef",
            "csi_profile": "standard",
        },
        "health": {
            "status": "healthy",
            "online": True,
            "uptime_s": 12345,
        },
        "sensing": {
            "enabled": True,
            "ready": True,
            "calibrating": False,
            "mode": "passive",
            "detector": "lightweight",
            "threshold": 1.25,
            "traffic_generator_mode": "disabled",
            "csi_target_pps": 100,
            "csi_sample_spacing": "uniform",
        },
        "wifi": {
            "configured": True,
            "connected": True,
            "band": "2.4GHz",
            "channel": 6,
            "rssi_dbm": -55,
            "ssid": "SuperSecretOfficeWiFi",
            "bssid": "18:fe:34:d2:91:a4",
            "ip": "192.168.1.42",
        },
        "diagnostics": {
            "csi_packets_rx": 45000,
            "csi_buffer_overruns": 0,
            "wifi_sta_ip": "192.168.1.42",
            "wifi_sta_mac": "18:fe:34:d2:91:a4",
        },
    }


def test_redact_mac_address():
    result, count = redact_mac_address("18:fe:34:d2:91:a4")
    assert result == "18:fe:**:**:**:**"
    assert count == 1

    result_dash, count_dash = redact_mac_address("18-FE-34-D2-91-A4")
    assert result_dash == "18-FE-**-**-**-**"
    assert count_dash == 1

    text = "AP at 00:11:22:33:44:55 connected to 66:77:88:99:aa:bb"
    result_multi, count_multi = redact_mac_address(text)
    assert result_multi == "AP at 00:11:**:**:**:** connected to 66:77:**:**:**:**"
    assert count_multi == 2


def test_redact_ipv4_address():
    result, count = redact_ipv4_address("192.168.1.42")
    assert result == "192.168.x.x"
    assert count == 1

    result_loopback, count_loopback = redact_ipv4_address("127.0.0.1")
    assert result_loopback == "127.0.0.1"
    assert count_loopback == 0

    result_broadcast, count_broadcast = redact_ipv4_address("255.255.255.255")
    assert result_broadcast == "255.255.255.255"
    assert count_broadcast == 0

    result_unspecified, count_unspecified = redact_ipv4_address("0.0.0.0")
    assert result_unspecified == "0.0.0.0"
    assert count_unspecified == 0


def test_redact_diagnostic_value():
    stats: dict[str, int] = {}

    assert redact_diagnostic_value("password", "secret123", stats) == "[REDACTED]"
    assert stats["redacted_fields"] == 1

    assert redact_diagnostic_value("ssid", "MyHomeWifi", stats) == "<redacted_ssid>"
    assert stats["ssid"] == 1

    assert redact_diagnostic_value("bssid", "11:22:33:44:55:66", stats) == "11:22:**:**:**:**"
    assert stats["mac"] >= 1

    assert redact_diagnostic_value("ip", "10.0.0.5", stats) == "10.0.x.x"
    assert stats["ip"] >= 1

    assert redact_diagnostic_value("device_id", "0123456789abcdef", stats) == "0123...cdef"
    assert stats["device_id"] == 1


def test_redact_diagnostics_full_payload():
    payload = sample_snapshot_payload()
    payload["diagnostics"]["admin_password"] = "mock-admin-password"
    payload["diagnostics"]["api_token"] = "mock-bearer-token"
    redacted, stats = redact_diagnostics(payload)

    assert redacted["wifi"]["ssid"] == "<redacted_ssid>"
    assert redacted["wifi"]["bssid"] == "18:fe:**:**:**:**"
    assert redacted["wifi"]["ip"] == "192.168.x.x"
    assert redacted["device"]["device_id"] == "0123...cdef"
    assert redacted["diagnostics"]["admin_password"] == "[REDACTED]"
    assert redacted["diagnostics"]["api_token"] == "[REDACTED]"
    assert redacted["diagnostics"]["csi_packets_rx"] == 45000
    assert "redacted_fields" in stats
    assert "mac" in stats
    assert "ip" in stats
    assert "ssid" in stats


def test_collect_diagnostic_snapshot():
    mock_client = MagicMock()

    def mock_request(method, resource, params=None):
        if resource == "device":
            return {"chip": "esp32c3"}
        if resource == "health":
            return {"status": "ok"}
        if resource == "sensing":
            return {"enabled": True}
        if resource == "wifi":
            return {"connected": True}
        if resource == "diagnostics":
            raise RuntimeError("diagnostics endpoint timed out")
        return {}

    mock_client.request.side_effect = mock_request
    snapshot = collect_diagnostic_snapshot(mock_client)

    assert snapshot["device"] == {"chip": "esp32c3"}
    assert snapshot["health"] == {"status": "ok"}
    assert snapshot["sensing"] == {"enabled": True}
    assert snapshot["wifi"] == {"connected": True}
    assert "error" in snapshot["diagnostics"]
    assert "timed out" in snapshot["diagnostics"]["error"]


def test_format_markdown_report():
    payload = sample_snapshot_payload()
    payload["diagnostics"]["admin_password"] = "mock-admin-password"
    redacted, stats = redact_diagnostics(payload)
    report = format_markdown_report(redacted, stats)

    assert "# ESPectre Diagnostic Report" in report
    assert "Privacy Notice" in report
    assert "Redaction Summary" in report
    assert "## Device & Firmware" in report
    assert "## Health & Runtime" in report
    assert "## Sensing State" in report
    assert "## Wi-Fi & Connectivity" in report
    assert "## CSI Diagnostics & Metrics" in report
    assert "<redacted_ssid>" in report
    assert "18:fe:**:**:**:**" in report
    assert "[REDACTED]" in report


def test_format_text_report():
    payload = sample_snapshot_payload()
    redacted, stats = redact_diagnostics(payload)
    report = format_text_report(redacted, stats)

    assert "=== ESPectre Diagnostic Report ===" in report
    assert "[DEVICE]" in report
    assert "[HEALTH]" in report
    assert "[SENSING]" in report
    assert "[WIFI]" in report
    assert "[DIAGNOSTICS]" in report


def test_cli_parser_report_arguments():
    parser = build_parser()

    args_file = parser.parse_args(["report", "--file", "input.json", "--format", "json", "-o", "out.json"])
    assert args_file.namespace == "report"
    assert args_file.file == "input.json"
    assert args_file.format == "json"
    assert args_file.output == "out.json"

    args_target = parser.parse_args(["report", "--target", "192.168.1.100", "--format", "text"])
    assert args_target.namespace == "report"
    assert args_target.target == "192.168.1.100"
    assert args_target.format == "text"


def test_run_diagnostic_report_command_from_file(tmp_path: Path, capsys: pytest.CaptureFixture):
    raw_payload = sample_snapshot_payload()
    input_file = tmp_path / "raw.json"
    output_file = tmp_path / "report.md"

    input_file.write_text(json.dumps(raw_payload), encoding="utf-8")

    args = argparse.Namespace(
        file=str(input_file),
        target=None,
        format="markdown",
        output=str(output_file),
    )

    exit_code = run_diagnostic_report_command(args)
    assert exit_code == 0
    assert output_file.exists()
    content = output_file.read_text(encoding="utf-8")
    assert "# ESPectre Diagnostic Report" in content
    assert "<redacted_ssid>" in content


def test_run_diagnostic_report_command_stdout(tmp_path: Path, capsys: pytest.CaptureFixture):
    raw_payload = sample_snapshot_payload()
    input_file = tmp_path / "raw.json"
    input_file.write_text(json.dumps(raw_payload), encoding="utf-8")

    args = argparse.Namespace(
        file=str(input_file),
        target=None,
        format="json",
        output=None,
    )

    exit_code = run_diagnostic_report_command(args)
    assert exit_code == 0
    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert parsed["wifi"]["ssid"] == "<redacted_ssid>"


def test_run_diagnostic_report_command_file_not_found(capsys: pytest.CaptureFixture):
    args = argparse.Namespace(
        file="/nonexistent/path/diagnostic.json",
        target=None,
        format="markdown",
        output=None,
    )
    exit_code = run_diagnostic_report_command(args)
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Diagnostic report generation failed" in captured.err
