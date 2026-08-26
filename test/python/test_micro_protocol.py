# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Micro-ESPectre protocol and read-only Direct facade contracts."""

import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import protocol


def test_device_id_matches_native_sha256_pseudonym():
    assert protocol._derive_device_id_from_mac(bytes.fromhex("7c2c6742bbac")) == "3cf79180d3a0aca4"


def test_cpp_and_python_protocol_catalogs_match():
    repo_root = Path(__file__).resolve().parents[2]
    build_dir = repo_root / "test" / "cpp" / "build"
    probe = build_dir / "suites" / "espectre_capabilities_probe"
    if not probe.exists():
        subprocess.run(["cmake", "-S", str(repo_root / "test" / "cpp"), "-B", str(build_dir)], check=True)
        subprocess.run(["cmake", "--build", str(build_dir), "--target", "espectre_capabilities_probe"], check=True)

    cpp_catalog = json.loads(subprocess.run(
        [str(probe), "micro"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout)
    python_catalog = {
        "capabilities": protocol.build_capabilities_payload("0000000000000000"),
        "message_model": protocol.build_protocol_catalog(),
    }
    assert cpp_catalog == python_catalog


def test_micro_capabilities_are_read_only_and_minimal():
    payload = protocol.build_capabilities_payload("0123456789abcdef")
    commands = {command["name"] for command in payload["commands"]}

    assert commands == {"capabilities", "info", "status", "config"}
    assert all(command["access"] == "read" for command in payload["commands"])
    assert payload["events"] == ["telemetry"]
    assert payload["config_sections"] == ["runtime", "device", "wifi"]
    assert payload["features"] == {"raw_csi": False}


def test_direct_facade_starts_and_publishes_canonical_telemetry(monkeypatch):
    native = MagicMock()
    monkeypatch.setitem(sys.modules, "espectre_native_direct", native)
    if not hasattr(time, "ticks_ms"):
        monkeypatch.setattr(time, "ticks_ms", lambda: 1000, raising=False)
    if not hasattr(time, "ticks_diff"):
        monkeypatch.setattr(time, "ticks_diff", lambda current, previous: current - previous, raising=False)
    sys.modules.pop("direct_api", None)
    from direct_api import DirectApi

    wlan = MagicMock()
    wlan.config.side_effect = lambda key: {
        "mac": bytes.fromhex("7c2c6742bbac"),
        "channel": 6,
        "ssid": "lab",
        "bssid": bytes.fromhex("001122334455"),
    }[key]
    wlan.status.return_value = -50
    wlan.active.return_value = True
    wlan.isconnected.return_value = True
    detector = MagicMock()
    detector.get_threshold.return_value = 0.25
    detector.is_ready.return_value = True
    policy = SimpleNamespace(motion_on_hits=4, motion_off_hits=3)
    traffic = MagicMock()
    traffic.is_running.return_value = True
    state = SimpleNamespace(chip_type="C3", current_channel=6, calibration_mode=False)
    config = SimpleNamespace(
        WIFI_SSID="lab",
        CSI_TARGET_PPS=100,
        EVALUATION_INTERVAL_MS=250,
        PUBLISH_INTERVAL_MS=1000,
    )

    facade = DirectApi(config, wlan, detector, state, policy, traffic)
    facade.start()
    facade.publish(0.75, 1, 0.25, 2000)

    start_args = native.start.call_args.kwargs
    capabilities = json.loads(start_args["capabilities"])
    info = json.loads(start_args["info"])
    assert {entry["name"] for entry in capabilities["commands"]} == {
        "capabilities", "info", "status", "config"
    }
    assert start_args["hostname"].startswith("espectre-micro-")
    assert start_args["protocol_version"] == protocol.PROTOCOL_VERSION
    assert start_args["dns_sd_schema_version"] == protocol.DNS_SD_TXT_SCHEMA_VERSION
    assert info["csi_traffic_mode"] == "internal"
    assert info["traffic_mode"] == "ping"
    traffic.is_running.return_value = False
    assert facade._info()["csi_traffic_mode"] == "external"
    event_name, event_json = native.publish.call_args.args
    telemetry = json.loads(event_json)
    assert event_name == "telemetry"
    assert telemetry["frontend"] == "micro"
    assert telemetry["detector"] == "lightweight"
    assert telemetry["motion_state"] == "motion"

    facade.stop()
    native.stop.assert_called_once_with()
