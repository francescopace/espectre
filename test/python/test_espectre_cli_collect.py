# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Current HTTP-only collector and Direct discovery contracts."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from espectre_cli.app import build_parser
from espectre_cli import device_discovery, host
from espectre_cli.device_discovery import DiscoveredDevice, ESPECTRE_DIRECT_PORT, ESPECTRE_SERVICE_TYPE
from tools.espectre_traffic_generator import ExternalTrafficGenerator


def discovered_device(
    *,
    frontend: str = "native",
    port: int = ESPECTRE_DIRECT_PORT,
    device_id: int = 0x1234,
    capabilities: tuple[str, ...] = ("config", "monitor", "raw_csi"),
) -> DiscoveredDevice:
    authority = "192.168.1.23" if port == 80 else f"192.168.1.23:{port}"
    return DiscoveredDevice(
        service_name=f"ESPectre {frontend}._espectre._tcp.local.",
        service_type=ESPECTRE_SERVICE_TYPE,
        frontend=frontend,
        device_id=device_id,
        device_id_text=f"{device_id:016x}",
        name=f"ESPectre {frontend}",
        chip="esp32c3",
        ip_address="192.168.1.23",
        port=port,
        transport="http",
        endpoint=f"http://{authority}/espectre/v1/request",
        protocol="1",
        events_endpoint=f"http://{authority}/espectre/v1/events",
        capabilities=capabilities,
    )


def collect_args(**overrides) -> argparse.Namespace:
    values = {
        "target": "192.168.1.23",
        "frontend": None,
        "source_ip": None,
        "pps": 100,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_collect_parser_exposes_only_http_collection_options() -> None:
    args = build_parser().parse_args(
        [
            "collect",
            "--target",
            f"espectre.local:{ESPECTRE_DIRECT_PORT}",
            "--frontend",
            "esphome",
            "--source-ip",
            "192.168.1.8",
            "--pps",
            "325",
            "--label",
            "benchmark",
        ]
    )

    assert args.target == f"espectre.local:{ESPECTRE_DIRECT_PORT}"
    assert args.frontend == "esphome"
    assert args.source_ip == "192.168.1.8"
    assert args.pps == 325
    assert args.label == "benchmark"
    for removed in ("transport", "udp_port", "target_port", "fixed", "list_devices"):
        assert not hasattr(args, removed)


@pytest.mark.parametrize(
    "removed_option",
    ["--transport", "--udp-port", "--target-port", "--fixed", "--list-devices"],
)
def test_collect_parser_rejects_removed_udp_and_streamer_options(removed_option: str) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["collect", removed_option])


def test_discovery_frontends_exclude_streamer() -> None:
    assert device_discovery.SUPPORTED_DISCOVERY_FRONTENDS == ("native", "esphome", "matter")


def test_collect_explicit_esphome_target_uses_shared_direct_port(monkeypatch) -> None:
    monkeypatch.setattr(host.socket, "gethostbyname", lambda _host: "192.168.1.23")
    monkeypatch.setattr(host, "discover_devices", lambda **_kwargs: [])
    args = collect_args(target="espectre.local", frontend="esphome")

    host._resolve_collect_target_via_discovery(args)

    assert args.direct_endpoint == f"http://espectre.local:{ESPECTRE_DIRECT_PORT}/espectre/v1/request"
    assert args.traffic_target == "192.168.1.23"
    assert args.expected_discovery_device_id is None


def test_collect_explicit_endpoint_preserves_nondefault_direct_port(monkeypatch) -> None:
    monkeypatch.setattr(host.socket, "gethostbyname", lambda _host: "192.168.1.23")
    args = collect_args(target="http://espectre.local:61443/espectre/v1/request")

    host._resolve_collect_target_via_discovery(args)

    assert args.direct_endpoint == "http://espectre.local:61443/espectre/v1/request"
    assert args.traffic_target == "192.168.1.23"


def test_collect_bare_hostname_uses_discovered_esphome_port(monkeypatch) -> None:
    selected = discovered_device(frontend="esphome", port=ESPECTRE_DIRECT_PORT)
    monkeypatch.setattr(host.socket, "gethostbyname", lambda _host: selected.ip_address)
    monkeypatch.setattr(host, "discover_devices", lambda **_kwargs: [selected])
    args = collect_args(target="espectre.local")

    host._resolve_collect_target_via_discovery(args)

    assert args.direct_endpoint == selected.endpoint
    assert args.target_frontend == "esphome"
    assert args.expected_discovery_device_id == selected.device_id


def test_collect_discovers_only_raw_capable_direct_devices(monkeypatch) -> None:
    raw = discovered_device(frontend="matter", device_id=0x1234)
    no_raw = discovered_device(frontend="native", device_id=0x5678, capabilities=("config", "monitor"))
    monkeypatch.setattr(host, "discover_devices", lambda **_kwargs: [no_raw, raw])
    args = collect_args(target=None)

    host._resolve_collect_target_via_discovery(args)

    assert args.direct_endpoint == raw.endpoint
    assert args.traffic_target == raw.ip_address
    assert args.target_frontend == "matter"
    assert args.expected_discovery_device_id == raw.device_id


def test_collect_resolves_full_device_id_through_direct_discovery(monkeypatch) -> None:
    selected = discovered_device(device_id=0x1122334455667788)
    other = discovered_device(device_id=0x8877665544332211)
    monkeypatch.setattr(host, "discover_devices", lambda **_kwargs: [other, selected])
    args = collect_args(target="1122334455667788")

    host._resolve_collect_target_via_discovery(args)

    assert args.direct_endpoint == selected.endpoint
    assert args.expected_discovery_device_id == selected.device_id


def test_prepare_raw_collection_persists_external_before_constructing_data_plane() -> None:
    calls: list[tuple[str, object]] = []

    class FakeControl:
        def __init__(self, endpoint):
            calls.append(("open", endpoint))

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            calls.append(("close", None))

        def request(self, method, params=None):
            calls.append((method, params))
            if method == "capabilities":
                return {
                    "raw_csi": {
                        "protocol_version": 2,
                        "traffic_udp_port": 6123,
                        "marker": ExternalTrafficGenerator.TRAFFIC_MARKER,
                    }
                }
            if method == "config":
                return {
                    "runtime": {
                        "csi_traffic_mode": "external",
                        "csi_traffic_udp_port": 6123,
                    }
                }
            return {}

    class FakeReceiver:
        def __init__(self, endpoint, **kwargs):
            self.endpoint = endpoint
            self.kwargs = kwargs

    class FakeGenerator:
        TRAFFIC_MARKER = ExternalTrafficGenerator.TRAFFIC_MARKER

        def __init__(self, targets, **kwargs):
            self.targets = targets
            self.kwargs = kwargs

    args = SimpleNamespace(
        direct_endpoint="http://192.168.1.23/espectre/v1/request",
        traffic_target="192.168.1.23",
        source_ip="192.168.1.8",
        pps=400,
    )

    receiver, generator, port = host._prepare_raw_http_collection(
        args, FakeControl, FakeReceiver, FakeGenerator)

    assert calls == [
        ("open", args.direct_endpoint),
        ("capabilities", None),
        ("set_csi_traffic_mode", {"csi_traffic_mode": "external"}),
        ("config", None),
        ("close", None),
    ]
    assert receiver.endpoint == args.direct_endpoint
    assert generator.targets == [args.traffic_target]
    assert generator.kwargs == {"port": 6123, "rate_pps": 400.0, "source_ip": "192.168.1.8"}
    assert port == 6123


def test_prepare_raw_collection_rejects_unconfirmed_persistent_mode() -> None:
    class FakeControl:
        def __init__(self, _endpoint):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def request(self, method, _params=None):
            if method == "capabilities":
                return {
                    "raw_csi": {
                        "protocol_version": 2,
                        "marker": ExternalTrafficGenerator.TRAFFIC_MARKER,
                    }
                }
            if method == "config":
                return {"runtime": {"csi_traffic_mode": "internal"}}
            return {}

    args = SimpleNamespace(
        direct_endpoint="http://192.168.1.23/espectre/v1/request",
        traffic_target="192.168.1.23",
        source_ip=None,
        pps=100,
    )
    with pytest.raises(RuntimeError, match="did not persist"):
        host._prepare_raw_http_collection(args, FakeControl, object, ExternalTrafficGenerator)


def test_prepare_raw_collection_rejects_raw_v1_without_fallback() -> None:
    class FakeControl:
        def __init__(self, _endpoint):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def request(self, method, _params=None):
            assert method == "capabilities"
            return {"raw_csi": {"protocol_version": 1}}

    args = SimpleNamespace(
        direct_endpoint="http://192.168.1.23/espectre/v1/request",
        traffic_target="192.168.1.23",
        source_ip=None,
        pps=100,
    )
    with pytest.raises(RuntimeError, match="raw HTTP v2"):
        host._prepare_raw_http_collection(args, FakeControl, object, object)


@pytest.mark.parametrize("raw_capability", [
    {"protocol_version": 2, "marker": "."},
    {"protocol_version": 2, "traffic_marker": "👻"},
])
def test_prepare_raw_collection_rejects_noncanonical_marker(raw_capability) -> None:
    class FakeControl:
        def __init__(self, _endpoint):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def request(self, method, _params=None):
            assert method == "capabilities"
            return {"raw_csi": raw_capability}

    args = SimpleNamespace(
        direct_endpoint="http://192.168.1.23/espectre/v1/request",
        traffic_target="192.168.1.23",
        source_ip=None,
        pps=100,
    )
    with pytest.raises(RuntimeError, match="canonical external traffic marker"):
        host._prepare_raw_http_collection(args, FakeControl, object, ExternalTrafficGenerator)


def test_start_raw_collection_opens_session_then_starts_traffic_before_http_bind() -> None:
    calls: list[str] = []

    class FakeReceiver:
        def start_session(self):
            calls.append("session")

        def bind_stream(self):
            calls.append("bind")

        def stop(self):
            calls.append("receiver_stop")

    class FakeGenerator:
        def start(self):
            calls.append("generator")

        def stop(self):
            calls.append("generator_stop")

    host._start_raw_http_collection(FakeReceiver(), FakeGenerator())

    assert calls == ["session", "generator", "bind"]


def test_start_raw_collection_stops_traffic_when_http_bind_fails() -> None:
    calls: list[str] = []

    class FailingReceiver:
        def start_session(self):
            calls.append("session")

        def bind_stream(self):
            calls.append("bind")
            raise TimeoutError("bind failed")

        def stop(self):
            calls.append("receiver_stop")

    class FakeGenerator:
        def start(self):
            calls.append("generator")

        def stop(self):
            calls.append("generator_stop")

    with pytest.raises(TimeoutError, match="bind failed"):
        host._start_raw_http_collection(FailingReceiver(), FakeGenerator())

    assert calls == ["session", "generator", "bind", "generator_stop", "receiver_stop"]
