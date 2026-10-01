# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Current HTTP-only collector and Direct discovery contracts."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest
import numpy as np

from espectre_cli.app import build_parser
from espectre_cli import device_discovery, host
from espectre_cli.device_discovery import DiscoveredDevice, ESPECTRE_DIRECT_PORT, ESPECTRE_SERVICE_TYPE
from tools.ha_traffic_generator_addon.espectre_traffic_generator import ExternalTrafficGenerator
from tools.lib import csi_io


def discovered_device(
    *,
    frontend: str = "native",
    port: int = ESPECTRE_DIRECT_PORT,
    device_id: int = 0x1234,
    capabilities: tuple[str, ...] = ("sensing", "motion", "csi"),
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
        endpoint=f"http://{authority}/espectre/v1",
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


def test_collect_rejects_unsafe_label_before_resolving_device(monkeypatch) -> None:
    target_resolution_attempted = False

    def resolve_target(_args):
        nonlocal target_resolution_attempted
        target_resolution_attempted = True

    monkeypatch.setattr(host, "_resolve_collect_target_via_discovery", resolve_target)
    args = collect_args(
        label="../outside",
        info=False,
        ready_stable_seconds=3.0,
    )

    with pytest.raises(SystemExit):
        host.collect_csi_data(args)

    assert target_resolution_attempted is False


def test_collect_allows_live_inspection_without_label(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        host,
        "_resolve_collect_target_via_discovery",
        lambda _args: calls.append("resolve"),
    )
    monkeypatch.setattr(host, "_run_live_collect", lambda _args: calls.append("run"))
    args = collect_args(
        label=None,
        info=False,
        ready_stable_seconds=3.0,
    )

    host.collect_csi_data(args)

    assert calls == ["resolve", "run"]


@pytest.mark.parametrize("label", [None, "capture"])
def test_collect_duration_expires_after_packets_stop(monkeypatch, tmp_path, label):
    clock = [0.0]
    saved_packets = []
    monkeypatch.setattr(host.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(host.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(host, "_run_post_collect_quality_checks", lambda _paths: True)
    # Exercise the recording deadline independently of startup calibration.
    monkeypatch.setattr("detector_interface.detector_needs_startup_calibration", lambda _kind: False)
    packet = csi_io.CSIPacket(
        timestamp=20.0,
        seq_num=0,
        num_subcarriers=64,
        iq_raw=np.ones(128, dtype=np.int8),
        device_id=1,
        device_ticks_us=10000,
        source_ip="192.0.2.1",
    )

    class Receiver:
        effective_socket_rcvbuf_bytes = None
        calls = 0
        stopped = False

        def add_callback(self, callback):
            self.callback = callback

        def run(self, **_kwargs):
            self.calls += 1
            assert self.calls <= 2, "collection exceeded its deadline without new packets"
            if self.calls == 1:
                # Waiting for the first packet does not consume recording time.
                clock[0] = 20.0
                self.callback(packet)
            else:
                clock[0] += 1.0

        def stop(self):
            self.stopped = True

    class Writer:
        def __init__(self, **_kwargs):
            pass

        def save_samples_by_device(self, packets):
            saved_packets.extend(packets)
            return [tmp_path / "capture.npz"]

    receiver = Receiver()
    generator_stops = []
    generator = SimpleNamespace(stop=lambda: generator_stops.append(True))
    monkeypatch.setattr(host, "_prepare_raw_http_collection", lambda *_args: (receiver, generator, 5555))
    monkeypatch.setattr(host, "_start_raw_http_collection", lambda *_args: None)
    monkeypatch.setattr(csi_io, "CSICollector", Writer)
    args = build_parser().parse_args([
        "collect", "--target", "192.0.2.1", "--duration", "1",
        "--ready-stable-seconds", "0",
    ])
    args.label = label
    args.direct_endpoint = "http://192.0.2.1:8080/espectre/v1"
    args.traffic_target = "192.0.2.1"

    host._run_live_collect(args)

    assert receiver.calls == 2
    assert receiver.stopped
    assert generator_stops
    assert saved_packets == ([packet] if label else [])


@pytest.mark.parametrize("motion", [False, True])
def test_collect_calibration_applies_quiet_evidence_and_rejects_motion(monkeypatch, motion):
    from tools.lib.lightweight_detector import LightweightDetector

    clock = [20.0]
    monkeypatch.setattr(host.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(host.signal, "signal", lambda *_args: None)
    if motion:
        # Every ready evaluation crosses the motion reference.
        monkeypatch.setattr("threshold.get_detector_calibration_motion_ceiling", lambda _detector: -1.0)
    applied_evidence = []
    abandoned = []
    set_adaptive_threshold = LightweightDetector.set_adaptive_threshold
    abandon = LightweightDetector.on_startup_calibration_abandoned

    def spy_set_adaptive_threshold(self, shared_threshold):
        applied_evidence.append(len(self._startup_logits))
        set_adaptive_threshold(self, shared_threshold)

    def spy_abandon(self):
        abandoned.append(True)
        abandon(self)

    monkeypatch.setattr(LightweightDetector, "set_adaptive_threshold", spy_set_adaptive_threshold)
    monkeypatch.setattr(LightweightDetector, "on_startup_calibration_abandoned", spy_abandon)

    class Receiver:
        effective_socket_rcvbuf_bytes = None
        calls = 0

        def add_callback(self, callback):
            self.callback = callback

        def run(self, **_kwargs):
            self.calls += 1
            if self.calls > 1:
                clock[0] += 1000.0
                return
            # Enough packets for a quiet calibration and for a full motion budget.
            for seq in range(4500):
                self.callback(csi_io.CSIPacket(
                    timestamp=20.0,
                    seq_num=seq,
                    num_subcarriers=64,
                    iq_raw=np.ones(128, dtype=np.int8),
                    device_id=1,
                    device_ticks_us=10000 * (seq + 1),
                    source_ip="192.0.2.1",
                ))

        def stop(self):
            pass

    generator = SimpleNamespace(stop=lambda: None)
    monkeypatch.setattr(host, "_prepare_raw_http_collection", lambda *_args: (Receiver(), generator, 5555))
    monkeypatch.setattr(host, "_start_raw_http_collection", lambda *_args: None)
    args = build_parser().parse_args(["collect", "--target", "192.0.2.1", "--duration", "60"])
    args.label = None
    args.direct_endpoint = "http://192.0.2.1:8080/espectre/v1"
    args.traffic_target = "192.0.2.1"

    host._run_live_collect(args)

    if motion:
        assert applied_evidence == []
        assert abandoned
    else:
        # The threshold reads the evidence the calibration collected.
        assert len(applied_evidence) == 1 and applied_evidence[0] > 0
        assert not abandoned


def test_discovery_frontends_exclude_streamer() -> None:
    assert device_discovery.SUPPORTED_DISCOVERY_FRONTENDS == ("native", "esphome", "matter", "micro")


def discovery_info(**overrides):
    properties = {
        b"device_id": b"0x1234",
        b"frontend": b"native",
        b"transport": b"http",
        b"path": b"/espectre/v1",
        b"txtvers": device_discovery.DNS_SD_TXT_SCHEMA_VERSION.encode(),
        b"protovers": device_discovery.PROTOCOL_VERSION.encode(),
        b"capabilities": b"sensing, motion, ,csi",
    }
    properties.update(overrides.pop("properties", {}))
    return SimpleNamespace(
        properties=properties,
        port=overrides.pop("port", ESPECTRE_DIRECT_PORT),
        parsed_addresses=lambda _version: overrides.get("addresses", ["192.168.1.23"]),
    )


@pytest.mark.parametrize("frontend", device_discovery.SUPPORTED_DISCOVERY_FRONTENDS)
@pytest.mark.parametrize("port", [80, ESPECTRE_DIRECT_PORT])
def test_discovery_parses_canonical_records(frontend, port):
    info = discovery_info(port=port, properties={b"frontend": frontend})
    record = device_discovery._parse_record(ESPECTRE_SERVICE_TYPE, "sensor._espectre._tcp.local.", info)

    assert record.frontend == frontend
    assert record.device_id == 0x1234
    assert record.display_id == "0000000000001234"
    assert record.target_port == port
    assert record.name == "sensor"
    assert record.chip == "unknown"
    authority = "192.168.1.23" if port == 80 else f"192.168.1.23:{port}"
    assert record.endpoint == f"http://{authority}/espectre/v1"
    assert record.events_endpoint == f"http://{authority}/espectre/v1/events"
    serialized = record.as_serializable_dict()
    assert serialized["device_id"] == record.display_id
    assert serialized["capabilities"] == ["sensing", "motion", "csi"]
    assert serialized["metadata"] == {}


@pytest.mark.parametrize("properties", [
    {b"device_id": None}, {b"device_id": b"invalid"}, {b"device_id": b"0"},
    {b"device_id": b"10000000000000000"}, {b"frontend": b"unknown"},
    {b"transport": b"mqtt"}, {b"path": b"/other"},
    {b"txtvers": b"unsupported"}, {b"protovers": b"unsupported"},
])
def test_discovery_rejects_incompatible_record_metadata(properties):
    assert device_discovery._parse_record(
        ESPECTRE_SERVICE_TYPE, "sensor._espectre._tcp.local.", discovery_info(properties=properties)
    ) is None


@pytest.mark.parametrize("service_type,overrides", [
    ("_http._tcp.local.", {}), (ESPECTRE_SERVICE_TYPE, {"addresses": []}),
    (ESPECTRE_SERVICE_TYPE, {"port": 0}), (ESPECTRE_SERVICE_TYPE, {"port": 65536}),
])
def test_discovery_rejects_unusable_service_endpoints(service_type, overrides):
    assert device_discovery._parse_record(service_type, "sensor", discovery_info(**overrides)) is None


@pytest.mark.parametrize("firmware", [b"3.0.0-rc1", b"3.0.0-rc2"])
@pytest.mark.parametrize("key_case", ["lower", "upper", "mixed"])
def test_discovery_accepts_case_insensitive_txt_and_release_announcements(firmware, key_case):
    info = discovery_info(properties={b"name": b"Office Sensor", b"firmware": firmware})
    if key_case == "upper":
        info.properties = {key.upper(): value for key, value in info.properties.items()}
    elif key_case == "mixed":
        info.properties = {key[:1].upper() + key[1:]: value for key, value in info.properties.items()}
    info.properties[b"UNKNOWN"] = b"ignored"
    service_type = "_ESpectre._TCP.LOCAL."
    service_name = "Office Sensor." + service_type
    record = device_discovery._parse_record(service_type, service_name, info)
    assert record is not None
    assert record.name == "Office Sensor"
    assert record.service_name == service_name
    assert record.service_type == service_type
    assert record.firmware == firmware.decode()
    assert record.endpoint == f"http://192.168.1.23:{ESPECTRE_DIRECT_PORT}/espectre/v1"


@pytest.mark.parametrize("first", [None, b"", b"native", b"invalid"])
def test_discovery_first_txt_key_wins_even_without_value(first):
    info = discovery_info()
    info.properties = {b"FrOnTeNd": first, **info.properties}
    record = device_discovery._parse_record(ESPECTRE_SERVICE_TYPE, "Sensor", info)
    assert (record is not None) == (first == b"native")
    assert device_discovery._decode_txt({b"NAME": None, b"name": b"Second"}, "name") is None


def test_discovery_compares_dns_names_with_ascii_case_only():
    info = discovery_info(properties={b"name": b"Office Sensor"})
    zeroconf = SimpleNamespace(get_service_info=lambda *_args, **_kwargs: info)
    listener = device_discovery._DeviceListener(zeroconf)
    listener.add_service(zeroconf, ESPECTRE_SERVICE_TYPE, "Office._espectre._tcp.local.")
    listener.update_service(zeroconf, ESPECTRE_SERVICE_TYPE.upper(), "OFFICE._ESPECTRE._TCP.LOCAL.")
    assert len(listener.snapshot()) == 1
    assert listener.snapshot()[0].name == "Office Sensor"
    listener.remove_service(zeroconf, ESPECTRE_SERVICE_TYPE, "office._espectre._tcp.local.")
    assert listener.snapshot() == []
    assert device_discovery._dns_key("ÄA") == "Äa"


def test_discovery_listener_updates_filters_and_removes_records():
    records = {"native": discovery_info(), "micro": discovery_info(properties={b"frontend": b"micro"})}
    zeroconf = SimpleNamespace(get_service_info=lambda _type, name, **_kwargs: records.get(name))
    listener = device_discovery._DeviceListener(zeroconf)
    for name in ["native", "micro", "missing", "native"]:
        listener.add_service(zeroconf, ESPECTRE_SERVICE_TYPE, name)
    assert [record.frontend for record in listener.snapshot()] == ["micro", "native"]
    assert len(listener.snapshot("native")) == 1
    records["native"] = discovery_info(properties={b"name": b"Renamed sensor", b"chip": b"c3"})
    listener.update_service(zeroconf, ESPECTRE_SERVICE_TYPE, "native")
    assert listener.snapshot("native")[0].name == "Renamed sensor"
    for name in ["native", "native"]:
        listener.remove_service(zeroconf, ESPECTRE_SERVICE_TYPE, name)
    assert listener.snapshot("native") == []
    assert len(listener.snapshot()) == 1


@pytest.mark.parametrize("has_record", [False, True])
def test_discovery_wait_uses_deadline_or_quiet_window(monkeypatch, has_record):
    clock = [0.0]
    waits = []
    monkeypatch.setattr(device_discovery.time, "monotonic", lambda: clock[0])
    zeroconf = SimpleNamespace(get_service_info=lambda *_args, **_kwargs: discovery_info())
    listener = device_discovery._DeviceListener(zeroconf)
    if has_record:
        listener.add_service(zeroconf, ESPECTRE_SERVICE_TYPE, "sensor")

    def wait(duration):
        waits.append(duration)
        clock[0] += duration

    monkeypatch.setattr(listener._records_changed, "wait", wait)
    listener.wait_for_quiet(2.0, 0.25)
    assert waits == [0.25 if has_record else 2.0]


@pytest.mark.parametrize("late_arrival_s", [0.9, 1.6, 5.4])
def test_discovery_default_browse_includes_delayed_devices(monkeypatch, late_arrival_s):
    clock = [0.0]
    monkeypatch.setattr(device_discovery.time, "monotonic", lambda: clock[0])
    zeroconf = SimpleNamespace(get_service_info=lambda *_args, **_kwargs: discovery_info())
    listener = device_discovery._DeviceListener(zeroconf)
    listener.add_service(zeroconf, ESPECTRE_SERVICE_TYPE, "first")
    pending = [late_arrival_s]

    def wait(duration):
        if pending and clock[0] + duration >= pending[0]:
            clock[0] = pending.pop()
            listener.add_service(zeroconf, ESPECTRE_SERVICE_TYPE, "later")
        else:
            clock[0] += duration

    monkeypatch.setattr(listener._records_changed, "wait", wait)
    listener.wait_for_quiet(device_discovery.DISCOVERY_TIMEOUT_S,
                            device_discovery.DISCOVERY_QUIET_WINDOW_S)
    assert len(listener.snapshot()) == 2
    assert clock[0] == device_discovery.DISCOVERY_TIMEOUT_S


@pytest.mark.parametrize("browser_fails", [False, True])
def test_discovery_closes_resources_after_browse(monkeypatch, browser_fails):
    events = []

    def resolve(_type, _name, *, timeout, question_type):
        assert timeout == 1000
        assert question_type is device_discovery.DNSQuestionType.QM
        return discovery_info()

    zeroconf = SimpleNamespace(close=lambda: events.append("close"), get_service_info=resolve)

    def open_discovery(*, unicast, ip_version):
        assert unicast is True
        assert ip_version is device_discovery.IPVersion.V4Only
        return zeroconf

    monkeypatch.setattr(device_discovery, "Zeroconf", open_discovery)
    monkeypatch.setattr(device_discovery._DeviceListener, "wait_for_quiet", lambda *_args: None)

    def browser(_zeroconf, service_type, listener, *, question_type):
        assert service_type == ESPECTRE_SERVICE_TYPE
        assert question_type is device_discovery.DNSQuestionType.QM
        if browser_fails:
            raise OSError("browse failed")
        listener.add_service(zeroconf, service_type, "sensor")
        return SimpleNamespace(cancel=lambda: events.append("cancel"))

    monkeypatch.setattr(device_discovery, "ServiceBrowser", browser)
    if browser_fails:
        with pytest.raises(OSError):
            device_discovery.discover_devices()
        assert events == ["close"]
    else:
        result = device_discovery.discover_devices(frontend="native")
        assert len(result) == 1
        assert result[0].device_id == 0x1234
        assert events == ["cancel", "close"]


@pytest.mark.parametrize("options", [
    {"frontend": "unknown"}, {"timeout_s": 0}, {"timeout_s": float("nan")},
    {"quiet_window_s": -1}, {"quiet_window_s": float("inf")},
])
def test_discovery_validates_options_before_opening_network(monkeypatch, options):
    opened = []
    monkeypatch.setattr(device_discovery, "Zeroconf", lambda **_kwargs: opened.append(True))
    with pytest.raises(ValueError):
        device_discovery.discover_devices(**options)
    assert opened == []


def test_discovery_reports_unavailable_dependency_or_interfaces(monkeypatch):
    def unavailable(**_kwargs):
        raise OSError("no multicast interface")

    monkeypatch.setattr(device_discovery, "Zeroconf", unavailable)
    with pytest.raises(device_discovery.DeviceDiscoveryError) as error:
        device_discovery.discover_devices()
    assert isinstance(error.value.__cause__, OSError)
    monkeypatch.setattr(device_discovery, "Zeroconf", None)
    with pytest.raises(device_discovery.DeviceDiscoveryError):
        device_discovery.discover_devices()


def test_discovery_requires_explicit_selection_for_ambiguous_devices(monkeypatch):
    records = [discovered_device(device_id=1), discovered_device(device_id=2)]
    with pytest.raises(device_discovery.DeviceDiscoveryError):
        device_discovery.select_discovered_device(records, interactive=False)
    with pytest.raises(device_discovery.DeviceDiscoveryError):
        device_discovery.select_discovered_device(records, chip="s3")
    choices = iter(["", "invalid", "0", "3", "2"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(choices))
    assert device_discovery.select_discovered_device(records, frontend_label="native") == records[1]


def test_collect_explicit_esphome_target_uses_shared_direct_port(monkeypatch) -> None:
    monkeypatch.setattr(host.socket, "gethostbyname", lambda _host: "192.168.1.23")
    monkeypatch.setattr(host, "discover_devices", lambda **_kwargs: [])
    args = collect_args(target="espectre.local", frontend="esphome")

    host._resolve_collect_target_via_discovery(args)

    assert args.direct_endpoint == f"http://espectre.local:{ESPECTRE_DIRECT_PORT}/espectre/v1"
    assert args.traffic_target == "192.168.1.23"
    assert args.expected_discovery_device_id is None


def test_collect_explicit_endpoint_preserves_nondefault_direct_port(monkeypatch) -> None:
    monkeypatch.setattr(host.socket, "gethostbyname", lambda _host: "192.168.1.23")
    args = collect_args(target="http://espectre.local:61443/espectre/v1")

    host._resolve_collect_target_via_discovery(args)

    assert args.direct_endpoint == "http://espectre.local:61443/espectre/v1"
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
    no_raw = discovered_device(frontend="native", device_id=0x5678, capabilities=("sensing", "motion"))
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

        def request(self, verb, resource, data=None):
            calls.append((verb, resource, data))
            if resource == "capabilities":
                return {
                    "csi": {
                        "protocol_version": 1,
                        "traffic_udp_port": 6123,
                        "marker": ExternalTrafficGenerator.TRAFFIC_MARKER,
                    }
                }
            if resource == "sensing":
                return {
                    "traffic_generator_mode": "external",
                    "csi_traffic_udp_port": 6123,
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
        direct_endpoint="http://192.168.1.23/espectre/v1",
        traffic_target="192.168.1.23",
        source_ip="192.168.1.8",
        pps=400,
    )

    receiver, generator, port = host._prepare_raw_http_collection(
        args, FakeControl, FakeReceiver, FakeGenerator)

    assert calls == [
        ("open", args.direct_endpoint),
        ("get", "capabilities", None),
        ("patch", "sensing", {"traffic_generator_mode": "external"}),
        ("get", "sensing", None),
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

        def request(self, verb, resource, _data=None):
            assert verb in {"get", "patch"}
            if resource == "capabilities":
                return {
                    "csi": {
                        "protocol_version": 1,
                        "marker": ExternalTrafficGenerator.TRAFFIC_MARKER,
                    }
                }
            if resource == "sensing" and verb == "get":
                return {"traffic_generator_mode": "ping"}
            return {}

    args = SimpleNamespace(
        direct_endpoint="http://192.168.1.23/espectre/v1",
        traffic_target="192.168.1.23",
        source_ip=None,
        pps=100,
    )
    with pytest.raises(RuntimeError, match="did not persist"):
        host._prepare_raw_http_collection(args, FakeControl, object, ExternalTrafficGenerator)


def test_prepare_raw_collection_rejects_incompatible_protocol_version() -> None:
    class FakeControl:
        def __init__(self, _endpoint):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def request(self, verb, resource, _data=None):
            assert (verb, resource) == ("get", "capabilities")
            return {"csi": {"protocol_version": 2}}

    args = SimpleNamespace(
        direct_endpoint="http://192.168.1.23/espectre/v1",
        traffic_target="192.168.1.23",
        source_ip=None,
        pps=100,
    )
    with pytest.raises(RuntimeError, match="raw HTTP v1"):
        host._prepare_raw_http_collection(args, FakeControl, object, object)


@pytest.mark.parametrize("raw_capability", [
    {"protocol_version": 1, "marker": "."},
    {"protocol_version": 1, "traffic_marker": "👻"},
])
def test_prepare_raw_collection_rejects_noncanonical_marker(raw_capability) -> None:
    class FakeControl:
        def __init__(self, _endpoint):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def request(self, verb, resource, _data=None):
            assert (verb, resource) == ("get", "capabilities")
            return {"csi": raw_capability}

    args = SimpleNamespace(
        direct_endpoint="http://192.168.1.23/espectre/v1",
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


@pytest.mark.parametrize("target", ["https://espectre.local/espectre/v1", "http://"])
def test_collect_rejects_unusable_direct_targets(target, capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        host._resolve_collect_target_via_discovery(collect_args(target=target))

    assert exit_info.value.code == 1
    assert "Invalid Direct target" in capsys.readouterr().out


def test_collect_reports_an_unresolvable_hostname(monkeypatch, capsys) -> None:
    def fail(_host):
        raise OSError("no such host")

    monkeypatch.setattr(host.socket, "gethostbyname", fail)

    with pytest.raises(SystemExit) as exit_info:
        host._resolve_collect_target_via_discovery(collect_args(target="missing.local"))

    assert exit_info.value.code == 1
    assert "Cannot resolve Direct target missing.local" in capsys.readouterr().out


def test_collect_falls_back_to_the_default_port_when_discovery_is_unavailable(monkeypatch) -> None:
    def unavailable(**_kwargs):
        raise device_discovery.DeviceDiscoveryError("mDNS unavailable")

    monkeypatch.setattr(host.socket, "gethostbyname", lambda _host: "192.168.1.23")
    monkeypatch.setattr(host, "discover_devices", unavailable)
    args = collect_args(target="espectre.local")

    host._resolve_collect_target_via_discovery(args)

    assert args.direct_endpoint == f"http://espectre.local:{ESPECTRE_DIRECT_PORT}/espectre/v1"
    assert args.target_frontend == "unknown"


def test_collect_ignores_an_ambiguous_discovery_match_for_a_hostname(monkeypatch) -> None:
    first = discovered_device(device_id=0x1)
    second = discovered_device(device_id=0x2)
    monkeypatch.setattr(host.socket, "gethostbyname", lambda _host: first.ip_address)
    monkeypatch.setattr(host, "discover_devices", lambda **_kwargs: [first, second])
    args = collect_args(target="espectre.local")

    host._resolve_collect_target_via_discovery(args)

    assert args.expected_discovery_device_id is None


def test_collect_exits_when_discovery_itself_fails(monkeypatch, capsys) -> None:
    def unavailable(**_kwargs):
        raise device_discovery.DeviceDiscoveryError("mDNS unavailable")

    monkeypatch.setattr(host, "discover_devices", unavailable)

    with pytest.raises(SystemExit) as exit_info:
        host._resolve_collect_target_via_discovery(collect_args(target=None))

    assert exit_info.value.code == 1
    assert "mDNS unavailable" in capsys.readouterr().out


@pytest.mark.parametrize("frontend, label", [(None, "raw-capable Direct"), ("matter", "Matter")])
def test_collect_exits_when_no_raw_capable_device_is_discovered(monkeypatch, capsys, frontend, label) -> None:
    monkeypatch.setattr(host, "discover_devices", lambda **_kwargs: [])

    with pytest.raises(SystemExit) as exit_info:
        host._resolve_collect_target_via_discovery(collect_args(target=None, frontend=frontend))

    assert exit_info.value.code == 1
    assert f"No {label} devices discovered" in capsys.readouterr().out


def test_collect_lets_the_operator_choose_among_several_devices(monkeypatch) -> None:
    first = discovered_device(device_id=0x1)
    second = discovered_device(device_id=0x2)
    monkeypatch.setattr(host, "discover_devices", lambda **_kwargs: [first, second])
    monkeypatch.setattr(host, "choose_device_interactively", lambda records, **_kwargs: records[1])
    args = collect_args(target=None)

    host._resolve_collect_target_via_discovery(args)

    assert args.expected_discovery_device_id == second.device_id
    assert args.target == second.ip_address


def test_collect_exits_when_device_selection_is_cancelled(monkeypatch) -> None:
    def cancel(_records, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(
        host, "discover_devices",
        lambda **_kwargs: [discovered_device(device_id=0x1), discovered_device(device_id=0x2)],
    )
    monkeypatch.setattr(host, "choose_device_interactively", cancel)

    with pytest.raises(SystemExit) as exit_info:
        host._resolve_collect_target_via_discovery(collect_args(target=None))

    assert exit_info.value.code == 1


@pytest.mark.parametrize("device_id, expected", [(None, "unknown"), (0x1234, "0000000000001234")])
def test_expected_device_id_is_rendered_as_sixteen_hex_digits(device_id, expected) -> None:
    assert host._format_expected_device_id(device_id) == expected


def test_collection_countdown_is_skipped_without_a_delay(monkeypatch) -> None:
    monkeypatch.setattr(host.time, "sleep", lambda _seconds: pytest.fail("must not sleep"))

    host._wait_before_collection(0)


def test_collection_countdown_sleeps_in_steps_of_at_most_one_second(monkeypatch) -> None:
    sleeps = []
    monkeypatch.setattr(host.time, "sleep", sleeps.append)

    host._wait_before_collection(2.5)

    assert sleeps == [1.0, 1.0, 0.5]


@pytest.mark.parametrize("bad_pps", [0, -5])
def test_prepare_raw_collection_rejects_a_non_positive_rate(bad_pps) -> None:
    args = SimpleNamespace(
        direct_endpoint="http://192.168.1.23/espectre/v1",
        traffic_target="192.168.1.23",
        source_ip=None,
        pps=bad_pps,
    )

    with pytest.raises(ValueError, match="must be > 0 pps"):
        host._prepare_raw_http_collection(args, object, object, ExternalTrafficGenerator)


def test_post_collect_quality_orders_issues_by_priority() -> None:
    names = ["stream_seq_gaps", "other", "temporal_occupancy", "inter_packet_gap", "stream_seq_max_gap"]

    ordered = sorted(
        (SimpleNamespace(name=name) for name in names),
        key=host._post_collect_quality_issue_sort_key,
    )

    assert [result.name for result in ordered] == [
        "temporal_occupancy", "stream_seq_max_gap", "inter_packet_gap", "stream_seq_gaps", "other",
    ]


def _quality_result(name, status, message=""):
    return SimpleNamespace(name=name, status=status, message=message)


def test_post_collect_quality_fails_only_when_a_check_fails(monkeypatch, tmp_path, capsys) -> None:
    outcomes = {
        "clean.csv": [_quality_result("a", "PASS")],
        "warned.csv": [_quality_result("temporal_occupancy", "WARN", "low"), _quality_result("b", "PASS")],
        "failed.csv": [_quality_result("inter_packet_gap", "FAIL", "gap")],
    }

    def validate(path, **_kwargs):
        return outcomes[path.name]

    monkeypatch.setattr("tools.lib.dataset_quality.capture.validate_capture_file", validate)

    assert host._run_post_collect_quality_checks([tmp_path / "clean.csv", tmp_path / "warned.csv"]) is True
    assert host._run_post_collect_quality_checks([tmp_path / "failed.csv"]) is False
    output = capsys.readouterr().out
    assert "clean.csv: quality checks all pass" in output
    assert "1 warn, 0 fail" in output
    assert "inter_packet_gap: gap" in output


def test_post_collect_quality_skips_files_it_cannot_validate(monkeypatch, tmp_path, capsys) -> None:
    def broken(_path, **_kwargs):
        raise ValueError("unreadable")

    monkeypatch.setattr("tools.lib.dataset_quality.capture.validate_capture_file", broken)

    assert host._run_post_collect_quality_checks([tmp_path / "broken.csv"]) is True
    assert "quality checks skipped (unreadable)" in capsys.readouterr().out


def test_post_collect_quality_is_available_in_a_source_checkout(capsys) -> None:
    assert host._run_post_collect_quality_checks([]) is True

    assert "unavailable" not in capsys.readouterr().out


def test_dataset_stats_explain_how_to_collect_when_empty(capsys) -> None:
    host._print_dataset_catalog_stats({"environments": [], "chips": []})

    assert "No samples collected yet." in capsys.readouterr().out


def test_dataset_stats_total_every_environment(capsys) -> None:
    stats = {
        "chips": ["c3", "s3"],
        "total_samples": 7,
        "environments": [{
            "environment": "lab",
            "total_samples": 7,
            "rows": [
                {"label": "baseline", "counts": {"c3": 2, "s3": 1}, "total": 3},
                {"label": "wave", "counts": {"c3": 1, "s3": 3}, "total": 4},
            ],
        }],
    }

    host._print_dataset_catalog_stats(stats)

    output = capsys.readouterr().out
    assert "lab" in output
    rows = {line.split()[0]: line.split()[1:] for line in output.splitlines() if line.strip()[:1].isalpha()}
    assert rows["baseline"] == ["2", "1", "3"]
    assert rows["wave"] == ["1", "3", "4"]
    assert "Grand total:" in output


def test_dataset_info_prints_the_catalog_statistics(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "tools.lib.dataset_metadata.get_dataset_catalog_stats",
        lambda: {"environments": [], "chips": []},
    )

    host._show_dataset_info()

    assert "No samples collected yet." in capsys.readouterr().out


def test_collect_rejects_a_negative_ready_gate_before_any_work(capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        host.collect_csi_data(collect_args(ready_stable_seconds=-1))

    assert exit_info.value.code == 1
    assert "Ready gate seconds must be >= 0" in capsys.readouterr().out
