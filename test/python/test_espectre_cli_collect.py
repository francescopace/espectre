"""
ESPectre - CLI Collect Tests

Tests for ESPectre host-side collect and UI CLI options.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import ModuleType
import builtins

import pytest
from espectre_cli.app import build_parser
from espectre_cli import host
from tools.lib.csi_io import AdaptivePacingController


def _make_collect_args(**overrides) -> argparse.Namespace:
    args = {
        "info": False,
        "label": "static_presence",
        "samples": 2,
        "duration": 10.0,
        "ready_stable_seconds": 3.0,
        "start_delay": 5.0,
        "udp_port": 5001,
        "bind_ip": None,
        "target": "192.168.1.15",
        "target_port": 9999,
        "pps": 100,
        "adaptive": True,
        "contributor": None,
        "description": None,
    }
    args.update(overrides)
    return argparse.Namespace(**args)


def _make_live_collect_args(**overrides) -> argparse.Namespace:
    args = {
        "info": False,
        "label": "test",
        "samples": 1,
        "duration": None,
        "ready_stable_seconds": 3.0,
        "start_delay": 0.0,
        "udp_port": 5001,
        "bind_ip": None,
        "target": "192.168.1.15",
        "target_port": 9999,
        "pps": 100,
        "adaptive": True,
        "detector": "classic",
        "contributor": None,
        "description": None,
    }
    args.update(overrides)
    return argparse.Namespace(**args)


def _install_live_collect_modules(monkeypatch, receiver_cls, pacing_cls, collector_cls=object, config_overrides=None) -> None:
    fake_csi_utils = ModuleType("tools.lib.csi_io")
    fake_config = ModuleType("config")
    fake_ml_detector = ModuleType("ml_detector")
    fake_classic_detector = ModuleType("classic_detector")
    fake_runtime_policy = ModuleType("runtime_policy")
    fake_threshold = ModuleType("threshold")

    fake_config.DEFAULT_SUBCARRIERS = [12, 14]
    fake_config.SEG_WINDOW_SIZE = 2
    fake_config.CALIBRATION_BUFFER_SIZE = 20
    fake_config.ENABLE_LOWPASS_FILTER = False
    fake_config.LOWPASS_CUTOFF = 11.0
    fake_config.ENABLE_HAMPEL_FILTER = True
    fake_config.HAMPEL_WINDOW = 7
    fake_config.HAMPEL_THRESHOLD = 5.0
    fake_config.PUBLISH_INTERVAL = 1
    fake_config.EVALUATION_INTERVAL = 1
    fake_config.MOTION_ON_HITS = 3
    fake_config.MOTION_OFF_HITS = 3
    if config_overrides:
        for key, value in config_overrides.items():
            setattr(fake_config, key, value)

    class FakeContext:
        last_turbulence = 0.0
        buffer_count = 0
        window_size = 2
        buffer_index = 0
        turbulence_buffer = []

    class FakeMLDetector:
        def __init__(self, **kwargs):
            self._context = FakeContext()
            self._threshold = kwargs.get("threshold", 0.5)

        def process_packet(self, csi_data, subcarriers):
            pass

        def update_state(self):
            return {"probability": 0.0, "threshold": self._threshold, "state": 0}

        def get_threshold(self):
            return self._threshold

        def set_threshold(self, threshold):
            self._threshold = threshold
            return True

        def reset(self):
            pass

        def set_cv_normalization(self, enabled):
            pass

        def is_ready(self):
            return False

        def _extract_features(self):
            return []

    class FakeClassicBaseDetector(FakeMLDetector):
        def update_state(self):
            return {"moving_variance": 0.0, "threshold": self._threshold, "state": 0}

        def get_motion_metric(self):
            return 0.0

        def set_adaptive_threshold(self, threshold):
            self._threshold = threshold

        def get_name(self):
            return "Classic"

    class FakeClassicDetector(FakeClassicBaseDetector):
        ALGORITHM = "classic"
        BASE_THRESHOLD = 1.0

        def get_name(self):
            return "Classic"

    class FakeRuntimeMotionPolicy:
        def __init__(self, **kwargs):
            pass

        def note_packet(self):
            pass

        def should_evaluate(self, should_publish):
            return True

        def apply_state(self, state):
            return state, None

        def after_evaluation(self):
            pass

        def reset(self):
            pass

    class FakeStartupThresholdCalibrator:
        def __init__(self, target_packets, auto_factor=1.3, gate_enabled=False):
            self.target_packets = int(target_packets)
            self.auto_factor = float(auto_factor)
            self.gate_enabled = bool(gate_enabled)
            self.packet_count = 0
            self.max_moving_variance = None

        def observe_detector(self, detector, packet_weight=1):
            self.packet_count += packet_weight
            if not detector.is_ready():
                return None
            current_mv = float(detector.get_motion_metric())
            if self.max_moving_variance is None or current_mv > self.max_moving_variance:
                self.max_moving_variance = current_mv
            return current_mv

        def is_complete(self):
            return self.packet_count >= self.target_packets

        def is_successful(self):
            return self.max_moving_variance is not None

        def calculate_threshold(self):
            return 1.5, "max x 1.3"

    fake_csi_utils.CSICollector = collector_cls
    fake_csi_utils.CSIReceiver = receiver_cls
    fake_csi_utils.UdpPacingSender = pacing_cls
    fake_csi_utils.AdaptivePacingController = AdaptivePacingController
    fake_csi_utils.get_default_bind_host = lambda: "127.0.0.1"
    fake_ml_detector.FEATURE_NAMES = ["f1", "f2"]
    fake_ml_detector.ML_DEFAULT_THRESHOLD = 0.5
    fake_ml_detector.ML_METRIC_SCALE = 1.0
    fake_ml_detector.MLDetector = FakeMLDetector
    fake_classic_detector.ClassicDetector = FakeClassicDetector
    fake_runtime_policy.RuntimeMotionPolicy = FakeRuntimeMotionPolicy
    fake_threshold.StartupThresholdCalibrator = FakeStartupThresholdCalibrator
    fake_threshold.get_detector_auto_factor = lambda detector: getattr(detector, "STARTUP_THRESHOLD_FACTOR", 1.3)
    fake_threshold.get_detector_startup_gate = lambda detector: bool(getattr(detector, "STARTUP_GATE", False))

    monkeypatch.setitem(sys.modules, "tools.lib.csi_io", fake_csi_utils)
    monkeypatch.setitem(sys.modules, "config", fake_config)
    monkeypatch.setitem(sys.modules, "ml_detector", fake_ml_detector)
    monkeypatch.setitem(sys.modules, "classic_detector", fake_classic_detector)
    monkeypatch.setitem(sys.modules, "runtime_policy", fake_runtime_policy)
    monkeypatch.setitem(sys.modules, "threshold", fake_threshold)


def test_collect_parser_accepts_count_alias() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "collect",
            "--label",
            "static_presence",
            "--duration",
            "10",
            "--count",
            "3",
            "--start-delay",
            "15",
            "--target",
            "239.1.1.15",
        ]
    )

    assert args.namespace == "collect"
    assert args.samples == 3
    assert args.start_delay == 15.0


def test_collect_parser_accepts_ready_gate_override() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "collect",
            "--label",
            "low_rssi",
            "--ready-stable-seconds",
            "0",
            "--target",
            "192.168.1.15",
        ]
    )

    assert args.namespace == "collect"
    assert args.ready_stable_seconds == 0.0


def test_collect_parser_keeps_samples_option() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "collect",
            "--label",
            "motion",
            "--samples",
            "4",
            "--target",
            "192.168.1.15",
        ]
    )

    assert args.samples == 4
    assert args.start_delay == 0.0


def test_collect_parser_accepts_comma_separated_targets() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "collect",
            "--label",
            "test",
            "--target",
            "192.168.1.17,192.168.1.24,192.168.1.29",
        ]
    )

    assert args.target == "192.168.1.17,192.168.1.24,192.168.1.29"


def test_collect_parser_accepts_target_short_flag() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "collect",
            "--label",
            "motion",
            "-t",
            "192.168.1.15",
        ]
    )

    assert args.target == "192.168.1.15"


def test_collect_parser_accepts_live_options() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "collect",
            "--target",
            "192.168.1.15",
            "--label",
            "empty",
            "--duration",
            "45",
            "--description",
            "live collect ML, idle-motion-idle",
        ]
    )

    assert args.namespace == "collect"
    assert args.detector == "classic"
    assert args.label == "empty"
    assert args.duration == 45.0
    assert args.pps == 100
    assert args.description == "live collect ML, idle-motion-idle"


def test_collect_timed_allows_test_label(monkeypatch) -> None:
    routed_labels: list[str] = []

    monkeypatch.setattr(
        host,
        "_collect_dataset_csi_data",
        lambda args: routed_labels.append(args.label),
    )

    host.collect_csi_data(_make_collect_args(target="192.168.1.15", label="test"))

    assert routed_labels == ["test"]


def test_collect_parser_accepts_pps() -> None:
    parser = build_parser()

    args = parser.parse_args(["collect", "--target", "192.168.1.15", "--pps", "42"])

    assert args.pps == 42
    assert args.adaptive is True


def test_collect_parser_defaults_to_adaptive() -> None:
    parser = build_parser()

    args = parser.parse_args(["collect", "--target", "192.168.1.15"])

    assert args.adaptive is True


def test_collect_parser_accepts_fixed() -> None:
    parser = build_parser()

    args = parser.parse_args(["collect", "--target", "192.168.1.15", "--fixed"])

    assert args.adaptive is False


def test_collect_parser_accepts_detector_choice() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "collect",
            "--target",
            "192.168.1.15",
            "--detector",
            "classic",
        ]
    )

    assert args.namespace == "collect"
    assert args.detector == "classic"
    assert args.label is None
    assert args.duration is None


def test_collect_parser_accepts_comma_separated_detectors() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "collect",
            "--target",
            "192.168.1.15",
            "--detector",
            "classic,ml",
        ]
    )

    assert args.namespace == "collect"
    assert args.detector == "classic,ml"


def test_collect_live_rejects_unknown_detector(monkeypatch, capsys) -> None:
    class FakeReceiver:
        def __init__(self, **kwargs):
            pass

    class FakePacingSender:
        def __init__(self, **kwargs):
            pass

    _install_live_collect_modules(monkeypatch, FakeReceiver, FakePacingSender)

    with pytest.raises(SystemExit):
        host.collect_csi_data(_make_live_collect_args(detector="classic,bogus", label=None))

    output = capsys.readouterr().out
    assert "Unsupported detector(s): bogus" in output


def test_ui_parser_accepts_ble_interface() -> None:
    parser = build_parser()

    args = parser.parse_args(["ui", "ble"])

    assert args.namespace == "ui"
    assert args.interface == "ble"


def test_ui_parser_accepts_theremin_interface() -> None:
    parser = build_parser()

    args = parser.parse_args(["ui", "theremin"])

    assert args.namespace == "ui"
    assert args.interface == "theremin"


def test_open_web_ui_opens_ble_file(monkeypatch, tmp_path) -> None:
    ble_file = tmp_path / "espectre-ble.html"
    ble_file.write_text("<html></html>", encoding="utf-8")
    served_routes: list[str] = []

    monkeypatch.setattr(
        host,
        "_WEB_UI_FILES",
        {
            "mqtt": tmp_path / "espectre-mqtt.html",
            "ble": ble_file,
            "theremin": tmp_path / "espectre-theremin.html",
        },
    )
    monkeypatch.setattr(host, "_serve_web_ui", lambda route: served_routes.append(route))

    host.open_web_ui("ble")

    assert served_routes == ["configure/"]


def test_open_web_ui_opens_theremin_file(monkeypatch, tmp_path) -> None:
    theremin_file = tmp_path / "espectre-theremin.html"
    theremin_file.write_text("<html></html>", encoding="utf-8")
    served_routes: list[str] = []

    monkeypatch.setattr(
        host,
        "_WEB_UI_FILES",
        {
            "mqtt": tmp_path / "espectre-mqtt.html",
            "ble": tmp_path / "espectre-ble.html",
            "theremin": theremin_file,
        },
    )
    monkeypatch.setattr(host, "_serve_web_ui", lambda route: served_routes.append(route))

    host.open_web_ui("theremin")

    assert served_routes == ["theremin/"]


def test_open_web_ui_reports_unknown_missing_and_browser_error(monkeypatch, tmp_path, capsys) -> None:
    missing = tmp_path / "missing.html"
    existing = tmp_path / "espectre-mqtt.html"
    existing.write_text("<html></html>", encoding="utf-8")

    monkeypatch.setattr(
        host,
        "_WEB_UI_FILES",
        {
            "mqtt": existing,
            "ble": missing,
            "theremin": tmp_path / "espectre-theremin.html",
        },
    )

    host.open_web_ui("unknown")
    host.open_web_ui("ble")

    def raise_browser(_route: str) -> None:
        raise RuntimeError("browser blocked")

    monkeypatch.setattr(host, "_serve_web_ui", raise_browser)
    host.open_web_ui("mqtt")

    output = capsys.readouterr().out
    assert "unknown web UI" in output
    assert "not found" in output
    assert "Error opening browser" in output


def test_serve_web_ui_opens_local_route_and_closes_server(monkeypatch) -> None:
    opened_urls: list[str] = []

    class FakeServer:
        server_address = ("127.0.0.1", 43210)

        def __init__(self) -> None:
            self.served = False
            self.closed = False

        def serve_forever(self) -> None:
            self.served = True

        def server_close(self) -> None:
            self.closed = True

    fake_server = FakeServer()
    monkeypatch.setattr(host, "ThreadingHTTPServer", lambda address, handler: fake_server)
    monkeypatch.setattr(host.webbrowser, "open", lambda url: opened_urls.append(url))

    host._serve_web_ui("monitor/")

    assert opened_urls == ["http://127.0.0.1:43210/monitor/"]
    assert fake_server.served is True
    assert fake_server.closed is True


def test_wait_before_collection_counts_down(monkeypatch, capsys) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(host.time, "sleep", lambda seconds: sleeps.append(seconds))

    host._wait_before_collection(2.5)

    output = capsys.readouterr().out
    assert "Starting collection in 2.5s" in output
    assert "2.5s remaining" in output
    assert "1.5s remaining" in output
    assert "0.5s remaining" in output
    assert "Starting now." in output
    assert sleeps == [1.0, 1.0, 0.5]


def test_collect_info_shows_empty_dataset(monkeypatch, capsys) -> None:
    fake_csi_utils = ModuleType("tools.lib.csi_io")
    fake_csi_utils.CSICollector = object
    fake_csi_utils.UdpPacingSender = object
    fake_csi_utils.get_default_bind_host = lambda: "127.0.0.1"
    fake_csi_utils.get_dataset_stats = lambda: {"labels": {}, "total_samples": 0}
    monkeypatch.setitem(sys.modules, "tools.lib.csi_io", fake_csi_utils)

    host.collect_csi_data(_make_collect_args(info=True))

    output = capsys.readouterr().out
    assert "Dataset Statistics" in output
    assert "No samples collected yet." in output


def test_collect_info_shows_label_table(monkeypatch, capsys) -> None:
    fake_csi_utils = ModuleType("tools.lib.csi_io")
    fake_csi_utils.CSICollector = object
    fake_csi_utils.UdpPacingSender = object
    fake_csi_utils.get_default_bind_host = lambda: "127.0.0.1"
    fake_csi_utils.get_dataset_stats = lambda: {
        "labels": {"motion": {"samples": 4}, "empty": {"samples": 2}},
        "total_samples": 6,
    }
    monkeypatch.setitem(sys.modules, "tools.lib.csi_io", fake_csi_utils)

    host.collect_csi_data(_make_collect_args(info=True))

    output = capsys.readouterr().out
    assert "motion" in output
    assert "empty" in output
    assert "Total" in output


def test_collect_csi_data_validates_arguments_and_imports(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "tools.lib.csi_io", raising=False)

    with pytest.raises(SystemExit):
        host.collect_csi_data(_make_collect_args(label=None))

    fake_csi_utils = ModuleType("tools.lib.csi_io")
    fake_csi_utils.CSICollector = object
    fake_csi_utils.UdpPacingSender = object
    fake_csi_utils.get_dataset_stats = lambda: {"labels": {}, "total_samples": 0}
    fake_csi_utils.get_default_bind_host = lambda: "127.0.0.1"
    monkeypatch.setitem(sys.modules, "tools.lib.csi_io", fake_csi_utils)

    with pytest.raises(SystemExit):
        host.collect_csi_data(_make_collect_args(target=None))

    with pytest.raises(SystemExit):
        host.collect_csi_data(_make_collect_args(start_delay=-1))


def test_collect_csi_data_handles_interrupt_and_runtime_error(monkeypatch) -> None:
    events: list[str] = []
    fake_csi_utils = ModuleType("tools.lib.csi_io")

    class InterruptCollector:
        def __init__(self, **kwargs):
            pass

        def collect_timed(self, duration: float, num_samples: int, **kwargs):
            raise KeyboardInterrupt

    class FakePacingSender:
        def __init__(self, **kwargs):
            pass

        def start(self):
            events.append("start")

        def stop(self):
            events.append("stop")

    fake_csi_utils.CSICollector = InterruptCollector
    fake_csi_utils.UdpPacingSender = FakePacingSender
    fake_csi_utils.get_dataset_stats = lambda: {"labels": {}, "total_samples": 0}
    fake_csi_utils.get_default_bind_host = lambda: "127.0.0.1"
    monkeypatch.setitem(sys.modules, "tools.lib.csi_io", fake_csi_utils)
    monkeypatch.setattr(host, "_wait_before_collection", lambda delay: None)

    host.collect_csi_data(_make_collect_args())
    assert events == ["start", "stop"]


def test_collect_applies_start_delay_before_starting_pacing(monkeypatch, capsys) -> None:
    events: list[object] = []
    fake_csi_utils = ModuleType("tools.lib.csi_io")
    fake_quality = ModuleType("tools.validate_dataset_quality")

    class FakeResult:
        def __init__(self, name: str, status: str, message: str):
            self.name = name
            self.status = status
            self.message = message

    class FakeData(dict):
        @property
        def files(self):
            return list(self.keys())

    class FakeCollector:
        def __init__(self, **kwargs):
            events.append(
                (
                    "collector_init",
                    kwargs["label"],
                    kwargs["bind_host"],
                    kwargs["expected_device_count"],
                    kwargs["detector_algorithm"],
                )
            )

        def collect_timed(self, duration: float, num_samples: int, **kwargs):
            events.append(("collect_timed", duration, num_samples, kwargs))
            return [Path("sample_1.npz"), Path("sample_2.npz")]

    class FakePacingSender:
        last_instance = None

        def __init__(self, **kwargs):
            events.append(
                ("sender_init", kwargs["target_host"], kwargs.get("interval_s"))
            )
            self.kwargs = kwargs
            self.rate_updates = []
            FakePacingSender.last_instance = self

        def start(self):
            events.append("start")

        def stop(self):
            events.append("stop")

        def set_rate_pps(self, rate_pps):
            self.rate_updates.append(float(rate_pps))

    fake_csi_utils.CSICollector = FakeCollector
    fake_csi_utils.UdpPacingSender = FakePacingSender
    fake_csi_utils.get_dataset_stats = lambda: {"labels": {}, "total_samples": 0}
    fake_csi_utils.get_default_bind_host = lambda: "127.0.0.1"
    fake_quality.validate_file_integrity = lambda path: (
        [
            FakeResult("file_load", "PASS", "File loads successfully"),
            FakeResult("sensing_contract", "PASS", "Sensing view keeps packets"),
        ],
        FakeData({"csi_data": object()}),
    )
    fake_quality.validate_signal_quality = lambda csi_data: [
        FakeResult("packet_count", "PASS", "7000 packets"),
        FakeResult("signal_level", "PASS", "Mean amplitude: 23.4"),
    ]
    fake_quality.validate_capture_continuity = lambda data, csi_data: [
        FakeResult("packet_rate", "PASS", "Packet rate: 100.0 pkt/s"),
        FakeResult(
            "stream_seq_max_gap",
            "PASS",
            "Largest stream gap: 0 packets with no missing packets detected (warn > 10, fail > 20)",
        ),
        FakeResult(
            "inter_packet_gap",
            "PASS",
            "Largest inter-packet gap: 10.0 ms via device_ticks_us at packet 0->1 (warn > 150.0 ms, fail > 250.0 ms)",
        ),
    ]

    monkeypatch.setitem(sys.modules, "tools.lib.csi_io", fake_csi_utils)
    monkeypatch.setitem(sys.modules, "tools.validate_dataset_quality", fake_quality)
    monkeypatch.setattr(host, "_wait_before_collection", lambda delay: events.append(("delay", delay)))

    host.collect_csi_data(_make_collect_args(target="192.168.1.17,192.168.1.24,192.168.1.29"))
    output = capsys.readouterr().out

    assert ("delay", 5.0) in events
    assert ("sender_init", ["192.168.1.17", "192.168.1.24", "192.168.1.29"], 0.01) in events
    collect_event = next(event for event in events if isinstance(event, tuple) and event[0] == "collect_timed")
    assert collect_event[1] == 10.0
    assert collect_event[2] == 2
    assert collect_event[3]["adaptive"] is True
    assert collect_event[3]["adaptive_controller_kwargs"] == {"boost_allowed": False}
    assert collect_event[3]["pacing_sender"] is FakePacingSender.last_instance
    assert collect_event[3]["ready_stable_seconds"] == 3.0
    assert FakePacingSender.last_instance.rate_updates == []
    assert ("collector_init", "static_presence", "127.0.0.1", 3, "classic") in events
    assert events.index(("delay", 5.0)) < events.index("start")
    assert events[-1] == "stop"
    assert "Post-collect quality:" in output
    assert "sample_1.npz: quality checks all pass" in output
    assert "sample_2.npz: quality checks all pass" in output
    assert "Collected 2 device file(s)" in output


def test_collect_reports_post_collect_gap_details(monkeypatch, capsys) -> None:
    fake_csi_utils = ModuleType("tools.lib.csi_io")
    fake_quality = ModuleType("tools.validate_dataset_quality")

    class FakeResult:
        def __init__(self, name: str, status: str, message: str):
            self.name = name
            self.status = status
            self.message = message

    class FakeData(dict):
        @property
        def files(self):
            return list(self.keys())

    class FakeCollector:
        def __init__(self, **kwargs):
            pass

        def collect_timed(self, duration: float, num_samples: int, **kwargs):
            return [Path("sample_1.npz")]

    class FakePacingSender:
        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            pass

    fake_csi_utils.CSICollector = FakeCollector
    fake_csi_utils.UdpPacingSender = FakePacingSender
    fake_csi_utils.get_dataset_stats = lambda: {"labels": {}, "total_samples": 0}
    fake_csi_utils.get_default_bind_host = lambda: "127.0.0.1"
    fake_quality.validate_file_integrity = lambda path: (
        [FakeResult("file_load", "PASS", "File loads successfully")],
        FakeData({"csi_data": object()}),
    )
    fake_quality.validate_signal_quality = lambda csi_data: []
    fake_quality.validate_capture_continuity = lambda data, csi_data: [
        FakeResult(
            "inter_packet_gap",
            "FAIL",
            "Largest inter-packet gap: 187.0 ms via device_ticks_us at packet 431->432 (warn > 150.0 ms, fail > 250.0 ms)",
        ),
        FakeResult(
            "stream_seq_max_gap",
            "WARN",
            "Largest stream gap: 14 packets after packet 430 (seq 1821 -> 1836) (warn > 10, fail > 20)",
        ),
    ]

    monkeypatch.setitem(sys.modules, "tools.lib.csi_io", fake_csi_utils)
    monkeypatch.setitem(sys.modules, "tools.validate_dataset_quality", fake_quality)
    monkeypatch.setattr(host, "_wait_before_collection", lambda delay: None)

    host.collect_csi_data(_make_collect_args(samples=1, start_delay=1.0))
    output = capsys.readouterr().out

    assert "sample_1.npz: 1 warn, 1 fail" in output
    assert "Largest stream gap: 14 packets after packet 430 (seq 1821 -> 1836)" in output
    assert "Largest inter-packet gap: 187.0 ms via device_ticks_us at packet 431->432" in output


def test_collect_timed_adaptive_adjusts_pacing(monkeypatch) -> None:
    import importlib

    csi_io = importlib.import_module("tools.lib.csi_io")
    clock = {"now": 0.0}

    class FakeReadyDetector:
        def __init__(self):
            self.window_size = 1
            self.current_metric = 0.0
            self.current_threshold = 1.0

        def process_packet(self, packet):
            self.current_metric = 0.0

        def is_ready(self):
            return True

    class FakePacket:
        def __init__(self, seq_num: int, tx_backpressure_total: int):
            self.seq_num = seq_num
            self.device_id = 0xABC123
            self.iq_raw = [seq_num, seq_num + 1, seq_num + 2, seq_num + 3]
            self.source_ip = "192.168.1.29"
            self.channel = 8
            self.rssi_dbm = -47
            self.chip = "s3"
            self.tx_backpressure_total = tx_backpressure_total

    packets_by_token = {
        b"ready": [FakePacket(1, 0)],
        b"rec1": [FakePacket(2, 6)],
        b"rec2": [FakePacket(3, 12)],
        b"rec3": [FakePacket(4, 12)],
        b"rec4": [FakePacket(5, 12)],
    }

    class FakeSocket:
        def __init__(self, *args, **kwargs):
            self.timeout = None
            self.events = [
                (0.0, b"ready"),
                (1.2, b"rec1"),
                (2.4, b"rec2"),
                (3.6, b"rec3"),
                (4.8, b"rec4"),
            ]

        def bind(self, addr):
            pass

        def settimeout(self, value):
            self.timeout = value

        def gettimeout(self):
            return self.timeout

        def recvfrom(self, max_bytes):
            if self.timeout == 0.0:
                raise BlockingIOError
            if not self.events:
                raise csi_io.socket.timeout
            next_time, token = self.events.pop(0)
            clock["now"] = next_time
            return token, ("192.168.1.29", 5001)

        def close(self):
            pass

    class FakePacingSender:
        def __init__(self):
            self.rate_updates = []

        def get_rate_pps(self):
            return 100.0

        def set_rate_pps(self, rate_pps):
            self.rate_updates.append(float(rate_pps))

    monkeypatch.setattr(csi_io.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(csi_io.socket, "socket", lambda *args, **kwargs: FakeSocket(*args, **kwargs))
    monkeypatch.setattr(csi_io.CSICollector, "_build_ready_detector", lambda self: FakeReadyDetector())
    monkeypatch.setattr(csi_io.CSICollector, "READY_STABLE_SECONDS", 0.0)
    monkeypatch.setattr(
        csi_io.CollectionDetectorGate,
        "default_window_size",
        staticmethod(lambda: 1),
    )
    monkeypatch.setattr(csi_io.CSICollector, "save_samples_by_device", lambda self, packets: [Path("sample_1.npz")])

    collector = csi_io.CSICollector(
        label="motion",
        port=5001,
        bind_host="127.0.0.1",
        expected_device_count=1,
        expected_source_hosts=["192.168.1.29"],
    )
    monkeypatch.setattr(collector.receiver, "_parse_packets", lambda data: packets_by_token[data])

    pacing_sender = FakePacingSender()
    saved_files = collector.collect_timed(
        duration=1.0,
        num_samples=1,
        quiet=True,
        pacing_sender=pacing_sender,
        adaptive=True,
    )

    assert saved_files == [Path("sample_1.npz")]
    assert pacing_sender.rate_updates == pytest.approx([85.0, 87.0])


def test_collect_timed_zero_ready_gate_skips_wait(monkeypatch) -> None:
    import importlib

    csi_io = importlib.import_module("tools.lib.csi_io")

    class FakeSocket:
        def bind(self, addr):
            pass

        def settimeout(self, value):
            pass

        def close(self):
            pass

    monkeypatch.setattr(csi_io.socket, "socket", lambda *args, **kwargs: FakeSocket())

    collector = csi_io.CSICollector(
        label="motion",
        port=5001,
        bind_host="127.0.0.1",
        expected_device_count=1,
        expected_source_hosts=["192.168.1.29"],
    )

    waited = {"called": False}

    def fake_wait(*args, **kwargs):
        waited["called"] = True
        return {}

    monkeypatch.setattr(collector, "_wait_for_ready_state_with_pacing", fake_wait)
    monkeypatch.setattr(collector, "_drain_udp_backlog", lambda: None)
    monkeypatch.setattr(collector, "_reset_live_status_block", lambda: None)
    monkeypatch.setattr(collector, "_collect_with_live_status", lambda *args, **kwargs: [])
    monkeypatch.setattr(collector, "save_samples_by_device", lambda packets: [Path("sample_1.npz")])

    saved_files = collector.collect_timed(
        duration=1.0,
        num_samples=1,
        quiet=True,
        ready_stable_seconds=0.0,
    )

    assert saved_files == [Path("sample_1.npz")]
    assert waited["called"] is False


def test_csi_collector_source_filter_rejects_off_target_unicast_and_accepts_multicast() -> None:
    import importlib

    csi_io = importlib.import_module("tools.lib.csi_io")

    unicast_collector = csi_io.CSICollector(
        label="motion",
        port=5001,
        bind_host="127.0.0.1",
        expected_device_count=2,
        expected_source_hosts=["192.168.1.29", "192.168.1.35"],
    )
    multicast_collector = csi_io.CSICollector(
        label="motion",
        port=5001,
        bind_host="127.0.0.1",
        expected_device_count=1,
        expected_source_hosts=["239.1.1.1"],
    )

    assert unicast_collector._should_accept_source_ip("192.168.1.29") is True
    assert unicast_collector._should_accept_source_ip("192.168.1.30") is False
    assert multicast_collector._should_accept_source_ip("192.168.1.30") is True


def test_collect_live_saves_raw_packets_with_collector(monkeypatch, capsys) -> None:
    events: list[object] = []
    clock = {"now": 0.0}
    fake_csi_utils = ModuleType("tools.lib.csi_io")
    fake_config = ModuleType("config")
    fake_ml_detector = ModuleType("ml_detector")
    fake_quality = ModuleType("tools.validate_dataset_quality")
    fake_runtime_policy = ModuleType("runtime_policy")

    fake_config.DEFAULT_SUBCARRIERS = [12, 14]
    fake_config.SEG_WINDOW_SIZE = 2
    fake_config.ENABLE_LOWPASS_FILTER = False
    fake_config.LOWPASS_CUTOFF = 11.0
    fake_config.ENABLE_HAMPEL_FILTER = True
    fake_config.HAMPEL_WINDOW = 7
    fake_config.HAMPEL_THRESHOLD = 5.0
    fake_config.PUBLISH_INTERVAL = 1
    fake_config.EVALUATION_INTERVAL = 1
    fake_config.MOTION_ON_HITS = 3
    fake_config.MOTION_OFF_HITS = 3

    class FakePacket:
        def __init__(self, seq_num: int):
            self.seq_num = seq_num
            self.device_id = 0xABC123
            self.iq_raw = [seq_num, seq_num + 1, seq_num + 2, seq_num + 3]
            self.source_ip = "192.168.1.29"
            self.channel = 8
            self.rssi_dbm = -47
            self.chip = "s3"

    class FakeResult:
        def __init__(self, name: str, status: str, message: str):
            self.name = name
            self.status = status
            self.message = message

    class FakeData(dict):
        @property
        def files(self):
            return list(self.keys())

    class FakeCollector:
        def __init__(self, **kwargs):
            events.append(("collector_init", kwargs["label"], kwargs["description"], kwargs["expected_device_count"]))

        def save_samples_by_device(self, packets):
            events.append(("save_sample", [p.seq_num for p in packets]))
            return [Path("empty_c3_64sc_dev0000000000abc123_20260630_120000_000001_0001.npz")]

    class FakeReceiver:
        def __init__(self, **kwargs):
            self._callbacks = []
            self.dropped_count = 0
            self.pps = 100

        def add_callback(self, callback):
            self._callbacks.append(callback)

        def run(self, timeout: float = 0, quiet: bool = False):
            packet_times = [
                (0.0, FakePacket(1)),
                (1.0, FakePacket(2)),
                (2.0, FakePacket(3)),
                (3.1, FakePacket(4)),
                (4.1, FakePacket(5)),
            ]
            for current_time, packet in packet_times:
                clock["now"] = current_time
                for callback in self._callbacks:
                    callback(packet)
            raise KeyboardInterrupt

        def stop(self):
            events.append("receiver_stop")

    class FakePacingSender:
        def __init__(self, **kwargs):
            events.append(("sender_init", kwargs["target_host"]))

        def start(self):
            events.append("start")

        def stop(self):
            events.append("stop")

    class FakeContext:
        last_turbulence = 0.0

    class FakeMLDetector:
        def __init__(self, **kwargs):
            self._context = FakeContext()

        def process_packet(self, csi_data, subcarriers):
            pass

        def update_state(self):
            return {"probability": 0.0, "threshold": 0.5, "state": 0}

        def is_ready(self):
            return True

        def _extract_features(self):
            return [0.0, 0.0]

    class FakeRuntimeMotionPolicy:
        def __init__(self, **kwargs):
            pass

        def note_packet(self):
            pass

        def should_evaluate(self, should_publish):
            return True

        def apply_state(self, state):
            return state, None

        def after_evaluation(self):
            pass

    fake_csi_utils.CSICollector = FakeCollector
    fake_csi_utils.CSIReceiver = FakeReceiver
    fake_csi_utils.UdpPacingSender = FakePacingSender
    fake_csi_utils.get_default_bind_host = lambda: "127.0.0.1"
    fake_csi_utils.AdaptivePacingController = AdaptivePacingController
    fake_ml_detector.FEATURE_NAMES = ["a", "b"]
    fake_ml_detector.ML_DEFAULT_THRESHOLD = 0.5
    fake_ml_detector.ML_METRIC_SCALE = 1.0
    fake_ml_detector.MLDetector = FakeMLDetector
    fake_quality.validate_file_integrity = lambda path: (
        [FakeResult("file_load", "PASS", "File loads successfully")],
        FakeData({"csi_data": object()}),
    )
    fake_quality.validate_signal_quality = lambda csi_data: [
        FakeResult("packet_count", "PASS", "6000 packets"),
        FakeResult("signal_level", "PASS", "Mean amplitude: 22.0"),
    ]
    fake_quality.validate_capture_continuity = lambda data, csi_data: [
        FakeResult("packet_rate", "PASS", "Packet rate: 100.0 pkt/s"),
    ]
    fake_runtime_policy.RuntimeMotionPolicy = FakeRuntimeMotionPolicy

    monkeypatch.setitem(sys.modules, "tools.lib.csi_io", fake_csi_utils)
    monkeypatch.setitem(sys.modules, "config", fake_config)
    monkeypatch.setitem(sys.modules, "ml_detector", fake_ml_detector)
    monkeypatch.setitem(sys.modules, "tools.validate_dataset_quality", fake_quality)
    monkeypatch.setitem(sys.modules, "runtime_policy", fake_runtime_policy)
    monkeypatch.setattr(host.time, "monotonic", lambda: clock["now"])

    host.collect_csi_data(
        _make_live_collect_args(
            target="192.168.1.29",
            label="empty",
            description="live collect ML, idle-motion-idle",
            detector="ml",
        )
    )

    output = capsys.readouterr().out
    assert ("collector_init", "empty", "live collect ML, idle-motion-idle", 1) in events
    assert ("sender_init", ["192.168.1.29"]) in events
    assert ("save_sample", [4, 5]) in events
    assert "STATUS: RECORDING 1/1" in output
    assert "recording until Ctrl+C" in output
    assert "Post-collect quality:" in output
    assert "quality checks all pass" in output
    assert "stop" in events
    assert "receiver_stop" in events


def test_collect_live_filters_off_target_packets_for_multi_unicast(monkeypatch, capsys) -> None:
    events: list[object] = []
    clock = {"now": 0.0}
    fake_quality = ModuleType("tools.validate_dataset_quality")

    class FakePacket:
        def __init__(self, seq_num: int, device_id: int, source_ip: str):
            self.seq_num = seq_num
            self.device_id = device_id
            self.iq_raw = [seq_num, seq_num + 1, seq_num + 2, seq_num + 3]
            self.source_ip = source_ip
            self.channel = 8
            self.rssi_dbm = -47
            self.chip = "s3"

    class FakeResult:
        def __init__(self, name: str, status: str, message: str):
            self.name = name
            self.status = status
            self.message = message

    class FakeData(dict):
        @property
        def files(self):
            return list(self.keys())

    class FakeCollector:
        def __init__(self, **kwargs):
            pass

        def save_samples_by_device(self, packets):
            events.append(
                (
                    "save_sample",
                    [(p.seq_num, p.device_id, p.source_ip) for p in packets],
                )
            )
            return [Path("motion_s3_64sc_dev0000000000000029_20260724_130000_0001.npz")]

    class FakeReceiver:
        def __init__(self, **kwargs):
            self._callbacks = []
            self.dropped_count = 0
            self.pps = 100

        def add_callback(self, callback):
            self._callbacks.append(callback)

        def run(self, timeout: float = 0, quiet: bool = False):
            packet_times = [
                (0.0, FakePacket(1, 0x29, "192.168.1.29")),
                (0.5, FakePacket(2, 0x30, "192.168.1.30")),
                (1.0, FakePacket(3, 0x35, "192.168.1.35")),
            ]
            for current_time, packet in packet_times:
                clock["now"] = current_time
                for callback in self._callbacks:
                    callback(packet)
            raise KeyboardInterrupt

        def stop(self):
            pass

    class FakePacingSender:
        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            pass

    fake_quality.validate_file_integrity = lambda path: (
        [FakeResult("file_load", "PASS", "File loads successfully")],
        FakeData({"csi_data": object()}),
    )
    fake_quality.validate_signal_quality = lambda csi_data: []
    fake_quality.validate_capture_continuity = lambda data, csi_data: []

    _install_live_collect_modules(monkeypatch, FakeReceiver, FakePacingSender, FakeCollector)
    monkeypatch.setitem(sys.modules, "tools.validate_dataset_quality", fake_quality)
    monkeypatch.setattr(host.time, "monotonic", lambda: clock["now"])

    host.collect_csi_data(
        _make_live_collect_args(
            target="192.168.1.29,192.168.1.35",
            label="motion",
            detector="ml",
            ready_stable_seconds=0.0,
        )
    )
    output = capsys.readouterr().out

    save_event = next(event for event in events if event[0] == "save_sample")
    assert save_event[1] == [
        (1, 0x29, "192.168.1.29"),
        (3, 0x35, "192.168.1.35"),
    ]
    assert "Saved 1 live capture file(s)" in output


def test_collect_live_accepts_all_sources_for_multicast_targets(monkeypatch, capsys) -> None:
    events: list[object] = []
    clock = {"now": 0.0}
    fake_quality = ModuleType("tools.validate_dataset_quality")

    class FakePacket:
        def __init__(self, seq_num: int, device_id: int, source_ip: str):
            self.seq_num = seq_num
            self.device_id = device_id
            self.iq_raw = [seq_num, seq_num + 1, seq_num + 2, seq_num + 3]
            self.source_ip = source_ip
            self.channel = 8
            self.rssi_dbm = -47
            self.chip = "s3"

    class FakeResult:
        def __init__(self, name: str, status: str, message: str):
            self.name = name
            self.status = status
            self.message = message

    class FakeData(dict):
        @property
        def files(self):
            return list(self.keys())

    class FakeCollector:
        def __init__(self, **kwargs):
            pass

        def save_samples_by_device(self, packets):
            events.append(
                (
                    "save_sample",
                    [(p.seq_num, p.device_id, p.source_ip) for p in packets],
                )
            )
            return [Path("motion_s3_64sc_dev0000000000000029_20260724_130000_0001.npz")]

    class FakeReceiver:
        def __init__(self, **kwargs):
            self._callbacks = []
            self.dropped_count = 0
            self.pps = 100

        def add_callback(self, callback):
            self._callbacks.append(callback)

        def run(self, timeout: float = 0, quiet: bool = False):
            packet_times = [
                (0.0, FakePacket(1, 0x29, "192.168.1.29")),
                (0.5, FakePacket(2, 0x30, "192.168.1.30")),
            ]
            for current_time, packet in packet_times:
                clock["now"] = current_time
                for callback in self._callbacks:
                    callback(packet)
            raise KeyboardInterrupt

        def stop(self):
            pass

    class FakePacingSender:
        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            pass

    fake_quality.validate_file_integrity = lambda path: (
        [FakeResult("file_load", "PASS", "File loads successfully")],
        FakeData({"csi_data": object()}),
    )
    fake_quality.validate_signal_quality = lambda csi_data: []
    fake_quality.validate_capture_continuity = lambda data, csi_data: []

    _install_live_collect_modules(monkeypatch, FakeReceiver, FakePacingSender, FakeCollector)
    monkeypatch.setitem(sys.modules, "tools.validate_dataset_quality", fake_quality)
    monkeypatch.setattr(host.time, "monotonic", lambda: clock["now"])

    host.collect_csi_data(
        _make_live_collect_args(
            target="239.1.1.1",
            label="motion",
            detector="ml",
            ready_stable_seconds=0.0,
        )
    )
    capsys.readouterr()

    save_event = next(event for event in events if event[0] == "save_sample")
    assert save_event[1] == [
        (1, 0x29, "192.168.1.29"),
        (2, 0x30, "192.168.1.30"),
    ]


def test_collect_live_zero_ready_gate_starts_saving_immediately(monkeypatch, capsys) -> None:
    events: list[object] = []
    clock = {"now": 0.0}
    fake_csi_utils = ModuleType("tools.lib.csi_io")
    fake_config = ModuleType("config")
    fake_ml_detector = ModuleType("ml_detector")
    fake_runtime_policy = ModuleType("runtime_policy")

    fake_config.DEFAULT_SUBCARRIERS = [12, 14]
    fake_config.SEG_WINDOW_SIZE = 2
    fake_config.ENABLE_LOWPASS_FILTER = False
    fake_config.LOWPASS_CUTOFF = 11.0
    fake_config.ENABLE_HAMPEL_FILTER = True
    fake_config.HAMPEL_WINDOW = 7
    fake_config.HAMPEL_THRESHOLD = 5.0
    fake_config.PUBLISH_INTERVAL = 1
    fake_config.EVALUATION_INTERVAL = 1
    fake_config.MOTION_ON_HITS = 3
    fake_config.MOTION_OFF_HITS = 3

    class FakePacket:
        def __init__(self, seq_num: int):
            self.seq_num = seq_num
            self.device_id = 0xABC123
            self.iq_raw = [seq_num, seq_num + 1, seq_num + 2, seq_num + 3]
            self.source_ip = "192.168.1.29"
            self.channel = 8
            self.rssi_dbm = -47
            self.chip = "s3"

    class FakeCollector:
        def __init__(self, **kwargs):
            events.append(("collector_init", kwargs["label"]))

        def save_samples_by_device(self, packets):
            events.append(("save_sample", [p.seq_num for p in packets]))
            return [Path("empty_c3_64sc_dev0000000000abc123_20260630_120000_000001_0001.npz")]

    class FakeReceiver:
        def __init__(self, **kwargs):
            self._callbacks = []
            self.dropped_count = 0
            self.pps = 100

        def add_callback(self, callback):
            self._callbacks.append(callback)

        def run(self, timeout: float = 0, quiet: bool = False):
            packet_times = [
                (0.0, FakePacket(1)),
                (1.0, FakePacket(2)),
                (2.0, FakePacket(3)),
                (3.1, FakePacket(4)),
                (4.1, FakePacket(5)),
            ]
            for current_time, packet in packet_times:
                clock["now"] = current_time
                for callback in self._callbacks:
                    callback(packet)
            raise KeyboardInterrupt

        def stop(self):
            events.append("receiver_stop")

    class FakePacingSender:
        def __init__(self, **kwargs):
            events.append(("sender_init", kwargs["target_host"]))

        def start(self):
            events.append("start")

        def stop(self):
            events.append("stop")

    class FakeContext:
        last_turbulence = 0.0

    class FakeMLDetector:
        def __init__(self, **kwargs):
            self._context = FakeContext()

        def process_packet(self, csi_data, subcarriers):
            pass

        def update_state(self):
            return {"probability": 0.0, "threshold": 0.5, "state": 0}

        def is_ready(self):
            return True

        def _extract_features(self):
            return [0.0, 0.0]

    class FakeRuntimeMotionPolicy:
        def __init__(self, **kwargs):
            pass

        def note_packet(self):
            pass

        def should_evaluate(self, should_publish):
            return True

        def apply_state(self, state):
            return state, None

        def after_evaluation(self):
            pass

    fake_csi_utils.CSICollector = FakeCollector
    fake_csi_utils.CSIReceiver = FakeReceiver
    fake_csi_utils.UdpPacingSender = FakePacingSender
    fake_csi_utils.get_default_bind_host = lambda: "127.0.0.1"
    fake_csi_utils.AdaptivePacingController = AdaptivePacingController
    fake_ml_detector.FEATURE_NAMES = ["a", "b"]
    fake_ml_detector.ML_DEFAULT_THRESHOLD = 0.5
    fake_ml_detector.ML_METRIC_SCALE = 1.0
    fake_ml_detector.MLDetector = FakeMLDetector
    fake_runtime_policy.RuntimeMotionPolicy = FakeRuntimeMotionPolicy

    monkeypatch.setitem(sys.modules, "tools.lib.csi_io", fake_csi_utils)
    monkeypatch.setitem(sys.modules, "config", fake_config)
    monkeypatch.setitem(sys.modules, "ml_detector", fake_ml_detector)
    monkeypatch.setitem(sys.modules, "runtime_policy", fake_runtime_policy)
    monkeypatch.setattr(host.time, "monotonic", lambda: clock["now"])

    host.collect_csi_data(
        _make_live_collect_args(
            target="192.168.1.29",
            label="empty",
            detector="ml",
            ready_stable_seconds=0.0,
        )
    )

    output = capsys.readouterr().out
    assert ("save_sample", [1, 2, 3, 4, 5]) in events
    assert "Ready gate:" in output and "disabled" in output


def test_collect_live_duration_interrupt_discards_partial_capture(monkeypatch, capsys) -> None:
    events: list[object] = []
    clock = {"now": 0.0}
    fake_csi_utils = ModuleType("tools.lib.csi_io")
    fake_config = ModuleType("config")
    fake_ml_detector = ModuleType("ml_detector")
    fake_runtime_policy = ModuleType("runtime_policy")

    fake_config.DEFAULT_SUBCARRIERS = [12, 14]
    fake_config.SEG_WINDOW_SIZE = 2
    fake_config.ENABLE_LOWPASS_FILTER = False
    fake_config.LOWPASS_CUTOFF = 11.0
    fake_config.ENABLE_HAMPEL_FILTER = True
    fake_config.HAMPEL_WINDOW = 7
    fake_config.HAMPEL_THRESHOLD = 5.0
    fake_config.PUBLISH_INTERVAL = 1
    fake_config.EVALUATION_INTERVAL = 1
    fake_config.MOTION_ON_HITS = 3
    fake_config.MOTION_OFF_HITS = 3

    class FakePacket:
        def __init__(self, seq_num: int):
            self.seq_num = seq_num
            self.device_id = 0xABC123
            self.iq_raw = [seq_num, seq_num + 1, seq_num + 2, seq_num + 3]
            self.source_ip = "192.168.1.29"
            self.channel = 8
            self.rssi_dbm = -47
            self.chip = "s3"

    class FakeCollector:
        def __init__(self, **kwargs):
            events.append(("collector_init", kwargs["label"]))

        def save_samples_by_device(self, packets):
            events.append(("save_sample", [p.seq_num for p in packets]))
            return [Path("should_not_exist.npz")]

    class FakeReceiver:
        def __init__(self, **kwargs):
            self._callbacks = []
            self.dropped_count = 0
            self.pps = 100

        def add_callback(self, callback):
            self._callbacks.append(callback)

        def run(self, timeout: float = 0, quiet: bool = False):
            packet_times = [
                (0.0, FakePacket(1)),
                (1.0, FakePacket(2)),
                (2.0, FakePacket(3)),
                (3.1, FakePacket(4)),
                (4.1, FakePacket(5)),
            ]
            for current_time, packet in packet_times:
                clock["now"] = current_time
                for callback in self._callbacks:
                    callback(packet)
            raise KeyboardInterrupt

        def stop(self):
            events.append("receiver_stop")

    class FakePacingSender:
        def __init__(self, **kwargs):
            events.append(("sender_init", kwargs["target_host"]))

        def start(self):
            events.append("start")

        def stop(self):
            events.append("stop")

    class FakeContext:
        last_turbulence = 0.0

    class FakeMLDetector:
        def __init__(self, **kwargs):
            self._context = FakeContext()

        def process_packet(self, csi_data, subcarriers):
            pass

        def update_state(self):
            return {"probability": 0.0, "threshold": 0.5, "state": 0}

        def is_ready(self):
            return True

        def _extract_features(self):
            return [0.0, 0.0]

    class FakeRuntimeMotionPolicy:
        def __init__(self, **kwargs):
            pass

        def note_packet(self):
            pass

        def should_evaluate(self, should_publish):
            return True

        def apply_state(self, state):
            return state, None

        def after_evaluation(self):
            pass

    fake_csi_utils.CSICollector = FakeCollector
    fake_csi_utils.CSIReceiver = FakeReceiver
    fake_csi_utils.UdpPacingSender = FakePacingSender
    fake_csi_utils.get_default_bind_host = lambda: "127.0.0.1"
    fake_csi_utils.AdaptivePacingController = AdaptivePacingController
    fake_ml_detector.FEATURE_NAMES = ["a", "b"]
    fake_ml_detector.ML_DEFAULT_THRESHOLD = 0.5
    fake_ml_detector.ML_METRIC_SCALE = 1.0
    fake_ml_detector.MLDetector = FakeMLDetector
    fake_runtime_policy.RuntimeMotionPolicy = FakeRuntimeMotionPolicy

    monkeypatch.setitem(sys.modules, "tools.lib.csi_io", fake_csi_utils)
    monkeypatch.setitem(sys.modules, "config", fake_config)
    monkeypatch.setitem(sys.modules, "ml_detector", fake_ml_detector)
    monkeypatch.setitem(sys.modules, "runtime_policy", fake_runtime_policy)
    monkeypatch.setattr(host.time, "monotonic", lambda: clock["now"])

    host.collect_csi_data(
        _make_live_collect_args(
            target="192.168.1.29",
            label="empty",
            duration=10,
            description="interrupted run",
            detector="ml",
        )
    )

    output = capsys.readouterr().out
    assert ("collector_init", "empty") in events
    assert ("save_sample", [4, 5]) not in events
    assert "Live capture interrupted before duration elapsed; nothing saved" in output


def test_collect_live_validates_save_arguments(monkeypatch) -> None:
    class FakeReceiver:
        def __init__(self, **kwargs):
            pass

        def add_callback(self, callback):
            pass

        def run(self, timeout: float = 0, quiet: bool = False):
            raise KeyboardInterrupt

        def stop(self):
            pass

    class FakePacingSender:
        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            pass

    _install_live_collect_modules(monkeypatch, FakeReceiver, FakePacingSender)

    with pytest.raises(SystemExit):
        host.collect_csi_data(_make_live_collect_args(duration=0))

    host.collect_csi_data(_make_live_collect_args(label=None, duration=5))


def test_collect_live_handles_import_failure(monkeypatch) -> None:
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        blocked = {
            "tools.lib.csi_io",
            "config",
            "classic_detector",
            "ml_detector",
            "runtime_policy",
            "threshold",
            "src.config",
            "src.classic_detector",
            "src.ml_detector",
            "src.runtime_policy",
            "src.threshold",
        }
        if name in blocked:
            raise ImportError(f"blocked import: {name}")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(SystemExit):
        host.collect_csi_data(_make_live_collect_args())


def test_collect_live_handles_save_without_packets(monkeypatch, capsys) -> None:
    fake_csi_utils = ModuleType("tools.lib.csi_io")
    fake_config = ModuleType("config")
    fake_ml_detector = ModuleType("ml_detector")
    fake_runtime_policy = ModuleType("runtime_policy")

    fake_config.DEFAULT_SUBCARRIERS = [12, 14]
    fake_config.SEG_WINDOW_SIZE = 2
    fake_config.ENABLE_LOWPASS_FILTER = True
    fake_config.LOWPASS_CUTOFF = 11.0
    fake_config.ENABLE_HAMPEL_FILTER = True
    fake_config.HAMPEL_WINDOW = 7
    fake_config.HAMPEL_THRESHOLD = 5.0
    fake_config.PUBLISH_INTERVAL = 1
    fake_config.EVALUATION_INTERVAL = 1
    fake_config.MOTION_ON_HITS = 3
    fake_config.MOTION_OFF_HITS = 3

    class FakePacket:
        iq_raw = [1, 2, 3, 4]
        chip = "c6"
        source_ip = "192.168.1.29"
        channel = 8
        rssi_dbm = -47

    class FakeCollector:
        def __init__(self, **kwargs):
            pass

        def save_samples_by_device(self, packets):
            return None

    class FakeReceiver:
        def __init__(self, **kwargs):
            self._callbacks = []
            self.dropped_count = 2
            self.pps = 50

        def add_callback(self, callback):
            self._callbacks.append(callback)

        def run(self, timeout: float = 0, quiet: bool = False):
            for callback in self._callbacks:
                callback(FakePacket())
            raise KeyboardInterrupt

        def stop(self):
            pass

    class FakePacingSender:
        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            pass

    class FakeContext:
        def __init__(self):
            self.last_turbulence = 0.5
            self.buffer_count = 2
            self.window_size = 2
            self.buffer_index = 0
            self.turbulence_buffer = [0.25, 0.5]

        def _compute_spatial_turbulence_in_buffer(self, iq_raw, subcarriers):
            return 0.75

    class FakeMLDetector:
        def __init__(self, **kwargs):
            self._context = FakeContext()

        def process_packet(self, csi_data, subcarriers):
            pass

        def update_state(self):
            return {"probability": 0.4, "threshold": 0.5, "state": 1}

        def is_ready(self):
            return True

        def _extract_features(self):
            return [0.1, 0.2]

    class FakeRuntimeMotionPolicy:
        def __init__(self, **kwargs):
            pass

        def note_packet(self):
            pass

        def should_evaluate(self, should_publish):
            return True

        def apply_state(self, state):
            return state, None

        def after_evaluation(self):
            pass

    fake_csi_utils.CSICollector = FakeCollector
    fake_csi_utils.CSIReceiver = FakeReceiver
    fake_csi_utils.UdpPacingSender = FakePacingSender
    fake_csi_utils.get_default_bind_host = lambda: "127.0.0.1"
    fake_csi_utils.AdaptivePacingController = AdaptivePacingController
    fake_ml_detector.FEATURE_NAMES = ["f1", "f2"]
    fake_ml_detector.ML_DEFAULT_THRESHOLD = 0.5
    fake_ml_detector.ML_METRIC_SCALE = 1.0
    fake_ml_detector.MLDetector = FakeMLDetector
    fake_runtime_policy.RuntimeMotionPolicy = FakeRuntimeMotionPolicy

    monkeypatch.setitem(sys.modules, "tools.lib.csi_io", fake_csi_utils)
    monkeypatch.setitem(sys.modules, "config", fake_config)
    monkeypatch.setitem(sys.modules, "ml_detector", fake_ml_detector)
    monkeypatch.setitem(sys.modules, "runtime_policy", fake_runtime_policy)

    host.collect_csi_data(
        _make_live_collect_args(
            target="192.168.1.29",
            detector="ml",
            description="feature run",
        )
    )

    output = capsys.readouterr().out
    assert "Threshold:" in output and "0.5" in output
    assert "Low-pass:" in output and "ON" in output
    assert "Save:" in output and "label=test duration=until Ctrl+C" in output
    assert "STATUS: STABILIZING 1/1" in output
    assert "No live capture packets received; nothing saved" in output


def test_collect_live_keeps_fixed_pacing_with_fixed_flag(monkeypatch, capsys) -> None:
    clock = {"now": 0.0}

    class FakePacket:
        def __init__(self, seq_num: int, tx_backpressure_total: int):
            self.seq_num = seq_num
            self.device_id = 0xABC123
            self.iq_raw = [seq_num, seq_num + 1, seq_num + 2, seq_num + 3]
            self.source_ip = "192.168.1.29"
            self.channel = 8
            self.rssi_dbm = -47
            self.chip = "s3"
            self.tx_backpressure_total = tx_backpressure_total

    class FakeReceiver:
        def __init__(self, **kwargs):
            self._callbacks = []
            self.dropped_count = 0
            self.pps = 100

        def add_callback(self, callback):
            self._callbacks.append(callback)

        def run(self, timeout: float = 0, quiet: bool = False):
            packet_schedule = [
                (0.0, FakePacket(1, 0)),
                (1.2, FakePacket(2, 5)),
                (2.4, FakePacket(3, 9)),
                (3.6, FakePacket(4, 9)),
            ]
            for current_time, packet in packet_schedule:
                clock["now"] = current_time
                for callback in self._callbacks:
                    callback(packet)
            raise KeyboardInterrupt

        def stop(self):
            pass

    class FakePacingSender:
        last_instance = None

        def __init__(self, **kwargs):
            self.rate_updates = []
            FakePacingSender.last_instance = self

        def start(self):
            pass

        def stop(self):
            pass

        def set_rate_pps(self, rate_pps):
            self.rate_updates.append(float(rate_pps))

    _install_live_collect_modules(
        monkeypatch,
        FakeReceiver,
        FakePacingSender,
        config_overrides={"PUBLISH_INTERVAL": 1, "EVALUATION_INTERVAL": 1},
    )
    monkeypatch.setattr(host.time, "monotonic", lambda: clock["now"])

    host.collect_csi_data(
        _make_live_collect_args(target="192.168.1.29", detector="ml", label=None, adaptive=False)
    )

    output = capsys.readouterr().out
    assert FakePacingSender.last_instance is not None
    assert FakePacingSender.last_instance.rate_updates == []
    assert "bp:active(+4)" in output
    assert "Pps:" in output and "(fixed)" in output


def test_collect_live_adapts_pacing_from_backpressure_feedback(monkeypatch, capsys) -> None:
    clock = {"now": 0.0}

    class FakePacket:
        def __init__(self, seq_num: int, tx_backpressure_total: int):
            self.seq_num = seq_num
            self.device_id = 0xABC123
            self.iq_raw = [seq_num, seq_num + 1, seq_num + 2, seq_num + 3]
            self.source_ip = "192.168.1.29"
            self.channel = 8
            self.rssi_dbm = -47
            self.chip = "s3"
            self.tx_backpressure_total = tx_backpressure_total

    class FakeReceiver:
        def __init__(self, **kwargs):
            self._callbacks = []
            self.dropped_count = 0
            self.pps = 100

        def add_callback(self, callback):
            self._callbacks.append(callback)

        def run(self, timeout: float = 0, quiet: bool = False):
            packet_schedule = [
                (0.0, FakePacket(1, 0)),
                (1.2, FakePacket(2, 6)),
                (2.4, FakePacket(3, 12)),
                (3.6, FakePacket(4, 12)),
                (4.8, FakePacket(5, 12)),
                (6.0, FakePacket(6, 12)),
            ]
            for current_time, packet in packet_schedule:
                clock["now"] = current_time
                for callback in self._callbacks:
                    callback(packet)
            raise KeyboardInterrupt

        def stop(self):
            pass

    class FakePacingSender:
        last_instance = None

        def __init__(self, **kwargs):
            self.rate_updates = []
            FakePacingSender.last_instance = self

        def start(self):
            pass

        def stop(self):
            pass

        def set_rate_pps(self, rate_pps):
            self.rate_updates.append(float(rate_pps))

    _install_live_collect_modules(
        monkeypatch,
        FakeReceiver,
        FakePacingSender,
        config_overrides={"PUBLISH_INTERVAL": 1, "EVALUATION_INTERVAL": 1},
    )
    monkeypatch.setattr(host.time, "monotonic", lambda: clock["now"])

    host.collect_csi_data(_make_live_collect_args(target="192.168.1.29", detector="ml", label=None))

    output = capsys.readouterr().out
    assert FakePacingSender.last_instance is not None
    assert FakePacingSender.last_instance.rate_updates == pytest.approx([85.0, 87.0, 89.0])
    assert "bp:active(+6)" in output
    assert "Pps:" in output and "(adaptive)" in output


def test_collect_live_sets_detector_window_from_pps(monkeypatch, capsys) -> None:
    class FakePacket:
        def __init__(self, seq_num: int):
            self.seq_num = seq_num
            self.device_id = 0xABC123
            self.iq_raw = [seq_num, seq_num + 1, seq_num + 2, seq_num + 3]
            self.source_ip = "192.168.1.29"
            self.channel = 8
            self.rssi_dbm = -47
            self.chip = "s3"
            self.tx_backpressure_total = 0

    class FakeReceiver:
        def __init__(self, **kwargs):
            self._callbacks = []
            self.dropped_count = 0
            self.pps = 42

        def add_callback(self, callback):
            self._callbacks.append(callback)

        def run(self, timeout: float = 0, quiet: bool = False):
            for callback in self._callbacks:
                callback(FakePacket(1))
            raise KeyboardInterrupt

        def stop(self):
            pass

    class FakePacingSender:
        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            pass

    _install_live_collect_modules(
        monkeypatch,
        FakeReceiver,
        FakePacingSender,
        config_overrides={"SEG_WINDOW_SIZE": 2},
    )
    classic_module = sys.modules["classic_detector"]
    base_detector = classic_module.ClassicDetector
    runtime_policy_module = sys.modules["runtime_policy"]
    base_runtime_policy = runtime_policy_module.RuntimeMotionPolicy

    class CapturingClassicDetector(base_detector):
        windows = []

        def __init__(self, **kwargs):
            self.__class__.windows.append(int(kwargs["window_size"]))
            super().__init__(**kwargs)

    class CapturingRuntimeMotionPolicy(base_runtime_policy):
        evaluation_intervals = []

        def __init__(self, **kwargs):
            self.__class__.evaluation_intervals.append(int(kwargs["evaluation_interval"]))
            super().__init__(**kwargs)

    classic_module.ClassicDetector = CapturingClassicDetector
    runtime_policy_module.RuntimeMotionPolicy = CapturingRuntimeMotionPolicy

    host.collect_csi_data(
        _make_live_collect_args(target="192.168.1.29", detector="classic", label=None, pps=42)
    )

    capsys.readouterr()
    assert CapturingClassicDetector.windows == [42, 42]
    assert CapturingRuntimeMotionPolicy.evaluation_intervals == [10, 10]


def test_collect_live_tracks_interleaved_devices_independently(monkeypatch, capsys) -> None:
    fake_csi_utils = ModuleType("tools.lib.csi_io")
    fake_config = ModuleType("config")
    fake_ml_detector = ModuleType("ml_detector")
    fake_runtime_policy = ModuleType("runtime_policy")

    fake_config.DEFAULT_SUBCARRIERS = [12, 14]
    fake_config.SEG_WINDOW_SIZE = 2
    fake_config.ENABLE_LOWPASS_FILTER = False
    fake_config.LOWPASS_CUTOFF = 11.0
    fake_config.ENABLE_HAMPEL_FILTER = True
    fake_config.HAMPEL_WINDOW = 7
    fake_config.HAMPEL_THRESHOLD = 5.0
    fake_config.PUBLISH_INTERVAL = 2
    fake_config.EVALUATION_INTERVAL = 99
    fake_config.MOTION_ON_HITS = 1
    fake_config.MOTION_OFF_HITS = 1

    class FakePacket:
        def __init__(self, seq_num: int, device_id: int):
            self.seq_num = seq_num
            self.device_id = device_id
            self.iq_raw = [seq_num, seq_num + 1, seq_num + 2, seq_num + 3]
            self.chip = "c6" if device_id == 0x11 else "s3"
            self.source_ip = "192.168.1.17" if device_id == 0x11 else "192.168.1.24"
            self.channel = 8 if device_id == 0x11 else 11
            self.rssi_dbm = -47 if device_id == 0x11 else -51

    class FakeReceiver:
        def __init__(self, **kwargs):
            self._callbacks = []
            self.dropped_count = 0
            self.pps = 100

        def add_callback(self, callback):
            self._callbacks.append(callback)

        def run(self, timeout: float = 0, quiet: bool = False):
            packets = [
                FakePacket(1, 0x11),
                FakePacket(2, 0x22),
                FakePacket(3, 0x11),
                FakePacket(4, 0x22),
            ]
            for packet in packets:
                for callback in self._callbacks:
                    callback(packet)
            raise KeyboardInterrupt

        def stop(self):
            pass

    class FakePacingSender:
        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            pass

    class FakeContext:
        def __init__(self):
            self.last_turbulence = 0.0
            self.buffer_count = 0
            self.window_size = 2
            self.buffer_index = 0
            self.turbulence_buffer = []

    class FakeMLDetector:
        instances = []

        def __init__(self, **kwargs):
            self._context = FakeContext()
            self.seen = []
            self.__class__.instances.append(self)

        def process_packet(self, csi_data, subcarriers):
            self.seen.append(int(csi_data[0]))

        def update_state(self):
            probability = float(sum(self.seen)) / 10.0
            state = 1 if probability > 0.5 else 0
            return {"probability": probability, "threshold": 0.5, "state": state}

        def is_ready(self):
            return True

        def _extract_features(self):
            return [float(value) for value in self.seen[-2:]]

    class FakeRuntimeMotionPolicy:
        def __init__(self, **kwargs):
            pass

        def note_packet(self):
            pass

        def should_evaluate(self, should_publish):
            return should_publish

        def apply_state(self, state):
            return state, None

        def after_evaluation(self):
            pass

    fake_csi_utils.CSICollector = object
    fake_csi_utils.CSIReceiver = FakeReceiver
    fake_csi_utils.UdpPacingSender = FakePacingSender
    fake_csi_utils.get_default_bind_host = lambda: "127.0.0.1"
    fake_csi_utils.AdaptivePacingController = AdaptivePacingController
    fake_ml_detector.FEATURE_NAMES = ["f1", "f2"]
    fake_ml_detector.ML_DEFAULT_THRESHOLD = 0.5
    fake_ml_detector.ML_METRIC_SCALE = 1.0
    fake_ml_detector.MLDetector = FakeMLDetector
    fake_runtime_policy.RuntimeMotionPolicy = FakeRuntimeMotionPolicy

    monkeypatch.setitem(sys.modules, "tools.lib.csi_io", fake_csi_utils)
    monkeypatch.setitem(sys.modules, "config", fake_config)
    monkeypatch.setitem(sys.modules, "ml_detector", fake_ml_detector)
    monkeypatch.setitem(sys.modules, "runtime_policy", fake_runtime_policy)

    host.collect_csi_data(
        _make_live_collect_args(target="192.168.1.17,192.168.1.24", detector="ml", label=None)
    )

    output = capsys.readouterr().out
    assert len(FakeMLDetector.instances) == 2
    assert "STATUS: COLLECTING 2/2" in output
    assert "collecting until Ctrl+C" in output
    assert "ip=192.168.1.17 chip=C6 ch=08 rssi=-47" in output
    assert "ip=192.168.1.24 chip=S3 ch=11 rssi=-51" in output
    assert " 80% | mvmt:0.400000 thr:0.500000 | IDLE | 0 pkt/s | drop 33.3%" in output
    assert "120% | mvmt:0.600000 thr:0.500000 | MOTION | 0 pkt/s | drop 33.3%" in output
    assert "mvmt:1.000000" not in output


def test_collect_live_calibrates_classic_per_device(monkeypatch, capsys) -> None:
    fake_csi_utils = ModuleType("tools.lib.csi_io")
    fake_config = ModuleType("config")
    fake_classic_detector = ModuleType("classic_detector")
    fake_ml_detector = ModuleType("ml_detector")
    fake_runtime_policy = ModuleType("runtime_policy")
    fake_threshold = ModuleType("threshold")

    fake_config.DEFAULT_SUBCARRIERS = [12, 14]
    fake_config.SEG_WINDOW_SIZE = 2
    fake_config.CALIBRATION_BUFFER_SIZE = 2
    fake_config.ENABLE_LOWPASS_FILTER = False
    fake_config.LOWPASS_CUTOFF = 11.0
    fake_config.ENABLE_HAMPEL_FILTER = True
    fake_config.HAMPEL_WINDOW = 7
    fake_config.HAMPEL_THRESHOLD = 5.0
    fake_config.PUBLISH_INTERVAL = 1
    fake_config.EVALUATION_INTERVAL = 1
    fake_config.MOTION_ON_HITS = 1
    fake_config.MOTION_OFF_HITS = 1

    class FakePacket:
        def __init__(self, seq_num: int, device_id: int):
            self.seq_num = seq_num
            self.device_id = device_id
            self.iq_raw = [seq_num, seq_num + 1, seq_num + 2, seq_num + 3]
            self.chip = "c6" if device_id == 0x11 else "s3"
            self.source_ip = "192.168.1.17" if device_id == 0x11 else "192.168.1.24"
            self.channel = 8 if device_id == 0x11 else 11
            self.rssi_dbm = -47 if device_id == 0x11 else -51

    class FakeReceiver:
        def __init__(self, **kwargs):
            self._callbacks = []
            self.dropped_count = 0
            self.pps = 100

        def add_callback(self, callback):
            self._callbacks.append(callback)

        def run(self, timeout: float = 0, quiet: bool = False):
            packets = [
                FakePacket(1, 0x11),
                FakePacket(1, 0x22),
                FakePacket(2, 0x11),
                FakePacket(2, 0x22),
                FakePacket(3, 0x11),
                FakePacket(3, 0x22),
                FakePacket(4, 0x11),
                FakePacket(4, 0x22),
            ]
            for packet in packets:
                for callback in self._callbacks:
                    callback(packet)
            raise KeyboardInterrupt

        def stop(self):
            pass

    class FakePacingSender:
        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            pass

    class FakeMLDetector:
        def __init__(self, **kwargs):
            self._threshold = kwargs.get("threshold", 0.5)
            self._context = type("Ctx", (), {"last_turbulence": 0.0, "buffer_count": 0, "window_size": 2, "buffer_index": 0, "turbulence_buffer": []})()

        def get_threshold(self):
            return self._threshold

        def set_threshold(self, threshold):
            self._threshold = threshold
            return True

        def set_cv_normalization(self, enabled):
            pass

        def process_packet(self, csi_data, subcarriers):
            pass

        def update_state(self):
            return {"probability": 0.0, "threshold": self._threshold, "state": 0}

        def reset(self):
            pass

        def is_ready(self):
            return False

    class FakeClassicBaseDetector:
        adaptive_thresholds = []

        def __init__(self, **kwargs):
            self._threshold = kwargs.get("threshold", 1.0)
            self._seen = []
            self._context = type("Ctx", (), {"last_turbulence": 0.0, "buffer_count": 0, "window_size": 2, "buffer_index": 0, "turbulence_buffer": []})()

        def set_cv_normalization(self, enabled):
            pass

        def process_packet(self, csi_data, subcarriers):
            self._seen.append(int(csi_data[0]))
            self._context.last_turbulence = float(self._seen[-1])
            self._context.buffer_count = min(len(self._seen), self._context.window_size)

        def update_state(self):
            metric = float(sum(self._seen[-2:])) if self._seen else 0.0
            state = 1 if metric > self._threshold else 0
            return {"moving_variance": metric, "threshold": self._threshold, "state": state}

        def get_motion_metric(self):
            return float(sum(self._seen[-2:])) if self._seen else 0.0

        def get_threshold(self):
            return self._threshold

        def set_threshold(self, threshold):
            self._threshold = threshold
            return True

        def set_adaptive_threshold(self, threshold):
            self._threshold = threshold
            self.__class__.adaptive_thresholds.append(threshold)

        def reset(self):
            self._seen = []
            self._context.buffer_count = 0

        def is_ready(self):
            return len(self._seen) >= 2

    class FakeClassicDetector(FakeClassicBaseDetector):
        ALGORITHM = "classic"
        BASE_THRESHOLD = 1.0
        STARTUP_THRESHOLD_FACTOR = 1.1
        STARTUP_GATE = True

        def get_name(self):
            return "Classic"

    class FakeRuntimeMotionPolicy:
        def __init__(self, **kwargs):
            pass

        def note_packet(self):
            pass

        def should_evaluate(self, should_publish):
            return True

        def apply_state(self, state):
            return state, None

        def after_evaluation(self):
            pass

        def reset(self):
            pass

    calibration_calls = []

    class FakeStartupThresholdCalibrator:
        def __init__(self, target_packets, auto_factor=1.3, gate_enabled=False):
            self.target_packets = int(target_packets)
            self.auto_factor = float(auto_factor)
            self.gate_enabled = bool(gate_enabled)
            self.packet_count = 0
            self.max_moving_variance = None

        def observe_detector(self, detector, packet_weight=1):
            self.packet_count += packet_weight
            if not detector.is_ready():
                return None
            current_mv = float(detector.get_motion_metric())
            if self.max_moving_variance is None or current_mv > self.max_moving_variance:
                self.max_moving_variance = current_mv
            return current_mv

        def is_complete(self):
            return self.packet_count >= self.target_packets

        def is_successful(self):
            return self.max_moving_variance is not None

        def calculate_threshold(self):
            calibration_calls.append(float(self.max_moving_variance or 0.0))
            return 8.0, "max x 1.3"

    fake_csi_utils.CSICollector = object
    fake_csi_utils.CSIReceiver = FakeReceiver
    fake_csi_utils.UdpPacingSender = FakePacingSender
    fake_csi_utils.get_default_bind_host = lambda: "127.0.0.1"
    fake_csi_utils.AdaptivePacingController = AdaptivePacingController
    fake_ml_detector.FEATURE_NAMES = ["f1", "f2"]
    fake_ml_detector.ML_DEFAULT_THRESHOLD = 0.5
    fake_ml_detector.ML_METRIC_SCALE = 1.0
    fake_ml_detector.MLDetector = FakeMLDetector
    fake_classic_detector.ClassicDetector = FakeClassicDetector
    fake_runtime_policy.RuntimeMotionPolicy = FakeRuntimeMotionPolicy
    fake_threshold.StartupThresholdCalibrator = FakeStartupThresholdCalibrator
    fake_threshold.get_detector_auto_factor = lambda detector: getattr(detector, "STARTUP_THRESHOLD_FACTOR", 1.3)
    fake_threshold.get_detector_startup_gate = lambda detector: bool(getattr(detector, "STARTUP_GATE", False))

    monkeypatch.setitem(sys.modules, "tools.lib.csi_io", fake_csi_utils)
    monkeypatch.setitem(sys.modules, "config", fake_config)
    monkeypatch.setitem(sys.modules, "ml_detector", fake_ml_detector)
    monkeypatch.setitem(sys.modules, "classic_detector", fake_classic_detector)
    monkeypatch.setitem(sys.modules, "runtime_policy", fake_runtime_policy)
    monkeypatch.setitem(sys.modules, "threshold", fake_threshold)

    host.collect_csi_data(
        _make_live_collect_args(target="192.168.1.17,192.168.1.24", detector="classic", label=None, pps=4)
    )

    output = capsys.readouterr().out
    assert "Detector:" in output and "CLASSIC" in output
    assert "STATUS: CALIBRATING" in output
    assert calibration_calls == [7.0, 7.0]
    assert FakeClassicDetector.adaptive_thresholds == [8.0, 8.0]
    assert "thr:8.000000 | IDLE | 0 pkt/s | drop 0.0% | bp:--" in output
    assert "STATUS: COLLECTING 2/2" in output


def test_collect_live_runs_parallel_detectors_per_device(monkeypatch, capsys) -> None:
    class FakePacket:
        def __init__(self, seq_num: int):
            self.seq_num = seq_num
            self.device_id = 0x22
            self.iq_raw = [seq_num, seq_num + 1, seq_num + 2, seq_num + 3]
            self.chip = "c3"
            self.source_ip = "192.168.1.24"
            self.channel = 6
            self.rssi_dbm = -45

    class FakeReceiver:
        def __init__(self, **kwargs):
            self._callbacks = []
            self.dropped_count = 0
            self.pps = 100

        def add_callback(self, callback):
            self._callbacks.append(callback)

        def run(self, timeout: float = 0, quiet: bool = False):
            for seq_num in range(1, 5):
                for callback in self._callbacks:
                    callback(FakePacket(seq_num))
            raise KeyboardInterrupt

        def stop(self):
            pass

    class FakePacingSender:
        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            pass

    _install_live_collect_modules(
        monkeypatch,
        FakeReceiver,
        FakePacingSender,
        config_overrides={"CALIBRATION_BUFFER_SIZE": 2},
    )

    host.collect_csi_data(
        _make_live_collect_args(target="192.168.1.24", detector="classic,ml", label=None, pps=4)
    )

    output = capsys.readouterr().out
    assert "Detector:" in output and "CLASSIC, ML" in output
    assert "STATUS: CALIBRATING 1/1" in output
    assert "STATUS: COLLECTING 1/1" in output
    # One live line per (device, detector) pair.
    assert "ip=192.168.1.24 chip=C3 ch=06 rssi=-45 [classic]" in output
    assert "ip=192.168.1.24 chip=C3 ch=06 rssi=-45 [ml     ]" in output


def test_collect_live_shows_drop_rate_during_calibration(monkeypatch, capsys) -> None:
    class FakePacket:
        def __init__(self, seq_num: int):
            self.seq_num = seq_num
            self.device_id = 0x11
            self.iq_raw = [seq_num, seq_num + 1, seq_num + 2, seq_num + 3]
            self.chip = "s3"
            self.source_ip = "192.168.1.24"
            self.channel = 8
            self.rssi_dbm = -49

    class FakeReceiver:
        def __init__(self, **kwargs):
            self._callbacks = []
            self.dropped_count = 0
            self.pps = 100

        def add_callback(self, callback):
            self._callbacks.append(callback)

        def run(self, timeout: float = 0, quiet: bool = False):
            for seq_num in (1, 3):
                for callback in self._callbacks:
                    callback(FakePacket(seq_num))
            raise KeyboardInterrupt

        def stop(self):
            pass

    class FakePacingSender:
        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            pass

    _install_live_collect_modules(
        monkeypatch,
        FakeReceiver,
        FakePacingSender,
        config_overrides={"CALIBRATION_BUFFER_SIZE": 4, "EVALUATION_INTERVAL": 1},
    )

    host.collect_csi_data(
        _make_live_collect_args(target="192.168.1.24", detector="classic", label=None)
    )

    output = capsys.readouterr().out
    assert "STATUS: CALIBRATING 1/1" in output
    assert "drop 33.3%" in output


def test_collect_live_surfaces_runtime_error(monkeypatch) -> None:
    class FakeReceiver:
        def __init__(self, **kwargs):
            self.dropped_count = 0
            self.pps = 0

        def add_callback(self, callback):
            pass

        def run(self, timeout: float = 0, quiet: bool = False):
            raise RuntimeError("udp failure")

        def stop(self):
            pass

    class FakePacingSender:
        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            pass

    _install_live_collect_modules(monkeypatch, FakeReceiver, FakePacingSender)

    with pytest.raises(SystemExit):
        host.collect_csi_data(_make_live_collect_args(label=None))


def test_collect_live_displays_device_drop_rate(monkeypatch, capsys) -> None:
    class FakePacket:
        def __init__(self, seq_num: int):
            self.seq_num = seq_num
            self.device_id = 0x22
            self.iq_raw = [seq_num, seq_num + 1, seq_num + 2, seq_num + 3]
            self.chip = "s3"
            self.source_ip = "192.168.1.34"
            self.channel = 8
            self.rssi_dbm = -46

    class FakeReceiver:
        def __init__(self, **kwargs):
            self._callbacks = []
            self.dropped_count = 0
            self.pps = 100

        def add_callback(self, callback):
            self._callbacks.append(callback)

        def run(self, timeout: float = 0, quiet: bool = False):
            for seq_num in (1, 2, 5, 6):
                for callback in self._callbacks:
                    callback(FakePacket(seq_num))
            raise KeyboardInterrupt

        def stop(self):
            pass

    class FakePacingSender:
        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            pass

    _install_live_collect_modules(monkeypatch, FakeReceiver, FakePacingSender)

    host.collect_csi_data(
        _make_live_collect_args(target="192.168.1.34", detector="ml", label=None)
    )

    output = capsys.readouterr().out
    assert "ip=192.168.1.34 chip=S3 ch=08 rssi=-46" in output
    assert "drop 33.3%" in output
