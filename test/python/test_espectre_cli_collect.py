"""Tests for `espectre micro collect` CLI options."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import ModuleType

from espectre_cli.app import build_parser
from espectre_cli import host


def _make_collect_args(**overrides) -> argparse.Namespace:
    args = {
        "info": False,
        "label": "static_presence",
        "samples": 2,
        "duration": 10.0,
        "start_delay": 5.0,
        "interactive": False,
        "udp_port": 5001,
        "bind_ip": None,
        "streamer_ip": "192.168.1.15",
        "stimulus_port": 9999,
        "stimulus_rate": 100,
        "reference_every": 20,
        "contributor": None,
        "description": None,
    }
    args.update(overrides)
    return argparse.Namespace(**args)


def _make_detect_args(**overrides) -> argparse.Namespace:
    args = {
        "udp_port": 5001,
        "bind_ip": None,
        "streamer_ip": "192.168.1.15",
        "stimulus_port": 9999,
        "stimulus_rate": 100,
        "reference_every": 0,
        "log_features": False,
        "log_turbulence": False,
        "log_only_motion": False,
        "window_tail": 16,
        "capture_label": None,
        "capture_duration": None,
        "contributor": None,
        "description": None,
    }
    args.update(overrides)
    return argparse.Namespace(**args)


def test_collect_parser_accepts_count_alias() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "micro",
            "collect",
            "--label",
            "static_presence",
            "--duration",
            "10",
            "--count",
            "3",
            "--start-delay",
            "15",
            "--streamer-ip",
            "192.168.1.15",
        ]
    )

    assert args.namespace == "micro"
    assert args.micro_command == "collect"
    assert args.samples == 3
    assert args.start_delay == 15.0


def test_collect_parser_keeps_samples_option() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "micro",
            "collect",
            "--label",
            "motion",
            "--samples",
            "4",
            "--streamer-ip",
            "192.168.1.15",
        ]
    )

    assert args.samples == 4
    assert args.start_delay == 0.0


def test_detect_parser_accepts_capture_options() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "micro",
            "detect",
            "--streamer-ip",
            "192.168.1.15",
            "--capture-label",
            "test",
            "--capture-duration",
            "45",
            "--description",
            "live detect ML, idle-motion-idle",
        ]
    )

    assert args.namespace == "micro"
    assert args.micro_command == "detect"
    assert args.capture_label == "test"
    assert args.capture_duration == 45.0
    assert args.description == "live detect ML, idle-motion-idle"


def test_ui_parser_accepts_ble_interface() -> None:
    parser = build_parser()

    args = parser.parse_args(["micro", "ui", "ble"])

    assert args.namespace == "micro"
    assert args.micro_command == "ui"
    assert args.interface == "ble"


def test_ui_parser_accepts_theremin_interface() -> None:
    parser = build_parser()

    args = parser.parse_args(["micro", "ui", "theremin"])

    assert args.namespace == "micro"
    assert args.micro_command == "ui"
    assert args.interface == "theremin"


def test_open_web_ui_opens_ble_file(monkeypatch, tmp_path) -> None:
    ble_file = tmp_path / "espectre-ble.html"
    ble_file.write_text("<html></html>", encoding="utf-8")
    opened_urls: list[str] = []

    monkeypatch.setattr(
        host,
        "_WEB_UI_FILES",
        {
            "mqtt": tmp_path / "espectre-mqtt.html",
            "ble": ble_file,
            "theremin": tmp_path / "espectre-theremin.html",
        },
    )
    monkeypatch.setattr(host.webbrowser, "open", lambda url: opened_urls.append(url))

    host.open_web_ui("ble")

    assert opened_urls == [ble_file.absolute().as_uri()]


def test_open_web_ui_opens_theremin_file(monkeypatch, tmp_path) -> None:
    theremin_file = tmp_path / "espectre-theremin.html"
    theremin_file.write_text("<html></html>", encoding="utf-8")
    opened_urls: list[str] = []

    monkeypatch.setattr(
        host,
        "_WEB_UI_FILES",
        {
            "mqtt": tmp_path / "espectre-mqtt.html",
            "ble": tmp_path / "espectre-ble.html",
            "theremin": theremin_file,
        },
    )
    monkeypatch.setattr(host.webbrowser, "open", lambda url: opened_urls.append(url))

    host.open_web_ui("theremin")

    assert opened_urls == [theremin_file.absolute().as_uri()]


def test_collect_applies_start_delay_before_starting_stimulus(monkeypatch) -> None:
    events: list[object] = []
    fake_csi_utils = ModuleType("tools.csi_utils")

    class FakeCollector:
        def __init__(self, **kwargs):
            events.append(("collector_init", kwargs["label"], kwargs["bind_host"]))

        def collect_timed(self, duration: float, num_samples: int):
            events.append(("collect_timed", duration, num_samples))
            return [Path("sample_1.npz"), Path("sample_2.npz")]

        def collect_interactive(self, num_samples: int, duration: float):
            events.append(("collect_interactive", duration, num_samples))
            return []

    class FakeStimulusSender:
        def __init__(self, **kwargs):
            events.append(("sender_init", kwargs["target_host"], kwargs["reference_every"]))

        def start(self):
            events.append("start")

        def stop(self):
            events.append("stop")

    fake_csi_utils.CSICollector = FakeCollector
    fake_csi_utils.StimulusSender = FakeStimulusSender
    fake_csi_utils.get_dataset_stats = lambda: {"labels": {}, "total_samples": 0}
    fake_csi_utils.get_default_bind_host = lambda: "127.0.0.1"

    monkeypatch.setitem(sys.modules, "tools.csi_utils", fake_csi_utils)
    monkeypatch.setattr(host, "_wait_before_collection", lambda delay: events.append(("delay", delay)))

    host.collect_csi_data(_make_collect_args())

    assert ("delay", 5.0) in events
    assert ("collect_timed", 10.0, 2) in events
    assert events.index(("delay", 5.0)) < events.index("start")
    assert events[-1] == "stop"


def test_detect_capture_saves_raw_packets_with_collector(monkeypatch) -> None:
    events: list[object] = []
    fake_csi_utils = ModuleType("tools.csi_utils")
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
            self.iq_raw = [seq_num, seq_num + 1, seq_num + 2, seq_num + 3]

    class FakeCollector:
        def __init__(self, **kwargs):
            events.append(("collector_init", kwargs["label"], kwargs["description"]))

        def save_sample(self, packets):
            events.append(("save_sample", [p.seq_num for p in packets]))
            return Path("test_c3_64sc_20260630_120000.npz")

    class FakeReceiver:
        def __init__(self, **kwargs):
            self._callbacks = []
            self.dropped_count = 0
            self.pps = 100

        def add_callback(self, callback):
            self._callbacks.append(callback)

        def run(self, timeout: float = 0, quiet: bool = False):
            for packet in [FakePacket(1), FakePacket(2)]:
                for callback in self._callbacks:
                    callback(packet)
            raise KeyboardInterrupt

        def stop(self):
            events.append("receiver_stop")

    class FakeStimulusSender:
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
            return {"probability": 0.0, "threshold": 5.0, "state": 0}

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
    fake_csi_utils.StimulusSender = FakeStimulusSender
    fake_csi_utils.get_default_bind_host = lambda: "127.0.0.1"
    fake_ml_detector.FEATURE_NAMES = ["a", "b"]
    fake_ml_detector.ML_DEFAULT_THRESHOLD = 5.0
    fake_ml_detector.ML_METRIC_SCALE = 10.0
    fake_ml_detector.MLDetector = FakeMLDetector
    fake_runtime_policy.RuntimeMotionPolicy = FakeRuntimeMotionPolicy

    monkeypatch.setitem(sys.modules, "tools.csi_utils", fake_csi_utils)
    monkeypatch.setitem(sys.modules, "config", fake_config)
    monkeypatch.setitem(sys.modules, "ml_detector", fake_ml_detector)
    monkeypatch.setitem(sys.modules, "runtime_policy", fake_runtime_policy)

    host.detect_live_motion(
        _make_detect_args(
            capture_label="test",
            description="live detect ML, idle-motion-idle",
        )
    )

    assert ("collector_init", "test", "live detect ML, idle-motion-idle") in events
    assert ("save_sample", [1, 2]) in events
    assert "stop" in events
    assert "receiver_stop" in events
