"""Tests for ESPectre host-side collect and UI CLI options."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import ModuleType
import builtins

import pytest
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
        "stimulus_target": "192.168.1.15",
        "stimulus_port": 9999,
        "stimulus_rate": 100,
        "reference_every": 20,
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
        "start_delay": 0.0,
        "interactive": False,
        "udp_port": 5001,
        "bind_ip": None,
        "stimulus_target": "192.168.1.15",
        "stimulus_port": 9999,
        "stimulus_rate": 100,
        "reference_every": 0,
        "detector": "mvs",
        "no_save": False,
        "log_features": False,
        "log_turbulence": False,
        "log_only_motion": False,
        "window_tail": 16,
        "contributor": None,
        "description": None,
    }
    args.update(overrides)
    return argparse.Namespace(**args)


def _install_live_collect_modules(monkeypatch, receiver_cls, stimulus_cls, collector_cls=object, config_overrides=None) -> None:
    fake_csi_utils = ModuleType("tools.lib.csi_io")
    fake_config = ModuleType("config")
    fake_ml_detector = ModuleType("ml_detector")
    fake_mvs_detector = ModuleType("mvs_detector")
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
            self._threshold = kwargs.get("threshold", 5.0)

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

    class FakeMVSDetector(FakeMLDetector):
        def update_state(self):
            return {"moving_variance": 0.0, "threshold": self._threshold, "state": 0}

        def get_motion_metric(self):
            return 0.0

        def set_adaptive_threshold(self, threshold):
            self._threshold = threshold

        def get_name(self):
            return "MVS"

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

        def observe_detector(self, detector):
            self.packet_count += 1
            if not detector.is_ready():
                return None
            current_mv = float(detector.get_motion_metric())
            if self.max_moving_variance is None or current_mv > self.max_moving_variance:
                self.max_moving_variance = current_mv
            return current_mv

        def is_complete(self):
            return self.packet_count >= self.target_packets

        def is_extending(self):
            return False

        def is_successful(self):
            return self.max_moving_variance is not None

        def calculate_threshold(self, mode="auto"):
            return 1.5, "max x 1.3"

    fake_csi_utils.CSICollector = collector_cls
    fake_csi_utils.CSIReceiver = receiver_cls
    fake_csi_utils.StimulusSender = stimulus_cls
    fake_csi_utils.get_default_bind_host = lambda: "127.0.0.1"
    fake_ml_detector.FEATURE_NAMES = ["f1", "f2"]
    fake_ml_detector.ML_DEFAULT_THRESHOLD = 5.0
    fake_ml_detector.ML_METRIC_SCALE = 10.0
    fake_ml_detector.MLDetector = FakeMLDetector
    fake_mvs_detector.MVSDetector = FakeMVSDetector
    fake_runtime_policy.RuntimeMotionPolicy = FakeRuntimeMotionPolicy
    fake_threshold.StartupThresholdCalibrator = FakeStartupThresholdCalibrator
    fake_threshold.get_detector_auto_factor = lambda detector: getattr(detector, "STARTUP_THRESHOLD_FACTOR", 1.3)
    fake_threshold.get_detector_startup_gate = lambda detector: bool(getattr(detector, "STARTUP_GATE", False))

    monkeypatch.setitem(sys.modules, "tools.lib.csi_io", fake_csi_utils)
    monkeypatch.setitem(sys.modules, "config", fake_config)
    monkeypatch.setitem(sys.modules, "ml_detector", fake_ml_detector)
    monkeypatch.setitem(sys.modules, "mvs_detector", fake_mvs_detector)
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
            "--stimulus-target",
            "239.1.1.15",
        ]
    )

    assert args.namespace == "collect"
    assert args.samples == 3
    assert args.start_delay == 15.0


def test_micro_collect_alias_is_rejected() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["micro", "collect", "--label", "motion", "--stimulus-target", "192.168.1.15"])


def test_collect_parser_keeps_samples_option() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "collect",
            "--label",
            "motion",
            "--samples",
            "4",
            "--stimulus-target",
            "192.168.1.15",
        ]
    )

    assert args.samples == 4
    assert args.start_delay == 0.0


def test_collect_parser_accepts_comma_separated_stimulus_targets() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "collect",
            "--label",
            "test",
            "--stimulus-target",
            "192.168.1.17,192.168.1.24,192.168.1.29",
        ]
    )

    assert args.stimulus_target == "192.168.1.17,192.168.1.24,192.168.1.29"


def test_collect_parser_accepts_live_options() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "collect",
            "--stimulus-target",
            "192.168.1.15",
            "--label",
            "test",
            "--duration",
            "45",
            "--description",
            "live collect ML, idle-motion-idle",
        ]
    )

    assert args.namespace == "collect"
    assert args.detector == "mvs"
    assert args.label == "test"
    assert args.duration == 45.0
    assert args.description == "live collect ML, idle-motion-idle"


def test_collect_parser_accepts_detector_choice_and_no_save() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "collect",
            "--stimulus-target",
            "192.168.1.15",
            "--detector",
            "mvs",
            "--no-save",
        ]
    )

    assert args.namespace == "collect"
    assert args.detector == "mvs"
    assert args.no_save is True
    assert args.duration is None


def test_collect_parser_accepts_comma_separated_detectors() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "collect",
            "--stimulus-target",
            "192.168.1.15",
            "--detector",
            "mvs,l1_delta",
            "--no-save",
        ]
    )

    assert args.namespace == "collect"
    assert args.detector == "mvs,l1_delta"


def test_collect_live_rejects_unknown_detector(monkeypatch, capsys) -> None:
    class FakeReceiver:
        def __init__(self, **kwargs):
            pass

    class FakeStimulusSender:
        def __init__(self, **kwargs):
            pass

    _install_live_collect_modules(monkeypatch, FakeReceiver, FakeStimulusSender)

    with pytest.raises(SystemExit):
        host.collect_csi_data(_make_live_collect_args(detector="mvs,bogus", no_save=True))

    output = capsys.readouterr().out
    assert "Unsupported detector(s): bogus" in output


def test_detect_command_is_rejected() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["detect", "--stimulus-target", "192.168.1.15"])


def test_ui_parser_accepts_ble_interface() -> None:
    parser = build_parser()

    args = parser.parse_args(["ui", "ble"])

    assert args.namespace == "ui"
    assert args.interface == "ble"


def test_micro_ui_alias_is_rejected() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["micro", "ui", "ble"])


def test_ui_parser_accepts_theremin_interface() -> None:
    parser = build_parser()

    args = parser.parse_args(["ui", "theremin"])

    assert args.namespace == "ui"
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

    def raise_browser(_url: str) -> None:
        raise RuntimeError("browser blocked")

    monkeypatch.setattr(host.webbrowser, "open", raise_browser)
    host.open_web_ui("mqtt")

    output = capsys.readouterr().out
    assert "unknown web UI" in output
    assert "not found" in output
    assert "Error opening browser" in output


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
    fake_csi_utils.StimulusSender = object
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
    fake_csi_utils.StimulusSender = object
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
    fake_csi_utils.StimulusSender = object
    fake_csi_utils.get_dataset_stats = lambda: {"labels": {}, "total_samples": 0}
    fake_csi_utils.get_default_bind_host = lambda: "127.0.0.1"
    monkeypatch.setitem(sys.modules, "tools.lib.csi_io", fake_csi_utils)

    with pytest.raises(SystemExit):
        host.collect_csi_data(_make_collect_args(stimulus_target=None))

    with pytest.raises(SystemExit):
        host.collect_csi_data(_make_collect_args(start_delay=-1))


def test_collect_csi_data_handles_interrupt_and_runtime_error(monkeypatch) -> None:
    events: list[str] = []
    fake_csi_utils = ModuleType("tools.lib.csi_io")

    class InterruptCollector:
        def __init__(self, **kwargs):
            pass

        def collect_timed(self, duration: float, num_samples: int):
            raise KeyboardInterrupt

        def collect_interactive(self, num_samples: int, duration: float):
            raise RuntimeError("boom")

    class FakeStimulusSender:
        def __init__(self, **kwargs):
            pass

        def start(self):
            events.append("start")

        def stop(self):
            events.append("stop")

    fake_csi_utils.CSICollector = InterruptCollector
    fake_csi_utils.StimulusSender = FakeStimulusSender
    fake_csi_utils.get_dataset_stats = lambda: {"labels": {}, "total_samples": 0}
    fake_csi_utils.get_default_bind_host = lambda: "127.0.0.1"
    monkeypatch.setitem(sys.modules, "tools.lib.csi_io", fake_csi_utils)
    monkeypatch.setattr(host, "_wait_before_collection", lambda delay: None)

    host.collect_csi_data(_make_collect_args())
    assert events == ["start", "stop"]

    with pytest.raises(SystemExit):
        host.collect_csi_data(_make_collect_args(interactive=True))


def test_collect_applies_start_delay_before_starting_stimulus(monkeypatch) -> None:
    events: list[object] = []
    fake_csi_utils = ModuleType("tools.lib.csi_io")

    class FakeCollector:
        def __init__(self, **kwargs):
            events.append(("collector_init", kwargs["label"], kwargs["bind_host"], kwargs["expected_device_count"]))

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

    monkeypatch.setitem(sys.modules, "tools.lib.csi_io", fake_csi_utils)
    monkeypatch.setattr(host, "_wait_before_collection", lambda delay: events.append(("delay", delay)))

    host.collect_csi_data(_make_collect_args(stimulus_target="192.168.1.17,192.168.1.24,192.168.1.29"))

    assert ("delay", 5.0) in events
    assert ("collect_timed", 10.0, 2) in events
    assert ("collector_init", "static_presence", "127.0.0.1", 3) in events
    assert ("sender_init", ["192.168.1.17", "192.168.1.24", "192.168.1.29"], 20) in events
    assert events.index(("delay", 5.0)) < events.index("start")
    assert events[-1] == "stop"


def test_collect_live_saves_raw_packets_with_collector(monkeypatch, capsys) -> None:
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
            events.append(("collector_init", kwargs["label"], kwargs["description"], kwargs["expected_device_count"]))

        def save_samples_by_device(self, packets):
            events.append(("save_sample", [p.seq_num for p in packets]))
            return [Path("test_c3_64sc_dev0000000000abc123_20260630_120000_000001_0001.npz")]

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

    monkeypatch.setitem(sys.modules, "tools.lib.csi_io", fake_csi_utils)
    monkeypatch.setitem(sys.modules, "config", fake_config)
    monkeypatch.setitem(sys.modules, "ml_detector", fake_ml_detector)
    monkeypatch.setitem(sys.modules, "runtime_policy", fake_runtime_policy)
    monkeypatch.setattr(host.time, "monotonic", lambda: clock["now"])

    host.collect_csi_data(
        _make_live_collect_args(
            stimulus_target="192.168.1.29",
            label="test",
            description="live collect ML, idle-motion-idle",
            detector="ml",
        )
    )

    output = capsys.readouterr().out
    assert ("collector_init", "test", "live collect ML, idle-motion-idle", 1) in events
    assert ("sender_init", ["192.168.1.29"]) in events
    assert ("save_sample", [4, 5]) in events
    assert "STATUS: RECORDING 1/1" in output
    assert "recording until Ctrl+C" in output
    assert "stop" in events
    assert "receiver_stop" in events


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

    monkeypatch.setitem(sys.modules, "tools.lib.csi_io", fake_csi_utils)
    monkeypatch.setitem(sys.modules, "config", fake_config)
    monkeypatch.setitem(sys.modules, "ml_detector", fake_ml_detector)
    monkeypatch.setitem(sys.modules, "runtime_policy", fake_runtime_policy)
    monkeypatch.setattr(host.time, "monotonic", lambda: clock["now"])

    host.collect_csi_data(
        _make_live_collect_args(
            stimulus_target="192.168.1.29",
            label="test",
            duration=10,
            description="interrupted run",
            detector="ml",
        )
    )

    output = capsys.readouterr().out
    assert ("collector_init", "test") in events
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

    class FakeStimulusSender:
        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            pass

    _install_live_collect_modules(monkeypatch, FakeReceiver, FakeStimulusSender)

    with pytest.raises(SystemExit):
        host.collect_csi_data(_make_live_collect_args(duration=0))

    with pytest.raises(SystemExit):
        host.collect_csi_data(_make_live_collect_args(label=None))

    host.collect_csi_data(_make_live_collect_args(label=None, no_save=True, duration=5))


def test_collect_live_handles_import_failure(monkeypatch) -> None:
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        blocked = {
            "tools.lib.csi_io",
            "config",
            "ml_detector",
            "mvs_detector",
            "runtime_policy",
            "threshold",
            "src.config",
            "src.ml_detector",
            "src.mvs_detector",
            "src.runtime_policy",
            "src.threshold",
        }
        if name in blocked:
            raise ImportError(f"blocked import: {name}")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(SystemExit):
        host.collect_csi_data(_make_live_collect_args())


def test_collect_live_logs_features_and_handles_save_without_packets(monkeypatch, capsys) -> None:
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
    fake_config.SEG_THRESHOLD = "auto"

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

    class FakeStimulusSender:
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
            return {"probability": 4.0, "threshold": 5.0, "state": 1}

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
    fake_csi_utils.StimulusSender = FakeStimulusSender
    fake_csi_utils.get_default_bind_host = lambda: "127.0.0.1"
    fake_ml_detector.FEATURE_NAMES = ["f1", "f2"]
    fake_ml_detector.ML_DEFAULT_THRESHOLD = 5.0
    fake_ml_detector.ML_METRIC_SCALE = 10.0
    fake_ml_detector.MLDetector = FakeMLDetector
    fake_runtime_policy.RuntimeMotionPolicy = FakeRuntimeMotionPolicy

    monkeypatch.setitem(sys.modules, "tools.lib.csi_io", fake_csi_utils)
    monkeypatch.setitem(sys.modules, "config", fake_config)
    monkeypatch.setitem(sys.modules, "ml_detector", fake_ml_detector)
    monkeypatch.setitem(sys.modules, "runtime_policy", fake_runtime_policy)

    host.collect_csi_data(
        _make_live_collect_args(
            detector="ml",
            log_features=True,
            log_turbulence=True,
            log_only_motion=True,
            description="feature run",
        )
    )

    output = capsys.readouterr().out
    assert "Threshold:" in output and "5.0" in output
    assert "Low-pass:" in output and "ON" in output
    assert "Save:" in output and "label=test duration=until Ctrl+C" in output
    assert "STATUS: STABILIZING 1/1" in output
    assert "ip=192.168.1.29 chip=C6 ch=08 rssi=-47" in output
    assert " 80% | mvmt:4.000000 thr:5.000000 | MOTION | 0 pkt/s" in output
    assert "turbulence: raw=0.7500 filtered=0.5000" in output
    assert "tail[2]:" in output
    assert "features: f1=0.1000 f2=0.2000" in output
    assert "No live capture packets received; nothing saved" in output


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
    fake_config.SEG_THRESHOLD = 5.0

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

    class FakeStimulusSender:
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
            probability = float(sum(self.seen))
            state = 1 if probability > 5.0 else 0
            return {"probability": probability, "threshold": 5.0, "state": state}

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
    fake_csi_utils.StimulusSender = FakeStimulusSender
    fake_csi_utils.get_default_bind_host = lambda: "127.0.0.1"
    fake_ml_detector.FEATURE_NAMES = ["f1", "f2"]
    fake_ml_detector.ML_DEFAULT_THRESHOLD = 5.0
    fake_ml_detector.ML_METRIC_SCALE = 10.0
    fake_ml_detector.MLDetector = FakeMLDetector
    fake_runtime_policy.RuntimeMotionPolicy = FakeRuntimeMotionPolicy

    monkeypatch.setitem(sys.modules, "tools.lib.csi_io", fake_csi_utils)
    monkeypatch.setitem(sys.modules, "config", fake_config)
    monkeypatch.setitem(sys.modules, "ml_detector", fake_ml_detector)
    monkeypatch.setitem(sys.modules, "runtime_policy", fake_runtime_policy)

    host.collect_csi_data(_make_live_collect_args(stimulus_target="192.168.1.17,192.168.1.24", no_save=True, detector="ml"))

    output = capsys.readouterr().out
    assert len(FakeMLDetector.instances) == 2
    assert "STATUS: COLLECTING 2/2" in output
    assert "collecting until Ctrl+C" in output
    assert "ip=192.168.1.17 chip=C6 ch=08 rssi=-47" in output
    assert "ip=192.168.1.24 chip=S3 ch=11 rssi=-51" in output
    assert " 80% | mvmt:4.000000 thr:5.000000 | IDLE | 0 pkt/s" in output
    assert "120% | mvmt:6.000000 thr:5.000000 | MOTION | 0 pkt/s" in output
    assert "mvmt:10.000000" not in output


def test_collect_live_calibrates_mvs_per_device(monkeypatch, capsys) -> None:
    fake_csi_utils = ModuleType("tools.lib.csi_io")
    fake_config = ModuleType("config")
    fake_ml_detector = ModuleType("ml_detector")
    fake_mvs_detector = ModuleType("mvs_detector")
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
    fake_config.SEG_THRESHOLD = "auto"

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

    class FakeStimulusSender:
        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            pass

    class FakeMLDetector:
        def __init__(self, **kwargs):
            self._threshold = kwargs.get("threshold", 5.0)
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

    class FakeMVSDetector:
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

        def observe_detector(self, detector):
            self.packet_count += 1
            if not detector.is_ready():
                return None
            current_mv = float(detector.get_motion_metric())
            if self.max_moving_variance is None or current_mv > self.max_moving_variance:
                self.max_moving_variance = current_mv
            return current_mv

        def is_complete(self):
            return self.packet_count >= self.target_packets

        def is_extending(self):
            return False

        def is_successful(self):
            return self.max_moving_variance is not None

        def calculate_threshold(self, mode="auto"):
            calibration_calls.append((float(self.max_moving_variance or 0.0), mode))
            return 8.0, "max x 1.3"

    fake_csi_utils.CSICollector = object
    fake_csi_utils.CSIReceiver = FakeReceiver
    fake_csi_utils.StimulusSender = FakeStimulusSender
    fake_csi_utils.get_default_bind_host = lambda: "127.0.0.1"
    fake_ml_detector.FEATURE_NAMES = ["f1", "f2"]
    fake_ml_detector.ML_DEFAULT_THRESHOLD = 5.0
    fake_ml_detector.ML_METRIC_SCALE = 10.0
    fake_ml_detector.MLDetector = FakeMLDetector
    fake_mvs_detector.MVSDetector = FakeMVSDetector
    fake_runtime_policy.RuntimeMotionPolicy = FakeRuntimeMotionPolicy
    fake_threshold.StartupThresholdCalibrator = FakeStartupThresholdCalibrator
    fake_threshold.get_detector_auto_factor = lambda detector: getattr(detector, "STARTUP_THRESHOLD_FACTOR", 1.3)
    fake_threshold.get_detector_startup_gate = lambda detector: bool(getattr(detector, "STARTUP_GATE", False))

    monkeypatch.setitem(sys.modules, "tools.lib.csi_io", fake_csi_utils)
    monkeypatch.setitem(sys.modules, "config", fake_config)
    monkeypatch.setitem(sys.modules, "ml_detector", fake_ml_detector)
    monkeypatch.setitem(sys.modules, "mvs_detector", fake_mvs_detector)
    monkeypatch.setitem(sys.modules, "runtime_policy", fake_runtime_policy)
    monkeypatch.setitem(sys.modules, "threshold", fake_threshold)

    host.collect_csi_data(_make_live_collect_args(stimulus_target="192.168.1.17,192.168.1.24", detector="mvs", no_save=True))

    output = capsys.readouterr().out
    assert "Detector:" in output and "MVS" in output
    assert "STATUS: CALIBRATING" in output
    assert calibration_calls == [(3.0, "auto"), (3.0, "auto")]
    assert FakeMVSDetector.adaptive_thresholds == [8.0, 8.0]
    assert " 87% | mvmt:7.000000 thr:8.000000 | IDLE | 0 pkt/s" in output
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

    class FakeStimulusSender:
        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            pass

    _install_live_collect_modules(
        monkeypatch,
        FakeReceiver,
        FakeStimulusSender,
        config_overrides={"CALIBRATION_BUFFER_SIZE": 2},
    )

    host.collect_csi_data(
        _make_live_collect_args(stimulus_target="192.168.1.24", detector="mvs,ml", no_save=True)
    )

    output = capsys.readouterr().out
    assert "Detector:" in output and "MVS, ML" in output
    assert "STATUS: CALIBRATING 1/1" in output
    assert "STATUS: COLLECTING 1/1" in output
    # One live line per (device, detector) pair.
    assert "ip=192.168.1.24 chip=C3 ch=06 rssi=-45 [mvs]" in output
    assert "ip=192.168.1.24 chip=C3 ch=06 rssi=-45 [ml ]" in output


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

    class FakeStimulusSender:
        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            pass

    _install_live_collect_modules(monkeypatch, FakeReceiver, FakeStimulusSender)

    with pytest.raises(SystemExit):
        host.collect_csi_data(_make_live_collect_args(no_save=True))
