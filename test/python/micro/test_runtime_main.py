# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Micro-ESPectre runtime startup contracts: helpers and fail-closed boot paths."""

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[3] / "src/python/micro_espectre"


class StopBoot(Exception):
    """Raised by fakes to end main() once the path under test is decided."""


@pytest.fixture
def runtime(monkeypatch):
    package = ModuleType("src")
    package.__path__ = [str(ROOT)]
    monkeypatch.setitem(sys.modules, "src", package)
    for name in ("config", "device_utils", "detector_interface", "runtime_motion_policy",
                 "console_output", "threshold"):
        module = importlib.import_module(name)
        monkeypatch.setitem(sys.modules, "src." + name, module)
        setattr(package, name, module)
    monkeypatch.setitem(sys.modules, "src.temporal_csi_sampler",
                        importlib.import_module("temporal_csi_sampler"))
    monkeypatch.setitem(sys.modules, "src.wifi_bootstrap", SimpleNamespace(
        cleanup_wifi=Mock(), connect_wifi=Mock(), print_wifi_status=Mock(), recover_wifi=Mock(),
    ))
    spec = importlib.util.spec_from_file_location("runtime_main_under_test", ROOT / "runtime_main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "time", SimpleNamespace(
        ticks_ms=lambda: 0, ticks_us=lambda: 0, ticks_diff=lambda a, b: a - b,
        sleep=lambda _: None, sleep_us=lambda _: None,
    ))
    monkeypatch.setattr(module, "gc", SimpleNamespace(
        collect=lambda: None, mem_free=lambda: 1, mem_alloc=lambda: 1,
    ))
    monkeypatch.setattr(module, "print_log", Mock())
    return module


@pytest.mark.parametrize("machine, expected", [
    ("ESP32S3 module with ESP32-S3", "S3"),
    ("ESP32C6 module", "C6"),
    ("ESP32C3 module", "C3"),
    ("ESP32C5 module", "C5"),
    ("ESP32S2 module", "S2"),
    ("Generic ESP32 module", "ESP32"),
    ("RP2040", "RP2040"),
])
def test_chip_type_prefers_the_specific_variant(runtime, monkeypatch, machine, expected):
    monkeypatch.setattr(runtime.os, "uname", lambda: SimpleNamespace(machine=machine))

    assert runtime.get_chip_type() == expected


def test_capture_context_follows_the_active_channel(runtime):
    runtime.g_state.chip_type = "S3"
    wlan = SimpleNamespace(config=lambda key: 6)

    runtime.refresh_csi_capture_context(wlan)

    assert runtime.g_state.current_channel == 6
    assert runtime.g_state.csi_capture_profile == runtime.select_csi_capture_profile("S3", 6)


def test_capture_context_survives_an_unreadable_channel(runtime):
    runtime.g_state.chip_type = "S3"

    def unreadable(_key):
        raise OSError("not associated")

    runtime.refresh_csi_capture_context(SimpleNamespace(config=unreadable))

    assert runtime.g_state.current_channel == 0


def test_capture_context_reports_a_profile_change(runtime, monkeypatch):
    runtime.g_state.chip_type = "S3"
    monkeypatch.setattr(runtime, "select_csi_capture_profile", lambda chip, channel: "lltf20")
    runtime.g_state.csi_capture_profile = "ht20"
    wlan = SimpleNamespace(config=lambda key: 1)

    assert runtime.refresh_csi_capture_context(wlan) is True
    assert runtime.refresh_csi_capture_context(wlan) is False


def test_recalibration_reuses_the_threshold_only_for_the_calibrated_setup(runtime):
    detector = SimpleNamespace(get_threshold=lambda: 0.42)
    runtime.g_state.current_channel = 6
    runtime.g_state.csi_capture_profile = "ht20"

    runtime.g_state.calibrated_setup = None
    assert runtime.recalibration_reference_threshold(detector) is None
    runtime.g_state.calibrated_setup = (11, "ht20")
    assert runtime.recalibration_reference_threshold(detector) is None
    runtime.g_state.calibrated_setup = (6, "lltf20")
    assert runtime.recalibration_reference_threshold(detector) is None
    runtime.g_state.calibrated_setup = (6, "ht20")
    assert runtime.recalibration_reference_threshold(detector) == 0.42


def test_heap_snapshot_reports_idf_regions_when_available(runtime, monkeypatch, capsys):
    esp32 = SimpleNamespace(
        HEAP_DATA=1,
        idf_heap_info=lambda kind: [(0, 100, 60, 10), (1, 50, 40, 5)],
    )
    monkeypatch.setitem(sys.modules, "esp32", esp32)

    runtime.print_heap("boot")

    line = capsys.readouterr().out
    assert "[MEM] boot:" in line
    assert "idf_free=150 idf_largest=60 idf_min=15" in line


def test_heap_snapshot_works_without_the_idf_module(runtime, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "esp32", None)

    runtime.collect_and_print_heap("boot")

    assert "idf_free" not in capsys.readouterr().out


@pytest.mark.parametrize("enabled, start_results, expected_starts", [
    (False, [], 0),
    (True, [True], 1),
    (True, [False, True], 2),
])
def test_traffic_generator_restart_retries_once(runtime, monkeypatch, enabled, start_results, expected_starts):
    monkeypatch.setattr(runtime.config, "TRAFFIC_GENERATOR_ENABLED", enabled, raising=False)
    traffic_gen = Mock()
    traffic_gen.start.side_effect = start_results

    runtime.restart_traffic_generator(traffic_gen)

    assert traffic_gen.start.call_count == expected_starts
    if expected_starts == 2:
        runtime.print_log.assert_any_call("WARN", "Failed to restart traffic generator, retrying...")


def test_traffic_generator_restart_ignores_a_missing_generator(runtime):
    runtime.restart_traffic_generator(None)


def test_unknown_detector_algorithm_is_rejected(runtime, monkeypatch):
    def unknown(_name):
        raise ValueError("unknown")

    monkeypatch.setattr(runtime, "load_detector_class", unknown)

    with pytest.raises(ValueError, match="Unsupported Micro detector: nope"):
        runtime.create_detector("nope", 100)


@pytest.mark.parametrize("backend", [None, "python"])
def test_detector_must_use_the_shared_core_backend(runtime, monkeypatch, backend):
    class Detector:
        def __init__(self, **_kwargs):
            pass

        def get_backend(self):
            return backend

    monkeypatch.setattr(runtime, "load_detector_class", lambda _name: Detector)

    with pytest.raises(RuntimeError, match="espectre_core detector backend"):
        runtime.create_detector("lightweight", 100)


def test_core_backed_detector_is_configured_from_the_runtime_config(runtime, monkeypatch):
    captured = {}

    class Detector:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def get_backend(self):
            return "espectre_core"

    monkeypatch.setattr(runtime, "load_detector_class", lambda _name: Detector)

    detector = runtime.create_detector("lightweight", 128)

    assert isinstance(detector, Detector)
    assert captured["window_size"] == 128
    assert captured["enable_lowpass"] == runtime.config.ENABLE_LOWPASS_FILTER
    assert captured["hampel_window"] == runtime.config.HAMPEL_WINDOW


@pytest.fixture
def boot(runtime, monkeypatch):
    """Drive main() with a stubbed network, traffic generator, and CSI source."""
    runtime.g_state.__init__()
    monkeypatch.setattr(runtime, "get_chip_type", lambda: "S3")
    monkeypatch.setattr(runtime, "refresh_csi_capture_context", lambda _wlan: False)
    traffic_gen = Mock()
    traffic_gen.start.return_value = True
    traffic_gen.get_mode.return_value = "ping"
    traffic_gen_module = SimpleNamespace(TrafficGenerator=lambda *_a, **_k: traffic_gen)
    monkeypatch.setitem(sys.modules, "src.traffic_generator", traffic_gen_module)
    machine = SimpleNamespace(reset=Mock(side_effect=StopBoot))
    monkeypatch.setitem(sys.modules, "machine", machine)
    monkeypatch.setattr(runtime.config, "TRAFFIC_GENERATOR_ENABLED", True, raising=False)
    return SimpleNamespace(runtime=runtime, traffic_gen=traffic_gen, machine=machine, wlan=object())


def test_boot_reboots_when_the_traffic_generator_cannot_start(boot):
    boot.traffic_gen.start.return_value = False

    with pytest.raises(StopBoot):
        boot.runtime.main(boot.wlan)

    boot.machine.reset.assert_called_once()
    assert boot.runtime.g_state.chip_type == "S3"


def test_boot_exits_when_no_csi_arrives_after_every_retry(boot, monkeypatch):
    monkeypatch.setattr(boot.runtime, "csi_read_frame", lambda *_args: None)

    with pytest.raises(SystemExit) as exit_info:
        boot.runtime.main(boot.wlan)

    assert exit_info.value.code == 1
    # One initial start plus one restart for each of the two failed probes.
    assert boot.traffic_gen.start.call_count == 3
    assert boot.traffic_gen.stop.call_count == 2


def test_boot_marks_c6_phy_metadata_as_missing(boot, monkeypatch):
    monkeypatch.setattr(boot.runtime, "get_chip_type", lambda: "C6")
    boot.traffic_gen.start.return_value = False

    with pytest.raises(StopBoot):
        boot.runtime.main(boot.wlan)

    assert boot.runtime.g_state.csi_phy_metadata_missing is True


def test_external_traffic_must_provide_advancing_timestamps(boot, monkeypatch):
    monkeypatch.setattr(boot.runtime.config, "TRAFFIC_GENERATOR_ENABLED", False, raising=False)
    monkeypatch.setattr(boot.runtime, "csi_read_frame", lambda *_args: [0, 6, 0, 0, 5000, b""])

    with pytest.raises(RuntimeError, match="advancing timestamps"):
        boot.runtime.main(boot.wlan)


def test_boot_stops_when_the_socket_cannot_reopen_after_detector_allocation(boot, monkeypatch):
    runtime = boot.runtime
    monkeypatch.setattr(runtime, "csi_read_frame", lambda *_args: [0, 6, 0, 0, 1, b""])
    detector = Mock()
    monkeypatch.setattr(runtime, "create_detector", lambda *_args: detector)
    boot.traffic_gen.start.side_effect = [True, False]
    cleanup = Mock()
    monkeypatch.setattr(runtime, "cleanup_wifi", cleanup)

    with pytest.raises(RuntimeError, match="restart failed after detector initialization"):
        runtime.main(boot.wlan)

    cleanup.assert_called_once_with(boot.wlan)
    detector.set_minimum_valid_samples.assert_called_once()


@pytest.mark.parametrize("running", [True, False])
def test_boot_stops_when_startup_calibration_fails(boot, monkeypatch, running):
    runtime = boot.runtime
    monkeypatch.setattr(runtime, "csi_read_frame", lambda *_args: [0, 6, 0, 0, 1, b""])
    monkeypatch.setattr(runtime, "create_detector", lambda *_args: Mock())
    monkeypatch.setattr(runtime, "run_startup_calibration", lambda *_args, **_kwargs: False)
    boot.traffic_gen.is_running.return_value = running
    cleanup = Mock()
    monkeypatch.setattr(runtime, "cleanup_wifi", cleanup)

    with pytest.raises(RuntimeError, match="Startup calibration failed"):
        runtime.main(boot.wlan)

    cleanup.assert_called_once_with(boot.wlan)
    assert boot.traffic_gen.stop.call_count == (2 if running else 1)
