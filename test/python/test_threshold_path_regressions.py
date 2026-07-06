"""Regression tests for shared threshold calibration paths."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from tools.lib import mvs_sweep_core


TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
VALIDATION_REAL_DATA_PATH = Path(__file__).resolve().parent / "test_validation_real_data.py"
ANALYZE_FILTER_TURBULENCE_PATH = TOOLS_DIR / "5_analyze_filter_turbulence.py"
BENCHMARK_MOTION_FEATURES_PATH = TOOLS_DIR / "12_benchmark_motion_features.py"
RESEARCH_MOTION_SCORE_PATH = TOOLS_DIR / "13_research_motion_score_benchmark.py"


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_run_l1_delta_calibration_uses_shared_runtime_helper(monkeypatch) -> None:
    module = _load_module("validation_real_data_threshold_paths", VALIDATION_REAL_DATA_PATH)
    packets = [{"csi_data": [0] * 128} for _ in range(3)]
    calls = {}

    def fake_estimate_runtime_threshold(packet_list, threshold_mode=None, selected_subcarriers=None):
        calls["packet_list"] = packet_list
        calls["threshold_mode"] = threshold_mode
        calls["selected_subcarriers"] = selected_subcarriers
        return 1.23

    monkeypatch.setattr(module, "estimate_runtime_threshold", fake_estimate_runtime_threshold)

    threshold = module.run_l1_delta_calibration(packets, selected_band=(14, 17), window_size=100)

    assert threshold == 1.23
    assert calls["packet_list"] == packets
    assert calls["threshold_mode"] is None
    assert calls["selected_subcarriers"] == (14, 17)


def test_run_fixed_subcarrier_calibration_uses_shared_mvs_helper(monkeypatch) -> None:
    module = _load_module("validation_real_data_mvs_paths", VALIDATION_REAL_DATA_PATH)
    packets = [{"csi_data": [0] * 128} for _ in range(3)]
    calls = {}

    def fake_calibrate_startup_threshold(packet_list, *, selected_band, window_size, filter_config=None):
        calls["packet_list"] = packet_list
        calls["selected_band"] = selected_band
        calls["window_size"] = window_size
        calls["filter_config"] = filter_config
        return 4.56, 0.78

    monkeypatch.setattr(module, "calibrate_startup_threshold", fake_calibrate_startup_threshold)

    selected_band, threshold = module.run_fixed_subcarrier_calibration(
        packets,
        num_subcarriers=64,
        hint_band=(14, 17),
        mvs_window_size=123,
    )

    assert selected_band == (14, 17)
    assert threshold == 4.56
    assert calls["packet_list"] == packets
    assert calls["selected_band"] == (14, 17)
    assert calls["window_size"] == 123
    assert calls["filter_config"] is None


def test_analyze_filter_turbulence_defaults_to_calibrate(monkeypatch) -> None:
    module = _load_module("analyze_filter_turbulence_threshold_defaults", ANALYZE_FILTER_TURBULENCE_PATH)
    monkeypatch.setattr(sys, "argv", ["5_analyze_filter_turbulence.py"])

    args = module.parse_args()

    assert not hasattr(args, "threshold_source")


def test_mvs_evaluate_pairs_defaults_to_calibrate(monkeypatch) -> None:
    calls = {}

    def fake_evaluate_pair(
        pair,
        *,
        variant=None,
        filter_config=None,
        window_size=None,
        selected_band=None,
        track_trace=None,
        threshold_source=None,
    ):
        calls["threshold_source"] = threshold_source
        return "ok"

    monkeypatch.setattr(mvs_sweep_core, "evaluate_pair", fake_evaluate_pair)

    results = mvs_sweep_core.evaluate_pairs([object()])

    assert results == ["ok"]
    assert calls["threshold_source"] == "calibrate"


def test_benchmark_motion_features_uses_shared_mvs_bootstrap(monkeypatch) -> None:
    module = _load_module("benchmark_motion_features_threshold_paths", BENCHMARK_MOTION_FEATURES_PATH)
    packets = [{"csi_data": [1] * 128} for _ in range(3)]
    calls = {}

    def fake_calibrate_startup_threshold(packet_list, *, selected_band, window_size, filter_config=None):
        calls["packet_list"] = packet_list
        calls["selected_band"] = selected_band
        calls["window_size"] = window_size
        calls["filter_config"] = filter_config
        return 7.89, 0.12

    monkeypatch.setattr(module, "calibrate_startup_threshold", fake_calibrate_startup_threshold)

    threshold = module._estimate_mvs_threshold_from_idle_prefix(packets)

    assert threshold == 7.89
    assert calls["packet_list"] == packets
    assert calls["selected_band"] == tuple(module.DEFAULT_SUBCARRIERS)
    assert calls["window_size"] == module.SEG_WINDOW_SIZE
    assert calls["filter_config"] is None


def test_research_motion_score_quiet_room_uses_shared_mvs_bootstrap(monkeypatch) -> None:
    module = _load_module("research_motion_score_threshold_paths", RESEARCH_MOTION_SCORE_PATH)
    calls = {}
    packets = [{"csi_data": [1] * 128} for _ in range(5)]

    def fake_calibrate_startup_threshold(packet_list, *, selected_band, window_size, filter_config=None):
        calls["packet_list"] = packet_list
        calls["selected_band"] = selected_band
        calls["window_size"] = window_size
        calls["filter_config"] = filter_config
        return 0.5, 0.1

    monkeypatch.setattr(module, "calibrate_startup_threshold", fake_calibrate_startup_threshold)

    idle_rec = module.IdleSeries(
        chip="C6",
        rec_id="empty:test",
        source="empty",
        packets=packets,
        turb_filtered=[],
        amp_sel=[],
        amp_all=[],
        fs=100.0,
    )

    result = module.production_mvs_quiet_room([idle_rec])

    assert "C6" in result
    assert calls["packet_list"] == packets
    assert calls["selected_band"] == tuple(module.config.DEFAULT_SUBCARRIERS)
    assert calls["window_size"] == module.config.SEG_WINDOW_SIZE
    assert calls["filter_config"] is None
