"""Tests for the detection comparison tool."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
MODULE_PATH = TOOLS_DIR / "7_compare_detection_methods.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("compare_detection_methods_tool", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _build_packet(seed: int) -> dict[str, list[int]]:
    # Provide a deterministic 64-subcarrier I/Q payload with small variation.
    csi_data = [((seed + idx) % 9) - 4 for idx in range(128)]
    return {"csi_data": csi_data}


def test_compare_detection_methods_uses_detector_specific_runtime_thresholds(monkeypatch) -> None:
    module = _load_module()
    static_presence_packets = [_build_packet(seed) for seed in range(130)]
    motion_packets = [_build_packet(seed + 1000) for seed in range(130)]
    calls = {}

    monkeypatch.setattr(module, "ML_AVAILABLE", False)

    def fake_estimate_runtime_threshold(packets, threshold_mode=None, selected_subcarriers=None):
        calls["l1_packets"] = packets
        calls["l1_threshold_mode"] = threshold_mode
        calls["l1_selected_subcarriers"] = selected_subcarriers
        return 1.23

    def fake_calibrate_startup_threshold(
        packets,
        *,
        selected_band,
        window_size,
        filter_config=None,
    ):
        calls["mvs_packets"] = packets
        calls["mvs_selected_band"] = selected_band
        calls["mvs_window_size"] = window_size
        calls["mvs_filter_config"] = filter_config
        return 4.56, 0.78

    monkeypatch.setattr(module, "estimate_runtime_threshold", fake_estimate_runtime_threshold)
    monkeypatch.setattr(module, "calibrate_startup_threshold", fake_calibrate_startup_threshold)

    *_unused, method_thresholds, _results = module.compare_detection_methods(
        static_presence_packets,
        motion_packets,
        module.WINDOW_SIZE,
        0.99,
    )

    # The comparison uses the caller-provided L1D threshold, which is already
    # calibrated from the selected static capture outside this function.
    assert method_thresholds["L1D"] == 0.99
    assert method_thresholds["MVS"] == 4.56
    assert calls["mvs_packets"] == static_presence_packets
    assert calls["mvs_selected_band"] == tuple(module.DEFAULT_SUBCARRIERS)
    assert calls["mvs_window_size"] == module.WINDOW_SIZE
    assert calls["mvs_filter_config"] is None


def test_compare_detection_methods_does_not_recompute_l1d_threshold(monkeypatch) -> None:
    module = _load_module()
    static_presence_packets = [_build_packet(seed) for seed in range(130)]
    motion_packets = [_build_packet(seed + 2000) for seed in range(130)]

    monkeypatch.setattr(module, "ML_AVAILABLE", False)
    monkeypatch.setattr(
        module,
        "estimate_runtime_threshold",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("compare should use the precomputed L1D threshold")),
    )
    monkeypatch.setattr(
        module,
        "calibrate_startup_threshold",
        lambda packets, *, selected_band, window_size, filter_config=None: (2.34, None),
    )

    *_unused, method_thresholds, _results = module.compare_detection_methods(
        static_presence_packets,
        motion_packets,
        module.WINDOW_SIZE,
        0.99,
    )

    assert method_thresholds["L1D"] == 0.99
    assert method_thresholds["MVS"] == 2.34


def test_run_all_chips_passes_static_capture_to_context_resolver(monkeypatch, capsys) -> None:
    module = _load_module()
    static_presence_packets = [_build_packet(seed) for seed in range(4)]
    motion_packets = [_build_packet(seed + 10) for seed in range(4)]
    calls = {}

    monkeypatch.setattr(module, "ML_AVAILABLE", False)
    monkeypatch.setattr(
        module,
        "load_dataset_info",
        lambda: {
            "files": {
                "static_presence": [
                    {"chip": "C6", "optimal_pair_motion_file": "motion_c6_example.npz"},
                ]
            }
        },
    )
    fake_pair = SimpleNamespace(
        static_presence=SimpleNamespace(path=Path("/tmp/static_presence_c6_example.npz")),
        motion=SimpleNamespace(path=Path("/tmp/motion_c6_example.npz")),
        chip="C6",
    )
    monkeypatch.setattr(module, "resolve_explicit_pair", lambda chip, num_sc=64: fake_pair)
    monkeypatch.setattr(
        module,
        "load_static_presence_and_motion",
        lambda static_presence_file, motion_file, chip=None, dataset=None: (
            static_presence_packets,
            motion_packets,
        ),
    )

    def fake_resolve_context_aware_config(pair, packets):
        calls["pair"] = pair
        calls["packets"] = packets
        return {
            "threshold": 0.99,
            "context_source": "explicit-pair runtime l1_delta calibration",
            "confidence_factor": 1.0,
        }

    monkeypatch.setattr(module, "resolve_context_aware_config", fake_resolve_context_aware_config)
    monkeypatch.setattr(
        module,
        "compare_detection_methods",
        lambda static_presence_packets, motion_packets, window_size, threshold: (
            {},
            None,
            None,
            {},
            None,
            None,
            {},
            [
                {"name": "MVS", "fp": 0, "tp": len(motion_packets), "fn": 0, "recall": 100.0, "precision": 100.0, "f1": 100.0},
                {"name": "L1D", "fp": 0, "tp": len(motion_packets), "fn": 0, "recall": 100.0, "precision": 100.0, "f1": 100.0},
            ],
        ),
    )

    module.run_all_chips()

    assert calls["packets"] == static_presence_packets
    captured = capsys.readouterr().out
    assert "Processing C6... done" in captured
