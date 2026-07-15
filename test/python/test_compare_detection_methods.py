"""
ESPectre - Detection Comparison Tests

Tests for the detection comparison tool.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
MODULE_PATH = TOOLS_DIR / "compare_detection_methods.py"


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


def test_compare_detection_methods_uses_runtime_thresholds_for_current_methods(monkeypatch) -> None:
    module = _load_module()
    static_presence_packets = [_build_packet(seed) for seed in range(130)]
    motion_packets = [_build_packet(seed + 1000) for seed in range(130)]
    calls = {"adaptive_threshold_inputs": []}

    monkeypatch.setattr(module, "ML_AVAILABLE", False)

    def fake_calculate_adaptive_threshold(values, threshold_mode=None, auto_factor=None):
        calls["adaptive_threshold_inputs"].append(
            {
                "values": values,
                "threshold_mode": threshold_mode,
                "auto_factor": auto_factor,
            }
        )
        return 4.56

    monkeypatch.setattr(module, "calculate_adaptive_threshold", fake_calculate_adaptive_threshold)

    *_unused, method_thresholds, _results = module.compare_detection_methods(
        static_presence_packets,
        motion_packets,
        module.WINDOW_SIZE,
        0.99,
    )

    # The comparison uses the caller-provided threshold, which is already
    # calibrated from the selected static capture outside this function.
    assert method_thresholds["Classic"] == 0.99
    assert method_thresholds["RSSI"] == 4.56
    assert set(method_thresholds) == {"RSSI", "Classic"}
    assert len(calls["adaptive_threshold_inputs"]) == 1
    assert calls["adaptive_threshold_inputs"][0]["threshold_mode"] is None
    assert calls["adaptive_threshold_inputs"][0]["auto_factor"] is None


def test_compare_detection_methods_adapts_classic_threshold_only_when_missing(monkeypatch) -> None:
    module = _load_module()
    static_presence_packets = [_build_packet(seed) for seed in range(130)]
    motion_packets = [_build_packet(seed + 2000) for seed in range(130)]
    calls = []

    monkeypatch.setattr(module, "ML_AVAILABLE", False)

    def fake_calculate_adaptive_threshold(values, threshold_mode=None, auto_factor=None):
        calls.append(
            {
                "values": values,
                "threshold_mode": threshold_mode,
                "auto_factor": auto_factor,
            }
        )
        return 1.11 if len(calls) == 1 else 2.34

    monkeypatch.setattr(module, "calculate_adaptive_threshold", fake_calculate_adaptive_threshold)

    *_unused, method_thresholds, _results = module.compare_detection_methods(
        static_presence_packets,
        motion_packets,
        module.WINDOW_SIZE,
        0.0,
    )

    assert method_thresholds["RSSI"] == 1.11
    assert method_thresholds["Classic"] == 2.34
    assert len(calls) == 2
    assert calls[0]["auto_factor"] is None
    assert calls[1]["auto_factor"] == module.L1_DELTA_STARTUP_THRESHOLD_FACTOR


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
                {"name": "Classic", "fp": 0, "tp": len(motion_packets), "fn": 0, "recall": 100.0, "precision": 100.0, "f1": 100.0},
                {"name": "RSSI", "fp": 0, "tp": len(motion_packets), "fn": 0, "recall": 100.0, "precision": 100.0, "f1": 100.0},
            ],
        ),
    )

    module.run_all_chips()

    assert calls["packets"] == static_presence_packets
    captured = capsys.readouterr().out
    assert "Processing C6... done" in captured
