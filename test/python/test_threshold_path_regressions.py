# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - Threshold Path Regression Tests

Regression tests for shared threshold calibration paths.

Author: Francesco Pace <francesco.pace@gmail.com>
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


VALIDATION_REAL_DATA_PATH = Path(__file__).resolve().parent / "test_validation_real_data.py"


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_run_classic_calibration_uses_shared_runtime_helper(monkeypatch) -> None:
    module = _load_module("validation_real_data_threshold_paths", VALIDATION_REAL_DATA_PATH)
    packets = [{"csi_data": [0] * 128} for _ in range(3)]
    calls = {}

    class FakeDetector:
        def __init__(self, **kwargs):
            calls["detector_kwargs"] = kwargs

        def process_packet(self, csi_data, selected_band):
            calls.setdefault("processed_packets", []).append(csi_data)
            calls["selected_band"] = selected_band

        def update_state(self):
            calls["updated"] = True

    class FakeCalibrator:
        def __init__(self, buffer_size, *, auto_factor, gate_enabled):
            calls["buffer_size"] = buffer_size
            calls["auto_factor"] = auto_factor
            calls["gate_enabled"] = gate_enabled

        def observe_detector(self, detector):
            calls["observed_detector"] = detector

        def is_complete(self):
            return True

        def is_successful(self):
            return True

        def calculate_threshold(self):
            return 1.23, "fake"

    monkeypatch.setattr(module, "ClassicDetector", FakeDetector)
    monkeypatch.setattr(module, "StartupThresholdCalibrator", FakeCalibrator)
    monkeypatch.setattr(module, "get_detector_auto_factor", lambda detector: 1.1)
    monkeypatch.setattr(module, "get_detector_startup_gate", lambda detector: True)

    threshold = module.run_classic_calibration(packets, selected_band=(14, 17), window_size=100)

    assert threshold == 1.23
    assert calls["detector_kwargs"]["window_size"] == 100
    assert calls["selected_band"] == (14, 17)


def test_run_fixed_subcarrier_calibration_uses_classic_helper(monkeypatch) -> None:
    module = _load_module("validation_real_data_classic_paths", VALIDATION_REAL_DATA_PATH)
    packets = [{"csi_data": [0] * 128} for _ in range(3)]
    calls = {}

    def fake_run_classic_calibration(packet_list, selected_band, window_size):
        calls["packet_list"] = packet_list
        calls["selected_band"] = selected_band
        calls["window_size"] = window_size
        return 4.56

    monkeypatch.setattr(module, "run_classic_calibration", fake_run_classic_calibration)

    selected_band, threshold = module.run_fixed_subcarrier_calibration(
        packets,
        num_subcarriers=64,
        hint_band=(14, 17),
        window_size_override=123,
    )

    assert selected_band == (14, 17)
    assert threshold == 4.56
    assert calls["packet_list"] == packets
    assert calls["selected_band"] == (14, 17)
    assert calls["window_size"] == 123
