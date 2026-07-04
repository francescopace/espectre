"""
Tests for `tools/11_validate_dataset_quality.py`.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


VALIDATOR_PATH = Path(__file__).resolve().parents[2] / "tools" / "11_validate_dataset_quality.py"


def _load_validator_module():
    """Load the validator script directly despite its numeric filename."""
    spec = importlib.util.spec_from_file_location("dataset_quality_validation", VALIDATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _meta(collected_at: str, duration_ms: int) -> dict:
    return {
        "collected_at": collected_at,
        "duration_ms": duration_ms,
    }


def _temporal_gap_result(module, bl_meta: dict, mv_meta: dict):
    csi = np.zeros((4, 128), dtype=np.int8)
    results, *_ = module.validate_pair(csi, csi, bl_meta, mv_meta)
    return next(result for result in results if result.name == "temporal_gap")


def test_temporal_gap_passes_when_motion_happens_first() -> None:
    module = _load_validator_module()

    result = _temporal_gap_result(
        module,
        bl_meta=_meta("2026-06-30T12:05:00", 180000),
        mv_meta=_meta("2026-06-30T12:00:00", 120000),
    )

    assert result.status == "PASS"
    assert result.value == 180.0
    assert result.message == "Gap: 180.0s"


def test_temporal_gap_is_zero_for_overlapping_captures() -> None:
    module = _load_validator_module()

    result = _temporal_gap_result(
        module,
        bl_meta=_meta("2026-06-30T12:04:00", 300000),
        mv_meta=_meta("2026-06-30T12:00:00", 300000),
    )

    assert result.status == "PASS"
    assert result.value == 0.0
    assert result.message == "Gap: 0.0s"


def test_temporal_gap_warns_only_for_large_order_agnostic_gaps() -> None:
    module = _load_validator_module()

    result = _temporal_gap_result(
        module,
        bl_meta=_meta("2026-06-30T12:40:01", 180000),
        mv_meta=_meta("2026-06-30T12:00:00", 120000),
    )

    assert result.status == "WARN"
    assert result.value == 2281.0
    assert result.message == "Large gap: 2281.0s > 1800s"


def test_empty_separation_uses_two_feature_score(monkeypatch) -> None:
    module = _load_validator_module()
    monkeypatch.setattr(module, "SEG_WINDOW_SIZE", 2)

    dataset_info = {
        "files": {
            "empty": [
                {"filename": "empty_a.npz", "chip": "C5", "environment": "bedroom"},
            ],
            "static_presence": [
                {
                    "filename": "static_a.npz",
                    "chip": "C5",
                    "environment": "bedroom",
                },
            ],
        }
    }

    fake_data = {
        "empty_a.npz": (np.zeros((4, 128), dtype=np.int8), np.zeros((4,), dtype=np.int8)),
        "static_a.npz": (np.zeros((4, 128), dtype=np.int8), np.zeros((4,), dtype=np.int8)),
    }

    def fake_resolve(entry, label):
        return Path(f"/tmp/{entry['filename']}")

    def fake_load(path, npz_cache):
        csi, refs = fake_data[path.name]
        return {"csi_data": csi, "is_reference": refs}, "csi_data"

    def fake_filter(csi_data, data):
        return csi_data

    def fake_compute(csi_data, use_cv_normalization=True):
        if csi_data is fake_data["empty_a.npz"][0]:
            # `turb_mean` separates empty from static, while moving variance stays identical.
            return np.array([1.0, 1.0, 1.0, 1.0]), np.array([0.2, 0.2, 0.2])
        return np.array([5.0, 5.0, 5.0, 5.0]), np.array([0.2, 0.2, 0.2])

    def fake_feature_series(values, feature_name, window_size=None):
        if feature_name != "waveform_length_over_mean":
            raise AssertionError(feature_name)
        if values[0] == 1.0:
            return [0.1, 0.1, 0.1]
        return [0.9, 0.9, 0.9]

    monkeypatch.setattr(module, "_resolve_dataset_entry_path", fake_resolve)
    monkeypatch.setattr(module, "_load_cached_or_npz", fake_load)
    monkeypatch.setattr(module, "_filter_measurement_frames", fake_filter)
    monkeypatch.setattr(module, "_compute_turbulence_and_moving_variance_series", fake_compute)
    monkeypatch.setattr(module, "_window_feature_series", fake_feature_series)

    results = module.validate_empty_sanity(dataset_info, npz_cache={})
    separation = next(r for r in results if r.name == "empty_separation_C5_bedroom")

    assert separation.status == "PASS"
    assert separation.value == 1.0
    assert separation.message.startswith(
        "Empty-vs-static score separates group ('C5', 'bedroom')"
    )
