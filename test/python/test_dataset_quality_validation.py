"""
Tests for `tools/11_validate_dataset_quality.py`.
"""

import importlib.util
from pathlib import Path

import numpy as np


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
