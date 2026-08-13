# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Tests for the production Lightweight detector."""

import math

import pytest

from lightweight_detector import LightweightDetector
from detector_interface import (
    MotionState,
    detector_needs_startup_calibration,
    get_detector_algorithm,
    load_detector_class,
)


def test_registry_exposes_lightweight_detector() -> None:
    detector = LightweightDetector()

    assert load_detector_class("lightweight") is LightweightDetector
    assert detector_needs_startup_calibration("lightweight")
    assert get_detector_algorithm(detector) == "lightweight"
    assert detector.get_name() == "Lightweight"


def test_linear_fusion_uses_exported_center_scale_and_weights() -> None:
    detector = LightweightDetector()

    assert detector._calculate_logit(
        detector.FEATURE_CENTER[0],
        detector.FEATURE_CENTER[1],
    ) == pytest.approx(detector.INTERCEPT)
    assert detector._calculate_logit(
        detector.FEATURE_CENTER[0] + detector.FEATURE_SCALE[0],
        detector.FEATURE_CENTER[1] + detector.FEATURE_SCALE[1],
    ) == pytest.approx(
        detector.INTERCEPT + sum(detector.FEATURE_WEIGHT)
    )


def test_hampel_master_switch_controls_turbulence_stream() -> None:
    enabled = LightweightDetector(enable_hampel=True)
    disabled = LightweightDetector(enable_hampel=False)

    assert enabled._context.hampel_filter is not None
    assert enabled._aggregated_context.hampel_filter is not None
    assert disabled._context.hampel_filter is None
    assert disabled._aggregated_context.hampel_filter is None


def test_lightweight_allocates_aggregated_turbulence_state() -> None:
    detector = LightweightDetector()

    assert len(detector._aggregated_context.turbulence_buffer) == 100
    assert detector._aggregated_context.buffer_count == 0


def test_startup_q95_adapts_probability_threshold() -> None:
    detector = LightweightDetector()
    detector._startup_logits = [-1.0, -0.8, -0.6, -0.4]

    detector.set_adaptive_threshold(0.01)

    q95 = detector._quantile(detector._startup_logits, 0.95)
    base_logit = math.log(
        detector.BASE_THRESHOLD / (1.0 - detector.BASE_THRESHOLD)
    )
    expected = detector._sigmoid(
        base_logit
        + detector.STARTUP_STRENGTH * (q95 - detector.TRAIN_IDLE_Q95_LOGIT)
    )
    assert detector.get_threshold() == pytest.approx(expected)


def test_noisy_startup_still_uses_the_shifted_logit_threshold() -> None:
    detector = LightweightDetector()
    detector._startup_logits = [10.0] * 4

    detector.set_adaptive_threshold(0.01)

    q95 = detector._quantile(detector._startup_logits, 0.95)
    base_logit = math.log(
        detector.BASE_THRESHOLD / (1.0 - detector.BASE_THRESHOLD)
    )
    expected = detector._sigmoid(
        base_logit
        + detector.STARTUP_STRENGTH * (q95 - detector.TRAIN_IDLE_Q95_LOGIT)
    )
    assert detector.get_threshold() == pytest.approx(expected)
    assert detector.get_threshold() > detector.BASE_THRESHOLD


def test_manual_threshold_uses_probability_scale() -> None:
    detector = LightweightDetector()

    assert detector.set_threshold(0.75)
    assert detector.get_threshold() == pytest.approx(0.75)
    assert not detector.set_threshold(1.01)
    assert detector.get_threshold() == pytest.approx(0.75)


def test_update_state_uses_weighted_probability(monkeypatch) -> None:
    detector = LightweightDetector(window_size=20, threshold=0.5)
    monkeypatch.setattr(detector, "is_ready", lambda: True)
    monkeypatch.setattr(detector, "_turb_autocorr", lambda: detector.FEATURE_CENTER[0])
    monkeypatch.setattr(
        detector,
        "_turb_iqr_over_mean_aggr",
        lambda: detector.FEATURE_CENTER[1],
    )

    metrics = detector.update_state()

    expected = detector._sigmoid(detector.INTERCEPT)
    assert metrics["probability"] == pytest.approx(expected)
    assert metrics["turb_autocorr"] == pytest.approx(detector.FEATURE_CENTER[0])
    assert metrics["turb_iqr_over_mean_aggr"] == pytest.approx(
        detector.FEATURE_CENTER[1]
    )
    # Derived from the probability rather than pinned, so a refit that moves the
    # intercept across the threshold does not read as a state-machine fault.
    assert metrics["state"] == (
        MotionState.MOTION if expected > detector.get_threshold() else MotionState.IDLE
    )


def test_reset_preserves_threshold_and_clears_feature_state() -> None:
    detector = LightweightDetector(threshold=0.7)
    detector._current_probability = 0.9
    detector._startup_logits = [1.0]

    detector.reset()

    assert detector.get_threshold() == pytest.approx(0.7)
    assert detector.get_motion_metric() == 0.0
    assert detector._startup_logits == []
    assert detector.get_state() == MotionState.IDLE
