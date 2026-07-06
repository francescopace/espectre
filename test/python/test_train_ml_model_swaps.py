"""
Tests for feature-swap helpers in `tools/10_train_ml_model.py`.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = REPO_ROOT / "tools" / "10_train_ml_model.py"
PYTHON_SRC = REPO_ROOT / "src" / "python" / "micro_espectre"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

from features import calc_l1_delta, extract_features_by_name


def _load_train_module():
    spec = importlib.util.spec_from_file_location("train_ml_model_swaps", TRAIN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_feature_swap_accepts_equals_and_colon():
    module = _load_train_module()

    assert module.parse_feature_swap("waveform_length_over_mean=l1_delta") == (
        "waveform_length_over_mean",
        "l1_delta",
    )
    assert module.parse_feature_swap("turb_skewness:l1_delta") == (
        "turb_skewness",
        "l1_delta",
    )


def test_apply_feature_swaps_replaces_requested_feature():
    module = _load_train_module()

    features = [
        "turb_std_over_mean",
        "waveform_length_over_mean",
        "turb_autocorr",
    ]
    swapped = module.apply_feature_swaps(
        features,
        [("waveform_length_over_mean", "l1_delta")],
    )

    assert swapped == [
        "turb_std_over_mean",
        "l1_delta",
        "turb_autocorr",
    ]


def test_parse_feature_drop_list_accepts_commas_and_spaces():
    module = _load_train_module()

    assert module.parse_feature_drop_list(
        "waveform_length_over_mean, turb_skewness ,turb_autocorr"
    ) == [
        "waveform_length_over_mean",
        "turb_skewness",
        "turb_autocorr",
    ]


def test_apply_feature_drops_removes_requested_features():
    module = _load_train_module()

    dropped = module.apply_feature_drops(
        [
            "turb_std_over_mean",
            "waveform_length_over_mean",
            "turb_skewness",
            "turb_autocorr",
        ],
        ["waveform_length_over_mean", "turb_skewness"],
    )

    assert dropped == [
        "turb_std_over_mean",
        "turb_autocorr",
    ]


def test_apply_feature_drops_rejects_missing_feature():
    module = _load_train_module()

    with pytest.raises(argparse.ArgumentTypeError):
        module.apply_feature_drops(
            ["turb_std_over_mean", "turb_autocorr"],
            ["waveform_length_over_mean"],
        )


def test_apply_feature_swaps_rejects_duplicate_replacement():
    module = _load_train_module()

    with pytest.raises(argparse.ArgumentTypeError):
        module.apply_feature_swaps(
            ["turb_std_over_mean", "waveform_length_over_mean", "l1_delta"],
            [("waveform_length_over_mean", "l1_delta")],
        )


def test_build_feature_sweep_candidates_replaces_each_slot_once():
    module = _load_train_module()

    candidates = module.build_feature_sweep_candidates(
        ["turb_std_over_mean", "waveform_length_over_mean", "turb_autocorr"],
        "l1_delta",
    )

    assert [candidate["replaced"] for candidate in candidates] == [
        "turb_std_over_mean",
        "waveform_length_over_mean",
        "turb_autocorr",
    ]
    assert candidates[1]["feature_names"] == [
        "turb_std_over_mean",
        "l1_delta",
        "turb_autocorr",
    ]


def test_build_feature_sweep_candidates_rejects_existing_feature():
    module = _load_train_module()

    with pytest.raises(argparse.ArgumentTypeError):
        module.build_feature_sweep_candidates(
            ["turb_std_over_mean", "l1_delta", "turb_autocorr"],
            "l1_delta",
        )


def test_extract_features_by_name_supports_l1_delta_feature():
    base_profile = [1.0, 2.0, 4.0, 8.0]
    changed_profile = [8.0, 4.0, 2.0, 1.0]
    amplitude_history = [base_profile] * 10 + [changed_profile] * 10
    turbulence = [0.1] * len(amplitude_history)

    expected = calc_l1_delta(amplitude_history, len(amplitude_history))
    features = extract_features_by_name(
        turbulence,
        len(turbulence),
        feature_names=["l1_delta"],
        amplitude_history=amplitude_history,
    )

    assert features == pytest.approx([expected])
    assert expected > 0.0
