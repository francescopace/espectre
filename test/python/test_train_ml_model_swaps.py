"""
Tests for the minimal Core-6 training helpers in `tools/train_ml_model.py`.
"""

from __future__ import annotations

import itertools
import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = REPO_ROOT / "tools" / "train_ml_model.py"

from features import calc_l1_delta, extract_features_by_name


def _load_train_module():
    spec = importlib.util.spec_from_file_location("train_ml_model_swaps", TRAIN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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


def test_training_cache_manifest_tracks_runtime_filter_defaults():
    module = _load_train_module()

    manifest = module._feature_cache_manifest(["turb_skewness"])

    assert manifest["enable_lowpass"] == module.ENABLE_LOWPASS_FILTER
    assert manifest["lowpass_cutoff"] == pytest.approx(module.LOWPASS_CUTOFF)
    assert manifest["enable_hampel"] == module.ENABLE_HAMPEL_FILTER
    assert manifest["hampel_window"] == module.HAMPEL_WINDOW
    assert manifest["hampel_threshold"] == pytest.approx(module.HAMPEL_THRESHOLD)


def test_extract_features_uses_runtime_filter_defaults(monkeypatch):
    module = _load_train_module()
    created = {}

    class FakeSegmentationContext:
        def __init__(self, **kwargs):
            created.update(kwargs)
            self.buffer_count = 0
            self.buffer_index = 0
            self.turbulence_buffer = []

        def calculate_spatial_turbulence(self, csi_data, selected_subcarriers=None, return_amplitudes=False):
            return (0.0, []) if return_amplitudes else 0.0

        def add_turbulence(self, turbulence):
            self.buffer_count += 1

    monkeypatch.setattr(module, "SegmentationContext", FakeSegmentationContext)

    module.extract_features(
        [{"source_file": "sample.npz", "csi_data": [0, 0], "is_motion": False}],
        window_size=2,
        feature_names=["turb_skewness"],
    )

    assert created["enable_lowpass"] == module.ENABLE_LOWPASS_FILTER
    assert created["lowpass_cutoff"] == pytest.approx(module.LOWPASS_CUTOFF)
    assert created["enable_hampel"] == module.ENABLE_HAMPEL_FILTER
    assert created["hampel_window"] == module.HAMPEL_WINDOW
    assert created["hampel_threshold"] == pytest.approx(module.HAMPEL_THRESHOLD)


def _cv_metrics(*, session_recall: float, chip_recall: float, session_fp: float, oof_f1: float, f1_mean: float):
    return {
        "oof_f1": oof_f1,
        "f1_mean": f1_mean,
        "group_reports": {
            "session_group": {
                "worst_recall": {"recall": session_recall},
                "worst_fp_rate": {"fp_rate": session_fp},
            },
            "chip": {
                "worst_recall": {"recall": chip_recall},
            },
        },
    }


def _long_metrics(*, mean_f1: float, worst_f1: float, total_fp: int, mean_recall: float, pass_count: int, max_fp_rate: float):
    return {
        "mean_f1": mean_f1,
        "worst_chip_f1": worst_f1,
        "total_fp": total_fp,
        "mean_recall": mean_recall,
        "pass_count": pass_count,
        "max_fp_rate": max_fp_rate,
    }


def test_search_candidate_key_prefers_passing_gate():
    module = _load_train_module()
    cv = _cv_metrics(session_recall=90.0, chip_recall=88.0, session_fp=8.0, oof_f1=85.0, f1_mean=84.0)
    failing_gate = module.ExportedMLGateResult(
        paired_returncode=1,
        paired_output="paired failed",
        long_metrics=_long_metrics(
            mean_f1=89.0,
            worst_f1=85.0,
            total_fp=30,
            mean_recall=90.0,
            pass_count=2,
            max_fp_rate=8.0,
        ),
        long_output="long gate",
    )
    passing_gate = module.ExportedMLGateResult(
        paired_returncode=0,
        paired_output="",
        long_metrics=_long_metrics(
            mean_f1=89.0,
            worst_f1=85.0,
            total_fp=30,
            mean_recall=90.0,
            pass_count=2,
            max_fp_rate=8.0,
        ),
        long_output="long gate",
    )

    assert module._search_candidate_key(cv, passing_gate) > module._search_candidate_key(cv, failing_gate)


def test_idle_runtime_policy_rejects_isolated_hits_and_counts_alarm_duration():
    module = _load_train_module()
    stride = module.EVALUATION_INTERVAL

    isolated = [0.0] * (stride * 4)
    isolated[stride - 1] = 0.9
    assert module.evaluate_idle_runtime_policy(isolated) == {
        "effective_alarms": 0,
        "false_motion_evaluations": 0,
    }

    burst = [0.0] * (stride * 8)
    for evaluation in range(4):
        burst[(evaluation + 1) * stride - 1] = 0.9
    assert module.evaluate_idle_runtime_policy(burst) == {
        "effective_alarms": 1,
        "false_motion_evaluations": 4,
    }


def test_real_gate_ranking_prefers_fewer_effective_alarms_over_f1():
    module = _load_train_module()
    quieter = _long_metrics(
        mean_f1=80.0,
        worst_f1=75.0,
        total_fp=20,
        mean_recall=90.0,
        pass_count=1,
        max_fp_rate=2.0,
    )
    noisier = _long_metrics(
        mean_f1=99.0,
        worst_f1=98.0,
        total_fp=10,
        mean_recall=99.0,
        pass_count=4,
        max_fp_rate=1.0,
    )
    quieter["total_effective_alarms"] = 0
    quieter["total_false_motion_evaluations"] = 0
    noisier["total_effective_alarms"] = 1
    noisier["total_false_motion_evaluations"] = 1

    assert module._real_ml_gate_key(quieter) > module._real_ml_gate_key(noisier)


def test_train_until_improvement_ranks_candidates_when_baseline_is_broken(monkeypatch):
    module = _load_train_module()

    baseline_cv = _cv_metrics(session_recall=60.0, chip_recall=70.0, session_fp=15.0, oof_f1=80.0, f1_mean=79.0)
    candidate_cv_a = _cv_metrics(session_recall=61.0, chip_recall=71.0, session_fp=14.0, oof_f1=80.5, f1_mean=79.5)
    candidate_cv_b = _cv_metrics(session_recall=64.0, chip_recall=74.0, session_fp=12.0, oof_f1=82.0, f1_mean=81.0)

    baseline_gate = module.ExportedMLGateResult(
        paired_returncode=1,
        paired_output="paired failed",
        long_metrics=_long_metrics(
            mean_f1=80.0,
            worst_f1=78.0,
            total_fp=120,
            mean_recall=82.0,
            pass_count=0,
            max_fp_rate=12.0,
        ),
        long_output="long gate",
    )
    candidate_gate_a = module.ExportedMLGateResult(
        paired_returncode=1,
        paired_output="paired failed",
        long_metrics=_long_metrics(
            mean_f1=81.0,
            worst_f1=79.0,
            total_fp=110,
            mean_recall=83.0,
            pass_count=0,
            max_fp_rate=11.0,
        ),
        long_output="long gate",
    )
    candidate_gate_b = module.ExportedMLGateResult(
        paired_returncode=1,
        paired_output="paired failed",
        long_metrics=_long_metrics(
            mean_f1=84.0,
            worst_f1=82.0,
            total_fp=90,
            mean_recall=86.0,
            pass_count=0,
            max_fp_rate=9.0,
        ),
        long_output="long gate",
    )

    train_calls = iter(
        [
            (0, 111, baseline_cv),
            (0, 201, candidate_cv_a),
            (0, 201, candidate_cv_a),
            (0, 202, candidate_cv_b),
            (0, 202, candidate_cv_b),
        ]
    )
    gate_calls = iter([baseline_gate, candidate_gate_a, candidate_gate_b])

    monkeypatch.setattr(module, "ensure_torch_available", lambda: object())
    monkeypatch.setattr(module, "describe_torch_device", lambda: "cpu")
    monkeypatch.setattr(module, "read_exported_seed", lambda: 111)
    monkeypatch.setattr(module, "train_all", lambda **kwargs: next(train_calls))
    monkeypatch.setattr(module, "run_exported_ml_gates", lambda **kwargs: next(gate_calls))

    backup_counter = itertools.count()
    restore_calls = []

    def fake_backup():
        idx = next(backup_counter)
        return f"backup-{idx}", [f"snapshot-{idx}"]

    def fake_restore(saved_files):
        restore_calls.append(tuple(saved_files))

    monkeypatch.setattr(module, "_backup_artifacts", fake_backup)
    monkeypatch.setattr(module, "_restore_artifacts", fake_restore)

    result = module.train_until_improvement(max_trials=2, use_cache=True)

    assert result == 0
    assert restore_calls[-1] == ("snapshot-2",)
