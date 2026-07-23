"""
ESPectre - ML Training Helper Tests

Tests for the minimal Core-6 training helpers in tools/train_ml_model.py.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

from __future__ import annotations

import itertools
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = REPO_ROOT / "tools" / "train_ml_model.py"

from config import MOTION_OFF_HITS, MOTION_ON_HITS
from csi_features import calc_l1_delta, l1_delta_series, extract_features_by_name


def _load_train_module():
    spec = importlib.util.spec_from_file_location("train_ml_model_swaps", TRAIN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_extract_features_by_name_requires_explicit_l1_stream():
    base_profile = [1.0, 2.0, 4.0, 8.0]
    changed_profile = [8.0, 4.0, 2.0, 1.0]
    amplitude_history = [base_profile] * 10 + [changed_profile] * 10
    turbulence = [0.1] * len(amplitude_history)

    l1_series = l1_delta_series(amplitude_history, len(amplitude_history))
    expected = calc_l1_delta(amplitude_history, len(amplitude_history))
    features = extract_features_by_name(
        turbulence,
        len(turbulence),
        feature_names=["l1_delta"],
        l1_series=l1_series,
    )

    assert features == pytest.approx([expected])
    assert expected > 0.0

    with pytest.raises(ValueError, match="l1_series is required"):
        extract_features_by_name(
            turbulence,
            len(turbulence),
            feature_names=["l1_delta"],
        )


def test_resolve_training_seed_prefers_exported_model_seed(monkeypatch, capsys):
    module = _load_train_module()
    monkeypatch.setattr(module, "read_exported_seed", lambda: 1194060148)

    assert module.resolve_training_seed(None) == 1194060148
    assert "Using exported model seed: 1194060148" in capsys.readouterr().out

    assert module.resolve_training_seed(42) == 42
    assert "Using provided seed: 42" in capsys.readouterr().out


def test_resolve_training_seed_can_force_random(monkeypatch, capsys):
    module = _load_train_module()
    monkeypatch.setattr(module, "read_exported_seed", lambda: 1194060148)
    monkeypatch.setattr(module, "generate_random_training_seed", lambda: 7)

    assert module.resolve_training_seed(None, prefer_exported=False) == 7
    assert "Generated random seed: 7" in capsys.readouterr().out


def test_resolve_training_seed_falls_back_when_exported_missing(monkeypatch, capsys):
    module = _load_train_module()
    monkeypatch.setattr(module, "read_exported_seed", lambda: None)
    monkeypatch.setattr(module, "generate_random_training_seed", lambda: 99)

    assert module.resolve_training_seed(None) == 99
    assert "No exported model seed found; generated random seed: 99" in capsys.readouterr().out


def test_resolve_training_augmentation_returns_robustness_winner_recipe():
    module = _load_train_module()

    feature_aug, packet_aug = module.resolve_training_augmentation(False)
    assert feature_aug == {}
    assert packet_aug == {}

    feature_aug, packet_aug = module.resolve_training_augmentation(True)
    assert feature_aug == {"jitter_sigma": 0.10}
    assert packet_aug == {
        "gain_sigma": 0.05,
        "noise_sigma": 0.01,
        "packet_loss": 0.05,
    }
    assert "jitter_010" in module.format_augmentation_config(feature_aug, packet_aug)


def test_append_augmented_training_rows_keeps_train_groups_only():
    module = _load_train_module()

    class _IdentityScaler:
        def transform(self, values):
            return np.asarray(values, dtype=np.float32)

    X_train = np.asarray([[1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    y_train = np.asarray([0, 1], dtype=np.int8)
    X_aug = np.asarray([[10.0, 0.0], [20.0, 0.0], [30.0, 0.0]], dtype=np.float32)
    y_aug = np.asarray([0, 1, 0], dtype=np.int8)
    groups_aug = np.asarray(["a", "b", "a"])

    X_out, y_out, sw_out = module._append_augmented_training_rows(
        X_train,
        y_train,
        _IdentityScaler(),
        X_aug,
        y_aug,
        groups_aug,
        train_groups=["a"],
        sample_weight=np.asarray([1.0, 1.0], dtype=np.float32),
    )

    assert X_out.shape == (4, 2)
    assert y_out.tolist() == [0, 1, 0, 0]
    assert sw_out.tolist() == [1.0, 1.0, 1.0, 1.0]
    assert X_out[2].tolist() == [10.0, 0.0]
    assert X_out[3].tolist() == [30.0, 0.0]


def test_training_cache_manifest_tracks_runtime_filter_defaults():
    module = _load_train_module()

    manifest = module._feature_cache_manifest(["turb_skewness"])

    assert manifest["enable_lowpass"] == module.ENABLE_LOWPASS_FILTER
    assert manifest["lowpass_cutoff"] == pytest.approx(module.LOWPASS_CUTOFF)
    assert manifest["enable_hampel"] == module.ENABLE_HAMPEL_FILTER
    assert manifest["hampel_window"] == module.HAMPEL_WINDOW
    assert manifest["hampel_threshold"] == pytest.approx(module.HAMPEL_THRESHOLD)


def test_training_cache_manifest_tracks_packet_augmentation():
    module = _load_train_module()
    config = {"gain_sigma": 0.05, "packet_loss": 0.1}

    manifest = module._feature_cache_manifest(
        ["turb_skewness"],
        packet_augmentation=config,
        augmentation_seed=123,
    )

    assert manifest["packet_augmentation"] == config
    assert manifest["augmentation_seed"] == 123


def test_training_cache_manifest_tracks_dataset_roles():
    module = _load_train_module()

    manifest = module._feature_cache_manifest(
        ["turb_skewness"],
        dataset_roles=("train", "selection"),
    )

    assert manifest["dataset"]["dataset_roles"] == ["selection", "train"]
    with pytest.raises(ValueError, match="Unsupported dataset role"):
        module.normalize_dataset_roles(("train", "unknown"))


def test_synthetic_metadata_shares_source_lineage(monkeypatch, tmp_path):
    module = _load_train_module()
    data_dir = tmp_path / "data"
    (data_dir / "static_presence").mkdir(parents=True)
    (data_dir / "motion").mkdir(parents=True)
    np.savez(data_dir / "static_presence" / "source.npz", csi_data=np.zeros((1, 2)))
    np.savez(
        data_dir / "motion" / "synthetic.npz",
        csi_data=np.zeros((1, 2)),
        source_dataset=np.asarray("source.npz"),
        generation_group=np.asarray("low-rssi-c3"),
        generation_mode=np.asarray("derived"),
        synthetic=np.asarray(True),
    )
    monkeypatch.setattr(module, "DATA_DIR", data_dir)
    dataset_info = {
        "files": {
            "static_presence": [{
                "filename": "source.npz",
                "chip": "C3",
                "session": "real-session",
            }],
            "motion": [{
                "filename": "synthetic.npz",
                "chip": "C3",
            }],
        },
    }

    metadata = module.get_file_metadata(dataset_info)

    assert metadata["source.npz"]["lineage_group"] == "real-session"
    assert metadata["synthetic.npz"]["lineage_group"] == "real-session"
    assert metadata["synthetic.npz"]["synthetic"] is True


def test_session_balanced_robust_scaler_is_deterministic_and_balanced():
    module = _load_train_module()
    X = np.arange(80, dtype=np.float32).reshape(40, 2)
    y = np.asarray([0] * 15 + [1] * 5 + [0] * 10 + [1] * 10)
    groups = np.asarray(["long"] * 20 + ["short"] * 20)

    first = module.SessionBalancedRobustScaler(max_samples_per_stratum=4)
    second = module.SessionBalancedRobustScaler(max_samples_per_stratum=4)
    first.fit(X, y=y, groups=groups)
    second.fit(X, y=y, groups=groups)

    assert first.selected_indices_.tolist() == second.selected_indices_.tolist()
    assert len(first.selected_indices_) == 16
    selected_strata = list(zip(groups[first.selected_indices_], y[first.selected_indices_]))
    assert all(selected_strata.count(key) == 4 for key in set(selected_strata))
    assert first.center_ == pytest.approx(second.center_)
    assert first.scale_ == pytest.approx(second.scale_)


@pytest.mark.parametrize(
    ("variant", "relative_std", "relative_waveform"),
    [
        ("l1_std_relative", True, False),
        ("l1_waveform_relative", False, True),
        ("l1_both_relative", True, True),
    ],
)
def test_l1_feature_variants_keep_delta_and_normalize_descriptors(
        variant, relative_std, relative_waveform):
    module = _load_train_module()
    names = list(module.DEFAULT_FEATURES)
    row = np.asarray([[0.1, 0.2, 0.3, 1.0, 2.0, 3.0]], dtype=np.float32)

    transformed = module.apply_l1_feature_variant(row, names, variant)

    assert transformed[0, names.index("l1_delta")] == pytest.approx(1.0)
    expected_std = 2.0 / 1.001 if relative_std else 2.0
    l1_steps = max(1, module.SEG_WINDOW_SIZE - module.L1_DELTA_LAG - 1)
    expected_waveform = 3.0 / (l1_steps * 1.001) if relative_waveform else 3.0
    assert transformed[0, names.index("l1_delta_std")] == pytest.approx(expected_std)
    assert transformed[0, names.index("l1_delta_waveform_length")] == pytest.approx(expected_waveform)


def test_feature_augmentation_is_reproducible_and_respects_bounds():
    module = _load_train_module()
    X = np.zeros((128, 6), dtype=np.float32)
    config = {
        "noise_sigma": 0.1,
        "jitter_sigma": 0.1,
        "dropout_probability": 0.02,
    }
    lower = np.asarray([-0.2] * 6, dtype=np.float32)
    upper = np.asarray([0.2] * 6, dtype=np.float32)

    first = module.augment_normalized_features(X, config, 42, (lower, upper))
    second = module.augment_normalized_features(X, config, 42, (lower, upper))

    assert first == pytest.approx(second)
    assert np.all(first >= lower)
    assert np.all(first <= upper)
    assert np.any(first != 0.0)
    assert np.any(np.all(first == 0.0, axis=1))


def test_packet_augmentation_is_reproducible_bounded_and_non_mutating():
    module = _load_train_module()
    packets = [
        {
            "source_file": "sample.npz",
            "packet_index": idx,
            "csi_data": np.asarray([100, -100] * 8, dtype=np.int16),
        }
        for idx in range(20)
    ]
    original = [packet["csi_data"].copy() for packet in packets]
    config = {"gain_sigma": 0.1, "noise_sigma": 0.03, "packet_loss": 0.5}

    first = module.augment_csi_packets(packets, config, 123)
    second = module.augment_csi_packets(packets, config, 123)

    assert [row["packet_index"] for row in first] == [row["packet_index"] for row in second]
    assert len(first) < len(packets)
    assert all(np.array_equal(a["csi_data"], b["csi_data"]) for a, b in zip(first, second))
    assert all(np.all(row["csi_data"] >= -128) and np.all(row["csi_data"] <= 127) for row in first)
    assert all(np.array_equal(packet["csi_data"], before) for packet, before in zip(packets, original))


def test_robustness_campaign_runs_staged_seed_schedule_without_promotion(monkeypatch, tmp_path):
    module = _load_train_module()
    matrix = {
        "X": np.zeros((2, 6), dtype=np.float32),
        "y": np.asarray([0, 1], dtype=np.int8),
        "feature_names": list(module.DEFAULT_FEATURES),
        "sample_context": {},
    }

    monkeypatch.setattr(module, "ensure_torch_available", lambda: None)
    monkeypatch.setattr(module, "load_training_matrix", lambda **_kwargs: (matrix, None))

    def fake_evaluate(candidate, seed, _matrix, augmented_matrix=None, **_kwargs):
        improvement = 1.0 if candidate["name"] != "baseline_standard" else 0.0
        folds = [
            {
                "fold": f"fold:{idx}",
                "recall": 96.0 + improvement,
                "fp_rate": 4.0 - improvement,
                "f1": 95.0 + improvement,
            }
            for idx in range(7)
        ]
        run = {
            "candidate": candidate,
            "seed": seed,
            "folds": folds,
            "holdout_count": len(folds),
            "seconds": 0.0,
        }
        run["rank_key"] = list(module.robustness_run_rank_key(run))
        return run

    monkeypatch.setattr(module, "evaluate_robustness_candidate", fake_evaluate)
    output = tmp_path / "robustness.json"

    payload = module.run_robustness_experiment(output_path=output)

    assert [stage["name"] for stage in payload["stages"]] == [
        "scalers", "l1_normalization", "feature_augmentation", "packet_augmentation"]
    assert [len(stage["runs"]) for stage in payload["stages"]] == [3, 3, 8, 14]
    assert len(payload["filter"]) == 3
    assert len(payload["final"]) == 2
    assert all(len(summary["seeds"]) == 5 for summary in payload["final"])
    assert payload["decision"]["generalization_qualified"] is True
    assert payload["decision"]["artifacts_changed"] is False
    assert json.loads(output.read_text())["decision"]["deployment_validation"] == "required"

    augmentation_output = tmp_path / "augmentation-only.json"
    augmentation_payload = module.run_robustness_experiment(
        output_path=augmentation_output,
        augmentation_only=True,
    )
    assert augmentation_payload["config"]["augmentation_only"] is True
    assert [stage["name"] for stage in augmentation_payload["stages"]] == [
        "baseline", "feature_augmentation", "packet_augmentation"]
    assert [len(stage["runs"]) for stage in augmentation_payload["stages"]] == [1, 8, 14]


def test_robustness_candidate_evaluates_all_holdouts_and_augments_training_only(monkeypatch):
    module = _load_train_module()
    environments = ("bedroom", "hobby_room", "living_room")
    chips = ("C3", "C5", "C6", "ESP32", "S3")
    rows = []
    labels = []
    context = {key: [] for key in (
        "environment_group", "chip", "session_group", "source_file", "label_name")}
    for environment, chip, label in itertools.product(environments, chips, (0, 1)):
        rows.append([float(label), 0.1, 0.2, 0.3, 0.4, 0.5])
        labels.append(label)
        suffix = f"{environment}-{chip}-{label}"
        context["environment_group"].append(environment)
        context["chip"].append(chip)
        context["session_group"].append(suffix)
        context["source_file"].append(suffix)
        context["label_name"].append("motion" if label else "empty")
    context = {key: np.asarray(values) for key, values in context.items()}
    matrix = {
        "X": np.asarray(rows, dtype=np.float32),
        "y": np.asarray(labels, dtype=np.int8),
        "feature_names": list(module.DEFAULT_FEATURES),
        "sample_context": context,
    }
    observed_train_rows = []

    def fake_train(values, _labels, feature_augmentation=None, **_kwargs):
        observed_train_rows.append(len(values))
        assert feature_augmentation == {"noise_sigma": 0.02}
        return object()

    monkeypatch.setattr(module, "train_model", fake_train)
    monkeypatch.setattr(
        module,
        "predict_probabilities",
        lambda _model, values: np.where(values[:, 0] > 0.0, 0.9, 0.1),
    )
    candidate = module._robustness_candidate(
        "synthetic", feature_augmentation={"noise_sigma": 0.02})

    result = module.evaluate_robustness_candidate(candidate, 42, matrix)

    assert len(result["folds"]) == 8
    assert result["holdout_count"] == 8
    assert {row["dimension"] for row in result["folds"]} == {"environment", "chip"}
    assert len(observed_train_rows) == 8
    assert all(row["test_windows"] > 0 for row in result["folds"])


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


def test_extract_features_applies_hampel_to_l1_stream(monkeypatch):
    module = _load_train_module()
    created = {}

    class FakeL1DeltaTracker:
        def __init__(self, **kwargs):
            created.update(kwargs)

        def process_amplitudes(self, amplitudes, amplitude_count):
            assert amplitude_count == len(amplitudes)

        def copy_deltas_into(self, out):
            out[0] = 0.25
            return 1

    monkeypatch.setattr(module, "L1DeltaTracker", FakeL1DeltaTracker)
    packets = [
        {
            "source_file": "sample.npz",
            "csi_data": [1] * 128,
            "is_motion": False,
        }
        for _ in range(2)
    ]

    X, _, _, _ = module.extract_features(
        packets,
        window_size=2,
        feature_names=["l1_delta"],
        enable_hampel=True,
        hampel_window=5,
        hampel_threshold=3.0,
    )

    assert X[-1, 0] == pytest.approx(0.25)
    assert created["enable_hampel"] is True
    assert created["hampel_window"] == 5
    assert created["hampel_threshold"] == pytest.approx(3.0)


def test_training_and_runtime_feature_streams_match_with_hampel():
    module = _load_train_module()
    window_size = 20
    packets = []
    runtime_rows = []
    detector = module.MLDetector(
        window_size=window_size,
        enable_lowpass=module.ENABLE_LOWPASS_FILTER,
        lowpass_cutoff=module.LOWPASS_CUTOFF,
        enable_hampel=True,
        hampel_window=module.HAMPEL_WINDOW,
        hampel_threshold=module.HAMPEL_THRESHOLD,
    )

    for packet_index in range(45):
        csi_data = [
            ((packet_index * 11 + value_index * 7) % 181) - 90
            for value_index in range(128)
        ]
        packets.append({
            "source_file": "parity.npz",
            "csi_data": csi_data,
            "is_motion": packet_index >= 30,
        })
        detector.process_packet(csi_data, module.DEFAULT_SUBCARRIERS)
        if detector.is_ready():
            runtime_rows.append(list(detector._extract_features()))

    training_rows, _, feature_names, _ = module.extract_features(
        packets,
        window_size=window_size,
        feature_names=list(module.EXPORTED_FEATURE_NAMES),
        enable_lowpass=module.ENABLE_LOWPASS_FILTER,
        lowpass_cutoff=module.LOWPASS_CUTOFF,
        enable_hampel=True,
        hampel_window=module.HAMPEL_WINDOW,
        hampel_threshold=module.HAMPEL_THRESHOLD,
    )

    assert feature_names == list(module.EXPORTED_FEATURE_NAMES)
    assert training_rows == pytest.approx(np.asarray(runtime_rows), abs=1e-6)


def test_select_balanced_shap_indices_is_deterministic_and_class_balanced():
    module = _load_train_module()
    y = np.asarray([0] * 18 + [1] * 6)
    context = {
        "chip": np.asarray(["C3"] * 12 + ["C5"] * 12),
        "session_group": np.asarray([f"session-{idx // 4}" for idx in range(24)]),
    }

    first = module.select_balanced_shap_indices(y, context, max_samples=10, seed=123)
    second = module.select_balanced_shap_indices(y, context, max_samples=10, seed=123)

    assert first.tolist() == second.tolist()
    assert np.sum(y[first] == 0) == 5
    assert np.sum(y[first] == 1) == 5
    assert len(set(context["session_group"][first])) >= 4


def test_cross_validate_shap_uses_disjoint_training_background_and_held_out_samples(monkeypatch):
    module = _load_train_module()
    rows = []
    labels = []
    sessions = []
    for group_idx in range(6):
        for label in (0, 1):
            for repeat in range(2):
                rows.append([float(len(rows)), float(label + repeat)])
                labels.append(label)
                sessions.append(f"session-{group_idx}")
    X = np.asarray(rows, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32)
    groups = np.asarray(sessions)
    context = {
        "chip": np.asarray(["C3" if idx < 12 else "C5" for idx in range(len(X))]),
        "session_group": groups,
        "source_file": np.asarray([f"source-{idx}" for idx in range(len(X))]),
    }

    class IdentityScaler:
        def fit_transform(self, values):
            return np.asarray(values)

        def transform(self, values):
            return np.asarray(values)

    observed = []

    class FakeExplainer:
        def __init__(self, _predict, background, algorithm, seed):
            assert algorithm == "permutation"
            assert seed is not None
            self.background = np.asarray(background)

        def __call__(self, explained):
            explained = np.asarray(explained)
            observed.append((self.background[:, 0].copy(), explained[:, 0].copy()))
            values = np.ones((len(explained), explained.shape[1], 1), dtype=np.float32)
            return SimpleNamespace(values=values)

    monkeypatch.setitem(sys.modules, "shap", SimpleNamespace(Explainer=FakeExplainer))
    monkeypatch.setattr(module, "build_preprocessor", lambda _mode: IdentityScaler())
    monkeypatch.setattr(module, "train_model", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        module,
        "predict_probabilities",
        lambda _model, values: np.full(len(values), 0.5, dtype=np.float32),
    )

    result = module.cross_validate(
        X,
        y,
        n_folds=3,
        groups=groups,
        sample_context=context,
        block_stride=1,
        report_group_keys=(),
        seed=10,
        shap_samples=6,
        shap_feature_names=["row_id", "signal"],
        shap_seed=20,
    )

    assert len(observed) == 3
    assert result["shap_samples"] == 6
    assert result["shap_importance"] == {"row_id": 1.0, "signal": 1.0}
    for background_ids, explained_ids in observed:
        assert set(background_ids).isdisjoint(explained_ids)


def test_build_feature_ablation_dataset_removes_only_requested_column():
    module = _load_train_module()
    context = {"session_group": np.asarray(["a", "b"])}
    dataset = {
        "X": np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        "y": np.asarray([0, 1]),
        "feature_names": ["first", "weak", "last"],
        "sample_context": context,
    }

    candidate = module.build_feature_ablation_dataset(dataset, "weak")

    assert candidate["feature_names"] == ["first", "last"]
    assert candidate["X"].tolist() == [[1.0, 3.0], [4.0, 6.0]]
    assert candidate["sample_context"] is context
    assert dataset["feature_names"] == ["first", "weak", "last"]

    with pytest.raises(ValueError, match="Unknown ablation feature"):
        module.build_feature_ablation_dataset(dataset, "missing")


def test_packet_csi_data_accepts_packet_dicts_and_compact_rows():
    module = _load_train_module()
    row = np.asarray([1, 2, 3], dtype=np.int8)

    assert module.packet_csi_data({"csi_data": row}) is row
    assert module.packet_csi_data(row) is row


def test_feature_ablation_comparison_prints_metric_rows(capsys):
    module = _load_train_module()

    def result(oof_f1, max_fp, worst_recall):
        return {
            "cv": {
                "oof_f1": oof_f1,
                "f1_mean": 90.0,
                "recall_mean": 91.0,
                "fp_rate_mean": 2.0,
                "worst_session_recall": 80.0,
                "worst_session_fp_rate": 10.0,
            },
            "paired": {
                "mean_f1": 98.0,
                "worst_chip_f1": 97.0,
                "max_fp_rate": max_fp,
                "worst_chip_recall": worst_recall,
            },
        }

    module._print_feature_ablation_comparison(
        result(92.4, 1.0, 98.0),
        result(93.5, 0.5, 99.0),
    )
    output = capsys.readouterr().out

    assert "Blocked OOF F1" in output
    assert "Paired max FP rate" in output
    assert "Paired worst-chip recall" in output
    assert "92.40%" in output
    assert "93.50%" in output


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


def _robust_cv_metrics(*, session_recall, session_fp, tail_recall=90.0,
                       tail_fp=5.0, chip_recall=90.0, chip_fp=5.0,
                       oof_f1=93.0):
    return {
        "oof_f1": oof_f1,
        "f1_mean": oof_f1,
        "group_reports": {
            "session_group": {
                "worst_recall": {
                    "recall": session_recall,
                    "positives": 100,
                },
                "worst_fp_rate": {
                    "fp_rate": session_fp,
                    "negatives": 100,
                },
                "tail_recall": {
                    "value": tail_recall,
                    "resolution": 1.0,
                },
                "tail_fp_rate": {
                    "value": tail_fp,
                    "resolution": 1.0,
                },
            },
            "chip": {
                "worst_recall": {
                    "recall": chip_recall,
                    "positives": 200,
                },
                "worst_fp_rate": {
                    "fp_rate": chip_fp,
                    "negatives": 200,
                },
            },
        },
    }


def _session_report(*, recall, fp, tail_recall=None, tail_fp=None):
    return {
        "worst_recall": {"recall": recall, "positives": 100},
        "worst_fp_rate": {"fp_rate": fp, "negatives": 100},
        "tail_recall": {
            "value": recall if tail_recall is None else tail_recall,
            "resolution": 1.0,
        },
        "tail_fp_rate": {
            "value": fp if tail_fp is None else tail_fp,
            "resolution": 1.0,
        },
    }


def test_robust_cv_promotes_worst_group_improvement_without_regressions():
    module = _load_train_module()
    baseline = _robust_cv_metrics(session_recall=80.0, session_fp=10.0)
    candidate = _robust_cv_metrics(session_recall=82.0, session_fp=10.0)

    comparison = module.compare_robust_cv(candidate, baseline)

    assert comparison["passed"] is True
    assert any(
        check["label"] == "CV worst-session recall" and check["improved"]
        for check in comparison["checks"]
    )


def test_robust_cv_rejects_material_worst_group_regression():
    module = _load_train_module()
    baseline = _robust_cv_metrics(session_recall=80.0, session_fp=10.0)
    candidate = _robust_cv_metrics(
        session_recall=84.0,
        session_fp=12.0,
        oof_f1=95.0,
    )

    comparison = module.compare_robust_cv(candidate, baseline)

    assert comparison["passed"] is False
    assert [row["label"] for row in comparison["regressions"]] == [
        "CV worst-session FP"
    ]


def test_candidate_key_prefers_real_session_report():
    module = _load_train_module()
    metrics = _robust_cv_metrics(
        session_recall=70.0,
        session_fp=20.0,
        tail_recall=75.0,
        tail_fp=15.0,
    )
    metrics["group_reports"]["real_session_group"] = _session_report(
        recall=90.0,
        fp=2.0,
        tail_recall=95.0,
        tail_fp=1.0,
    )

    key = module.build_candidate_key(metrics)

    assert key[0] == pytest.approx(95.0)
    assert key[1] == pytest.approx(-1.0)
    assert key[2] == pytest.approx(90.0)
    assert key[4] == pytest.approx(-2.0)


def test_robust_cv_synthetic_improvement_cannot_promote():
    module = _load_train_module()
    baseline = _robust_cv_metrics(session_recall=80.0, session_fp=10.0)
    candidate = _robust_cv_metrics(session_recall=80.0, session_fp=10.0)
    baseline["group_reports"]["synthetic_session_group"] = _session_report(
        recall=70.0, fp=20.0,
    )
    candidate["group_reports"]["synthetic_session_group"] = _session_report(
        recall=90.0, fp=5.0,
    )

    comparison = module.compare_robust_cv(candidate, baseline)

    assert comparison["non_regression"] is True
    assert comparison["material_improvement"] is False
    assert comparison["passed"] is False


def test_robust_cv_synthetic_regression_blocks_promotion():
    module = _load_train_module()
    baseline = _robust_cv_metrics(session_recall=80.0, session_fp=10.0)
    candidate = _robust_cv_metrics(session_recall=85.0, session_fp=10.0)
    baseline["group_reports"]["synthetic_session_group"] = _session_report(
        recall=90.0, fp=5.0,
    )
    candidate["group_reports"]["synthetic_session_group"] = _session_report(
        recall=90.0, fp=12.0,
    )

    comparison = module.compare_robust_cv(candidate, baseline)

    assert comparison["passed"] is False
    assert [row["label"] for row in comparison["regressions"]] == [
        "CV worst-synthetic-session FP",
        "CV tail-synthetic-session FP",
    ]


def test_robust_cv_real_sessions_lead_when_provenance_split_exists():
    module = _load_train_module()
    baseline = _robust_cv_metrics(session_recall=70.0, session_fp=20.0)
    candidate = _robust_cv_metrics(session_recall=60.0, session_fp=30.0)
    baseline["group_reports"]["real_session_group"] = _session_report(
        recall=80.0, fp=10.0,
    )
    candidate["group_reports"]["real_session_group"] = _session_report(
        recall=85.0, fp=10.0,
    )

    comparison = module.compare_robust_cv(candidate, baseline)

    assert comparison["passed"] is True
    assert any(
        check["label"] == "CV worst-session recall" and check["improved"]
        for check in comparison["checks"]
    )


def test_group_report_summarizes_the_five_worst_groups():
    module = _load_train_module()
    y_true = []
    y_prob = []
    groups = []
    for index in range(6):
        groups.extend([f"session-{index}"] * 20)
        y_true.extend([1] * 10 + [0] * 10)
        true_positives = 5 + index
        false_positives = 5 - min(index, 5)
        y_prob.extend(
            [0.9] * true_positives
            + [0.1] * (10 - true_positives)
            + [0.9] * false_positives
            + [0.1] * (10 - false_positives)
        )

    report = module.build_group_report(
        np.asarray(y_true),
        np.asarray(y_prob),
        np.asarray(groups),
    )

    assert report["count"] == 6
    assert len(report["tail_recall"]["groups"]) == module.ROBUST_TAIL_GROUPS
    assert report["tail_recall"]["value"] == pytest.approx(70.0)
    assert report["tail_fp_rate"]["value"] == pytest.approx(30.0)
    assert report["tail_recall"]["resolution"] == pytest.approx(10.0)


def test_search_candidate_key_prefers_passing_gate():
    module = _load_train_module()
    cv = _cv_metrics(session_recall=90.0, chip_recall=88.0, session_fp=8.0, oof_f1=85.0, f1_mean=84.0)
    failing_gate = module.ExportedMLGateResult(
        paired_returncode=1,
        paired_output="paired failed",
    )
    passing_gate = module.ExportedMLGateResult(
        paired_returncode=0,
        paired_output="",
    )

    assert module._search_candidate_key(cv, passing_gate) > module._search_candidate_key(cv, failing_gate)


def test_seed_promotion_rejects_paired_regression():
    module = _load_train_module()
    cv = _cv_metrics(
        session_recall=90.0,
        chip_recall=88.0,
        session_fp=8.0,
        oof_f1=85.0,
        f1_mean=84.0,
    )
    baseline_paired = {
        "pass_count": 4,
        "max_fp_rate": 0.0,
        "worst_chip_recall": 98.0,
        "worst_chip_f1": 99.0,
        "mean_f1": 99.0,
        "mean_recall": 98.0,
    }
    candidate_paired = dict(baseline_paired, worst_chip_recall=97.0)
    baseline_gate = module.ExportedMLGateResult(0, "", baseline_paired)
    candidate_gate = module.ExportedMLGateResult(0, "", candidate_paired)

    assert not module._candidate_beats_baseline(cv, candidate_gate, cv, baseline_gate)


def test_idle_runtime_policy_rejects_isolated_hits_and_counts_alarm_duration():
    module = _load_train_module()
    stride = module.EVALUATION_INTERVAL
    on_hits = MOTION_ON_HITS
    off_hits = MOTION_OFF_HITS

    isolated = [0.0] * (stride * on_hits)
    isolated[stride - 1] = 0.9
    assert module.evaluate_idle_runtime_policy(isolated) == {
        "effective_alarms": 0,
        "false_motion_evaluations": 0,
    }

    # Exactly on_hits consecutive MOTION evaluations, then enough IDLE ticks to
    # clear. Published MOTION starts on the confirming hit and stays through
    # (off_hits - 1) trailing idle evaluations before MOTION_OFF_HITS clears it.
    total_evaluations = on_hits + off_hits
    burst = [0.0] * (stride * total_evaluations)
    for evaluation in range(on_hits):
        burst[(evaluation + 1) * stride - 1] = 0.9
    assert module.evaluate_idle_runtime_policy(burst) == {
        "effective_alarms": 1,
        "false_motion_evaluations": off_hits,
    }


def test_runtime_policy_on_evaluation_ticks_requires_production_hit_count():
    module = _load_train_module()

    assert module.evaluate_runtime_policy_evaluations([True] * (MOTION_ON_HITS - 1)) == {
        "effective_alarms": 0,
        "false_motion_evaluations": 0,
    }
    assert module.evaluate_runtime_policy_evaluations(
        [True] * MOTION_ON_HITS + [False] * MOTION_OFF_HITS
    ) == {
        "effective_alarms": 1,
        "false_motion_evaluations": MOTION_OFF_HITS,
    }


def test_gate_row_passes_uses_stress_targets_for_low_rssi_rows():
    module = _load_train_module()
    row = {
        "recall": 92.8,
        "fp_rate": 4.4,
        "effective_alarms": 7,
    }

    assert module._gate_row_passes(row) is False
    assert module._gate_row_passes({**row, "low_rssi": True}) is True
    assert module._gate_row_passes(
        {"recall": 89.0, "fp_rate": 4.4, "effective_alarms": 0, "low_rssi": True}
    ) is False
    assert module._gate_row_passes(
        {"recall": 99.0, "fp_rate": 11.0, "effective_alarms": 0, "low_rssi": True}
    ) is False
    assert module._gate_row_passes(
        {"recall": 99.0, "fp_rate": 1.0, "effective_alarms": 0}
    ) is True


def test_paired_non_regression_uses_one_evaluation_per_recording_margin():
    module = _load_train_module()

    def summary(fp_rate):
        row = {
            "fp_rate": fp_rate,
            "recall": 99.0,
            "static_presence_eval_count": 100,
            "motion_eval_count": 100,
            "effective_alarms": 0,
        }
        return {
            "pass_count": 1,
            "max_fp_rate": fp_rate,
            "worst_chip_recall": 99.0,
            "worst_chip_f1": 99.0,
            "by_chip": {"C3:selection:capture.npz": row},
        }

    baseline = summary(1.0)

    assert module.paired_result_non_regression(summary(2.0), baseline)
    assert not module.paired_result_non_regression(summary(3.0), baseline)


def test_legacy_paired_fallback_never_selects_synthetic(monkeypatch, tmp_path):
    module = _load_train_module()
    data_dir = tmp_path / "data"
    for label in ("static_presence", "motion"):
        (data_dir / label).mkdir(parents=True)
    for label, filename in (
        ("static_presence", "real-static.npz"),
        ("motion", "real-motion.npz"),
        ("static_presence", "synthetic-static.npz"),
        ("motion", "synthetic-motion.npz"),
    ):
        (data_dir / label / filename).write_bytes(b"npz")
    dataset_info = {
        "files": {
            "static_presence": [
                {
                    "filename": "real-static.npz",
                    "chip": "C3",
                    "subcarriers": 64,
                    "collected_at": "2026-01-01T00:00:00Z",
                    "optimal_pair_motion_file": "real-motion.npz",
                },
                {
                    "filename": "synthetic-static.npz",
                    "chip": "C3",
                    "subcarriers": 64,
                    "collected_at": "2026-07-01T00:00:00Z",
                    "optimal_pair_motion_file": "synthetic-motion.npz",
                    "synthetic": True,
                },
            ],
            "motion": [
                {"filename": "real-motion.npz", "chip": "C3"},
                {
                    "filename": "synthetic-motion.npz",
                    "chip": "C3",
                    "synthetic": True,
                },
            ],
        },
    }
    monkeypatch.setattr(module, "DATA_DIR", data_dir)
    monkeypatch.setattr(module, "load_dataset_info", lambda: dataset_info)
    monkeypatch.setattr(module, "_load_npz_packets_cached", lambda path: [path.name])

    pairs = list(module._iter_paired_chip_packets(chips=("C3",)))

    assert pairs == [("C3", ["real-static.npz"], ["real-motion.npz"], False)]


def test_paired_gate_ranking_prefers_lower_fp_before_cv():
    module = _load_train_module()
    quieter = {
        "pass_count": 4,
        "max_fp_rate": 0.0,
        "worst_chip_recall": 97.0,
        "worst_chip_f1": 97.0,
        "mean_f1": 97.0,
        "mean_recall": 97.0,
    }
    noisier = {
        "pass_count": 4,
        "max_fp_rate": 2.0,
        "worst_chip_recall": 99.0,
        "worst_chip_f1": 99.0,
        "mean_f1": 99.0,
        "mean_recall": 99.0,
    }

    assert module._paired_gate_key(quieter) > module._paired_gate_key(noisier)


def test_architecture_ranking_prefers_robust_cv_after_equal_safety_passes():
    module = _load_train_module()

    def result(max_fp, oof_f1):
        return {
            "params": 100,
            "cv": {"oof_f1": oof_f1, "f1_mean": oof_f1},
            "paired": {
                "pass_count": 4,
                "max_fp_rate": max_fp,
                "worst_chip_recall": 98.0,
                "worst_chip_f1": 99.0,
            },
        }

    quieter = result(0.0, 90.0)
    noisier = result(1.5, 95.0)

    assert module.architecture_campaign_rank_key(noisier) < module.architecture_campaign_rank_key(quieter)


def test_parse_fp_weight_sweep_validates_and_deduplicates():
    module = _load_train_module()

    assert module.parse_fp_weight_sweep("1,1.5,2,1.5") == [1.0, 1.5, 2.0]
    with pytest.raises(module.argparse.ArgumentTypeError, match="positive"):
        module.parse_fp_weight_sweep("1,0")


def test_deployment_candidate_requires_paired_non_regression():
    module = _load_train_module()

    def result(worst_recall, max_fp, oof_f1=93.0):
        return {
            "cv": {
                "oof_f1": oof_f1,
                "f1_mean": oof_f1,
                "worst_session_recall": 90.0,
                "worst_chip_recall": 95.0,
                "worst_session_fp_rate": 5.0,
            },
            "paired": {
                "pass_count": 4,
                "max_fp_rate": max_fp,
                "worst_chip_recall": worst_recall,
                "worst_chip_f1": 99.0,
                "mean_f1": 99.0,
                "mean_recall": worst_recall,
            },
        }

    baseline = result(98.0, 1.0)
    lower_recall = result(97.0, 0.0)
    quieter = result(98.0, 0.0, oof_f1=94.0)
    fp_regression = result(98.0, 2.0)

    assert not module.deployment_candidate_beats_baseline(lower_recall, baseline)
    assert module.deployment_candidate_beats_baseline(quieter, baseline)
    assert not module.deployment_candidate_beats_baseline(fp_regression, baseline)


def test_paired_non_regression_prioritizes_more_passing_chips():
    module = _load_train_module()
    broken_baseline = {
        "pass_count": 2,
        "max_fp_rate": 100.0,
        "worst_chip_recall": 100.0,
        "worst_chip_f1": 66.0,
    }
    valid_candidate = {
        "pass_count": 5,
        "max_fp_rate": 3.0,
        "worst_chip_recall": 98.0,
        "worst_chip_f1": 98.0,
    }

    assert module.paired_result_non_regression(valid_candidate, broken_baseline)
    assert not module.paired_result_non_regression(broken_baseline, valid_candidate)


def test_fp_weight_campaign_is_multi_seed_and_non_promoting_by_default(monkeypatch, tmp_path):
    module = _load_train_module()
    context = {module.DEFAULT_PRIMARY_GROUP_KEY: np.asarray(["a", "b"])}
    matrix = {
        "X": np.asarray([[0.0], [1.0]], dtype=np.float32),
        "y": np.asarray([0, 1], dtype=np.int8),
        "feature_names": ["feature"],
        "sample_context": context,
        "sample_weights": np.ones(2, dtype=np.float32),
        "stats": {"chips": ["C3"]},
    }
    observed = []

    def fake_candidate(name, layers, seed, dataset, scaler, batch_size, fp_weight):
        observed.append((fp_weight, seed))
        max_fp = 0.0 if fp_weight == 3.0 else 1.5
        return {
            "name": name,
            "seed": seed,
            "fp_weight": fp_weight,
            "layers": list(layers),
            "architecture": "1 -> 2 -> 1",
            "params": 10,
            "weight_kb": 0.1,
            "flops": 5,
            "inference_us": 1.0,
            "cv": {"oof_f1": 92.0, "f1_mean": 92.0},
            "paired": {
                "pass_count": 4,
                "max_fp_rate": max_fp,
                "worst_chip_recall": 98.0,
                "worst_chip_f1": 99.0,
            },
        }

    monkeypatch.setattr(module, "ensure_torch_available", lambda: None)
    monkeypatch.setattr(module, "describe_torch_device", lambda: "cpu")
    monkeypatch.setattr(module, "read_exported_seed", lambda: 123)
    monkeypatch.setattr(module, "load_training_matrix", lambda **kwargs: (matrix, None))
    monkeypatch.setattr(
        module,
        "apply_positive_chip_boost",
        lambda weights, context, labels, boost: (weights, {}),
    )
    monkeypatch.setattr(module, "evaluate_architecture_candidate", fake_candidate)
    monkeypatch.setattr(
        module,
        "train_all",
        lambda **kwargs: pytest.fail("non-promoting campaign must not export"),
    )
    output_path = tmp_path / "fp_weights.json"

    result = module.experiment_fp_weights(
        fp_weights=[2.0, 3.0],
        hidden_layers=[2],
        output_path=output_path,
        promote_winner=False,
    )

    payload = json.loads(output_path.read_text())
    assert result == 0
    assert payload["promotion"]["winner"] == "fp_weight=3"
    assert payload["promotion"]["clear_winner"] is True
    assert {seed for _, seed in observed} >= set(module.DEFAULT_EXPERIMENT_FINAL_SEEDS)


def test_normal_training_evaluates_deployment_without_exporting(monkeypatch):
    module = _load_train_module()
    context = {
        module.DEFAULT_PRIMARY_GROUP_KEY: np.asarray(["a", "b"]),
        module.DEFAULT_BLOCK_GROUP_KEY: np.asarray(["one", "two"]),
    }
    matrix = {
        "X": np.asarray([[0.0], [1.0]], dtype=np.float32),
        "y": np.asarray([0, 1], dtype=np.int8),
        "feature_names": ["feature"],
        "sample_context": context,
        "sample_weights": np.ones(2, dtype=np.float32),
        "stats": {
            "chips": ["C3"],
            "labels": {"idle": 1, "motion": 1},
            "total": 2,
            "session_groups": ["a", "b"],
            "environment_groups": [],
        },
    }

    class IdentityScaler:
        def fit_transform(self, values):
            return values

    paired = {
        "by_chip": {"C3": {}},
        "pass_count": 1,
        "max_fp_rate": 0.0,
        "worst_chip_recall": 99.0,
    }
    quiet = {
        "by_dataset": {"C3:selection:quiet.npz": {}},
        "max_fp_rate": 0.0,
        "total_effective_alarms": 0,
        "max_effective_alarms": 0,
        "passed": True,
    }
    cv = {"oof_f1": 90.0, "f1_mean": 90.0, "f1_std": 0.0}

    monkeypatch.setattr(module, "ensure_torch_available", lambda: None)
    monkeypatch.setattr(module, "describe_torch_device", lambda: "cpu")
    monkeypatch.setattr(module, "set_global_determinism", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "load_training_matrix", lambda **kwargs: (matrix, None))
    monkeypatch.setattr(module, "apply_positive_chip_boost", lambda *args: (matrix["sample_weights"], {}))
    monkeypatch.setattr(module, "cross_validate", lambda *args, **kwargs: dict(cv))
    monkeypatch.setattr(module, "print_cv_summary", lambda results: None)
    monkeypatch.setattr(module, "build_preprocessor", lambda mode: IdentityScaler())
    monkeypatch.setattr(module, "train_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(module, "evaluate_paired_gate", lambda *args, **kwargs: paired)
    monkeypatch.setattr(module, "evaluate_quiet_gate", lambda *args, **kwargs: quiet)
    monkeypatch.setattr(
        module,
        "export_micropython",
        lambda *args, **kwargs: pytest.fail("artifacts must remain unchanged"),
    )

    result, seed, summary = module.train_all(
        seed=123,
        feature_names=["feature"],
        export_artifacts=False,
        evaluate_deployment=True,
    )

    assert result == 0
    assert seed == 123
    assert summary["paired"] is paired
    assert summary["quiet"] is quiet
    assert "long" not in summary


def test_force_export_bypasses_failed_deployment_gate(monkeypatch, tmp_path):
    module = _load_train_module()
    context = {
        module.DEFAULT_PRIMARY_GROUP_KEY: np.asarray(["a", "b"]),
        module.DEFAULT_BLOCK_GROUP_KEY: np.asarray(["one", "two"]),
    }
    matrix = {
        "X": np.asarray([[0.0], [1.0]], dtype=np.float32),
        "y": np.asarray([0, 1], dtype=np.int8),
        "feature_names": ["feature"],
        "sample_context": context,
        "sample_weights": np.ones(2, dtype=np.float32),
        "stats": {
            "chips": ["C3"],
            "labels": {"idle": 1, "motion": 1},
            "total": 2,
            "session_groups": ["a", "b"],
            "environment_groups": [],
        },
    }

    class IdentityScaler:
        def fit_transform(self, values):
            return values

    failing_paired = {
        "by_chip": {"C3": {}},
        "pass_count": 0,
        "max_fp_rate": 17.0,
        "worst_chip_recall": 98.0,
    }
    failing_quiet = {
        "by_dataset": {"C3:selection:quiet.npz": {}},
        "max_fp_rate": 6.0,
        "total_effective_alarms": 2,
        "max_effective_alarms": 2,
        "passed": False,
    }
    cv = {"oof_f1": 90.0, "f1_mean": 90.0, "f1_std": 0.0}
    exports = []

    def _unavailable_baseline(**_kwargs):
        raise FileNotFoundError("no exported baseline")

    monkeypatch.setattr(module, "ensure_torch_available", lambda: None)
    monkeypatch.setattr(module, "describe_torch_device", lambda: "cpu")
    monkeypatch.setattr(module, "set_global_determinism", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "load_training_matrix", lambda **kwargs: (matrix, None))
    monkeypatch.setattr(module, "apply_positive_chip_boost", lambda *args: (matrix["sample_weights"], {}))
    monkeypatch.setattr(module, "cross_validate", lambda *args, **kwargs: dict(cv))
    monkeypatch.setattr(module, "print_cv_summary", lambda results: None)
    monkeypatch.setattr(module, "build_preprocessor", lambda mode: IdentityScaler())
    monkeypatch.setattr(module, "train_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(module, "evaluate_paired_gate", lambda *args, **kwargs: failing_paired)
    monkeypatch.setattr(module, "evaluate_quiet_gate", lambda *args, **kwargs: failing_quiet)
    monkeypatch.setattr(module, "evaluate_exported_paired_gate", _unavailable_baseline)
    monkeypatch.setattr(
        module,
        "select_regression_subset_indices",
        lambda *args, **kwargs: np.asarray([0, 1]),
    )
    monkeypatch.setattr(module, "GENERATED_DATA_DIR", tmp_path)
    monkeypatch.setattr(
        module,
        "export_micropython",
        lambda *args, **kwargs: exports.append("micropython") or 1024,
    )
    monkeypatch.setattr(
        module,
        "export_cpp_weights",
        lambda *args, **kwargs: exports.append("cpp") or 1024,
    )
    monkeypatch.setattr(
        module,
        "export_test_data",
        lambda *args, **kwargs: exports.append("test_data") or 2,
    )

    result, seed, summary = module.train_all(
        seed=123,
        feature_names=["feature"],
        export_artifacts=True,
        evaluate_deployment=True,
        force_export=True,
    )

    assert result == 0
    assert seed == 123
    assert summary["paired"] is failing_paired
    assert exports == ["micropython", "cpp", "test_data"]


def test_load_all_data_propagates_sensing_contract_errors(monkeypatch, tmp_path):
    """A file emptied by format filtering must stop training, not be skipped."""
    module = _load_train_module()
    data_dir = tmp_path / "data"
    (data_dir / "motion").mkdir(parents=True)
    (data_dir / "motion" / "bad.npz").write_bytes(b"npz")
    monkeypatch.setattr(module, "DATA_DIR", data_dir)
    monkeypatch.setattr(module, "load_dataset_info", lambda: {"files": {}})
    monkeypatch.setattr(module, "get_file_metadata", lambda info: {})
    monkeypatch.setattr(module, "load_npz_as_packets", lambda path: [])

    with pytest.raises(RuntimeError, match="no HT20/HT-LTF/64-SC sensing packets"):
        module.load_all_data()


def test_features_cli_rejects_unknown_names(monkeypatch, capsys):
    module = _load_train_module()
    monkeypatch.setattr(
        "sys.argv",
        ["train_ml_model.py", "--features", "turb_zcr,bogus", "--no-export"],
    )

    assert module.main() == 1
    assert "unknown feature(s): bogus" in capsys.readouterr().out


def test_features_cli_blocks_candidate_features_on_exporting_flows(monkeypatch, capsys):
    module = _load_train_module()
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_ml_model.py",
            "--features", "l1_delta_cv,l1_delta",
            "--seed-search-until-improvement", "2",
        ],
    )

    assert module.main() == 1
    out = capsys.readouterr().out
    assert "without a C++ extractor id" in out
    assert "l1_delta_cv" in out


def test_features_cli_rejects_duplicates(monkeypatch, capsys):
    module = _load_train_module()
    monkeypatch.setattr(
        "sys.argv",
        ["train_ml_model.py", "--features", "l1_delta,l1_delta", "--no-export"],
    )

    assert module.main() == 1
    assert "duplicate" in capsys.readouterr().out


def test_force_promote_cli_requires_explicit_seed(monkeypatch, capsys):
    module = _load_train_module()
    monkeypatch.setattr(
        "sys.argv",
        ["train_ml_model.py", "--force-promote"],
    )

    assert module.main() == 1
    assert "--force-promote requires an explicit --seed" in capsys.readouterr().out


def test_train_until_improvement_ranks_candidates_when_baseline_is_broken(monkeypatch):
    module = _load_train_module()

    baseline_cv = _cv_metrics(session_recall=60.0, chip_recall=70.0, session_fp=15.0, oof_f1=80.0, f1_mean=79.0)
    candidate_cv_a = _cv_metrics(session_recall=61.0, chip_recall=71.0, session_fp=14.0, oof_f1=80.5, f1_mean=79.5)
    candidate_cv_b = _cv_metrics(session_recall=64.0, chip_recall=74.0, session_fp=12.0, oof_f1=82.0, f1_mean=81.0)

    baseline_gate = module.ExportedMLGateResult(
        paired_returncode=1,
        paired_output="paired failed",
        paired_metrics={
            "pass_count": 2,
            "max_fp_rate": 12.0,
            "worst_chip_recall": 80.0,
            "worst_chip_f1": 78.0,
            "mean_f1": 80.0,
            "mean_recall": 82.0,
        },
    )
    candidate_gate_a = module.ExportedMLGateResult(
        paired_returncode=0,
        paired_output="",
        paired_metrics={
            "pass_count": 2,
            "max_fp_rate": 11.0,
            "worst_chip_recall": 81.0,
            "worst_chip_f1": 79.0,
            "mean_f1": 81.0,
            "mean_recall": 83.0,
        },
    )
    candidate_gate_b = module.ExportedMLGateResult(
        paired_returncode=0,
        paired_output="",
        paired_metrics={
            "pass_count": 3,
            "max_fp_rate": 9.0,
            "worst_chip_recall": 84.0,
            "worst_chip_f1": 82.0,
            "mean_f1": 84.0,
            "mean_recall": 86.0,
        },
    )

    train_calls = iter(
        [
            (0, 111, baseline_cv),
            (0, 201, candidate_cv_a),
            (0, 202, candidate_cv_b),
        ]
    )
    no_holdout_gate = module.ExportedMLGateResult(1, "")
    gate_calls = iter([
        baseline_gate,
        no_holdout_gate,
        candidate_gate_a,
        candidate_gate_b,
        no_holdout_gate,
    ])

    monkeypatch.setattr(module, "ensure_torch_available", lambda: object())
    monkeypatch.setattr(module, "describe_torch_device", lambda: "cpu")
    monkeypatch.setattr(module, "read_exported_seed", lambda: 111)
    monkeypatch.setattr(module, "train_all", lambda **kwargs: next(train_calls))
    monkeypatch.setattr(
        module,
        "run_exported_ml_gates",
        lambda **_kwargs: next(gate_calls),
    )

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
