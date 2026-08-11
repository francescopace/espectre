import argparse

import numpy as np
import pytest

from tools import benchmark_classic_candidate_pairs as bench
from tools import replay_classic_candidates as replay


def test_parse_combination_specs_accepts_pairs_and_triplets() -> None:
    assert bench.parse_combination_specs(["a,b"], 2, "pair") == [("a", "b")]
    assert bench.parse_combination_specs(["a,b,c"], 3, "triple") == [
        ("a", "b", "c")
    ]


def test_resolve_candidate_combinations_accepts_single_features() -> None:
    args = argparse.Namespace(
        feature=["candidate"],
        pair=[],
        triple=[],
        all_runtime_triplets=False,
        all_host_triplets=False,
    )

    assert bench.resolve_candidate_combinations(args) == [("candidate",)]


def test_parse_combination_specs_rejects_duplicates() -> None:
    with pytest.raises(bench.BenchmarkError, match="features must differ"):
        bench.parse_combination_specs(["a,a"], 2, "pair")


def test_classic_replay_accepts_up_to_three_features() -> None:
    assert replay.parse_feature_sets(["a", "a,b", "a,b,c"]) == [
        ("a",),
        ("a", "b"),
        ("a", "b", "c"),
    ]

    with pytest.raises(replay.ReplayError, match="expected 1, 2, or 3"):
        replay.parse_feature_sets(["a,b,c,d"])


def test_classic_replay_exposes_individual_stress_scenarios() -> None:
    assert replay.STRESS_SCENARIOS["combined"] == (
        "base",
        "drift",
        "burst-loss",
    )


def test_classic_replay_nonlinear_fusion_surfaces() -> None:
    rows = np.asarray([[2.0, 3.0], [-1.0, 4.0]], dtype=np.float64)

    np.testing.assert_allclose(
        replay.transform_fusion_rows(rows, replay.FUSION_LINEAR),
        rows,
    )
    np.testing.assert_allclose(
        replay.transform_fusion_rows(rows, replay.FUSION_INTERACTION),
        [[2.0, 3.0, 6.0], [-1.0, 4.0, -4.0]],
    )
    np.testing.assert_allclose(
        replay.transform_fusion_rows(rows, replay.FUSION_QUADRATIC),
        [[2.0, 3.0, 6.0, 4.0, 9.0], [-1.0, 4.0, -4.0, 1.0, 16.0]],
    )

    with pytest.raises(replay.ReplayError, match="exactly two"):
        replay.transform_fusion_rows(
            np.asarray([[1.0]], dtype=np.float64),
            replay.FUSION_INTERACTION,
        )


def test_resolve_candidate_combinations_prefers_explicit_triplets() -> None:
    args = argparse.Namespace(
        feature=[],
        pair=[],
        triple=["x,y,z"],
        all_runtime_triplets=False,
        all_host_triplets=False,
    )
    assert bench.resolve_candidate_combinations(args) == [("x", "y", "z")]


def test_fit_lda_projection_supports_three_features() -> None:
    x = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.2, 0.1],
            [1.0, 1.0, 0.9],
            [1.1, 1.2, 1.0],
        ],
        dtype=np.float64,
    )
    y = np.asarray([0, 0, 1, 1], dtype=np.int8)

    weights, pooled = bench.fit_lda_projection(x, y)

    assert weights.shape == (3,)
    assert pooled.shape == (3, 3)
    assert np.all(np.isfinite(weights))
    assert np.all(np.isfinite(pooled))


def test_cumulative_threshold_search_matches_direct_masks() -> None:
    scores = np.asarray([-2.0, -1.0, -0.4, 0.2, -0.8, 0.1, 0.9, 1.8])
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8)
    weights = np.asarray([1.0, 1.5, 0.5, 1.0, 0.7, 1.3, 1.1, 0.9])
    sessions = np.asarray(["i1", "i1", "i2", "i2", "m1", "m1", "m2", "m2"])

    threshold, metrics = replay.choose_base_threshold(
        scores,
        labels,
        weights,
        session=sessions,
        fp_target=30.0,
    )

    probabilities = 1.0 / (1.0 + np.exp(-scores))
    candidates = np.unique(
        np.quantile(probabilities, np.linspace(0.001, 0.999, 999))
    )
    expected = None
    expected_recall = -1.0
    for candidate in candidates:
        predicted = probabilities > candidate
        tp = float(weights[(labels == 1) & predicted].sum())
        fn = float(weights[(labels == 1) & ~predicted].sum())
        fp = float(weights[(labels == 0) & predicted].sum())
        tn = float(weights[(labels == 0) & ~predicted].sum())
        recall = tp / (tp + fn) if tp > 0.0 else 0.0
        fp_rate = 100.0 * fp / (fp + tn)
        if fp_rate <= 30.0 and recall > expected_recall:
            expected = float(candidate)
            expected_recall = recall

    assert threshold == pytest.approx(expected)
    assert metrics["recall"] == pytest.approx(100.0 * expected_recall)


def test_safe_auc_reports_missing_optional_dependency(monkeypatch) -> None:
    def fail_import(_name: str):
        raise ImportError("missing optional dependency")

    monkeypatch.setattr(bench, "import_module", fail_import)

    with pytest.raises(bench.BenchmarkError, match="requirements-ml.txt"):
        bench.safe_auc(
            np.asarray([0, 1], dtype=np.int8),
            np.asarray([0.1, 0.9], dtype=np.float64),
        )


def test_correlation_summary_reports_pairwise_triplet_redundancy() -> None:
    values = np.asarray(
        [
            [1.0, 2.0, 1.0],
            [2.0, 4.0, 0.0],
            [3.0, 6.0, -1.0],
            [4.0, 8.0, -2.0],
        ],
        dtype=np.float64,
    )

    summary = bench.correlation_summary(values)

    assert summary["corr_abs"] == pytest.approx(1.0)
    assert summary["corr_abs_mean"] > 0.5
    assert set(summary["corr_pairs"]) == {"0-1", "0-2", "1-2"}


def test_startup_evaluation_limit_matches_nominal_runtime_calibration() -> None:
    assert replay.startup_evaluation_limit(
        calibration_duration_ms=10_000,
        window_size_ms=1_000,
        evaluation_interval_ms=250,
        sample_limit=64,
    ) == 37


def test_startup_evaluation_limit_honors_detector_storage_cap() -> None:
    assert replay.startup_evaluation_limit(
        calibration_duration_ms=100_000,
        window_size_ms=1_000,
        evaluation_interval_ms=250,
        sample_limit=64,
    ) == 64


def test_session_centering_uses_only_runtime_startup_prefix() -> None:
    scores = np.asarray([0.0, 1.0, 100.0, 4.0, 5.0], dtype=np.float64)
    labels = np.asarray([0, 0, 0, 1, 1], dtype=np.int8)
    sessions = np.asarray(["a"] * 5, dtype=object)

    centered = replay.session_centered_replay_scores(
        scores,
        labels,
        sessions,
        startup_strength=0.5,
        startup_sample_limit=2,
    )

    expected_shift = 0.5 * np.quantile([0.0, 1.0], 0.95)
    assert centered == pytest.approx(scores - expected_shift)


def test_robust_startup_calibration_maps_reference_standardized_threshold() -> None:
    rows = np.asarray([[6.0], [10.0], [14.0]], dtype=np.float64)
    coefficients = {
        "center": np.asarray([0.0]),
        "scale": np.asarray([1.0]),
        "weight": np.asarray([1.0]),
        "intercept": 0.0,
    }
    policy = replay.calibration_policy(
        replay.CALIBRATION_ROBUST_LOGIT,
        startup_strength=1.0,
        robust_scale_floor_ratio=0.25,
    )
    references = {
        "idle_q95_logit": 0.0,
        "final_location": 0.0,
        "final_scale": 2.0,
    }

    threshold = replay.calibrated_startup_threshold(
        rows,
        np.asarray([6.0, 10.0, 14.0]),
        coefficients,
        replay.probability(1.0),
        policy,
        references,
        startup_sample_limit=3,
    )

    # Reference z=0.5 maps to median=10 plus 0.5 * session IQR=4.
    assert replay.probability_logit(threshold) == pytest.approx(12.0)


def test_robust_oof_calibration_standardizes_raw_session_logits() -> None:
    scores = np.asarray([6.0, 10.0, 14.0, 18.0], dtype=np.float64)
    labels = np.asarray([0, 0, 0, 1], dtype=np.int8)
    sessions = np.asarray(["a"] * 4, dtype=object)
    coefficients = {
        "center": np.asarray([0.0]),
        "scale": np.asarray([1.0]),
        "weight": np.asarray([1.0]),
        "intercept": 0.0,
    }
    policy = replay.calibration_policy(
        replay.CALIBRATION_ROBUST_LOGIT,
        startup_strength=1.0,
        robust_scale_floor_ratio=0.25,
    )

    calibrated = replay.calibrated_replay_scores(
        scores,
        scores.reshape(-1, 1),
        labels,
        sessions,
        coefficients,
        policy,
        {"oof_location": 0.0, "oof_scale": 2.0},
        startup_sample_limit=3,
    )

    np.testing.assert_allclose(calibrated, [-1.0, 0.0, 1.0, 2.0])


def test_feature_startup_calibration_shifts_each_feature_before_fusion() -> None:
    rows = np.asarray([[3.0, 5.0], [3.0, 5.0]], dtype=np.float64)
    coefficients = {
        "center": np.asarray([0.0, 0.0]),
        "scale": np.asarray([2.0, 4.0]),
        "weight": np.asarray([2.0, 4.0]),
        "intercept": 0.0,
    }
    policy = replay.calibration_policy(
        replay.CALIBRATION_FEATURE_SHIFT,
        startup_strength=0.5,
        feature_startup_quantile=0.5,
    )
    references = {
        "idle_q95_logit": 0.0,
        "feature_location": [1.0, 2.0],
    }

    threshold = replay.calibrated_startup_threshold(
        rows,
        np.asarray([8.0, 8.0]),
        coefficients,
        0.5,
        policy,
        references,
        startup_sample_limit=2,
    )

    # w/scale=[1, 1], so half of the contribution shift 8-3 is 2.5 logits.
    assert replay.probability_logit(threshold) == pytest.approx(2.5)


def test_guarded_upward_recovery_is_bounded_by_startup_threshold() -> None:
    rows = np.asarray(
        [[value] for value in ([-4.0] * 240 + [-1.3] * 60 + [-0.5] * 40)],
        dtype=np.float64,
    )
    coefficients = {
        "center": np.asarray([0.0]),
        "scale": np.asarray([1.0]),
        "weight": np.asarray([1.0]),
        "intercept": 0.0,
    }
    policy = replay.calibration_policy(
        replay.CALIBRATION_GUARDED_UPWARD,
        startup_strength=0.5,
        upward_blocks=3,
        upward_quantile=0.95,
        upward_max_positive_fraction=0.5,
    )
    initial_threshold = replay.probability(1.0)

    metrics = replay.replay_one_stream(
        rows,
        coefficients,
        base_threshold=initial_threshold,
        idle_q95=0.0,
        startup_strength=0.5,
        startup_sample_limit=37,
        settle_margin_logits=2.8,
        initial_threshold=initial_threshold,
        calibration=policy,
        references={"idle_q95_logit": 0.0},
    )

    assert metrics["threshold"] == pytest.approx(initial_threshold)
    assert metrics["threshold"] <= metrics["initial_threshold"]
    assert metrics["positive_count"] == 0


def test_guarded_upward_recovery_freezes_on_predominantly_positive_blocks() -> None:
    rows = np.asarray(
        [[value] for value in ([-4.0] * 240 + [2.0] * 100)],
        dtype=np.float64,
    )
    coefficients = {
        "center": np.asarray([0.0]),
        "scale": np.asarray([1.0]),
        "weight": np.asarray([1.0]),
        "intercept": 0.0,
    }
    policy = replay.calibration_policy(
        replay.CALIBRATION_GUARDED_UPWARD,
        startup_strength=0.5,
        upward_blocks=3,
        upward_quantile=0.95,
        upward_max_positive_fraction=0.5,
    )
    initial_threshold = replay.probability(1.0)

    metrics = replay.replay_one_stream(
        rows,
        coefficients,
        base_threshold=initial_threshold,
        idle_q95=0.0,
        startup_strength=0.5,
        startup_sample_limit=37,
        settle_margin_logits=2.8,
        initial_threshold=initial_threshold,
        calibration=policy,
        references={"idle_q95_logit": 0.0},
    )

    assert metrics["threshold"] < initial_threshold
    assert metrics["positive_count"] >= 100


def test_threshold_free_fit_roles_do_not_include_holdout() -> None:
    assert bench.FIT_ROLES == ("train",)
    assert bench.HOLDOUT_ROLE not in bench.PRIMARY_ROLES


def test_train_empty_rows_are_grouped_idle_hard_negatives() -> None:
    corpus = {
        "x": np.asarray([[1.0, 2.0]], dtype=np.float64),
        "y": np.asarray([1], dtype=np.int8),
        "session": np.asarray(["pair-a"], dtype=object),
        "chip": np.asarray(["C6"], dtype=object),
        "deoverlapped": np.asarray([True]),
    }
    empties = [
        {
            "session": "empty-train",
            "chip": "C5",
            "role": "train",
            "path": "empty-train.npz",
        },
        {
            "session": "empty-holdout",
            "chip": "C3",
            "role": "holdout",
            "path": "empty-holdout.npz",
        },
    ]
    cache = {
        "empty-train.npz": {
            "rows": np.asarray([[3.0, 4.0]], dtype=np.float64),
            "deoverlapped": np.asarray([True]),
        },
        "empty-holdout.npz": {
            "rows": np.asarray([[100.0, 200.0]], dtype=np.float64),
            "deoverlapped": np.asarray([True]),
        },
    }

    augmented = replay.append_training_empty_rows(
        corpus,
        empties,
        cache,
        {"a": 0, "b": 1},
        ("a", "b"),
    )

    np.testing.assert_allclose(augmented["x"], [[1.0, 2.0], [3.0, 4.0]])
    assert augmented["y"].tolist() == [1, 0]
    assert augmented["session"].tolist() == ["pair-a", "empty-train"]


def test_load_feature_matrix_uses_time_aware_replay_for_host_features(
    monkeypatch, tmp_path
) -> None:
    source_path = tmp_path / "motion.npz"
    record = {
        "path": source_path,
        "label_name": "motion",
        "is_motion": True,
        "pair_id": "pair-a",
        "synthetic": False,
    }
    seen = {}

    monkeypatch.setattr(
        bench.train_ml_model,
        "_load_training_file_records",
        lambda **_kwargs: ([record], {"chips": ["C6"]}),
    )

    def fake_build_replay_cache(paths, feature_names, *, quiet):
        seen["paths"] = list(paths)
        seen["feature_names"] = list(feature_names)
        seen["quiet"] = quiet
        return {
            str(source_path): {
                "rows": np.asarray([[1.0, 2.0]], dtype=np.float32),
                "deoverlapped": np.asarray([True]),
            }
        }

    monkeypatch.setattr(
        bench.candidate_replay,
        "build_replay_cache",
        fake_build_replay_cache,
    )

    matrix = bench.load_feature_matrix(
        [("phase_resid_lag_ratio", "turb_autocorr")]
    )

    assert matrix["training_sample_contract"] == "time_aware_replay_tick"
    assert seen["paths"] == [source_path]
    assert seen["quiet"] is True
    np.testing.assert_allclose(matrix["X"], [[1.0, 2.0]])
