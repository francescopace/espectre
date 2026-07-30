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


def test_parse_combination_specs_rejects_duplicates() -> None:
    with pytest.raises(bench.BenchmarkError, match="features must differ"):
        bench.parse_combination_specs(["a,a"], 2, "pair")


def test_resolve_candidate_combinations_prefers_explicit_triplets() -> None:
    args = argparse.Namespace(
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
        calibration_packets=1000,
        window_packets=100,
        evaluation_interval=25,
        sample_limit=64,
    ) == 37


def test_startup_evaluation_limit_honors_detector_storage_cap() -> None:
    assert replay.startup_evaluation_limit(
        calibration_packets=10_000,
        window_packets=100,
        evaluation_interval=25,
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
