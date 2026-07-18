"""
ESPectre - Dataset Quality Validation Tests

Tests for tools/validate_dataset_quality.py.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


VALIDATOR_PATH = Path(__file__).resolve().parents[2] / "tools" / "validate_dataset_quality.py"


def _load_validator_module():
    """Load the validator script directly from the tools directory."""
    spec = importlib.util.spec_from_file_location("dataset_quality_validation", VALIDATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_empty_and_static_use_independent_classic_calibration(monkeypatch) -> None:
    module = _load_validator_module()

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

    empty_csi = np.zeros((4, 128), dtype=np.int8)
    static_csi = np.ones((4, 128), dtype=np.int8)

    def fake_resolve(entry, label):
        return Path(f"/tmp/{entry['filename']}")

    def fake_load(path, npz_cache):
        csi = empty_csi if path.name.startswith("empty") else static_csi
        return {"csi_data": csi}, "csi_data"

    def fake_stats(csi_data, packet_rate_pps=100.0, *, calibration_cache=None, cache_key=None):
        assert packet_rate_pps == 100.0
        if csi_data is empty_csi:
            margins = np.array([-2.0, -1.0, -2.0, -1.0])
            threshold = 0.3
            motion_count = 0
        else:
            margins = np.array([-1.0, 0.5, -1.0, 0.5])
            threshold = 0.6
            motion_count = 2
        median = float(np.median(margins))
        return {
            "threshold": threshold,
            "eval_count": len(margins),
            "motion_count": motion_count,
            "fp_rate": motion_count / len(margins),
            "margin_median": median,
            "margin_mad": float(np.median(np.abs(margins - median))),
            "margin_q95": float(np.quantile(margins, 0.95)),
            "margin_q99": float(np.quantile(margins, 0.99)),
            "margin_drift": 0.0,
            "margin_series": margins,
            "block_margins": np.array([median]),
            "burst_count": motion_count,
            "bursts_per_minute": 0.0,
            "longest_burst_seconds": 0.0,
            "eval_seconds": 1.0,
            "score": 0.0,
        }

    monkeypatch.setattr(module, "_resolve_dataset_entry_path", fake_resolve)
    monkeypatch.setattr(module, "_load_cached_or_npz", fake_load)
    monkeypatch.setattr(module, "_classic_self_baseline_stats", fake_stats)

    results, empty_rows, presence_rows = module.validate_empty_sanity(
        dataset_info, npz_cache={}
    )
    empty_result = next(r for r in results if r.name == "empty_quality/empty_a.npz")
    presence_result = next(
        r for r in results if r.name == "presence_quality/static_a.npz"
    )

    assert empty_result.status == "PASS"
    assert presence_result.status == "WARN"
    assert empty_rows[0]["baseline"]["threshold"] == pytest.approx(0.3)
    assert presence_rows[0]["baseline"]["threshold"] == pytest.approx(0.6)
    assert empty_rows[0]["baseline"]["fp_rate"] == 0.0
    assert presence_rows[0]["baseline"]["fp_rate"] == 0.5
    assert empty_rows[0]["verdict"] == "clean"
    assert presence_rows[0]["verdict"] == "motion-contaminated"


def test_classic_baseline_score_weights_cleanliness_stability_and_bursts() -> None:
    module = _load_validator_module()

    assert module.classic_baseline_score(0.0, 0.75, 0.0) == 100.0
    assert module.classic_baseline_score(0.10, 1.50, 5.0) == 0.0
    assert module.classic_baseline_score(0.05, 1.125, 2.5) == 53.8


def test_active_burst_metrics_reports_duration_and_rate() -> None:
    module = _load_validator_module()
    # States are evaluation ticks; with EVALUATION_INTERVAL=25 and 50 pps,
    # the eval sample rate is 2 Hz (one tick every 0.5 s).
    metrics = module._active_burst_metrics(
        np.array([0, 1, 1, 0, 1, 1, 1, 0], dtype=np.int8),
        packet_rate_pps=50.0,
    )

    assert metrics["burst_count"] == 2
    assert metrics["longest_burst_seconds"] == pytest.approx(1.5)
    assert metrics["bursts_per_minute"] == pytest.approx(30.0)


def test_empty_and_presence_verdicts_use_classic_baseline_only() -> None:
    module = _load_validator_module()
    baseline = {
        "fp_rate": 0.0,
        "margin_mad": 0.7,
        "longest_burst_seconds": 0.0,
    }

    assert module._empty_quality_verdict(baseline) == "clean"
    assert module._presence_quality_verdict(baseline) == "clean"

    motion_baseline = dict(baseline, fp_rate=0.08)
    assert module._empty_quality_verdict(motion_baseline) == "motion-like"
    assert module._presence_quality_verdict(motion_baseline) == "motion-contaminated"

    unstable_baseline = dict(baseline, margin_mad=1.25, longest_burst_seconds=2.0)
    assert module._empty_quality_verdict(unstable_baseline) == "unstable"
    assert module._presence_quality_verdict(unstable_baseline) == "unstable"


def test_metadata_refresh_recommendation_triggers_for_missing_pair_warning() -> None:
    module = _load_validator_module()

    assert module.should_recommend_dataset_metadata_refresh([], missing_motion_pair_count=1) is True


def test_refresh_metadata_respects_chip_filter() -> None:
    module = _load_validator_module()

    info = {
        "updated_at": "2026-07-12T10:00:00",
        "files": {
            "static_presence": [
                {
                    "filename": "static_presence_c3_64sc_dev1_20260712_100000_0001.npz",
                    "chip": "C3",
                    "subcarriers": 64,
                    "collected_at": "2026-07-12T10:00:00.000000",
                },
                {
                    "filename": "static_presence_c6_64sc_dev2_20260712_100000_0001.npz",
                    "chip": "C6",
                    "subcarriers": 64,
                    "collected_at": "2026-07-12T10:00:00.000000",
                    "optimal_pair_motion_file": "keep_motion.npz",
                },
            ],
            "motion": [
                {
                    "filename": "motion_c3_64sc_dev1_20260712_100500_0001.npz",
                    "chip": "C3",
                    "subcarriers": 64,
                    "collected_at": "2026-07-12T10:05:00.000000",
                },
                {
                    "filename": "motion_c6_64sc_dev2_20260712_100500_0001.npz",
                    "chip": "C6",
                    "subcarriers": 64,
                    "collected_at": "2026-07-12T10:05:00.000000",
                    "optimal_pair_static_presence_file": "keep_static.npz",
                },
            ],
        },
    }

    refreshed, pair_rows = module.refresh_metadata(info, chip_filter="C3")

    assert refreshed["files"]["static_presence"][0]["optimal_pair_motion_file"] == (
        "motion_c3_64sc_dev1_20260712_100500_0001.npz"
    )
    assert refreshed["files"]["motion"][0]["optimal_pair_static_presence_file"] == (
        "static_presence_c3_64sc_dev1_20260712_100000_0001.npz"
    )
    assert refreshed["files"]["static_presence"][1]["optimal_pair_motion_file"] == "keep_motion.npz"
    assert refreshed["files"]["motion"][1]["optimal_pair_static_presence_file"] == "keep_static.npz"
    assert len(pair_rows) == 1


def test_run_validation_refresh_metadata_writes_dataset_info(monkeypatch, tmp_path) -> None:
    module = _load_validator_module()

    dataset_info_path = tmp_path / "dataset_info.json"
    dataset_info_path.write_text(
        """
{
  "updated_at": "2026-07-12T10:00:00",
  "files": {
    "static_presence": [
      {
        "filename": "static_presence_c3_64sc_dev1_20260712_100000_0001.npz",
        "chip": "C3",
        "subcarriers": 64,
        "collected_at": "2026-07-12T10:00:00.000000",
        "environment": "lab"
      }
    ],
    "motion": [
      {
        "filename": "motion_c3_64sc_dev1_20260712_100500_0001.npz",
        "chip": "C3",
        "subcarriers": 64,
        "collected_at": "2026-07-12T10:05:00.000000",
        "environment": "lab"
      }
    ]
  }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(module, "DATASET_INFO", dataset_info_path)
    monkeypatch.setattr(module, "validate_metadata_completeness", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        module, "validate_empty_sanity", lambda *args, **kwargs: ([], [], [])
    )
    monkeypatch.setattr(
        module, "validate_quiet_test_recordings", lambda *args, **kwargs: ([], [])
    )
    monkeypatch.setattr(module, "validate_ml_readiness", lambda *args, **kwargs: [])
    monkeypatch.setattr(module, "should_recommend_dataset_metadata_refresh", lambda *args, **kwargs: False)

    exit_code = module.run_validation(generate_report=False)

    refreshed = module.load_dataset_info()
    static_entry = refreshed["files"]["static_presence"][0]
    motion_entry = refreshed["files"]["motion"][0]

    assert exit_code == 0
    assert static_entry["optimal_pair_motion_file"] == motion_entry["filename"]
    assert motion_entry["optimal_pair_static_presence_file"] == static_entry["filename"]
    assert refreshed["updated_at"] != "2026-07-12T10:00:00"


def test_run_validation_refresh_metadata_skips_write_when_unchanged(monkeypatch, tmp_path) -> None:
    module = _load_validator_module()

    dataset_info_path = tmp_path / "dataset_info.json"
    dataset_info_path.write_text(
        """
{
  "updated_at": "2026-07-12T10:00:00",
  "files": {
    "static_presence": [
      {
        "filename": "static_presence_c3_64sc_dev1_20260712_100000_0001.npz",
        "chip": "C3",
        "subcarriers": 64,
        "collected_at": "2026-07-12T10:00:00.000000",
        "environment": "lab",
        "optimal_pair_motion_file": "motion_c3_64sc_dev1_20260712_100500_0001.npz"
      }
    ],
    "motion": [
      {
        "filename": "motion_c3_64sc_dev1_20260712_100500_0001.npz",
        "chip": "C3",
        "subcarriers": 64,
        "collected_at": "2026-07-12T10:05:00.000000",
        "environment": "lab",
        "optimal_pair_static_presence_file": "static_presence_c3_64sc_dev1_20260712_100000_0001.npz"
      }
    ]
  }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    save_calls = []
    original_save = module.save_dataset_info

    def tracked_save(info):
        save_calls.append(info)
        original_save(info)

    monkeypatch.setattr(module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(module, "DATASET_INFO", dataset_info_path)
    monkeypatch.setattr(module, "save_dataset_info", tracked_save)
    monkeypatch.setattr(module, "validate_metadata_completeness", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        module, "validate_empty_sanity", lambda *args, **kwargs: ([], [], [])
    )
    monkeypatch.setattr(
        module, "validate_quiet_test_recordings", lambda *args, **kwargs: ([], [])
    )
    monkeypatch.setattr(module, "validate_ml_readiness", lambda *args, **kwargs: [])
    monkeypatch.setattr(module, "should_recommend_dataset_metadata_refresh", lambda *args, **kwargs: False)

    exit_code = module.run_validation(generate_report=False)

    assert exit_code == 0
    assert len(save_calls) == 0
    assert module.load_dataset_info()["updated_at"] == "2026-07-12T10:00:00"


def test_per_file_quality_labels_include_test_recordings() -> None:
    module = _load_validator_module()

    assert module.PER_FILE_QUALITY_LABELS == (
        "empty",
        "static_presence",
        "motion",
        "test",
    )


def test_ml_readiness_uses_empty_as_idle_and_warmup_per_file() -> None:
    module = _load_validator_module()
    dataset_info = {
        "files": {
            "empty": [
                {"filename": "empty.npz", "chip": "C3", "environment": "bedroom", "num_packets": 200},
            ],
            "static_presence": [
                {
                    "filename": "static.npz",
                    "chip": "C3",
                    "environment": "bedroom",
                    "num_packets": 300,
                    "optimal_pair_motion_file": "motion.npz",
                },
            ],
            "motion": [
                {
                    "filename": "motion.npz",
                    "chip": "C3",
                    "environment": "bedroom",
                    "num_packets": 400,
                    "optimal_pair_static_presence_file": "static.npz",
                },
            ],
        }
    }

    results = module.validate_ml_readiness(dataset_info)
    by_name = {result.name: result for result in results}

    assert by_name["sample_count"].value == 600
    assert "empty=100" in by_name["label_balance"].message
    assert "static_presence=200" in by_name["label_balance"].message


def test_ml_readiness_respects_chip_filter() -> None:
    module = _load_validator_module()
    dataset_info = {
        "files": {
            "empty": [
                {"filename": "empty_c3.npz", "chip": "C3", "num_packets": 200},
                {"filename": "empty_c6.npz", "chip": "C6", "num_packets": 900},
            ],
            "static_presence": [],
            "motion": [
                {"filename": "motion_c3.npz", "chip": "C3", "num_packets": 200},
                {"filename": "motion_c6.npz", "chip": "C6", "num_packets": 900},
            ],
        }
    }

    results = module.validate_ml_readiness(dataset_info, chip_filter="C3")
    sample_count = next(result for result in results if result.name == "sample_count")

    assert sample_count.value == 200


def test_file_integrity_rejects_subcarrier_shape_mismatch(tmp_path) -> None:
    module = _load_validator_module()
    path = tmp_path / "bad.npz"
    np.savez(
        path,
        csi_data=np.zeros((10, 128), dtype=np.int8),
        num_subcarriers=np.array(52),
    )

    results, data = module.validate_file_integrity(path)
    shape = next(result for result in results if result.name == "csi_shape")

    assert data is not None
    assert shape.status == "FAIL"
    assert "implies 64 subcarriers" in shape.message


def test_long_recording_coverage_warns_without_annotated_motion() -> None:
    module = _load_validator_module()
    dataset_info = {
        "files": {
            "test": [
                {
                    "filename": "quiet.npz",
                    "chip": "C3",
                    "description": "quiet long-run",
                    "num_packets": 60000,
                    "collected_at": "2026-07-04T11:23:18.928039",
                    "environment": "bedroom",
                },
            ],
        }
    }

    class FakeNpz:
        def __getitem__(self, key):
            return np.zeros((200, 128), dtype=np.int8)

    module._resolve_dataset_entry_path = lambda entry, label: Path("/tmp/quiet.npz")
    module._load_cached_or_npz = lambda filepath, cache: (FakeNpz(), "csi_data")
    module._classic_self_baseline_stats = (
        lambda csi, packet_rate_pps=100.0, *, calibration_cache=None, cache_key=None: {
            "score": 100.0,
            "fp_rate": 0.0,
            "margin_mad": 0.5,
            "longest_burst_seconds": 0.0,
            "threshold": 1.0,
            "eval_count": 100,
            "motion_count": 0,
        }
    )

    results, quiet_scores = module.validate_quiet_test_recordings(dataset_info, {})
    coverage = next(result for result in results if result.name == "long_test_event_coverage")
    quiet_result = next(result for result in results if result.name == "quiet_test_idle/quiet.npz")
    assert quiet_scores and quiet_scores[0]["baseline"]["score"] == 100.0
    assert quiet_scores[0]["verdict"] == "clean"
    assert quiet_scores[0]["display_date"] == "2026-07-04 11:23"
    assert quiet_result.status == "PASS"

    assert coverage.status == "WARN"
    assert coverage.value == 0


def test_metadata_completeness_fails_when_environment_is_missing() -> None:
    module = _load_validator_module()

    dataset_info = {
        "files": {
            "static_presence": [
                {
                    "filename": "static_a.npz",
                    "chip": "C5",
                    "optimal_pair_motion_file": "motion_a.npz",
                },
            ],
            "motion": [
                {
                    "filename": "motion_a.npz",
                    "chip": "C5",
                    "environment": "bedroom",
                    "optimal_pair_static_presence_file": "static_a.npz",
                },
            ],
        }
    }

    results = module.validate_metadata_completeness(dataset_info)
    missing_environment = next(
        result for result in results if result.name == "metadata_static_presence/static_a.npz"
    )

    assert missing_environment.status == "FAIL"
    assert "missing environment" in missing_environment.message


def test_capture_continuity_flags_low_rate_and_stream_gaps() -> None:
    module = _load_validator_module()

    class FakeNpz:
        files = ["duration_ms", "stream_seq_num"]

        def __init__(self):
            self.values = {
                "duration_ms": np.array(1000.0),
                "stream_seq_num": np.array([10, 11, 12, 60], dtype=np.uint32),
            }

        def __getitem__(self, key):
            return self.values[key]

    csi_data = np.zeros((4, 128), dtype=np.int8)

    results = module.validate_capture_continuity(FakeNpz(), csi_data)
    by_name = {result.name: result for result in results}

    assert by_name["packet_rate"].status == "WARN"
    assert "Low packet rate" in by_name["packet_rate"].message
    assert by_name["stream_seq_gaps"].status == "FAIL"
    assert "Missing stream packets: 92.2%" in by_name["stream_seq_gaps"].message
    assert by_name["stream_seq_max_gap"].status == "FAIL"


def test_capture_continuity_flags_large_inter_packet_gap() -> None:
    module = _load_validator_module()

    class FakeNpz:
        files = ["duration_ms", "stream_seq_num", "device_ticks_us"]

        def __init__(self):
            self.values = {
                "duration_ms": np.array(1000.0),
                "stream_seq_num": np.array([1, 2, 3, 4], dtype=np.uint32),
                "device_ticks_us": np.array([0, 10_000, 20_000, 2_500_000], dtype=np.uint64),
            }

        def __getitem__(self, key):
            return self.values[key]

    csi_data = np.zeros((4, 128), dtype=np.int8)

    results = module.validate_capture_continuity(FakeNpz(), csi_data)
    by_name = {result.name: result for result in results}

    assert by_name["inter_packet_gap"].status == "FAIL"
    assert "Largest inter-packet gap: 2480.0 ms" in by_name["inter_packet_gap"].message


def test_validate_pair_uses_classic_diagnostic_activation_logic(monkeypatch) -> None:
    module = _load_validator_module()

    static_csi = np.zeros((4, 128), dtype=np.int8)
    motion_csi = np.ones((4, 128), dtype=np.int8)
    threshold = 0.5
    detector = object()

    monkeypatch.setattr(
        module,
        "build_calibrated_classic_detector",
        lambda packets, selected_subcarriers=None: (detector, threshold),
    )

    def fake_replay(csi_data, replay_detector):
        assert replay_detector is detector
        if csi_data is static_csi:
            return {
                "score_series": np.array([0.10, 0.20, 0.30], dtype=np.float64),
                "state_series": np.array([0, 0, 0], dtype=np.int8),
            }
        return {
            "score_series": np.array([0.55, 0.70, 0.80], dtype=np.float64),
            "state_series": np.array([1, 1, 1], dtype=np.int8),
        }

    monkeypatch.setattr(module, "_replay_classic_metrics", fake_replay)

    results, static_active, motion_active, returned_threshold, pair_ratio = module.validate_pair(
        static_csi,
        motion_csi,
    )

    activation = results[0]
    assert activation.name == "classic_pair_activation"
    assert activation.status == "PASS"
    assert static_active == 0.0
    assert motion_active == 1.0
    assert returned_threshold == threshold
    # p95([0.55, 0.70, 0.80]) / 0.5
    assert pair_ratio == pytest.approx(1.58)


def test_validate_pair_warns_when_motion_stays_below_threshold(monkeypatch) -> None:
    module = _load_validator_module()

    static_csi = np.zeros((4, 128), dtype=np.int8)
    motion_csi = np.ones((4, 128), dtype=np.int8)
    threshold = 0.5
    detector = object()

    monkeypatch.setattr(
        module,
        "build_calibrated_classic_detector",
        lambda packets, selected_subcarriers=None: (detector, threshold),
    )

    def fake_replay(csi_data, replay_detector):
        assert replay_detector is detector
        if csi_data is static_csi:
            return {
                "score_series": np.array([0.10, 0.15, 0.20], dtype=np.float64),
                "state_series": np.array([0, 0, 0], dtype=np.int8),
            }
        return {
            "score_series": np.array([0.25, 0.30, 0.40], dtype=np.float64),
            "state_series": np.array([0, 0, 0], dtype=np.int8),
        }

    monkeypatch.setattr(module, "_replay_classic_metrics", fake_replay)

    results, static_active, motion_active, returned_threshold, pair_ratio = module.validate_pair(
        static_csi,
        motion_csi,
    )

    activation = results[0]
    assert activation.name == "classic_pair_activation"
    assert activation.status == "WARN"
    assert "motion_above=0.0%" in activation.message
    assert static_active == 0.0
    assert motion_active == 0.0
    assert returned_threshold == threshold
    # p95([0.25, 0.30, 0.40]) / 0.5
    assert pair_ratio == pytest.approx(0.78)


def test_classic_pair_score_rewards_clean_idle_and_full_motion() -> None:
    module = _load_validator_module()
    assert module.classic_pair_score(0.0, 1.0, 4.0) == 100.0
    assert module.classic_pair_score(0.10, 0.0, 1.0) == 0.0
    mid = module.classic_pair_score(0.05, 0.95, 2.5)
    assert 49.0 <= mid <= 76.0


def test_ratio_cells_mark_soft_warn_and_fail_thresholds() -> None:
    module = _load_validator_module()
    assert module._format_static_above_cell(0.0) == "0.0%"
    assert "⚠️" in module._format_static_above_cell(0.08)
    assert "❌" in module._format_static_above_cell(0.12)
    assert "⚠️" in module._format_motion_above_cell(0.94)
    assert "❌" in module._format_motion_above_cell(0.85)
    assert module._format_quiet_fp_cell(0.01) == "1.0%"
    assert "⚠️" in module._format_quiet_fp_cell(0.03)
    assert "❌" in module._format_quiet_fp_cell(0.06)
    assert "**" in module._format_static_above_cell(0.08, markdown=True)
    assert module._format_score_cell(99.0) == "99.0"
    assert "⚠️" in module._format_score_cell(80.0, "warn")
    assert "❌" in module._format_score_cell(40.0, "fail")
    assert module._format_margin_mad_cell(0.90) == "0.90"
    assert "⚠️" in module._format_margin_mad_cell(1.10)
    assert "❌" in module._format_margin_mad_cell(1.60)
    assert module._format_burst_cell(0.5) == "0.5s"
    assert "⚠️" in module._format_burst_cell(1.5)
    assert "❌" in module._format_burst_cell(5.5)
    assert module._score_value_severity(95.0) is None
    assert module._score_value_severity(94.9) is None
    assert module._score_value_severity(90.0) is None
    assert module._score_value_severity(89.9) is None
    assert module._score_value_severity(98.9) is None
    assert module._pair_ratio_severity(3.0) is None
    assert module._pair_ratio_severity(2.9) == "warn"
    assert module._pair_ratio_severity(2.0) == "warn"
    assert module._pair_ratio_severity(1.9) == "fail"
    assert "❌" in module._format_pair_ratio_cell(1.73)
    assert module._pair_ratio(
        np.array([0.55, 0.70, 0.80]),
        threshold=0.5,
    ) == pytest.approx(1.58)
    assert module._pair_ratio(
        np.array([0.25, 0.30, 0.40]),
        threshold=0.5,
    ) == pytest.approx(0.78)


def test_idle_evidence_results_never_fail_the_run() -> None:
    module = _load_validator_module()

    dirty_baseline = {
        "score": 10.0,
        "fp_rate": 0.5,
        "margin_mad": 2.0,
        "longest_burst_seconds": 30.0,
    }
    module._compute_idle_evidence_for_entry = (
        lambda entry, label, npz_cache, calibration_cache=None: (dirty_baseline, None)
    )

    results, rows = module._evaluate_idle_evidence_files(
        [{"filename": "quiet.npz", "chip": "C3", "environment": "bedroom"}],
        label="test",
        check_kind="quiet_test_idle",
        kind_title="Long-test",
        verdict_fn=module._empty_quality_verdict,
        npz_cache={},
    )

    assert results[0].status == "WARN"
    assert "verdict=motion-like" in results[0].message
    assert rows[0]["baseline"] is dirty_baseline


def test_pair_review_profile_derives_empirical_ratio_and_score_thresholds() -> None:
    module = _load_validator_module()
    pair_rows = [
        {"chip": "C3", "classic_status": "PASS", "pair_ratio": 4.8, "classic_score": 99.8},
        {"chip": "C3", "classic_status": "PASS", "pair_ratio": 4.4, "classic_score": 98.9},
        {"chip": "C3", "classic_status": "PASS", "pair_ratio": 4.0, "classic_score": 98.0},
        {"chip": "C3", "classic_status": "PASS", "pair_ratio": 3.6, "classic_score": 97.2},
    ]

    profile_map = module._table_review_profiles(pair_rows, [], [], [])
    severity_profile = profile_map["pair"]["C3"]

    assert module._pair_ratio_severity(3.7, severity_profile) == "warn"
    assert module._score_value_severity(97.3, severity_profile) is None


def test_idle_review_profile_derives_empirical_mad_and_burst_thresholds() -> None:
    module = _load_validator_module()
    idle_rows = [
        {
            "chip": "S3",
            "verdict": "clean",
            "baseline": {
                "margin_mad": 0.60,
                "longest_burst_seconds": 0.0,
                "score": 100.0,
            },
        },
        {
            "chip": "S3",
            "verdict": "clean",
            "baseline": {
                "margin_mad": 0.70,
                "longest_burst_seconds": 0.1,
                "score": 99.0,
            },
        },
        {
            "chip": "S3",
            "verdict": "clean",
            "baseline": {
                "margin_mad": 0.80,
                "longest_burst_seconds": 0.2,
                "score": 97.0,
            },
        },
        {
            "chip": "S3",
            "verdict": "clean",
            "baseline": {
                "margin_mad": 0.90,
                "longest_burst_seconds": 0.3,
                "score": 95.0,
            },
        },
    ]

    profile_map = module._table_review_profiles([], idle_rows, [], [])
    severity_profile = profile_map["static_presence"]["S3"]

    assert "⚠️" in module._format_margin_mad_cell(0.88, severity_profile=severity_profile)
    assert "⚠️" in module._format_burst_cell(0.28, severity_profile=severity_profile)
    assert module._score_value_severity(95.4, severity_profile) is None
