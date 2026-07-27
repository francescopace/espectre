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

    def fake_stats(csi_data, packet_rate_pps=100.0, *, rssi_dbm=None, calibration_cache=None, cache_key=None):
        assert packet_rate_pps == 100.0
        assert rssi_dbm is None
        if csi_data is empty_csi:
            margins = np.array([-0.5, 0.0, -0.5, 0.5])
            threshold = 0.3
            motion_count = 0
        else:
            # A tall upper tail above this capture's own median is what marks it
            # as contaminated now; the threshold plays no part.
            margins = np.array([0.0, 0.0, 0.0, 8.0])
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
            "margin_q95": float(np.quantile(margins, 0.95)) - median,
            "margin_q99": float(np.quantile(margins, 0.99)) - median,
            "margin_drift": 0.0,
            "margin_drift_abs": 0.0,
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


def test_classic_baseline_score_weights_tail_height_and_bursts() -> None:
    module = _load_validator_module()

    # The first argument is the capture's own q95 above its own median, in
    # logits, so the score no longer moves with the calibrated threshold.
    assert module.classic_baseline_score(2.0, 0.0) == 100.0
    assert module.classic_baseline_score(6.0, 5.0) == 0.0
    assert module.classic_baseline_score(4.0, 2.5) == 50.0

    # A uniformly noisy capture must not score well. This is what an excursion
    # rate normalized by the capture's own spread would have missed, because a
    # wide MAD lifts its own bound.
    assert module.classic_baseline_score(7.5, 0.0) < 40.0


def test_validate_file_integrity_rejects_object_arrays(tmp_path) -> None:
    module = _load_validator_module()
    filepath = tmp_path / "malicious_dataset.npz"
    np.savez_compressed(
        filepath,
        csi_data=np.zeros((1, 128), dtype=np.int8),
        num_subcarriers=64,
        chip=np.array("c6", dtype=object),
        label="motion",
    )

    results, data = module.validate_file_integrity(filepath)

    assert data is None
    assert results[0].name == "file_load"
    assert results[0].status == "FAIL"
    assert "Unsafe NPZ dataset" in results[0].message


def test_active_burst_metrics_reports_duration_and_rate() -> None:
    module = _load_validator_module()
    # States are evaluation ticks; the replay now treats Classic cadence as the
    # production 250 ms wall-clock step rather than scaling durations with raw
    # packet rate.
    metrics = module._active_burst_metrics(
        np.array([0, 1, 1, 0, 1, 1, 1, 0], dtype=np.int8),
        packet_rate_pps=50.0,
    )

    assert metrics["burst_count"] == 2
    assert metrics["longest_burst_seconds"] == pytest.approx(0.75)
    assert metrics["bursts_per_minute"] == pytest.approx(60.0)


def test_classic_calibration_cache_preserves_full_detector_state(monkeypatch) -> None:
    module = _load_validator_module()

    class FakeDetector:
        def __init__(self):
            self.threshold = 0.8
            self.adapted_threshold_ready = True
            self.settle_blocks = [1.0, 2.0]

    calibration_calls = []

    def fake_calibrate(*_args, **_kwargs):
        calibration_calls.append(True)
        return FakeDetector(), 0.8

    monkeypatch.setattr(
        module, "build_calibrated_classic_detector", fake_calibrate
    )
    cache = {}
    first = module._calibrated_classic_for(
        np.zeros((10, 128), dtype=np.int8),
        calibration_cache=cache,
        cache_key="low-rssi",
    )
    first[0].settle_blocks.append(99.0)
    second = module._calibrated_classic_for(
        np.zeros((10, 128), dtype=np.int8),
        calibration_cache=cache,
        cache_key="low-rssi",
    )

    assert len(calibration_calls) == 1
    assert second[0] is not first[0]
    assert second[0].threshold == pytest.approx(0.8)
    assert second[0].adapted_threshold_ready is True
    assert second[0].settle_blocks == [1.0, 2.0]


def test_calibrated_classic_for_passes_rssi_into_calibration(monkeypatch) -> None:
    module = _load_validator_module()
    captured_packets = []

    def fake_calibrate(packets, **_kwargs):
        captured_packets.extend(list(packets))
        return object(), 0.5

    monkeypatch.setattr(module, "build_calibrated_classic_detector", fake_calibrate)

    csi_data = np.zeros((3, 128), dtype=np.int8)
    rssi_dbm = np.array([-80, -79, -78], dtype=np.int16)
    module._calibrated_classic_for(csi_data, rssi_dbm=rssi_dbm)

    assert [pkt["rssi_dbm"] for pkt in captured_packets] == [-80, -79, -78]


def test_replay_classic_metrics_passes_rssi_into_detector() -> None:
    module = _load_validator_module()

    class FakeDetector:
        def __init__(self):
            self.calls = []

        def reset(self):
            return None

        def process_packet(self, packet, subcarriers, rssi_dbm=None):
            self.calls.append((packet, tuple(subcarriers), rssi_dbm))

        def update_state(self):
            return {"motion_metric": 0.0}

        def is_ready(self):
            return True

        def get_state(self):
            return module.MotionState.IDLE

        def get_threshold(self):
            return 0.5

    detector = FakeDetector()
    csi_data = np.zeros((3, 128), dtype=np.int8)
    rssi_dbm = np.array([-77, -76, -75], dtype=np.int16)

    module._replay_classic_metrics(csi_data, detector, rssi_dbm=rssi_dbm)

    assert [call[2] for call in detector.calls] == [-77, -76, -75]


def test_empty_and_presence_verdicts_use_classic_baseline_only() -> None:
    module = _load_validator_module()
    baseline = {
        "fp_rate": 0.0,
        "margin_mad": 0.7,
        "margin_q95": 2.0,
        "margin_drift": 0.0,
        "margin_drift_abs": 0.0,
        "longest_burst_seconds": 0.0,
    }

    assert module._empty_quality_verdict(baseline) == "clean"
    assert module._presence_quality_verdict(baseline) == "clean"

    # The verdict turns on the threshold-free tail, not on how often the
    # capture crossed a calibrated threshold.
    motion_baseline = dict(baseline, margin_q95=7.0)
    assert module._empty_quality_verdict(motion_baseline) == "motion-like"
    assert module._presence_quality_verdict(motion_baseline) == "motion-contaminated"

    # A capture that alarms constantly under a badly placed threshold is still
    # clean data when its own tail is low.
    assert module._empty_quality_verdict(dict(baseline, fp_rate=0.9)) == "clean"

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
                    "duration_ms": 4000,
                    "num_packets": 2000,
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
                    "duration_ms": 4000,
                    "num_packets": 2000,
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
    assert refreshed["files"]["static_presence"][0]["average_packet_rate"] == 500.0
    assert refreshed["files"]["motion"][0]["optimal_pair_static_presence_file"] == (
        "static_presence_c3_64sc_dev1_20260712_100000_0001.npz"
    )
    assert refreshed["files"]["motion"][0]["average_packet_rate"] == 500.0
    assert "nominal_packet_rate" not in refreshed["files"]["static_presence"][0]
    assert "nominal_packet_rate" not in refreshed["files"]["motion"][0]
    assert refreshed["files"]["static_presence"][1]["optimal_pair_motion_file"] == "keep_motion.npz"
    assert refreshed["files"]["motion"][1]["optimal_pair_static_presence_file"] == "keep_static.npz"
    assert len(pair_rows) == 1


def test_refresh_metadata_never_pairs_real_with_synthetic(
    monkeypatch, tmp_path
) -> None:
    module = _load_validator_module()
    monkeypatch.setattr(module.dataset_metadata, "DATA_DIR", tmp_path)
    (tmp_path / "static_presence").mkdir()
    (tmp_path / "motion").mkdir()
    np.savez_compressed(
        tmp_path / "static_presence" / "synthetic_static.npz",
        generation_group=np.asarray("synthetic-pair"),
    )
    np.savez_compressed(
        tmp_path / "motion" / "synthetic_motion.npz",
        generation_group=np.asarray("synthetic-pair"),
    )
    info = {
        "files": {
            "static_presence": [
                {
                    "filename": "real_static.npz",
                    "chip": "C3",
                    "subcarriers": 64,
                    "collected_at": "2026-07-22T10:00:00",
                },
                {
                    "filename": "synthetic_static.npz",
                    "chip": "C3",
                    "subcarriers": 64,
                    "collected_at": "2026-07-22T10:00:00",
                    "synthetic": True,
                },
            ],
            "motion": [
                {
                    "filename": "real_motion.npz",
                    "chip": "C3",
                    "subcarriers": 64,
                    "collected_at": "2026-07-22T10:01:00",
                },
                {
                    "filename": "synthetic_motion.npz",
                    "chip": "C3",
                    "subcarriers": 64,
                    "collected_at": "2026-07-22T10:01:00",
                    "synthetic": True,
                },
            ],
        }
    }

    refreshed, rows = module.refresh_metadata(info)

    assert len(rows) == 2
    assert refreshed["files"]["static_presence"][0]["optimal_pair_motion_file"] == (
        "real_motion.npz"
    )
    assert refreshed["files"]["static_presence"][1]["optimal_pair_motion_file"] == (
        "synthetic_motion.npz"
    )


def test_metadata_completeness_skips_excluded_entries() -> None:
    module = _load_validator_module()

    dataset_info = {
        "files": {
            "static_presence": [
                {
                    "filename": "excluded_static.npz",
                    "chip": "C5",
                    "dataset_role": "exclude",
                },
            ],
            "motion": [],
        }
    }

    results = module.validate_metadata_completeness(dataset_info)

    assert all("excluded_static.npz" not in result.name for result in results)


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


def test_ml_readiness_skips_excluded_entries() -> None:
    module = _load_validator_module()
    dataset_info = {
        "files": {
            "empty": [
                {"filename": "empty_train.npz", "chip": "C3", "num_packets": 200},
            ],
            "static_presence": [
                {
                    "filename": "static_excluded.npz",
                    "chip": "C3",
                    "num_packets": 400,
                    "dataset_role": "exclude",
                },
            ],
            "motion": [
                {
                    "filename": "motion_excluded.npz",
                    "chip": "C3",
                    "num_packets": 500,
                    "dataset_role": "exclude",
                },
            ],
        }
    }

    results = module.validate_ml_readiness(dataset_info)
    by_name = {result.name: result for result in results}

    assert by_name["sample_count"].value == 100
    assert "empty=100" in by_name["label_balance"].message
    assert "static_presence=0" in by_name["label_balance"].message
    assert "28.7%" not in by_name["label_balance"].message


def test_run_validation_skips_excluded_files_in_directory_scan(
    monkeypatch, tmp_path
) -> None:
    module = _load_validator_module()

    dataset_info_path = tmp_path / "dataset_info.json"
    dataset_info_path.write_text(
        """
{
  "updated_at": "2026-07-12T10:00:00",
  "files": {
    "static_presence": [
      {
        "filename": "included_static.npz",
        "chip": "C3",
        "subcarriers": 64,
        "collected_at": "2026-07-12T10:00:00.000000",
        "environment": "lab"
      },
      {
        "filename": "excluded_static.npz",
        "chip": "C3",
        "subcarriers": 64,
        "collected_at": "2026-07-12T10:01:00.000000",
        "environment": "lab",
        "dataset_role": "exclude"
      }
    ],
    "motion": []
  }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    static_dir = tmp_path / "static_presence"
    static_dir.mkdir()
    (static_dir / "included_static.npz").touch()
    (static_dir / "excluded_static.npz").touch()

    processed = []

    def fake_validate_file_integrity(path):
        processed.append(path.name)
        return [module.ValidationResult("file_load", "PASS", "ok")], {
            "csi_data": np.zeros((1, 128), dtype=np.int8),
        }

    monkeypatch.setattr(module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(module, "DATASET_INFO", dataset_info_path)
    monkeypatch.setattr(module, "validate_metadata_completeness", lambda *args, **kwargs: [])
    monkeypatch.setattr(module, "validate_file_integrity", fake_validate_file_integrity)
    monkeypatch.setattr(module, "validate_signal_quality", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        module, "validate_capture_continuity", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(module, "validate_empty_sanity", lambda *args, **kwargs: ([], [], []))
    monkeypatch.setattr(
        module, "validate_quiet_test_recordings", lambda *args, **kwargs: ([], [])
    )
    monkeypatch.setattr(module, "validate_ml_readiness", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        module,
        "should_recommend_dataset_metadata_refresh",
        lambda *args, **kwargs: False,
    )

    exit_code = module.run_validation(generate_report=False)

    assert exit_code == 0
    assert processed == ["included_static.npz"]


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


def test_file_integrity_returns_ht20_sensing_view(tmp_path) -> None:
    module = _load_validator_module()
    path = tmp_path / "mixed_phy.npz"
    np.savez(
        path,
        csi_data=np.zeros((5, 128), dtype=np.int8),
        num_subcarriers=np.array(64),
        label=np.array("motion"),
        phy_mode=np.array(["ht", "legacy", "ht", "legacy", "ht"]),
        ltf_type=np.array(["ht-ltf", "lltf", "ht-ltf", "lltf", "ht-ltf"]),
        channel_width=np.array(["20", "20", "20", "20", "20"]),
        stream_seq_num=np.array([1, 2, 3, 4, 5], dtype=np.uint32),
    )

    results, data = module.validate_file_integrity(path)
    by_name = {result.name: result for result in results}

    assert by_name["file_load"].status == "PASS"
    assert data is not None
    assert data["csi_data"].shape[0] == 3
    np.testing.assert_array_equal(data["stream_seq_num"], np.array([1, 3, 5], dtype=np.uint32))
    np.testing.assert_array_equal(data["phy_mode"], np.array(["ht", "ht", "ht"]))


def test_capture_continuity_sees_gaps_after_ht20_filter(tmp_path) -> None:
    module = _load_validator_module()
    path = tmp_path / "legacy_heavy.npz"
    # Alternate HT/legacy so the HT20 view has a gap on every other seq number.
    phy_mode = np.array(["ht", "legacy"] * 20)
    ltf_type = np.array(["ht-ltf", "lltf"] * 20)
    stream_seq = np.arange(1, 41, dtype=np.uint32)
    np.savez(
        path,
        csi_data=np.zeros((40, 128), dtype=np.int8),
        num_subcarriers=np.array(64),
        label=np.array("static_presence"),
        duration_ms=np.array(1000.0),
        phy_mode=phy_mode,
        ltf_type=ltf_type,
        channel_width=np.array(["20"] * 40),
        stream_seq_num=stream_seq,
    )

    _, data = module.validate_file_integrity(path)
    assert data is not None
    assert data["csi_data"].shape[0] == 20

    results = module.validate_capture_continuity(data, data["csi_data"])
    by_name = {result.name: result for result in results}

    assert by_name["stream_seq_gaps"].status in {"WARN", "FAIL"}
    assert by_name["stream_seq_gaps"].value > 0.0


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
        lambda csi, packet_rate_pps=100.0, *, rssi_dbm=None, calibration_cache=None, cache_key=None: {
            "score": 100.0,
            "fp_rate": 0.0,
            "margin_mad": 0.5,
            "margin_q95": -0.5,
            "margin_drift": 0.0,
            "margin_drift_abs": 0.0,
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
    assert "after packet 2 (seq 12 -> 60)" in by_name["stream_seq_max_gap"].message


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
    assert "at packet 2->3" in by_name["inter_packet_gap"].message


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

    def fake_replay(csi_data, replay_detector, *, rssi_dbm=None):
        assert replay_detector is detector
        assert rssi_dbm is None
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

    (
        results,
        static_active,
        motion_active,
        returned_threshold,
        pair_separation,
        _idle_tail,
        _motion_coverage,
    ) = module.validate_pair(
        static_csi,
        motion_csi,
    )

    activation = results[0]
    assert activation.name == "classic_pair_activation"
    assert activation.status == "PASS"
    assert static_active == 0.0
    assert motion_active == 1.0
    assert returned_threshold == threshold
    # Every motion score outranks every idle score.
    assert pair_separation == pytest.approx(1.0)


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

    def fake_replay(csi_data, replay_detector, *, rssi_dbm=None):
        assert replay_detector is detector
        assert rssi_dbm is None
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

    (
        results,
        static_active,
        motion_active,
        returned_threshold,
        pair_separation,
        _idle_tail,
        _motion_coverage,
    ) = module.validate_pair(
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
    # This is the case the separation metric exists for. Not one motion
    # evaluation crosses the threshold, so the Classic diagnostic warns, yet the
    # two halves are perfectly ordered and the recording itself is fine.
    assert pair_separation == pytest.approx(1.0)


def test_classic_pair_score_uses_only_threshold_free_terms() -> None:
    module = _load_validator_module()
    # (idle tail in logits, motion coverage above the idle p95, idle/motion AUC)
    assert module.classic_pair_score(2.0, 1.0, 1.0) == 100.0
    assert module.classic_pair_score(6.0, 0.0, 0.90) == 0.0
    mid = module.classic_pair_score(4.0, 0.95, 0.995)
    assert 49.0 <= mid <= 90.0

    # Separation leads, so a pair that separates cleanly outscores one that does
    # not, even when the second covers more of its motion half.
    assert module.classic_pair_score(2.0, 0.70, 0.9990) > module.classic_pair_score(
        2.0, 1.0, 0.9200
    )


def test_ratio_cells_mark_soft_warn_and_fail_thresholds() -> None:
    module = _load_validator_module()
    assert module._format_static_above_cell(0.0) == "0.0%"
    assert "⚠️" in module._format_static_above_cell(0.08)
    assert "❌" in module._format_static_above_cell(0.12)
    assert "⚠️" in module._format_motion_above_cell(0.94)
    assert "❌" in module._format_motion_above_cell(0.85)
    assert module._format_quiet_fp_cell(0.01) == "1.0%"
    assert "⚠️" in module._format_quiet_fp_cell(0.09)
    assert "❌" in module._format_quiet_fp_cell(0.14)
    assert "**" in module._format_static_above_cell(0.08, markdown=True)
    assert module._format_score_cell(99.0) == "99.0"
    assert "⚠️" in module._format_score_cell(80.0, "warn")
    assert "❌" in module._format_score_cell(40.0, "fail")
    assert module._format_margin_mad_cell(0.90) == "0.90"
    assert module._format_margin_mad_cell(1.10) == "1.10"
    assert module._format_margin_mad_cell(1.60) == "1.60"
    assert module._format_packet_rate_cell(93.2) == "93.2"
    assert module._format_burst_cell(0.5) == "0.5s"
    assert "⚠️" in module._format_burst_cell(1.5)
    assert "❌" in module._format_burst_cell(5.5)
    assert module._format_margin_q95_cell(-0.20) == "-0.20"
    assert module._format_margin_drift_cell(0.20) == "0.20"
    assert module._score_value_severity(95.0) is None
    assert module._score_value_severity(94.9) is None
    assert module._score_value_severity(90.0) is None
    assert module._score_value_severity(89.9) is None
    assert module._score_value_severity(98.9) is None
    assert module._pair_separation_severity(0.995) is None
    assert module._pair_separation_severity(0.985) == "warn"
    assert module._pair_separation_severity(0.971) == "warn"
    assert module._pair_separation_severity(0.969) == "fail"
    assert "❌" in module._format_pair_separation_cell(0.9678)

    # Disjoint series separate perfectly whatever the scale.
    assert module._pair_separation(
        np.array([0.10, 0.20, 0.30]),
        np.array([0.55, 0.70, 0.80]),
    ) == pytest.approx(1.0)
    # The same ordering far below any plausible threshold scores the same.
    assert module._pair_separation(
        np.array([0.10, 0.15, 0.20]),
        np.array([0.25, 0.30, 0.40]),
    ) == pytest.approx(1.0)
    # Identical series carry no separation at all.
    assert module._pair_separation(
        np.array([0.4, 0.4, 0.4]),
        np.array([0.4, 0.4, 0.4]),
    ) == pytest.approx(0.5)
    # Fully reversed order is the opposite extreme.
    assert module._pair_separation(
        np.array([0.8, 0.9, 1.0]),
        np.array([0.1, 0.2, 0.3]),
    ) == pytest.approx(0.0)


def test_idle_evidence_results_never_fail_the_run() -> None:
    module = _load_validator_module()

    dirty_baseline = {
        "score": 10.0,
        "fp_rate": 0.5,
        "margin_mad": 2.0,
        "margin_q95": 0.8,
        "margin_drift": 0.6,
        "margin_drift_abs": 0.6,
        "longest_burst_seconds": 30.0,
    }
    module._compute_idle_evidence_for_entry = (
        lambda entry, label, npz_cache, calibration_cache=None: (
            dirty_baseline,
            None,
            None,
        )
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


def test_pair_separation_keeps_absolute_floors_instead_of_empirical_ones() -> None:
    module = _load_validator_module()
    pair_rows = [
        {"chip": "C3", "classic_status": "PASS", "pair_separation": 1.0000, "classic_score": 99.8},
        {"chip": "C3", "classic_status": "PASS", "pair_separation": 0.9998, "classic_score": 98.9},
        {"chip": "C3", "classic_status": "PASS", "pair_separation": 0.9995, "classic_score": 98.0},
        {"chip": "C3", "classic_status": "PASS", "pair_separation": 0.9990, "classic_score": 97.2},
    ]

    profile_map = module._table_review_profiles(pair_rows, [], [], [])

    # AUC saturates against its own ceiling, so a peer-relative outlier rule
    # would mark the bottom of a set of near-perfect pairs. The pair table
    # therefore derives no empirical thresholds at all.
    assert profile_map["pair"] == {}
    severity_profile = module._row_severity_profile(profile_map, "pair", "C3")
    assert module._review_basis_label(severity_profile) == "fixed"
    assert module._pair_separation_severity(0.9990, severity_profile) is None


def test_idle_review_profile_derives_empirical_burst_q95_and_drift_thresholds() -> None:
    module = _load_validator_module()
    idle_rows = [
        {
            "chip": "S3",
            "verdict": "clean",
            "baseline": {
                "margin_mad": 0.60,
                "margin_q95": -0.70,
                "margin_drift": 0.02,
                "margin_drift_abs": 0.02,
                "longest_burst_seconds": 0.0,
                "score": 100.0,
            },
        },
        {
            "chip": "S3",
            "verdict": "clean",
            "baseline": {
                "margin_mad": 0.70,
                "margin_q95": -0.60,
                "margin_drift": 0.04,
                "margin_drift_abs": 0.04,
                "longest_burst_seconds": 0.1,
                "score": 99.0,
            },
        },
        {
            "chip": "S3",
            "verdict": "clean",
            "baseline": {
                "margin_mad": 0.80,
                "margin_q95": -0.45,
                "margin_drift": 0.07,
                "margin_drift_abs": 0.07,
                "longest_burst_seconds": 0.2,
                "score": 97.0,
            },
        },
        {
            "chip": "S3",
            "verdict": "clean",
            "baseline": {
                "margin_mad": 0.90,
                "margin_q95": -0.35,
                "margin_drift": 0.10,
                "margin_drift_abs": 0.10,
                "longest_burst_seconds": 0.3,
                "score": 95.0,
            },
        },
    ]

    profile_map = module._table_review_profiles([], idle_rows, [], [])
    severity_profile = module._row_severity_profile(
        profile_map, "static_presence", "S3"
    )

    assert module._review_basis_label(severity_profile) == "chip"
    assert "⚠️" in module._format_burst_cell(0.28, severity_profile=severity_profile)
    # Tail height is absolute and shared with the verdict, so no peer-relative
    # profile can mark a low tail or leave a high one clean.
    assert module._format_margin_q95_cell(-0.37, severity_profile=severity_profile) == "-0.37"
    assert "⚠️" in module._format_margin_q95_cell(4.5, severity_profile=severity_profile)
    assert "❌" in module._format_margin_q95_cell(6.5, severity_profile=severity_profile)
    assert "q95" not in severity_profile
    assert "⚠️" in module._format_margin_drift_cell(0.095, severity_profile=severity_profile)
    assert module._score_value_severity(95.4, severity_profile) is None


def test_idle_review_profile_uses_fixed_basis_without_same_chip_references() -> None:
    module = _load_validator_module()
    idle_rows = [
        {
            "chip": "C3",
            "verdict": "clean",
            "baseline": {
                "margin_mad": 0.60,
                "margin_q95": -0.70,
                "margin_drift": 0.02,
                "margin_drift_abs": 0.02,
                "longest_burst_seconds": 0.0,
                "score": 100.0,
            },
        },
        {
            "chip": "C3",
            "verdict": "clean",
            "baseline": {
                "margin_mad": 0.70,
                "margin_q95": -0.60,
                "margin_drift": 0.04,
                "margin_drift_abs": 0.04,
                "longest_burst_seconds": 0.1,
                "score": 99.0,
            },
        },
        {
            "chip": "C3",
            "verdict": "clean",
            "baseline": {
                "margin_mad": 0.80,
                "margin_q95": -0.45,
                "margin_drift": 0.07,
                "margin_drift_abs": 0.07,
                "longest_burst_seconds": 0.2,
                "score": 97.0,
            },
        },
    ]

    profile_map = module._table_review_profiles([], idle_rows, [], [])
    severity_profile = module._row_severity_profile(profile_map, "static_presence", "C6")

    assert module._review_basis_label(severity_profile) == "fixed"
    assert "C3" not in profile_map["static_presence"]
    assert module.EMPIRICAL_PROFILE_GLOBAL_KEY not in profile_map["static_presence"]


def test_pair_rows_report_fixed_basis_for_every_chip() -> None:
    module = _load_validator_module()
    pair_rows = [
        {"chip": "C3", "classic_status": "PASS", "pair_separation": 1.0000, "classic_score": 99.8},
        {"chip": "C3", "classic_status": "PASS", "pair_separation": 0.9998, "classic_score": 98.9},
        {"chip": "C3", "classic_status": "PASS", "pair_separation": 0.9995, "classic_score": 98.0},
        {"chip": "C3", "classic_status": "PASS", "pair_separation": 0.9990, "classic_score": 97.2},
        {"chip": "C6", "classic_status": "PASS", "pair_separation": 0.9988, "classic_score": 96.1},
    ]

    profile_map = module._table_review_profiles(pair_rows, [], [], [])
    severity_profile = module._row_severity_profile(profile_map, "pair", "C6")

    assert module._review_basis_label(severity_profile) == "fixed"
    assert module._pair_separation_severity(0.9988, severity_profile) is None


def test_idle_score_row_reports_basis_and_alternative_metrics() -> None:
    module = _load_validator_module()
    row = {
        "chip": "S3",
        "environment": "bedroom",
        "filename": "static_presence_s3_demo.npz",
        "display_date": "2026-07-24 12:59",
        "rssi_dbm": -47.0,
        "baseline": {
            "score": 97.5,
            "packet_rate_pps": 93.2,
            "fp_rate": 0.0,
            "margin_mad": 0.88,
            "margin_q95": -0.37,
            "margin_drift": 0.09,
            "margin_drift_abs": 0.09,
            "longest_burst_seconds": 0.28,
        },
        "verdict": "clean",
    }
    review_profiles = {
        "static_presence": {
            "S3": {
                "mad": {"warn_above": 0.85, "fail_above": 0.95},
                "burst": {"warn_above": 0.25, "fail_above": 0.35},
                "q95": {"warn_above": -0.40, "fail_above": -0.30},
                "drift": {"warn_above": 0.08, "fail_above": 0.12},
            },
        },
    }

    rendered = module._format_idle_evidence_score_row(
        row,
        label="static_presence",
        markdown=True,
        review_profiles=review_profiles,
    )

    assert rendered.startswith("| S3 | bedroom |")
    assert "| 93.2 |" in rendered
    assert "⚠️" in rendered
