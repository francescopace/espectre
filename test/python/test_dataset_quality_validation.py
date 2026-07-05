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

    def fake_compute(csi_data):
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


def test_metadata_refresh_recommendation_triggers_for_missing_threshold() -> None:
    module = _load_validator_module()

    results = [
        module.ValidationResult(
            "metadata_test/example.npz",
            "FAIL",
            "optimal_threshold_gridsearch must be a positive number",
        )
    ]

    assert module.should_recommend_dataset_metadata_refresh(results) is True


def test_metadata_refresh_recommendation_triggers_for_missing_pair_warning() -> None:
    module = _load_validator_module()

    assert module.should_recommend_dataset_metadata_refresh([], missing_motion_pair_count=1) is True


def test_per_file_quality_labels_include_test_recordings() -> None:
    module = _load_validator_module()

    assert module.PER_FILE_QUALITY_LABELS == (
        "empty",
        "static_presence",
        "motion",
        "test",
    )


def test_metadata_completeness_fails_when_environment_is_missing() -> None:
    module = _load_validator_module()

    dataset_info = {
        "files": {
            "static_presence": [
                {
                    "filename": "static_a.npz",
                    "chip": "C5",
                    "optimal_threshold_gridsearch": 1.5,
                    "optimal_pair_motion_file": "motion_a.npz",
                },
            ],
            "motion": [
                {
                    "filename": "motion_a.npz",
                    "chip": "C5",
                    "environment": "bedroom",
                    "optimal_threshold_gridsearch": 1.8,
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
    assert by_name["stream_seq_max_gap"].status == "WARN"


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


def test_validate_pair_uses_threshold_activation_logic(monkeypatch) -> None:
    module = _load_validator_module()

    static_csi = np.zeros((4, 128), dtype=np.int8)
    motion_csi = np.ones((4, 128), dtype=np.int8)
    threshold = 0.5

    monkeypatch.setattr(module, "_filter_measurement_frames", lambda csi_data, data: csi_data)

    def fake_replay(csi_data, runtime_threshold):
        assert runtime_threshold == threshold
        if csi_data is static_csi:
            return np.array([0.10, 0.20, 0.30], dtype=np.float64)
        return np.array([0.55, 0.70, 0.80], dtype=np.float64)

    monkeypatch.setattr(module, "_replay_mvs_metric_series", fake_replay)

    results, static_active, motion_active, returned_threshold, motion_peak_ratio = module.validate_pair(
        static_csi,
        motion_csi,
        {},
        {},
        threshold,
    )

    activation = results[0]
    assert activation.name == "threshold_activation"
    assert activation.status == "PASS"
    assert static_active == 0.0
    assert motion_active == 1.0
    assert returned_threshold == threshold
    assert motion_peak_ratio == pytest.approx(1.6)


def test_validate_pair_fails_when_motion_stays_below_threshold(monkeypatch) -> None:
    module = _load_validator_module()

    static_csi = np.zeros((4, 128), dtype=np.int8)
    motion_csi = np.ones((4, 128), dtype=np.int8)
    threshold = 0.5

    monkeypatch.setattr(module, "_filter_measurement_frames", lambda csi_data, data: csi_data)

    def fake_replay(csi_data, runtime_threshold):
        assert runtime_threshold == threshold
        if csi_data is static_csi:
            return np.array([0.10, 0.15, 0.20], dtype=np.float64)
        return np.array([0.25, 0.30, 0.40], dtype=np.float64)

    monkeypatch.setattr(module, "_replay_mvs_metric_series", fake_replay)

    results, static_active, motion_active, returned_threshold, motion_peak_ratio = module.validate_pair(
        static_csi,
        motion_csi,
        {},
        {},
        threshold,
    )

    activation = results[0]
    assert activation.name == "threshold_activation"
    assert activation.status == "FAIL"
    assert "motion_above=0.0%" in activation.message
    assert static_active == 0.0
    assert motion_active == 0.0
    assert returned_threshold == threshold
    assert motion_peak_ratio == pytest.approx(0.8)
