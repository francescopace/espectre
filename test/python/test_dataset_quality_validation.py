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


def test_empty_separation_uses_turb_mean_score(monkeypatch) -> None:
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
        csi, _refs = fake_data[path.name]
        return {"csi_data": csi}, "csi_data"

    def fake_compute(csi_data):
        if csi_data is fake_data["empty_a.npz"][0]:
            return np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        return np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float64)

    monkeypatch.setattr(module, "_resolve_dataset_entry_path", fake_resolve)
    monkeypatch.setattr(module, "_load_cached_or_npz", fake_load)
    monkeypatch.setattr(module, "_compute_turbulence_series", fake_compute)

    results = module.validate_empty_sanity(dataset_info, npz_cache={})
    separation = next(r for r in results if r.name == "empty_separation_C5_bedroom")

    assert separation.status == "PASS"
    assert separation.value == 1.0
    assert separation.message.startswith(
        "Empty-vs-static score separates group ('C5', 'bedroom')"
    )


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
    monkeypatch.setattr(module, "validate_empty_sanity", lambda *args, **kwargs: [])
    monkeypatch.setattr(module, "validate_quiet_test_recordings", lambda *args, **kwargs: [])
    monkeypatch.setattr(module, "validate_ml_readiness", lambda *args, **kwargs: [])
    monkeypatch.setattr(module, "should_recommend_dataset_metadata_refresh", lambda *args, **kwargs: False)

    exit_code = module.run_validation(refresh_metadata_first=True)

    refreshed = module.load_dataset_info()
    static_entry = refreshed["files"]["static_presence"][0]
    motion_entry = refreshed["files"]["motion"][0]

    assert exit_code == 0
    assert static_entry["optimal_pair_motion_file"] == motion_entry["filename"]
    assert motion_entry["optimal_pair_static_presence_file"] == static_entry["filename"]


def test_run_validation_refresh_metadata_force_writes_when_unchanged(monkeypatch, tmp_path) -> None:
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
    monkeypatch.setattr(module, "validate_empty_sanity", lambda *args, **kwargs: [])
    monkeypatch.setattr(module, "validate_quiet_test_recordings", lambda *args, **kwargs: [])
    monkeypatch.setattr(module, "validate_ml_readiness", lambda *args, **kwargs: [])
    monkeypatch.setattr(module, "should_recommend_dataset_metadata_refresh", lambda *args, **kwargs: False)

    exit_code = module.run_validation(refresh_metadata_first=True)

    assert exit_code == 0
    assert len(save_calls) == 1


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
                },
            ],
        }
    }

    class FakeNpz:
        def __getitem__(self, key):
            return np.zeros((200, 128), dtype=np.int8)

    module._resolve_dataset_entry_path = lambda entry, label: Path("/tmp/quiet.npz")
    module._load_cached_or_npz = lambda filepath, cache: (FakeNpz(), "csi_data")
    module._evaluate_classic_quiet_fp = lambda csi: {
        "fp_rate": 0.0,
        "threshold": 1.0,
        "eval_count": 100,
    }

    results = module.validate_quiet_test_recordings(dataset_info, {})
    coverage = next(result for result in results if result.name == "long_test_event_coverage")

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


def test_validate_pair_uses_threshold_activation_logic(monkeypatch) -> None:
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

    results, static_active, motion_active, returned_threshold, motion_peak_ratio = module.validate_pair(
        static_csi,
        motion_csi,
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

    results, static_active, motion_active, returned_threshold, motion_peak_ratio = module.validate_pair(
        static_csi,
        motion_csi,
    )

    activation = results[0]
    assert activation.name == "threshold_activation"
    assert activation.status == "FAIL"
    assert "motion_above=0.0%" in activation.message
    assert static_active == 0.0
    assert motion_active == 0.0
    assert returned_threshold == threshold
    assert motion_peak_ratio == pytest.approx(0.8)
