"""
ESPectre - Long Recording Validation Tests

Validation tests for long CSI recordings.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import importlib.util
from pathlib import Path

import pytest

from tools.lib.performance_report import (
    evaluate_classic_long_recording as _evaluate_classic_long_recording,
    evaluate_ml_long_recording as _evaluate_ml_long_recording,
)
from tools.lib.repo_paths import tools_dir

from conftest import (
    DATA_DIR,
    DATASET_INFO_PATH,
    build_long_test_params,
    extract_motion_start_from_description,
    get_available_long_test_datasets,
    load_long_test_dataset,
)


TRAIN_ML_MODEL_PATH = tools_dir() / "train_ml_model.py"


def _load_train_ml_model_module():
    """Load the training script directly from the tools directory."""
    spec = importlib.util.spec_from_file_location("train_ml_model_gate", TRAIN_ML_MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestLongRecordings:
    """Validate MLDetector on the curated 60-second recordings."""

    _rows = []

    @classmethod
    def setup_class(cls):
        cls._rows = []

    @classmethod
    def teardown_class(cls):
        if not cls._rows:
            return

        print("")
        print("=" * 99)
        print("                    LONG RECORDING ML SUMMARY (for seed search)")
        print("=" * 99)
        print("| Chip   | Recall  | Precision | FP Rate | F1-Score | FP Count | Alarms | False Motion Evals |")
        print("|--------|---------|-----------|---------|----------|----------|--------|--------------------|")
        for row in sorted(cls._rows, key=lambda item: item["chip"]):
            print(
                f"| {row['chip']:<6} | {row['recall']:>6.1f}% | {row['precision']:>8.1f}% | "
                f"{row['fp_rate']:>6.1f}% | {row['f1']:>7.1f}% | {row['fp_count']:>8d} | "
                f"{row['effective_alarms']:>6d} | {row['false_motion_evaluations']:>18d} |"
            )
        print("-" * 99)

    @pytest.mark.parametrize("long_dataset", build_long_test_params(), indirect=False)
    def test_ml_vs_test_recordings(self, long_dataset):
        """
        Evaluate the ML detector on the 60-second test recordings.

        The output table is intentionally stable because train_ml_model.py
        parses it during seed search.
        """
        if long_dataset is None:
            pytest.skip("No long test recordings available in data/test")

        long_dataset = load_long_test_dataset(long_dataset)
        _, baseline_packets, movement_packets, motion_start_packet, chip, entry = long_dataset
        metrics = _evaluate_ml_long_recording(baseline_packets, movement_packets)
        self.__class__._rows.append(
            {
                "chip": chip,
                "motion_start_packet": motion_start_packet,
                "baseline_packets": len(baseline_packets),
                "movement_packets": len(movement_packets),
                "fp_count": metrics["fp"],
                **metrics,
            }
        )

        assert len(baseline_packets) == motion_start_packet
        if extract_motion_start_from_description(entry.get("description")) is None:
            assert len(movement_packets) == 0
        else:
            assert len(movement_packets) > 0
        assert metrics["baseline_eval_count"] >= 0
        assert metrics["movement_eval_count"] >= 0
        assert 0.0 <= metrics["recall"] <= 100.0
        assert 0.0 <= metrics["precision"] <= 100.0
        assert 0.0 <= metrics["fp_rate"] <= 100.0
        assert 0.0 <= metrics["f1"] <= 100.0
        assert metrics["effective_alarms"] >= 0
        assert metrics["false_motion_evaluations"] >= 0
        assert str(entry.get("chip", "")).upper() == chip


class TestLongRecordingsClassic:
    """Validate ClassicDetector on the curated 60-second recordings."""

    _rows = []

    @classmethod
    def setup_class(cls):
        cls._rows = []

    @classmethod
    def teardown_class(cls):
        if not cls._rows:
            return

        print("")
        print("=" * 108)
        print("                           LONG RECORDING CLASSIC SUMMARY")
        print("=" * 108)
        print("| Chip   | Recall  | Precision | FP Rate | F1-Score | FP Count |")
        print("|--------|---------|-----------|---------|----------|----------|")
        for row in sorted(cls._rows, key=lambda item: item["chip"]):
            print(
                f"| {row['chip']:<6} | {row['recall']:>6.1f}% | {row['precision']:>8.1f}% | "
                f"{row['fp_rate']:>6.1f}% | {row['f1']:>7.1f}% | {row['fp_count']:>8d} |"
            )
        print("-" * 108)

    @pytest.mark.parametrize("long_dataset", build_long_test_params(), indirect=False)
    def test_classic_vs_test_recordings(self, long_dataset):
        """Evaluate startup-calibrated ClassicDetector on the 60-second test recordings."""
        if long_dataset is None:
            pytest.skip("No long test recordings available in data/test")

        long_dataset = load_long_test_dataset(long_dataset)
        _, baseline_packets, movement_packets, motion_start_packet, chip, entry = long_dataset
        metrics = _evaluate_classic_long_recording(baseline_packets, movement_packets)
        assert metrics is not None, "Classic startup calibration failed"
        self.__class__._rows.append(
            {
                "chip": chip,
                "motion_start_packet": motion_start_packet,
                "baseline_packets": len(baseline_packets),
                "movement_packets": len(movement_packets),
                "fp_count": metrics["fp"],
                **metrics,
            }
        )

        assert len(baseline_packets) == motion_start_packet
        if extract_motion_start_from_description(entry.get("description")) is None:
            assert len(movement_packets) == 0
        else:
            assert len(movement_packets) > 0
        assert metrics["baseline_eval_count"] >= 0
        assert metrics["movement_eval_count"] >= 0
        assert 0.0 <= metrics["adaptive_threshold"] <= 10.0
        assert 0.0 <= metrics["recall"] <= 100.0
        assert 0.0 <= metrics["precision"] <= 100.0
        assert 0.0 <= metrics["fp_rate"] <= 100.0
        assert 0.0 <= metrics["f1"] <= 100.0
        assert str(entry.get("chip", "")).upper() == chip


class TestLongRecordingHelpers:
    """Regression tests for long recording metadata and parser helpers."""

    def test_motion_start_packet_uses_metadata_or_full_capture_fallback(self):
        datasets = get_available_long_test_datasets()
        assert datasets, f"No datasets found via {DATASET_INFO_PATH}"

        chips = {chip for _, _, _, _, chip, _ in datasets}
        assert chips

        for _, baseline_packets, movement_packets, motion_start_packet, _, entry in datasets:
            expected = extract_motion_start_from_description(entry.get("description"))
            assert len(baseline_packets) == motion_start_packet
            if expected is None:
                assert motion_start_packet == len(baseline_packets) + len(movement_packets)
                assert len(movement_packets) == 0
            else:
                assert expected == motion_start_packet
                assert len(movement_packets) > 0

    @pytest.mark.parametrize("long_dataset", build_long_test_params(), indirect=False)
    def test_long_test_loader_splits_stream_consistently(self, long_dataset):
        if long_dataset is None:
            pytest.skip("No long test recordings available in data/test")

        long_dataset = load_long_test_dataset(long_dataset)
        test_path, baseline_packets, movement_packets, motion_start_packet, chip, entry = long_dataset

        assert test_path.parent == DATA_DIR / "test"
        assert test_path.exists()
        assert len(baseline_packets) == motion_start_packet
        if extract_motion_start_from_description(entry.get("description")) is None:
            assert len(movement_packets) == 0
        else:
            assert len(movement_packets) > 0
        assert str(entry.get("chip", "")).upper() == chip

    def test_long_gate_output_parser_extracts_rows(self):
        train_ml_model = _load_train_ml_model_module()
        output = """
===================================================================================
                    LONG RECORDING ML SUMMARY (for seed search)
===================================================================================
| Chip   | Recall  | Precision | FP Rate | F1-Score | FP Count | Alarms | False Motion Evals |
|--------|---------|-----------|---------|----------|----------|--------|--------------------|
| C3     |   99.9% |     97.1% |    3.0% |    98.5% |       89 |      0 |                  0 |
| C5     |   98.4% |     96.3% |    4.1% |    97.3% |      121 |      1 |                  4 |
| C6     |   96.7% |     94.8% |    5.2% |    95.7% |      165 |      2 |                  9 |
-----------------------------------------------------------------------------------
"""
        metrics = train_ml_model._parse_long_recording_metrics(output)
        assert metrics is not None
        assert metrics["pass_count"] == 2
        assert metrics["total_fp"] == 375
        assert metrics["total_effective_alarms"] == 3
        assert metrics["total_false_motion_evaluations"] == 13
        assert metrics["mean_f1"] == pytest.approx((98.5 + 97.3 + 95.7) / 3.0)
        assert metrics["worst_chip_f1"] == pytest.approx(95.7)
