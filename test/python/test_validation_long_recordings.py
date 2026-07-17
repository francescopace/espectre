"""
ESPectre - Long Recording Validation Tests

Validation tests for long CSI recordings.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

from pathlib import Path

import pytest

from tools.lib.performance_report import (
    evaluate_classic_long_recording as _evaluate_classic_long_recording,
    evaluate_ml_long_recording as _evaluate_ml_long_recording,
)

from conftest import (
    DATA_DIR,
    DATASET_INFO_PATH,
    build_long_test_params,
    extract_motion_start_from_description,
    get_available_long_test_datasets,
    load_long_test_dataset,
)


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
        print("=" * 88)
        print("                    LONG RECORDING ML SUMMARY (for seed search)")
        print("=" * 88)
        print("| Chip   | Recall  | Precision | FP Rate | F1-Score | FP Count | Alarms |")
        print("|--------|---------|-----------|---------|----------|----------|--------|")
        for row in sorted(cls._rows, key=lambda item: item["chip"]):
            print(
                f"| {row['chip']:<6} | {row['recall']:>6.1f}% | {row['precision']:>8.1f}% | "
                f"{row['fp_rate']:>6.1f}% | {row['f1']:>7.1f}% | {row['fp_count']:>8d} | "
                f"{row['effective_alarms']:>6d} |"
            )
        print("-" * 88)

    @pytest.mark.parametrize("long_dataset", build_long_test_params(), indirect=False)
    def test_ml_vs_test_recordings(self, long_dataset):
        """Evaluate the ML detector on the 60-second test recordings."""
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
        assert 0.0 <= metrics["adaptive_threshold"] <= 1.0
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
