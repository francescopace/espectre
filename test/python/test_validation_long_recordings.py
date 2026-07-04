"""
Micro-ESPectre - Long recording validation tests.

These tests evaluate the ML detector on the 60-second recordings stored in
data/test/ and print a stable summary table used by the training seed-search
gate.
"""

import importlib.util
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from repo_paths import tools_dir

TESTS_DIR = Path(__file__).parent
sys.path.insert(0, str(TESTS_DIR))

from config import DEFAULT_SUBCARRIERS, SEG_WINDOW_SIZE
from detector_interface import MotionState
from ml_detector import MLDetector
from segmentation import SegmentationContext
from conftest import (
    DATA_DIR,
    DATASET_INFO_PATH,
    build_long_test_params,
    extract_motion_start_from_description,
    get_available_long_test_datasets,
)
from test_validation_real_data import run_calibration


TRAIN_ML_MODEL_PATH = tools_dir() / "10_train_ml_model.py"


def _load_train_ml_model_module():
    """Load the training script directly despite its numeric filename."""
    spec = importlib.util.spec_from_file_location("train_ml_model_gate", TRAIN_ML_MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _evaluate_ml_long_recording(baseline_packets, movement_packets):
    """Run MLDetector across a long recording split and return packet metrics."""
    use_cv_normalization = _long_recording_uses_cv_normalization(
        baseline_packets, movement_packets
    )
    detector = MLDetector(
        threshold=5.0,
        window_size=SEG_WINDOW_SIZE,
        use_cv_normalization=use_cv_normalization,
    )
    warmup = SEG_WINDOW_SIZE

    baseline_eval_count = max(len(baseline_packets) - warmup, 0)
    movement_eval_count = max(len(movement_packets) - warmup, 0)
    baseline_motion_packets = 0
    movement_with_motion = 0
    movement_without_motion = 0

    for i, pkt in enumerate(baseline_packets):
        detector.process_packet(pkt["csi_data"], DEFAULT_SUBCARRIERS)
        detector.update_state()
        if i >= warmup and detector.get_state() == MotionState.MOTION:
            baseline_motion_packets += 1

    for i, pkt in enumerate(movement_packets):
        detector.process_packet(pkt["csi_data"], DEFAULT_SUBCARRIERS)
        detector.update_state()
        if i >= warmup:
            if detector.get_state() == MotionState.MOTION:
                movement_with_motion += 1
            else:
                movement_without_motion += 1

    tp = movement_with_motion
    fn = movement_without_motion
    fp = baseline_motion_packets
    tn = max(baseline_eval_count - baseline_motion_packets, 0)

    recall = tp / (tp + fn) * 100.0 if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) * 100.0 if (tp + fp) > 0 else 0.0
    fp_rate = fp / baseline_eval_count * 100.0 if baseline_eval_count > 0 else 0.0
    f1 = (
        2 * (precision / 100.0) * (recall / 100.0) / ((precision + recall) / 100.0) * 100.0
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "baseline_eval_count": baseline_eval_count,
        "movement_eval_count": movement_eval_count,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "recall": recall,
        "precision": precision,
        "fp_rate": fp_rate,
        "f1": f1,
    }


def _long_recording_uses_cv_normalization(baseline_packets, movement_packets):
    """Return the shared production normalization mode for long recordings."""
    return True


def _evaluate_mvs_long_recording(baseline_packets, movement_packets):
    """Run production-style MVS with fixed subcarriers across a long recording split."""
    use_cv_normalization = _long_recording_uses_cv_normalization(
        baseline_packets, movement_packets
    )
    selected_band, adaptive_threshold = run_calibration(
        baseline_packets,
        num_subcarriers=64,
        algorithm="fixed_default",
        hint_band=DEFAULT_SUBCARRIERS,
        mvs_window_size=SEG_WINDOW_SIZE,
        use_cv_normalization=use_cv_normalization,
    )

    ctx = SegmentationContext(
        window_size=SEG_WINDOW_SIZE,
        threshold=adaptive_threshold,
        enable_hampel=True,
    )
    ctx.use_cv_normalization = use_cv_normalization

    warmup = SEG_WINDOW_SIZE
    baseline_eval_count = max(len(baseline_packets) - warmup, 0)
    movement_eval_count = max(len(movement_packets) - warmup, 0)
    baseline_motion_packets = 0
    movement_with_motion = 0
    movement_without_motion = 0

    for i, pkt in enumerate(baseline_packets):
        turbulence = ctx.calculate_spatial_turbulence(pkt["csi_data"], selected_band)
        ctx.add_turbulence(turbulence)
        ctx.update_state()
        if i >= warmup and ctx.get_state() == MotionState.MOTION:
            baseline_motion_packets += 1

    for i, pkt in enumerate(movement_packets):
        turbulence = ctx.calculate_spatial_turbulence(pkt["csi_data"], selected_band)
        ctx.add_turbulence(turbulence)
        ctx.update_state()
        if i >= warmup:
            if ctx.get_state() == MotionState.MOTION:
                movement_with_motion += 1
            else:
                movement_without_motion += 1

    tp = movement_with_motion
    fn = movement_without_motion
    fp = baseline_motion_packets
    tn = max(baseline_eval_count - baseline_motion_packets, 0)

    recall = tp / (tp + fn) * 100.0 if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) * 100.0 if (tp + fp) > 0 else 0.0
    fp_rate = fp / baseline_eval_count * 100.0 if baseline_eval_count > 0 else 0.0
    f1 = (
        2 * (precision / 100.0) * (recall / 100.0) / ((precision + recall) / 100.0) * 100.0
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "selected_band": list(selected_band) if selected_band is not None else [],
        "adaptive_threshold": adaptive_threshold,
        "use_cv_normalization": use_cv_normalization,
        "baseline_eval_count": baseline_eval_count,
        "movement_eval_count": movement_eval_count,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "recall": recall,
        "precision": precision,
        "fp_rate": fp_rate,
        "f1": f1,
    }


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
        print("| Chip   | Recall  | Precision | FP Rate | F1-Score | FP Count |")
        print("|--------|---------|-----------|---------|----------|----------|")
        for row in sorted(cls._rows, key=lambda item: item["chip"]):
            print(
                f"| {row['chip']:<6} | {row['recall']:>6.1f}% | {row['precision']:>8.1f}% | "
                f"{row['fp_rate']:>6.1f}% | {row['f1']:>7.1f}% | {row['fp_count']:>8d} |"
            )
        print("-" * 99)

    @pytest.mark.parametrize("long_dataset", build_long_test_params(), indirect=False)
    def test_ml_vs_test_recordings(self, long_dataset):
        """
        Evaluate the ML detector on the 60-second test recordings.

        The output table is intentionally stable because 10_train_ml_model.py
        parses it during seed search.
        """
        if long_dataset is None:
            pytest.skip("No long test recordings available in data/test")

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
        assert len(movement_packets) > 0
        assert metrics["baseline_eval_count"] >= 0
        assert metrics["movement_eval_count"] >= 0
        assert 0.0 <= metrics["recall"] <= 100.0
        assert 0.0 <= metrics["precision"] <= 100.0
        assert 0.0 <= metrics["fp_rate"] <= 100.0
        assert 0.0 <= metrics["f1"] <= 100.0
        assert str(entry.get("chip", "")).upper() == chip


class TestLongRecordingsMVS:
    """Validate MVS with fixed subcarriers on the curated 60-second recordings."""

    _rows = []

    @classmethod
    def setup_class(cls):
        cls._rows = []

    @classmethod
    def teardown_class(cls):
        if not cls._rows:
            return

        print("")
        print("=" * 118)
        print("                             LONG RECORDING MVS FIXED SUMMARY")
        print("=" * 118)
        print("| Chip   | Recall  | Precision | FP Rate | F1-Score | FP Count | CV Norm |")
        print("|--------|---------|-----------|---------|----------|----------|---------|")
        for row in sorted(cls._rows, key=lambda item: item["chip"]):
            print(
                f"| {row['chip']:<6} | {row['recall']:>6.1f}% | {row['precision']:>8.1f}% | "
                f"{row['fp_rate']:>6.1f}% | {row['f1']:>7.1f}% | {row['fp_count']:>8d} | "
                f"{'ON' if row['use_cv_normalization'] else 'OFF':>7} |"
            )
        print("-" * 118)

    @pytest.mark.parametrize("long_dataset", build_long_test_params(), indirect=False)
    def test_mvs_vs_test_recordings(self, long_dataset):
        """Evaluate MVS with fixed subcarriers on the 60-second test recordings."""
        if long_dataset is None:
            pytest.skip("No long test recordings available in data/test")

        _, baseline_packets, movement_packets, motion_start_packet, chip, entry = long_dataset
        metrics = _evaluate_mvs_long_recording(baseline_packets, movement_packets)
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
        assert len(movement_packets) > 0
        assert len(metrics["selected_band"]) == 12
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

    def test_motion_start_packet_is_parsed_from_dataset_metadata(self):
        datasets = get_available_long_test_datasets()
        assert datasets, f"No datasets found via {DATASET_INFO_PATH}"

        chips = {chip for _, _, _, _, chip, _ in datasets}
        assert {"C3", "C5", "C6"}.issubset(chips)

        for _, baseline_packets, movement_packets, motion_start_packet, _, entry in datasets:
            expected = extract_motion_start_from_description(entry.get("description"))
            assert expected == motion_start_packet
            assert len(baseline_packets) == motion_start_packet
            assert len(movement_packets) > 0

    @pytest.mark.parametrize("long_dataset", build_long_test_params(), indirect=False)
    def test_long_test_loader_splits_stream_consistently(self, long_dataset):
        if long_dataset is None:
            pytest.skip("No long test recordings available in data/test")

        test_path, baseline_packets, movement_packets, motion_start_packet, chip, entry = long_dataset

        assert test_path.parent == DATA_DIR / "test"
        assert test_path.exists()
        assert len(baseline_packets) == motion_start_packet
        assert len(movement_packets) > 0
        assert str(entry.get("chip", "")).upper() == chip

    def test_long_gate_output_parser_extracts_rows(self):
        train_ml_model = _load_train_ml_model_module()
        output = """
===================================================================================
                    LONG RECORDING ML SUMMARY (for seed search)
===================================================================================
| Chip   | Recall  | Precision | FP Rate | F1-Score | FP Count |
|--------|---------|-----------|---------|----------|----------|
| C3     |   99.9% |     97.1% |    3.0% |    98.5% |       89 |
| C5     |   98.4% |     96.3% |    4.1% |    97.3% |      121 |
| C6     |   96.7% |     94.8% |    5.2% |    95.7% |      165 |
-----------------------------------------------------------------------------------
"""
        metrics = train_ml_model._parse_long_recording_metrics(output)
        assert metrics is not None
        assert metrics["pass_count"] == 2
        assert metrics["total_fp"] == 375
        assert metrics["mean_f1"] == pytest.approx((98.5 + 97.3 + 95.7) / 3.0)
        assert metrics["worst_chip_f1"] == pytest.approx(95.7)
