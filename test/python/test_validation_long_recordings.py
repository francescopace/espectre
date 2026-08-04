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
    get_available_long_test_dataset_specs,
)

from conftest import (
    build_long_test_params,
    extract_motion_start_from_description,
    load_long_test_dataset,
)


def _assert_classic_replays_match(cached_metrics, packet_metrics) -> None:
    assert packet_metrics is not None, "Classic packet calibration failed"
    assert cached_metrics is not None, "Classic startup calibration failed"
    for key in (
        "baseline_eval_count",
        "movement_eval_count",
        "tp",
        "fn",
        "fp",
        "tn",
        "effective_alarms",
        "false_motion_evaluations",
    ):
        assert cached_metrics[key] == packet_metrics[key]
    for key in (
        "adaptive_threshold",
        "recall",
        "precision",
        "fp_rate",
        "f1",
    ):
        assert cached_metrics[key] == pytest.approx(
            packet_metrics[key], abs=1e-12
        )


def _representative_long_recording_param():
    specs = get_available_long_test_dataset_specs()
    if not specs:
        return pytest.param(
            None,
            marks=pytest.mark.skip(
                reason="No long-recording replays available in dataset_info.json"
            ),
            id="no_long_test_recordings",
        )
    spec = min(specs, key=lambda item: (int(item[2]), str(item[0])))
    test_path, motion_start_packet, num_packets, chip, _entry = spec
    return pytest.param(
        spec,
        id=(
            f"{chip.lower()}_representative_{Path(test_path).stem}_"
            f"{motion_start_packet}b_{num_packets - motion_start_packet}m"
        ),
    )


class TestLongRecordings:
    """Validate the curated 60-second recordings once per dataset."""

    _ml_rows = []
    _classic_rows = []

    @classmethod
    def setup_class(cls):
        cls._ml_rows = []
        cls._classic_rows = []

    @classmethod
    def teardown_class(cls):
        if not cls._ml_rows and not cls._classic_rows:
            return

        if cls._ml_rows:
            print("")
            print("=" * 88)
            print("                    LONG RECORDING ML SUMMARY (for seed search)")
            print("=" * 88)
            print("| Chip   | Recall  | Precision | FP Rate | F1-Score | FP Count | Alarms |")
            print("|--------|---------|-----------|---------|----------|----------|--------|")
            for row in sorted(cls._ml_rows, key=lambda item: item["chip"]):
                print(
                    f"| {row['chip']:<6} | {row['recall']:>6.1f}% | {row['precision']:>8.1f}% | "
                    f"{row['fp_rate']:>6.1f}% | {row['f1']:>7.1f}% | {row['fp_count']:>8d} | "
                    f"{row['effective_alarms']:>6d} |"
                )
            print("-" * 88)

        if cls._classic_rows:
            print("")
            print("=" * 108)
            print("                           LONG RECORDING CLASSIC SUMMARY")
            print("=" * 108)
            print("| Chip   | Recall  | Precision | FP Rate | F1-Score | FP Count |")
            print("|--------|---------|-----------|---------|----------|----------|")
            for row in sorted(cls._classic_rows, key=lambda item: item["chip"]):
                print(
                    f"| {row['chip']:<6} | {row['recall']:>6.1f}% | {row['precision']:>8.1f}% | "
                    f"{row['fp_rate']:>6.1f}% | {row['f1']:>7.1f}% | {row['fp_count']:>8d} |"
                )
            print("-" * 108)

    @pytest.mark.parametrize("long_dataset", build_long_test_params(), indirect=False)
    def test_long_recording_replays(self, long_dataset):
        """Validate split metadata plus ML and Classic replays once per recording."""
        if long_dataset is None:
            pytest.skip("No long-recording replays available in dataset_info.json")

        long_dataset = load_long_test_dataset(long_dataset)
        test_path, baseline_packets, movement_packets, motion_start_packet, chip, entry = long_dataset

        assert test_path.exists()
        assert len(baseline_packets) == motion_start_packet
        if extract_motion_start_from_description(entry.get("description")) is None:
            assert len(movement_packets) == 0
        else:
            assert len(movement_packets) > 0
        assert str(entry.get("chip", "")).upper() == chip

        ml_metrics = _evaluate_ml_long_recording(
            baseline_packets,
            movement_packets,
            source_path=test_path,
            motion_start_packet=motion_start_packet,
        )
        self.__class__._ml_rows.append(
            {
                "chip": chip,
                "motion_start_packet": motion_start_packet,
                "baseline_packets": len(baseline_packets),
                "movement_packets": len(movement_packets),
                "fp_count": ml_metrics["fp"],
                **ml_metrics,
            }
        )

        assert ml_metrics["baseline_eval_count"] >= 0
        assert ml_metrics["movement_eval_count"] >= 0
        assert 0.0 <= ml_metrics["recall"] <= 100.0
        assert 0.0 <= ml_metrics["precision"] <= 100.0
        assert 0.0 <= ml_metrics["fp_rate"] <= 100.0
        assert 0.0 <= ml_metrics["f1"] <= 100.0
        assert ml_metrics["effective_alarms"] >= 0
        assert ml_metrics["false_motion_evaluations"] >= 0

        classic_metrics = _evaluate_classic_long_recording(
            baseline_packets,
            movement_packets,
            source_path=test_path,
            motion_start_packet=motion_start_packet,
        )
        assert classic_metrics is not None, "Classic startup calibration failed"
        self.__class__._classic_rows.append(
            {
                "chip": chip,
                "motion_start_packet": motion_start_packet,
                "baseline_packets": len(baseline_packets),
                "movement_packets": len(movement_packets),
                "fp_count": classic_metrics["fp"],
                **classic_metrics,
            }
        )

        assert classic_metrics["baseline_eval_count"] >= 0
        assert classic_metrics["movement_eval_count"] >= 0
        assert 0.0 <= classic_metrics["adaptive_threshold"] <= 1.0
        assert 0.0 <= classic_metrics["recall"] <= 100.0
        assert 0.0 <= classic_metrics["precision"] <= 100.0
        assert 0.0 <= classic_metrics["fp_rate"] <= 100.0
        assert 0.0 <= classic_metrics["f1"] <= 100.0


@pytest.mark.parametrize(
    "long_dataset",
    [_representative_long_recording_param()],
    indirect=False,
)
def test_classic_long_recording_cached_rows_match_packet_replay(long_dataset):
    """Keep exact raw-versus-row parity on one deterministic long replay."""
    if long_dataset is None:
        pytest.skip("No long-recording replays available in dataset_info.json")

    test_path, baseline_packets, movement_packets, motion_start_packet, _chip, _entry = (
        load_long_test_dataset(long_dataset)
    )
    packet_metrics = _evaluate_classic_long_recording(
        baseline_packets,
        movement_packets,
    )
    cached_metrics = _evaluate_classic_long_recording(
        baseline_packets,
        movement_packets,
        source_path=test_path,
        motion_start_packet=motion_start_packet,
    )
    _assert_classic_replays_match(cached_metrics, packet_metrics)
