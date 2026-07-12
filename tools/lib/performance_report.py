"""
Shared performance-report helpers for tests and tooling.
"""

from __future__ import annotations

import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np

from .bootstrap import setup_paths
from .dataset_metadata import build_calibrated_classic_detector
from .repo_paths import data_dir, repo_root

setup_paths()

from config import (
    DEFAULT_SUBCARRIERS,
    EVALUATION_INTERVAL,
    MOTION_OFF_HITS,
    MOTION_ON_HITS,
    SEG_WINDOW_SIZE as DETECTOR_DEFAULT_WINDOW_SIZE,
)
from detector_interface import MotionState
from features import FEATURE_NAMES as RUNTIME_FEATURE_NAMES
from runtime_policy import RuntimeMotionPolicy
from tools.lib.csi_io import load_npz_as_packets


DATA_DIR = data_dir()
PERFORMANCE_DOC_PATH = repo_root() / "docs" / "PERFORMANCE.md"
KEY_RUNTIME_FEATURE_GATES = {
    "turb_mad_over_mean": 0.5,
    "turb_autocorr": 0.5,
    "l1_delta": 0.5,
    "l1_delta_std": 0.5,
    "l1_delta_waveform_length": 0.5,
}


def evaluate_idle_runtime_policy(raw_motion_states: Sequence[bool]) -> Dict[str, int]:
    """Evaluate runtime cadence and consecutive-hit filtering on IDLE data."""
    policy = RuntimeMotionPolicy(EVALUATION_INTERVAL, MOTION_ON_HITS, MOTION_OFF_HITS)
    effective_alarms = 0
    false_motion_evaluations = 0

    for raw_motion in raw_motion_states[EVALUATION_INTERVAL - 1::EVALUATION_INTERVAL]:
        raw_state = MotionState.MOTION if raw_motion else MotionState.IDLE
        effective_state, changed = policy.apply_state(raw_state)
        if changed and effective_state == MotionState.MOTION:
            effective_alarms += 1
        if effective_state == MotionState.MOTION:
            false_motion_evaluations += 1

    return {
        "effective_alarms": effective_alarms,
        "false_motion_evaluations": false_motion_evaluations,
    }


CHIP_ORDER = ("C3", "C5", "C6", "S3")
PAIRED_CHIP_LABELS = {
    "C3": "ESP32-C3",
    "C5": "ESP32-C5",
    "C6": "ESP32-C6",
    "S3": "ESP32-S3",
}


def _load_dataset_info() -> Dict[str, Any]:
    dataset_info_path = DATA_DIR / "dataset_info.json"
    if not dataset_info_path.exists():
        return {"files": {}}
    with dataset_info_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def _get_available_paired_datasets_cached() -> tuple[tuple[Path, Path, int, str, str], ...]:
    """Return explicit static-presence/motion pairs (HT20: 64 SC only)."""
    dataset_info = _load_dataset_info()
    files = dataset_info.get("files", {})
    motion_by_filename = {
        entry.get("filename"): entry
        for entry in files.get("motion", [])
        if entry.get("filename")
    }

    pair_entries = []
    for static_entry in files.get("static_presence", []):
        if static_entry.get("subcarriers") != 64:
            continue
        motion_filename = static_entry.get("optimal_pair_motion_file")
        motion_entry = motion_by_filename.get(motion_filename)
        if not motion_entry or motion_entry.get("subcarriers") != 64:
            continue

        chip = static_entry.get("chip")
        static_path = DATA_DIR / "static_presence" / static_entry["filename"]
        motion_path = DATA_DIR / "motion" / motion_filename
        if not chip or not static_path.exists() or not motion_path.exists():
            continue

        environment = static_entry.get("environment") or "unknown"
        dataset_id = f"{str(chip).lower()}_{environment}_{static_path.stem}"
        pair_entries.append((static_path, motion_path, 64, str(chip).upper(), dataset_id))

    pair_entries.sort(key=lambda item: (item[3], item[4]))
    return tuple(pair_entries)


def get_available_paired_datasets() -> list[tuple[Path, Path, int, str, str]]:
    """Return the paired real-data datasets used by the performance report."""
    return list(_get_available_paired_datasets_cached())


@lru_cache(maxsize=1)
def _get_available_empty_datasets_cached() -> tuple[Path, ...]:
    empty_dir = DATA_DIR / "empty"
    return tuple(sorted(empty_dir.glob("empty_*_64sc_*.npz")))


def get_available_empty_datasets() -> list[Path]:
    """Return the empty-room recordings used by the ML FP gate."""
    return list(_get_available_empty_datasets_cached())


def get_available_chip_types() -> list[str]:
    """Return the stable set of chips covered by the paired real-data datasets."""
    chips = []
    for _static_path, _motion_path, _num_sc, chip, _dataset_id in get_available_paired_datasets():
        chips.append(chip)
    return sorted(dict.fromkeys(chips))


@lru_cache(maxsize=None)
def _load_packets_cached(path_value: str) -> tuple[dict[str, Any], ...]:
    """Load and cache packet dictionaries for one .npz dataset."""
    return tuple(load_npz_as_packets(Path(path_value)))


@lru_cache(maxsize=None)
def load_real_data_cached(static_presence_path: str | Path, motion_path: str | Path) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    """Cache paired static-presence and motion packet streams."""
    return (
        _load_packets_cached(str(static_presence_path)),
        _load_packets_cached(str(motion_path)),
    )


@lru_cache(maxsize=None)
def load_empty_room_packets(empty_dataset_path: str | Path) -> tuple[dict[str, Any], ...]:
    """Cache empty-room packet streams across ML FP checks."""
    return _load_packets_cached(str(empty_dataset_path))


def evaluate_detector_packets(
    detector: Any,
    static_presence_packets: Sequence[dict[str, Any]],
    motion_packets: Sequence[dict[str, Any]],
    selected_band: Sequence[int],
) -> Dict[str, float]:
    """Replay one baseline/motion pair through a detector."""
    num_baseline = len(static_presence_packets)
    num_movement = len(motion_packets)

    static_presence_motion_packets = 0
    for pkt in static_presence_packets:
        detector.process_packet(pkt["csi_data"], selected_band)
        detector.update_state()
        if detector.get_state() == MotionState.MOTION:
            static_presence_motion_packets += 1

    motion_with_motion = 0
    motion_without_motion = 0
    for pkt in motion_packets:
        detector.process_packet(pkt["csi_data"], selected_band)
        detector.update_state()
        if detector.get_state() == MotionState.MOTION:
            motion_with_motion += 1
        else:
            motion_without_motion += 1

    pkt_tp = motion_with_motion
    pkt_fn = motion_without_motion
    pkt_tn = num_baseline - static_presence_motion_packets
    pkt_fp = static_presence_motion_packets
    pkt_recall = pkt_tp / (pkt_tp + pkt_fn) * 100.0 if (pkt_tp + pkt_fn) > 0 else 0.0
    pkt_precision = pkt_tp / (pkt_tp + pkt_fp) * 100.0 if (pkt_tp + pkt_fp) > 0 else 0.0
    pkt_fp_rate = pkt_fp / num_baseline * 100.0 if num_baseline > 0 else 0.0
    pkt_f1 = (
        2 * (pkt_precision / 100.0) * (pkt_recall / 100.0) / ((pkt_precision + pkt_recall) / 100.0) * 100.0
        if (pkt_precision + pkt_recall) > 0
        else 0.0
    )
    return {
        "tp": pkt_tp,
        "fn": pkt_fn,
        "tn": pkt_tn,
        "fp": pkt_fp,
        "recall": pkt_recall,
        "precision": pkt_precision,
        "fp_rate": pkt_fp_rate,
        "f1": pkt_f1,
        "num_baseline": num_baseline,
        "num_movement": num_movement,
    }


@lru_cache(maxsize=None)
def compute_classic_dataset_result(
    static_presence_path: str | Path,
    motion_path: str | Path,
    selected_band: tuple[int, ...],
    window_size: int,
) -> Optional[tuple[float, Dict[str, float]]]:
    """Run the Classic replay once per dataset and cache the resulting metrics."""
    static_presence_packets, motion_packets = load_real_data_cached(
        static_presence_path,
        motion_path,
    )
    calibrated = build_calibrated_classic_detector(
        static_presence_packets,
        selected_subcarriers=selected_band,
    )
    if calibrated is None:
        return None

    detector, adaptive_threshold = calibrated
    metrics = evaluate_detector_packets(
        detector,
        static_presence_packets,
        motion_packets,
        selected_band,
    )
    return adaptive_threshold, metrics


@lru_cache(maxsize=None)
def compute_ml_dataset_result(
    static_presence_path: str | Path,
    motion_path: str | Path,
    selected_subcarriers: tuple[int, ...],
    window_size: int,
    threshold: float,
    feature_names: tuple[str, ...] = (),
) -> tuple[Dict[str, float], Dict[str, Dict[str, tuple[float, ...]]]]:
    """Run the ML replay once per dataset and cache metrics plus optional features."""
    from ml_detector import MLDetector, FEATURE_NAMES as EXPORTED_FEATURE_NAMES, predict

    assert tuple(EXPORTED_FEATURE_NAMES) == tuple(RUNTIME_FEATURE_NAMES)

    static_presence_packets, motion_packets = load_real_data_cached(
        static_presence_path,
        motion_path,
    )
    detector = MLDetector(window_size=window_size, threshold=threshold)

    feature_indices = {
        feature_name: EXPORTED_FEATURE_NAMES.index(feature_name)
        for feature_name in feature_names
    }
    static_series = {feature_name: [] for feature_name in feature_names}
    motion_series = {feature_name: [] for feature_name in feature_names}

    num_baseline = len(static_presence_packets)
    num_movement = len(motion_packets)
    warmup = window_size
    static_presence_motion_packets = 0
    static_presence_eval_count = max(num_baseline - warmup, 0)

    for i, pkt in enumerate(static_presence_packets):
        detector.process_packet(pkt["csi_data"], selected_subcarriers)
        if not detector.is_ready():
            continue

        values = detector._extract_features()
        probability = predict(values)
        current_state = MotionState.MOTION if probability > threshold else MotionState.IDLE
        if i >= warmup and feature_indices:
            for feature_name, feature_idx in feature_indices.items():
                value = float(values[feature_idx])
                if np.isfinite(value):
                    static_series[feature_name].append(value)
        if i >= warmup and current_state == MotionState.MOTION:
            static_presence_motion_packets += 1

    motion_with_motion = 0
    motion_without_motion = 0
    motion_eval_count = max(num_movement - warmup, 0)
    for i, pkt in enumerate(motion_packets):
        detector.process_packet(pkt["csi_data"], selected_subcarriers)
        values = detector._extract_features()
        probability = predict(values)
        current_state = MotionState.MOTION if probability > threshold else MotionState.IDLE
        if i >= warmup:
            if feature_indices:
                for feature_name, feature_idx in feature_indices.items():
                    value = float(values[feature_idx])
                    if np.isfinite(value):
                        motion_series[feature_name].append(value)
            if current_state == MotionState.MOTION:
                motion_with_motion += 1
            else:
                motion_without_motion += 1

    pkt_tp = motion_with_motion
    pkt_fn = motion_without_motion
    pkt_tn = static_presence_eval_count - static_presence_motion_packets if static_presence_eval_count > 0 else 0
    pkt_fp = static_presence_motion_packets
    pkt_recall = pkt_tp / (pkt_tp + pkt_fn) * 100.0 if (pkt_tp + pkt_fn) > 0 else 0.0
    pkt_precision = pkt_tp / (pkt_tp + pkt_fp) * 100.0 if (pkt_tp + pkt_fp) > 0 else 0.0
    pkt_fp_rate = pkt_fp / static_presence_eval_count * 100.0 if static_presence_eval_count > 0 else 0.0
    pkt_f1 = (
        2 * (pkt_precision / 100.0) * (pkt_recall / 100.0) / ((pkt_precision + pkt_recall) / 100.0) * 100.0
        if (pkt_precision + pkt_recall) > 0
        else 0.0
    )

    metrics = {
        "tp": pkt_tp,
        "fn": pkt_fn,
        "tn": pkt_tn,
        "fp": pkt_fp,
        "recall": pkt_recall,
        "precision": pkt_precision,
        "fp_rate": pkt_fp_rate,
        "f1": pkt_f1,
        "num_baseline": static_presence_eval_count,
        "num_movement": motion_eval_count,
    }
    feature_payload = {
        "baseline": {name: tuple(values) for name, values in static_series.items()},
        "motion": {name: tuple(values) for name, values in motion_series.items()},
    }
    return metrics, feature_payload


@lru_cache(maxsize=None)
def compute_ml_empty_fp_result(
    empty_dataset_path: str | Path,
    selected_subcarriers: tuple[int, ...],
    window_size: int,
    threshold: float,
) -> Dict[str, float]:
    """Run the empty-room ML FP replay once per dataset and cache the result."""
    from ml_detector import MLDetector

    packets = load_empty_room_packets(empty_dataset_path)
    detector = MLDetector(window_size=window_size, threshold=threshold)

    eval_count = max(len(packets) - window_size, 0)
    motion_packets = 0
    for i, pkt in enumerate(packets):
        detector.process_packet(pkt["csi_data"], selected_subcarriers)
        detector.update_state()
        if i >= window_size and detector.get_state() == MotionState.MOTION:
            motion_packets += 1

    fp_rate = motion_packets / eval_count * 100.0 if eval_count > 0 else 0.0
    return {
        "motion_packets": motion_packets,
        "eval_count": eval_count,
        "fp_rate": fp_rate,
    }


def extract_motion_start_from_description(description: Optional[str]) -> Optional[int]:
    """Extract motion start packet index from free-text test metadata."""
    if not description:
        return None

    import re

    match = re.search(
        r"motion\s+starts\s+at\s+packet(?:\s+index)?(?:\s+n\.)?\s+(\d+)",
        str(description),
        re.IGNORECASE,
    )
    if match:
        return int(match.group(1))
    return None


def _normalize_long_test_chip_filter(chips: Optional[Iterable[str]]) -> Optional[tuple[str, ...]]:
    if not chips:
        return None
    return tuple(sorted({str(chip).upper() for chip in chips}))


@lru_cache(maxsize=None)
def _get_available_long_test_datasets_cached(chips_key: Optional[tuple[str, ...]]) -> tuple[tuple[Any, ...], ...]:
    """Return available long test recordings with validated split metadata."""
    dataset_info = _load_dataset_info()
    test_entries = dataset_info.get("files", {}).get("test", [])
    if not test_entries:
        return tuple()

    requested = set(chips_key) if chips_key else None
    datasets = []

    for entry in test_entries:
        chip = str(entry.get("chip", "")).upper()
        if requested and chip not in requested:
            continue

        filename = entry.get("filename")
        if not filename:
            continue

        test_path = DATA_DIR / "test" / filename
        if not test_path.exists():
            continue

        packets = load_npz_as_packets(test_path)
        if len(packets) < 2:
            continue

        motion_start_packet = extract_motion_start_from_description(entry.get("description"))
        if motion_start_packet is None:
            motion_start_packet = len(packets)

        if motion_start_packet <= 0 or motion_start_packet > len(packets):
            continue

        static_presence_packets = packets[:motion_start_packet]
        motion_packets = packets[motion_start_packet:]
        datasets.append(
            (
                test_path,
                static_presence_packets,
                motion_packets,
                motion_start_packet,
                chip,
                entry,
            )
        )

    datasets.sort(key=lambda item: item[4])
    return tuple(datasets)


def get_available_long_test_datasets(chips: Optional[Iterable[str]] = None) -> list[tuple[Any, ...]]:
    """Return cached long test recordings with validated split metadata."""
    return list(_get_available_long_test_datasets_cached(_normalize_long_test_chip_filter(chips)))


def evaluate_ml_long_recording(
    baseline_packets: Sequence[dict[str, Any]],
    movement_packets: Sequence[dict[str, Any]],
) -> Dict[str, float]:
    """Run MLDetector across a long recording split and return packet metrics."""
    from ml_detector import MLDetector

    detector = MLDetector(
        threshold=0.5,
        window_size=DETECTOR_DEFAULT_WINDOW_SIZE,
    )
    warmup = DETECTOR_DEFAULT_WINDOW_SIZE

    baseline_eval_count = max(len(baseline_packets) - warmup, 0)
    movement_eval_count = max(len(movement_packets) - warmup, 0)
    baseline_motion_packets = 0
    baseline_motion_states = []
    movement_with_motion = 0
    movement_without_motion = 0

    for i, pkt in enumerate(baseline_packets):
        detector.process_packet(pkt["csi_data"], DEFAULT_SUBCARRIERS)
        detector.update_state()
        if i >= warmup and detector.get_state() == MotionState.MOTION:
            baseline_motion_packets += 1
        if i >= warmup:
            baseline_motion_states.append(detector.get_state() == MotionState.MOTION)

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

    policy_metrics = evaluate_idle_runtime_policy(baseline_motion_states)
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
        **policy_metrics,
    }


def evaluate_classic_long_recording(
    baseline_packets: Sequence[dict[str, Any]],
    movement_packets: Sequence[dict[str, Any]],
) -> Optional[Dict[str, float]]:
    """Run startup-calibrated ClassicDetector across a long recording split."""
    calibrated = build_calibrated_classic_detector(
        baseline_packets,
        selected_subcarriers=DEFAULT_SUBCARRIERS,
    )
    if calibrated is None:
        return None
    detector, adaptive_threshold = calibrated
    warmup = DETECTOR_DEFAULT_WINDOW_SIZE
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
        "adaptive_threshold": adaptive_threshold,
        "warmup": warmup,
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


def _average_detector_metrics(entries: Sequence[Dict[str, float]]) -> Optional[Dict[str, float]]:
    if not entries:
        return None
    return {
        "count": len(entries),
        "recall": sum(entry["recall"] for entry in entries) / len(entries),
        "precision": sum(entry["precision"] for entry in entries) / len(entries),
        "fp_rate": sum(entry["fp_rate"] for entry in entries) / len(entries),
        "f1": sum(entry["f1"] for entry in entries) / len(entries),
    }


def compute_performance_report_data() -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """Compute all metrics published in docs/PERFORMANCE.md."""
    paired_results: dict[str, dict[str, list[Dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    long_results: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for static_path, motion_path, _num_sc, chip, _dataset_id in get_available_paired_datasets():
        classic_result = compute_classic_dataset_result(
            static_path,
            motion_path,
            tuple(DEFAULT_SUBCARRIERS),
            DETECTOR_DEFAULT_WINDOW_SIZE,
        )
        if classic_result is not None:
            _adaptive_threshold, classic_metrics = classic_result
            paired_results["classic"][chip].append(classic_metrics)

        ml_metrics, _feature_payload = compute_ml_dataset_result(
            static_path,
            motion_path,
            tuple(DEFAULT_SUBCARRIERS),
            DETECTOR_DEFAULT_WINDOW_SIZE,
            0.5,
        )
        paired_results["ml"][chip].append(ml_metrics)

    for _test_path, baseline_packets, movement_packets, _motion_start, chip, _entry in get_available_long_test_datasets():
        classic_metrics = evaluate_classic_long_recording(baseline_packets, movement_packets)
        if classic_metrics is not None:
            long_results["classic"][chip].append(classic_metrics["fp_rate"])

        ml_metrics = evaluate_ml_long_recording(baseline_packets, movement_packets)
        long_results["ml"][chip].append(ml_metrics["fp_rate"])

    paired_summary: Dict[str, Dict[str, Dict[str, float]]] = {"classic": {}, "ml": {}}
    for algorithm, by_chip in paired_results.items():
        for chip, entries in by_chip.items():
            averaged = _average_detector_metrics(entries)
            if averaged is not None:
                paired_summary[algorithm][chip] = averaged

    long_summary: Dict[str, Dict[str, Dict[str, float]]] = {"classic": {}, "ml": {}}
    for algorithm, by_chip in long_results.items():
        for chip, fp_rates in by_chip.items():
            if not fp_rates:
                continue
            long_summary[algorithm][chip] = {
                "avg_fp_rate": sum(fp_rates) / len(fp_rates),
                "max_fp_rate": max(fp_rates),
            }

    return {
        "paired": paired_summary,
        "long_quiet": long_summary,
    }


def render_performance_report_markdown(
    report_data: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
) -> str:
    """Render the published performance markdown from computed metrics."""
    lines = [
        "# Performance Metrics",
        "",
        "This document provides detailed performance metrics for ESPectre's motion detection algorithms.",
        "",
        "Generated by: `tools/generate_performance_report.py`",
        "",
        "- **Classic Detector**: Uses L1-Delta as the primary metric, with a gated moving-variance recovery vote.",
        "- **ML Detector**: Uses a pretrained neural network model based on turbulence and spectral features.",
        "",
        "See [ALGORITHMS.md](ALGORITHMS.md) for the full detector design.",
        "",
        "---",
        "",
        "## Performance Targets",
        "",
        "| Metric | Target | ",
        "|-------|--------|",
        "| Recall | >95% |",
        "| FP Rate | <5% |",
        "",
        "---",
        "",
        "## Test Scripts",
        "",
        "- C++ `test_motion_detection`",
        "- C++ `test_long_recordings`",
        "- Python `TestPerformanceMetrics`",
        "- Python `test_validation_long_recordings.py`",
        "",
        "---",
        "",
        "## Paired Real-Data Validation (empty+static_presence / motion)",
        "",
        "### Classic Detector",
        "",
    ]

    paired = report_data["paired"]
    paired_header = "| Metric | " + " | ".join(PAIRED_CHIP_LABELS[chip] for chip in CHIP_ORDER) + " |"
    paired_divider = "|--------|" + "|".join("----------" for _ in CHIP_ORDER) + "|"
    lines.append(paired_header)
    lines.append(paired_divider)
    for key, label in (
        ("recall", "Recall"),
        ("precision", "Precision"),
        ("fp_rate", "FP Rate"),
        ("f1", "F1-Score"),
    ):
        values = []
        for chip in CHIP_ORDER:
            metrics = paired["classic"].get(chip)
            values.append(f"{metrics[key]:.1f}%" if metrics is not None else "N/A")
        lines.append(f"| {label} | " + " | ".join(values) + " |")

    lines.extend([
        "",
        "### ML Detector",
        "",
        paired_header,
        paired_divider,
    ])
    for key, label in (
        ("recall", "Recall"),
        ("precision", "Precision"),
        ("fp_rate", "FP Rate"),
        ("f1", "F1-Score"),
    ):
        values = []
        for chip in CHIP_ORDER:
            metrics = paired["ml"].get(chip)
            values.append(f"{metrics[key]:.1f}%" if metrics is not None else "N/A")
        lines.append(f"| {label} | " + " | ".join(values) + " |")

    lines.extend([
        "",
        "---",
        "",
        "## Long Quiet Real-Data Validation",
        "",
        "### Classic Detector",
        "",
        "| Metric | C3 | C5 | C6 | S3 |",
        "|--------|----|----|----|----|",
    ])

    long_quiet = report_data["long_quiet"]
    for key, label in (
        ("avg_fp_rate", "Avg FP Rate"),
        ("max_fp_rate", "Max FP Rate"),
    ):
        values = []
        for chip in CHIP_ORDER:
            metrics = long_quiet["classic"].get(chip)
            values.append(f"{metrics[key]:.2f}%" if metrics is not None else "N/A")
        lines.append(f"| {label} | " + " | ".join(values) + " |")

    lines.extend([
        "",
        "### ML Detector",
        "",
        "| Metric | C3 | C5 | C6 | S3 |",
        "|--------|----|----|----|----|",
    ])
    for key, label in (
        ("avg_fp_rate", "Avg FP Rate"),
        ("max_fp_rate", "Max FP Rate"),
    ):
        values = []
        for chip in CHIP_ORDER:
            metrics = long_quiet["ml"].get(chip)
            values.append(f"{metrics[key]:.2f}%" if metrics is not None else "N/A")
        lines.append(f"| {label} | " + " | ".join(values) + " |")

    return "\n".join(lines) + "\n"


def write_performance_report(output_path: Optional[Path] = None) -> Path:
    """Compute and write docs/PERFORMANCE.md."""
    destination = PERFORMANCE_DOC_PATH if output_path is None else Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        render_performance_report_markdown(compute_performance_report_data()),
        encoding="utf-8",
    )
    return destination
