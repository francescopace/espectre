"""
Shared performance-report helpers for tests and tooling.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Sequence

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
from tools.lib.csi_io import load_npz_as_packets, load_npz_csi_data


DATA_DIR = data_dir()
PERFORMANCE_DOC_PATH = repo_root() / "docs" / "performance" / "README.md"
KEY_RUNTIME_FEATURE_GATES = {
    "turb_mad_over_mean": 0.5,
    "turb_autocorr": 0.5,
    "l1_delta": 0.5,
    "l1_delta_std": 0.5,
    "l1_delta_waveform_length": 0.5,
}


def evaluate_idle_runtime_policy(raw_motion_states: Sequence[bool]) -> Dict[str, int]:
    """Evaluate runtime cadence and consecutive-hit filtering on IDLE data."""
    return _evaluate_idle_runtime_policy_evaluations(
        raw_motion_states[EVALUATION_INTERVAL - 1::EVALUATION_INTERVAL]
    )


def _evaluate_idle_runtime_policy_evaluations(raw_motion_states: Sequence[bool]) -> Dict[str, int]:
    """Apply production hit filtering to states sampled at evaluation ticks."""
    policy = RuntimeMotionPolicy(EVALUATION_INTERVAL, MOTION_ON_HITS, MOTION_OFF_HITS)
    effective_alarms = 0
    false_motion_evaluations = 0

    for raw_motion in raw_motion_states:
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

ProgressCallback = Callable[[str], None]
ExecutionInfo = Dict[str, Any]


class _CsiRowView(Sequence[Any]):
    """Zero-copy unsigned-byte rows over one contiguous CSI matrix."""

    def __init__(self, matrix: np.ndarray, start: int = 0, stop: Optional[int] = None):
        if matrix.ndim != 2 or matrix.dtype != np.int8 or not matrix.flags.c_contiguous:
            raise ValueError("CSI matrix must be a contiguous two-dimensional int8 array")
        matrix_stop = len(matrix) if stop is None else int(stop)
        if start < 0 or matrix_stop < start or matrix_stop > len(matrix):
            raise ValueError("Invalid CSI row view bounds")
        self._matrix = matrix
        self._bytes = memoryview(matrix).cast("B")
        self._row_size = matrix.shape[1]
        self._start = int(start)
        self._stop = matrix_stop

    def __len__(self) -> int:
        return self._stop - self._start

    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step != 1:
                return [self[i] for i in range(start, stop, step)]
            return _CsiRowView(
                self._matrix,
                self._start + start,
                self._start + stop,
            )

        row_index = int(index)
        if row_index < 0:
            row_index += len(self)
        if row_index < 0 or row_index >= len(self):
            raise IndexError("CSI row index out of range")
        byte_start = (self._start + row_index) * self._row_size
        return self._bytes[byte_start:byte_start + self._row_size]


def _emit_progress(progress: Optional[ProgressCallback], message: str) -> None:
    if progress is not None:
        progress(message)


def _format_progress_duration(seconds: float) -> str:
    if seconds < 60.0:
        return f"{seconds:.2f}s"
    minutes, remaining_seconds = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m {remaining_seconds:.2f}s"
    hours, remaining_minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h {int(remaining_minutes)}m {remaining_seconds:.2f}s"


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
def _get_available_long_test_dataset_specs_cached(
    chips_key: Optional[tuple[str, ...]],
) -> tuple[tuple[Any, ...], ...]:
    """Return long-recording metadata without loading packet payloads."""
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

        num_packets = int(entry.get("num_packets", 0) or 0)
        if num_packets < 2:
            continue

        motion_start_packet = extract_motion_start_from_description(entry.get("description"))
        if motion_start_packet is None:
            motion_start_packet = num_packets

        if motion_start_packet <= 0 or motion_start_packet > num_packets:
            continue

        datasets.append(
            (
                test_path,
                motion_start_packet,
                num_packets,
                chip,
                entry,
            )
        )

    datasets.sort(key=lambda item: item[3])
    return tuple(datasets)


def get_available_long_test_dataset_specs(
    chips: Optional[Iterable[str]] = None,
) -> list[tuple[Any, ...]]:
    """Return lightweight long-recording specs suitable for parametrization."""
    return list(
        _get_available_long_test_dataset_specs_cached(
            _normalize_long_test_chip_filter(chips)
        )
    )


@lru_cache(maxsize=None)
def _load_long_test_csi_cached(path_value: str) -> np.ndarray:
    """Load one compact CSI matrix per worker process."""
    return load_npz_csi_data(Path(path_value))


def load_long_test_dataset(spec: tuple[Any, ...]) -> tuple[Any, ...]:
    """Materialize one long-recording spec as baseline and movement views."""
    test_path, motion_start_packet, num_packets, chip, entry = spec
    matrix = _load_long_test_csi_cached(str(test_path))
    if len(matrix) != num_packets:
        raise ValueError(
            f"Packet count mismatch for {test_path}: metadata={num_packets}, npz={len(matrix)}"
        )
    packets = _CsiRowView(matrix)
    return (
        test_path,
        packets[:motion_start_packet],
        packets[motion_start_packet:],
        motion_start_packet,
        chip,
        entry,
    )


def get_available_long_test_datasets(chips: Optional[Iterable[str]] = None) -> list[tuple[Any, ...]]:
    """Load long test recordings with validated split metadata."""
    return [
        load_long_test_dataset(spec)
        for spec in get_available_long_test_dataset_specs(chips=chips)
    ]


def _packet_csi_data(packet: Any) -> Any:
    """Return CSI bytes from a packet dictionary or a compact CSI row."""
    return packet["csi_data"] if isinstance(packet, dict) else packet


def evaluate_ml_long_recording(
    baseline_packets: Sequence[Any],
    movement_packets: Sequence[Any],
) -> Dict[str, float]:
    """Run MLDetector at the production evaluation cadence."""
    from ml_detector import MLDetector

    detector = MLDetector(
        threshold=0.5,
        window_size=DETECTOR_DEFAULT_WINDOW_SIZE,
    )
    warmup = DETECTOR_DEFAULT_WINDOW_SIZE

    baseline_eval_count = 0
    movement_eval_count = 0
    baseline_motion_packets = 0
    baseline_motion_states = []
    movement_with_motion = 0
    movement_without_motion = 0

    packets_since_evaluation = 0
    for i, pkt in enumerate(baseline_packets):
        detector.process_packet(_packet_csi_data(pkt), DEFAULT_SUBCARRIERS)
        packets_since_evaluation += 1
        if packets_since_evaluation < EVALUATION_INTERVAL:
            continue
        detector.update_state()
        packets_since_evaluation = 0
        if i >= warmup:
            baseline_eval_count += 1
        if i >= warmup and detector.get_state() == MotionState.MOTION:
            baseline_motion_packets += 1
        if i >= warmup:
            baseline_motion_states.append(detector.get_state() == MotionState.MOTION)

    for i, pkt in enumerate(movement_packets):
        detector.process_packet(_packet_csi_data(pkt), DEFAULT_SUBCARRIERS)
        packets_since_evaluation += 1
        if packets_since_evaluation < EVALUATION_INTERVAL:
            continue
        detector.update_state()
        packets_since_evaluation = 0
        if i >= warmup:
            movement_eval_count += 1
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

    policy_metrics = _evaluate_idle_runtime_policy_evaluations(baseline_motion_states)
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
    baseline_packets: Sequence[Any],
    movement_packets: Sequence[Any],
) -> Optional[Dict[str, float]]:
    """Run startup-calibrated ClassicDetector at the production cadence."""
    calibrated = build_calibrated_classic_detector(
        baseline_packets,
        selected_subcarriers=DEFAULT_SUBCARRIERS,
    )
    if calibrated is None:
        return None
    detector, adaptive_threshold = calibrated
    warmup = DETECTOR_DEFAULT_WINDOW_SIZE
    baseline_eval_count = 0
    movement_eval_count = 0
    baseline_motion_packets = 0
    movement_with_motion = 0
    movement_without_motion = 0

    baseline_motion_states = []

    packets_since_evaluation = 0
    for i, pkt in enumerate(baseline_packets):
        detector.process_packet(_packet_csi_data(pkt), DEFAULT_SUBCARRIERS)
        packets_since_evaluation += 1
        if packets_since_evaluation < EVALUATION_INTERVAL:
            continue
        detector.update_state()
        packets_since_evaluation = 0
        if i >= warmup:
            baseline_eval_count += 1
        if i >= warmup and detector.get_state() == MotionState.MOTION:
            baseline_motion_packets += 1
        if i >= warmup:
            baseline_motion_states.append(detector.get_state() == MotionState.MOTION)

    for i, pkt in enumerate(movement_packets):
        detector.process_packet(_packet_csi_data(pkt), DEFAULT_SUBCARRIERS)
        packets_since_evaluation += 1
        if packets_since_evaluation < EVALUATION_INTERVAL:
            continue
        detector.update_state()
        packets_since_evaluation = 0
        if i >= warmup:
            movement_eval_count += 1
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
    policy_metrics = _evaluate_idle_runtime_policy_evaluations(baseline_motion_states)
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
        **policy_metrics,
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


def compute_performance_report_data(
    progress: Optional[ProgressCallback] = None,
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """Compute all metrics published in docs/performance/README.md."""
    paired_results: dict[str, dict[str, list[Dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    long_results: dict[str, dict[str, list[Dict[str, float]]]] = defaultdict(lambda: defaultdict(list))

    paired_datasets = get_available_paired_datasets()
    _emit_progress(progress, f"discovered {len(paired_datasets)} paired validation datasets")
    paired_phase_started = time.perf_counter()
    for index, (static_path, motion_path, _num_sc, chip, dataset_id) in enumerate(paired_datasets, start=1):
        dataset_started = time.perf_counter()
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
        _emit_progress(
            progress,
            (
                f"paired dataset {index}/{len(paired_datasets)} complete: "
                f"{chip} ({dataset_id}) in {_format_progress_duration(time.perf_counter() - dataset_started)}"
            ),
        )

    _emit_progress(
        progress,
        (
            f"paired validation complete: {len(paired_datasets)} datasets in "
            f"{_format_progress_duration(time.perf_counter() - paired_phase_started)}"
        ),
    )

    long_test_datasets = get_available_long_test_datasets()
    _emit_progress(progress, f"discovered {len(long_test_datasets)} long quiet validation datasets")
    long_phase_started = time.perf_counter()
    for index, (test_path, baseline_packets, movement_packets, _motion_start, chip, _entry) in enumerate(
        long_test_datasets,
        start=1,
    ):
        dataset_started = time.perf_counter()
        classic_metrics = evaluate_classic_long_recording(baseline_packets, movement_packets)
        if classic_metrics is not None:
            long_results["classic"][chip].append(classic_metrics)

        ml_metrics = evaluate_ml_long_recording(baseline_packets, movement_packets)
        long_results["ml"][chip].append(ml_metrics)
        _emit_progress(
            progress,
            (
                f"long quiet dataset {index}/{len(long_test_datasets)} complete: "
                f"{chip} ({test_path.stem}) in {_format_progress_duration(time.perf_counter() - dataset_started)}"
            ),
        )

    _emit_progress(
        progress,
        (
            f"long quiet validation complete: {len(long_test_datasets)} datasets in "
            f"{_format_progress_duration(time.perf_counter() - long_phase_started)}"
        ),
    )

    summary_started = time.perf_counter()
    paired_summary: Dict[str, Dict[str, Dict[str, float]]] = {"classic": {}, "ml": {}}
    for algorithm, by_chip in paired_results.items():
        for chip, entries in by_chip.items():
            averaged = _average_detector_metrics(entries)
            if averaged is not None:
                paired_summary[algorithm][chip] = averaged

    long_summary: Dict[str, Dict[str, Dict[str, float]]] = {"classic": {}, "ml": {}}
    for algorithm, by_chip in long_results.items():
        for chip, entries in by_chip.items():
            if not entries:
                continue
            fp_rates = [entry["fp_rate"] for entry in entries]
            long_summary[algorithm][chip] = {
                "avg_fp_rate": sum(fp_rates) / len(fp_rates),
                "max_fp_rate": max(fp_rates),
                "effective_alarms": sum(entry["effective_alarms"] for entry in entries),
                "false_motion_evaluations": sum(
                    entry["false_motion_evaluations"] for entry in entries
                ),
            }

    _emit_progress(
        progress,
        f"summary aggregation complete in {_format_progress_duration(time.perf_counter() - summary_started)}",
    )
    _emit_progress(progress, "render data ready")
    return {
        "paired": paired_summary,
        "long_quiet": long_summary,
    }


def render_performance_report_markdown(
    report_data: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    execution_info: Optional[ExecutionInfo] = None,
) -> str:
    """Render the published performance markdown from computed metrics."""
    lines = [
        "<!-- Generated file. Do not edit manually. -->",
        "",
        "# Performance Metrics",
        "",
    ]

    if execution_info is not None:
        lines.extend([
            f"Last update: {execution_info['last_update']}",
            f"Source: `{execution_info['source']}`",
            f"Generated by: `{execution_info['generated_by']}`",
            f"Run started: `{execution_info['run_started']}`",
            f"Run duration: `{execution_info['run_duration']}`",
            (
                "Inputs: "
                f"`{execution_info['paired_dataset_count']}` paired datasets, "
                f"`{execution_info['long_quiet_dataset_count']}` long quiet datasets"
            ),
            "",
        ])

    lines.extend([
        "This document provides detailed performance metrics for ESPectre's motion detection algorithms.",
        "",
        (
            "Per-chip live firmware reports in this directory are generated by "
            "`tools/benchmark_firmware.py`."
        ),
        "",
        "- **Classic Detector**: Uses L1-Delta as the primary metric, with a gated moving-variance recovery vote.",
        "- **ML Detector**: Uses a pretrained neural network model based on turbulence and spectral features.",
        "",
        "See [ALGORITHMS.md](../ALGORITHMS.md) for the full detector design.",
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
    ])

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
        "Long-recording detector states are sampled at the deploy runtime cadence: "
        f"one evaluation every {EVALUATION_INTERVAL} packets. Effective Alarms and "
        "False Motion Evals then apply "
        f"{MOTION_ON_HITS} consecutive hits to enter MOTION, and {MOTION_OFF_HITS} to leave it. "
        "They count triggered alarms and evaluations spent in a false MOTION state across "
        "all quiet recordings per chip.",
        "",
        "### Classic Detector",
        "",
    ])

    long_quiet = report_data["long_quiet"]
    long_row_specs = (
        ("avg_fp_rate", "Avg FP Rate", lambda value: f"{value:.2f}%"),
        ("max_fp_rate", "Max FP Rate", lambda value: f"{value:.2f}%"),
        ("effective_alarms", "Effective Alarms", lambda value: f"{int(value)}"),
        ("false_motion_evaluations", "False Motion Evals", lambda value: f"{int(value)}"),
    )

    def _append_long_quiet_table(algorithm):
        lines.append("| Metric | C3 | C5 | C6 | S3 |")
        lines.append("|--------|----|----|----|----|")
        for key, label, formatter in long_row_specs:
            values = []
            for chip in CHIP_ORDER:
                metrics = long_quiet[algorithm].get(chip)
                value = metrics.get(key) if metrics is not None else None
                values.append(formatter(value) if value is not None else "N/A")
            lines.append(f"| {label} | " + " | ".join(values) + " |")

    _append_long_quiet_table("classic")

    lines.extend([
        "",
        "### ML Detector",
        "",
    ])
    _append_long_quiet_table("ml")

    return "\n".join(lines) + "\n"


def write_performance_report(
    output_path: Optional[Path] = None,
    report_data: Optional[Dict[str, Dict[str, Dict[str, Dict[str, float]]]]] = None,
    progress: Optional[ProgressCallback] = None,
    execution_info: Optional[ExecutionInfo] = None,
) -> Path:
    """Compute and write docs/performance/README.md."""
    destination = PERFORMANCE_DOC_PATH if output_path is None else Path(output_path)
    _emit_progress(progress, f"ensuring output directory exists: {destination.parent}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _emit_progress(progress, f"writing markdown to {destination}")
    computed_report_data = report_data
    if computed_report_data is None:
        computed_report_data = (
            compute_performance_report_data()
            if progress is None
            else compute_performance_report_data(progress=progress)
        )
    destination.write_text(
        render_performance_report_markdown(
            computed_report_data,
            execution_info=execution_info,
        ),
        encoding="utf-8",
    )
    _emit_progress(progress, f"report written to {destination}")
    return destination
