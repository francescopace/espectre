#!/usr/bin/env python3
"""
Benchmark physically grounded offline motion scores on real CSI datasets.

This script does not modify the production runtime. It evaluates candidate
standalone scores against the existing MVS and ML baselines using:

- explicit static_presence/motion pairs
- empty-room recordings
- long test recordings under data/test
- leave-one-pair-out threshold selection
- leave-one-chip-out threshold selection
- quiet-room stability diagnostics
- gain-scaling and narrowband-spike sensitivity diagnostics

The benchmark keeps the production packet path aligned with the repo defaults:
- fixed default 12-subcarrier set
- AGC-active CV turbulence: std(amplitudes) / mean(amplitudes)
- Hampel filtering on the turbulence stream
- window size from config
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PYTHON_SRC = REPO_ROOT / "src" / "python" / "micro_espectre"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

from tools.lib.csi_io import load_npz_as_packets
from tools.lib.dataset_metadata import DATA_DIR, load_dataset_info

from config import (
    CALIBRATION_BUFFER_SIZE,
    DEFAULT_SUBCARRIERS,
    HAMPEL_THRESHOLD,
    HAMPEL_WINDOW,
    SEG_WINDOW_SIZE,
)
from detector_interface import MotionState
from filters import HampelFilter
from ml_detector import MLDetector, ML_DEFAULT_THRESHOLD
from segmentation import SegmentationContext
from threshold import get_threshold_factor


PAIR_THRESHOLD_GRID_SIZE = 256
GLOBAL_GAIN_SCALES = (0.75, 1.25, 1.50)
SPIKE_PACKET_RATE = 0.01
SPIKE_SUBCARRIER_FACTOR = 2.0
SPIKE_SUBCARRIER_COUNT = 2
QUIET_TEST_DESCRIPTION = "quiet long-run"


@dataclass(frozen=True)
class StreamRecord:
    """One continuous labeled stream evaluated by the benchmark."""

    name: str
    chip: str
    environment: str
    kind: str
    group_id: str
    idle_prefix_len: int
    amplitudes: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True)
class SequenceMetrics:
    """Thresholded metrics for one scored stream."""

    name: str
    chip: str
    kind: str
    threshold: float
    positives: int
    negatives: int
    tp: int
    fp: int
    tn: int
    fn: int
    recall: Optional[float]
    precision: Optional[float]
    fp_rate: Optional[float]
    f1: Optional[float]


@dataclass(frozen=True)
class ThresholdSelection:
    """Selected threshold and training-fold quality summary."""

    threshold: float
    macro_f1: float
    macro_fp_rate: float
    macro_recall: float


@dataclass(frozen=True)
class CandidateConfig:
    """One benchmarked score."""

    name: str
    family: str
    description: str


CANDIDATES: Tuple[CandidateConfig, ...] = (
    CandidateConfig(
        name="amp_mean_var",
        family="gain-sensitive control",
        description="variance over time of mean packet amplitude",
    ),
    CandidateConfig(
        name="turbulence_mean",
        family="scalar turbulence level",
        description="mean filtered turbulence in the window",
    ),
    CandidateConfig(
        name="mvs_var",
        family="temporal spread of turbulence",
        description="variance of filtered turbulence, the production MVS score",
    ),
    CandidateConfig(
        name="turb_iqr_rel",
        family="robust relative turbulence spread",
        description="IQR of filtered turbulence divided by local mean turbulence",
    ),
    CandidateConfig(
        name="turb_mad_rel",
        family="robust relative turbulence spread",
        description="MAD of filtered turbulence divided by local mean turbulence",
    ),
    CandidateConfig(
        name="profile_rms",
        family="normalized profile-shape instability",
        description="RMS dispersion of gain-normalized subcarrier profiles",
    ),
    CandidateConfig(
        name="profile_l1_med",
        family="robust normalized profile-shape instability",
        description="median L1 distance to the median normalized subcarrier profile",
    ),
)


def _extract_motion_start_from_description(description: str) -> Optional[int]:
    """Extract motion start packet index from free-text metadata."""
    if not description:
        return None
    match = re.search(
        r"motion\s+starts\s+at\s+packet(?:\s+index)?(?:\s+n\.)?\s+(\d+)",
        description,
        re.IGNORECASE,
    )
    if match:
        return int(match.group(1))
    return None


def _packet_to_amplitudes(packet: Dict[str, object]) -> np.ndarray:
    """Extract amplitude vector for the shared production subcarriers."""
    csi_data = packet["csi_data"]
    amplitudes = np.empty(len(DEFAULT_SUBCARRIERS), dtype=np.float64)
    for idx, sc_idx in enumerate(DEFAULT_SUBCARRIERS):
        q = float(csi_data[sc_idx * 2])
        i = float(csi_data[sc_idx * 2 + 1])
        amplitudes[idx] = math.sqrt(i * i + q * q)
    return amplitudes


def _load_stream_records() -> List[StreamRecord]:
    """Load all pair, empty, and long-test streams."""
    dataset_info = load_dataset_info()
    streams: List[StreamRecord] = []

    motion_entries = {
        str(entry.get("filename")): entry
        for entry in dataset_info.get("files", {}).get("motion", [])
        if entry.get("filename")
    }

    # Paired static_presence + motion streams.
    for static_entry in dataset_info.get("files", {}).get("static_presence", []):
        static_name = static_entry.get("filename")
        motion_name = static_entry.get("optimal_pair_motion_file")
        if not static_name or not motion_name:
            continue
        motion_entry = motion_entries.get(str(motion_name))
        if motion_entry is None:
            continue
        static_path = DATA_DIR / "static_presence" / str(static_name)
        motion_path = DATA_DIR / "motion" / str(motion_name)
        if not static_path.exists() or not motion_path.exists():
            continue

        static_packets = load_npz_as_packets(static_path)
        motion_packets = load_npz_as_packets(motion_path)
        amplitudes = np.vstack(
            [_packet_to_amplitudes(pkt) for pkt in (static_packets + motion_packets)]
        )
        labels = np.concatenate(
            [
                np.zeros(len(static_packets), dtype=np.int8),
                np.ones(len(motion_packets), dtype=np.int8),
            ]
        )
        names = sorted([str(static_name), str(motion_name)])
        pair_id = f"pair:{names[0]}::{names[1]}"
        streams.append(
            StreamRecord(
                name=pair_id,
                chip=str(static_entry.get("chip", "unknown")).upper(),
                environment=str(static_entry.get("environment", "unknown")),
                kind="pair",
                group_id=pair_id,
                idle_prefix_len=len(static_packets),
                amplitudes=amplitudes,
                labels=labels,
            )
        )

    # Empty-room streams.
    for empty_entry in dataset_info.get("files", {}).get("empty", []):
        empty_name = empty_entry.get("filename")
        if not empty_name:
            continue
        empty_path = DATA_DIR / "empty" / str(empty_name)
        if not empty_path.exists():
            continue
        packets = load_npz_as_packets(empty_path)
        amplitudes = np.vstack([_packet_to_amplitudes(pkt) for pkt in packets])
        labels = np.zeros(len(packets), dtype=np.int8)
        streams.append(
            StreamRecord(
                name=str(empty_name),
                chip=str(empty_entry.get("chip", "unknown")).upper(),
                environment=str(empty_entry.get("environment", "unknown")),
                kind="empty",
                group_id=f"file:{empty_name}",
                idle_prefix_len=len(packets),
                amplitudes=amplitudes,
                labels=labels,
            )
        )

    # Long test recordings from data/test.
    for test_entry in dataset_info.get("files", {}).get("test", []):
        test_name = test_entry.get("filename")
        if not test_name:
            continue
        test_path = DATA_DIR / "test" / str(test_name)
        if not test_path.exists():
            continue
        packets = load_npz_as_packets(test_path)
        motion_start = _extract_motion_start_from_description(
            str(test_entry.get("description", ""))
        )
        if motion_start is None:
            motion_start = len(packets)
        motion_start = max(0, min(int(motion_start), len(packets)))
        amplitudes = np.vstack([_packet_to_amplitudes(pkt) for pkt in packets])
        labels = np.zeros(len(packets), dtype=np.int8)
        if motion_start < len(packets):
            labels[motion_start:] = 1
            kind = "test_mixed"
        else:
            kind = "test_quiet"
        streams.append(
            StreamRecord(
                name=str(test_name),
                chip=str(test_entry.get("chip", "unknown")).upper(),
                environment=str(test_entry.get("environment", "unknown")),
                kind=kind,
                group_id=f"file:{test_name}",
                idle_prefix_len=motion_start,
                amplitudes=amplitudes,
                labels=labels,
            )
        )

    if not streams:
        raise RuntimeError("No benchmark streams found in data/dataset_info.json")
    return streams


def _median_abs_dev(values: np.ndarray) -> float:
    """Median absolute deviation."""
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)))


def _lag1_autocorr(values: np.ndarray) -> float:
    """Lag-1 autocorrelation helper for diagnostics."""
    if len(values) < 3:
        return 0.0
    centered = values - np.mean(values)
    variance = float(np.mean(centered * centered))
    if variance <= 1e-12:
        return 1.0
    return float(np.mean(centered[:-1] * centered[1:]) / variance)


def _apply_hampel(series: np.ndarray) -> np.ndarray:
    """Apply the production Hampel filter sequentially to a scalar series."""
    hampel = HampelFilter(window_size=HAMPEL_WINDOW, threshold=HAMPEL_THRESHOLD)
    filtered = np.empty_like(series, dtype=np.float64)
    for idx, value in enumerate(series):
        filtered[idx] = float(hampel.filter(float(value)))
    return filtered


def _apply_narrowband_spikes(
    amplitudes: np.ndarray,
    *,
    seed_key: str,
    packet_rate: float = SPIKE_PACKET_RATE,
    factor: float = SPIKE_SUBCARRIER_FACTOR,
    subcarrier_count: int = SPIKE_SUBCARRIER_COUNT,
) -> np.ndarray:
    """Inject deterministic narrowband spikes into a copy of the amplitude matrix."""
    rng_seed = abs(hash(seed_key)) % (2**32)
    rng = np.random.default_rng(rng_seed)
    stressed = np.array(amplitudes, dtype=np.float64, copy=True)
    if len(stressed) == 0:
        return stressed

    packet_mask = rng.random(len(stressed)) < float(packet_rate)
    packet_indices = np.nonzero(packet_mask)[0]
    if len(packet_indices) == 0:
        return stressed

    sc_count = stressed.shape[1]
    choose_k = max(1, min(int(subcarrier_count), sc_count))
    for pkt_idx in packet_indices:
        sc_indices = rng.choice(sc_count, size=choose_k, replace=False)
        stressed[pkt_idx, sc_indices] *= float(factor)
    return stressed


def _compute_base_series(amplitudes: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute filtered turbulence and normalized profile series."""
    safe_amplitudes = np.asarray(amplitudes, dtype=np.float64)
    packet_mean = safe_amplitudes.mean(axis=1)
    safe_mean = np.maximum(packet_mean, 1e-6)
    normalized_profile = safe_amplitudes / safe_mean[:, None]
    raw_turbulence = safe_amplitudes.std(axis=1) / safe_mean
    filtered_turbulence = _apply_hampel(raw_turbulence)
    filtered_packet_mean = _apply_hampel(packet_mean)
    return {
        "packet_mean": packet_mean,
        "packet_mean_filtered": filtered_packet_mean,
        "normalized_profile": normalized_profile,
        "raw_turbulence": raw_turbulence,
        "filtered_turbulence": filtered_turbulence,
    }


def _compute_candidate_score(
    candidate_name: str,
    base_series: Dict[str, np.ndarray],
    *,
    window_size: int = SEG_WINDOW_SIZE,
) -> np.ndarray:
    """Compute one candidate score as a trailing-window series."""
    n_packets = len(base_series["filtered_turbulence"])
    scores = np.full(n_packets, np.nan, dtype=np.float64)
    filtered_turbulence = base_series["filtered_turbulence"]
    filtered_packet_mean = base_series["packet_mean_filtered"]
    normalized_profile = base_series["normalized_profile"]

    if n_packets < window_size:
        return scores

    turb_windows = np.lib.stride_tricks.sliding_window_view(
        filtered_turbulence,
        window_shape=window_size,
    )
    mean_windows = np.lib.stride_tricks.sliding_window_view(
        filtered_packet_mean,
        window_shape=window_size,
    )
    profile_windows = np.lib.stride_tricks.sliding_window_view(
        normalized_profile,
        window_shape=window_size,
        axis=0,
    )
    profile_windows = np.moveaxis(profile_windows, -1, 1)

    turb_mean = np.mean(np.abs(turb_windows), axis=1)
    denom = np.maximum(turb_mean, 1e-6)

    if candidate_name == "amp_mean_var":
        window_scores = np.var(mean_windows, axis=1)
    elif candidate_name == "turbulence_mean":
        window_scores = np.mean(turb_windows, axis=1)
    elif candidate_name == "mvs_var":
        window_scores = np.var(turb_windows, axis=1)
    elif candidate_name == "turb_iqr_rel":
        q25 = np.percentile(turb_windows, 25.0, axis=1)
        q75 = np.percentile(turb_windows, 75.0, axis=1)
        window_scores = (q75 - q25) / denom
    elif candidate_name == "turb_mad_rel":
        turb_median = np.median(turb_windows, axis=1)
        abs_dev = np.abs(turb_windows - turb_median[:, None])
        window_scores = np.median(abs_dev, axis=1) / denom
    elif candidate_name == "profile_rms":
        center = np.mean(profile_windows, axis=1)
        diff = profile_windows - center[:, None, :]
        window_scores = np.sqrt(np.mean(diff * diff, axis=(1, 2)))
    elif candidate_name == "profile_l1_med":
        center = np.median(profile_windows, axis=1)
        distances = np.sum(np.abs(profile_windows - center[:, None, :]), axis=2)
        window_scores = np.median(distances, axis=1)
    else:
        raise ValueError(f"Unknown candidate: {candidate_name}")

    scores[window_size - 1 :] = np.asarray(window_scores, dtype=np.float64)
    return scores


def _precompute_candidate_scores(streams: Iterable[StreamRecord]) -> Dict[str, Dict[str, np.ndarray]]:
    """Precompute base candidate score series for all streams."""
    all_scores: Dict[str, Dict[str, np.ndarray]] = {}
    for stream in streams:
        base_series = _compute_base_series(stream.amplitudes)
        stream_scores = {}
        for candidate in CANDIDATES:
            stream_scores[candidate.name] = _compute_candidate_score(candidate.name, base_series)
        all_scores[stream.name] = stream_scores
    return all_scores


def _build_labels_and_mask(stream: StreamRecord) -> Tuple[np.ndarray, np.ndarray]:
    """Build evaluation labels and valid-mask aligned to trailing windows."""
    valid_mask = ~np.isnan(np.full(len(stream.labels), np.nan))
    valid_mask = np.arange(len(stream.labels)) >= (SEG_WINDOW_SIZE - 1)
    return stream.labels.astype(np.int8), valid_mask


def _evaluate_thresholded_stream(
    stream: StreamRecord,
    score_series: np.ndarray,
    threshold: float,
) -> SequenceMetrics:
    """Evaluate one stream at a fixed threshold."""
    labels, valid_mask = _build_labels_and_mask(stream)
    valid_scores = score_series[valid_mask]
    valid_labels = labels[valid_mask]
    predictions = valid_scores > float(threshold)

    positive_mask = valid_labels == 1
    negative_mask = valid_labels == 0
    positives = int(np.sum(positive_mask))
    negatives = int(np.sum(negative_mask))
    tp = int(np.sum(predictions & positive_mask))
    fp = int(np.sum(predictions & negative_mask))
    tn = int(np.sum((~predictions) & negative_mask))
    fn = int(np.sum((~predictions) & positive_mask))

    recall = tp / positives * 100.0 if positives > 0 else None
    precision = tp / (tp + fp) * 100.0 if (tp + fp) > 0 else None
    fp_rate = fp / negatives * 100.0 if negatives > 0 else None
    if positives > 0 and negatives > 0 and (2 * tp + fp + fn) > 0:
        f1 = 2.0 * tp / (2.0 * tp + fp + fn) * 100.0
    elif positives > 0 and (2 * tp + fp + fn) > 0:
        f1 = 2.0 * tp / (2.0 * tp + fp + fn) * 100.0
    else:
        f1 = None

    return SequenceMetrics(
        name=stream.name,
        chip=stream.chip,
        kind=stream.kind,
        threshold=float(threshold),
        positives=positives,
        negatives=negatives,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        recall=recall,
        precision=precision,
        fp_rate=fp_rate,
        f1=f1,
    )


def _macro_average(values: Iterable[Optional[float]]) -> Optional[float]:
    """Mean over available values."""
    usable = [float(value) for value in values if value is not None]
    if not usable:
        return None
    return float(np.mean(usable))


def _aggregate_metrics(metrics: Iterable[SequenceMetrics]) -> Dict[str, Optional[float]]:
    """Aggregate micro counts and macro averages."""
    rows = list(metrics)
    if not rows:
        return {
            "count": 0,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "recall": None,
            "precision": None,
            "fp_rate": None,
            "f1": None,
            "macro_recall": None,
            "macro_precision": None,
            "macro_fp_rate": None,
            "macro_f1": None,
            "worst_f1": None,
            "worst_fp_rate": None,
        }

    tp = int(sum(row.tp for row in rows))
    fp = int(sum(row.fp for row in rows))
    tn = int(sum(row.tn for row in rows))
    fn = int(sum(row.fn for row in rows))

    recall = tp / (tp + fn) * 100.0 if (tp + fn) > 0 else None
    precision = tp / (tp + fp) * 100.0 if (tp + fp) > 0 else None
    fp_rate = fp / (fp + tn) * 100.0 if (fp + tn) > 0 else None
    f1 = 2.0 * tp / (2.0 * tp + fp + fn) * 100.0 if (2 * tp + fp + fn) > 0 else None

    macro_recall = _macro_average(row.recall for row in rows if row.positives > 0)
    macro_precision = _macro_average(row.precision for row in rows if row.positives > 0)
    macro_fp_rate = _macro_average(row.fp_rate for row in rows if row.negatives > 0)
    macro_f1 = _macro_average(row.f1 for row in rows if row.f1 is not None)

    f1_rows = [row.f1 for row in rows if row.f1 is not None]
    fp_rows = [row.fp_rate for row in rows if row.fp_rate is not None]
    worst_f1 = float(min(f1_rows)) if f1_rows else None
    worst_fp_rate = float(max(fp_rows)) if fp_rows else None

    return {
        "count": len(rows),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "recall": recall,
        "precision": precision,
        "fp_rate": fp_rate,
        "f1": f1,
        "macro_recall": macro_recall,
        "macro_precision": macro_precision,
        "macro_fp_rate": macro_fp_rate,
        "macro_f1": macro_f1,
        "worst_f1": worst_f1,
        "worst_fp_rate": worst_fp_rate,
    }


def _candidate_thresholds(score_arrays: Iterable[np.ndarray]) -> np.ndarray:
    """Build a compact threshold grid from training scores."""
    concatenated = np.concatenate([scores[np.isfinite(scores)] for scores in score_arrays])
    if concatenated.size == 0:
        return np.array([0.0], dtype=np.float64)
    quantiles = np.unique(np.quantile(concatenated, np.linspace(0.0, 1.0, PAIR_THRESHOLD_GRID_SIZE + 1)))
    if quantiles.size == 1:
        return quantiles.astype(np.float64)
    return quantiles.astype(np.float64)


def _select_global_threshold(
    candidate_name: str,
    train_pairs: Iterable[StreamRecord],
    score_map: Dict[str, Dict[str, np.ndarray]],
) -> ThresholdSelection:
    """Tune one global threshold on training pair streams only."""
    pair_streams = list(train_pairs)
    thresholds = _candidate_thresholds(
        [score_map[stream.name][candidate_name] for stream in pair_streams]
    )

    best: Optional[Tuple[float, float, float, float]] = None
    for threshold in thresholds:
        fold_metrics = [
            _evaluate_thresholded_stream(stream, score_map[stream.name][candidate_name], float(threshold))
            for stream in pair_streams
        ]
        macro_f1 = _macro_average(row.f1 for row in fold_metrics if row.f1 is not None) or 0.0
        macro_fp_rate = _macro_average(row.fp_rate for row in fold_metrics if row.fp_rate is not None) or 0.0
        macro_recall = _macro_average(row.recall for row in fold_metrics if row.recall is not None) or 0.0
        candidate_tuple = (macro_f1, -macro_fp_rate, macro_recall, float(threshold))
        if best is None or candidate_tuple > best:
            best = candidate_tuple

    assert best is not None
    return ThresholdSelection(
        threshold=best[3],
        macro_f1=best[0],
        macro_fp_rate=-best[1],
        macro_recall=best[2],
    )


def _evaluate_global_threshold_family(
    streams: List[StreamRecord],
    score_map: Dict[str, Dict[str, np.ndarray]],
) -> Dict[str, Dict[str, object]]:
    """Evaluate all candidate scores under leave-one-pair-out and leave-one-chip-out."""
    pair_streams = [stream for stream in streams if stream.kind == "pair"]
    empty_streams = [stream for stream in streams if stream.kind == "empty"]
    test_streams = [stream for stream in streams if stream.kind in ("test_quiet", "test_mixed")]
    quiet_test_streams = [stream for stream in streams if stream.kind == "test_quiet"]
    mixed_test_streams = [stream for stream in streams if stream.kind == "test_mixed"]
    chips = sorted({stream.chip for stream in pair_streams})

    summary: Dict[str, Dict[str, object]] = {}
    for candidate in CANDIDATES:
        candidate_name = candidate.name
        pair_fold_metrics: List[SequenceMetrics] = []
        pair_fold_thresholds: Dict[str, float] = {}
        for held_out in pair_streams:
            train_pairs = [stream for stream in pair_streams if stream.group_id != held_out.group_id]
            selection = _select_global_threshold(candidate_name, train_pairs, score_map)
            pair_fold_thresholds[held_out.name] = selection.threshold
            pair_fold_metrics.append(
                _evaluate_thresholded_stream(
                    held_out,
                    score_map[held_out.name][candidate_name],
                    selection.threshold,
                )
            )

        chip_fold_rows = {}
        for chip in chips:
            train_pairs = [stream for stream in pair_streams if stream.chip != chip]
            test_pairs = [stream for stream in pair_streams if stream.chip == chip]
            selection = _select_global_threshold(candidate_name, train_pairs, score_map)
            chip_metrics = {
                "threshold": selection.threshold,
                "pairs": _aggregate_metrics(
                    _evaluate_thresholded_stream(
                        stream,
                        score_map[stream.name][candidate_name],
                        selection.threshold,
                    )
                    for stream in test_pairs
                ),
                "empty": _aggregate_metrics(
                    _evaluate_thresholded_stream(
                        stream,
                        score_map[stream.name][candidate_name],
                        selection.threshold,
                    )
                    for stream in empty_streams
                    if stream.chip == chip
                ),
                "test_quiet": _aggregate_metrics(
                    _evaluate_thresholded_stream(
                        stream,
                        score_map[stream.name][candidate_name],
                        selection.threshold,
                    )
                    for stream in quiet_test_streams
                    if stream.chip == chip
                ),
                "test_mixed": _aggregate_metrics(
                    _evaluate_thresholded_stream(
                        stream,
                        score_map[stream.name][candidate_name],
                        selection.threshold,
                    )
                    for stream in mixed_test_streams
                    if stream.chip == chip
                ),
            }
            chip_fold_rows[chip] = chip_metrics

        summary[candidate_name] = {
            "family": candidate.family,
            "description": candidate.description,
            "leave_one_pair_out": _aggregate_metrics(pair_fold_metrics),
            "pair_thresholds": pair_fold_thresholds,
            "leave_one_chip_out": {
                "per_chip": chip_fold_rows,
                "pairs_macro_f1_mean": _macro_average(
                    chip_fold_rows[chip]["pairs"]["macro_f1"] for chip in chips
                ),
                "pairs_macro_fp_rate_mean": _macro_average(
                    chip_fold_rows[chip]["pairs"]["macro_fp_rate"] for chip in chips
                ),
                "pairs_worst_chip_f1": min(
                    (
                        chip_fold_rows[chip]["pairs"]["macro_f1"]
                        for chip in chips
                        if chip_fold_rows[chip]["pairs"]["macro_f1"] is not None
                    ),
                    default=None,
                ),
                "empty_fp_rate_mean": _macro_average(
                    chip_fold_rows[chip]["empty"]["fp_rate"] for chip in chips
                ),
                "empty_worst_fp_rate": max(
                    (
                        chip_fold_rows[chip]["empty"]["fp_rate"]
                        for chip in chips
                        if chip_fold_rows[chip]["empty"]["fp_rate"] is not None
                    ),
                    default=None,
                ),
                "test_quiet_fp_rate_mean": _macro_average(
                    chip_fold_rows[chip]["test_quiet"]["fp_rate"] for chip in chips
                ),
                "test_quiet_worst_fp_rate": max(
                    (
                        chip_fold_rows[chip]["test_quiet"]["fp_rate"]
                        for chip in chips
                        if chip_fold_rows[chip]["test_quiet"]["fp_rate"] is not None
                    ),
                    default=None,
                ),
                "test_mixed_macro_f1_mean": _macro_average(
                    chip_fold_rows[chip]["test_mixed"]["macro_f1"] for chip in chips
                ),
            },
        }
    return summary


def _estimate_mvs_threshold_from_idle_prefix(amplitudes: np.ndarray, idle_prefix_len: int) -> float:
    """Replay the production startup threshold bootstrap on an idle prefix."""
    calibration_len = min(int(idle_prefix_len), int(CALIBRATION_BUFFER_SIZE))
    calibration_len = max(0, calibration_len)

    ctx = SegmentationContext(window_size=SEG_WINDOW_SIZE, threshold=1.0, enable_hampel=True)
    max_moving_variance: Optional[float] = None
    for idx in range(calibration_len):
        packet_amplitudes = amplitudes[idx]
        mean_amp = float(np.mean(packet_amplitudes))
        turbulence = float(np.std(packet_amplitudes) / max(mean_amp, 1e-6))
        ctx.add_turbulence(turbulence)
        ctx.update_state()
        if ctx.buffer_count >= ctx.window_size:
            current_mv = float(ctx.current_moving_variance)
            if max_moving_variance is None or current_mv > max_moving_variance:
                max_moving_variance = current_mv

    if max_moving_variance is None:
        return 1.0
    return max(max_moving_variance * get_threshold_factor("auto"), 1e-6)


def _evaluate_mvs_session_baseline(stream: StreamRecord) -> Dict[str, object]:
    """Evaluate production-style per-stream MVS."""
    threshold = _estimate_mvs_threshold_from_idle_prefix(stream.amplitudes, stream.idle_prefix_len)
    ctx = SegmentationContext(window_size=SEG_WINDOW_SIZE, threshold=threshold, enable_hampel=True)
    score_series = np.full(len(stream.labels), np.nan, dtype=np.float64)
    state_series = np.zeros(len(stream.labels), dtype=np.int8)
    for idx, amplitudes in enumerate(stream.amplitudes):
        mean_amp = float(np.mean(amplitudes))
        turbulence = float(np.std(amplitudes) / max(mean_amp, 1e-6))
        ctx.add_turbulence(turbulence)
        ctx.update_state()
        score_series[idx] = float(ctx.current_moving_variance) if idx >= (SEG_WINDOW_SIZE - 1) else np.nan
        state_series[idx] = 1 if ctx.get_state() == MotionState.MOTION else 0

    labels, valid_mask = _build_labels_and_mask(stream)
    valid_labels = labels[valid_mask]
    predictions = state_series[valid_mask] == 1
    positive_mask = valid_labels == 1
    negative_mask = valid_labels == 0
    metrics = SequenceMetrics(
        name=stream.name,
        chip=stream.chip,
        kind=stream.kind,
        threshold=threshold,
        positives=int(np.sum(positive_mask)),
        negatives=int(np.sum(negative_mask)),
        tp=int(np.sum(predictions & positive_mask)),
        fp=int(np.sum(predictions & negative_mask)),
        tn=int(np.sum((~predictions) & negative_mask)),
        fn=int(np.sum((~predictions) & positive_mask)),
        recall=(float(np.sum(predictions & positive_mask)) / int(np.sum(positive_mask)) * 100.0)
        if np.any(positive_mask)
        else None,
        precision=(
            float(np.sum(predictions & positive_mask))
            / max(int(np.sum(predictions)), 1)
            * 100.0
        )
        if np.any(predictions)
        else None,
        fp_rate=(float(np.sum(predictions & negative_mask)) / int(np.sum(negative_mask)) * 100.0)
        if np.any(negative_mask)
        else None,
        f1=(
            2.0 * float(np.sum(predictions & positive_mask))
            / (
                2.0 * float(np.sum(predictions & positive_mask))
                + float(np.sum(predictions & negative_mask))
                + float(np.sum((~predictions) & positive_mask))
            )
            * 100.0
        )
        if np.any(positive_mask)
        and (
            2.0 * float(np.sum(predictions & positive_mask))
            + float(np.sum(predictions & negative_mask))
            + float(np.sum((~predictions) & positive_mask))
        )
        > 0.0
        else None,
    )
    return {
        "metrics": metrics,
        "threshold": threshold,
        "score_series": score_series,
    }


def _evaluate_ml_fixed_baseline(stream: StreamRecord, packets_map: Dict[str, List[Dict[str, object]]]) -> Dict[str, object]:
    """Evaluate the exported production ML detector on one stream."""
    detector = MLDetector(threshold=ML_DEFAULT_THRESHOLD, window_size=SEG_WINDOW_SIZE)
    score_series = np.full(len(stream.labels), np.nan, dtype=np.float64)
    state_series = np.zeros(len(stream.labels), dtype=np.int8)
    for idx, packet in enumerate(packets_map[stream.name]):
        detector.process_packet(packet["csi_data"], DEFAULT_SUBCARRIERS)
        metrics = detector.update_state()
        if idx >= (SEG_WINDOW_SIZE - 1):
            score_series[idx] = float(metrics.get("probability", 0.0))
        state_series[idx] = 1 if detector.get_state() == MotionState.MOTION else 0

    labels, valid_mask = _build_labels_and_mask(stream)
    valid_labels = labels[valid_mask]
    predictions = state_series[valid_mask] == 1
    positive_mask = valid_labels == 1
    negative_mask = valid_labels == 0
    metrics = SequenceMetrics(
        name=stream.name,
        chip=stream.chip,
        kind=stream.kind,
        threshold=float(ML_DEFAULT_THRESHOLD),
        positives=int(np.sum(positive_mask)),
        negatives=int(np.sum(negative_mask)),
        tp=int(np.sum(predictions & positive_mask)),
        fp=int(np.sum(predictions & negative_mask)),
        tn=int(np.sum((~predictions) & negative_mask)),
        fn=int(np.sum((~predictions) & positive_mask)),
        recall=(float(np.sum(predictions & positive_mask)) / int(np.sum(positive_mask)) * 100.0)
        if np.any(positive_mask)
        else None,
        precision=(
            float(np.sum(predictions & positive_mask))
            / max(int(np.sum(predictions)), 1)
            * 100.0
        )
        if np.any(predictions)
        else None,
        fp_rate=(float(np.sum(predictions & negative_mask)) / int(np.sum(negative_mask)) * 100.0)
        if np.any(negative_mask)
        else None,
        f1=(
            2.0 * float(np.sum(predictions & positive_mask))
            / (
                2.0 * float(np.sum(predictions & positive_mask))
                + float(np.sum(predictions & negative_mask))
                + float(np.sum((~predictions) & positive_mask))
            )
            * 100.0
        )
        if np.any(positive_mask)
        and (
            2.0 * float(np.sum(predictions & positive_mask))
            + float(np.sum(predictions & negative_mask))
            + float(np.sum((~predictions) & positive_mask))
        )
        > 0.0
        else None,
    )
    return {
        "metrics": metrics,
        "score_series": score_series,
    }


def _load_packet_map(streams: Iterable[StreamRecord]) -> Dict[str, List[Dict[str, object]]]:
    """Reload packet dictionaries by stream name for the ML baseline."""
    dataset_info = load_dataset_info()
    packet_map: Dict[str, List[Dict[str, object]]] = {}

    motion_entries = {
        str(entry.get("filename")): entry
        for entry in dataset_info.get("files", {}).get("motion", [])
        if entry.get("filename")
    }

    for static_entry in dataset_info.get("files", {}).get("static_presence", []):
        static_name = static_entry.get("filename")
        motion_name = static_entry.get("optimal_pair_motion_file")
        if not static_name or not motion_name:
            continue
        motion_entry = motion_entries.get(str(motion_name))
        if motion_entry is None:
            continue
        static_path = DATA_DIR / "static_presence" / str(static_name)
        motion_path = DATA_DIR / "motion" / str(motion_name)
        if not static_path.exists() or not motion_path.exists():
            continue
        names = sorted([str(static_name), str(motion_name)])
        pair_id = f"pair:{names[0]}::{names[1]}"
        packet_map[pair_id] = load_npz_as_packets(static_path) + load_npz_as_packets(motion_path)

    for label in ("empty", "test"):
        for entry in dataset_info.get("files", {}).get(label, []):
            name = entry.get("filename")
            if not name:
                continue
            path = DATA_DIR / label / str(name)
            if path.exists():
                packet_map[str(name)] = load_npz_as_packets(path)

    return packet_map


def _aggregate_reference_baselines(
    streams: List[StreamRecord],
    packets_map: Dict[str, List[Dict[str, object]]],
) -> Dict[str, Dict[str, object]]:
    """Evaluate the current production MVS and ML baselines."""
    chips = sorted({stream.chip for stream in streams if stream.kind == "pair"})
    pair_streams = [stream for stream in streams if stream.kind == "pair"]
    empty_streams = [stream for stream in streams if stream.kind == "empty"]
    quiet_streams = [stream for stream in streams if stream.kind == "test_quiet"]
    mixed_streams = [stream for stream in streams if stream.kind == "test_mixed"]

    # MVS session-calibrated baseline.
    mvs_rows = {stream.name: _evaluate_mvs_session_baseline(stream) for stream in streams}
    ml_rows = {stream.name: _evaluate_ml_fixed_baseline(stream, packets_map) for stream in streams}

    def summarize(rows: Dict[str, Dict[str, object]]) -> Dict[str, object]:
        per_chip = {}
        for chip in chips:
            per_chip[chip] = {
                "pairs": _aggregate_metrics(
                    rows[stream.name]["metrics"] for stream in pair_streams if stream.chip == chip
                ),
                "empty": _aggregate_metrics(
                    rows[stream.name]["metrics"] for stream in empty_streams if stream.chip == chip
                ),
                "test_quiet": _aggregate_metrics(
                    rows[stream.name]["metrics"] for stream in quiet_streams if stream.chip == chip
                ),
                "test_mixed": _aggregate_metrics(
                    rows[stream.name]["metrics"] for stream in mixed_streams if stream.chip == chip
                ),
            }

        return {
            "pairs": _aggregate_metrics(rows[stream.name]["metrics"] for stream in pair_streams),
            "empty": _aggregate_metrics(rows[stream.name]["metrics"] for stream in empty_streams),
            "test_quiet": _aggregate_metrics(rows[stream.name]["metrics"] for stream in quiet_streams),
            "test_mixed": _aggregate_metrics(rows[stream.name]["metrics"] for stream in mixed_streams),
            "per_chip": per_chip,
        }

    return {
        "mvs_session": summarize(mvs_rows),
        "ml_fixed": summarize(ml_rows),
    }


def _score_gap(candidate_name: str, streams: List[StreamRecord], score_map: Dict[str, Dict[str, np.ndarray]]) -> float:
    """Median motion-idle score gap on paired streams."""
    idle_values = []
    motion_values = []
    for stream in streams:
        if stream.kind != "pair":
            continue
        scores = score_map[stream.name][candidate_name]
        labels, valid_mask = _build_labels_and_mask(stream)
        valid_scores = scores[valid_mask]
        valid_labels = labels[valid_mask]
        idle_values.append(valid_scores[valid_labels == 0])
        motion_values.append(valid_scores[valid_labels == 1])

    idle_concat = np.concatenate(idle_values)
    motion_concat = np.concatenate(motion_values)
    gap = float(np.median(motion_concat) - np.median(idle_concat))
    return gap if abs(gap) > 1e-9 else 1e-9


def _compute_candidate_diagnostics(
    streams: List[StreamRecord],
    score_map: Dict[str, Dict[str, np.ndarray]],
) -> Dict[str, Dict[str, float]]:
    """Compute threshold-free stability, gain, and spike diagnostics."""
    quiet_streams = [stream for stream in streams if stream.kind in ("empty", "test_quiet")]
    diagnostics: Dict[str, Dict[str, float]] = {}

    quiet_variant_scores: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    for stream in quiet_streams:
        variants = {
            "base": stream.amplitudes,
            "spike": _apply_narrowband_spikes(stream.amplitudes, seed_key=stream.name),
        }
        for gain_scale in GLOBAL_GAIN_SCALES:
            variants[f"gain:{gain_scale:.2f}"] = stream.amplitudes * float(gain_scale)

        stream_variant_scores: Dict[str, Dict[str, np.ndarray]] = {}
        for variant_name, variant_amplitudes in variants.items():
            base_series = _compute_base_series(variant_amplitudes)
            stream_variant_scores[variant_name] = {
                candidate.name: _compute_candidate_score(candidate.name, base_series)
                for candidate in CANDIDATES
            }
        quiet_variant_scores[stream.name] = stream_variant_scores

    for candidate in CANDIDATES:
        candidate_name = candidate.name
        gap = _score_gap(candidate_name, streams, score_map)

        jitter_ratios = []
        drift_ratios = []
        gain_ratios = []
        spike_ratios = []

        for stream in quiet_streams:
            base_scores = quiet_variant_scores[stream.name]["base"][candidate_name]
            finite_scores = base_scores[np.isfinite(base_scores)]
            if finite_scores.size < 8:
                continue

            diffs = np.diff(finite_scores)
            jitter = _median_abs_dev(diffs) if diffs.size > 0 else 0.0
            quarter = max(1, finite_scores.size // 4)
            drift = abs(float(np.median(finite_scores[-quarter:]) - np.median(finite_scores[:quarter])))
            jitter_ratios.append(jitter / abs(gap))
            drift_ratios.append(drift / abs(gap))

            # Gain scaling.
            baseline_values = finite_scores
            for gain_scale in GLOBAL_GAIN_SCALES:
                scaled_scores = quiet_variant_scores[stream.name][f"gain:{gain_scale:.2f}"][candidate_name]
                scaled_values = scaled_scores[np.isfinite(scaled_scores)]
                common_len = min(len(baseline_values), len(scaled_values))
                if common_len > 0:
                    gain_delta = np.median(
                        np.abs(scaled_values[:common_len] - baseline_values[:common_len])
                    )
                    gain_ratios.append(float(gain_delta) / abs(gap))

            # Narrowband spikes.
            spiked_scores = quiet_variant_scores[stream.name]["spike"][candidate_name]
            spiked_values = spiked_scores[np.isfinite(spiked_scores)]
            common_len = min(len(finite_scores), len(spiked_values))
            if common_len > 0:
                spike_delta = np.median(
                    np.maximum(spiked_values[:common_len] - finite_scores[:common_len], 0.0)
                )
                spike_ratios.append(float(spike_delta) / abs(gap))

        diagnostics[candidate_name] = {
            "motion_idle_gap": gap,
            "quiet_jitter_ratio": float(np.median(jitter_ratios)) if jitter_ratios else float("nan"),
            "quiet_drift_ratio": float(np.median(drift_ratios)) if drift_ratios else float("nan"),
            "gain_sensitivity_ratio": float(np.median(gain_ratios)) if gain_ratios else float("nan"),
            "spike_sensitivity_ratio": float(np.median(spike_ratios)) if spike_ratios else float("nan"),
        }
    return diagnostics


def _print_reference_summary(reference: Dict[str, Dict[str, object]]) -> None:
    """Print production baseline metrics."""
    print("\n" + "=" * 100)
    print("CURRENT PRODUCTION BASELINES")
    print("=" * 100)
    for baseline_name, summary in reference.items():
        pairs = summary["pairs"]
        empty = summary["empty"]
        quiet = summary["test_quiet"]
        mixed = summary["test_mixed"]
        print(f"\n{baseline_name}")
        print(
            f"  paired real data:    recall={pairs['recall']:.1f}% precision={pairs['precision']:.1f}% "
            f"fp_rate={pairs['fp_rate']:.1f}% f1={pairs['f1']:.1f}%"
        )
        print(
            f"  empty-room FP:       mean={empty['macro_fp_rate']:.1f}% worst={empty['worst_fp_rate']:.1f}%"
        )
        print(
            f"  quiet long-run FP:   mean={quiet['macro_fp_rate']:.1f}% worst={quiet['worst_fp_rate']:.1f}%"
        )
        if mixed["macro_f1"] is not None:
            print(
                f"  mixed long tests:    recall={mixed['macro_recall']:.1f}% precision={mixed['macro_precision']:.1f}% "
                f"fp_rate={mixed['macro_fp_rate']:.1f}% f1={mixed['macro_f1']:.1f}%"
            )


def _print_candidate_summary(
    candidate_summary: Dict[str, Dict[str, object]],
    diagnostics: Dict[str, Dict[str, float]],
) -> None:
    """Print a compact candidate ranking view."""
    print("\n" + "=" * 100)
    print("GLOBAL-THRESHOLD STANDALONE CANDIDATES")
    print("=" * 100)

    ranking = sorted(
        candidate_summary.items(),
        key=lambda item: (
            item[1]["leave_one_chip_out"]["pairs_macro_f1_mean"] or -1.0,
            -(item[1]["leave_one_chip_out"]["empty_fp_rate_mean"] or 1e9),
        ),
        reverse=True,
    )

    header = (
        f"{'candidate':<18} {'family':<34} {'LOCO F1':>8} {'worst chip F1':>13} "
        f"{'empty FP':>10} {'quiet FP':>10} {'LODO F1':>9} {'gain':>8} {'spike':>8} {'drift':>8}"
    )
    print(header)
    print("-" * len(header))
    for candidate_name, summary in ranking:
        loco = summary["leave_one_chip_out"]
        lodo = summary["leave_one_pair_out"]
        diag = diagnostics[candidate_name]
        print(
            f"{candidate_name:<18} "
            f"{summary['family']:<34} "
            f"{(loco['pairs_macro_f1_mean'] or float('nan')):>7.1f}% "
            f"{(loco['pairs_worst_chip_f1'] or float('nan')):>12.1f}% "
            f"{(loco['empty_fp_rate_mean'] or float('nan')):>9.1f}% "
            f"{(loco['test_quiet_fp_rate_mean'] or float('nan')):>9.1f}% "
            f"{(lodo['macro_f1'] or float('nan')):>8.1f}% "
            f"{diag['gain_sensitivity_ratio']:>8.3f} "
            f"{diag['spike_sensitivity_ratio']:>8.3f} "
            f"{diag['quiet_drift_ratio']:>8.3f}"
        )

    print("\nPer-candidate notes:")
    for candidate_name, summary in ranking:
        diag = diagnostics[candidate_name]
        loco = summary["leave_one_chip_out"]
        print(
            f"  - {candidate_name}: {summary['description']} | "
            f"LOCO pair F1={loco['pairs_macro_f1_mean']:.1f}% | "
            f"empty FP={loco['empty_fp_rate_mean']:.1f}% | "
            f"quiet FP={loco['test_quiet_fp_rate_mean']:.1f}% | "
            f"gain={diag['gain_sensitivity_ratio']:.3f} | "
            f"spike={diag['spike_sensitivity_ratio']:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark offline CSI motion scores")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path for a JSON report",
    )
    args = parser.parse_args()

    streams = _load_stream_records()
    packet_map = _load_packet_map(streams)
    score_map = _precompute_candidate_scores(streams)
    diagnostics = _compute_candidate_diagnostics(streams, score_map)
    candidate_summary = _evaluate_global_threshold_family(streams, score_map)
    reference = _aggregate_reference_baselines(streams, packet_map)

    _print_reference_summary(reference)
    _print_candidate_summary(candidate_summary, diagnostics)

    result = {
        "config": {
            "window_size": SEG_WINDOW_SIZE,
            "calibration_buffer_size": CALIBRATION_BUFFER_SIZE,
            "default_subcarriers": list(DEFAULT_SUBCARRIERS),
            "hampel_window": HAMPEL_WINDOW,
            "hampel_threshold": HAMPEL_THRESHOLD,
            "gain_scales": list(GLOBAL_GAIN_SCALES),
            "spike_packet_rate": SPIKE_PACKET_RATE,
            "spike_subcarrier_factor": SPIKE_SUBCARRIER_FACTOR,
            "spike_subcarrier_count": SPIKE_SUBCARRIER_COUNT,
        },
        "stream_counts": {
            "pairs": sum(1 for stream in streams if stream.kind == "pair"),
            "empty": sum(1 for stream in streams if stream.kind == "empty"),
            "test_quiet": sum(1 for stream in streams if stream.kind == "test_quiet"),
            "test_mixed": sum(1 for stream in streams if stream.kind == "test_mixed"),
        },
        "reference": reference,
        "candidates": candidate_summary,
        "diagnostics": diagnostics,
    }

    if args.json_output is not None:
        args.json_output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nSaved JSON report to {args.json_output}")


if __name__ == "__main__":
    main()
