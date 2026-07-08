#!/usr/bin/env python3
"""
Shared helpers for the historical moving-variance baseline over paired datasets.

This is the single public module for host-side variance-baseline tooling. It
mirrors the paired Python/C++ validation path:
- explicit static_presence -> motion pairs from dataset_info.json
- startup calibration from packet 0 using CALIBRATION_BUFFER_SIZE
- continuous baseline -> motion evaluation on a single warm context
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.lib.csi_io import load_npz_as_packets
from tools.lib.dataset_metadata import load_dataset_info
from config import (
    CALIBRATION_BUFFER_SIZE,
    DEFAULT_SUBCARRIERS,
    ENABLE_HAMPEL_FILTER,
    ENABLE_LOWPASS_FILTER,
    HAMPEL_THRESHOLD,
    HAMPEL_WINDOW,
    LOWPASS_CUTOFF,
    SEG_WINDOW_SIZE,
)
from tools.lib.repo_paths import data_dir
from segmentation import SegmentationContext
from threshold import get_threshold_factor


DATA_DIR = data_dir()


@dataclass(frozen=True)
class PairedDataset:
    chip: str
    environment: str
    dataset_id: str
    static_presence_path: Path
    motion_path: Path
    num_subcarriers: int = 64


@dataclass(frozen=True)
class VarianceFilterConfig:
    enable_hampel: bool = ENABLE_HAMPEL_FILTER
    enable_lowpass: bool = ENABLE_LOWPASS_FILTER
    hampel_window: int = HAMPEL_WINDOW
    hampel_threshold: float = HAMPEL_THRESHOLD
    lowpass_cutoff: float = LOWPASS_CUTOFF


@dataclass(frozen=True)
class BaselineTrackingConfig:
    idle_percentile: float = 99.0
    factor: float = 1.10
    idle_history_size: int = 512
    min_idle_samples: int = 24
    margin_ratio: float = 0.98
    transition_guard_packets: int = SEG_WINDOW_SIZE // 2


@dataclass(frozen=True)
class SubcarrierEMANormConfig:
    alpha: float = 0.01
    margin_ratio: float = 0.75
    transition_guard_packets: int = SEG_WINDOW_SIZE
    warmup_packets: int = SEG_WINDOW_SIZE


@dataclass(frozen=True)
class VarianceVariantConfig:
    name: str
    baseline_tracking: Optional[BaselineTrackingConfig] = None
    subcarrier_ema_norm: Optional[SubcarrierEMANormConfig] = None


@dataclass
class PacketTrace:
    moving_variance: float
    threshold: float
    motion: bool


@dataclass
class VarianceEvaluationResult:
    dataset: PairedDataset
    variant_name: str
    filter_config: VarianceFilterConfig
    startup_threshold: float
    final_threshold: float
    threshold_source: str
    tp: int
    fn: int
    fp: int
    tn: int
    recall: float
    precision: float
    fp_rate: float
    f1: float
    baseline_count: int
    motion_count: int
    threshold_update_count: int = 0
    ema_update_count: int = 0
    idle_reference_count: int = 0
    tracking_gate_hit_count: int = 0
    tracking_state_block_count: int = 0
    tracking_transition_block_count: int = 0
    tracking_margin_block_count: int = 0
    max_idle_reference_mv: float = 0.0
    max_candidate_threshold: float = 0.0
    selected_band: tuple[int, ...] = DEFAULT_SUBCARRIERS
    baseline_trace: Optional[list[PacketTrace]] = None
    motion_trace: Optional[list[PacketTrace]] = None


def iter_paired_datasets(
    *,
    chip: Optional[str] = None,
    dataset_id: Optional[str] = None,
    num_subcarriers: int = 64,
    limit: Optional[int] = None,
) -> list[PairedDataset]:
    """Return explicit static_presence/motion pairs from dataset_info.json."""
    info = load_dataset_info()
    files = info.get("files", {})
    selected_chip = chip.upper() if chip else None
    selected_dataset_key = dataset_id.strip().lower() if dataset_id else None

    motion_by_filename = {
        str(entry.get("filename")): entry
        for entry in files.get("motion", [])
        if entry.get("filename")
    }

    pairs: list[PairedDataset] = []
    for static_entry in files.get("static_presence", []):
        if int(static_entry.get("subcarriers", 0) or 0) != int(num_subcarriers):
            continue

        static_chip = str(static_entry.get("chip", "")).upper()
        if selected_chip and static_chip != selected_chip:
            continue

        motion_name = static_entry.get("optimal_pair_motion_file")
        motion_entry = motion_by_filename.get(str(motion_name))
        if motion_entry is None:
            continue
        if int(motion_entry.get("subcarriers", 0) or 0) != int(num_subcarriers):
            continue

        static_name = static_entry.get("filename")
        if not static_name or not motion_name:
            continue

        static_path = DATA_DIR / "static_presence" / str(static_name)
        motion_path = DATA_DIR / "motion" / str(motion_name)
        if not static_path.exists() or not motion_path.exists():
            continue

        environment = str(static_entry.get("environment") or "unknown")
        current_dataset_id = f"{static_chip.lower()}_{environment}_{static_path.stem}"
        if selected_dataset_key and selected_dataset_key not in {
            current_dataset_id.lower(),
            static_path.name.lower(),
            static_path.stem.lower(),
            str(motion_name).lower(),
            Path(str(motion_name)).stem.lower(),
        }:
            continue

        pairs.append(
            PairedDataset(
                chip=static_chip,
                environment=environment,
                dataset_id=current_dataset_id,
                static_presence_path=static_path,
                motion_path=motion_path,
                num_subcarriers=int(num_subcarriers),
            )
        )

    pairs.sort(key=lambda item: (item.chip, item.environment, item.static_presence_path.name))
    if limit is not None:
        return pairs[: max(0, int(limit))]
    return pairs


def load_paired_packets(pair: PairedDataset) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load packet lists for a paired dataset."""
    return load_npz_as_packets(pair.static_presence_path), load_npz_as_packets(pair.motion_path)


def build_segmentation_context(
    *,
    threshold: float,
    window_size: int = SEG_WINDOW_SIZE,
    filter_config: Optional[VarianceFilterConfig] = None,
) -> SegmentationContext:
    """Create one production-style segmentation context."""
    cfg = filter_config or VarianceFilterConfig()
    return SegmentationContext(
        window_size=window_size,
        threshold=threshold,
        enable_hampel=cfg.enable_hampel,
        hampel_window=cfg.hampel_window,
        hampel_threshold=cfg.hampel_threshold,
        enable_lowpass=cfg.enable_lowpass,
        lowpass_cutoff=cfg.lowpass_cutoff,
    )


def calibrate_startup_threshold(
    static_presence_packets: list[dict[str, Any]],
    *,
    selected_band: tuple[int, ...] = DEFAULT_SUBCARRIERS,
    window_size: int = SEG_WINDOW_SIZE,
    filter_config: Optional[VarianceFilterConfig] = None,
) -> tuple[float, Optional[float]]:
    """Mirror production startup calibration from packet 0."""
    ctx = build_segmentation_context(threshold=1.0, window_size=window_size, filter_config=filter_config)
    max_moving_variance: Optional[float] = None
    calibration_packets = min(CALIBRATION_BUFFER_SIZE, len(static_presence_packets))

    for pkt in static_presence_packets[:calibration_packets]:
        turbulence = ctx.calculate_spatial_turbulence(pkt["csi_data"], selected_band)
        ctx.add_turbulence(turbulence)
        ctx.update_state()
        if ctx.buffer_count >= ctx.window_size:
            current_moving_variance = float(ctx.current_moving_variance)
            if max_moving_variance is None or current_moving_variance > max_moving_variance:
                max_moving_variance = current_moving_variance

    if max_moving_variance is None:
        return 1.0, None

    threshold = max_moving_variance * get_threshold_factor("auto")
    return max(float(threshold), 1e-6), max_moving_variance


def production_variant() -> VarianceVariantConfig:
    return VarianceVariantConfig(name="baseline")


def baseline_tracking_variant(
    *,
    idle_percentile: float = 99.0,
    factor: float = 1.10,
    idle_history_size: int = 512,
    min_idle_samples: int = 24,
    margin_ratio: float = 0.98,
    transition_guard_packets: int = SEG_WINDOW_SIZE // 2,
) -> VarianceVariantConfig:
    return VarianceVariantConfig(
        name="baseline_tracking",
        baseline_tracking=BaselineTrackingConfig(
            idle_percentile=idle_percentile,
            factor=factor,
            idle_history_size=idle_history_size,
            min_idle_samples=min_idle_samples,
            margin_ratio=margin_ratio,
            transition_guard_packets=transition_guard_packets,
        ),
    )


def subcarrier_ema_norm_variant(
    *,
    alpha: float = 0.01,
    margin_ratio: float = 0.75,
    transition_guard_packets: int = SEG_WINDOW_SIZE,
    warmup_packets: int = SEG_WINDOW_SIZE,
) -> VarianceVariantConfig:
    return VarianceVariantConfig(
        name="subcarrier_ema_norm",
        subcarrier_ema_norm=SubcarrierEMANormConfig(
            alpha=alpha,
            margin_ratio=margin_ratio,
            transition_guard_packets=transition_guard_packets,
            warmup_packets=warmup_packets,
        )
    )


def coefficient_of_variation(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean_value = float(np.mean(values))
    if mean_value <= 0.0:
        return 0.0
    return float(np.std(values) / mean_value)


def extract_subcarrier_amplitudes(csi_data: Any, selected_band: tuple[int, ...]) -> list[float]:
    """Return amplitudes for the selected subcarriers."""
    _turbulence, amplitudes = SegmentationContext.compute_spatial_turbulence(csi_data, selected_band)
    return [float(value) for value in amplitudes]


def _seed_ema_baseline(
    packets: list[dict[str, Any]],
    selected_band: tuple[int, ...],
    config: SubcarrierEMANormConfig,
) -> Optional[list[float]]:
    seed_vectors: list[list[float]] = []
    limit = min(len(packets), max(config.warmup_packets, 1))
    for pkt in packets[:limit]:
        amplitudes = extract_subcarrier_amplitudes(pkt["csi_data"], selected_band)
        if len(amplitudes) == len(selected_band):
            seed_vectors.append(amplitudes)
    if not seed_vectors:
        return None
    return list(np.mean(np.array(seed_vectors, dtype=float), axis=0))


def _ema_normalized_turbulence(
    amplitudes: list[float],
    baseline: Optional[list[float]],
) -> float:
    if not amplitudes:
        return 0.0
    if baseline is None or len(baseline) != len(amplitudes):
        return coefficient_of_variation(amplitudes)

    normalized = []
    for amp, reference in zip(amplitudes, baseline):
        normalized.append(float(amp) / max(float(reference), 1e-6))
    return coefficient_of_variation(normalized)


def _idle_reference_gate_reason(
    *,
    current_state: int,
    idle_state: int,
    moving_variance: float,
    threshold: float,
    packets_since_transition: int,
    margin_ratio: float,
    transition_guard_packets: int,
) -> bool:
    if current_state != idle_state:
        return "state"
    if packets_since_transition < transition_guard_packets:
        return "transition"
    if moving_variance <= 0.0:
        return "nonpositive"
    if moving_variance > (threshold * margin_ratio):
        return "margin"
    return "ok"


def evaluate_pair(
    pair: PairedDataset,
    *,
    variant: Optional[VarianceVariantConfig] = None,
    filter_config: Optional[VarianceFilterConfig] = None,
    window_size: int = SEG_WINDOW_SIZE,
    selected_band: tuple[int, ...] = DEFAULT_SUBCARRIERS,
    track_trace: bool = False,
    threshold_source: str = "calibrate",
) -> VarianceEvaluationResult:
    """Evaluate one paired dataset with a continuous baseline -> motion pass."""
    cfg = filter_config or VarianceFilterConfig()
    variant_cfg = variant or production_variant()
    static_presence_packets, motion_packets = load_paired_packets(pair)

    startup_threshold, _calibration_mv = calibrate_startup_threshold(
        static_presence_packets,
        selected_band=selected_band,
        window_size=window_size,
        filter_config=cfg,
    )
    effective_threshold_source = "calibrate"

    ctx = build_segmentation_context(threshold=startup_threshold, window_size=window_size, filter_config=cfg)
    tracking = variant_cfg.baseline_tracking
    ema_norm = variant_cfg.subcarrier_ema_norm
    ema_baseline = _seed_ema_baseline(static_presence_packets, selected_band, ema_norm) if ema_norm else None
    idle_history = deque(maxlen=tracking.idle_history_size if tracking else 0)

    tp = fn = fp = tn = 0
    threshold_update_count = 0
    ema_update_count = 0
    idle_reference_count = 0
    tracking_gate_hit_count = 0
    tracking_state_block_count = 0
    tracking_transition_block_count = 0
    tracking_margin_block_count = 0
    max_idle_reference_mv = 0.0
    max_candidate_threshold = startup_threshold
    packets_since_transition = window_size
    last_state = ctx.get_state()
    baseline_trace: Optional[list[PacketTrace]] = [] if track_trace else None
    motion_trace: Optional[list[PacketTrace]] = [] if track_trace else None

    def process_stream(
        packets: list[dict[str, Any]],
        *,
        expected_motion: bool,
        trace: Optional[list[PacketTrace]],
    ) -> tuple[int, int, int, int]:
        nonlocal last_state
        nonlocal packets_since_transition
        nonlocal threshold_update_count
        nonlocal ema_update_count
        nonlocal idle_reference_count
        nonlocal tracking_gate_hit_count
        nonlocal tracking_state_block_count
        nonlocal tracking_transition_block_count
        nonlocal tracking_margin_block_count
        nonlocal max_idle_reference_mv
        nonlocal max_candidate_threshold
        nonlocal ema_baseline

        stream_tp = stream_fn = stream_fp = stream_tn = 0

        for pkt in packets:
            if ema_norm:
                amplitudes = extract_subcarrier_amplitudes(pkt["csi_data"], selected_band)
                turbulence = _ema_normalized_turbulence(amplitudes, ema_baseline)
            else:
                amplitudes = None
                turbulence = ctx.calculate_spatial_turbulence(pkt["csi_data"], selected_band)

            ctx.add_turbulence(turbulence)
            ctx.update_state()
            current_state = ctx.get_state()
            moving_variance = float(ctx.current_moving_variance)
            is_motion = current_state == ctx.STATE_MOTION

            if expected_motion:
                if is_motion:
                    stream_tp += 1
                else:
                    stream_fn += 1
            else:
                if is_motion:
                    stream_fp += 1
                else:
                    stream_tn += 1

            if current_state != last_state:
                packets_since_transition = 0
            else:
                packets_since_transition += 1
            last_state = current_state

            tracking_gate_reason = _idle_reference_gate_reason(
                current_state=current_state,
                idle_state=ctx.STATE_IDLE,
                moving_variance=moving_variance,
                threshold=ctx.threshold,
                packets_since_transition=packets_since_transition,
                margin_ratio=tracking.margin_ratio if tracking else 0.0,
                transition_guard_packets=tracking.transition_guard_packets if tracking else 0,
            ) if tracking else None

            if tracking:
                if tracking_gate_reason == "state":
                    tracking_state_block_count += 1
                elif tracking_gate_reason == "transition":
                    tracking_transition_block_count += 1
                elif tracking_gate_reason == "margin":
                    tracking_margin_block_count += 1

            if tracking and tracking_gate_reason == "ok":
                idle_history.append(moving_variance)
                idle_reference_count += 1
                tracking_gate_hit_count += 1
                max_idle_reference_mv = max(max_idle_reference_mv, moving_variance)
                if len(idle_history) >= tracking.min_idle_samples:
                    rolling_idle_p = float(np.percentile(np.array(idle_history, dtype=float), tracking.idle_percentile))
                    candidate_threshold = max(startup_threshold, rolling_idle_p * tracking.factor)
                    max_candidate_threshold = max(max_candidate_threshold, candidate_threshold)
                    if abs(candidate_threshold - ctx.threshold) > 1e-6:
                        ctx.set_adaptive_threshold(candidate_threshold)
                        threshold_update_count += 1

            ema_gate_reason = _idle_reference_gate_reason(
                current_state=current_state,
                idle_state=ctx.STATE_IDLE,
                moving_variance=moving_variance,
                threshold=ctx.threshold,
                packets_since_transition=packets_since_transition,
                margin_ratio=ema_norm.margin_ratio if ema_norm else 0.0,
                transition_guard_packets=ema_norm.transition_guard_packets if ema_norm else 0,
            ) if ema_norm else None

            if ema_norm and amplitudes and ema_gate_reason == "ok" and ema_baseline is not None and len(ema_baseline) == len(amplitudes):
                idle_reference_count += 1
                for index, value in enumerate(amplitudes):
                    ema_baseline[index] = (
                        (1.0 - ema_norm.alpha) * float(ema_baseline[index])
                        + ema_norm.alpha * float(value)
                    )
                ema_update_count += 1

            if trace is not None:
                trace.append(
                    PacketTrace(
                        moving_variance=moving_variance,
                        threshold=float(ctx.threshold),
                        motion=is_motion,
                    )
                )

        return stream_tp, stream_fn, stream_fp, stream_tn

    base_tp, base_fn, base_fp, base_tn = process_stream(
        static_presence_packets, expected_motion=False, trace=baseline_trace
    )
    tp += base_tp
    fn += base_fn
    fp += base_fp
    tn += base_tn

    move_tp, move_fn, move_fp, move_tn = process_stream(
        motion_packets, expected_motion=True, trace=motion_trace
    )
    tp += move_tp
    fn += move_fn
    fp += move_fp
    tn += move_tn

    recall = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
    precision = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
    fp_rate = (fp / len(static_presence_packets) * 100.0) if static_presence_packets else 0.0
    f1 = (
        2.0 * (precision / 100.0) * (recall / 100.0) / ((precision + recall) / 100.0) * 100.0
        if (precision + recall) > 0.0
        else 0.0
    )

    return VarianceEvaluationResult(
        dataset=pair,
        variant_name=variant_cfg.name,
        filter_config=cfg,
        startup_threshold=float(startup_threshold),
        final_threshold=float(ctx.threshold),
        threshold_source=effective_threshold_source,
        tp=tp,
        fn=fn,
        fp=fp,
        tn=tn,
        recall=float(recall),
        precision=float(precision),
        fp_rate=float(fp_rate),
        f1=float(f1),
        baseline_count=len(static_presence_packets),
        motion_count=len(motion_packets),
        threshold_update_count=threshold_update_count,
        ema_update_count=ema_update_count,
        idle_reference_count=idle_reference_count,
        tracking_gate_hit_count=tracking_gate_hit_count,
        tracking_state_block_count=tracking_state_block_count,
        tracking_transition_block_count=tracking_transition_block_count,
        tracking_margin_block_count=tracking_margin_block_count,
        max_idle_reference_mv=max_idle_reference_mv,
        max_candidate_threshold=max_candidate_threshold,
        selected_band=tuple(int(sc) for sc in selected_band),
        baseline_trace=baseline_trace,
        motion_trace=motion_trace,
    )


def evaluate_pairs(
    pairs: list[PairedDataset],
    *,
    variant: Optional[VarianceVariantConfig] = None,
    filter_config: Optional[VarianceFilterConfig] = None,
    window_size: int = SEG_WINDOW_SIZE,
    selected_band: tuple[int, ...] = DEFAULT_SUBCARRIERS,
    track_trace: bool = False,
    threshold_source: str = "calibrate",
) -> list[VarianceEvaluationResult]:
    """Evaluate a list of paired datasets with the same configuration."""
    return [
        evaluate_pair(
            pair,
            variant=variant,
            filter_config=filter_config,
            window_size=window_size,
            selected_band=selected_band,
            track_trace=track_trace,
            threshold_source=threshold_source,
        )
        for pair in pairs
    ]


def summarize_results(results: list[VarianceEvaluationResult]) -> dict[str, Any]:
    """Aggregate metrics across datasets and chips for reporting."""
    if not results:
        return {
            "dataset_count": 0,
            "recall": 0.0,
            "precision": 0.0,
            "fp_rate": 0.0,
            "f1": 0.0,
            "threshold_updates": 0,
            "ema_updates": 0,
            "idle_reference_count": 0,
            "tracking_gate_hits": 0,
            "tracking_state_blocks": 0,
            "tracking_transition_blocks": 0,
            "tracking_margin_blocks": 0,
            "max_candidate_threshold": 0.0,
            "raise_capable_datasets": 0,
            "per_chip": {},
            "worst_fp_pair": None,
            "worst_recall_pair": None,
        }

    total_tp = sum(r.tp for r in results)
    total_fn = sum(r.fn for r in results)
    total_fp = sum(r.fp for r in results)
    total_baseline = sum(r.baseline_count for r in results)
    dataset_count = len(results)

    recall = (total_tp / (total_tp + total_fn) * 100.0) if (total_tp + total_fn) > 0 else 0.0
    precision = (total_tp / (total_tp + total_fp) * 100.0) if (total_tp + total_fp) > 0 else 0.0
    fp_rate = (total_fp / total_baseline * 100.0) if total_baseline > 0 else 0.0
    f1 = (
        2.0 * (precision / 100.0) * (recall / 100.0) / ((precision + recall) / 100.0) * 100.0
        if (precision + recall) > 0.0
        else 0.0
    )

    per_chip: dict[str, dict[str, Any]] = {}
    for result in results:
        chip_bucket = per_chip.setdefault(
            result.dataset.chip,
            {
                "datasets": 0,
                "tp": 0,
                "fn": 0,
                "fp": 0,
                "baseline_count": 0,
                "threshold_updates": 0,
                "ema_updates": 0,
            },
        )
        chip_bucket["datasets"] += 1
        chip_bucket["tp"] += result.tp
        chip_bucket["fn"] += result.fn
        chip_bucket["fp"] += result.fp
        chip_bucket["baseline_count"] += result.baseline_count
        chip_bucket["threshold_updates"] += result.threshold_update_count
        chip_bucket["ema_updates"] += result.ema_update_count

    for chip_bucket in per_chip.values():
        chip_tp = chip_bucket["tp"]
        chip_fn = chip_bucket["fn"]
        chip_fp = chip_bucket["fp"]
        chip_baseline = chip_bucket["baseline_count"]
        chip_recall = (chip_tp / (chip_tp + chip_fn) * 100.0) if (chip_tp + chip_fn) > 0 else 0.0
        chip_precision = (chip_tp / (chip_tp + chip_fp) * 100.0) if (chip_tp + chip_fp) > 0 else 0.0
        chip_fp_rate = (chip_fp / chip_baseline * 100.0) if chip_baseline > 0 else 0.0
        chip_f1 = (
            2.0 * (chip_precision / 100.0) * (chip_recall / 100.0) / ((chip_precision + chip_recall) / 100.0) * 100.0
            if (chip_precision + chip_recall) > 0.0
            else 0.0
        )
        chip_bucket["recall"] = chip_recall
        chip_bucket["precision"] = chip_precision
        chip_bucket["fp_rate"] = chip_fp_rate
        chip_bucket["f1"] = chip_f1

    return {
        "dataset_count": dataset_count,
        "recall": recall,
        "precision": precision,
        "fp_rate": fp_rate,
        "f1": f1,
        "threshold_updates": sum(r.threshold_update_count for r in results),
        "ema_updates": sum(r.ema_update_count for r in results),
        "idle_reference_count": sum(r.idle_reference_count for r in results),
        "tracking_gate_hits": sum(r.tracking_gate_hit_count for r in results),
        "tracking_state_blocks": sum(r.tracking_state_block_count for r in results),
        "tracking_transition_blocks": sum(r.tracking_transition_block_count for r in results),
        "tracking_margin_blocks": sum(r.tracking_margin_block_count for r in results),
        "max_candidate_threshold": max(r.max_candidate_threshold for r in results),
        "raise_capable_datasets": sum(1 for r in results if r.max_candidate_threshold > (r.startup_threshold + 1e-6)),
        "per_chip": per_chip,
        "worst_fp_pair": max(results, key=lambda item: item.fp_rate),
        "worst_recall_pair": min(results, key=lambda item: item.recall),
    }


__all__ = [
    "BaselineTrackingConfig",
    "PacketTrace",
    "PairedDataset",
    "SubcarrierEMANormConfig",
    "VarianceEvaluationResult",
    "VarianceFilterConfig",
    "VarianceVariantConfig",
    "baseline_tracking_variant",
    "build_segmentation_context",
    "calibrate_startup_threshold",
    "coefficient_of_variation",
    "evaluate_pair",
    "evaluate_pairs",
    "extract_subcarrier_amplitudes",
    "iter_paired_datasets",
    "load_paired_packets",
    "production_variant",
    "subcarrier_ema_norm_variant",
    "summarize_results",
]
