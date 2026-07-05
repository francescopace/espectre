#!/usr/bin/env python3
"""
Independent offline research benchmark: motion-score candidates vs. production MVS.

Scope and intent
-----------------
This is a standalone, read-only research tool. It does not modify the
production runtime, and it is deliberately separate from
`tools/12_benchmark_motion_features.py` (a different, concurrent research
track using the same repository).

It evaluates a family of candidate single-scalar motion scores against the
real datasets already collected in `data/`, using the same fixed 12-subcarrier
band and the same conditioning (Hampel filter, window=100) as production, so
that every candidate is fed the same signal MVS/ML would see. All scores are
computed once from raw CSI; no synthetic data is used except for the
dedicated AGC-drift and RF-spike robustness stress tests, which are clearly
labeled as such.

Evaluation protocol
--------------------
- Fair baseline: the production MVS numbers are computed by running the
  actual production code (`tools/lib/mvs_sweep_core`), not a re-implementation.
- Cross-chip generalization: leave-one-chip-out (LOCO). A global decision
  threshold is selected only from the three training chips (paired sessions
  plus their empty/quiet-long-run hard negatives), then frozen and applied,
  unseen, to the held-out chip.
- Calibration-fragility gap: a second, production-style variant re-derives a
  per-session threshold from each pair's own boot-calibration segment
  (mirroring `max(calibration) x factor`), with the factor itself tuned only
  on the training chips. The gap between the global (no-calibration) and the
  per-session-calibrated variant quantifies how much a candidate secretly
  depends on session-specific calibration.
- Quiet-room false positives are tracked separately from paired
  static_presence/motion recall, using `data/empty` and the long quiet-room
  `data/test` recordings (10 minutes each).
- Temporal stability, AGC/gain-drift sensitivity, and RF-spike sensitivity are
  evaluated as explicit stress diagnostics, not folded into the headline
  ranking metric.

Usage
-----
    python tools/research_motion_score_benchmark.py
    python tools/research_motion_score_benchmark.py --hop 25 --quick
    python tools/research_motion_score_benchmark.py --save-json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.lib.bootstrap import setup_paths

setup_paths()

from tools.lib import motion_score_lab as msl
from tools.lib import mvs_sweep_core as sweep
from tools.lib.csi_io import load_npz_as_packets
from tools.lib.dataset_metadata import load_dataset_info
from tools.lib.repo_paths import data_dir

import config

CHIPS = ("C3", "C5", "C6", "S3")
CALIBRATION_PACKETS = int(config.CALIBRATION_BUFFER_SIZE)  # 1000, matches production boot calibration
GAIN_FACTORS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
SPIKE_CONFIGS = (
    {"spike_rate": 0.002, "spike_factor_range": (3.0, 8.0)},
    {"spike_rate": 0.01, "spike_factor_range": (3.0, 8.0)},
)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class PairSeries:
    chip: str
    pair_id: str
    environment: str
    turb_filtered: np.ndarray
    amp_sel: np.ndarray
    amp_all: np.ndarray
    label: np.ndarray  # True = motion, aligned 1:1 with turb_filtered
    static_len: int
    fs: float


@dataclass
class IdleSeries:
    chip: str
    rec_id: str
    source: str  # "empty" | "test_quiet"
    turb_filtered: np.ndarray
    amp_sel: np.ndarray
    amp_all: np.ndarray
    fs: float


def _process_packets(packets: Sequence[Dict[str, Any]], selected_band: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    csi = msl.packets_to_csi_matrix(packets)
    amp_all = msl.amplitudes_from_csi_batch(csi)
    amp_sel = amp_all[:, list(selected_band)]
    turb_raw = msl.raw_turbulence(amp_sel)
    turb_filtered = msl.hampel_filter_series(turb_raw)
    return amp_all, amp_sel, turb_filtered


def load_all_pairs(chip: Optional[str] = None, limit: Optional[int] = None) -> List[PairSeries]:
    pairs = sweep.iter_paired_datasets(chip=chip, limit=limit)
    out: List[PairSeries] = []
    for pair in pairs:
        static_packets, motion_packets = sweep.load_paired_packets(pair)
        amp_all_s, amp_sel_s, turb_s = _process_packets(static_packets, config.DEFAULT_SUBCARRIERS)
        amp_all_m, amp_sel_m, turb_m = _process_packets(motion_packets, config.DEFAULT_SUBCARRIERS)

        amp_all = np.concatenate([amp_all_s, amp_all_m], axis=0)
        amp_sel = np.concatenate([amp_sel_s, amp_sel_m], axis=0)
        turb_filtered = np.concatenate([turb_s, turb_m], axis=0)
        label = np.concatenate([np.zeros(len(turb_s), dtype=bool), np.ones(len(turb_m), dtype=bool)])

        static_entry = next(
            (e for e in load_dataset_info()["files"].get("static_presence", []) if e.get("filename") == pair.static_presence_path.name),
            {},
        )
        fs = msl.estimate_sample_rate_hz(static_entry, len(static_packets)) if static_entry else 100.0

        out.append(
            PairSeries(
                chip=pair.chip,
                pair_id=pair.dataset_id,
                environment=pair.environment,
                turb_filtered=turb_filtered,
                amp_sel=amp_sel,
                amp_all=amp_all,
                label=label,
                static_len=len(turb_s),
                fs=fs,
            )
        )
    return out


def _extract_motion_start_packet(description: Optional[str]) -> Optional[int]:
    if not description:
        return None
    import re

    match = re.search(r"motion[_ ]?start[_ ]?(?:packet)?[:=]?\s*(\d+)", str(description), re.IGNORECASE)
    return int(match.group(1)) if match else None


def load_idle_recordings(chip: Optional[str] = None) -> List[IdleSeries]:
    """Load all `empty/` and long quiet `test/` recordings as idle-only streams.

    Any `test/` entry whose description encodes an explicit motion-start
    packet is skipped defensively (none currently do, verified against
    `data/dataset_info.json`, but this keeps the loader honest).
    """
    info = load_dataset_info()
    data_root = data_dir()
    out: List[IdleSeries] = []

    for label, source in (("empty", "empty"), ("test", "test_quiet")):
        for entry in info.get("files", {}).get(label, []):
            if int(entry.get("subcarriers", 0) or 0) != 64:
                continue
            entry_chip = str(entry.get("chip", "")).upper()
            if chip and entry_chip != chip.upper():
                continue
            if source == "test_quiet" and _extract_motion_start_packet(entry.get("description")) is not None:
                continue
            path = data_root / label / str(entry.get("filename"))
            if not path.exists():
                continue
            packets = load_npz_as_packets(path)
            if len(packets) < config.SEG_WINDOW_SIZE + 1:
                continue
            amp_all, amp_sel, turb_filtered = _process_packets(packets, config.DEFAULT_SUBCARRIERS)
            fs = msl.estimate_sample_rate_hz(entry, len(packets))
            out.append(
                IdleSeries(
                    chip=entry_chip,
                    rec_id=f"{source}:{path.stem}",
                    source=source,
                    turb_filtered=turb_filtered,
                    amp_sel=amp_sel,
                    amp_all=amp_all,
                    fs=fs,
                )
            )
    return out


# ---------------------------------------------------------------------------
# Window dataset assembly
# ---------------------------------------------------------------------------


@dataclass
class WindowPool:
    scores: Dict[str, np.ndarray] = field(default_factory=dict)
    label: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    chip: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))
    kind: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))
    group_id: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))
    source: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))


def build_window_pool(
    pairs: Sequence[PairSeries],
    idle_recs: Sequence[IdleSeries],
    *,
    window: int,
    hop: int,
) -> Tuple[WindowPool, Dict[str, msl.WindowScores], Dict[str, IdleSeries]]:
    """Build the flat evaluation pool plus per-recording idle scores (for stability)."""
    per_candidate_scores: Dict[str, List[np.ndarray]] = {name: [] for name in msl.CANDIDATE_NAMES}
    labels: List[np.ndarray] = []
    chips: List[np.ndarray] = []
    kinds: List[np.ndarray] = []
    group_ids: List[np.ndarray] = []
    sources: List[np.ndarray] = []

    for pair in pairs:
        ws = msl.compute_window_scores(pair.turb_filtered, pair.amp_sel, pair.amp_all, window=window, hop=hop, fs=pair.fs)
        n = len(ws.end_index)
        if n == 0:
            continue
        for name in msl.CANDIDATE_NAMES:
            per_candidate_scores[name].append(ws.scores[name])
        labels.append(pair.label[ws.end_index])
        chips.append(np.full(n, pair.chip, dtype=object))
        kinds.append(np.full(n, "pair", dtype=object))
        group_ids.append(np.full(n, pair.pair_id, dtype=object))
        sources.append(np.full(n, "paired", dtype=object))

    idle_window_scores: Dict[str, msl.WindowScores] = {}
    idle_by_id: Dict[str, IdleSeries] = {}
    for rec in idle_recs:
        ws = msl.compute_window_scores(rec.turb_filtered, rec.amp_sel, rec.amp_all, window=window, hop=hop, fs=rec.fs)
        idle_window_scores[rec.rec_id] = ws
        idle_by_id[rec.rec_id] = rec
        n = len(ws.end_index)
        if n == 0:
            continue
        for name in msl.CANDIDATE_NAMES:
            per_candidate_scores[name].append(ws.scores[name])
        labels.append(np.zeros(n, dtype=bool))
        chips.append(np.full(n, rec.chip, dtype=object))
        kinds.append(np.full(n, "idle", dtype=object))
        group_ids.append(np.full(n, rec.rec_id, dtype=object))
        sources.append(np.full(n, rec.source, dtype=object))

    pool = WindowPool(
        scores={name: np.concatenate(chunks) for name, chunks in per_candidate_scores.items()},
        label=np.concatenate(labels),
        chip=np.concatenate(chips),
        kind=np.concatenate(kinds),
        group_id=np.concatenate(group_ids),
        source=np.concatenate(sources),
    )
    return pool, idle_window_scores, idle_by_id


# ---------------------------------------------------------------------------
# Threshold search and metrics
# ---------------------------------------------------------------------------


def _confusion_metrics(pred_motion: np.ndarray, pos_mask: np.ndarray, neg_mask: np.ndarray) -> Dict[str, float]:
    tp = int(np.sum(pred_motion & pos_mask))
    fn = int(np.sum(~pred_motion & pos_mask))
    fp = int(np.sum(pred_motion & neg_mask))
    tn = int(np.sum(~pred_motion & neg_mask))
    recall = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
    precision = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
    fp_rate = (fp / (fp + tn) * 100.0) if (fp + tn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"recall": recall, "precision": precision, "fp_rate": fp_rate, "f1": f1, "tp": tp, "fn": fn, "fp": fp, "tn": tn}


def evaluate_signed_threshold(
    signed_scores: np.ndarray, label: np.ndarray, kind: np.ndarray, mask: np.ndarray, threshold: float
) -> Dict[str, float]:
    sub_scores = signed_scores[mask]
    sub_label = label[mask]
    sub_kind = kind[mask]
    pred_motion = sub_scores > threshold
    pos_mask = (sub_kind == "pair") & sub_label
    neg_mask = ((sub_kind == "pair") & ~sub_label) | (sub_kind == "idle")
    return _confusion_metrics(pred_motion, pos_mask, neg_mask)


def determine_sign(scores: np.ndarray, label: np.ndarray, kind: np.ndarray, mask: np.ndarray) -> float:
    """Return +1.0 or -1.0 so that `sign * score` increases with motion, per training data."""
    sub = mask & (kind == "pair")
    motion_mean = scores[sub & label].mean() if np.any(sub & label) else 0.0
    idle_mean = scores[sub & ~label].mean() if np.any(sub & ~label) else 0.0
    return 1.0 if motion_mean >= idle_mean else -1.0


def search_best_threshold(signed_scores: np.ndarray, label: np.ndarray, kind: np.ndarray, mask: np.ndarray, num_grid: int = 300) -> Tuple[float, Dict[str, float]]:
    sub_scores = signed_scores[mask]
    if len(sub_scores) == 0:
        return 0.0, _confusion_metrics(np.array([], dtype=bool), np.array([], dtype=bool), np.array([], dtype=bool))
    quantiles = np.linspace(0.5, 99.9, num_grid)
    candidates = np.unique(np.percentile(sub_scores, quantiles))
    best_threshold = candidates[0] if len(candidates) else 0.0
    best_f1 = -1.0
    best_metrics: Dict[str, float] = {}
    for threshold in candidates:
        metrics = evaluate_signed_threshold(signed_scores, label, kind, mask, threshold)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = threshold
            best_metrics = metrics
    return float(best_threshold), best_metrics


def search_best_calibration_factor(
    signed_scores: np.ndarray,
    label: np.ndarray,
    kind: np.ndarray,
    group_id: np.ndarray,
    train_mask: np.ndarray,
    calib_windows_per_group: Dict[str, int],
) -> float:
    """Pick the `max(calibration) x factor` multiplier that maximizes F1 on training pairs."""
    factors = np.array([0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0])
    train_pair_groups = sorted({g for g in np.unique(group_id[train_mask & (kind == "pair")])})
    if not train_pair_groups:
        return 1.3

    best_factor = 1.3
    best_f1 = -1.0
    for factor in factors:
        pred_all = np.zeros(len(signed_scores), dtype=bool)
        active_mask = np.zeros(len(signed_scores), dtype=bool)
        for group in train_pair_groups:
            group_mask = group_id == group
            n_calib = calib_windows_per_group.get(group)
            if not n_calib:
                continue
            group_indices = np.flatnonzero(group_mask)
            calib_indices = group_indices[:n_calib]
            if len(calib_indices) == 0:
                continue
            calib_max = signed_scores[calib_indices].max()
            threshold = calib_max * factor
            active_mask[group_indices] = True
            pred_all[group_indices] = signed_scores[group_indices] > threshold
        sub_active = active_mask & train_mask
        sub_label = label[sub_active]
        sub_kind = kind[sub_active]
        pos_mask = (sub_kind == "pair") & sub_label
        neg_mask = ((sub_kind == "pair") & ~sub_label) | (sub_kind == "idle")
        metrics = _confusion_metrics(pred_all[sub_active], pos_mask, neg_mask)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_factor = float(factor)
    return best_factor


def evaluate_calibrated_variant(
    signed_scores: np.ndarray,
    label: np.ndarray,
    kind: np.ndarray,
    group_id: np.ndarray,
    test_mask: np.ndarray,
    factor: float,
    calib_windows_per_group: Dict[str, int],
) -> Dict[str, float]:
    test_pair_groups = sorted({g for g in np.unique(group_id[test_mask & (kind == "pair")])})
    pred_all = np.zeros(len(signed_scores), dtype=bool)
    active_mask = np.zeros(len(signed_scores), dtype=bool)
    for group in test_pair_groups:
        group_mask = group_id == group
        n_calib = calib_windows_per_group.get(group)
        if not n_calib:
            continue
        group_indices = np.flatnonzero(group_mask)
        calib_indices = group_indices[:n_calib]
        if len(calib_indices) == 0:
            continue
        calib_max = signed_scores[calib_indices].max()
        threshold = calib_max * factor
        active_mask[group_indices] = True
        pred_all[group_indices] = signed_scores[group_indices] > threshold
    sub_active = active_mask & test_mask
    sub_label = label[sub_active]
    sub_kind = kind[sub_active]
    pos_mask = (sub_kind == "pair") & sub_label
    neg_mask = ((sub_kind == "pair") & ~sub_label) | (sub_kind == "idle")
    return _confusion_metrics(pred_all[sub_active], pos_mask, neg_mask)


# ---------------------------------------------------------------------------
# Main LOCO evaluation per candidate
# ---------------------------------------------------------------------------


def run_loco_for_candidate(
    name: str,
    pool: WindowPool,
    calib_windows_per_group: Dict[str, int],
) -> Dict[str, Any]:
    scores = pool.scores[name]
    folds = []
    for held_out in CHIPS:
        train_mask = pool.chip != held_out
        test_mask = pool.chip == held_out
        if not np.any(train_mask) or not np.any(test_mask):
            continue
        sign = determine_sign(scores, pool.label, pool.kind, train_mask)
        signed_scores = scores * sign

        threshold, train_metrics = search_best_threshold(signed_scores, pool.label, pool.kind, train_mask)
        test_metrics_global = evaluate_signed_threshold(signed_scores, pool.label, pool.kind, test_mask, threshold)

        # Quiet-room-only FP within the held-out chip (idle kind only).
        idle_test_mask = test_mask & (pool.kind == "idle")
        quiet_fp_rate = 0.0
        if np.any(idle_test_mask):
            pred_idle = signed_scores[idle_test_mask] > threshold
            quiet_fp_rate = float(np.mean(pred_idle) * 100.0)

        factor = search_best_calibration_factor(signed_scores, pool.label, pool.kind, pool.group_id, train_mask, calib_windows_per_group)
        test_metrics_calibrated = evaluate_calibrated_variant(
            signed_scores, pool.label, pool.kind, pool.group_id, test_mask, factor, calib_windows_per_group
        )

        folds.append(
            {
                "held_out_chip": held_out,
                "sign": sign,
                "global_threshold": threshold,
                "calibration_factor": factor,
                "global": test_metrics_global,
                "quiet_fp_rate_global": quiet_fp_rate,
                "calibrated": test_metrics_calibrated,
            }
        )
    return {"name": name, "folds": folds}


def aggregate_loco(loco_result: Dict[str, Any]) -> Dict[str, float]:
    folds = loco_result["folds"]
    if not folds:
        return {}

    def mean_of(path_fn):
        values = [path_fn(f) for f in folds]
        return float(np.mean(values))

    def worst_of(path_fn):
        values = [path_fn(f) for f in folds]
        return float(np.max(values))

    return {
        "mean_recall_global": mean_of(lambda f: f["global"]["recall"]),
        "mean_precision_global": mean_of(lambda f: f["global"]["precision"]),
        "mean_fp_rate_global": mean_of(lambda f: f["global"]["fp_rate"]),
        "mean_f1_global": mean_of(lambda f: f["global"]["f1"]),
        "mean_quiet_fp_rate_global": mean_of(lambda f: f["quiet_fp_rate_global"]),
        "worst_fp_rate_global": worst_of(lambda f: f["global"]["fp_rate"]),
        "worst_quiet_fp_rate_global": worst_of(lambda f: f["quiet_fp_rate_global"]),
        "worst_recall_global": float(np.min([f["global"]["recall"] for f in folds])),
        "mean_recall_calibrated": mean_of(lambda f: f["calibrated"]["recall"]),
        "mean_precision_calibrated": mean_of(lambda f: f["calibrated"]["precision"]),
        "mean_fp_rate_calibrated": mean_of(lambda f: f["calibrated"]["fp_rate"]),
        "mean_f1_calibrated": mean_of(lambda f: f["calibrated"]["f1"]),
        "calibration_gap_f1": mean_of(lambda f: f["calibrated"]["f1"]) - mean_of(lambda f: f["global"]["f1"]),
    }


# ---------------------------------------------------------------------------
# Stability and stress diagnostics
# ---------------------------------------------------------------------------


def compute_temporal_stability(idle_window_scores: Dict[str, msl.WindowScores]) -> Dict[str, Dict[str, float]]:
    per_candidate: Dict[str, List[Tuple[float, float]]] = {name: [] for name in msl.CANDIDATE_NAMES}
    for rec_id, ws in idle_window_scores.items():
        for name in msl.CANDIDATE_NAMES:
            series = ws.scores[name]
            if len(series) < 10:
                continue
            mean_abs = max(abs(float(np.mean(series))), 1e-9)
            cv = float(np.std(series) / mean_abs)
            half = len(series) // 2
            first_mean = float(np.mean(series[:half]))
            second_mean = float(np.mean(series[half:]))
            drift = (second_mean - first_mean) / mean_abs
            per_candidate[name].append((cv, drift))

    summary: Dict[str, Dict[str, float]] = {}
    for name, values in per_candidate.items():
        if not values:
            summary[name] = {"median_cv": 0.0, "max_abs_drift": 0.0}
            continue
        cvs = [v[0] for v in values]
        drifts = [abs(v[1]) for v in values]
        summary[name] = {
            "median_cv": float(np.median(cvs)),
            "max_abs_drift": float(np.max(drifts)),
            "median_abs_drift": float(np.median(drifts)),
        }
    return summary


def compute_gain_stress(
    idle_recs: Sequence[IdleSeries],
    loco_by_candidate: Dict[str, Dict[str, Any]],
    *,
    window: int,
    hop: int,
) -> Dict[str, Dict[str, Any]]:
    """For each candidate, measure induced quiet-room FP rate and score drift under synthetic gain shift."""
    chip_threshold: Dict[str, Dict[str, float]] = {}
    for name, result in loco_by_candidate.items():
        chip_threshold[name] = {}
        for fold in result["folds"]:
            chip_threshold[name][fold["held_out_chip"]] = {"threshold": fold["global_threshold"], "sign": fold["sign"]}

    fp_by_factor: Dict[str, Dict[float, List[float]]] = {name: {factor: [] for factor in GAIN_FACTORS} for name in msl.CANDIDATE_NAMES}

    for rec in idle_recs:
        chip_info_by_name = {name: chip_threshold[name].get(rec.chip) for name in msl.CANDIDATE_NAMES}
        for factor in GAIN_FACTORS:
            amp_all_g, amp_sel_g = msl.apply_gain_shift(rec.amp_all, rec.amp_sel, factor)
            turb_raw_g = msl.raw_turbulence(amp_sel_g)
            turb_filtered_g = msl.hampel_filter_series(turb_raw_g)
            ws = msl.compute_window_scores(turb_filtered_g, amp_sel_g, amp_all_g, window=window, hop=hop, fs=rec.fs)
            for name in msl.CANDIDATE_NAMES:
                series = ws.scores[name]
                info = chip_info_by_name[name]
                if info is None or len(series) == 0:
                    continue
                signed = series * info["sign"]
                pred_motion = signed > info["threshold"]
                fp_by_factor[name][factor].append(float(np.mean(pred_motion) * 100.0))
    summary: Dict[str, Dict[str, Any]] = {}
    for name in msl.CANDIDATE_NAMES:
        summary[name] = {
            f"quiet_fp_rate_at_{factor}x": (float(np.mean(fp_by_factor[name][factor])) if fp_by_factor[name][factor] else 0.0)
            for factor in GAIN_FACTORS
        }
    return summary


def compute_spike_stress(
    idle_recs: Sequence[IdleSeries],
    loco_by_candidate: Dict[str, Dict[str, Any]],
    *,
    window: int,
    hop: int,
    seed: int = 12345,
) -> Dict[str, Dict[str, Any]]:
    chip_threshold: Dict[str, Dict[str, Dict[str, float]]] = {}
    for name, result in loco_by_candidate.items():
        chip_threshold[name] = {}
        for fold in result["folds"]:
            chip_threshold[name][fold["held_out_chip"]] = {"threshold": fold["global_threshold"], "sign": fold["sign"]}

    rng = np.random.default_rng(seed)
    baseline_fp: Dict[str, List[float]] = {name: [] for name in msl.CANDIDATE_NAMES}
    spike_fp: Dict[str, Dict[float, List[float]]] = {
        name: {cfg["spike_rate"]: [] for cfg in SPIKE_CONFIGS} for name in msl.CANDIDATE_NAMES
    }

    selected_indices = list(config.DEFAULT_SUBCARRIERS)
    for rec in idle_recs:
        base_ws = msl.compute_window_scores(rec.turb_filtered, rec.amp_sel, rec.amp_all, window=window, hop=hop, fs=rec.fs)
        chip_info_by_name = {name: chip_threshold[name].get(rec.chip) for name in msl.CANDIDATE_NAMES}
        for name in msl.CANDIDATE_NAMES:
            info = chip_info_by_name[name]
            series = base_ws.scores[name]
            if info is None or len(series) == 0:
                continue
            signed = series * info["sign"]
            baseline_fp[name].append(float(np.mean(signed > info["threshold"]) * 100.0))

        for cfg in SPIKE_CONFIGS:
            amp_all_s, amp_sel_s = msl.inject_spike_noise(
                rec.amp_all,
                rec.amp_sel,
                rng=rng,
                spike_rate=cfg["spike_rate"],
                spike_factor_range=cfg["spike_factor_range"],
                selected_indices=selected_indices,
            )
            turb_raw_s = msl.raw_turbulence(amp_sel_s)
            turb_filtered_s = msl.hampel_filter_series(turb_raw_s)
            ws = msl.compute_window_scores(turb_filtered_s, amp_sel_s, amp_all_s, window=window, hop=hop, fs=rec.fs)
            for name in msl.CANDIDATE_NAMES:
                info = chip_info_by_name[name]
                series = ws.scores[name]
                if info is None or len(series) == 0:
                    continue
                signed = series * info["sign"]
                spike_fp[name][cfg["spike_rate"]].append(float(np.mean(signed > info["threshold"]) * 100.0))

    summary: Dict[str, Dict[str, Any]] = {}
    for name in msl.CANDIDATE_NAMES:
        entry = {"baseline_fp_rate": float(np.mean(baseline_fp[name])) if baseline_fp[name] else 0.0}
        for cfg in SPIKE_CONFIGS:
            rate = cfg["spike_rate"]
            values = spike_fp[name][rate]
            entry[f"fp_rate_spike_{rate}"] = float(np.mean(values)) if values else 0.0
        summary[name] = entry
    return summary


# ---------------------------------------------------------------------------
# Production reference (real MVS code, real metadata thresholds)
# ---------------------------------------------------------------------------


def production_mvs_reference() -> Dict[str, Any]:
    pairs = sweep.iter_paired_datasets()
    results = sweep.evaluate_pairs(pairs, threshold_source="metadata")
    summary = sweep.summarize_results(results)
    return summary


def production_mvs_quiet_room(idle_recs: Sequence[IdleSeries]) -> Dict[str, float]:
    """Run the exact SegmentationContext with each recording's own metadata threshold."""
    from segmentation import SegmentationContext
    from detector_interface import MotionState

    info = load_dataset_info()
    threshold_by_filename: Dict[str, float] = {}
    for label in ("empty", "test"):
        for entry in info.get("files", {}).get(label, []):
            value = entry.get("optimal_threshold_gridsearch")
            if value:
                threshold_by_filename[str(entry.get("filename"))] = float(value)

    per_chip_motion = {}
    per_chip_total = {}
    for rec in idle_recs:
        filename = rec.rec_id.split(":", 1)[1]
        # rec_id stem does not include extension; look up by stem match.
        matched_threshold = None
        for filename_key, threshold in threshold_by_filename.items():
            if Path(filename_key).stem == filename:
                matched_threshold = threshold
                break
        if matched_threshold is None:
            continue
        # Replay from raw amplitudes so ctx applies its own (single) filter chain.
        ctx = SegmentationContext(window_size=config.SEG_WINDOW_SIZE, threshold=matched_threshold, enable_hampel=True)
        motion_count = 0
        total = 0
        for row in rec.amp_sel:
            turbulence = SegmentationContext._turbulence_from_amplitude_buffer(list(row), len(row))
            ctx.add_turbulence(turbulence)
            ctx.update_state()
            total += 1
            if ctx.get_state() == MotionState.MOTION:
                motion_count += 1
        per_chip_motion[rec.chip] = per_chip_motion.get(rec.chip, 0) + motion_count
        per_chip_total[rec.chip] = per_chip_total.get(rec.chip, 0) + total

    return {chip: (per_chip_motion[chip] / per_chip_total[chip] * 100.0) for chip in per_chip_total if per_chip_total[chip] > 0}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_report(
    loco_by_candidate: Dict[str, Dict[str, Any]],
    stability: Dict[str, Dict[str, float]],
    gain_stress: Dict[str, Dict[str, Any]],
    spike_stress: Dict[str, Dict[str, Any]],
    mvs_reference: Dict[str, Any],
    mvs_quiet: Dict[str, float],
) -> None:
    print("\n" + "=" * 120)
    print("PRODUCTION MVS REFERENCE (real code, per-session metadata threshold)")
    print("=" * 120)
    print(f"Aggregate: recall={mvs_reference['recall']:.1f}% precision={mvs_reference['precision']:.1f}% "
          f"fp_rate={mvs_reference['fp_rate']:.1f}% f1={mvs_reference['f1']:.1f}%")
    for chip, bucket in sorted(mvs_reference.get("per_chip", {}).items()):
        print(f"  {chip:<4} recall={bucket['recall']:.1f}% precision={bucket['precision']:.1f}% "
              f"fp_rate={bucket['fp_rate']:.1f}% f1={bucket['f1']:.1f}%")
    print("Quiet-room (empty + long quiet test) FP rate, per chip, own-session calibrated threshold:")
    for chip, rate in sorted(mvs_quiet.items()):
        print(f"  {chip:<4} fp_rate={rate:.2f}%")

    print("\n" + "=" * 120)
    print("CANDIDATE RANKING — LEAVE-ONE-CHIP-OUT (global threshold, no per-session calibration)")
    print("=" * 120)
    header = f"{'Candidate':<20} {'F1':>7} {'Recall':>7} {'Prec':>7} {'FP%':>7} {'QuietFP%':>9} {'WorstFP%':>9} {'CalibGapF1':>11}"
    print(header)
    print("-" * len(header))
    rows = []
    for name, result in loco_by_candidate.items():
        agg = aggregate_loco(result)
        if not agg:
            continue
        rows.append((name, agg))
    rows.sort(key=lambda item: item[1]["mean_f1_global"], reverse=True)
    for name, agg in rows:
        print(
            f"{name:<20} {agg['mean_f1_global']:>6.1f}% {agg['mean_recall_global']:>6.1f}% "
            f"{agg['mean_precision_global']:>6.1f}% {agg['mean_fp_rate_global']:>6.1f}% "
            f"{agg['mean_quiet_fp_rate_global']:>8.1f}% {agg['worst_quiet_fp_rate_global']:>8.1f}% "
            f"{agg['calibration_gap_f1']:>+10.1f}%"
        )

    print("\nPer-chip breakdown (global LOCO threshold):")
    for name, result in loco_by_candidate.items():
        print(f"\n  {name}:")
        for fold in result["folds"]:
            g = fold["global"]
            print(
                f"    held_out={fold['held_out_chip']:<4} recall={g['recall']:>6.1f}% precision={g['precision']:>6.1f}% "
                f"fp_rate={g['fp_rate']:>6.1f}% f1={g['f1']:>6.1f}% quiet_fp={fold['quiet_fp_rate_global']:>6.1f}%"
            )

    print("\n" + "=" * 120)
    print("TEMPORAL STABILITY (idle long recordings; lower is better)")
    print("=" * 120)
    print(f"{'Candidate':<20} {'MedianCV':>10} {'MedianAbsDrift':>15} {'MaxAbsDrift':>12}")
    for name in msl.CANDIDATE_NAMES:
        s = stability.get(name, {})
        print(f"{name:<20} {s.get('median_cv', 0.0):>10.3f} {s.get('median_abs_drift', 0.0):>15.3f} {s.get('max_abs_drift', 0.0):>12.3f}")

    if gain_stress:
        print("\n" + "=" * 120)
        print("AGC / GAIN-DRIFT STRESS: induced quiet-room FP rate vs. synthetic gain factor")
        print("=" * 120)
        header2 = f"{'Candidate':<20}" + "".join(f"{('x' + str(f)):>9}" for f in GAIN_FACTORS)
        print(header2)
        for name in msl.CANDIDATE_NAMES:
            s = gain_stress.get(name, {})
            row = f"{name:<20}"
            for factor in GAIN_FACTORS:
                row += f"{s.get(f'quiet_fp_rate_at_{factor}x', 0.0):>8.1f}%"
            print(row)

    if spike_stress:
        print("\n" + "=" * 120)
        print("RF-SPIKE STRESS: induced quiet-room FP rate (baseline vs. synthetic spike injection)")
        print("=" * 120)
        print(f"{'Candidate':<20} {'Baseline%':>10} {'Spike0.2%%':>11} {'Spike1%%':>10}")
        for name in msl.CANDIDATE_NAMES:
            s = spike_stress.get(name, {})
            print(
                f"{name:<20} {s.get('baseline_fp_rate', 0.0):>9.1f}% "
                f"{s.get('fp_rate_spike_0.002', 0.0):>10.1f}% {s.get('fp_rate_spike_0.01', 0.0):>9.1f}%"
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hop", type=int, default=10, help="Window hop in packets for the main LOCO evaluation (default: 10)")
    parser.add_argument("--stress-hop", type=int, default=25, help="Window hop for stress diagnostics (default: 25)")
    parser.add_argument("--window", type=int, default=config.SEG_WINDOW_SIZE, help="Score window size in packets")
    parser.add_argument("--save-json", type=str, default=None, help="Optional path to save full numeric results as JSON")
    parser.add_argument("--skip-stress", action="store_true", help="Skip AGC/RF-spike stress tests (faster)")
    args = parser.parse_args()

    t0 = time.time()
    print("Loading paired static_presence/motion datasets...")
    pairs = load_all_pairs()
    print(f"  {len(pairs)} pairs loaded across chips: {sorted({p.chip for p in pairs})}")

    print("Loading idle-only datasets (empty + long quiet-room test recordings)...")
    idle_recs = load_idle_recordings()
    print(f"  {len(idle_recs)} idle recordings loaded across chips: {sorted({r.chip for r in idle_recs})}")

    print(f"Building window pool (window={args.window}, hop={args.hop})...")
    pool, idle_window_scores, _idle_by_id = build_window_pool(pairs, idle_recs, window=args.window, hop=args.hop)
    print(f"  {len(pool.label)} total windows ({int(np.sum(pool.kind == 'pair'))} paired, {int(np.sum(pool.kind == 'idle'))} idle)")

    calib_windows_per_group: Dict[str, int] = {}
    for pair in pairs:
        n_calib_packets = min(CALIBRATION_PACKETS, pair.static_len)
        n_calib_windows = max(0, (n_calib_packets - args.window) // args.hop + 1) if n_calib_packets >= args.window else 0
        calib_windows_per_group[pair.pair_id] = n_calib_windows

    print("Running leave-one-chip-out evaluation per candidate...")
    loco_by_candidate: Dict[str, Dict[str, Any]] = {}
    for name in msl.CANDIDATE_NAMES:
        loco_by_candidate[name] = run_loco_for_candidate(name, pool, calib_windows_per_group)
    print(f"  done ({time.time() - t0:.1f}s elapsed)")

    print("Computing temporal stability on idle long recordings...")
    stability = compute_temporal_stability(idle_window_scores)

    gain_stress: Dict[str, Dict[str, Any]] = {}
    spike_stress: Dict[str, Dict[str, Any]] = {}
    if not args.skip_stress:
        print("Running AGC/gain-drift stress test...")
        gain_stress = compute_gain_stress(idle_recs, loco_by_candidate, window=args.window, hop=args.stress_hop)
        print("Running RF-spike stress test...")
        spike_stress = compute_spike_stress(idle_recs, loco_by_candidate, window=args.window, hop=args.stress_hop)

    print("Computing production MVS reference (real code)...")
    mvs_reference = production_mvs_reference()
    mvs_quiet = production_mvs_quiet_room(idle_recs)

    print_report(loco_by_candidate, stability, gain_stress, spike_stress, mvs_reference, mvs_quiet)

    if args.save_json:
        out_path = Path(args.save_json)
        payload = {
            "loco": {name: aggregate_loco(result) for name, result in loco_by_candidate.items()},
            "loco_folds": {
                name: [
                    {k: v for k, v in fold.items() if k not in ("global", "calibrated")}
                    | {"global": fold["global"], "calibrated": fold["calibrated"]}
                    for fold in result["folds"]
                ]
                for name, result in loco_by_candidate.items()
            },
            "stability": stability,
            "gain_stress": gain_stress,
            "spike_stress": spike_stress,
            "mvs_reference": {k: v for k, v in mvs_reference.items() if k not in ("worst_fp_pair", "worst_recall_pair")},
            "mvs_quiet_room": mvs_quiet,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, default=str))
        print(f"\nSaved full results to {out_path}")

    print(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
