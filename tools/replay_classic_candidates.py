#!/usr/bin/env python3
"""
ESPectre - Classic Candidate Replay

Research-only fitter and replay harness for pair and triplet Classic detector
candidates. It mirrors the grouped logistic fit and startup-threshold workflow
of `fit_classic_detector.py`, but never writes runtime artifacts.

Usage:
    python tools/replay_classic_candidates.py \
        --features turb_autocorr,chan_freq_coh_curve_std \
        --features turb_autocorr,chan_freq_coh_curve_std,chan_coh_gap
    python tools/replay_classic_candidates.py --json \
        --features turb_mad_over_mean,turb_autocorr,l1_delta_lag_ratio

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.lib.bootstrap import setup_paths  # noqa: E402

setup_paths()

import config  # noqa: E402
import tools.train_ml_model as train_ml_model  # noqa: E402
from classic_detector import ClassicDetector  # noqa: E402
from ml_weights import FEATURE_NAMES  # noqa: E402
from tools.fit_classic_detector import (  # noqa: E402
    IDLE_LABEL,
    MOTION_LABEL,
    balanced_sample_weights,
    choose_base_threshold,
    fit_coefficients,
    logits,
)
from tools.lib.candidate_features import CANDIDATE_FEATURES  # noqa: E402
from tools.lib.csi_io import load_npz_as_packets  # noqa: E402
from tools.lib.dataset_metadata import (  # noqa: E402
    load_dataset_info,
    measure_packet_interval_us,
    paired_dataset_role,
    resolve_entry_path,
)
from tools.lib.performance_report import (  # noqa: E402
    load_or_compute_ml_replay_rows,
    note_evaluation_tick,
    timing_cadence_for_window,
)

DISCOVERY_ROLES = ("train", "selection")
HOLDOUT_ROLE = "holdout"
PRIMARY_ROLES = DISCOVERY_ROLES + (HOLDOUT_ROLE,)
EXCLUDE_ROLE = "exclude"
REPLAY_ROLES = PRIMARY_ROLES + (EXCLUDE_ROLE,)
CURRENT_CLASSIC_COMBINATION = ("turb_autocorr", "chan_freq_coh_curve_std")
RUNTIME_READY_FEATURES = tuple(FEATURE_NAMES)
HOST_ONLY_FEATURES = tuple(CANDIDATE_FEATURES)


class ReplayError(RuntimeError):
    """Raised when the requested replay cannot run."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--features",
        action="append",
        default=[],
        help="candidate feature set as feature_a,feature_b[,feature_c]",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the full replay payload as JSON",
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=5,
        help="grouped OOF folds for the operating point (default: 5)",
    )
    parser.add_argument(
        "--fp-target",
        type=float,
        default=3.0,
        help="false-positive ceiling for the operating point (default: 3.0)",
    )
    parser.add_argument(
        "--min-session-recall",
        type=float,
        default=0.0,
        help="reject operating points whose worst train session falls below this recall",
    )
    parser.add_argument(
        "--startup-strength",
        action="append",
        type=float,
        default=[],
        help=(
            "startup calibration strength to replay; repeat for a grid "
            f"(default: {ClassicDetector.STARTUP_STRENGTH})"
        ),
    )
    parser.add_argument(
        "--settle-margin-logits",
        action="append",
        type=float,
        default=[],
        help=(
            "settled-level margin in logits; repeat for a grid "
            f"(default: {ClassicDetector.SETTLE_MARGIN_LOGITS})"
        ),
    )
    parser.add_argument(
        "--include-train-empty",
        action="store_true",
        help=(
            "include train-role empty recordings as idle hard negatives in "
            "the coefficient and operating-point fit"
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="limit printed rankings to the top K candidates (default: 20)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress per-file extraction progress",
    )
    return parser.parse_args()


def parse_feature_sets(raw_specs: Sequence[str]) -> List[Tuple[str, ...]]:
    if not raw_specs:
        raise ReplayError("Pass at least one --features combination to replay")
    parsed: List[Tuple[str, ...]] = []
    for raw in raw_specs:
        names = [part.strip() for part in raw.split(",") if part.strip()]
        if len(names) < 2 or len(names) > 3:
            raise ReplayError(
                f"Invalid --features {raw!r}; expected 2 or 3 distinct features"
            )
        if len(set(names)) != len(names):
            raise ReplayError(f"Invalid --features {raw!r}; features must differ")
        parsed.append(tuple(names))
    return parsed


def iter_replay_pairs() -> List[Dict[str, Any]]:
    files = load_dataset_info()["files"]
    motion_by_name = {entry["filename"]: entry for entry in files["motion"]}
    pairs: List[Dict[str, Any]] = []
    for static_entry in files["static_presence"]:
        if bool(static_entry.get("synthetic")):
            continue
        motion_name = static_entry.get("optimal_pair_motion_file")
        motion_entry = motion_by_name.get(motion_name) if motion_name else None
        if motion_entry is None or bool(motion_entry.get("synthetic")):
            continue
        role = paired_dataset_role(
            static_entry,
            motion_entry,
            admitted_roles=REPLAY_ROLES,
        )
        if role is None:
            continue
        pairs.append(
            {
                "session": static_entry["filename"],
                "chip": str(static_entry.get("chip", "unknown")).upper(),
                "role": role,
                "low_rssi": bool(static_entry.get("low_rssi"))
                or bool(motion_entry.get("low_rssi")),
                "static_path": resolve_entry_path("static_presence", static_entry),
                "motion_path": resolve_entry_path("motion", motion_entry),
            }
        )
    if not pairs:
        raise ReplayError("No real paired datasets found for replay")
    return pairs


def iter_empty_replays() -> List[Dict[str, Any]]:
    files = load_dataset_info()["files"]
    empties: List[Dict[str, Any]] = []
    for entry in files["empty"]:
        if bool(entry.get("synthetic")):
            continue
        role = str(entry.get("dataset_role", "train")).strip().lower() or "train"
        if role not in REPLAY_ROLES:
            continue
        empties.append(
            {
                "session": entry["filename"],
                "chip": str(entry.get("chip", "unknown")).upper(),
                "role": role,
                "path": resolve_entry_path("empty", entry),
            }
        )
    return empties


def extract_window_features(
    packets: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    extractor = train_ml_model.StreamingFeatureExtractor(feature_names)
    timing_tracker, cadence = timing_cadence_for_window(
        config.SEG_WINDOW_SIZE,
        measure_packet_interval_us(packets),
    )
    rows: List[Sequence[float]] = []
    deoverlapped: List[bool] = []
    since_window = 0
    for packet in packets:
        should_evaluate, contaminated = note_evaluation_tick(
            cadence,
            packet=packet,
            timing_tracker=timing_tracker,
        )
        if contaminated:
            extractor = train_ml_model.StreamingFeatureExtractor(feature_names)
            cadence.reset()
            timing_tracker.reset()
            should_evaluate, _ = note_evaluation_tick(
                cadence,
                packet=packet,
                timing_tracker=timing_tracker,
            )
            since_window = 0
        values = extractor.process_packet(packet["csi_data"])
        since_window += 1
        if not should_evaluate or values is None:
            continue
        rows.append(values)
        deoverlapped.append(since_window >= config.SEG_WINDOW_SIZE)
        if since_window >= config.SEG_WINDOW_SIZE:
            since_window = 0
    return (
        np.asarray(rows, dtype=np.float64).reshape(-1, len(feature_names)),
        np.asarray(deoverlapped, dtype=bool),
    )


def build_replay_cache(
    paths: Iterable[Path],
    feature_names: Sequence[str],
    *,
    quiet: bool,
) -> Dict[str, Dict[str, np.ndarray]]:
    unique_paths = sorted({str(path) for path in paths})
    cache: Dict[str, Dict[str, np.ndarray]] = {}
    runtime_ready = all(name in RUNTIME_READY_FEATURES for name in feature_names)
    for index, path_text in enumerate(unique_paths, start=1):
        path = Path(path_text)
        if not quiet:
            print(f"  [{index}/{len(unique_paths)}] {path.name}", flush=True)
        if runtime_ready:
            replay_rows = load_or_compute_ml_replay_rows(
                path,
                selected_subcarriers=config.DEFAULT_SUBCARRIERS,
                window_size=config.SEG_WINDOW_SIZE,
                feature_names=feature_names,
                sample_contract="replay_tick",
            )
            rows = np.asarray(replay_rows["X"], dtype=np.float64)
            packet_index = np.asarray(replay_rows["packet_index"], dtype=np.int64)
            reset_index = np.asarray(replay_rows["reset_index"], dtype=np.int64)
            deoverlapped = np.zeros(len(rows), dtype=bool)
            last_boundary_by_reset: Dict[int, int] = {}
            for row_index, (packet, reset) in enumerate(zip(packet_index, reset_index)):
                last_boundary = last_boundary_by_reset.get(int(reset))
                if (
                    last_boundary is None
                    or int(packet) - last_boundary >= config.SEG_WINDOW_SIZE
                ):
                    deoverlapped[row_index] = True
                    last_boundary_by_reset[int(reset)] = int(packet)
        else:
            rows, deoverlapped = extract_window_features(
                load_npz_as_packets(path),
                feature_names,
            )
        cache[path_text] = {
            "rows": rows,
            "deoverlapped": deoverlapped,
        }
    return cache


def build_training_corpus(
    pairs: Sequence[Mapping[str, Any]],
    cache: Mapping[str, Mapping[str, np.ndarray]],
    feature_index: Mapping[str, int],
    combination: Sequence[str],
) -> Dict[str, np.ndarray]:
    cols = [feature_index[name] for name in combination]
    features: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    sessions: List[np.ndarray] = []
    chips: List[np.ndarray] = []
    deoverlapped: List[np.ndarray] = []
    for pair in pairs:
        if pair["role"] != "train":
            continue
        for path_key, label in (
            (str(pair["static_path"]), IDLE_LABEL),
            (str(pair["motion_path"]), MOTION_LABEL),
        ):
            rows = np.asarray(cache[path_key]["rows"][:, cols], dtype=np.float64)
            if rows.size == 0:
                continue
            features.append(rows)
            labels.append(np.full(len(rows), label, dtype=np.int8))
            sessions.append(np.full(len(rows), pair["session"], dtype=object))
            chips.append(np.full(len(rows), pair["chip"], dtype=object))
            deoverlapped.append(np.asarray(cache[path_key]["deoverlapped"], dtype=bool))
    if not features:
        raise ReplayError(f"Replay produced no train rows for {tuple(combination)!r}")
    return {
        "x": np.vstack(features),
        "y": np.concatenate(labels),
        "session": np.concatenate(sessions),
        "chip": np.concatenate(chips),
        "deoverlapped": np.concatenate(deoverlapped),
    }


def append_training_empty_rows(
    corpus: Mapping[str, np.ndarray],
    empties: Sequence[Mapping[str, Any]],
    cache: Mapping[str, Mapping[str, np.ndarray]],
    feature_index: Mapping[str, int],
    combination: Sequence[str],
) -> Dict[str, np.ndarray]:
    """Append train-role empty streams as grouped idle hard negatives."""
    cols = [feature_index[name] for name in combination]
    features = [np.asarray(corpus["x"], dtype=np.float64)]
    labels = [np.asarray(corpus["y"], dtype=np.int8)]
    sessions = [np.asarray(corpus["session"], dtype=object)]
    chips = [np.asarray(corpus["chip"], dtype=object)]
    deoverlapped = [np.asarray(corpus["deoverlapped"], dtype=bool)]
    for empty in empties:
        if empty["role"] != "train":
            continue
        path_key = str(empty["path"])
        rows = np.asarray(cache[path_key]["rows"][:, cols], dtype=np.float64)
        if rows.size == 0:
            continue
        features.append(rows)
        labels.append(np.full(len(rows), IDLE_LABEL, dtype=np.int8))
        sessions.append(
            np.full(len(rows), empty["session"], dtype=object)
        )
        chips.append(np.full(len(rows), empty["chip"], dtype=object))
        deoverlapped.append(
            np.asarray(cache[path_key]["deoverlapped"], dtype=bool)
        )
    return {
        "x": np.vstack(features),
        "y": np.concatenate(labels),
        "session": np.concatenate(sessions),
        "chip": np.concatenate(chips),
        "deoverlapped": np.concatenate(deoverlapped),
    }


def probability(logit: float) -> float:
    if logit < -20.0:
        return 0.0
    if logit > 20.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-logit))


def startup_evaluation_limit(
    calibration_packets: int,
    window_packets: int,
    evaluation_interval: int,
    sample_limit: int,
) -> int:
    """Return how many ready evaluations production startup can observe.

    The detector is evaluated throughout calibration, but it cannot emit a
    feature row before one full window is ready. At nominal cadence this is 37
    rows, not ``CALIBRATION_BUFFER_SIZE / SEG_WINDOW_SIZE`` (10) and not the
    detector's storage cap (64).
    """
    calibration_packets = max(0, int(calibration_packets))
    window_packets = max(1, int(window_packets))
    evaluation_interval = max(1, int(evaluation_interval))
    sample_limit = max(0, int(sample_limit))
    first_ready_tick = (
        (window_packets + evaluation_interval - 1) // evaluation_interval
    ) * evaluation_interval
    if first_ready_tick > calibration_packets:
        return 0
    available = 1 + (
        calibration_packets - first_ready_tick
    ) // evaluation_interval
    return min(sample_limit, available)


def session_centered_replay_scores(
    scores: np.ndarray,
    y: np.ndarray,
    sessions: np.ndarray,
    startup_strength: float,
    startup_sample_limit: int,
) -> np.ndarray:
    """Center dense OOF logits with the same quiet-prefix rule as replay."""
    centered = np.asarray(scores, dtype=np.float64).copy()
    sessions = np.asarray(sessions, dtype=object)
    for session_name in np.unique(sessions):
        session_mask = sessions == session_name
        idle_scores = scores[session_mask & (y == IDLE_LABEL)][
            :startup_sample_limit
        ]
        if idle_scores.size == 0:
            continue
        session_q95 = float(
            np.quantile(idle_scores, ClassicDetector.STARTUP_QUANTILE)
        )
        centered[session_mask] -= float(startup_strength) * session_q95
    return centered


def dense_out_of_fold_logits(
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    fit_weights: np.ndarray,
    fit_sessions: np.ndarray,
    dense_x: np.ndarray,
    dense_sessions: np.ndarray,
    splits: int,
) -> Optional[np.ndarray]:
    """Fit folds on de-overlapped rows, then score every held-out runtime tick."""
    from sklearn.model_selection import StratifiedGroupKFold

    if len(np.unique(fit_sessions)) < splits:
        return None
    oof = np.full(len(dense_x), np.nan, dtype=np.float64)
    splitter = StratifiedGroupKFold(
        n_splits=splits,
        shuffle=True,
        random_state=0,
    )
    for train_index, test_index in splitter.split(
        fit_x,
        fit_y,
        groups=fit_sessions,
    ):
        fold = fit_coefficients(
            fit_x[train_index],
            fit_y[train_index],
            fit_weights[train_index],
        )
        held_out_sessions = np.unique(fit_sessions[test_index])
        dense_test = np.isin(dense_sessions, held_out_sessions)
        oof[dense_test] = logits(dense_x[dense_test], fold)
    return oof if not np.isnan(oof).any() else None


def startup_threshold(
    series_logits: np.ndarray,
    base_threshold: float,
    idle_q95: float,
    startup_strength: float,
    startup_sample_limit: int,
) -> float:
    prefix_count = min(len(series_logits), int(startup_sample_limit))
    if prefix_count <= 0:
        return float(base_threshold)
    startup_q95 = float(
        np.quantile(series_logits[:prefix_count], ClassicDetector.STARTUP_QUANTILE)
    )
    base_logit = float(np.log(base_threshold / (1.0 - base_threshold)))
    adapted_logit = base_logit + float(startup_strength) * (
        startup_q95 - idle_q95
    )
    return probability(adapted_logit)


def replay_one_stream(
    rows: np.ndarray,
    coefficients: Dict[str, Any],
    base_threshold: float,
    idle_q95: float,
    *,
    startup_strength: float,
    startup_sample_limit: int,
    settle_margin_logits: float,
    initial_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    if rows.size == 0:
        return {"eval_count": 0, "positive_count": 0, "positive_rate": 0.0}
    series_logits = logits(rows, coefficients)
    threshold = (
        startup_threshold(
            series_logits,
            base_threshold,
            idle_q95,
            startup_strength,
            startup_sample_limit,
        )
        if initial_threshold is None
        else float(initial_threshold)
    )
    settle_blocks: List[float] = []
    block_max = -1e9
    block_count = 0
    positive_count = 0
    for logit_value in series_logits:
        if logit_value > block_max:
            block_max = float(logit_value)
        block_count += 1
        if block_count >= ClassicDetector.SETTLE_BLOCK_EVALUATIONS:
            settle_blocks.append(block_max)
            if len(settle_blocks) > ClassicDetector.SETTLE_BLOCKS:
                settle_blocks.pop(0)
            block_max = -1e9
            block_count = 0
            if len(settle_blocks) >= ClassicDetector.SETTLE_BLOCKS:
                settled_logit = sorted(settle_blocks)[len(settle_blocks) // 2]
                settled_threshold = probability(
                    settled_logit + float(settle_margin_logits)
                )
                if settled_threshold < threshold:
                    threshold = settled_threshold
        if probability(float(logit_value)) > threshold:
            positive_count += 1
    eval_count = int(len(series_logits))
    return {
        "eval_count": eval_count,
        "positive_count": positive_count,
        "positive_rate": 100.0 * positive_count / eval_count if eval_count else 0.0,
        "threshold": float(threshold),
    }


def replay_one_pair(
    static_rows: np.ndarray,
    motion_rows: np.ndarray,
    coefficients: Dict[str, Any],
    base_threshold: float,
    idle_q95: float,
    *,
    startup_strength: float,
    startup_sample_limit: int,
    settle_margin_logits: float,
) -> Dict[str, Any]:
    initial_threshold = startup_threshold(
        logits(static_rows, coefficients),
        base_threshold,
        idle_q95,
        startup_strength,
        startup_sample_limit,
    )
    static_metrics = replay_one_stream(
        static_rows,
        coefficients,
        base_threshold,
        idle_q95,
        startup_strength=startup_strength,
        startup_sample_limit=startup_sample_limit,
        settle_margin_logits=settle_margin_logits,
        initial_threshold=initial_threshold,
    )
    motion_metrics = replay_one_stream(
        motion_rows,
        coefficients,
        base_threshold,
        idle_q95,
        startup_strength=startup_strength,
        startup_sample_limit=startup_sample_limit,
        settle_margin_logits=settle_margin_logits,
        initial_threshold=initial_threshold,
    )
    return {
        "static": static_metrics,
        "motion": motion_metrics,
        "initial_threshold": float(initial_threshold),
    }


def aggregate_paired(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "weighted_recall": float("nan"),
            "weighted_fp_rate": float("nan"),
            "mean_recall": float("nan"),
            "mean_fp_rate": float("nan"),
            "worst_recall": float("nan"),
            "worst_low_rssi_recall": float("nan"),
            "max_fp_rate": float("nan"),
            "worst_recall_session": None,
            "max_fp_session": None,
        }
    recall_weight_total = sum(int(row["motion_eval_count"]) for row in rows)
    fp_weight_total = sum(int(row["static_eval_count"]) for row in rows)
    weighted_recall = (
        sum(float(row["recall"]) * int(row["motion_eval_count"]) for row in rows)
        / recall_weight_total
        if recall_weight_total
        else float("nan")
    )
    weighted_fp_rate = (
        sum(float(row["fp_rate"]) * int(row["static_eval_count"]) for row in rows)
        / fp_weight_total
        if fp_weight_total
        else float("nan")
    )
    worst_recall_row = min(rows, key=lambda row: float(row["recall"]))
    max_fp_row = max(rows, key=lambda row: float(row["fp_rate"]))
    low_rssi_rows = [row for row in rows if bool(row["low_rssi"])]
    worst_low_rssi = (
        min(float(row["recall"]) for row in low_rssi_rows)
        if low_rssi_rows
        else float("nan")
    )
    return {
        "count": len(rows),
        "weighted_recall": float(weighted_recall),
        "weighted_fp_rate": float(weighted_fp_rate),
        "mean_recall": float(np.mean([float(row["recall"]) for row in rows])),
        "mean_fp_rate": float(np.mean([float(row["fp_rate"]) for row in rows])),
        "worst_recall": float(worst_recall_row["recall"]),
        "worst_low_rssi_recall": float(worst_low_rssi),
        "max_fp_rate": float(max_fp_row["fp_rate"]),
        "worst_recall_session": str(worst_recall_row["session"]),
        "max_fp_session": str(max_fp_row["session"]),
    }


def aggregate_idle(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "mean_fp_rate": float("nan"),
            "max_fp_rate": float("nan"),
            "worst_session": None,
        }
    worst_row = max(rows, key=lambda row: float(row["fp_rate"]))
    return {
        "count": len(rows),
        "mean_fp_rate": float(np.mean([float(row["fp_rate"]) for row in rows])),
        "max_fp_rate": float(worst_row["fp_rate"]),
        "worst_session": str(worst_row["session"]),
    }


def replay_score(primary_pairs: Mapping[str, Any], primary_idle: Mapping[str, Any]) -> float:
    penalties = 0.0
    penalties += max(0.0, 95.0 - float(primary_pairs["worst_recall"])) * 3.0
    low_rssi = float(primary_pairs["worst_low_rssi_recall"])
    if not math.isnan(low_rssi):
        penalties += max(0.0, 85.0 - low_rssi) * 4.0
    penalties += max(0.0, float(primary_pairs["weighted_fp_rate"]) - 3.0) * 2.0
    penalties += max(0.0, float(primary_idle["max_fp_rate"]) - 6.0) * 4.0
    return penalties


def evaluate_candidate(
    combination: Sequence[str],
    pairs: Sequence[Mapping[str, Any]],
    empties: Sequence[Mapping[str, Any]],
    cache: Mapping[str, Mapping[str, np.ndarray]],
    feature_index: Mapping[str, int],
    args: argparse.Namespace,
    *,
    startup_strength: float,
    settle_margin_logits: float,
) -> Dict[str, Any]:
    combination = tuple(combination)
    runtime_ready = all(name in RUNTIME_READY_FEATURES for name in combination)
    corpus = build_training_corpus(pairs, cache, feature_index, combination)
    include_train_empty = bool(
        getattr(args, "include_train_empty", False)
    )
    if include_train_empty:
        corpus = append_training_empty_rows(
            corpus,
            empties,
            cache,
            feature_index,
            combination,
        )
    x, y = corpus["x"], corpus["y"]
    deoverlapped = np.asarray(corpus["deoverlapped"], dtype=bool)
    fit_x, fit_y = x[deoverlapped], y[deoverlapped]
    fit_weights = balanced_sample_weights(
        fit_y,
        corpus["chip"][deoverlapped],
        corpus["session"][deoverlapped],
    )
    coefficients = fit_coefficients(fit_x, fit_y, fit_weights)
    all_weights = balanced_sample_weights(y, corpus["chip"], corpus["session"])
    oof = dense_out_of_fold_logits(
        fit_x,
        fit_y,
        fit_weights,
        corpus["session"][deoverlapped],
        x,
        corpus["session"],
        args.splits,
    )
    score_source = (
        f"grouped de-overlapped fit / dense OOF score ({args.splits} folds)"
    )
    if oof is None:
        score_source = "in-sample"
        oof = logits(x, coefficients)
    idle_logits = logits(fit_x[fit_y == IDLE_LABEL], coefficients)
    idle_q95 = float(np.quantile(idle_logits, ClassicDetector.STARTUP_QUANTILE))
    startup_sample_limit = startup_evaluation_limit(
        config.CALIBRATION_BUFFER_SIZE,
        config.SEG_WINDOW_SIZE,
        config.EVALUATION_INTERVAL,
        ClassicDetector.STARTUP_SAMPLE_LIMIT,
    )
    centered = session_centered_replay_scores(
        oof,
        y,
        corpus["session"],
        startup_strength,
        startup_sample_limit,
    )
    centered_threshold, train_metrics = choose_base_threshold(
        centered,
        y,
        all_weights,
        session=corpus["session"],
        fp_target=args.fp_target,
        min_session_recall=args.min_session_recall,
    )
    centered_logit = float(np.log(centered_threshold / (1.0 - centered_threshold)))
    base_threshold = probability(
        centered_logit + float(startup_strength) * idle_q95
    )

    cols = [feature_index[name] for name in combination]
    paired_rows: List[Dict[str, Any]] = []
    for pair in pairs:
        static_rows = np.asarray(
            cache[str(pair["static_path"])]["rows"][:, cols],
            dtype=np.float64,
        )
        motion_rows = np.asarray(
            cache[str(pair["motion_path"])]["rows"][:, cols],
            dtype=np.float64,
        )
        replay_metrics = replay_one_pair(
            static_rows,
            motion_rows,
            coefficients,
            base_threshold,
            idle_q95,
            startup_strength=startup_strength,
            startup_sample_limit=startup_sample_limit,
            settle_margin_logits=settle_margin_logits,
        )
        static_metrics = replay_metrics["static"]
        motion_metrics = replay_metrics["motion"]
        paired_rows.append(
            {
                "session": pair["session"],
                "role": pair["role"],
                "chip": pair["chip"],
                "low_rssi": bool(pair["low_rssi"]),
                "static_eval_count": int(static_metrics["eval_count"]),
                "motion_eval_count": int(motion_metrics["eval_count"]),
                "fp_rate": float(static_metrics["positive_rate"]),
                "recall": float(motion_metrics["positive_rate"]),
                "initial_threshold": float(replay_metrics["initial_threshold"]),
                "static_threshold": float(static_metrics["threshold"]),
                "motion_threshold": float(motion_metrics["threshold"]),
            }
        )
    idle_rows: List[Dict[str, Any]] = []
    for empty in empties:
        rows = np.asarray(cache[str(empty["path"])]["rows"][:, cols], dtype=np.float64)
        metrics = replay_one_stream(
            rows,
            coefficients,
            base_threshold,
            idle_q95,
            startup_strength=startup_strength,
            startup_sample_limit=startup_sample_limit,
            settle_margin_logits=settle_margin_logits,
        )
        idle_rows.append(
            {
                "session": empty["session"],
                "role": empty["role"],
                "chip": empty["chip"],
                "fp_rate": float(metrics["positive_rate"]),
                "eval_count": int(metrics["eval_count"]),
                "threshold": float(metrics["threshold"]),
            }
        )

    discovery_pairs = [
        row for row in paired_rows if row["role"] in DISCOVERY_ROLES
    ]
    holdout_pairs = [
        row for row in paired_rows if row["role"] == HOLDOUT_ROLE
    ]
    exclude_pairs = [row for row in paired_rows if row["role"] == EXCLUDE_ROLE]
    discovery_idle = [
        row for row in idle_rows if row["role"] in DISCOVERY_ROLES
    ]
    holdout_idle = [
        row for row in idle_rows if row["role"] == HOLDOUT_ROLE
    ]
    exclude_idle = [row for row in idle_rows if row["role"] == EXCLUDE_ROLE]
    discovery_pair_summary = aggregate_paired(discovery_pairs)
    discovery_idle_summary = aggregate_idle(discovery_idle)
    holdout_pair_summary = aggregate_paired(holdout_pairs)
    holdout_idle_summary = aggregate_idle(holdout_idle)
    exclude_pair_summary = aggregate_paired(exclude_pairs)
    exclude_idle_summary = aggregate_idle(exclude_idle)
    return {
        "combination": list(combination),
        "combination_size": len(combination),
        "runtime_ready": runtime_ready,
        "score": replay_score(
            discovery_pair_summary,
            discovery_idle_summary,
        ),
        "startup_strength": float(startup_strength),
        "startup_sample_limit": int(startup_sample_limit),
        "settle_margin_logits": float(settle_margin_logits),
        "train_empty_hard_negatives": include_train_empty,
        "coefficients": {
            "center": [float(value) for value in coefficients["center"]],
            "scale": [float(value) for value in coefficients["scale"]],
            "weight": [float(value) for value in coefficients["weight"]],
            "intercept": float(coefficients["intercept"]),
        },
        "base_threshold": float(base_threshold),
        "train_idle_q95_logit": float(idle_q95),
        "train_operating_point": dict(train_metrics, source=score_source),
        "discovery": {
            "roles": list(DISCOVERY_ROLES),
            "paired": discovery_pair_summary,
            "idle": discovery_idle_summary,
        },
        "holdout": {
            "role": HOLDOUT_ROLE,
            "paired": holdout_pair_summary,
            "idle": holdout_idle_summary,
        },
        "exclude": {
            "paired": exclude_pair_summary,
            "idle": exclude_idle_summary,
        },
        "replay_rows": {
            "paired": paired_rows,
            "idle": idle_rows,
        },
    }


def print_summary(result: Mapping[str, Any]) -> None:
    discovery_pairs = result["discovery"]["paired"]
    discovery_idle = result["discovery"]["idle"]
    holdout_pairs = result["holdout"]["paired"]
    holdout_idle = result["holdout"]["idle"]
    exclude_pairs = result["exclude"]["paired"]
    exclude_idle = result["exclude"]["idle"]
    bucket = "runtime-ready" if result["runtime_ready"] else "host-only"
    print(
        f"#{result['rank']}  {' + '.join(result['combination'])}  "
        f"[{bucket}]  score={result['score']:.3f}  "
        f"startup={result['startup_strength']:.3f}  "
        f"settle_margin={result['settle_margin_logits']:.3f}"
    )
    print(
        "  "
        f"train OOF recall={result['train_operating_point']['recall']:.2f}%  "
        f"fp={result['train_operating_point']['fp_rate']:.2f}%  "
        f"base={result['base_threshold']:.4f}"
    )
    print(
        "  "
        f"discovery recall weighted={discovery_pairs['weighted_recall']:.2f}%  "
        f"worst={discovery_pairs['worst_recall']:.2f}%  "
        f"low_rssi_worst={discovery_pairs['worst_low_rssi_recall']:.2f}%"
    )
    print(
        "  "
        f"discovery fp weighted={discovery_pairs['weighted_fp_rate']:.2f}%  "
        f"paired max={discovery_pairs['max_fp_rate']:.2f}%  "
        f"idle max={discovery_idle['max_fp_rate']:.2f}%"
    )
    if holdout_pairs["count"] or holdout_idle["count"]:
        print(
            "  "
            f"holdout recall weighted={holdout_pairs['weighted_recall']:.2f}%  "
            f"worst={holdout_pairs['worst_recall']:.2f}%  "
            f"fp weighted={holdout_pairs['weighted_fp_rate']:.2f}%  "
            f"paired max={holdout_pairs['max_fp_rate']:.2f}%  "
            f"idle max={holdout_idle['max_fp_rate']:.2f}%"
        )
    if exclude_pairs["count"] or exclude_idle["count"]:
        print(
            "  "
            f"exclude recall weighted={exclude_pairs['weighted_recall']:.2f}%  "
            f"fp weighted={exclude_pairs['weighted_fp_rate']:.2f}%  "
            f"idle max={exclude_idle['max_fp_rate']:.2f}%"
        )


def main() -> int:
    args = parse_args()
    if args.top_k < 1:
        raise ReplayError("--top-k must be at least 1")
    startup_strengths = (
        args.startup_strength
        if args.startup_strength
        else [ClassicDetector.STARTUP_STRENGTH]
    )
    settle_margins = (
        args.settle_margin_logits
        if args.settle_margin_logits
        else [ClassicDetector.SETTLE_MARGIN_LOGITS]
    )
    if any(value < 0.0 or value > 1.0 for value in startup_strengths):
        raise ReplayError("--startup-strength must be between 0 and 1")
    if any(value < 0.0 for value in settle_margins):
        raise ReplayError("--settle-margin-logits must be non-negative")
    candidates = parse_feature_sets(args.features)
    candidate_set = {tuple(candidate) for candidate in candidates}
    candidate_set.add(CURRENT_CLASSIC_COMBINATION)
    available = set(train_ml_model.selectable_features())
    unknown = sorted(
        {name for candidate in candidate_set for name in candidate}.difference(available)
    )
    if unknown:
        raise ReplayError("Unknown feature(s) requested: " + ", ".join(unknown))

    feature_surface = sorted(
        {name for candidate in candidate_set for name in candidate}
    )
    pairs = iter_replay_pairs()
    empties = iter_empty_replays()
    paths = [pair["static_path"] for pair in pairs]
    paths.extend(pair["motion_path"] for pair in pairs)
    paths.extend(entry["path"] for entry in empties)
    runtime_candidates = [
        candidate
        for candidate in candidate_set
        if all(name in RUNTIME_READY_FEATURES for name in candidate)
    ]
    runtime_surface = sorted(
        {name for candidate in runtime_candidates for name in candidate}
    )
    print(
        f"Extracting runtime replay rows for {len(paths)} files and "
        f"{len(runtime_surface)} features",
        flush=True,
    )
    runtime_cache = build_replay_cache(
        paths,
        runtime_surface,
        quiet=args.quiet,
    )
    runtime_index = {
        name: index for index, name in enumerate(runtime_surface)
    }
    host_bundles: Dict[
        Tuple[str, ...],
        Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, int]],
    ] = {}
    for candidate in candidate_set:
        if candidate in runtime_candidates:
            continue
        candidate_surface = list(candidate)
        print(
            f"Extracting host-only replay rows for {' + '.join(candidate)}",
            flush=True,
        )
        host_bundles[candidate] = (
            build_replay_cache(
                paths,
                candidate_surface,
                quiet=args.quiet,
            ),
            {
                name: index
                for index, name in enumerate(candidate_surface)
            },
        )
    results = []
    for candidate in sorted(candidate_set):
        if candidate in runtime_candidates:
            candidate_cache = runtime_cache
            candidate_index = runtime_index
        else:
            candidate_cache, candidate_index = host_bundles[candidate]
        for startup_strength, settle_margin_logits in product(
            startup_strengths,
            settle_margins,
        ):
            results.append(
                evaluate_candidate(
                    candidate,
                    pairs,
                    empties,
                    candidate_cache,
                    candidate_index,
                    args,
                    startup_strength=startup_strength,
                    settle_margin_logits=settle_margin_logits,
                )
            )
    ranked = sorted(results, key=lambda row: row["score"])
    baseline_rows = []
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
        if tuple(row["combination"]) == CURRENT_CLASSIC_COMBINATION:
            baseline_rows.append(row)
    payload = {
        "feature_surface": feature_surface,
        "runtime_ready_feature_surface": list(RUNTIME_READY_FEATURES),
        "host_only_feature_surface": list(HOST_ONLY_FEATURES),
        "baseline_combination": list(CURRENT_CLASSIC_COMBINATION),
        "baseline": baseline_rows,
        "discovery_roles": list(DISCOVERY_ROLES),
        "holdout_role": HOLDOUT_ROLE,
        "train_empty_hard_negatives": bool(args.include_train_empty),
        "candidates": ranked,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    if baseline_rows:
        print("\nRefitted feature-pair surrogate (not the exported runtime baseline):")
        for row in baseline_rows:
            print_summary(row)
        print()
    print("Candidate replay ranking:")
    for row in ranked[: args.top_k]:
        print_summary(row)
        print()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReplayError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
