#!/usr/bin/env python3
"""
ESPectre - Classic Detector Coefficient Fit

Refits the Classic detector's weighted `l1_delta + turb_autocorr` fusion and
exports the constants consumed by both runtimes.

The fit follows the recipe recorded alongside the previous coefficients: a
grouped, de-overlapped out-of-fold fit balanced by class, chip, and session.

- grouped: folds split on session, so windows from one recording never appear on
  both sides of a split
- de-overlapped: one feature vector per full window, so consecutive samples share
  no packets and the fit is not fed a smoothed random walk
- balanced: sample weights equalize class, chip, and session mass, so the long
  recordings and the better represented chips do not dominate

Only `train` datasets are fitted. `holdout` stays sealed, `selection` is left for
operating-point work, and `exclude` is dropped.

Usage:
    python tools/fit_classic_detector.py
    python tools/fit_classic_detector.py --apply

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.lib.bootstrap import setup_paths  # noqa: E402

setup_paths()

import config  # noqa: E402
from classic_detector import ClassicDetector  # noqa: E402
from tools.lib.csi_io import load_npz_as_packets  # noqa: E402
from tools.lib.dataset_metadata import (  # noqa: E402
    derive_detector_timing,
    load_dataset_info,
    measure_packet_interval_us,
    resolve_entry_path,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON_SOURCE = REPO_ROOT / "src" / "python" / "micro_espectre" / "classic_detector.py"
CPP_SOURCE = REPO_ROOT / "src" / "cpp" / "core" / "classic_detector.h"

FITTED_ROLES = ("train",)
IDLE_LABEL = 0
MOTION_LABEL = 1


class FitError(RuntimeError):
    """Raised when the corpus or the fit cannot produce usable coefficients."""


def iter_training_pairs() -> List[Dict[str, Any]]:
    """Collect paired static-presence and motion recordings used for the fit."""
    files = load_dataset_info()["files"]
    motion_by_name = {entry["filename"]: entry for entry in files["motion"]}

    pairs: List[Dict[str, Any]] = []
    for static_entry in files["static_presence"]:
        role = static_entry.get("dataset_role")
        if role not in FITTED_ROLES:
            continue
        motion_name = static_entry.get("optimal_pair_motion_file")
        motion_entry = motion_by_name.get(motion_name) if motion_name else None
        if motion_entry is None or motion_entry.get("dataset_role") == "exclude":
            continue
        pairs.append(
            {
                "session": static_entry["filename"],
                "chip": str(static_entry.get("chip", "unknown")),
                "static_path": resolve_entry_path("static_presence", static_entry),
                "motion_path": resolve_entry_path("motion", motion_entry),
            }
        )
    if not pairs:
        raise FitError("No train-role paired datasets found")
    return pairs


def extract_window_features(
    packets: Sequence[Dict[str, Any]],
    selected_band: Sequence[int],
    window_size: int,
) -> np.ndarray:
    """Replay one recording at the runtime evaluation cadence.

    Returns one feature row per evaluation, plus a mask marking the rows that are
    de-overlapped. The two consumers need different sampling: the coefficient fit
    wants de-overlapped rows so consecutive samples share no packets and it is
    not fed a smoothed random walk, while the operating-point sweep wants every
    evaluation, because that is what the runtime thresholds. Sweeping on the
    de-overlapped subset understates false positives, since a brief excursion
    above the threshold spans several consecutive evaluations but only one
    de-overlapped window.

    Features come from the production detector rather than a reimplementation, so
    the fit can never drift from what the runtime computes. That includes the
    timing contract: the detector is sized from this recording's measured
    cadence exactly as the runtime sizes it, because the lags define what the
    two features mean and fitting them under a different cadence would produce
    coefficients for a feature the runtime never computes.
    """
    timing = derive_detector_timing(measure_packet_interval_us(packets))
    detector = ClassicDetector(
        window_size=timing["window_packets"],
        lag=timing["lag"],
        autocorr_lag=timing["autocorr_lag"],
    )
    window = timing["window_packets"]
    cadence = max(1, timing["evaluation_interval"])
    rows: List[Tuple[float, float]] = []
    deoverlapped: List[bool] = []
    since_evaluation = 0
    since_window = 0
    for packet in packets:
        detector.process_packet(packet["csi_data"], selected_band)
        since_evaluation += 1
        since_window += 1
        if since_evaluation < cadence or not detector.is_ready():
            continue
        since_evaluation = 0
        metrics = detector.update_state()
        rows.append((metrics["l1_delta"], metrics["turb_autocorr"]))
        deoverlapped.append(since_window >= window)
        if since_window >= window:
            since_window = 0
    return (
        np.asarray(rows, dtype=np.float64).reshape(-1, 2),
        np.asarray(deoverlapped, dtype=bool),
    )


def build_corpus(
    pairs: Sequence[Dict[str, Any]],
    selected_band: Sequence[int],
    window_size: int,
    progress: bool = True,
) -> Dict[str, np.ndarray]:
    """Replay every training pair into one labelled, grouped feature matrix."""
    features: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    sessions: List[np.ndarray] = []
    chips: List[np.ndarray] = []
    deoverlapped: List[np.ndarray] = []

    for index, pair in enumerate(pairs, start=1):
        if progress:
            print(f"  [{index}/{len(pairs)}] {pair['session'][:64]}", flush=True)
        for path, label in ((pair["static_path"], IDLE_LABEL), (pair["motion_path"], MOTION_LABEL)):
            rows, row_deoverlapped = extract_window_features(
                load_npz_as_packets(path), selected_band, window_size
            )
            if rows.size == 0:
                continue
            features.append(rows)
            labels.append(np.full(len(rows), label, dtype=np.int8))
            sessions.append(np.full(len(rows), pair["session"], dtype=object))
            chips.append(np.full(len(rows), pair["chip"], dtype=object))
            deoverlapped.append(row_deoverlapped)

    if not features:
        raise FitError("Replay produced no windows; check the corpus and window size")

    return {
        "x": np.vstack(features),
        "y": np.concatenate(labels),
        "session": np.concatenate(sessions),
        "chip": np.concatenate(chips),
        "deoverlapped": np.concatenate(deoverlapped),
    }


def balanced_sample_weights(
    y: np.ndarray,
    chip: np.ndarray,
    session: np.ndarray,
) -> np.ndarray:
    """Equalize class, chip, and session mass so no stratum dominates the fit."""
    weights = np.ones(len(y), dtype=np.float64)
    for key in (y, chip, session):
        values, counts = np.unique(key, return_counts=True)
        lookup = {value: count for value, count in zip(values, counts)}
        weights *= np.asarray([1.0 / lookup[value] for value in key], dtype=np.float64)
    return weights * (len(y) / weights.sum())


def fit_coefficients(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> Dict[str, Any]:
    """Standardize the features and fit the two-feature logistic fusion."""
    from sklearn.linear_model import LogisticRegression

    center = np.average(x, axis=0, weights=weights)
    scale = np.sqrt(np.average((x - center) ** 2, axis=0, weights=weights))
    if not np.all(scale > 0.0):
        raise FitError(f"Degenerate feature scale {scale}; the corpus has no variation")

    standardized = (x - center) / scale
    model = LogisticRegression(max_iter=1000)
    model.fit(standardized, y, sample_weight=weights)

    # Plain Python floats: NumPy scalars repr as "np.float64(...)", which would be
    # written verbatim into both runtimes by the exporter.
    return {
        "center": tuple(float(value) for value in center),
        "scale": tuple(float(value) for value in scale),
        "weight": tuple(float(value) for value in model.coef_[0]),
        "intercept": float(model.intercept_[0]),
    }


def logits(x: np.ndarray, coefficients: Dict[str, Any]) -> np.ndarray:
    """Evaluate the fused logit exactly as both runtimes do."""
    standardized = (x - np.asarray(coefficients["center"])) / np.asarray(coefficients["scale"])
    return coefficients["intercept"] + standardized @ np.asarray(coefficients["weight"])


def out_of_fold_logits(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    session: np.ndarray,
    splits: int,
) -> Optional[np.ndarray]:
    """Compute grouped out-of-fold logits so the operating point is not in-sample."""
    from sklearn.model_selection import StratifiedGroupKFold

    groups = np.unique(session)
    if len(groups) < splits:
        return None

    oof = np.full(len(y), np.nan, dtype=np.float64)
    splitter = StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=0)
    for train_index, test_index in splitter.split(x, y, groups=session):
        fold = fit_coefficients(x[train_index], y[train_index], weights[train_index])
        oof[test_index] = logits(x[test_index], fold)
    return oof if not np.isnan(oof).any() else None


def session_centered_scores(
    scores: np.ndarray,
    y: np.ndarray,
    sessions: Sequence[str],
    window_size: int,
) -> np.ndarray:
    """Re-express logits the way the runtime compares them.

    Startup calibration does not threshold the raw logit. It shifts the
    threshold by the session's own quiet level, so the effective decision is

        logit - STARTUP_STRENGTH * session_q95
            > base_logit - STARTUP_STRENGTH * TRAIN_IDLE_Q95_LOGIT

    Choosing an operating point on the raw logit therefore prices in a session
    shift of zero, which no session actually has. The left-hand side is the
    quantity the runtime compares, so it is the one the fit has to sweep.

    ``session_q95`` mirrors the runtime: the quantile is taken over the quiet
    prefix the calibration buffer covers, not over the whole quiet segment,
    because the runtime only ever sees that prefix and its noise.
    """
    prefix_windows = max(1, config.CALIBRATION_BUFFER_SIZE // max(1, window_size))
    sessions = np.asarray(sessions)
    centered = np.array(scores, dtype=np.float64)
    for session in np.unique(sessions):
        in_session = sessions == session
        idle_scores = scores[in_session & (y == IDLE_LABEL)][:prefix_windows]
        if idle_scores.size == 0:
            continue
        session_q95 = float(
            np.quantile(idle_scores, ClassicDetector.STARTUP_QUANTILE)
        )
        centered[in_session] -= ClassicDetector.STARTUP_STRENGTH * session_q95
    return centered


def base_threshold_from_centered(centered_threshold: float, idle_q95: float) -> float:
    """Convert a session-centered operating point back to BASE_THRESHOLD."""
    base_logit = centered_threshold + ClassicDetector.STARTUP_STRENGTH * idle_q95
    return float(1.0 / (1.0 + np.exp(-base_logit)))


def choose_base_threshold(
    scores: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    session: Optional[np.ndarray] = None,
    fp_target: float = 3.0,
    min_session_recall: float = 0.0,
) -> Tuple[float, Dict[str, float]]:
    """Pick the operating point the production promotion gate actually asks for.

    The gate is a false-positive ceiling with recall maximized underneath it, not
    best F1. Maximizing F1 lands well above the ceiling, because F1 trades recall
    against precision on equal terms while the runtime contract does not. Falls
    back to best F1 only when no candidate clears the ceiling.

    Every rate here is pooled over the whole corpus, and a pooled rate hides the
    one recording that fails. A single-feature experiment scored `98.0%` recall
    on this sweep while one capture sat at `62%`, because twenty-six good pairs
    outvoted it; the per-pair replay caught what the sweep could not. So the
    chosen point now also carries `worst_session_recall`, and `min_session_recall`
    can make it binding. Report it, and do not read the pooled recall alone.
    """
    probabilities = 1.0 / (1.0 + np.exp(-scores))
    candidates = np.unique(np.quantile(probabilities, np.linspace(0.001, 0.999, 999)))

    gated_threshold: Optional[float] = None
    gated_recall = -1.0
    gated: Dict[str, float] = {}

    best_threshold = 0.5
    best_f1 = -1.0
    best: Dict[str, float] = {}
    for threshold in candidates:
        predicted = probabilities > threshold
        tp = float(weights[(y == MOTION_LABEL) & predicted].sum())
        fn = float(weights[(y == MOTION_LABEL) & ~predicted].sum())
        fp = float(weights[(y == IDLE_LABEL) & predicted].sum())
        tn = float(weights[(y == IDLE_LABEL) & ~predicted].sum())
        if tp <= 0.0:
            continue
        recall = tp / (tp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0.0 else 0.0
        if precision <= 0.0:
            continue
        f1 = 2.0 * precision * recall / (precision + recall)
        fp_rate = 100.0 * fp / (fp + tn) if (fp + tn) > 0.0 else 0.0
        worst_session_recall = 100.0 * recall
        if session is not None:
            motion = y == MOTION_LABEL
            per_session = []
            for name in np.unique(session[motion]):
                rows = motion & (session == name)
                hit = float(weights[rows & predicted].sum())
                total = float(weights[rows].sum())
                if total > 0.0:
                    per_session.append(100.0 * hit / total)
            if per_session:
                worst_session_recall = float(min(per_session))
        point = {
            "f1": 100.0 * f1,
            "recall": 100.0 * recall,
            "fp_rate": fp_rate,
            "worst_session_recall": worst_session_recall,
        }

        if worst_session_recall < min_session_recall:
            continue

        if fp_rate <= fp_target and recall > gated_recall:
            gated_recall = recall
            gated_threshold = float(threshold)
            gated = point
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
            best = point

    if gated_threshold is not None:
        gated["gate"] = fp_target
        return gated_threshold, gated
    if best_f1 < 0.0:
        raise FitError("No usable operating point found")
    print(
        f"WARNING: no operating point holds false positives at or under {fp_target}%; "
        "falling back to best F1",
        file=sys.stderr,
    )
    return best_threshold, best


def replace_assignment(text: str, pattern: str, replacement: str, label: str) -> str:
    """Replace exactly one assignment, refusing to guess when the anchor moved."""
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise FitError(f"Could not locate {label}; update the exporter alongside the source")
    return updated


def render_python(coefficients: Dict[str, Any], base_threshold: float, idle_q95: float) -> str:
    """Rewrite the Python detector constants in place."""
    center, scale = coefficients["center"], coefficients["scale"]
    weight, intercept = coefficients["weight"], coefficients["intercept"]
    text = PYTHON_SOURCE.read_text()
    for pattern, replacement, label in (
        (r"^    FEATURE_CENTER = .*$", f"    FEATURE_CENTER = ({center[0]!r}, {center[1]!r})", "FEATURE_CENTER"),
        (r"^    FEATURE_SCALE = .*$", f"    FEATURE_SCALE = ({scale[0]!r}, {scale[1]!r})", "FEATURE_SCALE"),
        (r"^    FEATURE_WEIGHT = .*$", f"    FEATURE_WEIGHT = ({weight[0]!r}, {weight[1]!r})", "FEATURE_WEIGHT"),
        (r"^    INTERCEPT = .*$", f"    INTERCEPT = {intercept!r}", "INTERCEPT"),
        (r"^    BASE_THRESHOLD = .*$", f"    BASE_THRESHOLD = {base_threshold!r}", "BASE_THRESHOLD"),
        (r"^    TRAIN_IDLE_Q95_LOGIT = .*$", f"    TRAIN_IDLE_Q95_LOGIT = {idle_q95!r}", "TRAIN_IDLE_Q95_LOGIT"),
    ):
        text = replace_assignment(text, pattern, replacement, label)
    return text


def render_cpp(coefficients: Dict[str, Any], base_threshold: float, idle_q95: float) -> str:
    """Rewrite the C++ detector constants in place."""
    center, scale = coefficients["center"], coefficients["scale"]
    weight, intercept = coefficients["weight"], coefficients["intercept"]
    text = CPP_SOURCE.read_text()
    for name, value in (
        ("CLASSIC_L1_CENTER", center[0]),
        ("CLASSIC_L1_SCALE", scale[0]),
        ("CLASSIC_L1_WEIGHT", weight[0]),
        ("CLASSIC_AUTOCORR_CENTER", center[1]),
        ("CLASSIC_AUTOCORR_SCALE", scale[1]),
        ("CLASSIC_AUTOCORR_WEIGHT", weight[1]),
        ("CLASSIC_INTERCEPT", intercept),
        ("CLASSIC_DEFAULT_THRESHOLD", base_threshold),
        ("CLASSIC_TRAIN_IDLE_Q95_LOGIT", idle_q95),
    ):
        text = replace_assignment(
            text,
            rf"^constexpr float {name} = .*$",
            f"constexpr float {name} = {float(value)!r}f;",
            name,
        )
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="write the constants into both runtimes")
    parser.add_argument("--splits", type=int, default=5, help="grouped OOF folds (default: 5)")
    parser.add_argument("--fp-target", type=float, default=3.0,
                        help="false-positive ceiling for the operating point (default: 3.0)")
    parser.add_argument("--min-session-recall", type=float, default=0.0,
                        help="reject operating points whose worst single session falls below "
                             "this recall (default: 0.0, report only)")
    parser.add_argument("--quiet", action="store_true", help="suppress per-dataset progress")
    args = parser.parse_args()

    selected_band = tuple(config.DEFAULT_SUBCARRIERS)
    window_size = config.SEG_WINDOW_SIZE

    pairs = iter_training_pairs()
    print(f"Fitting Classic on {len(pairs)} train pairs (band={selected_band}, window={window_size})")
    corpus = build_corpus(pairs, selected_band, window_size, progress=not args.quiet)

    x, y = corpus["x"], corpus["y"]
    deoverlapped = corpus["deoverlapped"]
    weights = balanced_sample_weights(y, corpus["chip"], corpus["session"])
    print(
        f"\nEvaluations: {len(y)} ({int((y == MOTION_LABEL).sum())} motion, "
        f"{int((y == IDLE_LABEL).sum())} idle); "
        f"{int(deoverlapped.sum())} de-overlapped windows fit the coefficients"
    )

    # Coefficients come from the de-overlapped subset so consecutive samples
    # share no packets; the operating point is swept over every evaluation
    # below, because that is the cadence the runtime thresholds at.
    fit_x, fit_y = x[deoverlapped], y[deoverlapped]
    fit_weights = balanced_sample_weights(
        fit_y, corpus["chip"][deoverlapped], corpus["session"][deoverlapped]
    )
    coefficients = fit_coefficients(fit_x, fit_y, fit_weights)

    oof = out_of_fold_logits(x, y, weights, corpus["session"], args.splits)
    if oof is None:
        print("WARNING: grouped OOF unavailable; operating point is in-sample", file=sys.stderr)
        oof = logits(x, coefficients)
        source = "in-sample"
    else:
        source = f"grouped OOF ({args.splits} folds)"
    idle_logits = logits(fit_x[fit_y == IDLE_LABEL], coefficients)
    idle_q95 = float(np.quantile(idle_logits, ClassicDetector.STARTUP_QUANTILE))

    # Sweep the quantity the runtime compares, then convert the chosen point
    # back into the constant the runtime stores.
    centered = session_centered_scores(oof, y, corpus["session"], window_size)
    centered_threshold, metrics = choose_base_threshold(
        centered,
        y,
        weights,
        session=corpus["session"],
        fp_target=args.fp_target,
        min_session_recall=args.min_session_recall,
    )
    centered_logit = float(np.log(centered_threshold / (1.0 - centered_threshold)))
    base_threshold = base_threshold_from_centered(centered_logit, idle_q95)

    center, scale = coefficients["center"], coefficients["scale"]
    weight, intercept = coefficients["weight"], coefficients["intercept"]
    print(f"\nOperating point from {source}:")
    print(f"  F1={metrics['f1']:.3f}%  recall={metrics['recall']:.3f}%  fp_rate={metrics['fp_rate']:.3f}%")
    worst_session = metrics.get("worst_session_recall")
    if worst_session is not None:
        print(f"  worst session recall={worst_session:.3f}%")
        if metrics["recall"] - worst_session > 10.0:
            print(
                f"WARNING: pooled recall {metrics['recall']:.1f}% hides a session at "
                f"{worst_session:.1f}%; check the per-pair replay before trusting this point",
                file=sys.stderr,
            )
    print("\nFitted constants:")
    print(f"  FEATURE_CENTER       = ({center[0]!r}, {center[1]!r})")
    print(f"  FEATURE_SCALE        = ({scale[0]!r}, {scale[1]!r})")
    print(f"  FEATURE_WEIGHT       = ({weight[0]!r}, {weight[1]!r})")
    print(f"  INTERCEPT            = {intercept!r}")
    print(f"  BASE_THRESHOLD       = {base_threshold!r}")
    print(f"  TRAIN_IDLE_Q95_LOGIT = {idle_q95!r}")

    if not args.apply:
        print("\nDry run; pass --apply to write both runtimes.")
        return 0

    python_text = render_python(coefficients, base_threshold, idle_q95)
    cpp_text = render_cpp(coefficients, base_threshold, idle_q95)
    PYTHON_SOURCE.write_text(python_text)
    CPP_SOURCE.write_text(cpp_text)
    print(f"\nWrote {PYTHON_SOURCE.relative_to(REPO_ROOT)} and {CPP_SOURCE.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
