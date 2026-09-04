#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - Adjacent-Subcarrier Aggregation Benchmark

Research-only harness that measures what averaging adjacent bins into each of
the twelve selected subcarriers does to the detectors. It never writes runtime
artifacts.

Aggregation is injected by replacing the production amplitude-buffer fill for
the duration of a run, so the whole runtime chain replays unchanged behind it
and the features come from the production detectors rather than from a
reimplementation. Only the twelve-tone path can move: the channel-shape and
coherence features read the 56-bin live complex profile and are bit-identical
under aggregation, which every run re-checks in `features` mode.

Modes:
    channel     per-tone noise, adjacent-bin coherence, and the predicted
                signal-to-noise gain, using no detection metric
    classic     Lightweight per-pair separability across group widths, with the
                fusion coefficients refit per configuration
    features    per-feature effect across the production eight-feature set
    candidates  dispersion and order statistics of the turbulence series,
                including candidates retired before this evidence existed

Usage:
    python tools/benchmark_subcarrier_aggregation.py --mode channel
    python tools/benchmark_subcarrier_aggregation.py --mode classic
    python tools/benchmark_subcarrier_aggregation.py --mode classic --coherent
    python tools/benchmark_subcarrier_aggregation.py --mode candidates --json out.json

Author: Francesco Pace <francesco.pace@gmail.com>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.lib.bootstrap import setup_paths  # noqa: E402

setup_paths()

import config  # noqa: E402
from tools.lib.lightweight_detector import LightweightDetector  # noqa: E402
from tools.lib.high_accuracy_detector import HighAccuracyDetector  # noqa: E402
from tools.lib.temporal_csi_sampler import minimum_valid_slots, temporal_window_slots  # noqa: E402
from tools.lib.ml_weights import FEATURE_NAMES  # noqa: E402
from tools.fit_lightweight_detector import (  # noqa: E402
    balanced_sample_weights,
    build_corpus,
    fit_coefficients,
    iter_training_pairs,
    logits,
)
from tools.lib.csi_io import load_npz_as_packets, load_npz_csi_data  # noqa: E402
from tools.lib.dataset_metadata import (  # noqa: E402
    derive_detector_timing,
    load_dataset_info,
    measure_packet_interval_us,
    resolve_entry_path,
)
from tools.lib.adjacent_aggregation import aggregated_amplitudes  # noqa: E402
from tools.lib.performance_report import temporal_detector_ticks  # noqa: E402
from tools.lib.temporal_replay import target_pps_for_packets  # noqa: E402

LIVE_BINS: Tuple[int, ...] = tuple(range(4, 32)) + tuple(range(33, 61))
DEFAULT_WIDTHS: Tuple[int, ...] = (2, 3, 5)
CORRELATION_LAGS: Tuple[int, ...] = (1, 2, 3, 4, 5, 10, 14)

# Features that read the twelve-tone amplitude buffer. The rest come from the
# live complex profile and must not move under aggregation.
BUFFER_FED_FEATURES = frozenset({
    "turb_mad_over_mean",
    "turb_autocorr",
    "turb_zcr",
    "l1_delta_lag_ratio",
})

CANDIDATE_NAMES: Tuple[str, ...] = (
    "turb_mad_over_mean",
    "turb_cv",
    "turb_iqr_over_mean",
    "turb_p95_over_mean",
    "turb_p05_over_mean",
    "turb_max_over_mean",
    "turb_min_over_mean",
    "turb_range_over_mean",
    "turb_peak_over_mad",
    "waveform_length_over_mean",
    "turb_skewness",
    "corr_amp_d1",
)

def configurations(widths: Sequence[int], coherent: bool) -> List[Tuple[str, Any]]:
    """Baseline first, then one entry per requested group width."""
    kind = "coherent" if coherent else "magnitude"
    return [("baseline", None)] + [(f"W={w} {kind}", w) for w in widths]


# =============================================================================
# Metrics
# =============================================================================
def auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Rank-based AUC with ties sharing the average rank."""
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    values = np.concatenate([positive, negative])
    order = values.argsort()
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    rank_sum = ranks[: len(positive)].sum()
    return float(
        (rank_sum - len(positive) * (len(positive) + 1) / 2)
        / (len(positive) * len(negative))
    )


def separation(values: np.ndarray) -> np.ndarray:
    """Discrimination strength regardless of polarity.

    A feature at AUC 0.00 separates as well as one at 1.00, and the production
    set contains both, so raw AUC is not comparable across features.
    """
    return np.maximum(values, 1.0 - values)


def report_paired(
    names: Sequence[str],
    baseline: np.ndarray,
    candidate: np.ndarray,
    title: str,
) -> None:
    """Print median, worst pair, and whether the worst pair is the same one.

    The median saturates near 1.0 for most of these features, so the worst pair
    carries the evidence. It is only a paired comparison when the limiting
    recording is the same in both configurations, which is reported per row
    because it often is not.
    """
    base, cand = separation(baseline), separation(candidate)
    print(f"\n{title}")
    print(
        f"{'feature':<32}{'med base':>9}{'med cand':>9}{'worst base':>11}"
        f"{'worst cand':>11}{'same pair':>10}{'mean delta':>11}"
    )
    print("-" * 93)
    for i, name in enumerate(names):
        same = base[:, i].argmin() == cand[:, i].argmin()
        print(
            f"{name:<32}{np.median(base[:, i]):>9.4f}{np.median(cand[:, i]):>9.4f}"
            f"{base[:, i].min():>11.4f}{cand[:, i].min():>11.4f}"
            f"{str(same):>10}{(cand[:, i] - base[:, i]).mean():>+11.4f}"
        )


# =============================================================================
# Mode: channel statistics
# =============================================================================
def live_amplitudes(csi: np.ndarray) -> np.ndarray:
    """(packets, 56) live-band magnitudes from a raw int8 CSI matrix."""
    iq = csi.reshape(csi.shape[0], -1, 2).astype(np.float64)
    complex_profile = iq[:, :, 1] + 1j * iq[:, :, 0]
    return np.abs(complex_profile[:, list(LIVE_BINS)])


def fast_fluctuation(series: np.ndarray) -> np.ndarray:
    """Packet-to-packet fluctuation: the lag-1 first difference over sqrt(2).

    The turbulence autocorrelation runs at lag 1 packet, so this is the band
    that matters, not slow session drift.
    """
    return np.diff(series, axis=0) / math.sqrt(2.0)


def adjacent_correlation(fluctuation: np.ndarray, lag: int) -> float:
    """Mean correlation between tones `lag` bins apart, skipping the DC gap."""
    values = []
    for i in range(fluctuation.shape[1] - lag):
        if LIVE_BINS[i + lag] - LIVE_BINS[i] != lag:
            continue
        left, right = fluctuation[:, i], fluctuation[:, i + lag]
        left_sd, right_sd = left.std(), right.std()
        if left_sd > 0 and right_sd > 0:
            values.append(
                float(
                    ((left - left.mean()) * (right - right.mean())).mean()
                    / (left_sd * right_sd)
                )
            )
    return float(np.mean(values)) if values else float("nan")


def run_channel(args: argparse.Namespace) -> Dict[str, Any]:
    """Measure the noise and the coherence that decide whether averaging helps."""
    files = load_dataset_info()["files"]
    quiet_entries = [("empty", e) for e in files["empty"][: args.limit]]
    motion_entries = [("motion", e) for e in files["motion"][: args.limit]]

    raw_quiet: List[float] = []
    correlations: Dict[str, Dict[int, List[float]]] = {"quiet": {}, "motion": {}}
    scaled: Dict[str, List[float]] = {"quiet": [], "motion": []}
    for lag in CORRELATION_LAGS:
        correlations["quiet"][lag] = []
        correlations["motion"][lag] = []

    for label, entries in (("quiet", quiet_entries), ("motion", motion_entries)):
        for index, (folder, entry) in enumerate(entries, 1):
            csi = load_npz_csi_data(resolve_entry_path(folder, entry))
            if csi.shape[1] != 128 or csi.shape[0] < 400:
                continue
            amplitude = live_amplitudes(csi)
            packet_mean = amplitude.mean(axis=1, keepdims=True)
            normalized = np.divide(
                amplitude, packet_mean, out=np.zeros_like(amplitude), where=packet_mean > 0
            )
            fluctuation = fast_fluctuation(normalized)
            scaled[label].append(float(fluctuation.std(axis=0).mean()))
            if label == "quiet":
                raw = fast_fluctuation(amplitude / amplitude.mean(axis=0, keepdims=True))
                raw_quiet.append(float(raw.std(axis=0).mean()))
            for lag in CORRELATION_LAGS:
                correlations[label][lag].append(adjacent_correlation(fluctuation, lag))
            if args.progress:
                print(f"  [{label} {index}/{len(entries)}]", end="\r", flush=True)

    raw = float(np.nanmean(raw_quiet))
    quiet = float(np.nanmean(scaled["quiet"]))
    motion = float(np.nanmean(scaled["motion"]))
    print("\nPacket-to-packet per-tone fluctuation (relative units)")
    print(f"  quiet, common-mode gain included : {raw:.5f}")
    print(f"  quiet, common-mode gain removed  : {quiet:.5f}")
    print(f"  motion, common-mode gain removed : {motion:.5f}")
    print(f"  gain share of raw quiet jitter   : {1 - quiet / raw:.1%}")
    print("\nThe gain share is common to every tone, so no cross-tone average")
    print("removes it; the production features already discard it by being")
    print("scale-invariant, so only the remaining residual is in play.")

    print("\nAdjacent-bin correlation of the gain-removed fast fluctuation")
    print(f"{'lag (sc)':>9}{'MHz':>8}{'quiet':>9}{'motion':>9}")
    curve: Dict[str, Dict[int, float]] = {"quiet": {}, "motion": {}}
    for lag in CORRELATION_LAGS:
        curve["quiet"][lag] = float(np.nanmean(correlations["quiet"][lag]))
        curve["motion"][lag] = float(np.nanmean(correlations["motion"][lag]))
        print(
            f"{lag:>9}{lag * 0.3125:>8.2f}"
            f"{curve['quiet'][lag]:>9.3f}{curve['motion'][lag]:>9.3f}"
        )

    print("\nPredicted effect of averaging W adjacent bins, from the measured curve")
    print(f"{'W':>3}{'ideal noise':>13}{'noise':>9}{'signal':>9}{'SNR gain':>10}")
    predicted: Dict[int, Dict[str, float]] = {}
    for width in args.widths:
        if any(lag not in curve["quiet"] for lag in range(1, width)):
            continue

        def factor(kind: str, w: int = width) -> float:
            total = w + 2 * sum((w - lag) * curve[kind][lag] for lag in range(1, w))
            return float(math.sqrt(total) / w)

        noise, signal = factor("quiet"), factor("motion")
        predicted[width] = {"noise": noise, "signal": signal, "snr_gain": signal / noise}
        print(
            f"{width:>3}{1 / math.sqrt(width):>13.3f}{noise:>9.3f}"
            f"{signal:>9.3f}{signal / noise:>10.3f}"
        )

    return {
        "quiet_raw": raw,
        "quiet_scaled": quiet,
        "motion_scaled": motion,
        "correlation": {k: {str(a): b for a, b in v.items()} for k, v in curve.items()},
        "predicted": {str(k): v for k, v in predicted.items()},
    }


# =============================================================================
# Mode: Lightweight
# =============================================================================
def run_classic(args: argparse.Namespace) -> Dict[str, Any]:
    """Score Lightweight per-pair separability across group widths.

    Coefficients are refit per configuration, because scoring a changed input
    under the baseline's coefficients measures the mismatch rather than the
    change.
    """
    pairs = iter_training_pairs()
    band = list(config.DEFAULT_SUBCARRIERS)
    print(f"train pairs: {len(pairs)}")

    results: Dict[str, Any] = {}
    per_config: List[Tuple[str, np.ndarray, np.ndarray]] = []
    for label, width in configurations(args.widths, args.coherent):
        with aggregated_amplitudes(width, args.coherent):
            corpus = build_corpus(pairs, band, progress=False)

        features, labels = corpus["x"], corpus["y"]
        session, chip = corpus["session"], corpus["chip"]
        deoverlapped = corpus["deoverlapped"]
        weights = balanced_sample_weights(
            labels[deoverlapped], chip[deoverlapped], session[deoverlapped]
        )
        coefficients = fit_coefficients(
            features[deoverlapped], labels[deoverlapped], weights
        )
        fused = logits(features, coefficients)

        turb_rows, fused_rows = [], []
        for name in np.unique(session):
            mask = session == name
            positive, negative = mask & (labels == 1), mask & (labels == 0)
            if positive.sum() == 0 or negative.sum() == 0:
                continue
            turb_rows.append(auc(features[positive, 0], features[negative, 0]))
            fused_rows.append(auc(fused[positive], fused[negative]))

        turb = separation(np.asarray(turb_rows))
        fusion = separation(np.asarray(fused_rows))
        per_config.append((label, turb, fusion))
        results[label] = {
            "turb_autocorr": {"median": float(np.median(turb)), "worst": float(turb.min())},
            "fused": {"median": float(np.median(fusion)), "worst": float(fusion.min())},
        }
        print(
            f"{label:<20} turb_autocorr med={np.median(turb):.4f} worst={turb.min():.4f}"
            f" | fused med={np.median(fusion):.4f} worst={fusion.min():.4f}"
        )

    base_turb, base_fused = per_config[0][1], per_config[0][2]
    print("\nPaired per-pair delta against baseline (positive favours aggregation)")
    print(f"{'configuration':<20}{'turb mean':>11}{'turb wins':>11}"
          f"{'fused mean':>12}{'fused wins':>12}")
    for label, turb, fusion in per_config[1:]:
        turb_delta, fused_delta = turb - base_turb, fusion - base_fused
        print(
            f"{label:<20}{turb_delta.mean():>+11.4f}"
            f"{f'{(turb_delta > 0).sum()}/{len(turb_delta)}':>11}"
            f"{fused_delta.mean():>+12.4f}"
            f"{f'{(fused_delta > 0).sum()}/{len(fused_delta)}':>12}"
        )
    return results


# =============================================================================
# Mode: production feature set
# =============================================================================
def ml_feature_rows(packets: Sequence[Dict[str, Any]], band: Sequence[int]) -> np.ndarray:
    """Replay one recording and collect the production feature vectors.

    Reads the detector's own extractor rather than reassembling the feature
    vector here, so the trackers, filters, and timing stay production behaviour.
    """
    interval_us = measure_packet_interval_us(packets)
    target_pps = target_pps_for_packets(packets, interval_us)
    timing = derive_detector_timing(max(1, int(round(1_000_000.0 / target_pps))))
    timing["window_packets"] = temporal_window_slots(
        target_pps, config.SEGMENTATION_WINDOW_SIZE_MS
    )
    detector = HighAccuracyDetector(
        window_size=timing["window_packets"],
        lag=timing["lag"],
        autocorr_lag=timing["autocorr_lag"],
    )
    detector.set_minimum_valid_samples(
        minimum_valid_slots(timing["window_packets"])
    )
    rows = []
    for admission, should_evaluate, _ in temporal_detector_ticks(
        detector, packets, interval_us
    ):
        packet = admission.packet
        detector.process_packet(packet["csi_data"], band)
        if not should_evaluate or not detector.is_ready():
            continue
        rows.append(list(detector._extract_features()))
    return np.asarray(rows, dtype=np.float64).reshape(-1, len(FEATURE_NAMES))


def run_features(args: argparse.Namespace) -> Dict[str, Any]:
    """Score every production feature under one group width."""
    width = args.widths[0]
    pairs = iter_training_pairs()
    band = list(config.DEFAULT_SUBCARRIERS)

    scores: Dict[str, np.ndarray] = {}
    for label, configured in configurations([width], args.coherent):
        rows = []
        with aggregated_amplitudes(configured, args.coherent):
            for index, pair in enumerate(pairs, 1):
                quiet = ml_feature_rows(load_npz_as_packets(pair["static_path"]), band)
                motion = ml_feature_rows(load_npz_as_packets(pair["motion_path"]), band)
                if len(quiet) == 0 or len(motion) == 0:
                    continue
                rows.append(
                    [auc(motion[:, i], quiet[:, i]) for i in range(len(FEATURE_NAMES))]
                )
                if args.progress:
                    print(f"  [{label}] {index}/{len(pairs)}", end="\r", flush=True)
        scores[label] = np.asarray(rows)

    baseline, candidate = scores["baseline"], scores[list(scores)[1]]
    report_paired(FEATURE_NAMES, baseline, candidate, f"Production feature set, W={width}")

    untouched = [n for n in FEATURE_NAMES if n not in BUFFER_FED_FEATURES]
    moved = [
        name
        for name in untouched
        if not np.allclose(
            baseline[:, FEATURE_NAMES.index(name)],
            candidate[:, FEATURE_NAMES.index(name)],
        )
    ]
    if moved:
        print(f"\nWARNING: full-width features moved under aggregation: {moved}")
        print("They read the live complex profile and must be unaffected; the")
        print("injection reached further than the twelve-tone buffer.")
    else:
        print(f"\nSelf-check passed: the {len(untouched)} full-width features are")
        print("bit-identical, so the injection reached only the twelve-tone buffer.")
    return {
        name: {
            "baseline": float(np.median(separation(baseline)[:, i])),
            "candidate": float(np.median(separation(candidate)[:, i])),
        }
        for i, name in enumerate(FEATURE_NAMES)
    }


# =============================================================================
# Mode: turbulence-series candidates
# =============================================================================
def candidate_values(turbulence: np.ndarray, profiles: np.ndarray) -> List[float]:
    """Dispersion and order statistics of one turbulence window.

    `turb_mad_over_mean` is the historical production reference for this
    screen: it must reproduce the `features` mode result, which is what makes
    the retired candidates around it trustworthy. It uses the true median
    absolute deviation; the mean absolute deviation is a different statistic
    and does not reproduce the historical value.
    """
    count = len(turbulence)
    mean = float(turbulence.mean())
    if mean <= 0.0 or count < 4:
        return [0.0] * len(CANDIDATE_NAMES)

    median = float(np.median(turbulence))
    mad = float(np.median(np.abs(turbulence - median)))
    q05, q25, q75, q95 = (float(v) for v in np.percentile(turbulence, [5, 25, 75, 95]))
    sd = float(turbulence.std())
    high, low = float(turbulence.max()), float(turbulence.min())
    waveform_length = float(np.abs(np.diff(turbulence)).mean())
    skewness = float(((turbulence - mean) ** 3).mean() / sd**3) if sd > 0 else 0.0

    # corr_amp_d1: mean correlation between consecutive amplitude profiles.
    # Retired because at lag 1 it mostly measured receiver noise, which makes it
    # the strongest prior candidate for benefiting from aggregation.
    current, previous = profiles[1:], profiles[:-1]
    current = current - current.mean(axis=1, keepdims=True)
    previous = previous - previous.mean(axis=1, keepdims=True)
    denominator = np.sqrt((current * current).sum(axis=1) * (previous * previous).sum(axis=1))
    correlation = np.divide(
        (current * previous).sum(axis=1),
        denominator,
        out=np.zeros(len(current)),
        where=denominator > 0,
    )

    return [
        mad / mean,
        sd / mean,
        (q75 - q25) / mean,
        q95 / mean,
        q05 / mean,
        high / mean,
        low / mean,
        (high - low) / mean,
        (high - mean) / mad if mad > 0 else 0.0,
        waveform_length / mean,
        skewness,
        float(correlation.mean()),
    ]


def candidate_rows(packets: Sequence[Dict[str, Any]], band: Sequence[int]) -> np.ndarray:
    """Replay one recording and score the candidates on each evaluated window.

    The turbulence series and the amplitude profiles come from the unmodified
    production path; only the statistics computed over them are defined here.
    """
    interval_us = measure_packet_interval_us(packets)
    target_pps = target_pps_for_packets(packets, interval_us)
    timing = derive_detector_timing(max(1, int(round(1_000_000.0 / target_pps))))
    timing["window_packets"] = temporal_window_slots(
        target_pps, config.SEGMENTATION_WINDOW_SIZE_MS
    )
    window = timing["window_packets"]
    detector = LightweightDetector(
        window_size=window, autocorr_lag=timing["autocorr_lag"]
    )
    detector.set_minimum_valid_samples(minimum_valid_slots(window))
    context = detector._context
    history: List[List[float]] = []
    rows = []
    for admission, should_evaluate, _ in temporal_detector_ticks(
        detector, packets, interval_us
    ):
        packet = admission.packet
        if admission.reset_required:
            history.clear()
        detector.process_packet(packet["csi_data"], band)
        history.append(list(context._amplitude_buffer[: context._amplitude_count]))
        if len(history) > window:
            del history[0]
        if not should_evaluate or not detector.is_ready():
            continue

        count = context.buffer_count
        if count < context.window_size:
            turbulence = np.asarray(context.turbulence_buffer[:count], dtype=np.float64)
        else:
            # The ring is full, so the oldest sample sits at the write index.
            turbulence = np.asarray(
                [
                    context.turbulence_buffer[(context.buffer_index + i) % count]
                    for i in range(count)
                ],
                dtype=np.float64,
            )
        profiles = np.asarray(history, dtype=np.float64)
        if profiles.ndim != 2 or profiles.shape[0] < 2:
            continue
        rows.append(candidate_values(turbulence, profiles))
    return np.asarray(rows, dtype=np.float64).reshape(-1, len(CANDIDATE_NAMES))


def run_candidates(args: argparse.Namespace) -> Dict[str, Any]:
    """Score the turbulence-series candidates, retired ones included."""
    width = args.widths[0]
    pairs = iter_training_pairs()
    band = list(config.DEFAULT_SUBCARRIERS)

    scores: Dict[str, np.ndarray] = {}
    for label, configured in configurations([width], args.coherent):
        rows = []
        with aggregated_amplitudes(configured, args.coherent):
            for index, pair in enumerate(pairs, 1):
                quiet = candidate_rows(load_npz_as_packets(pair["static_path"]), band)
                motion = candidate_rows(load_npz_as_packets(pair["motion_path"]), band)
                if len(quiet) == 0 or len(motion) == 0:
                    continue
                rows.append(
                    [auc(motion[:, i], quiet[:, i]) for i in range(len(CANDIDATE_NAMES))]
                )
                if args.progress:
                    print(f"  [{label}] {index}/{len(pairs)}", end="\r", flush=True)
        scores[label] = np.asarray(rows)

    baseline, candidate = scores["baseline"], scores[list(scores)[1]]
    report_paired(
        CANDIDATE_NAMES, baseline, candidate, f"Turbulence-series candidates, W={width}"
    )
    print("\n`turb_mad_over_mean` is the reference row: compare it against the")
    print("`features` mode result. If the two disagree, the statistics defined")
    print("here have drifted from the production feature and the retired")
    print("candidates around them are not trustworthy.")
    return {
        name: {
            "baseline_worst": float(separation(baseline)[:, i].min()),
            "candidate_worst": float(separation(candidate)[:, i].min()),
        }
        for i, name in enumerate(CANDIDATE_NAMES)
    }


# =============================================================================
# Entry point
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark adjacent-subcarrier aggregation on the 12-tone path",
    )
    parser.add_argument(
        "--mode",
        choices=("channel", "classic", "features", "candidates"),
        default="classic",
        help="which measurement to run (default: classic)",
    )
    parser.add_argument(
        "--widths",
        type=int,
        nargs="+",
        default=list(DEFAULT_WIDTHS),
        help="group widths in bins; features and candidates modes use the first",
    )
    parser.add_argument(
        "--coherent",
        action="store_true",
        help="average complex values instead of magnitudes",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="channel mode only: recordings per class (default: 20)",
    )
    parser.add_argument("--json", type=Path, help="also write the results as JSON")
    parser.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="suppress per-recording progress output",
    )
    args = parser.parse_args()
    if any(w < 2 for w in args.widths):
        parser.error("group widths must be at least 2; the baseline is always included")
    return args


def main() -> int:
    args = parse_args()
    runners = {
        "channel": run_channel,
        "classic": run_classic,
        "features": run_features,
        "candidates": run_candidates,
    }
    results = runners[args.mode](args)
    if args.json:
        args.json.write_text(json.dumps({"mode": args.mode, "results": results}, indent=1))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
