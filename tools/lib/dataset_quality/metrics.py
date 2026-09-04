# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Detector-agnostic dataset quality metrics and scores."""

from . import core
import numpy as np

from config import DEFAULT_SUBCARRIERS
from tools.lib.csi_io import load_npz_packet_view
from tools.lib.occupancy_thinning import mean_window_occupancy
from tools.lib.performance_report import (
    build_ml_replay_rows,
    load_or_compute_ml_replay_rows,
)
from .severity import (
    BASELINE_BLOCK_SECONDS,
    BASELINE_EXCURSION_MADS,
    BASELINE_LONGEST_BURST_ZERO_SECONDS,
    CLASSIC_SCORE_MOTION_FULL,
    CLASSIC_SCORE_SEPARATION_FULL,
    CLASSIC_SCORE_SEPARATION_ZERO,
    CLASSIC_SCORE_TAIL_FULL,
    CLASSIC_SCORE_TAIL_ZERO,
    FEATURE_EVIDENCE_DIRECTIONS,
    REFERENCE_BLOCK_SECONDS,
    REFERENCE_EXCURSION_EXPECTED_RATIO,
    REFERENCE_EXCURSION_ZERO_RATIO,
    REFERENCE_LONGEST_BURST_FAIL_SECONDS,
    REFERENCE_MAX_BLOCKS_PER_CAPTURE,
    VALIDATION_FEATURE_NAMES,
)

def _clamp_score(value):
    """Clamp an indicative score into [0, 100]."""
    return float(max(0.0, min(100.0, value)))


def occupancy_quality_score(mean_occupancy):
    """Return the 0-100 score ceiling imposed by temporal occupancy."""
    if mean_occupancy is None:
        return 0.0
    return round(_clamp_score(100.0 * float(mean_occupancy)), 1)


def cap_quality_score_by_occupancy(score, *occupancies):
    """Cap one quality score by every capture occupancy in its scope."""
    ceilings = [occupancy_quality_score(value) for value in occupancies]
    return round(min(float(score), *ceilings), 1) if ceilings else round(float(score), 1)


def agnostic_pair_score(motion_coverage, pair_separation):
    """Return an indicative 0-100 separation score for one pair.

    Separation is the static/motion AUC, and motion coverage is the share of the
    motion half rising above the static half's p95. Static cleanliness is not
    inferred from a self-normalized tail; it is scored independently against
    external idle references and caps the final pair quality score.
    """
    motion_cover = _clamp_score(
        100.0 * float(motion_coverage) / CLASSIC_SCORE_MOTION_FULL
    )
    separation_value = float(pair_separation)
    if not np.isfinite(separation_value):
        separation_value = CLASSIC_SCORE_SEPARATION_FULL
    separation_score = _clamp_score(
        100.0
        * (separation_value - CLASSIC_SCORE_SEPARATION_ZERO)
        / (CLASSIC_SCORE_SEPARATION_FULL - CLASSIC_SCORE_SEPARATION_ZERO)
    )
    return round(0.7 * separation_score + 0.3 * motion_cover, 1)


def reference_cleanliness_score(excursion_ratio, longest_burst_seconds):
    """Return a 0-100 idle-cleanliness score against external references."""
    excursion_clean = _clamp_score(
        100.0
        * (REFERENCE_EXCURSION_ZERO_RATIO - float(excursion_ratio))
        / (
            REFERENCE_EXCURSION_ZERO_RATIO
            - REFERENCE_EXCURSION_EXPECTED_RATIO
        )
    )
    burst_clean = _clamp_score(
        100.0
        * (
            1.0
            - float(longest_burst_seconds)
            / REFERENCE_LONGEST_BURST_FAIL_SECONDS
        )
    )
    return round(0.7 * excursion_clean + 0.3 * burst_clean, 1)


def agnostic_baseline_score(margin_q95, longest_burst_seconds):
    """Return a 0-100 within-capture stability score.

    Tail height carries most of the score. It is the capture's own q95 above its
    own median, so it does not depend on a detector threshold. External-reference
    cleanliness is required to detect a uniform session shift. This remains a
    review-only diagnostic, not a dataset-admission gate.
    """
    cleanliness = _clamp_score(
        100.0
        * (CLASSIC_SCORE_TAIL_ZERO - float(margin_q95))
        / (CLASSIC_SCORE_TAIL_ZERO - CLASSIC_SCORE_TAIL_FULL)
    )
    burst_clean = _clamp_score(
        100.0
        * (
            1.0
            - float(longest_burst_seconds)
            / BASELINE_LONGEST_BURST_ZERO_SECONDS
        )
    )
    return round(0.7 * cleanliness + 0.3 * burst_clean, 1)


classic_pair_score = agnostic_pair_score


classic_baseline_score = agnostic_baseline_score


def _mean_temporal_occupancy(packets, target_pps):
    """Return mean valid-slot occupancy across complete temporal windows."""
    if not packets or not target_pps:
        return None
    return float(
        mean_window_occupancy(
            packets,
            target_pps=max(1, int(target_pps)),
        )
    )


def _resolve_temporal_occupancy_target_pps(packets, *, fallback=None):
    """Resolve the recorded detector grid, with one legacy metadata fallback."""
    embedded = packets[0].get("csi_target_pps") if packets else None
    for candidate in (embedded, fallback):
        try:
            resolved = int(candidate)
        except (TypeError, ValueError):
            continue
        if resolved > 0:
            return resolved
    return None


def _pair_separation(baseline_scores, motion_scores):
    """Return the rank-based AUC between idle and motion probability series.

    This is the Mann-Whitney statistic: the probability that a random motion
    evaluation scores above a random idle one. It reads only the ordering of the
    two series, so it is unchanged by where the threshold sits and by any other
    monotone rescaling of the metric.
    """
    baseline_scores = np.asarray(baseline_scores, dtype=np.float64)
    motion_scores = np.asarray(motion_scores, dtype=np.float64)
    if baseline_scores.size == 0 or motion_scores.size == 0:
        return float("nan")

    combined = np.concatenate([baseline_scores, motion_scores])
    order = combined.argsort(kind="mergesort")
    ranks = np.empty(combined.size, dtype=np.float64)
    ranks[order] = np.arange(1, combined.size + 1, dtype=np.float64)

    # Ties share their average rank, otherwise a flat stretch of the metric
    # would score as separation purely from input order.
    sorted_values = combined[order]
    start = 0
    for stop in range(1, sorted_values.size + 1):
        if stop == sorted_values.size or sorted_values[stop] != sorted_values[start]:
            if stop - start > 1:
                ranks[order[start:stop]] = ranks[order[start:stop]].mean()
            start = stop

    motion_rank_sum = float(ranks[baseline_scores.size:].sum())
    motion_count = float(motion_scores.size)
    baseline_count = float(baseline_scores.size)
    return float(
        (motion_rank_sum - motion_count * (motion_count + 1.0) / 2.0)
        / (baseline_count * motion_count)
    )


def _robust_axis_location_and_scale(values):
    """Return (median, MAD floor-applied) for one feature axis."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0, 1.0
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    return center, max(mad, 1e-6)


def _feature_matrix_packets(packets, *, feature_names=None):
    """Return the canonical time-aware dense feature stream for packets."""
    rows = build_ml_replay_rows(
        tuple(packets),
        DEFAULT_SUBCARRIERS,
        None,
        feature_names=list(feature_names or VALIDATION_FEATURE_NAMES),
        sample_contract="stream_dense",
    )
    return (
        np.asarray(rows["X"], dtype=np.float64),
        tuple(rows["feature_names"]),
        {
            "slot_index": np.asarray(rows.get("slot_index", ()), dtype=np.int64),
            "reset_index": np.asarray(rows.get("reset_index", ()), dtype=np.int32),
            "target_pps": int(rows.get("target_pps", 0)),
        },
    )


def _load_or_compute_validation_feature_matrix(filepath, *, feature_names=None, use_cache=True):
    """Return the shared time-aware dense feature stream for validation."""
    requested_feature_names = tuple(feature_names or VALIDATION_FEATURE_NAMES)
    if core.DIAGNOSTIC_ALL_PHY:
        return _feature_matrix_packets(
            load_npz_packet_view(filepath, keep_all_phy=True),
            feature_names=requested_feature_names,
        )
    rows = load_or_compute_ml_replay_rows(
        filepath,
        selected_subcarriers=DEFAULT_SUBCARRIERS,
        window_size=None,
        feature_names=requested_feature_names,
        sample_contract="stream_dense",
        use_cache=use_cache,
    )
    return (
        np.asarray(rows["X"], dtype=np.float64),
        tuple(rows["feature_names"]),
        {
            "slot_index": np.asarray(rows.get("slot_index", ()), dtype=np.int64),
            "reset_index": np.asarray(rows.get("reset_index", ()), dtype=np.int32),
            "target_pps": int(rows.get("target_pps", 0)),
        },
    )


def _feature_direction_vector(feature_names):
    """Return one fixed direction per validation feature."""
    return np.asarray(
        [FEATURE_EVIDENCE_DIRECTIONS.get(name, 1.0) for name in feature_names],
        dtype=np.float64,
    )


def _feature_evidence_series(feature_matrix, *, centers=None, scales=None, directions=None):
    """Collapse per-window invariant features into one robust evidence series."""
    matrix = np.asarray(feature_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return np.asarray([], dtype=np.float64)
    width = matrix.shape[1]
    if centers is None or scales is None:
        centers = np.zeros(width, dtype=np.float64)
        scales = np.ones(width, dtype=np.float64)
        for index in range(width):
            centers[index], scales[index] = _robust_axis_location_and_scale(
                matrix[:, index]
            )
    else:
        centers = np.asarray(centers, dtype=np.float64)
        scales = np.asarray(scales, dtype=np.float64)
    directions = np.ones(width, dtype=np.float64) if directions is None else np.asarray(
        directions, dtype=np.float64
    )
    normalized = directions * ((matrix - centers) / scales)
    normalized = np.clip(normalized, -8.0, 8.0)
    return np.mean(normalized, axis=1)


def _consensus_pair_evidence(idle_matrix, motion_matrix, feature_names):
    """Return directional evidence series and idle robust axis stats."""
    idle_matrix = np.asarray(idle_matrix, dtype=np.float64)
    motion_matrix = np.asarray(motion_matrix, dtype=np.float64)
    if idle_matrix.ndim != 2 or motion_matrix.ndim != 2:
        return None
    if idle_matrix.shape[0] == 0 or motion_matrix.shape[0] == 0:
        return None
    centers = np.median(idle_matrix, axis=0)
    scales = np.zeros(idle_matrix.shape[1], dtype=np.float64)
    for index in range(idle_matrix.shape[1]):
        _center, scales[index] = _robust_axis_location_and_scale(idle_matrix[:, index])
    directions = _feature_direction_vector(feature_names)
    idle_evidence = _feature_evidence_series(
        idle_matrix, centers=centers, scales=scales, directions=directions
    )
    motion_evidence = _feature_evidence_series(
        motion_matrix, centers=centers, scales=scales, directions=directions
    )
    return idle_evidence, motion_evidence, centers, scales


def _temporal_block_medians(values, row_timing, block_seconds):
    """Return physical-time block medians and their reset segments."""
    samples = np.asarray(values, dtype=np.float64)
    row_count = samples.shape[0] if samples.ndim else 0
    if row_count == 0 or not row_timing:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.int32)
    slots = np.asarray(row_timing.get("slot_index", ()), dtype=np.int64)
    resets = np.asarray(row_timing.get("reset_index", ()), dtype=np.int32)
    target_pps = int(row_timing.get("target_pps", 0))
    if len(slots) != row_count or len(resets) != row_count or target_pps <= 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.int32)

    block_slots = max(1, int(round(target_pps * float(block_seconds))))
    medians = []
    segment_values = []
    start = 0
    while start < row_count:
        reset = int(resets[start])
        stop = start + 1
        while stop < row_count and int(resets[stop]) == reset:
            stop += 1
        relative_slots = slots[start:stop] - slots[start]
        complete_blocks = int((relative_slots[-1] + 1) // block_slots)
        for block in range(complete_blocks):
            mask = (
                (relative_slots >= block * block_slots)
                & (relative_slots < (block + 1) * block_slots)
            )
            if np.any(mask):
                medians.append(np.median(samples[start:stop][mask], axis=0))
                segment_values.append(reset)
        start = stop

    if not medians:
        return (
            np.asarray([np.median(samples, axis=0)], dtype=np.float64),
            np.asarray([int(resets[0])], dtype=np.int32),
        )
    return np.asarray(medians, dtype=np.float64), np.asarray(segment_values, dtype=np.int32)


def _temporal_coverage_seconds(row_timing, row_count):
    """Return elapsed sampler-grid coverage without compacting missing slots."""
    if row_count <= 0 or not row_timing:
        return 0.0
    slots = np.asarray(row_timing.get("slot_index", ()), dtype=np.int64)
    resets = np.asarray(row_timing.get("reset_index", ()), dtype=np.int32)
    target_pps = int(row_timing.get("target_pps", 0))
    if len(slots) != row_count or len(resets) != row_count or target_pps <= 0:
        return 0.0
    elapsed_slots = 0
    start = 0
    while start < row_count:
        reset = int(resets[start])
        stop = start + 1
        while stop < row_count and int(resets[stop]) == reset:
            stop += 1
        elapsed_slots += int(slots[stop - 1] - slots[start] + 1)
        start = stop
    return elapsed_slots / float(target_pps)


def _feature_block_medians(feature_matrix, row_timing):
    """Return contiguous five-second feature medians on the sampler grid."""
    matrix = np.asarray(feature_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        return np.asarray([], dtype=np.float64)
    blocks, _segments = _temporal_block_medians(
        matrix,
        row_timing,
        REFERENCE_BLOCK_SECONDS,
    )
    return blocks


def _sample_reference_blocks(blocks):
    """Sample a bounded, deterministic number of blocks from one capture."""
    blocks = np.asarray(blocks, dtype=np.float64)
    if len(blocks) <= REFERENCE_MAX_BLOCKS_PER_CAPTURE:
        return blocks
    indices = np.linspace(
        0,
        len(blocks) - 1,
        REFERENCE_MAX_BLOCKS_PER_CAPTURE,
    ).round().astype(np.int64)
    return blocks[indices]


def _agnostic_baseline_stats_from_series(evidence_series, row_timing):
    """Summarize one dense idle feature-evidence series.

    The canonical ``stream_dense`` matrix contributes one ready feature row per
    admitted slot, not one row per production evaluation tick. Temporal
    aggregation therefore follows the sampler slot coordinates and preserves
    missing slots instead of compacting them into the next observation.
    """
    evidence = np.asarray(evidence_series, dtype=np.float64)
    if evidence.size == 0 or not row_timing:
        return None
    margin_center = float(np.median(evidence))
    margins = evidence - margin_center
    margin_median = float(np.median(margins))
    margin_mad = float(np.median(np.abs(margins - margin_median)))
    excursion_bound = margin_median + BASELINE_EXCURSION_MADS * max(margin_mad, 1e-9)
    states = (margins > excursion_bound).astype(np.int8)

    block_margins, block_segments = _temporal_block_medians(
        margins,
        row_timing,
        BASELINE_BLOCK_SECONDS,
    )
    if block_margins.size == 0:
        return None

    split = len(margins) // 2
    margin_drift = (
        float(np.median(margins[split:]) - np.median(margins[:split]))
        if split > 0
        else 0.0
    )
    block_center = float(np.median(block_margins))
    block_mad = float(np.median(np.abs(block_margins - block_center)))
    block_excursion_bound = block_center + BASELINE_EXCURSION_MADS * max(block_mad, 1e-9)
    block_states = (block_margins > block_excursion_bound).astype(np.int8)
    burst_count = 0
    longest_blocks = 0
    for segment in np.unique(block_segments):
        segment_states = block_states[block_segments == segment]
        padded = np.concatenate([[0], segment_states, [0]])
        edges = np.diff(padded)
        burst_starts = np.flatnonzero(edges == 1)
        burst_lengths = np.flatnonzero(edges == -1) - burst_starts
        burst_count += int(burst_starts.size)
        if burst_lengths.size:
            longest_blocks = max(longest_blocks, int(burst_lengths.max()))
    longest_burst_seconds = float(longest_blocks) * BASELINE_BLOCK_SECONDS
    eval_seconds = _temporal_coverage_seconds(row_timing, len(margins))
    bursts_per_minute = (
        burst_count * 60.0 / eval_seconds if eval_seconds > 0.0 else 0.0
    )
    fp_rate = float(states.mean())
    margin_q95 = float(np.quantile(margins, 0.95))
    score = agnostic_baseline_score(
        margin_q95,
        longest_burst_seconds,
    )
    return {
        "packet_rate_pps": float(row_timing["target_pps"]),
        "eval_count": int(len(evidence)),
        "motion_count": int(states.sum()),
        "fp_rate": fp_rate,
        "excursion_bound": float(excursion_bound),
        "margin_center": margin_center,
        "margin_median": margin_median,
        "margin_mad": margin_mad,
        "margin_q95": margin_q95,
        "margin_q99": float(np.quantile(margins, 0.99)),
        "margin_drift": margin_drift,
        "margin_drift_abs": float(abs(margin_drift)),
        "margin_series": margins,
        "block_margins": block_margins,
        "score": score,
        "burst_count": burst_count,
        "bursts_per_minute": float(bursts_per_minute),
        "longest_burst_seconds": longest_burst_seconds,
        "eval_seconds": float(eval_seconds),
    }
