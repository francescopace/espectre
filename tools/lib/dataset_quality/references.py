# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Idle-reference selection and external cleanliness metrics."""

import numpy as np

from tools.lib import dataset_metadata
from .catalog import (
    _entry_environment,
    _entry_matches_chip,
    _is_excluded_entry,
    _is_long_recording_entry,
    _packet_rate_from_entry,
)
from .metrics import (
    _feature_block_medians,
    _feature_direction_vector,
    _feature_evidence_series,
    _load_or_compute_validation_feature_matrix,
    _robust_axis_location_and_scale,
    _sample_reference_blocks,
    reference_cleanliness_score,
)
from .severity import (
    REFERENCE_BLOCK_SECONDS,
    REFERENCE_HIGH_RATE_PPS,
    REFERENCE_MIN_CAPTURES,
)

def _idle_reference_stratum(entry):
    """Return link- and packet-rate classes that must not be mixed."""
    link_class = "low-rssi" if bool(entry.get("low_rssi")) else "normal-rssi"
    packet_rate_pps = _packet_rate_from_entry(entry)
    if packet_rate_pps is None:
        rate_class = "unknown-rate"
    elif packet_rate_pps >= REFERENCE_HIGH_RATE_PPS:
        rate_class = "high-rate"
    else:
        rate_class = "nominal-rate"
    return link_class, rate_class


def _unique_entries_by_resolved_path(label, entries):
    """Keep one logical catalog entry for each canonical NPZ path."""
    unique = []
    seen_paths = set()
    for entry in entries:
        path = dataset_metadata.resolve_entry_path(label, entry).resolve()
        if path in seen_paths:
            continue
        seen_paths.add(path)
        unique.append(entry)
    return unique


def _build_idle_reference_records(dataset_info, *, chip_filter=None, use_cache=True):
    """Build admitted, non-long idle references for cross-capture review."""
    records = []
    for label in ("empty", "static_presence"):
        entries = _unique_entries_by_resolved_path(
            label,
            dataset_info.get("files", {}).get(label, []),
        )
        for entry in entries:
            if _is_excluded_entry(entry) or not _entry_matches_chip(entry, chip_filter):
                continue
            if label == "empty" and _is_long_recording_entry(entry):
                continue
            filepath = dataset_metadata.resolve_entry_path(label, entry)
            if not filepath.exists():
                continue
            try:
                matrix, feature_names, row_timing = _load_or_compute_validation_feature_matrix(
                    filepath,
                    use_cache=use_cache,
                )
            except Exception:
                continue
            blocks = _feature_block_medians(
                matrix,
                row_timing,
            )
            if blocks.size == 0:
                continue
            records.append({
                "filename": filepath.name,
                "chip": str(entry.get("chip", "unknown")).upper(),
                "environment": _entry_environment(entry),
                "stratum": _idle_reference_stratum(entry),
                "feature_names": tuple(feature_names),
                "blocks": _sample_reference_blocks(blocks),
            })
    return records


def _select_idle_reference_records(records, entry, feature_names, *, exclude_filename=None):
    """Choose same-environment references when sufficient, then same-chip."""
    chip = str(entry.get("chip", "unknown")).upper()
    environment = _entry_environment(entry)
    stratum = _idle_reference_stratum(entry)
    feature_names = tuple(feature_names)
    candidates = [
        record
        for record in records
        if record["chip"] == chip
        and record["feature_names"] == feature_names
        and record.get("stratum", ("normal-rssi", "nominal-rate")) == stratum
        and record["filename"] != exclude_filename
    ]
    environment_records = [
        record
        for record in candidates
        if record["environment"] == environment
    ]
    if len(environment_records) >= REFERENCE_MIN_CAPTURES:
        return environment_records, "chip+env+stratum"
    if len(candidates) >= REFERENCE_MIN_CAPTURES:
        return candidates, "chip+stratum"
    return [], "unavailable"


def _reference_idle_stats(
    feature_matrix,
    entry,
    feature_names,
    reference_records,
    *,
    row_timing,
    exclude_filename=None,
):
    """Compare one idle capture with independent same-chip feature blocks."""
    references, basis = _select_idle_reference_records(
        reference_records,
        entry,
        feature_names,
        exclude_filename=exclude_filename,
    )
    if not references:
        return None

    reference_blocks = np.concatenate(
        [record["blocks"] for record in references],
        axis=0,
    )
    centers = np.median(reference_blocks, axis=0)
    scales = np.zeros(reference_blocks.shape[1], dtype=np.float64)
    for index in range(reference_blocks.shape[1]):
        _center, scales[index] = _robust_axis_location_and_scale(
            reference_blocks[:, index]
        )
    directions = _feature_direction_vector(feature_names)
    reference_evidence = _feature_evidence_series(
        reference_blocks,
        centers=centers,
        scales=scales,
        directions=directions,
    )
    target_blocks = _feature_block_medians(
        feature_matrix,
        row_timing,
    )
    target_evidence = _feature_evidence_series(
        target_blocks,
        centers=centers,
        scales=scales,
        directions=directions,
    )
    if target_evidence.size == 0:
        return None

    excursion_bound = float(np.quantile(reference_evidence, 0.95))
    extreme_bound = float(np.quantile(reference_evidence, 0.99))
    excursion_ratio = float((target_evidence > excursion_bound).mean())
    extreme_states = (target_evidence > extreme_bound).astype(np.int8)
    padded = np.concatenate([[0], extreme_states, [0]])
    edges = np.diff(padded)
    burst_starts = np.flatnonzero(edges == 1)
    burst_lengths = np.flatnonzero(edges == -1) - burst_starts
    longest_burst_seconds = (
        float(int(burst_lengths.max())) * REFERENCE_BLOCK_SECONDS
        if burst_starts.size
        else 0.0
    )
    score = reference_cleanliness_score(
        excursion_ratio,
        longest_burst_seconds,
    )
    return {
        "basis": basis,
        "reference_count": len(references),
        "block_count": int(len(target_evidence)),
        "excursion_bound": excursion_bound,
        "extreme_bound": extreme_bound,
        "excursion_ratio": excursion_ratio,
        "extreme_ratio": float(extreme_states.mean()),
        "longest_burst_seconds": longest_burst_seconds,
        "evidence_median": float(np.median(target_evidence)),
        "evidence_q95": float(np.quantile(target_evidence, 0.95)),
        "score": score,
    }
