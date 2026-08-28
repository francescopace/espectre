# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Pair metadata refresh, pair evaluation, and excluded diagnostics."""

import datetime
from copy import deepcopy

import numpy as np

from tools.lib import dataset_metadata
from .capture import (
    _load_validation_packet_view,
)
from .catalog import (
    _entry_environment,
    _entry_matches_chip,
    _estimate_average_packet_rate_from_capture,
    _is_excluded_entry,
    _packet_rate_from_entry,
)
from .core import (
    ValidationResult,
)
from .metrics import (
    _agnostic_baseline_stats_from_series,
    _consensus_pair_evidence,
    _load_or_compute_validation_feature_matrix,
    _mean_temporal_occupancy,
    _pair_separation,
    _resolve_temporal_occupancy_target_pps,
    agnostic_pair_score,
    cap_quality_score_by_occupancy,
)
from .references import (
    _reference_idle_stats,
)
from .rendering import (
    _entry_display_date,
)
from .replay import (
    _calibrated_classic_for,
    _call_replay_classic_metrics,
    _probability_logit,
)
from .severity import (
    FAIL_MOTION_COVERAGE_RATIO,
    MAX_STATIC_ACTIVE_RATIO,
    MIN_ACTIVE_RATIO_MARGIN,
    MIN_MOTION_ACTIVE_RATIO,
    MIN_MOTION_COVERAGE_RATIO,
    _pair_separation_severity,
    _reference_cleanliness_severity,
    _threshold_severity,
)

PAIR_MAX_DELTA_SECONDS = 30 * 60


def parse_iso_timestamp(value):
    """Parse an ISO timestamp string, returning None when unavailable."""
    if not value:
        return None
    try:
        return datetime.datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _entry_matches_selected_chips(entry, selected_chips):
    """Return True when an entry should be refreshed for the selected chips."""
    if selected_chips is None:
        return True
    return str(entry.get("chip", "")).upper() in selected_chips


def _synthetic_group_from_npz(label, entry, cache):
    """Read the pairing group from a generated NPZ without catalog fields."""
    path = dataset_metadata.resolve_entry_path(label, entry)
    if path in cache:
        return cache[path]
    group_id = ""
    try:
        with np.load(path, allow_pickle=False) as generated:
            if "generation_group" in generated:
                group_id = str(np.asarray(generated["generation_group"]).item())
    except (OSError, ValueError):
        pass
    cache[path] = group_id
    return group_id


def refresh_pair_metadata(files, *, selected_chips=None):
    """
    Refresh explicit static_presence/motion pairing fields.

    Pairing policy:
    - same chip
    - same subcarrier count
    - timestamps within PAIR_MAX_DELTA_SECONDS
    - nearest 1:1 greedy assignment by time delta
    """
    static_entries = files.get("static_presence", [])
    motion_entries = files.get("motion", [])
    synthetic_group_cache = {}

    for entry in static_entries:
        if (
            _entry_matches_selected_chips(entry, selected_chips)
            and not _is_excluded_entry(entry)
        ):
            entry.pop("optimal_pair_motion_file", None)
    for entry in motion_entries:
        if (
            _entry_matches_selected_chips(entry, selected_chips)
            and not _is_excluded_entry(entry)
        ):
            entry.pop("optimal_pair_static_presence_file", None)

    candidates = []
    for static_index, static_entry in enumerate(static_entries):
        if not _entry_matches_selected_chips(static_entry, selected_chips):
            continue
        if _is_excluded_entry(static_entry):
            continue
        static_name = static_entry.get("filename")
        static_ts = parse_iso_timestamp(static_entry.get("collected_at"))
        static_chip = str(static_entry.get("chip", "")).upper()
        static_sc = int(static_entry.get("subcarriers", 0) or 0)
        if not static_name or static_ts is None or not static_chip or static_sc <= 0:
            continue

        for motion_index, motion_entry in enumerate(motion_entries):
            if not _entry_matches_selected_chips(motion_entry, selected_chips):
                continue
            if _is_excluded_entry(motion_entry):
                continue
            motion_name = motion_entry.get("filename")
            motion_ts = parse_iso_timestamp(motion_entry.get("collected_at"))
            motion_chip = str(motion_entry.get("chip", "")).upper()
            motion_sc = int(motion_entry.get("subcarriers", 0) or 0)
            if not motion_name or motion_ts is None:
                continue
            if motion_chip != static_chip or motion_sc != static_sc:
                continue

            static_synthetic = bool(static_entry.get("synthetic"))
            motion_synthetic = bool(motion_entry.get("synthetic"))
            if static_synthetic != motion_synthetic:
                continue
            if static_synthetic:
                static_group = _synthetic_group_from_npz(
                    "static_presence", static_entry, synthetic_group_cache
                )
                motion_group = _synthetic_group_from_npz(
                    "motion", motion_entry, synthetic_group_cache
                )
                if not static_group or static_group != motion_group:
                    continue

            static_device = str(static_entry.get("device_id", "")).strip()
            motion_device = str(motion_entry.get("device_id", "")).strip()
            if static_device and motion_device and static_device != motion_device:
                continue

            static_environment = str(static_entry.get("environment", "")).strip()
            motion_environment = str(motion_entry.get("environment", "")).strip()
            if (
                static_environment
                and motion_environment
                and static_environment != motion_environment
            ):
                continue

            delta = abs((motion_ts - static_ts).total_seconds())
            if delta > PAIR_MAX_DELTA_SECONDS:
                continue

            candidates.append(
                (
                    delta,
                    str(static_name),
                    str(motion_name),
                    static_index,
                    motion_index,
                )
            )

    used_static = set()
    used_motion = set()
    pair_rows = []

    for delta, static_name, motion_name, static_index, motion_index in sorted(candidates):
        if static_index in used_static or motion_index in used_motion:
            continue

        static_entry = static_entries[static_index]
        motion_entry = motion_entries[motion_index]
        static_entry["optimal_pair_motion_file"] = motion_name
        motion_entry["optimal_pair_static_presence_file"] = static_name
        used_static.add(static_index)
        used_motion.add(motion_index)
        pair_rows.append(
            {
                "static_presence": static_name,
                "motion": motion_name,
                "delta_seconds": round(float(delta), 3),
            }
        )

    return pair_rows


def refresh_metadata(info, chip_filter=None):
    """Return a refreshed copy of dataset_info and derived metadata summaries.

    Does not bump ``updated_at``; callers should set it only when the refreshed
    content differs from the previous dataset_info.
    """
    refreshed = deepcopy(info)
    files = refreshed.get("files", {})
    if chip_filter:
        if isinstance(chip_filter, str):
            selected_chips = {chip_filter.upper()}
        else:
            selected_chips = {str(chip).upper() for chip in chip_filter}
    else:
        selected_chips = None
    for label, entries in files.items():
        for entry in entries:
            if selected_chips and str(entry.get("chip", "")).upper() not in selected_chips:
                continue
            average_packet_rate = _estimate_average_packet_rate_from_capture(label, entry)
            if average_packet_rate is not None:
                entry["average_packet_rate"] = round(float(average_packet_rate), 3)
            try:
                explicit_nominal_packet_rate = int(entry.get("nominal_packet_rate", 0) or 0)
            except (TypeError, ValueError):
                explicit_nominal_packet_rate = 0
            if explicit_nominal_packet_rate > 0:
                entry["nominal_packet_rate"] = explicit_nominal_packet_rate
            elif average_packet_rate is not None and abs(float(average_packet_rate) - 100.0) <= 10.0:
                entry["nominal_packet_rate"] = 100
            else:
                entry.pop("nominal_packet_rate", None)
    pair_rows = refresh_pair_metadata(files, selected_chips=selected_chips)
    return refreshed, pair_rows


def summarize_pair_rows(pair_rows):
    """Print a compact summary of refreshed static_presence/motion pairs."""
    print(f"Resolved {len(pair_rows)} static_presence/motion pairs")
    if not pair_rows:
        return
    by_chip = {}
    for row in pair_rows:
        filename = row["static_presence"]
        parts = filename.split("_")
        chip = parts[2].upper() if len(parts) >= 3 else "UNKNOWN"
        by_chip[chip] = by_chip.get(chip, 0) + 1
    for chip in sorted(by_chip):
        print(f"  {chip:<15} count={by_chip[chip]:2d}")


def validate_pair(
    bl_csi,
    mv_csi,
    *,
    bl_rssi_dbm=None,
    mv_rssi_dbm=None,
    bl_stream_seq_num=None,
    mv_stream_seq_num=None,
    bl_device_ticks_us=None,
    mv_device_ticks_us=None,
    bl_wifi_rx_ts_us=None,
    mv_wifi_rx_ts_us=None,
    calibration_cache=None,
    cache_key=None,
):
    """Lightweight indicative replay for a static-presence/motion pair.

    Results are non-blocking: soft misses become WARN and never veto admission.

    Args:
        bl_csi: static-presence CSI array (num_packets, 128)
        mv_csi: motion CSI array (num_packets, 128)
        calibration_cache: optional per-run startup-threshold memo
        cache_key: cache key identifying the static capture
    Returns:
        tuple: (
            results,
            static_active_ratio,
            motion_active_ratio,
            threshold,
            pair_separation,  # idle/motion AUC, threshold-free
        )
    """
    results = []
    calibrated = _calibrated_classic_for(
        bl_csi,
        rssi_dbm=bl_rssi_dbm,
        stream_seq_num=bl_stream_seq_num,
        device_ticks_us=bl_device_ticks_us,
        wifi_rx_ts_us=bl_wifi_rx_ts_us,
        calibration_cache=calibration_cache,
        cache_key=cache_key,
    )
    if calibrated is None:
        results.append(ValidationResult(
            "classic_pair_activation",
            "WARN",
            "Could not calibrate the classic startup threshold from the static capture",
        ))
        return results, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    detector, threshold = calibrated
    bl_replay = _call_replay_classic_metrics(
        bl_csi,
        detector,
        rssi_dbm=bl_rssi_dbm,
        stream_seq_num=bl_stream_seq_num,
        device_ticks_us=bl_device_ticks_us,
        wifi_rx_ts_us=bl_wifi_rx_ts_us,
    )
    mv_replay = _call_replay_classic_metrics(
        mv_csi,
        detector,
        rssi_dbm=mv_rssi_dbm,
        stream_seq_num=mv_stream_seq_num,
        device_ticks_us=mv_device_ticks_us,
        wifi_rx_ts_us=mv_wifi_rx_ts_us,
    )
    mv_metric = mv_replay["score_series"]
    bl_states = bl_replay["state_series"]
    mv_states = mv_replay["state_series"]
    if len(bl_states) == 0 or len(mv_states) == 0:
        results.append(ValidationResult(
            "classic_pair_activation",
            "WARN",
            "Insufficient full-window Lightweight samples for pair diagnostic",
        ))
        return results, 0.0, 0.0, threshold, 0.0, 0.0, 0.0

    static_active_ratio = float(bl_states.mean())
    motion_active_ratio = float(mv_states.mean())
    pair_separation = _pair_separation(bl_replay["score_series"], mv_metric)

    # Threshold-free companions to the two activation ratios above. The idle
    # half is judged by its own tail, and motion coverage is measured against
    # the idle half's p95 rather than against the calibrated threshold.
    bl_metric = bl_replay["score_series"]
    bl_logits = _probability_logit(bl_metric)
    idle_tail = float(np.quantile(bl_logits, 0.95) - np.median(bl_logits))
    idle_p95 = float(np.quantile(bl_metric, 0.95))
    motion_coverage = float((np.asarray(mv_metric, dtype=np.float64) > idle_p95).mean())
    active_ratio_delta = motion_active_ratio - static_active_ratio

    passes = (
        static_active_ratio <= MAX_STATIC_ACTIVE_RATIO
        and motion_active_ratio >= MIN_MOTION_ACTIVE_RATIO
        and active_ratio_delta >= MIN_ACTIVE_RATIO_MARGIN
    )
    message = (
        "Lightweight diagnostic probability activation: "
        f"static_above={static_active_ratio:.1%}, "
        f"motion_above={motion_active_ratio:.1%}, "
        f"delta={active_ratio_delta:+.1%}, "
        f"separation={pair_separation:.4f} idle/motion AUC, "
        f"threshold={threshold:.6f}"
    )
    results.append(ValidationResult(
        "classic_pair_activation",
        "PASS" if passes else "WARN",
        message,
        round(motion_active_ratio, 4),
    ))
    return (
        results,
        static_active_ratio,
        motion_active_ratio,
        threshold,
        pair_separation,
        idle_tail,
        motion_coverage,
    )


def _evaluate_pair_capture(
    static_entry,
    motion_entry,
    *,
    idle_reference_records=None,
    use_cache=True,
):
    """Score one static_presence/motion pair from feature-space evidence."""
    bl_file = dataset_metadata.resolve_entry_path("static_presence", static_entry)
    mv_file = dataset_metadata.resolve_entry_path("motion", motion_entry)

    try:
        static_packets = _load_validation_packet_view(bl_file)
        motion_packets = _load_validation_packet_view(mv_file)
    except Exception as exc:
        return (
            [ValidationResult("pair_load", "FAIL", f"Cannot load pair: {exc}")],
            None,
            bl_file,
            mv_file,
        )

    static_packet_rate_pps = _packet_rate_from_entry(static_entry)
    motion_packet_rate_pps = _packet_rate_from_entry(motion_entry)
    if static_packet_rate_pps is None or motion_packet_rate_pps is None:
        missing = []
        if static_packet_rate_pps is None:
            missing.append(bl_file.name)
        if motion_packet_rate_pps is None:
            missing.append(mv_file.name)
        return (
            [ValidationResult(
                "pair_timing_metadata",
                "FAIL",
                "Insufficient timing metadata for temporal pair scoring: "
                + ", ".join(missing),
            )],
            None,
            bl_file,
            mv_file,
        )

    static_matrix, feature_names, static_row_timing = _load_or_compute_validation_feature_matrix(
        bl_file,
        use_cache=use_cache,
    )
    motion_matrix, motion_feature_names, motion_row_timing = _load_or_compute_validation_feature_matrix(
        mv_file,
        use_cache=use_cache,
    )
    if feature_names != motion_feature_names:
        return (
            [ValidationResult(
                "pair_feature_alignment",
                "FAIL",
                "Static and motion feature matrices do not share the same feature schema",
            )],
            None,
            bl_file,
            mv_file,
        )
    consensus = _consensus_pair_evidence(static_matrix, motion_matrix, feature_names)
    if consensus is None:
        return (
            [ValidationResult(
                "pair_feature_windows",
                "WARN",
                "Insufficient feature windows for agnostic pair scoring",
            )],
            None,
            bl_file,
            mv_file,
        )
    idle_evidence, motion_evidence, _centers, _scales = consensus
    pair_separation = _pair_separation(idle_evidence, motion_evidence)
    idle_p95 = float(np.quantile(idle_evidence, 0.95))
    motion_coverage = float((np.asarray(motion_evidence, dtype=np.float64) > idle_p95).mean())
    idle_baseline = _agnostic_baseline_stats_from_series(
        idle_evidence,
        static_row_timing,
    )
    pair_score = agnostic_pair_score(
        motion_coverage,
        pair_separation,
    )
    static_mean_occupancy = _mean_temporal_occupancy(
        static_packets,
        _resolve_temporal_occupancy_target_pps(
            static_packets,
            fallback=(
                static_entry.get("nominal_packet_rate")
                or static_row_timing.get("target_pps")
            ),
        ),
    )
    motion_mean_occupancy = _mean_temporal_occupancy(
        motion_packets,
        _resolve_temporal_occupancy_target_pps(
            motion_packets,
            fallback=(
                motion_entry.get("nominal_packet_rate")
                or motion_row_timing.get("target_pps")
            ),
        ),
    )
    reference_cleanliness = _reference_idle_stats(
        static_matrix,
        static_entry,
        feature_names,
        idle_reference_records or [],
        row_timing=static_row_timing,
        exclude_filename=bl_file.name,
    )
    score = cap_quality_score_by_occupancy(
        min(
            pair_score,
            reference_cleanliness["score"] if reference_cleanliness else pair_score,
        ),
        static_mean_occupancy,
        motion_mean_occupancy,
    )
    severity = _pair_separation_severity(pair_separation)
    coverage_severity = _threshold_severity(
        motion_coverage,
        warn_below=MIN_MOTION_COVERAGE_RATIO,
        fail_below=FAIL_MOTION_COVERAGE_RATIO,
    )
    reference_severity = _reference_cleanliness_severity(reference_cleanliness)
    status = "WARN" if severity or coverage_severity or reference_severity else "PASS"
    reference_message = (
        f", reference_excursions={reference_cleanliness['excursion_ratio']:.1%}, "
        f"reference_burst={reference_cleanliness['longest_burst_seconds']:.1f}s, "
        f"cleanliness={reference_cleanliness['score']:.1f}/100"
        if reference_cleanliness
        else ", reference_cleanliness=unavailable"
    )
    pair_res = [ValidationResult(
        "pair_feature_quality",
        status,
        (
            f"Feature-space quality score={score:.1f}/100; "
            f"pair_score={pair_score:.1f}/100, "
            f"motion_cover={motion_coverage:.1%}, "
            f"separation={pair_separation:.4f}, "
            f"occupancy={static_mean_occupancy:.1%}/{motion_mean_occupancy:.1%}"
            f"{reference_message}"
        ),
        score,
    )]

    pair_row = {
        "static_presence": bl_file.name,
        "motion": mv_file.name,
        "static_date": _entry_display_date(static_entry, bl_file.name),
        "motion_date": _entry_display_date(motion_entry, mv_file.name),
        "static_rssi_dbm": float(np.median([pkt.get("rssi_dbm") for pkt in static_packets if pkt.get("rssi_dbm") is not None])) if any(pkt.get("rssi_dbm") is not None for pkt in static_packets) else None,
        "motion_rssi_dbm": float(np.median([pkt.get("rssi_dbm") for pkt in motion_packets if pkt.get("rssi_dbm") is not None])) if any(pkt.get("rssi_dbm") is not None for pkt in motion_packets) else None,
        "static_packet_rate_pps": static_packet_rate_pps,
        "motion_packet_rate_pps": motion_packet_rate_pps,
        "static_mean_occupancy": static_mean_occupancy,
        "motion_mean_occupancy": motion_mean_occupancy,
        "chip": str(static_entry.get("chip", "unknown")).upper(),
        "environment": _entry_environment(static_entry),
        "idle_tail": idle_baseline["margin_q95"],
        "motion_coverage": motion_coverage,
        "pair_separation": pair_separation,
        "pair_score": pair_score,
        "reference_cleanliness": reference_cleanliness,
        "feature_score": score,
        "feature_names": feature_names,
        "status": status,
    }
    return pair_res, pair_row, bl_file, mv_file


def _collect_excluded_pair_rows(
    dataset_info,
    *,
    chip_filter=None,
    idle_reference_records=None,
    use_cache=True,
):
    """Return informational score rows for pairs whose role is `exclude`."""
    excluded_rows = []
    static_entries = [
        entry
        for entry in dataset_info.get("files", {}).get("static_presence", [])
        if _is_excluded_entry(entry) and _entry_matches_chip(entry, chip_filter)
    ]
    motion_entries_by_name = {
        str(item.get("filename", "")): item
        for item in dataset_info.get("files", {}).get("motion", [])
        if _is_excluded_entry(item)
    }

    for entry in static_entries:
        motion_entry = motion_entries_by_name.get(
            str(entry.get("optimal_pair_motion_file", ""))
        )
        if motion_entry is None or not _entry_matches_chip(motion_entry, chip_filter):
            continue
        bl_file = dataset_metadata.resolve_entry_path("static_presence", entry)
        mv_file = dataset_metadata.resolve_entry_path("motion", motion_entry)
        if not bl_file.exists() or not mv_file.exists():
            continue
        _pair_res, pair_row, _bl_file, _mv_file = _evaluate_pair_capture(
            entry,
            motion_entry,
            idle_reference_records=idle_reference_records,
            use_cache=use_cache,
        )
        if pair_row is None:
            continue
        excluded_rows.append(pair_row)

    return excluded_rows


def _collect_excluded_idle_rows(
    dataset_info,
    *,
    chip_filter=None,
    idle_reference_records=None,
    use_cache=True,
):
    """Return cross-capture cleanliness diagnostics for excluded idle files."""
    rows = []
    for label in ("empty",):
        for entry in dataset_info.get("files", {}).get(label, []):
            if not _is_excluded_entry(entry) or not _entry_matches_chip(entry, chip_filter):
                continue
            filepath = dataset_metadata.resolve_entry_path(label, entry)
            if not filepath.exists():
                continue
            try:
                matrix, feature_names, row_timing = _load_or_compute_validation_feature_matrix(
                    filepath,
                    use_cache=use_cache,
                )
                packets = _load_validation_packet_view(filepath)
            except Exception:
                continue
            unusable = int(np.asarray(matrix).shape[0]) == 0
            reference_cleanliness = None if unusable else _reference_idle_stats(
                matrix,
                entry,
                feature_names,
                idle_reference_records or [],
                row_timing=row_timing,
                exclude_filename=filepath.name,
            )
            rssi_values = [
                packet.get("rssi_dbm")
                for packet in packets
                if packet.get("rssi_dbm") is not None
            ]
            mean_occupancy = _mean_temporal_occupancy(
                packets,
                _resolve_temporal_occupancy_target_pps(
                    packets,
                    fallback=(
                        entry.get("nominal_packet_rate")
                        or row_timing.get("target_pps")
                    ),
                ),
            )
            if reference_cleanliness is not None:
                reference_cleanliness["intrinsic_score"] = reference_cleanliness["score"]
                reference_cleanliness["score"] = cap_quality_score_by_occupancy(
                    reference_cleanliness["score"],
                    mean_occupancy,
                )
            rows.append({
                "label": label,
                "filename": filepath.name,
                "display_date": _entry_display_date(entry, filepath.name),
                "chip": str(entry.get("chip", "unknown")).upper(),
                "environment": _entry_environment(entry),
                "rssi_dbm": float(np.median(rssi_values)) if rssi_values else None,
                "packet_rate_pps": _packet_rate_from_entry(entry),
                "mean_occupancy": mean_occupancy,
                "reference_cleanliness": reference_cleanliness,
                "unusable": unusable,
            })
    return rows


def _excluded_idle_unusable_results(rows):
    """Return WARN results for excluded idle captures with no feature rows."""
    results = []
    for row in rows:
        if not row.get("unusable"):
            continue
        filename = str(row.get("filename", "?"))
        results.append(ValidationResult(
            f"excluded_idle_unusable/{filename}",
            "WARN",
            (
                "Excluded idle capture produces no usable feature rows after "
                "fixed temporal admission, so cleanliness cannot be scored"
            ),
        ))
    return results
