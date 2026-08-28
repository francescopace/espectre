# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Idle evidence, long-recording checks, and ML readiness."""

import numpy as np

from config import SEGMENTATION_WINDOW_SIZE_MS
from tools.lib import dataset_metadata
from tools.lib.runtime_policy import derive_detector_timing
from .capture import (
    _load_validation_packet_view,
)
from .catalog import (
    _entry_environment,
    _entry_matches_chip,
    _extract_motion_start_from_description,
    _is_excluded_entry,
    _is_long_recording_entry,
    _is_missing_metadata_value,
    _long_recording_entry_records,
    _packet_rate_from_entry,
)
from .core import (
    ValidationResult,
)
from .metrics import (
    _agnostic_baseline_stats_from_series,
    _feature_direction_vector,
    _feature_evidence_series,
    _load_or_compute_validation_feature_matrix,
    _mean_temporal_occupancy,
    _resolve_temporal_occupancy_target_pps,
    cap_quality_score_by_occupancy,
)
from .references import (
    _unique_entries_by_resolved_path,
)
from .rendering import (
    _entry_display_date,
)
from .replay import (
    _resolve_dataset_entry_path,
)
from .severity import (
    REQUIRED_PAIR_FIELD_BY_LABEL,
    VALIDATION_FEATURE_NAMES,
    _empty_quality_verdict,
    _presence_quality_verdict,
)

def _training_session_group(label, entry):
    """Mirror the trainer's explicit-session, pair, then file grouping policy."""
    for field in ('session', 'session_id', 'session_name'):
        value = entry.get(field)
        if not _is_missing_metadata_value(value):
            return str(value)

    pair_field = REQUIRED_PAIR_FIELD_BY_LABEL.get(label)
    counterpart = entry.get(pair_field) if pair_field else None
    filename = str(entry.get('filename', 'unknown'))
    if counterpart:
        names = sorted((filename, str(counterpart)))
        return f"pair:{names[0]}::{names[1]}"
    return f"file:{filename}"


def _usable_window_count(label, entry, *, use_cache=True):
    """Return admitted feature windows after the production readiness gate."""
    try:
        filepath = dataset_metadata.resolve_entry_path(label, entry)
        matrix, _feature_names, _row_timing = (
            _load_or_compute_validation_feature_matrix(
                filepath,
                use_cache=use_cache,
            )
        )
    except (OSError, ValueError, KeyError):
        return 0
    return int(matrix.shape[0])


def validate_ml_readiness(dataset_info, chip_filter=None, *, use_cache=True):
    """Check if the binary empty/static-presence/motion dataset is ML-ready."""
    results = []

    files_by_label = dataset_info.get('files', {})
    training_files = {
        label: [
            entry for entry in files_by_label.get(label, [])
            if _entry_matches_chip(entry, chip_filter)
            and not _is_excluded_entry(entry)
            and not bool(entry.get('synthetic'))
            and not (label == 'empty' and _is_long_recording_entry(entry))
        ]
        for label in ('empty', 'static_presence', 'motion')
    }

    windows_by_label = {
        label: sum(
            _usable_window_count(label, entry, use_cache=use_cache)
            for entry in entries
        )
        for label, entries in training_files.items()
    }
    missing_timing = sorted(
        str(entry.get("filename", "unknown"))
        for entries in training_files.values()
        for entry in entries
        if _packet_rate_from_entry(entry) is None
    )
    results.append(ValidationResult(
        "timing_metadata",
        "FAIL" if missing_timing else "PASS",
        (
            "Missing usable average_packet_rate or num_packets/duration_ms timing metadata: "
            + ", ".join(missing_timing)
            if missing_timing
            else "All ML training captures provide usable timing metadata"
        ),
        len(missing_timing),
    ))
    idle_windows = windows_by_label['empty'] + windows_by_label['static_presence']
    motion_windows = windows_by_label['motion']
    total = idle_windows + motion_windows

    if total > 0:
        idle_ratio = idle_windows / total
        if 0.3 <= idle_ratio <= 0.7:
            results.append(ValidationResult("label_balance", "PASS",
                (
                    f"Binary window balance: {idle_ratio:.1%} IDLE "
                    f"(empty={windows_by_label['empty']}, "
                    f"static_presence={windows_by_label['static_presence']}), "
                    f"{1-idle_ratio:.1%} MOTION"
                ), idle_ratio))
        else:
            results.append(ValidationResult("label_balance", "WARN",
                (
                    f"Imbalanced binary windows: {idle_ratio:.1%} IDLE "
                    f"(empty={windows_by_label['empty']}, "
                    f"static_presence={windows_by_label['static_presence']}), "
                    f"{1-idle_ratio:.1%} MOTION"
                ), idle_ratio))
    else:
        results.append(ValidationResult(
            "label_balance", "FAIL", "No usable ML windows after per-file warm-up"
        ))

    min_windows = 1000
    estimated_windows = total
    if estimated_windows < min_windows:
        results.append(ValidationResult("sample_count", "WARN",
            f"Low sample count: ~{estimated_windows} windows (target: {min_windows}+)", estimated_windows))
    else:
        results.append(ValidationResult("sample_count", "PASS",
            f"~{estimated_windows} feature windows available", estimated_windows))

    all_training_entries = [
        entry for entries in training_files.values() for entry in entries
    ]
    chips = {str(entry.get('chip', 'unknown')).upper() for entry in all_training_entries}
    if chip_filter and chips:
        results.append(ValidationResult("chip_diversity", "PASS",
            f"Filtered ML scope contains chip: {sorted(chips)}", len(chips)))
    elif len(chips) >= 3:
        results.append(ValidationResult("chip_diversity", "PASS",
            f"{len(chips)} chip types: {sorted(chips)}", len(chips)))
    else:
        results.append(ValidationResult("chip_diversity", "WARN",
            f"Only {len(chips)} chip type(s): {sorted(chips)}", len(chips)))

    sessions_by_target = {'IDLE': set(), 'MOTION': set()}
    for label, entries in training_files.items():
        target = 'MOTION' if label == 'motion' else 'IDLE'
        sessions_by_target[target].update(
            _training_session_group(label, entry) for entry in entries
        )

    all_sessions = sessions_by_target['IDLE'] | sessions_by_target['MOTION']
    min_folds = 3
    if min(len(sessions_by_target['IDLE']), len(sessions_by_target['MOTION'])) >= min_folds:
        session_status = "PASS"
    else:
        session_status = "WARN"
    results.append(ValidationResult(
        "session_group_coverage",
        session_status,
        (
            f"{len(all_sessions)} grouped sessions: "
            f"IDLE={len(sessions_by_target['IDLE'])}, "
            f"MOTION={len(sessions_by_target['MOTION'])}; "
            f"three-fold grouped CV expects at least {min_folds} per target"
        ),
        len(all_sessions),
    ))

    environments = {
        str(entry.get('environment', 'unknown')) for entry in all_training_entries
    }
    unknown_environment = 'unknown' in environments or '' in environments
    results.append(ValidationResult(
        "environment_coverage",
        "WARN" if unknown_environment or len(environments) < 2 else "PASS",
        f"{len(environments)} ML environment group(s): {sorted(environments)}",
        len(environments),
    ))

    return results


def _group_entries_by_chip_env(entries):
    """Group dataset entries by (chip, environment)."""
    group_map = {}
    for entry in entries:
        group = (
            str(entry.get("chip", "unknown")).upper(),
            str(entry.get("environment", "unknown")),
        )
        group_map.setdefault(group, []).append(entry)
    return group_map


def _compute_idle_evidence_for_entry(entry, label, *, use_cache=True):
    """Return baseline, median RSSI, mean occupancy, and an optional error."""
    try:
        filepath = _resolve_dataset_entry_path(entry, label)
        packet_rate_pps = _packet_rate_from_entry(entry)
        if packet_rate_pps is None:
            return None, None, None, "insufficient timing metadata"
        feature_names = tuple(VALIDATION_FEATURE_NAMES)
        packets = _load_validation_packet_view(filepath)
        feature_matrix, feature_names, row_timing = _load_or_compute_validation_feature_matrix(
            filepath,
            feature_names=feature_names,
            use_cache=use_cache,
        )
        evidence = _feature_evidence_series(
            feature_matrix,
            directions=_feature_direction_vector(feature_names),
        )
        baseline = _agnostic_baseline_stats_from_series(evidence, row_timing)
        if baseline is None:
            return None, None, None, "insufficient data"
        rssi_values = [pkt.get("rssi_dbm") for pkt in packets if pkt.get("rssi_dbm") is not None]
        median_rssi = float(np.median(rssi_values)) if rssi_values else None
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
        baseline["intrinsic_score"] = baseline["score"]
        baseline["score"] = cap_quality_score_by_occupancy(
            baseline["score"],
            mean_occupancy,
        )
        return baseline, median_rssi, mean_occupancy, None
    except (OSError, ValueError, KeyError) as exc:
        return None, None, None, str(exc)


def _idle_evidence_score_row(entry, baseline, verdict, rssi_dbm, mean_occupancy):
    """Build one shared idle-evidence score-table row."""
    filename = str(entry.get("filename", "?"))
    return {
        "chip": str(entry.get("chip", "?")).upper(),
        "environment": _entry_environment(entry),
        "filename": filename,
        "display_date": _entry_display_date(entry, filename),
        "rssi_dbm": rssi_dbm,
        "mean_occupancy": mean_occupancy,
        "baseline": baseline,
        "verdict": verdict,
    }


def _evaluate_idle_evidence_files(
    entries,
    *,
    label,
    check_kind,
    kind_title,
    verdict_fn,
    use_cache=True,
):
    """Score one empty or static_presence label set into results + table rows."""
    results = []
    score_rows = []
    for entry in entries:
        filename = str(entry.get("filename", "?"))
        baseline, rssi_dbm, mean_occupancy, error = _compute_idle_evidence_for_entry(
            entry,
            label,
            use_cache=use_cache,
        )
        if baseline is None:
            results.append(ValidationResult(
                f"{check_kind}/{filename}",
                "WARN",
                (
                    f"Could not compute {kind_title.lower()} quality diagnostics: "
                    f"{error or 'insufficient data'}"
                ),
            ))
            continue

        verdict = verdict_fn(baseline)
        status = "PASS" if verdict == "clean" else "WARN"
        results.append(ValidationResult(
            f"{check_kind}/{filename}",
            status,
            (
                f"{kind_title} quality: verdict={verdict}, "
                f"feature_score={baseline['score']:.1f}, "
                f"excursion_rate={baseline['fp_rate']:.1%}"
            ),
            baseline["score"],
        ))
        score_rows.append(
            _idle_evidence_score_row(
                entry,
                baseline,
                verdict,
                rssi_dbm,
                mean_occupancy,
            )
        )
    return results, score_rows


def validate_empty_sanity(dataset_info, chip_filter=None, use_cache=True):
    """Score empty and static-presence captures from Lightweight idle baselines.

    Returns:
        tuple: (results, empty_score_rows, presence_score_rows)
    """
    results = []

    empty_files = _unique_entries_by_resolved_path("empty", [
        entry for entry in dataset_info.get('files', {}).get('empty', [])
        if _entry_matches_chip(entry, chip_filter)
        and not _is_excluded_entry(entry)
        and not bool(entry.get('synthetic'))
        and not _is_long_recording_entry(entry)
    ])
    static_presence_files = _unique_entries_by_resolved_path("static_presence", [
        entry for entry in dataset_info.get('files', {}).get('static_presence', [])
        if _entry_matches_chip(entry, chip_filter)
        and not _is_excluded_entry(entry)
        and not bool(entry.get('synthetic'))
    ])

    if not empty_files:
        results.append(ValidationResult(
            "empty_dataset_presence", "WARN",
            "No empty datasets available for validation"
        ))
    else:
        results.append(ValidationResult(
            "empty_dataset_presence", "PASS",
            f"{len(empty_files)} empty file(s) available", len(empty_files)
        ))

    empty_group_map = _group_entries_by_chip_env(empty_files)
    static_group_map = _group_entries_by_chip_env(static_presence_files)
    overlap_groups = sorted(set(empty_group_map) & set(static_group_map))

    if not overlap_groups:
        results.append(ValidationResult(
            "empty_overlap_groups", "WARN",
            "No overlapping chip/environment groups with static presence"
        ))
    else:
        results.append(ValidationResult(
            "empty_overlap_groups", "PASS",
            f"{len(overlap_groups)} overlapping chip/environment group(s): {overlap_groups}",
            len(overlap_groups)
        ))

    empty_results, empty_score_rows = _evaluate_idle_evidence_files(
        empty_files,
        label="empty",
        check_kind="empty_quality",
        kind_title="Empty",
        verdict_fn=_empty_quality_verdict,
        use_cache=use_cache,
    )
    presence_results, presence_score_rows = _evaluate_idle_evidence_files(
        static_presence_files,
        label="static_presence",
        check_kind="presence_quality",
        kind_title="Presence",
        verdict_fn=_presence_quality_verdict,
        use_cache=use_cache,
    )
    results.extend(empty_results)
    results.extend(presence_results)

    return results, empty_score_rows, presence_score_rows


def validate_quiet_test_recordings(dataset_info, chip_filter=None, use_cache=True):
    """Validate long-recording coverage and score idle-only Lightweight baselines."""
    results = []
    idle_candidates = []
    mixed_candidates = []
    for label_group, entry in _long_recording_entry_records(
        dataset_info, chip_filter=chip_filter
    ):
        motion_start = _extract_motion_start_from_description(entry.get("description"))
        if motion_start is None:
            idle_candidates.append((label_group, entry))
        else:
            mixed_candidates.append((label_group, entry, motion_start))

    results.append(ValidationResult(
        "long_test_event_coverage",
        "PASS" if mixed_candidates else "WARN",
        (
            f"{len(mixed_candidates)} mixed long recording(s) with an annotated motion start; "
            "event recall and detection latency are unavailable" if not mixed_candidates else
            f"{len(mixed_candidates)} mixed long recording(s) with an annotated motion start"
        ),
        len(mixed_candidates),
    ))

    for _label_group, entry, motion_start in mixed_candidates:
        filename = str(entry.get("filename", "<missing filename>"))
        try:
            num_packets = int(entry.get("num_packets", 0) or 0)
        except (TypeError, ValueError):
            num_packets = 0
        packet_rate_pps = _packet_rate_from_entry(entry)
        if packet_rate_pps is None:
            results.append(ValidationResult(
                f"long_test_annotation/{filename}",
                "FAIL",
                "Insufficient timing metadata to resolve the temporal detector window",
            ))
            continue
        window_packets = derive_detector_timing(
            max(1, int(round(1_000_000.0 / packet_rate_pps))),
            SEGMENTATION_WINDOW_SIZE_MS,
        )["window_packets"]
        valid = motion_start > window_packets and num_packets - motion_start > window_packets
        results.append(ValidationResult(
            f"long_test_annotation/{filename}",
            "PASS" if valid else "FAIL",
            (
                f"motion_start={motion_start}, packets={num_packets}; both IDLE and MOTION "
                f"segments must exceed the {SEGMENTATION_WINDOW_SIZE_MS} ms warm-up "
                f"({window_packets} packets at the recorded cadence)"
            ),
            motion_start,
        ))

    quiet_score_rows = []
    if not idle_candidates:
        results.append(ValidationResult(
            "quiet_test_presence",
            "WARN",
            "No idle-only long recordings available for validation",
        ))
        return results, quiet_score_rows

    results.append(ValidationResult(
        "quiet_test_presence",
        "PASS",
        f"{len(idle_candidates)} idle-only long-recording file(s) available",
        len(idle_candidates),
    ))

    idle_results = []
    quiet_score_rows = []
    for label_group, entry in idle_candidates:
        filename = str(entry.get("filename", "?"))
        baseline, rssi_dbm, mean_occupancy, error = _compute_idle_evidence_for_entry(
            entry,
            label_group,
            use_cache=use_cache,
        )
        if baseline is None:
            idle_results.append(ValidationResult(
                f"quiet_test_idle/{filename}",
                "WARN",
                (
                    "Could not compute long-recording quality diagnostics: "
                    f"{error or 'insufficient data'}"
                ),
            ))
            continue

        verdict = _empty_quality_verdict(baseline)
        idle_results.append(ValidationResult(
            f"quiet_test_idle/{filename}",
            "PASS" if verdict == "clean" else "WARN",
            (
                f"Long-recording quality: verdict={verdict}, "
                f"feature_score={baseline['score']:.1f}, "
                f"excursion_rate={baseline['fp_rate']:.1%}"
            ),
            baseline["score"],
        ))
        quiet_score_rows.append(
            _idle_evidence_score_row(
                entry,
                baseline,
                verdict,
                rssi_dbm,
                mean_occupancy,
            )
        )
    results.extend(idle_results)
    return results, quiet_score_rows
