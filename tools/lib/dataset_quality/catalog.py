# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""dataset_info.json loading, derived metadata, and validation."""

from . import core
import re

import numpy as np

from tools.lib import dataset_metadata
from tools.lib.dataset_metadata import (
    DATASET_ROLES,
    admitted_dataset_role,
    dataset_role,
)
from .core import (
    ValidationResult,
    _result_counts,
)
from .severity import (
    METADATA_LABELS,
    PAIR_COUNTERPART_LABEL,
    REQUIRED_PAIR_FIELD_BY_LABEL,
    VALIDATION_DOMAINS,
    VALIDATION_DOMAIN_LABELS,
)

def _entry_environment(entry):
    """Return a compact environment label for table display."""
    value = entry.get("environment") if isinstance(entry, dict) else None
    if _is_missing_metadata_value(value):
        return "?"
    return str(value)


def _domain_summary_rows(all_results):
    """Return (label, counts) rows for the per-domain summary tables."""
    return [
        (
            VALIDATION_DOMAIN_LABELS[domain],
            _result_counts([
                result for result in all_results if result.domain == domain
            ]),
        )
        for domain in VALIDATION_DOMAINS
    ]


def _is_missing_metadata_value(value):
    """Return True when a dataset_info field is absent or semantically empty."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) == 0
    return False


def _entry_matches_chip(entry, chip_filter):
    """Return True when an entry should be included for the optional chip filter."""
    if not chip_filter:
        return True
    entry_chip = str(entry.get('chip', '')).lower()
    filename = str(entry.get('filename', '')).lower()
    chip = str(chip_filter).lower()
    return entry_chip == chip or chip in filename


def _dataset_role(entry):
    """Return the normalized dataset role for one dataset_info entry."""
    return dataset_role(entry)


def _is_excluded_entry(entry):
    """Return True unless an entry has an explicit admitted dataset role."""
    return admitted_dataset_role(entry) is None


def _is_long_recording_entry(entry):
    """Return True when an empty-room entry is reserved for long-recording replay."""
    return bool(entry.get("long_recording"))


def _long_recording_entry_records(dataset_info, chip_filter=None):
    """Return (label_group, entry) pairs for long-recording replay validation.

    Preferred layout stores quiet long-runs under `empty` with `long_recording:
    true`. Older datasets may still keep them under `test`.
    """
    files = dataset_info.get("files", {})
    explicit = [
        ("empty", entry)
        for entry in files.get("empty", [])
        if _is_long_recording_entry(entry)
        and _entry_matches_chip(entry, chip_filter)
        and not _is_excluded_entry(entry)
    ]
    if explicit:
        return explicit
    return [
        ("test", entry)
        for entry in files.get("test", [])
        if _entry_matches_chip(entry, chip_filter)
        and not _is_excluded_entry(entry)
    ]


def _extract_motion_start_from_description(description):
    """Extract motion start packet index from free-text test metadata."""
    if not description:
        return None

    match = re.search(
        r"motion\s+starts\s+at\s+packet(?:\s+index)?(?:\s+n\.)?\s+(\d+)",
        str(description),
        re.IGNORECASE,
    )
    if match:
        return int(match.group(1))
    return None


def load_dataset_info():
    """Load dataset_info.json."""
    return dataset_metadata.load_dataset_info(core.DATASET_INFO)


def save_dataset_info(info):
    """Write dataset_info.json with stable formatting."""
    dataset_metadata.save_dataset_info(info, core.DATASET_INFO)


def validate_metadata_completeness(dataset_info, chip_filter=None):
    """Check derived/manual dataset_info fields required by training workflows."""
    results = []
    files_by_label = dataset_info.get('files', {})
    filtered_entries = {}
    filename_index = {}

    for label in METADATA_LABELS:
        for entry in files_by_label.get(label, []):
            if not _entry_matches_chip(entry, chip_filter):
                continue
            filename = str(entry.get('filename', '<missing filename>'))
            raw_role = entry.get("dataset_role")
            if _is_missing_metadata_value(raw_role):
                results.append(ValidationResult(
                    f"metadata_{label}/{filename}",
                    "FAIL",
                    (
                        "missing dataset_role; entries default to exclude and "
                        "must be admitted explicitly"
                    ),
                ))
            elif _dataset_role(entry) not in DATASET_ROLES:
                results.append(ValidationResult(
                    f"metadata_{label}/{filename}",
                    "FAIL",
                    f"invalid dataset_role: {raw_role!r}",
                ))
        entries = [
            entry for entry in files_by_label.get(label, [])
            if _entry_matches_chip(entry, chip_filter)
            and not _is_excluded_entry(entry)
        ]
        filtered_entries[label] = entries
        filename_index[label] = {
            str(entry.get('filename')): entry
            for entry in entries
            if entry.get('filename')
        }

    for label, entries in filtered_entries.items():
        for entry in entries:
            filename = str(entry.get('filename', '<missing filename>'))
            entry_errors = []

            if _is_missing_metadata_value(entry.get('environment')):
                entry_errors.append("missing environment")
            for required_field in ('filename', 'chip', 'subcarriers', 'num_packets', 'collected_at'):
                if _is_missing_metadata_value(entry.get(required_field)):
                    entry_errors.append(f"missing {required_field}")

            primary_path = dataset_metadata.resolve_entry_path(label, entry)
            if filename != '<missing filename>' and not primary_path.exists():
                entry_errors.append("metadata entry target file is missing")

            pair_field = REQUIRED_PAIR_FIELD_BY_LABEL.get(label)
            if pair_field:
                counterpart_label = PAIR_COUNTERPART_LABEL[label]
                counterpart_name = entry.get(pair_field)
                if _is_missing_metadata_value(counterpart_name):
                    entry_errors.append(f"missing {pair_field}")
                else:
                    counterpart_name = str(counterpart_name)
                    counterpart_entry = filename_index[counterpart_label].get(counterpart_name)
                    counterpart_path = (
                        dataset_metadata.resolve_entry_path(
                            counterpart_label, counterpart_entry
                        )
                        if counterpart_entry is not None
                        else core.DATA_DIR / counterpart_label / counterpart_name
                    )
                    if counterpart_entry is None:
                        entry_errors.append(
                            f"{pair_field} does not reference a {counterpart_label} metadata entry"
                        )
                    if not counterpart_path.exists():
                        entry_errors.append(f"{pair_field} target file is missing")
                    elif counterpart_entry is not None:
                        if bool(entry.get("synthetic")) != bool(
                            counterpart_entry.get("synthetic")
                        ):
                            entry_errors.append(
                                f"{pair_field} mixes real and synthetic datasets"
                            )
                    if counterpart_entry is not None:
                        reverse_field = REQUIRED_PAIR_FIELD_BY_LABEL[counterpart_label]
                        if counterpart_entry.get(reverse_field) != filename:
                            entry_errors.append(f"{pair_field} is not reciprocal")
                        for shared_field in ('chip', 'subcarriers', 'device_id', 'environment'):
                            left = entry.get(shared_field)
                            right = counterpart_entry.get(shared_field)
                            if (
                                not _is_missing_metadata_value(left)
                                and not _is_missing_metadata_value(right)
                                and str(left) != str(right)
                            ):
                                entry_errors.append(
                                    f"{pair_field} has mismatched {shared_field}"
                                )

            result_name = f"metadata_{label}/{filename}"
            if entry_errors:
                results.append(ValidationResult(
                    result_name,
                    "FAIL",
                    "; ".join(entry_errors),
                ))
            else:
                results.append(ValidationResult(
                    result_name,
                    "PASS",
                    "Required dataset_info metadata is complete",
                ))

    if not any(filtered_entries.values()):
        results.append(ValidationResult(
            "metadata_entries",
            "FAIL",
            "No dataset_info entries found for metadata validation",
        ))

    for label, entries in filtered_entries.items():
        all_label_entries = files_by_label.get(label, [])
        if any(entry.get("relative_path") for entry in all_label_entries):
            # A source-native directory can contain additional views that are
            # intentionally outside this catalog. Orphan discovery is only
            # unambiguous for the conventional <label>/<filename> layout.
            continue
        metadata_paths = {
            dataset_metadata.resolve_entry_path(label, entry).resolve()
            for entry in entries
            if entry.get('filename')
        }
        excluded_metadata_paths = {
            dataset_metadata.resolve_entry_path(label, entry).resolve()
            for entry in all_label_entries
            if entry.get('filename')
            and _entry_matches_chip(entry, chip_filter)
            and _is_excluded_entry(entry)
        }
        label_dir = core.DATA_DIR / label
        if not label_dir.exists():
            continue
        disk_paths = {
            path.resolve() for path in label_dir.glob('*.npz')
            if _entry_matches_chip({'filename': path.name}, chip_filter)
        }
        for orphan_path in sorted(
            disk_paths - metadata_paths - excluded_metadata_paths,
            key=str,
        ):
            results.append(ValidationResult(
                f"metadata_orphan/{label}/{orphan_path.name}",
                "FAIL",
                "Capture exists on disk but is absent from dataset_info.json",
            ))

    return results


def should_recommend_dataset_metadata_refresh(results, missing_motion_pair_count=0):
    """Return True when validation suggests refreshing derived dataset metadata."""
    if missing_motion_pair_count > 0:
        return True

    for result in results:
        message = str(getattr(result, "message", ""))
        if "optimal_pair_motion_file" in message:
            return True
        if "optimal_pair_static_presence_file" in message:
            return True
    return False


def _estimate_average_packet_rate_from_capture(label, entry):
    """Estimate capture packet rate from replay timing metadata when possible."""
    path = dataset_metadata.resolve_entry_path(label, entry)
    if not path.exists():
        return dataset_metadata.estimate_average_packet_rate(
            entry.get("num_packets"),
            entry.get("duration_ms"),
        )
    try:
        with np.load(path, allow_pickle=False) as data:
            csi_data = data.get("csi_data")
            if csi_data is None:
                return dataset_metadata.estimate_average_packet_rate(
                    entry.get("num_packets"),
                    entry.get("duration_ms"),
                )
            num_packets = int(csi_data.shape[0])
            if num_packets < 2:
                return dataset_metadata.estimate_average_packet_rate(
                    entry.get("num_packets"),
                    entry.get("duration_ms"),
                )

            stream_seq_num = data.get("stream_seq_num")
            device_ticks_us = data.get("device_ticks_us")
            wifi_rx_ts_us = data.get("wifi_rx_ts_us")
            packet_views = []
            for index in range(num_packets):
                packet_view = {}
                if stream_seq_num is not None and index < len(stream_seq_num):
                    seq_num = int(stream_seq_num[index])
                    packet_view["seq_num"] = seq_num
                    packet_view["stream_seq_num"] = seq_num
                if device_ticks_us is not None and index < len(device_ticks_us):
                    packet_view["device_ticks_us"] = int(device_ticks_us[index])
                if wifi_rx_ts_us is not None and index < len(wifi_rx_ts_us):
                    packet_view["wifi_rx_ts_us"] = int(wifi_rx_ts_us[index])
                packet_views.append(packet_view)

            interval_us = dataset_metadata.measure_packet_interval_us(packet_views)
            if interval_us > 0:
                return 1_000_000.0 / float(interval_us)
    except (OSError, KeyError, ValueError, TypeError, IndexError):
        pass
    return dataset_metadata.estimate_average_packet_rate(
        entry.get("num_packets"),
        entry.get("duration_ms"),
    )


def _packet_rate_from_entry(entry):
    """Estimate capture packet rate from metadata, or return None if unknown."""
    average_packet_rate = entry.get("average_packet_rate")
    if average_packet_rate is not None:
        try:
            resolved = float(average_packet_rate)
        except (TypeError, ValueError):
            resolved = 0.0
        if resolved > 0.0:
            return resolved
    estimated = dataset_metadata.estimate_average_packet_rate(
        entry.get("num_packets"),
        entry.get("duration_ms"),
    )
    if estimated is not None:
        return estimated
    return None
