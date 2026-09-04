# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""NPZ integrity, signal, occupancy, and continuity validation."""

from . import core
import numpy as np

from tools.lib.csi_analysis import extract_amplitudes_matrix
from tools.lib.csi_io import (
    filter_npz_arrays_sensing,
    load_npz_arrays,
    load_npz_packet_view,
)
from tools.lib.timing_quality import (
    MAX_INTER_PACKET_GAP_FAIL_MS,
    MAX_INTER_PACKET_GAP_WARN_MS,
    MAX_STREAM_SEQ_GAP_FAIL_PACKETS,
    MAX_STREAM_SEQ_GAP_WARN_PACKETS,
    MAX_STREAM_SEQ_MISSING_FAIL_RATIO,
    MAX_STREAM_SEQ_MISSING_WARN_RATIO,
    MIN_CAPTURE_PACKET_RATE_PPS,
)
from .core import (
    MINIMUM_TEMPORAL_OCCUPANCY_RATIO,
    TEMPORAL_OCCUPANCY_WARN_RATIO,
    ValidationResult,
)
from .metrics import (
    _mean_temporal_occupancy,
    _resolve_temporal_occupancy_target_pps,
)
from .severity import (
    MAX_LOW_RSSI_STREAM_SEQ_MISSING_FAIL_RATIO,
    MAX_ZERO_PACKET_RATIO,
    METADATA_LABELS,
    MIN_AMPLITUDE_MEAN,
    MIN_PACKETS,
)

class _MaterializedNpz(dict):
    """Materialized NPZ contents; indexing does not re-read the archive.

    ``NpzFile`` decompresses an array on every ``data[key]`` access, so caching
    the raw handle would re-decompress CSI matrices in every validation phase.
    Materializing once also releases the underlying file handle immediately.
    """

    @property
    def files(self):
        return list(self.keys())


def _load_npz_materialized(filepath):
    """Load one NPZ file into a fully materialized key/array mapping."""
    return _MaterializedNpz(load_npz_arrays(filepath).items())


def _sensing_view_npz(data):
    """Return the sensing view used by continuity and Lightweight/High Accuracy quality."""
    filtered = filter_npz_arrays_sensing(dict(data))
    if filtered is data or (
        len(filtered) == len(data)
        and all(filtered[key] is data[key] for key in data)
    ):
        return data
    return _MaterializedNpz(filtered.items())


def _get_csi_key(data):
    """Return the key for CSI data inside an NPZ mapping."""
    keys = list(data.keys())
    if 'csi_data' in keys:
        return 'csi_data'
    if 'csi' in keys:
        return 'csi'
    return keys[0] if keys else None


def validate_file_integrity(filepath):
    """Check file can be loaded and has expected structure.

    Structural checks use the on-disk arrays. The returned mapping normally is
    the HT20 sensing view so later phases match training and host tooling. The
    explicit diagnostic mode returns all rows after retaining the supported
    sensing-contract result.
    """
    results = []

    try:
        raw_data = _load_npz_materialized(filepath)
    except Exception as e:
        results.append(ValidationResult("file_load", "FAIL", f"Cannot load: {e}"))
        return results, None

    results.append(ValidationResult("file_load", "PASS", "File loads successfully"))

    csi_key = _get_csi_key(raw_data)
    if csi_key is None:
        results.append(ValidationResult("csi_key", "FAIL", "No data keys found"))
        return results, None

    csi = raw_data[csi_key]
    if csi_key == 'csi_data':
        results.append(ValidationResult("csi_key", "PASS",
            f"CSI data found (key: {csi_key})", f"shape={csi.shape}"))
    elif csi_key == 'csi':
        results.append(ValidationResult("csi_key", "WARN",
            "Legacy CSI key found; current captures should use csi_data", f"shape={csi.shape}"))
    else:
        results.append(ValidationResult("csi_key", "FAIL",
            f"No supported CSI key; first key is {csi_key}", f"shape={csi.shape}"))
        return results, None

    if csi.ndim != 2:
        results.append(ValidationResult(
            "csi_shape", "FAIL", f"CSI matrix must be 2D, got shape {csi.shape}"
        ))
        return results, None

    if csi.shape[1] <= 0 or csi.shape[1] % 2 != 0:
        results.append(ValidationResult(
            "csi_shape", "FAIL", f"CSI width must contain I/Q pairs, got {csi.shape[1]}"
        ))
        return results, None

    actual_subcarriers = csi.shape[1] // 2
    declared_subcarriers = _read_scalar_metadata(raw_data, 'num_subcarriers')
    if declared_subcarriers is not None:
        try:
            declared_subcarriers = int(declared_subcarriers)
        except (TypeError, ValueError):
            declared_subcarriers = -1
        if declared_subcarriers != actual_subcarriers:
            results.append(ValidationResult(
                "csi_shape",
                "FAIL",
                (
                    f"CSI width implies {actual_subcarriers} subcarriers, but "
                    f"num_subcarriers={declared_subcarriers}"
                ),
            ))
        else:
            results.append(ValidationResult(
                "csi_shape", "PASS", f"Valid {actual_subcarriers}-subcarrier I/Q matrix"
            ))
    else:
        results.append(ValidationResult(
            "csi_shape",
            "WARN",
            f"Valid {actual_subcarriers}-subcarrier I/Q matrix without num_subcarriers metadata",
        ))

    packet_metadata_keys = (
        'stream_seq_num', 'device_ticks_us', 'wifi_rx_ts_us', 'wifi_rx_start_ts_ns',
        'channel', 'rssi_dbm', 'noise_floor_dbm',
    )
    mismatched = [
        key for key in packet_metadata_keys
        if key in raw_data.files and np.asarray(raw_data[key]).ndim > 0
        and len(raw_data[key]) != csi.shape[0]
    ]
    if mismatched:
        results.append(ValidationResult(
            "packet_metadata_shape",
            "FAIL",
            f"Per-packet metadata length mismatch: {', '.join(mismatched)}",
        ))
    else:
        results.append(ValidationResult(
            "packet_metadata_shape", "PASS", "Per-packet metadata lengths are coherent"
        ))

    embedded_label = _read_scalar_metadata(raw_data, 'label')
    directory_label = filepath.parent.name
    if embedded_label is None:
        results.append(ValidationResult(
            "embedded_label", "WARN", "Capture has no embedded label metadata"
        ))
    elif directory_label in METADATA_LABELS and str(embedded_label).lower() != directory_label:
        results.append(ValidationResult(
            "embedded_label",
            "FAIL",
            f"Embedded label {embedded_label!r} does not match directory {directory_label!r}",
        ))
    else:
        results.append(ValidationResult(
            "embedded_label", "PASS", f"Embedded label is {embedded_label!r}"
        ))

    sensing_view = _sensing_view_npz(raw_data)
    sensing_key = _get_csi_key(sensing_view)
    sensing_rows = 0 if sensing_key is None else int(np.asarray(sensing_view[sensing_key]).shape[0])
    if sensing_rows == 0:
        results.append(ValidationResult(
            "sensing_contract",
            "FAIL",
            "No HT20/HT-LTF/64-SC sensing packets remain after format filtering",
        ))
    else:
        results.append(ValidationResult(
            "sensing_contract",
            "PASS",
            f"Sensing view keeps {sensing_rows} HT20/HT-LTF/64-SC packet(s)",
        ))

    return results, raw_data if core.DIAGNOSTIC_ALL_PHY else sensing_view


def _load_validation_packet_view(filepath):
    """Load the packet view selected for report diagnostics."""
    return load_npz_packet_view(filepath, keep_all_phy=core.DIAGNOSTIC_ALL_PHY)


def validate_signal_quality(csi_data):
    """Check signal quality metrics."""
    results = []

    num_packets = csi_data.shape[0]

    # Packet count
    if num_packets < MIN_PACKETS:
        results.append(ValidationResult("packet_count", "FAIL",
            f"Too few packets: {num_packets} < {MIN_PACKETS}", num_packets))
    else:
        results.append(ValidationResult("packet_count", "PASS",
            f"{num_packets} packets", num_packets))

    # Zero-packet detection (vectorized)
    zero_packets = int(np.all(csi_data == 0, axis=1).sum())
    zero_ratio = zero_packets / num_packets if num_packets > 0 else 0
    if zero_ratio > MAX_ZERO_PACKET_RATIO:
        results.append(ValidationResult("zero_packets", "WARN",
            f"Zero-packet ratio: {zero_ratio:.4f} ({zero_packets}/{num_packets})", zero_ratio))
    else:
        results.append(ValidationResult("zero_packets", "PASS",
            f"Zero-packet ratio: {zero_ratio:.4f}", zero_ratio))

    # Mean amplitude check (vectorized, first 100 packets)
    sample = csi_data[:min(100, num_packets)]
    amps = extract_amplitudes_matrix(sample)
    mean_amp = float(amps.mean()) if amps.size > 0 else 0.0

    if mean_amp < MIN_AMPLITUDE_MEAN:
        results.append(ValidationResult("signal_level", "WARN",
            f"Low mean amplitude: {mean_amp:.2f}", mean_amp))
    else:
        results.append(ValidationResult("signal_level", "PASS",
            f"Mean amplitude: {mean_amp:.2f}", mean_amp))

    return results


def _read_scalar_metadata(data, key):
    """Return a scalar NPZ metadata value, or None when unavailable."""
    if key not in data.files:
        return None
    value = data[key]
    if np.shape(value) == ():
        return value.item()
    return value


def validate_temporal_occupancy(filepath, *, target_pps=None):
    """Check mean valid-slot occupancy on the recorded detector grid."""
    packets = _load_validation_packet_view(filepath)
    if not packets:
        return [ValidationResult(
            "temporal_occupancy",
            "FAIL",
            "Temporal occupancy unavailable: no sensing packets",
        )]

    target_pps = _resolve_temporal_occupancy_target_pps(
        packets,
        fallback=target_pps,
    )
    if target_pps is None:
        return [ValidationResult(
            "temporal_occupancy",
            "FAIL",
            "Temporal occupancy unavailable: missing csi_target_pps metadata",
        )]

    occupancy = _mean_temporal_occupancy(packets, target_pps)
    if occupancy < MINIMUM_TEMPORAL_OCCUPANCY_RATIO:
        status = "FAIL"
    elif occupancy < TEMPORAL_OCCUPANCY_WARN_RATIO:
        status = "WARN"
    else:
        status = "PASS"
    return [ValidationResult(
        "temporal_occupancy",
        status,
        (
            f"Mean temporal occupancy: {occupancy:.1%} "
            f"(warn < {TEMPORAL_OCCUPANCY_WARN_RATIO:.1%}, "
            f"fail < {MINIMUM_TEMPORAL_OCCUPANCY_RATIO:.1%})"
        ),
        round(occupancy, 4),
    )]


def validate_capture_continuity(
    data,
    csi_data,
    *,
    low_rssi=False,
    include_packet_rate=True,
):
    """Check packet cadence and stream continuity metadata when available.

    Real weak-link captures intentionally preserve bounded transport stress, so
    cataloged ``low_rssi`` recordings use a five-percent missing-sequence
    admission ceiling. Normal recordings retain the shared three-percent gate.
    """
    results = []
    num_packets = int(csi_data.shape[0])

    duration_ms = _read_scalar_metadata(data, 'duration_ms')
    try:
        duration_ms = float(duration_ms)
    except (TypeError, ValueError):
        duration_ms = 0.0

    if include_packet_rate and duration_ms > 0:
        packet_rate = num_packets / (duration_ms / 1000.0)
        if packet_rate < MIN_CAPTURE_PACKET_RATE_PPS:
            results.append(ValidationResult(
                "packet_rate",
                "WARN",
                (
                    f"Low packet rate: {packet_rate:.1f} pkt/s "
                    f"(< {MIN_CAPTURE_PACKET_RATE_PPS:.1f} pkt/s)"
                ),
                round(packet_rate, 1),
            ))
        else:
            results.append(ValidationResult(
                "packet_rate",
                "PASS",
                f"Packet rate: {packet_rate:.1f} pkt/s",
                round(packet_rate, 1),
            ))

    if 'stream_seq_num' not in data.files:
        return results

    stream_seq = np.asarray(data['stream_seq_num'], dtype=np.int64)
    if stream_seq.shape[0] != num_packets:
        results.append(ValidationResult(
            "stream_seq_num",
            "WARN",
            (
                "stream_seq_num length does not match CSI packets: "
                f"{stream_seq.shape[0]} != {num_packets}"
            ),
        ))
        return results

    if stream_seq.shape[0] < 2:
        results.append(ValidationResult(
            "stream_seq_gaps",
            "PASS",
            "Not enough packets to evaluate stream gaps",
        ))
        return results

    seq_delta = np.diff(stream_seq)
    missing_packets = int(np.maximum(seq_delta - 1, 0).sum())
    produced_packets = int(stream_seq[-1] - stream_seq[0] + 1)
    if produced_packets <= 0:
        results.append(ValidationResult(
            "stream_seq_gaps",
            "WARN",
            "stream_seq_num is not monotonic over the capture",
        ))
        return results

    missing_ratio = missing_packets / produced_packets
    nonunit_steps = int(np.sum(seq_delta != 1))
    seq_gap_sizes = np.maximum(seq_delta - 1, 0)
    max_seq_gap = int(seq_gap_sizes.max(initial=0))

    missing_fail_ratio = (
        MAX_LOW_RSSI_STREAM_SEQ_MISSING_FAIL_RATIO
        if low_rssi
        else MAX_STREAM_SEQ_MISSING_FAIL_RATIO
    )
    if missing_ratio > missing_fail_ratio:
        status = "FAIL"
    elif missing_ratio > MAX_STREAM_SEQ_MISSING_WARN_RATIO:
        status = "WARN"
    else:
        status = "PASS"

    threshold_note = (
        f", low_rssi fail > {missing_fail_ratio:.1%}"
        if low_rssi
        else ""
    )
    results.append(ValidationResult(
        "stream_seq_gaps",
        status,
        (
            f"Missing stream packets: {missing_ratio:.1%} "
            f"({missing_packets}/{produced_packets}, non-unit steps: {nonunit_steps}"
            f"{threshold_note})"
        ),
        round(missing_ratio, 4),
    ))

    if max_seq_gap > MAX_STREAM_SEQ_GAP_FAIL_PACKETS:
        status = "FAIL"
    elif max_seq_gap > MAX_STREAM_SEQ_GAP_WARN_PACKETS:
        status = "WARN"
    else:
        status = "PASS"

    max_gap_index = int(seq_gap_sizes.argmax()) if seq_gap_sizes.size and max_seq_gap > 0 else -1
    if max_gap_index >= 0:
        seq_before = int(stream_seq[max_gap_index])
        seq_after = int(stream_seq[max_gap_index + 1])
        gap_location = (
            f"after packet {max_gap_index} "
            f"(seq {seq_before} -> {seq_after})"
        )
    else:
        gap_location = "with no missing packets detected"

    results.append(ValidationResult(
        "stream_seq_max_gap",
        status,
        (
            f"Largest stream gap: {max_seq_gap} packets "
            f"{gap_location} "
            f"(warn > {MAX_STREAM_SEQ_GAP_WARN_PACKETS}, "
            f"fail > {MAX_STREAM_SEQ_GAP_FAIL_PACKETS})"
        ),
        max_seq_gap,
    ))

    timestamp_key = None
    if 'device_ticks_us' in data.files:
        timestamp_key = 'device_ticks_us'
    elif 'wifi_rx_ts_us' in data.files:
        timestamp_key = 'wifi_rx_ts_us'

    if timestamp_key is None:
        return results

    timestamps = np.asarray(data[timestamp_key], dtype=np.int64)
    if timestamps.shape[0] != num_packets:
        results.append(ValidationResult(
            "inter_packet_gap",
            "WARN",
            (
                f"{timestamp_key} length does not match CSI packets: "
                f"{timestamps.shape[0]} != {num_packets}"
            ),
        ))
        return results

    timestamp_delta = np.diff(timestamps)
    positive_delta = timestamp_delta[timestamp_delta > 0]
    if positive_delta.size == 0:
        results.append(ValidationResult(
            "inter_packet_gap",
            "WARN",
            f"{timestamp_key} is not monotonic enough to evaluate packet gaps",
        ))
        return results

    max_gap_index = int(timestamp_delta.argmax())
    max_gap_ms = float(timestamp_delta[max_gap_index]) / 1000.0
    if max_gap_ms > MAX_INTER_PACKET_GAP_FAIL_MS:
        status = "FAIL"
    elif max_gap_ms > MAX_INTER_PACKET_GAP_WARN_MS:
        status = "WARN"
    else:
        status = "PASS"

    results.append(ValidationResult(
        "inter_packet_gap",
        status,
        (
            f"Largest inter-packet gap: {max_gap_ms:.1f} ms via {timestamp_key} "
            f"at packet {max_gap_index}->{max_gap_index + 1} "
            f"(warn > {MAX_INTER_PACKET_GAP_WARN_MS:.1f} ms, "
            f"fail > {MAX_INTER_PACKET_GAP_FAIL_MS:.1f} ms)"
        ),
        round(max_gap_ms, 1),
    ))

    return results


def validate_capture_file(
    filepath,
    *,
    low_rssi=False,
    include_packet_rate=True,
    target_pps=None,
):
    """Run the canonical per-file admission checks used by CLI workflows."""
    results, data = validate_file_integrity(filepath)
    if data is None:
        return results

    csi_key = _get_csi_key(data)
    if csi_key is None:
        return results
    csi_data = data[csi_key]
    results.extend(validate_signal_quality(csi_data))
    results.extend(validate_temporal_occupancy(filepath, target_pps=target_pps))
    results.extend(validate_capture_continuity(
        data,
        csi_data,
        low_rssi=low_rssi,
        include_packet_rate=include_packet_rate,
    ))
    return results
