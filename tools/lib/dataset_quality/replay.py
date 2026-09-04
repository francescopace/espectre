# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Classic detector replay and self-baseline diagnostics."""

from copy import deepcopy

import numpy as np

from config import (
    CALIBRATION_DURATION_MS,
    CSI_TARGET_PPS,
    DEFAULT_SUBCARRIERS,
    EVALUATION_INTERVAL_MS,
    SEGMENTATION_WINDOW_SIZE_MS,
)
from detector_interface import MotionState
from tools.lib import dataset_metadata
from tools.lib.dataset_metadata import build_calibrated_lightweight_detector
from tools.lib.runtime_policy import (
    make_evaluation_cadence,
    nominal_packet_interval_us,
)
from tools.lib.temporal_replay import (
    TemporalReplayController,
    apply_temporal_admission,
)
from .metrics import (
    classic_baseline_score,
)
from .severity import (
    BASELINE_BLOCK_SECONDS,
    BASELINE_EXCURSION_MADS,
)

def _resolve_dataset_entry_path(entry, label_group):
    """Resolve an NPZ path from label group + filename, with legacy fallback."""
    return dataset_metadata.resolve_entry_path(str(label_group), entry)


def _coerce_rssi_series(rssi_dbm, expected_length):
    """Normalize optional per-packet RSSI metadata to one aligned series."""
    if rssi_dbm is None:
        return None
    series = np.asarray(rssi_dbm)
    if series.ndim == 0:
        return np.full(int(expected_length), int(series.item()), dtype=np.int16)
    if len(series) != int(expected_length):
        return None
    return series


def _packet_rssi_at(rssi_dbm, index):
    """Return one optional RSSI sample from a normalized series."""
    if rssi_dbm is None:
        return None
    if index < 0 or index >= len(rssi_dbm):
        return None
    value = rssi_dbm[index]
    if value is None:
        return None
    return int(value)


def _mapping_optional_value(data, key):
    """Return one optional field from a mapping-like NPZ container."""
    if hasattr(data, "get"):
        return data.get(key)
    try:
        return data[key]
    except (KeyError, IndexError, TypeError):
        return None


def _call_classic_self_baseline_stats(csi_data, packet_rate_pps, **kwargs):
    """Call the idle-baseline helper through its current timing-aware contract."""
    return _classic_self_baseline_stats(csi_data, packet_rate_pps, **kwargs)


def _call_replay_classic_metrics(csi_data, detector, **kwargs):
    """Call Lightweight replay through its current timing-aware contract."""
    return _replay_classic_metrics(csi_data, detector, **kwargs)


def _calibration_packets(
    csi_data,
    rssi_dbm=None,
    *,
    stream_seq_num=None,
    device_ticks_us=None,
    wifi_rx_ts_us=None,
):
    """Yield calibration packets with optional RSSI metadata."""
    normalized_rssi = _coerce_rssi_series(rssi_dbm, len(csi_data))
    normalized_seq = None if stream_seq_num is None else np.asarray(stream_seq_num)
    normalized_ticks = None if device_ticks_us is None else np.asarray(device_ticks_us)
    normalized_wifi = None if wifi_rx_ts_us is None else np.asarray(wifi_rx_ts_us)

    for index, packet in enumerate(csi_data):
        rssi_value = _packet_rssi_at(normalized_rssi, index)
        payload = {"csi_data": packet}
        if rssi_value is not None:
            payload["rssi_dbm"] = rssi_value
        if normalized_seq is not None and index < len(normalized_seq):
            payload["seq_num"] = int(normalized_seq[index])
        if normalized_ticks is not None and index < len(normalized_ticks):
            payload["device_ticks_us"] = int(normalized_ticks[index])
        if normalized_wifi is not None and index < len(normalized_wifi):
            payload["wifi_rx_ts_us"] = int(normalized_wifi[index])
        yield payload


def _replay_classic_metrics(
    csi_data,
    detector,
    *,
    rssi_dbm=None,
    stream_seq_num=None,
    device_ticks_us=None,
    wifi_rx_ts_us=None,
    target_pps=CSI_TARGET_PPS,
):
    """Replay one capture through LightweightDetector at evaluation cadence.

    The detector is reset first so every replay starts from a clean window,
    matching a production boot instead of inheriting the previous stream.
    """
    detector.reset()
    score_series = []
    state_series = []
    target_pps = max(1, int(target_pps))
    nominal_interval_us = nominal_packet_interval_us(target_pps)
    cadence = make_evaluation_cadence(EVALUATION_INTERVAL_MS)
    temporal = TemporalReplayController(
        target_pps,
        SEGMENTATION_WINDOW_SIZE_MS,
        nominal_interval_us,
    )
    normalized_rssi = _coerce_rssi_series(rssi_dbm, len(csi_data))
    normalized_seq = None if stream_seq_num is None else np.asarray(stream_seq_num)
    normalized_ticks = None if device_ticks_us is None else np.asarray(device_ticks_us)
    normalized_wifi = None if wifi_rx_ts_us is None else np.asarray(wifi_rx_ts_us)

    def consume_admission(admission):
        admitted_packet = admission.packet
        if admission.reset_required:
            cadence.reset()
        apply_temporal_admission(detector, admission)
        detector.process_packet(
            admitted_packet["csi_data"],
            DEFAULT_SUBCARRIERS,
            rssi_dbm=admitted_packet["rssi_dbm"],
        )
        cadence.note_packet(elapsed_us=admission.coverage_us)
        if not cadence.should_evaluate():
            return
        metrics = detector.update_state()
        cadence.after_evaluation()
        if detector.is_ready():
            score_series.append(float(metrics.get("motion_metric", 0.0)))
            state_series.append(int(detector.get_state() == MotionState.MOTION))

    for index, packet in enumerate(csi_data):
        packet_view = {
            "csi_data": packet,
            "rssi_dbm": _packet_rssi_at(normalized_rssi, index),
        }
        if normalized_seq is not None and index < len(normalized_seq):
            packet_view["seq_num"] = int(normalized_seq[index])
        if normalized_ticks is not None and index < len(normalized_ticks):
            packet_view["device_ticks_us"] = int(normalized_ticks[index])
        if normalized_wifi is not None and index < len(normalized_wifi):
            packet_view["wifi_rx_ts_us"] = int(normalized_wifi[index])
        admission = temporal.admit(packet_view)
        if admission is None:
            continue
        consume_admission(admission)
    admission = temporal.finish()
    if admission is not None:
        consume_admission(admission)

    return {
        "threshold": float(detector.get_threshold()),
        "score_series": np.asarray(score_series, dtype=np.float64),
        "state_series": np.asarray(state_series, dtype=np.int8),
    }


def _calibrated_classic_for(
    csi_data,
    *,
    rssi_dbm=None,
    stream_seq_num=None,
    device_ticks_us=None,
    wifi_rx_ts_us=None,
    calibration_cache=None,
    cache_key=None,
):
    """Return a (detector, threshold) tuple calibrated on a capture's startup.

    The startup calibration replays packets through the detector in Python and
    is the expensive step, so a pristine calibrated detector snapshot is
    memoized per capture. The full snapshot matters because low-RSSI calibration
    also sets the session L1 floor and noise blend, not only the threshold.

    Keep this path aligned with ``tools.lib.dataset_metadata``: let the
    production-like calibrator walk the full stream so gap-aware restarts can
    recover from a contaminated prefix instead of failing on the first
    configured calibration duration only.
    """
    if calibration_cache is not None and cache_key in calibration_cache:
        calibrated = calibration_cache[cache_key]
        if calibrated is None:
            return None
        return deepcopy(calibrated)

    calibrated = build_calibrated_lightweight_detector(
        _calibration_packets(
            csi_data,
            rssi_dbm=rssi_dbm,
            stream_seq_num=stream_seq_num,
            device_ticks_us=device_ticks_us,
            wifi_rx_ts_us=wifi_rx_ts_us,
        ),
        selected_subcarriers=tuple(DEFAULT_SUBCARRIERS),
    )
    if calibration_cache is not None and cache_key is not None:
        calibration_cache[cache_key] = (
            None if calibrated is None else deepcopy(calibrated)
        )
    return calibrated


def _probability_logit(values):
    """Convert probabilities to finite logits for session-relative margins."""
    probabilities = np.asarray(values, dtype=np.float64)
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _active_burst_metrics(states, packet_rate_pps):
    """Return active burst count/rate and longest duration.

    ``states`` are sampled at the production evaluation cadence, which is now
    treated as elapsed-time driven (about one sample every 250 ms) instead of
    depending on the raw packet rate of the capture.
    """
    padded = np.concatenate([[0], np.asarray(states, dtype=np.int8), [0]])
    edges = np.diff(padded)
    burst_starts = np.flatnonzero(edges == 1)
    burst_lengths = np.flatnonzero(edges == -1) - burst_starts
    burst_count = int(burst_starts.size)
    longest = int(burst_lengths.max()) if burst_count else 0

    del packet_rate_pps
    eval_rate_hz = 1000.0 / float(EVALUATION_INTERVAL_MS)
    eval_seconds = len(states) / eval_rate_hz
    bursts_per_minute = (
        burst_count * 60.0 / eval_seconds if eval_seconds > 0.0 else 0.0
    )
    return {
        "burst_count": burst_count,
        "bursts_per_minute": float(bursts_per_minute),
        "longest_burst_seconds": longest / eval_rate_hz,
        "eval_seconds": float(eval_seconds),
    }


def _classic_self_baseline_stats(
    csi_data,
    packet_rate_pps,
    *,
    rssi_dbm=None,
    stream_seq_num=None,
    device_ticks_us=None,
    wifi_rx_ts_us=None,
    calibration_cache=None,
    cache_key=None,
):
    """Self-calibrate one idle capture and evaluate its post-bootstrap tail."""
    calibration_packets = max(
        1,
        int(round(packet_rate_pps * CALIBRATION_DURATION_MS / 1000.0)),
    )
    if len(csi_data) <= calibration_packets:
        return None

    calibrated = _calibrated_classic_for(
        csi_data,
        rssi_dbm=rssi_dbm,
        stream_seq_num=stream_seq_num,
        device_ticks_us=device_ticks_us,
        wifi_rx_ts_us=wifi_rx_ts_us,
        calibration_cache=calibration_cache,
        cache_key=cache_key,
    )
    if calibrated is None:
        return None
    detector, threshold = calibrated
    replay = _replay_classic_metrics(
        csi_data[calibration_packets:],
        detector,
        target_pps=max(1, int(round(packet_rate_pps))),
        rssi_dbm=(
            None
            if rssi_dbm is None
            else rssi_dbm[calibration_packets:]
        ),
        stream_seq_num=(
            None
            if stream_seq_num is None
            else stream_seq_num[calibration_packets:]
        ),
        device_ticks_us=(
            None
            if device_ticks_us is None
            else device_ticks_us[calibration_packets:]
        ),
        wifi_rx_ts_us=(
            None
            if wifi_rx_ts_us is None
            else wifi_rx_ts_us[calibration_packets:]
        ),
    )
    scores = replay["score_series"]
    if len(scores) == 0:
        return None

    # Every quantity below is measured against this capture's own typical
    # level, never against the calibrated threshold. A startup calibration can
    # land badly, or be recomputed later from the data, and a dataset verdict
    # must not move when it does.
    score_logits = _probability_logit(scores)
    margin_center = float(np.median(score_logits))
    margins = score_logits - margin_center
    margin_median = float(np.median(margins))
    margin_mad = float(np.median(np.abs(margins - margin_median)))

    # Excursions are read against a robust bound built from the capture itself,
    # so "how often does this idle recording leave its own baseline" replaces
    # "how often does it cross the detector's threshold".
    excursion_bound = margin_median + BASELINE_EXCURSION_MADS * max(margin_mad, 1e-9)
    states = (margins > excursion_bound).astype(np.int8)

    eval_rate_hz = 1000.0 / float(EVALUATION_INTERVAL_MS)
    block_size = max(1, int(round(eval_rate_hz * BASELINE_BLOCK_SECONDS)))
    full_block_count = len(margins) // block_size
    if full_block_count:
        block_margins = np.asarray([
            np.median(margins[index * block_size:(index + 1) * block_size])
            for index in range(full_block_count)
        ], dtype=np.float64)
    else:
        block_margins = np.asarray([margin_median], dtype=np.float64)

    split = len(margins) // 2
    margin_drift = (
        float(np.median(margins[split:]) - np.median(margins[:split]))
        if split > 0
        else 0.0
    )
    burst_metrics = _active_burst_metrics(states, packet_rate_pps)
    fp_rate = float(states.mean())
    margin_q95 = float(np.quantile(margins, 0.95))
    score = classic_baseline_score(
        margin_q95,
        burst_metrics["longest_burst_seconds"],
    )
    return {
        "threshold": float(threshold),
        "packet_rate_pps": float(packet_rate_pps),
        "eval_count": int(len(scores)),
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
        **burst_metrics,
    }
