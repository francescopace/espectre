# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - Timing Quality Helpers

Shared host-side timing provenance helpers for training, replay reporting, and
dataset validation.

Author: Francesco Pace <francesco.pace@gmail.com>
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from statistics import mean
from typing import Any

from .bootstrap import setup_paths

setup_paths()

from runtime_policy import PacketTimingTracker, nominal_packet_interval_us


MIN_CAPTURE_PACKET_RATE_PPS = 95.0
MAX_STREAM_SEQ_MISSING_WARN_RATIO = 0.01
MAX_STREAM_SEQ_MISSING_FAIL_RATIO = 0.03
MAX_STREAM_SEQ_GAP_WARN_PACKETS = 10
MAX_STREAM_SEQ_GAP_FAIL_PACKETS = 20
MAX_INTER_PACKET_GAP_WARN_MS = 150.0
MAX_INTER_PACKET_GAP_FAIL_MS = 250.0
TIMING_CONTAMINATION_WARN_RATIO = 0.01
TIMING_CONTAMINATION_FAIL_RATIO = 0.03

TIMING_STATUS_RANK = {
    "unknown": 0,
    "PASS": 1,
    "WARN": 2,
    "FAIL": 3,
}

TIMING_BUCKET_BY_STATUS = {
    "unknown": "unknown",
    "PASS": "clean",
    "WARN": "degraded",
    "FAIL": "poor",
}


def _packet_field(packet: Any, key: str) -> Any:
    """Return one optional field from a dict-like packet or packet object."""
    if isinstance(packet, Mapping):
        value = packet.get(key)
        if value is None and key == "seq_num":
            return packet.get("stream_seq_num")
        return value
    return getattr(packet, key, None)


def summarize_capture_timing(
    packets: Iterable[Any],
    *,
    nominal_interval_us: int | None = None,
) -> dict[str, Any]:
    """Summarize one capture's timing provenance and replay contamination risk."""
    packet_list = list(packets)
    nominal = (
        nominal_packet_interval_us(100)
        if nominal_interval_us is None
        else max(1, int(nominal_interval_us))
    )
    tracker = PacketTimingTracker(nominal)

    has_sequence_metadata = False
    has_timestamp_metadata = False
    contaminated_packets = 0
    missing_sequence_packets = 0
    max_sequence_gap_packets = 0
    max_gap_ms = 0.0
    clean_timed_deltas_us: list[int] = []

    for packet in packet_list:
        if _packet_field(packet, "seq_num") is not None:
            has_sequence_metadata = True
        if (
            _packet_field(packet, "device_ticks_us") is not None
            or _packet_field(packet, "wifi_rx_ts_us") is not None
        ):
            has_timestamp_metadata = True

        timing = tracker.observe_packet(packet)
        missing_seq = int(timing["missing_seq"])
        missing_sequence_packets += missing_seq
        max_sequence_gap_packets = max(max_sequence_gap_packets, missing_seq)
        max_gap_ms = max(max_gap_ms, float(timing["delta_us"]) / 1000.0)
        if timing["contaminated"]:
            contaminated_packets += 1
            continue
        if timing["source"] != "missing":
            clean_timed_deltas_us.append(int(timing["delta_us"]))

    total_packets = len(packet_list)
    sequence_intervals = max(0, total_packets - 1) if has_sequence_metadata else 0
    produced_packets = total_packets + missing_sequence_packets
    missing_sequence_ratio = (
        float(missing_sequence_packets) / float(produced_packets)
        if produced_packets > 0
        else 0.0
    )
    contaminated_ratio = (
        float(contaminated_packets) / float(total_packets)
        if total_packets > 0
        else 0.0
    )

    interval_us = (
        max(1, int(round(float(mean(clean_timed_deltas_us)))))
        if clean_timed_deltas_us
        else nominal
    )
    packet_rate_pps = 1_000_000.0 / float(interval_us)

    has_timing_metadata = has_sequence_metadata or has_timestamp_metadata
    if not has_timing_metadata:
        quality_status = "unknown"
        reasons = ["missing timing metadata"]
    else:
        failed = []
        warned = []
        if missing_sequence_ratio > MAX_STREAM_SEQ_MISSING_FAIL_RATIO:
            failed.append("sequence loss ratio")
        elif missing_sequence_ratio > MAX_STREAM_SEQ_MISSING_WARN_RATIO:
            warned.append("sequence loss ratio")

        if max_sequence_gap_packets > MAX_STREAM_SEQ_GAP_FAIL_PACKETS:
            failed.append("sequence gap")
        elif max_sequence_gap_packets > MAX_STREAM_SEQ_GAP_WARN_PACKETS:
            warned.append("sequence gap")

        if max_gap_ms > MAX_INTER_PACKET_GAP_FAIL_MS:
            failed.append("inter-packet gap")
        elif max_gap_ms > MAX_INTER_PACKET_GAP_WARN_MS:
            warned.append("inter-packet gap")

        if contaminated_ratio > TIMING_CONTAMINATION_FAIL_RATIO:
            failed.append("runtime contamination")
        elif contaminated_ratio > TIMING_CONTAMINATION_WARN_RATIO:
            warned.append("runtime contamination")

        if failed:
            quality_status = "FAIL"
            reasons = failed
        elif warned:
            quality_status = "WARN"
            reasons = warned
        else:
            quality_status = "PASS"
            reasons = []

    return {
        "total_packets": total_packets,
        "has_sequence_metadata": has_sequence_metadata,
        "has_timestamp_metadata": has_timestamp_metadata,
        "has_timing_metadata": has_timing_metadata,
        "sequence_intervals": sequence_intervals,
        "missing_sequence_packets": missing_sequence_packets,
        "missing_sequence_ratio": float(missing_sequence_ratio),
        "max_sequence_gap_packets": max_sequence_gap_packets,
        "contaminated_packets": contaminated_packets,
        "contaminated_ratio": float(contaminated_ratio),
        "max_gap_ms": float(max_gap_ms),
        "interval_us": interval_us,
        "packet_rate_pps": float(packet_rate_pps),
        "quality_status": quality_status,
        "quality_bucket": TIMING_BUCKET_BY_STATUS[quality_status],
        "quality_reasons": tuple(sorted(dict.fromkeys(reasons))),
    }


def merge_timing_summaries(*summaries: Mapping[str, Any]) -> dict[str, Any]:
    """Return one worst-case summary over multiple capture timing records."""
    valid = [dict(summary) for summary in summaries if summary]
    if not valid:
        return {
            "capture_count": 0,
            "quality_status": "unknown",
            "quality_bucket": "unknown",
            "quality_reasons": tuple(),
            "packet_rate_pps": 0.0,
            "contaminated_ratio": 0.0,
            "max_gap_ms": 0.0,
        }

    worst_status = max(
        (str(summary.get("quality_status", "unknown")) for summary in valid),
        key=lambda status: TIMING_STATUS_RANK.get(status, 0),
    )
    reasons = []
    for summary in valid:
        reasons.extend(summary.get("quality_reasons", ()))

    return {
        "capture_count": len(valid),
        "quality_status": worst_status,
        "quality_bucket": TIMING_BUCKET_BY_STATUS.get(worst_status, "unknown"),
        "quality_reasons": tuple(sorted(dict.fromkeys(str(reason) for reason in reasons))),
        "packet_rate_pps": sum(float(summary.get("packet_rate_pps", 0.0)) for summary in valid)
        / float(len(valid)),
        "contaminated_ratio": sum(
            float(summary.get("contaminated_ratio", 0.0)) for summary in valid
        )
        / float(len(valid)),
        "max_gap_ms": max(float(summary.get("max_gap_ms", 0.0)) for summary in valid),
    }
