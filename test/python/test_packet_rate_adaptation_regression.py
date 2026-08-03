"""
ESPectre - Packet-rate adaptation regression tests

Regression coverage for every explicit static-presence/motion pair whose
metadata reports ``average_packet_rate >= 500``. Each high-rate pair is
decimated to slower effective packet rates, and the test checks that:

- the derived detector timing follows the measured cadence,
- evaluation ticks stay time-based instead of packet-count based, and
- both Classic and ML keep good replay quality on the slower streams.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pytest

from config import DEFAULT_SUBCARRIERS, SEG_WINDOW_SIZE
from conftest import DATA_DIR, DATASET_INFO_PATH
from runtime_policy import derive_detector_timing
from tools.lib.dataset_metadata import measure_packet_interval_us
from tools.lib.performance_report import (
    compute_classic_dataset_result,
    compute_ml_dataset_result,
    compute_classic_packet_result,
    compute_ml_packet_result,
    load_real_data_cached,
)


MINIMUM_SOURCE_AVERAGE_PACKET_RATE = 500.0
TARGET_PPS = (500, 400, 300, 200, 120, 100, 80)
PACKET_RATE_REGRESSION_ENV = "ESPECTRE_RUN_PACKET_RATE_REGRESSION"


@dataclass(frozen=True)
class PacketRateSourcePair:
    pair_id: str
    static_filename: str
    motion_filename: str
    source_pps: int
    average_packet_rate: float


def _dataset_path(label: str, filename: str) -> Path:
    return DATA_DIR / label / filename


def _entry_average_packet_rate(entry: dict[str, object]) -> float:
    value = entry.get("average_packet_rate")
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        resolved = 0.0
    if resolved > 0.0:
        return resolved
    duration_ms = float(entry.get("duration_ms", 0.0) or 0.0)
    num_packets = int(entry.get("num_packets", 0) or 0)
    if duration_ms > 0.0 and num_packets > 0:
        return num_packets * 1000.0 / duration_ms
    return 0.0


def _entry_nominal_packet_rate(entry: dict[str, object]) -> int | None:
    value = entry.get("nominal_packet_rate")
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return None
    if resolved > 0:
        return resolved
    return None


@lru_cache(maxsize=1)
def _source_pairs() -> tuple[PacketRateSourcePair, ...]:
    if not DATASET_INFO_PATH.exists():
        return ()
    with DATASET_INFO_PATH.open("r", encoding="utf-8") as handle:
        dataset_info = json.load(handle)

    motion_by_filename = {
        str(entry.get("filename")): entry
        for entry in dataset_info.get("files", {}).get("motion", [])
        if entry.get("filename")
    }
    pairs: list[PacketRateSourcePair] = []
    for static_entry in dataset_info.get("files", {}).get("static_presence", []):
        static_filename = str(static_entry.get("filename") or "")
        if not static_filename:
            continue
        average_packet_rate = _entry_average_packet_rate(static_entry)
        if average_packet_rate < MINIMUM_SOURCE_AVERAGE_PACKET_RATE:
            continue
        motion_filename = str(static_entry.get("optimal_pair_motion_file") or "")
        if not motion_filename:
            continue
        motion_entry = motion_by_filename.get(motion_filename)
        if motion_entry is None:
            continue
        if _entry_average_packet_rate(motion_entry) < MINIMUM_SOURCE_AVERAGE_PACKET_RATE:
            continue
        source_pps = _entry_nominal_packet_rate(static_entry)
        if source_pps is None:
            continue
        pair_id = Path(static_filename).stem
        pairs.append(
            PacketRateSourcePair(
                pair_id=pair_id,
                static_filename=static_filename,
                motion_filename=motion_filename,
                source_pps=source_pps,
                average_packet_rate=average_packet_rate,
            )
        )
    pairs.sort(key=lambda pair: (pair.source_pps, pair.static_filename, pair.motion_filename))
    return tuple(pairs)


def _pair_params() -> list[object]:
    pairs = _source_pairs()
    if not pairs:
        return [
            pytest.param(
                None,
                marks=pytest.mark.skip(
                    reason="No explicit static_presence/motion pairs with average_packet_rate >= 500"
                ),
                id="no_high_packet_rate_pairs",
            )
        ]
    return [
        pytest.param(
            pair,
            id=f"{pair.source_pps}pps_{pair.pair_id}",
        )
        for pair in pairs
    ]


@lru_cache(maxsize=None)
def _load_source_pair(
    pair_spec: PacketRateSourcePair,
) -> tuple[tuple[dict, ...], tuple[dict, ...]]:
    static_path = _dataset_path("static_presence", pair_spec.static_filename)
    motion_path = _dataset_path("motion", pair_spec.motion_filename)
    assert static_path.exists(), f"Missing regression dataset: {static_path}"
    assert motion_path.exists(), f"Missing regression dataset: {motion_path}"
    return load_real_data_cached(static_path, motion_path)


def _decimate_packets(
    packets: tuple[dict, ...],
    *,
    source_pps: int,
    target_pps: int,
) -> tuple[dict, ...]:
    """Select packets at the target cadence and resequence them contiguously."""
    if target_pps >= source_pps:
        return packets

    stride = float(source_pps) / float(target_pps)
    interval_us = int(round(1_000_000.0 / float(target_pps)))
    decimated = []
    cursor = 0.0
    next_seq_num = None
    next_device_ticks_us = None
    next_wifi_rx_ts_us = None

    while True:
        source_index = int(round(cursor))
        if source_index >= len(packets):
            break

        packet = dict(packets[source_index])
        if next_seq_num is None:
            next_seq_num = int(packet.get("seq_num", packet.get("stream_seq_num", 0)) or 0)
        else:
            next_seq_num += 1
        packet["seq_num"] = next_seq_num
        packet["stream_seq_num"] = next_seq_num
        if "device_ticks_us" in packet and packet["device_ticks_us"] is not None:
            if next_device_ticks_us is None:
                next_device_ticks_us = int(packet["device_ticks_us"])
            else:
                next_device_ticks_us += interval_us
            packet["device_ticks_us"] = next_device_ticks_us
        if "wifi_rx_ts_us" in packet and packet["wifi_rx_ts_us"] is not None:
            if next_wifi_rx_ts_us is None:
                next_wifi_rx_ts_us = int(packet["wifi_rx_ts_us"])
            else:
                next_wifi_rx_ts_us = (next_wifi_rx_ts_us + interval_us) % (1 << 32)
            packet["wifi_rx_ts_us"] = next_wifi_rx_ts_us
        decimated.append(packet)
        cursor += stride

    return tuple(decimated)


@lru_cache(maxsize=None)
def _decimated_pair(
    pair_spec: PacketRateSourcePair,
    target_pps: int,
) -> tuple[tuple[dict, ...], tuple[dict, ...]]:
    static_packets, motion_packets = _load_source_pair(pair_spec)
    return (
        _decimate_packets(static_packets, source_pps=pair_spec.source_pps, target_pps=target_pps),
        _decimate_packets(motion_packets, source_pps=pair_spec.source_pps, target_pps=target_pps),
    )


@lru_cache(maxsize=None)
def _rate_summary(pair_spec: PacketRateSourcePair, target_pps: int) -> dict[str, object]:
    static_packets, motion_packets = _decimated_pair(pair_spec, target_pps)
    interval_us = measure_packet_interval_us(static_packets)
    timing = derive_detector_timing(interval_us)

    if target_pps == 500 and pair_spec.source_pps == 500:
        static_path = _dataset_path("static_presence", pair_spec.static_filename)
        motion_path = _dataset_path("motion", pair_spec.motion_filename)
        classic_dataset_result = compute_classic_dataset_result(
            static_path,
            motion_path,
            tuple(DEFAULT_SUBCARRIERS),
            SEG_WINDOW_SIZE,
        )
        assert classic_dataset_result is not None, "Classic startup calibration failed at 500 pps"
        classic_threshold, classic_metrics = classic_dataset_result
        ml_metrics, _feature_payload = compute_ml_dataset_result(
            static_path,
            motion_path,
            tuple(DEFAULT_SUBCARRIERS),
            SEG_WINDOW_SIZE,
            0.5,
        )
    else:
        classic_result = compute_classic_packet_result(
            static_packets,
            motion_packets,
            tuple(DEFAULT_SUBCARRIERS),
            SEG_WINDOW_SIZE,
        )
        assert classic_result is not None, f"Classic startup calibration failed at {target_pps} pps"
        classic_threshold, classic_metrics = classic_result

        ml_metrics, _feature_payload = compute_ml_packet_result(
            static_packets,
            motion_packets,
            tuple(DEFAULT_SUBCARRIERS),
            SEG_WINDOW_SIZE,
            0.5,
        )

    return {
        "pair_id": pair_spec.pair_id,
        "source_pps": pair_spec.source_pps,
        "average_packet_rate": pair_spec.average_packet_rate,
        "target_pps": target_pps,
        "static_packets": len(static_packets),
        "motion_packets": len(motion_packets),
        "interval_us": interval_us,
        "timing": timing,
        "classic_threshold": classic_threshold,
        "classic": classic_metrics,
        "ml": ml_metrics,
    }


def _format_compact_summary_table(summaries: list[dict[str, object]]) -> str:
    """Return one fixed-width summary table for the full packet-rate sweep."""
    headers = (
        "pair",
        "src",
        "pps",
        "timing",
        "Classic R/FP",
        "ML R/FP",
        "eval idle/motion",
    )
    rows = []
    for summary in summaries:
        timing = summary["timing"]
        classic = summary["classic"]
        ml = summary["ml"]
        rows.append(
            (
                f"{summary['pair_id']}",
                f"{summary['source_pps']}",
                f"{summary['target_pps']}",
                f"w{timing['window_packets']} l{timing['lag']} a{timing['autocorr_lag']} e{timing['evaluation_interval']}",
                f"{classic['recall']:.1f}% / {classic['fp_rate']:.1f}%",
                f"{ml['recall']:.1f}% / {ml['fp_rate']:.1f}%",
                f"{classic['num_baseline']} / {classic['num_movement']}",
            )
        )

    widths = [
        max(len(str(value)) for value in (header, *(row[index] for row in rows)))
        for index, header in enumerate(headers)
    ]

    def render_row(values: tuple[str, ...]) -> str:
        return " | ".join(
            str(value).ljust(widths[index]) for index, value in enumerate(values)
        )

    separator = "-+-".join("-" * width for width in widths)
    body = [render_row(headers), separator]
    body.extend(render_row(row) for row in rows)
    return "\n".join(body)


@pytest.mark.parametrize("pair_spec", _pair_params())
def test_packet_rate_adaptation_regression_matrix(pair_spec: PacketRateSourcePair) -> None:
    """Validate one full packet-rate sweep while reusing cached summaries."""
    if os.environ.get(PACKET_RATE_REGRESSION_ENV, "").strip().lower() not in {"1", "true", "yes"}:
        pytest.skip(
            f"set {PACKET_RATE_REGRESSION_ENV}=1 to run the extended packet-rate regression"
        )

    summaries = [_rate_summary(pair_spec, target_pps) for target_pps in TARGET_PPS]
    baseline_counts = [summary["classic"]["num_baseline"] for summary in summaries]
    movement_counts = [summary["classic"]["num_movement"] for summary in summaries]
    print(
        "\nPacket-rate adaptation summary "
        f"for {pair_spec.pair_id} (nominal={pair_spec.source_pps} pps, "
        f"average={pair_spec.average_packet_rate:.1f} pps)"
    )
    print(_format_compact_summary_table(summaries))

    assert min(baseline_counts) >= 675
    assert max(baseline_counts) <= 720
    assert max(baseline_counts) - min(baseline_counts) <= 30

    assert min(movement_counts) >= 335
    assert max(movement_counts) <= 360
    assert max(movement_counts) - min(movement_counts) <= 20

    baseline = summaries[0]
    for summary, target_pps in zip(summaries, TARGET_PPS):
        timing = summary["timing"]
        expected_interval_us = int(round(1_000_000.0 / float(target_pps)))
        expected = derive_detector_timing(expected_interval_us)

        assert abs(int(summary["interval_us"]) - expected_interval_us) <= 1, (
            f"{target_pps} pps measured interval {summary['interval_us']} "
            f"instead of {expected_interval_us}"
        )

        for field in ("window_packets", "lag", "autocorr_lag", "evaluation_interval"):
            expected_value = expected[field]
            assert timing[field] == expected_value, (
                f"{target_pps} pps resolved {field}={timing[field]} "
                f"instead of {expected_value}"
            )

        if target_pps == 500:
            continue

        classic = summary["classic"]
        ml = summary["ml"]
        if pair_spec.source_pps <= 500:
            assert classic["recall"] >= 95.0, (
                f"Classic recall regressed at {target_pps} pps: {classic['recall']:.1f}%"
            )
            assert classic["fp_rate"] <= 1.0, (
                f"Classic FP rate regressed at {target_pps} pps: {classic['fp_rate']:.1f}%"
            )
            assert ml["recall"] >= 95.0, (
                f"ML recall regressed at {target_pps} pps: {ml['recall']:.1f}%"
            )
            assert ml["fp_rate"] <= 1.0, (
                f"ML FP rate regressed at {target_pps} pps: {ml['fp_rate']:.1f}%"
            )
            continue

        assert classic["recall"] >= max(90.0, baseline["classic"]["recall"] - 2.0), (
            f"Classic recall regressed at {target_pps} pps: {classic['recall']:.1f}% "
            f"(baseline {baseline['classic']['recall']:.1f}%)"
        )
        assert classic["fp_rate"] <= max(1.0, baseline["classic"]["fp_rate"] + 1.0), (
            f"Classic FP rate regressed at {target_pps} pps: {classic['fp_rate']:.1f}% "
            f"(baseline {baseline['classic']['fp_rate']:.1f}%)"
        )
        assert ml["recall"] >= max(88.0, baseline["ml"]["recall"] - 2.0), (
            f"ML recall regressed at {target_pps} pps: {ml['recall']:.1f}% "
            f"(baseline {baseline['ml']['recall']:.1f}%)"
        )
        assert ml["fp_rate"] <= max(1.0, baseline["ml"]["fp_rate"] + 1.0), (
            f"ML FP rate regressed at {target_pps} pps: {ml['fp_rate']:.1f}% "
            f"(baseline {baseline['ml']['fp_rate']:.1f}%)"
        )
