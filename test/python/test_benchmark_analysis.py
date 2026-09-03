# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Benchmark Analysis contracts."""

from __future__ import annotations

import pytest

from tools.lib.firmware_benchmark import analysis as bench
from tools.lib.firmware_benchmark import settings as benchmark_settings


@pytest.mark.parametrize(
    ("counter", "label"),
    (
        ("direct_rejected_connections", "rejected connection"),
        ("direct_send_failures", "send failure"),
    ),
)
def test_direct_evidence_fails_when_transport_health_counters_increase(counter, label):
    samples = [
        {
            "host_elapsed_seconds": 0.0,
            "timestamp_ms": 1_000,
            "uptime": 1,
            "csi_admitted_pps": 84.0,
            "csi_occupancy_percent": 84.0,
            "free_memory_kb": 120.0,
            "direct_rejected_connections": 0,
            "direct_send_failures": 0,
        },
        {
            "host_elapsed_seconds": 1.0,
            "timestamp_ms": 2_000,
            "uptime": 2,
            "csi_admitted_pps": 84.0,
            "csi_occupancy_percent": 84.0,
            "free_memory_kb": 120.0,
            "direct_rejected_connections": 0,
            "direct_send_failures": 0,
        },
    ]
    samples[-1][counter] = 1

    _metrics, reasons = bench.analyze_direct_evidence(
        samples,
        [],
        duration_seconds=2,
        require_telemetry=False,
        require_detection_timing=False,
    )

    assert f"Direct transport recorded a {label} during the scored window" in reasons


def test_direct_evidence_counts_censored_attempts_as_failures():
    attempts = [
        {"method": "status", "succeeded": True, "censored": False, "duration_ms": 12.0},
        {
            "method": "diagnostics",
            "succeeded": False,
            "censored": True,
            "duration_ms": 30_000.0,
        },
    ]

    metrics, reasons = bench.analyze_direct_evidence(
        [{"host_elapsed_seconds": 0.0, "timestamp_ms": 1_000, "uptime": 1}],
        [],
        duration_seconds=1,
        require_telemetry=False,
        require_detection_timing=False,
        attempts=attempts,
    )

    assert metrics.direct_request_attempts == 2
    assert metrics.direct_request_failures == 1
    assert metrics.direct_request_censored == 1
    assert "1/2 Direct control attempts failed (1 censored)" in reasons


def test_direct_evidence_uses_device_time_when_host_clock_hides_a_gap():
    samples = [
        {"host_elapsed_seconds": 0.0, "timestamp_ms": 1_000, "uptime": 1},
        {"host_elapsed_seconds": 1.0, "timestamp_ms": 31_000, "uptime": 31},
    ]

    metrics, reasons = bench.analyze_direct_evidence(
        samples,
        [],
        duration_seconds=2,
        require_telemetry=False,
        require_detection_timing=False,
    )

    assert metrics.status_interval_max_ms == 30_000
    assert metrics.status_gap_count == 1
    assert "Direct diagnostics gap reached 30.00s" in reasons

def test_direct_evidence_does_not_treat_host_latency_as_a_runtime_gap():
    samples = [
        {"host_elapsed_seconds": 0.0, "timestamp_ms": 1_000, "uptime": 1},
        {"host_elapsed_seconds": 3.0, "timestamp_ms": 2_000, "uptime": 2},
    ]

    metrics, reasons = bench.analyze_direct_evidence(
        samples,
        [],
        duration_seconds=2,
        require_telemetry=False,
        require_detection_timing=False,
    )

    assert metrics.status_interval_max_ms == 1_000
    assert metrics.status_gap_count == 0
    assert not any("diagnostics gap" in reason for reason in reasons)

def test_direct_evidence_rejects_a_frozen_device_timestamp():
    samples = [
        {"host_elapsed_seconds": 0.0, "timestamp_ms": 1_000, "uptime": 1},
        {"host_elapsed_seconds": 1.0, "timestamp_ms": 1_000, "uptime": 1},
    ]

    _metrics, reasons = bench.analyze_direct_evidence(
        samples,
        [],
        duration_seconds=2,
        require_telemetry=False,
        require_detection_timing=False,
    )

    assert "Direct diagnostics timestamp did not advance in 1 sampled interval(s)" in reasons

def test_direct_evidence_accepts_micro_diagnostics_cadence():
    samples = [
        {"host_elapsed_seconds": 0.0, "timestamp_ms": 1_000, "uptime": 1},
        {"host_elapsed_seconds": 4.5, "timestamp_ms": 6_008, "uptime": 6},
        {"host_elapsed_seconds": 9.0, "timestamp_ms": 10_000, "uptime": 10},
        {"host_elapsed_seconds": 13.5, "timestamp_ms": 14_000, "uptime": 14},
    ]

    metrics, reasons = bench.analyze_direct_evidence(
        samples,
        [],
        duration_seconds=18,
        require_telemetry=False,
        require_detection_timing=False,
        sample_interval_seconds=benchmark_settings.MICRO_DIRECT_DIAGNOSTICS_INTERVAL_SECONDS,
        status_gap_tolerance_ms=benchmark_settings.MICRO_RUNTIME_STATUS_GAP_TOLERANCE_MS,
    )

    assert metrics.status_expected_samples == 4
    assert metrics.status_interval_max_ms == 5_008
    assert metrics.status_gap_count == 0
    assert not any("diagnostics gap" in reason for reason in reasons)


def test_direct_evidence_rejects_micro_diagnostics_refresh_gap():
    samples = [
        {"host_elapsed_seconds": 0.0, "timestamp_ms": 1_000, "uptime": 1},
        {"host_elapsed_seconds": 4.5, "timestamp_ms": 7_000, "uptime": 7},
    ]

    metrics, reasons = bench.analyze_direct_evidence(
        samples,
        [],
        duration_seconds=9,
        require_telemetry=False,
        require_detection_timing=False,
        sample_interval_seconds=benchmark_settings.MICRO_DIRECT_DIAGNOSTICS_INTERVAL_SECONDS,
        status_gap_tolerance_ms=benchmark_settings.MICRO_RUNTIME_STATUS_GAP_TOLERANCE_MS,
    )

    assert metrics.status_interval_max_ms == 6_000
    assert metrics.status_gap_count == 1
    assert "Direct diagnostics gap reached 6.00s" in reasons


def test_direct_evidence_accepts_heap_that_reaches_a_final_plateau():
    samples = [
        {
            "host_elapsed_seconds": float(second),
            "timestamp_ms": second * 1_000,
            "uptime": second,
            "free_memory_kb": 120.0 if second < 35 else 100.0,
        }
        for second in range(60)
    ]

    metrics, reasons = bench.analyze_direct_evidence(
        samples,
        [],
        duration_seconds=60,
        require_telemetry=False,
        require_detection_timing=False,
    )

    assert metrics.heap_free_settled_first == 100 * 1024
    assert metrics.heap_free_settled_last == 100 * 1024
    assert metrics.heap_free_settled_delta_percent == 0.0
    assert not any("free heap did not stabilize" in reason for reason in reasons)


def test_direct_evidence_rejects_heap_that_keeps_declining_in_final_window():
    samples = [
        {
            "host_elapsed_seconds": float(second),
            "timestamp_ms": second * 1_000,
            "uptime": second,
            "free_memory_kb": 100.0 if second < 50 else 90.0,
        }
        for second in range(60)
    ]

    metrics, reasons = bench.analyze_direct_evidence(
        samples,
        [],
        duration_seconds=60,
        require_telemetry=False,
        require_detection_timing=False,
    )

    assert metrics.heap_free_settled_first == 100 * 1024
    assert metrics.heap_free_settled_last == 90 * 1024
    assert metrics.heap_free_settled_delta_percent == -10.0
    assert any("free heap did not stabilize" in reason for reason in reasons)


@pytest.mark.parametrize("duration_seconds", [5, 15])
def test_direct_evidence_rejects_incomplete_heap_stability_windows(duration_seconds):
    samples = [
        {
            "host_elapsed_seconds": float(second),
            "timestamp_ms": second * 1_000,
            "uptime": second,
            "free_memory_kb": 100.0,
        }
        for second in range(duration_seconds)
    ]

    metrics, reasons = bench.analyze_direct_evidence(
        samples,
        [],
        duration_seconds=duration_seconds,
        require_telemetry=False,
        require_detection_timing=False,
    )

    assert metrics.heap_free_settled_first is None
    assert metrics.heap_free_settled_last is None
    assert metrics.heap_free_settled_delta_percent is None
    assert any("two complete consecutive 10-second windows" in reason for reason in reasons)
