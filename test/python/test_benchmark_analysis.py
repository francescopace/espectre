# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Benchmark Analysis contracts."""

from __future__ import annotations

from tools.lib.firmware_benchmark import analysis as bench
from tools.lib.firmware_benchmark import settings as benchmark_settings


def test_direct_evidence_fails_when_transport_health_counters_increase():
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
            "direct_slow_client_disconnects": 0,
        },
        {
            "host_elapsed_seconds": 1.0,
            "timestamp_ms": 2_000,
            "uptime": 2,
            "csi_admitted_pps": 84.0,
            "csi_occupancy_percent": 84.0,
            "free_memory_kb": 120.0,
            "direct_rejected_connections": 0,
            "direct_send_failures": 1,
            "direct_slow_client_disconnects": 0,
        },
    ]

    _metrics, reasons = bench.analyze_direct_evidence(
        samples,
        [],
        duration_seconds=2,
        require_telemetry=False,
        require_detection_timing=False,
    )

    assert "Direct transport recorded a send failure during the scored window" in reasons

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
        {"host_elapsed_seconds": 5.25, "timestamp_ms": 6_000, "uptime": 6},
        {"host_elapsed_seconds": 10.5, "timestamp_ms": 11_000, "uptime": 11},
    ]

    metrics, reasons = bench.analyze_direct_evidence(
        samples,
        [],
        duration_seconds=15,
        require_telemetry=False,
        require_detection_timing=False,
        sample_interval_seconds=benchmark_settings.MICRO_DIRECT_DIAGNOSTICS_INTERVAL_SECONDS,
    )

    assert metrics.status_expected_samples == 3
    assert metrics.status_interval_max_ms == 5_000
    assert metrics.status_gap_count == 0
    assert not any("diagnostics gap" in reason for reason in reasons)
