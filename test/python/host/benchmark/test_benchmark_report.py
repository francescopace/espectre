# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Benchmark Report contracts."""

from __future__ import annotations

from datetime import datetime
import json

import pytest
from tools.lib.firmware_benchmark import report as bench
from tools.lib.firmware_benchmark.models import (
    BenchmarkCase,
    BenchmarkResult,
    CommandResult,
    RepositoryState,
    RuntimeMetrics,
)


def test_benchmark_source_fingerprint_covers_split_owners():
    assert "tools/lib/firmware_benchmark" in bench.BENCHMARK_SOURCE_PATHS


def test_cpp_artifacts_store_only_normalized_direct_evidence(tmp_path):
    case = BenchmarkCase("native", "lightweight")
    result = BenchmarkResult(case=case, status="PASS")
    result.flash = CommandResult(
        ["espectre", "native", "flash", "--port", "/dev/cu.usb", "--target", "192.168.1.50"],
        0,
        2.0,
        "MAC: AA:BB:CC:DD:EE:FF connected to 192.168.1.50\n",
    )
    result.direct_samples = [{"host_elapsed_seconds": 1.0, "uptime": 7, "free_memory_kb": 120.0}]
    result.direct_attempts = [
        {
            "method": "status",
            "duration_ms": 30_000.0,
            "failed_phase": "body",
            "response_bytes": 177,
            "expected_response_bytes": 693,
            "censored": True,
            "succeeded": False,
            "error_type": "TimeoutError",
        }
    ]
    result.transport_evidence = {
        "transport": "http",
        "origin": "https://test.espectre.dev",
        "request_path": "/espectre/v1",
        "events_path": "/espectre/v1/events",
    }

    bench.write_benchmark_artifacts(
        tmp_path,
        chip="c3",
        port="/dev/cu.usb",
        started_at=datetime.fromisoformat("2026-08-22T12:00:00+02:00"),
        results=[result],
    )

    case_dir = tmp_path / "native-lightweight"
    serialized = (case_dir / "analysis.json").read_text(encoding="utf-8")
    manifest = (tmp_path / "manifest.json").read_text(encoding="utf-8")
    assert not (case_dir / "flash.log").exists()
    assert not (case_dir / "flash.jsonl").exists()
    assert "192.168.1.50" not in serialized + manifest
    assert "AA:BB:CC:DD:EE:FF" not in serialized + manifest
    assert '"transport": "http"' in serialized
    analysis = json.loads(serialized)
    assert analysis["direct_attempts"] == result.direct_attempts
def test_parse_report_results_accepts_na_packet_rate():
    text = """### Native High Accuracy

Result: **FAIL**

| Metric | Value |
|---|---:|
| Benchmark mode | runtime |
| Packet rate | N/A mean, N/A min, N/A max, N/A standard deviation |
| CSI occupancy | 0.00% mean, 0% min, 0% max |
| Status samples | 60/60 expected |

Failure reasons:

- mean CSI occupancy 0.0% is below the 70% detector-ready floor
"""

    results = bench.parse_report_results(text)

    assert len(results) == 1
    assert results[0].case.frontend == "native"
    assert results[0].case.detector == "high_accuracy"
    assert results[0].status == "FAIL"
    assert results[0].runtime_metrics.pps_mean is None
    assert results[0].runtime_metrics.occupancy_mean == 0.0
    assert results[0].runtime_metrics.status_samples == 60

def test_parse_report_results_reads_micro_deploy_metrics():
    text = """### Micro-ESPectre Lightweight

Result: **PASS**

| Metric | Value |
|---|---:|
| Benchmark mode | runtime |
| Deploy duration | 2.5s |
| Firmware binary | 1,024 bytes (1.0 KiB) |
| Deployed Python source | 2,048 bytes (2.0 KiB) |
"""

    results = bench.parse_report_results(text)

    assert results[0].deploy is not None
    assert results[0].deploy.duration_seconds == 2.5
    assert results[0].build_metrics.firmware_size_bytes == 1_024
    assert results[0].build_metrics.deployed_source_bytes == 2_048

def test_parse_report_results_skips_removed_legacy_case():
    text = """### Micro-ESPectre High Accuracy

Result: **FAIL**

| Metric | Value |
|---|---:|
| Benchmark mode | runtime |

### Micro-ESPectre Lightweight

Result: **PASS**

| Metric | Value |
|---|---:|
| Benchmark mode | runtime |
"""

    results = bench.parse_report_results(text)

    assert len(results) == 1
    assert results[0].case.frontend == "micro"
    assert results[0].case.detector == "lightweight"

def test_parse_report_results_rejects_unknown_case():
    with pytest.raises(ValueError, match="unknown benchmark case label"):
        bench.parse_report_results("### Unknown Legacy Case\n")

def test_report_round_trip_preserves_reboot_and_settled_heap_diagnostics():
    case = BenchmarkCase("micro", "lightweight")
    result = BenchmarkResult(case=case, status="FAIL")
    result.monitor = CommandResult(["monitor"], 0, 60.0, "")
    result.runtime_metrics = RuntimeMetrics(
        status_samples=58,
        direct_request_attempts=120,
        direct_request_failures=2,
        direct_request_censored=1,
        status_expected_samples=60,
        status_interval_mean_ms=1_050.0,
        status_interval_max_ms=3_000,
        status_gap_count=2,
        serial_framing_anomalies=3,
        device_reboots=1,
        heap_free_last=140_000,
        heap_free_settled_first=150_000,
        heap_free_settled_last=140_000,
        heap_free_settled_delta=-10_000,
        heap_free_settled_delta_percent=-6.6667,
        heap_free_post_gc_last=155_000,
        verified_detector="lightweight",
        packet_processing_samples=240,
        packet_processing_avg_us_mean=2_100.0,
        packet_processing_min_us=1_500,
        packet_processing_max_us=3_200,
        gc_pause_us_mean=4_250.0,
        gc_pause_us_max=4_800,
    )

    rendered = bench.render_report(
        "c3",
        "/dev/cu.test",
        datetime.fromisoformat("2026-08-22T12:00:00+02:00"),
        [result],
        [case],
    )
    parsed = bench.parse_report_results(rendered)[0].runtime_metrics

    assert parsed.device_reboots == 1
    assert parsed.direct_request_attempts == 120
    assert parsed.direct_request_failures == 2
    assert parsed.direct_request_censored == 1
    assert parsed.status_gap_count == 2
    assert parsed.serial_framing_anomalies == 0
    assert "Serial framing anomalies" not in rendered
    assert parsed.status_interval_mean_ms == 1_050.0
    assert parsed.heap_free_settled_first == 150_000
    assert parsed.heap_free_settled_delta == -10_000
    assert parsed.heap_free_settled_delta_percent == -6.67
    assert parsed.heap_free_post_gc_last == 155_000
    assert parsed.verified_detector == "lightweight"
    assert parsed.packet_processing_samples == 240
    assert parsed.packet_processing_avg_us_mean == 2_100.0
    assert parsed.packet_processing_max_us == 3_200
    assert parsed.gc_pause_us_mean == 4_250.0
    assert parsed.gc_pause_us_max == 4_800


def test_report_keeps_bssid_evidence_out_of_the_summary():
    case = BenchmarkCase("esphome", "lightweight")
    result = BenchmarkResult(case=case, status="PASS")
    result.transport_evidence = {
        "bssid_provisioning": {
            "requested": True,
            "applied": True,
            "already_associated": True,
            "reassociation_exercised": True,
            "verified": True,
        }
    }

    rendered = bench.render_report(
        "c3",
        "/dev/cu.test",
        datetime.fromisoformat("2026-08-22T12:00:00+02:00"),
        [result],
        [case],
    )
    parsed = bench.parse_report_results(rendered)[0]

    assert "| Frontend BSSID setup |" not in rendered
    assert "| Frontend setup final BSSID reassociation exercised | yes |" in rendered
    assert "BSSID reboot observed" not in rendered
    assert parsed.status == "PASS"
    assert parsed.transport_evidence["bssid_provisioning"] == result.transport_evidence["bssid_provisioning"]

def test_micro_artifacts_do_not_persist_runtime_serial_output(tmp_path, monkeypatch):
    monkeypatch.setenv("ESPECTRE_BENCHMARK_WIFI_PASSWORD", "super-secret-password")
    case = BenchmarkCase("micro", "lightweight")
    result = BenchmarkResult(case=case, status="PASS")
    result.monitor = CommandResult(
        ["monitor"],
        0,
        2.0,
        "I (1000) connected super-secret-password\nI (2000) ready\n",
        line_elapsed_seconds=[0.5, 1.5],
        analysis_start_line=1,
    )

    bench.write_benchmark_artifacts(
        tmp_path,
        chip="c3",
        port="/dev/cu.test",
        started_at=datetime.fromisoformat("2026-08-22T12:00:00+02:00"),
        results=[result],
    )

    case_dir = tmp_path / "micro-lightweight"
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert not (case_dir / "monitor.log").exists()
    assert not (case_dir / "monitor.jsonl").exists()
    assert manifest["cases"][0]["commands"]["monitor"]["returncode"] == 0
    assert manifest["schema_version"] == bench.BENCHMARK_ARTIFACT_SCHEMA_VERSION

def test_benchmark_artifacts_preserve_starting_source_provenance(tmp_path, monkeypatch):
    case = BenchmarkCase("esphome", "lightweight")
    result = BenchmarkResult(case=case, status="FAIL")
    state_start = RepositoryState("aaaaaaaaaaaa", True, "source-start")
    state_end = RepositoryState("bbbbbbbbbbbb", False, "source-end")
    monkeypatch.setattr(bench, "repository_state", lambda: state_end)

    bench.write_benchmark_artifacts(
        tmp_path,
        chip="c3",
        port="/dev/cu.test",
        started_at=datetime.fromisoformat("2026-08-26T12:49:37+02:00"),
        results=[result],
        repository_state_start=state_start,
        source_changed_during_run=True,
    )

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["git_revision"] == "aaaaaaaaaaaa"
    assert manifest["git_revision_end"] == "bbbbbbbbbbbb"
    assert manifest["git_revision_changed"] is True
    assert manifest["git_source_fingerprint"] == "source-start"
    assert manifest["git_source_fingerprint_end"] == "source-end"
    assert manifest["git_source_changed_during_run"] is True
    assert manifest["git_worktree_dirty"] is True
    assert manifest["git_worktree_dirty_end"] is False

def test_benchmark_source_provenance_only_invalidates_revision_changes():
    state_start = RepositoryState("aaaaaaaaaaaa", False, "source-start")

    assert bench.benchmark_revision_provenance_reason(state_start, state_start) is None
    assert bench.benchmark_revision_provenance_reason(
        state_start,
        RepositoryState("aaaaaaaaaaaa", True, "source-end"),
    ) is None
    assert bench.benchmark_revision_provenance_reason(
        state_start,
        RepositoryState("bbbbbbbbbbbb", False, "source-end"),
    ) == (
        "benchmark source provenance is invalid: Git revision changed from aaaaaaaaaaaa to "
        "bbbbbbbbbbbb during the run"
    )

def test_benchmark_source_changes_are_reported_as_warnings():
    state_start = RepositoryState("aaaaaaaaaaaa", True, "source-start")
    state_end = RepositoryState("aaaaaaaaaaaa", True, "source-end")
    case = BenchmarkCase("esphome", "lightweight")
    result = BenchmarkResult(case=case, status="PASS")

    assert bench.benchmark_source_change_warning(state_start, state_start) is None
    assert bench.benchmark_source_change_warning(state_start, state_end) == (
        "firmware or benchmark sources changed during the run"
    )

    bench.render_report(
        "c3",
        "/dev/cu.test",
        datetime.fromisoformat("2026-08-26T12:49:37+02:00"),
        [result],
        [case],
        repository_state_start=state_start,
        repository_state_end=state_end,
        source_changed_during_run=True,
    )


def test_s2_report_criteria_omit_unsupported_frontends():
    cases = (
        BenchmarkCase("native", "lightweight"),
        BenchmarkCase("esphome", "lightweight"),
    )
    results = [BenchmarkResult(case=case, status="PASS") for case in cases]

    rendered = bench.render_report(
        "s2",
        "/dev/cu.test",
        datetime.fromisoformat("2026-08-30T12:00:00+02:00"),
        results,
        cases,
    )
    pass_criteria = rendered.split("## Pass Criteria", maxsplit=1)[1]

    assert "Micro-ESPectre" not in pass_criteria
    assert "Matter" not in pass_criteria
    assert "Native and ESPHome negotiate Direct v1" in pass_criteria
