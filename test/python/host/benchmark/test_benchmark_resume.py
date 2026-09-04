# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Benchmark Resume contracts."""

from __future__ import annotations

import sys

import pytest

from tools import benchmark_firmware as bench
from tools.lib.firmware_benchmark import report as benchmark_report
from tools.lib.firmware_benchmark.models import (
    BenchmarkCase,
    BenchmarkResult,
    CASES,
    CommandResult,
    RepositoryState,
)


def test_cases_run_frontends_in_hardware_benchmark_order():
    frontends = [case.frontend for case in CASES]

    assert frontends == [
        "native",
        "native",
        "esphome",
        "esphome",
        "matter",
        "matter",
        "micro",
    ]

def test_main_executes_frontends_in_hardware_benchmark_order(tmp_path, monkeypatch):
    observed: list[str] = []
    observed_ports: list[str] = []
    state = RepositoryState("revision", False, "fingerprint")

    def run_direct(cases, _chip, port, *, on_result):
        observed.append(cases[0].frontend)
        observed_ports.append(port)
        direct_results = [BenchmarkResult(case=case, status="PASS") for case in cases]
        for result in direct_results:
            on_result(result)
        return direct_results

    def run_micro(case, _chip, port, **_kwargs):
        observed.append(case.frontend)
        observed_ports.append(port)
        return BenchmarkResult(case=case, status="PASS")

    monkeypatch.setattr(
        sys,
        "argv",
        ["benchmark_firmware.py", "--chip", "c5", "--port", "/dev/cu.explicit"],
    )
    monkeypatch.setattr(bench, "repository_state", lambda: state)
    monkeypatch.setattr(bench, "benchmark_artifact_dir", lambda *_args: tmp_path / "artifacts")
    monkeypatch.setattr(bench, "require_benchmark_prerequisites", lambda _cases: None)
    monkeypatch.setattr(bench, "run_direct_frontend_cases_safely", run_direct)
    monkeypatch.setattr(bench, "run_micro_case", run_micro)
    monkeypatch.setattr(bench, "write_report", lambda *_args, **_kwargs: tmp_path / "report.md")
    monkeypatch.setattr(bench, "write_benchmark_artifacts", lambda *_args, **_kwargs: None)

    assert bench.main() == 0
    assert observed == ["native", "esphome", "matter", "micro"]
    assert observed_ports == ["/dev/cu.explicit"] * 4


def test_main_passes_explicit_port_unchanged(tmp_path, monkeypatch):
    observed_ports: list[str] = []
    state = RepositoryState("revision", False, "fingerprint")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_firmware.py",
            "--chip",
            "c5",
            "--frontend",
            "native",
            "--port",
            "/dev/cu.requested",
        ],
    )
    monkeypatch.setattr(bench, "repository_state", lambda: state)
    monkeypatch.setattr(bench, "benchmark_artifact_dir", lambda *_args: tmp_path / "artifacts")
    monkeypatch.setattr(bench, "require_benchmark_prerequisites", lambda _cases: None)
    monkeypatch.setattr(
        bench,
        "run_direct_frontend_cases_safely",
        lambda _cases, _chip, port, **_kwargs: observed_ports.append(port) or [],
    )
    monkeypatch.setattr(bench, "write_report", lambda *_args, **_kwargs: tmp_path / "report.md")
    monkeypatch.setattr(bench, "write_benchmark_artifacts", lambda *_args, **_kwargs: None)

    assert bench.main() == 1
    assert observed_ports == ["/dev/cu.requested"]


def test_main_stops_after_first_failed_case(tmp_path, monkeypatch, capsys):
    observed: list[str] = []
    state = RepositoryState("revision", False, "fingerprint")

    def run_direct(cases, _chip, _port, *, on_result):
        observed.append(cases[0].frontend)
        flash = CommandResult(["flash"], 0, 1.0, "Flash completed")
        direct_results = [
            BenchmarkResult(
                case=case,
                status="FAIL",
                reasons=["provisioning exited with status 1"],
                flash=flash,
            )
            for case in cases
        ]
        for result in direct_results:
            on_result(result)
        return direct_results

    def fail_micro(*_args, **_kwargs):
        raise AssertionError("Micro must not run after a flash failure")

    monkeypatch.setattr(
        sys,
        "argv",
        ["benchmark_firmware.py", "--chip", "c5", "--port", "/dev/cu.test"],
    )
    monkeypatch.setattr(bench, "repository_state", lambda: state)
    monkeypatch.setattr(bench, "benchmark_artifact_dir", lambda *_args: tmp_path / "artifacts")
    monkeypatch.setattr(bench, "require_benchmark_prerequisites", lambda _cases: None)
    monkeypatch.setattr(bench, "run_direct_frontend_cases_safely", run_direct)
    monkeypatch.setattr(bench, "run_micro_case", fail_micro)
    monkeypatch.setattr(bench, "write_report", lambda *_args, **_kwargs: tmp_path / "report.md")
    monkeypatch.setattr(bench, "write_benchmark_artifacts", lambda *_args, **_kwargs: None)

    assert bench.main() == 1
    assert observed == ["native"]
    assert "stopping the benchmark at the first failed case" in capsys.readouterr().err

def test_main_warns_but_passes_when_sources_change_on_same_revision(
    tmp_path,
    monkeypatch,
    capsys,
):
    state_start = RepositoryState("revision", True, "source-start")
    state_end = RepositoryState("revision", True, "source-end")
    states = iter((state_start, state_end, state_end, state_end, state_end))
    reports: list[tuple[list[str], bool, str]] = []

    def run_direct(cases, _chip, _port, *, on_result):
        direct_results = [BenchmarkResult(case=case, status="PASS") for case in cases]
        for result in direct_results:
            on_result(result)
        return direct_results

    def write_report(_chip, _port, _started_at, results, _expected_cases, **kwargs):
        rendered = benchmark_report.render_report(
            _chip,
            _port,
            _started_at,
            results,
            _expected_cases,
            **kwargs,
        )
        reports.append(
            (
                [result.status for result in results],
                kwargs["source_changed_during_run"],
                rendered,
            )
        )
        return tmp_path / "report.md"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_firmware.py",
            "--chip",
            "c5",
            "--frontend",
            "native",
            "--port",
            "/dev/cu.test",
        ],
    )
    monkeypatch.setattr(bench, "repository_state", lambda: next(states))
    monkeypatch.setattr(bench, "benchmark_artifact_dir", lambda *_args: tmp_path / "artifacts")
    monkeypatch.setattr(bench, "require_benchmark_prerequisites", lambda _cases: None)
    monkeypatch.setattr(bench, "run_direct_frontend_cases_safely", run_direct)
    monkeypatch.setattr(bench, "write_report", write_report)
    monkeypatch.setattr(bench, "write_benchmark_artifacts", lambda *_args, **_kwargs: None)

    assert bench.main() == 0
    assert [(statuses, changed) for statuses, changed, _rendered in reports] == [
        (["PASS"], True),
        (["PASS", "PASS"], True),
        (["PASS", "PASS"], True),
        (["PASS", "PASS"], True),
    ]
    assert all("Source consistency: **WARNING**" in rendered for _, _, rendered in reports)
    assert all("Source fingerprint: `source-start` → `source-end`" in rendered for _, _, rendered in reports)
    assert capsys.readouterr().err.count("WARNING: firmware or benchmark sources changed") == 1

def test_cases_include_micro_espectre_lightweight_only():
    labels = [case.label for case in CASES]

    assert "Micro-ESPectre Lightweight" in labels
    assert "Micro-ESPectre High Accuracy" not in labels

def test_s2_cases_exclude_only_unsupported_matter_frontend():
    labels = [case.label for case in bench.select_cases(chip="s2")]

    assert "Matter Lightweight" not in labels
    assert "Matter High Accuracy" not in labels
    assert "Native Lightweight" in labels
    assert "Micro-ESPectre Lightweight" in labels
    assert "ESPHome Lightweight" in labels

def test_resume_selects_only_failed_and_missing_requested_cases():
    native_lightweight = BenchmarkCase("native", "lightweight")
    native_high_accuracy = BenchmarkCase("native", "high_accuracy")
    micro_lightweight = BenchmarkCase("micro", "lightweight")
    existing_results = [
        BenchmarkResult(case=native_lightweight, status="PASS"),
        BenchmarkResult(case=native_high_accuracy, status="FAIL"),
    ]

    selected = bench.select_resume_cases(
        (native_lightweight, native_high_accuracy, micro_lightweight),
        existing_results,
    )

    assert selected == (native_high_accuracy, micro_lightweight)

def test_resume_expected_cases_include_existing_and_requested_cases():
    native_lightweight = BenchmarkCase("native", "lightweight")
    micro_lightweight = BenchmarkCase("micro", "lightweight")
    existing_results = [BenchmarkResult(case=native_lightweight, status="PASS")]

    expected = bench.expected_preserved_cases(existing_results, (micro_lightweight,))

    assert expected == (native_lightweight, micro_lightweight)

def test_resume_with_no_failed_or_missing_cases_does_not_access_hardware(
    tmp_path,
    monkeypatch,
    capsys,
):
    report_path = tmp_path / "ESP32-C3.md"
    report_path.write_text(
        """### Micro-ESPectre Lightweight

Result: **PASS**

| Metric | Value |
|---|---:|
| Benchmark mode | runtime |
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(bench, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(bench, "report_path_for_chip", lambda _chip: report_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_firmware.py",
            "--chip",
            "c3",
            "--frontend",
            "micro",
            "--port",
            "/dev/cu.test",
            "--resume",
        ],
    )

    assert bench.main() == 0
    assert "no failed or missing selected cases" in capsys.readouterr().out


def test_main_requires_explicit_port(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["benchmark_firmware.py", "--chip", "s3"])

    with pytest.raises(SystemExit) as exc_info:
        bench.main()

    assert exc_info.value.code == 2
