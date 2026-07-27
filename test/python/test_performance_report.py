"""
ESPectre - Performance Report Tests

Tests for the shared performance-report helpers.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import sys
from pathlib import Path

import numpy as np
import tools.generate_performance_report as generate_report
from tools.lib import cpp_parity
from tools.lib import performance_report as report


class _FakeTimingTracker:
    def __init__(self, coverage_us: int, contaminated: bool = False) -> None:
        self.coverage_us = coverage_us
        self.contaminated = contaminated

    def observe_packet(self, _packet):
        return {
            "coverage_us": self.coverage_us,
            "contaminated": self.contaminated,
        }


def _fake_report_data():
    return {
        "paired": {
            "classic": {
                "C3": {
                    "count": 1,
                    "recall": 98.0,
                    "min_recall": 98.0,
                    "precision": 99.3,
                    "fp_rate": 0.3,
                    "max_fp_rate": 0.3,
                    "f1": 98.6,
                    "effective_alarms": 1,
                },
                "C5": {
                    "count": 1,
                    "recall": 99.9,
                    "min_recall": 99.9,
                    "precision": 100.0,
                    "fp_rate": 0.0,
                    "max_fp_rate": 0.0,
                    "f1": 100.0,
                    "effective_alarms": 0,
                },
            },
            "ml": {
                "C3": {
                    "count": 1,
                    "recall": 99.8,
                    "min_recall": 99.8,
                    "precision": 100.0,
                    "fp_rate": 0.0,
                    "max_fp_rate": 0.0,
                    "f1": 99.9,
                    "effective_alarms": 0,
                },
                "S3": {
                    "count": 1,
                    "recall": 100.0,
                    "min_recall": 100.0,
                    "precision": 100.0,
                    "fp_rate": 0.0,
                    "max_fp_rate": 0.0,
                    "f1": 100.0,
                    "effective_alarms": 2,
                },
            },
        },
        "paired_ml_roles": {
            "reserved": {
                "C3": {
                    "count": 1,
                    "recall": 99.8,
                    "min_recall": 99.8,
                    "precision": 100.0,
                    "fp_rate": 0.0,
                    "max_fp_rate": 0.0,
                    "f1": 99.9,
                    "effective_alarms": 0,
                },
                "S3": {
                    "count": 1,
                    "recall": 100.0,
                    "min_recall": 100.0,
                    "precision": 100.0,
                    "fp_rate": 0.0,
                    "max_fp_rate": 0.0,
                    "f1": 100.0,
                    "effective_alarms": 2,
                },
            },
            "train": {
                "C3": {
                    "count": 2,
                    "recall": 99.9,
                    "min_recall": 99.9,
                    "precision": 99.8,
                    "fp_rate": 0.2,
                    "max_fp_rate": 0.2,
                    "f1": 99.8,
                    "effective_alarms": 0,
                },
            },
        },
        "paired_classic_normal": {
            "C3": {
                "count": 1,
                "recall": 98.0,
                "min_recall": 98.0,
                "precision": 99.3,
                "fp_rate": 0.3,
                "max_fp_rate": 0.3,
                "f1": 98.6,
                "effective_alarms": 1,
            },
            "C5": {
                "count": 1,
                "recall": 99.9,
                "min_recall": 99.9,
                "precision": 100.0,
                "fp_rate": 0.0,
                "max_fp_rate": 0.0,
                "f1": 100.0,
                "effective_alarms": 0,
            },
        },
        "paired_stress_real": {
            "classic": {
                "S3": {
                    "count": 1,
                    "recall": 96.3,
                    "min_recall": 96.3,
                    "precision": 94.0,
                    "fp_rate": 6.3,
                    "max_fp_rate": 6.3,
                    "f1": 92.2,
                    "effective_alarms": 7,
                },
            },
            "ml": {
                "S3": {
                    "count": 1,
                    "recall": 92.8,
                    "min_recall": 92.8,
                    "precision": 95.5,
                    "fp_rate": 4.4,
                    "max_fp_rate": 4.4,
                    "f1": 92.0,
                    "effective_alarms": 7,
                },
            },
        },
        "paired_synthetic": {
            "classic": {
                "C3": {
                    "count": 1,
                    "recall": 87.0,
                    "min_recall": 87.0,
                    "precision": 99.0,
                    "fp_rate": 0.5,
                    "max_fp_rate": 0.5,
                    "f1": 92.6,
                    "effective_alarms": 1,
                },
            },
            "ml": {},
        },
        "long_quiet": {
            "classic": {
                "C3": {
                    "count": 1,
                    "min_recall": 97.5,
                    "avg_fp_rate": 0.30,
                    "max_fp_rate": 0.42,
                    "effective_alarms": 2,
                },
                "S3": {
                    "count": 1,
                    "min_recall": 94.0,
                    "avg_fp_rate": 1.20,
                    "max_fp_rate": 1.20,
                    "effective_alarms": 1,
                },
            },
            "ml": {
                "C3": {
                    "count": 1,
                    "min_recall": 99.0,
                    "avg_fp_rate": 0.00,
                    "max_fp_rate": 0.00,
                    "effective_alarms": 0,
                },
                "S3": {
                    "count": 1,
                    "min_recall": 98.7,
                    "avg_fp_rate": 0.13,
                    "max_fp_rate": 0.13,
                    "effective_alarms": 0,
                },
            },
        },
    }


def _fake_execution_info():
    return {
        "last_update": "2026-07-14",
        "source": "data/dataset_info.json",
        "generated_by": "tools/generate_performance_report.py",
        "run_started": "2026-07-14T15:43:00+02:00",
        "run_duration": "12.34s",
        "real_paired_dataset_count": 4,
        "synthetic_paired_dataset_count": 2,
        "long_quiet_dataset_count": 11,
    }


def test_note_evaluation_tick_resets_time_aware_cadence() -> None:
    cadence = report.RuntimeMotionPolicy(
        evaluation_interval=25,
        evaluation_interval_us=30,
    )
    tracker = _FakeTimingTracker(coverage_us=10)

    assert report.note_evaluation_tick(cadence, packet={}, timing_tracker=tracker) == (False, False)
    assert report.note_evaluation_tick(cadence, packet={}, timing_tracker=tracker) == (False, False)
    assert report.note_evaluation_tick(cadence, packet={}, timing_tracker=tracker) == (True, False)
    assert report.note_evaluation_tick(cadence, packet={}, timing_tracker=tracker) == (False, False)


def test_render_performance_report_markdown_formats_missing_values_as_na() -> None:
    markdown = report.render_performance_report_markdown(_fake_report_data())

    assert markdown.startswith("<!-- Generated file. Do not edit manually. -->\n")
    assert "\n# Performance Metrics\n" in markdown
    assert "| Recall | 98.0% | 99.9% | N/A | N/A | N/A |" in markdown
    assert "| Min Recall | 98.0% | 99.9% | N/A | N/A | N/A |" in markdown
    assert "| Recall | 99.8% | N/A | N/A | N/A | 100.0% |" in markdown
    assert "| Max FP Rate | 0.3% | 0.0% | N/A | N/A | N/A |" in markdown
    assert "| Effective Alarms | 1 | 0 | N/A | N/A | N/A |" in markdown
    assert "| Effective Alarms | 0 | N/A | N/A | N/A | 2 |" in markdown
    assert "| Min Recall | 99.0% | N/A | N/A | N/A | 98.7% |" in markdown
    assert "| Avg FP Rate | 0.30% | N/A | N/A | N/A | 1.20% |" in markdown
    assert "| Max FP Rate | 0.00% | N/A | N/A | N/A | 0.13% |" in markdown
    assert "| Effective Alarms | 2 | N/A | N/A | N/A | 1 |" in markdown
    assert "| Metric | ESP32-C3 | ESP32-C5 | ESP32-C6 | ESP32 | ESP32-S3 |" in markdown
    assert "| Metric | C3 | C5 | C6 | ESP32 | S3 |" in markdown
    assert "False Motion Evals" not in markdown
    assert "Per-chip live firmware reports" in markdown
    assert "also verifies that the host-side C++ integration suites stay aligned" in markdown
    assert "## Low-RSSI Stress Validation" in markdown


def test_render_performance_report_markdown_splits_ml_by_provenance() -> None:
    markdown = report.render_performance_report_markdown(_fake_report_data())

    assert "### ML Detector — Reserved Replays (out-of-sample)" in markdown
    assert "### ML Detector — Training Recordings (in-sample diagnostic)" in markdown
    reserved_index = markdown.index("Reserved Replays")
    train_index = markdown.index("Training Recordings")
    assert reserved_index < train_index
    reserved_section = markdown[reserved_index:train_index]
    train_section = markdown[train_index:markdown.index("## Low-RSSI Stress Validation")]
    assert "| Recall | 99.8% | N/A | N/A | N/A | 100.0% |" in reserved_section
    assert "| Recall | 99.9% | N/A | N/A | N/A | N/A |" in train_section


def test_load_long_test_dataset_uses_zero_copy_packet_view(monkeypatch, tmp_path) -> None:
    csi_matrix = np.arange(6 * 128, dtype=np.int8).reshape(6, 128)
    rssi_dbm = np.array([-80, -79, -78, -77, -76, -75], dtype=np.int16)

    monkeypatch.setattr(
        report,
        "load_npz_arrays",
        lambda _path: {"csi_data": csi_matrix, "rssi_dbm": rssi_dbm},
    )
    monkeypatch.setattr(report, "filter_npz_arrays_sensing", lambda arrays: arrays)
    report._load_long_test_packets_cached.cache_clear()

    spec = (tmp_path / "long_test.npz", 4, 6, "C3", {"chip": "C3"})
    _, baseline_packets, movement_packets, motion_start_packet, chip, _entry = report.load_long_test_dataset(spec)

    first_packet = baseline_packets[0]
    assert motion_start_packet == 4
    assert chip == "C3"
    assert len(baseline_packets) == 4
    assert len(movement_packets) == 2
    assert isinstance(first_packet, dict)
    assert isinstance(first_packet["csi_data"], memoryview)
    assert first_packet["rssi_dbm"] == -80
    assert movement_packets[0]["rssi_dbm"] == -76


def test_render_performance_report_markdown_reports_link_class_split() -> None:
    markdown = report.render_performance_report_markdown(_fake_report_data())

    assert "(static_presence / motion, normal link)" in markdown
    assert "### Real Weak-Link Pairs — Classic Detector (report-only)" in markdown
    assert "### Real Weak-Link Pairs — ML Detector" in markdown
    stress_index = markdown.index("## Low-RSSI Stress Validation")
    stress_section = markdown[stress_index:]
    assert "| Recall | N/A | N/A | N/A | N/A | 96.3% |" in stress_section
    assert "| Recall | N/A | N/A | N/A | N/A | 92.8% |" in stress_section
    assert "recall >90% and FP <10%" in stress_section


def test_render_performance_report_markdown_includes_execution_info_when_provided() -> None:
    markdown = report.render_performance_report_markdown(
        _fake_report_data(),
        execution_info=_fake_execution_info(),
    )

    assert "Last update: 2026-07-14" in markdown
    assert "Source: `data/dataset_info.json`" in markdown
    assert "Generated by: `tools/generate_performance_report.py`" in markdown
    assert "Run started: `2026-07-14T15:43:00+02:00`" in markdown
    assert "Run duration: `12.34s`" in markdown
    assert (
        "Inputs: `4` real paired datasets, `2` synthetic paired datasets, "
        "`11` long quiet datasets"
    ) in markdown


def test_write_performance_report_writes_rendered_markdown(tmp_path, monkeypatch) -> None:
    output_path = tmp_path / "PERFORMANCE.md"
    monkeypatch.setattr(report, "compute_performance_report_data", _fake_report_data)

    written_path = report.write_performance_report(output_path)

    assert written_path == output_path
    assert output_path.read_text(encoding="utf-8") == report.render_performance_report_markdown(_fake_report_data())
    assert output_path.parent == Path(tmp_path)


def test_write_performance_report_uses_provided_report_data(tmp_path, monkeypatch) -> None:
    output_path = tmp_path / "PERFORMANCE.md"

    def _unexpected_compute():
        raise AssertionError("compute_performance_report_data should not run when report_data is provided")

    monkeypatch.setattr(report, "compute_performance_report_data", _unexpected_compute)

    written_path = report.write_performance_report(
        output_path,
        report_data=_fake_report_data(),
        execution_info=_fake_execution_info(),
    )

    assert written_path == output_path
    assert output_path.read_text(encoding="utf-8") == report.render_performance_report_markdown(
        _fake_report_data(),
        execution_info=_fake_execution_info(),
    )


def test_compare_cpp_and_python_report_data_accepts_matching_payloads() -> None:
    cpp_report_data = {
        "paired": {
            "classic": {
                "C3": {
                    "count": 1,
                    "recall": 98.0,
                    "min_recall": 98.0,
                    "precision": 99.3,
                    "fp_rate": 0.3,
                    "max_fp_rate": 0.3,
                    "f1": 98.6,
                    "effective_alarms": 1,
                },
                "C5": {
                    "count": 1,
                    "recall": 99.9,
                    "min_recall": 99.9,
                    "precision": 100.0,
                    "fp_rate": 0.0,
                    "max_fp_rate": 0.0,
                    "f1": 100.0,
                    "effective_alarms": 0,
                },
            },
            "ml": {
                "C3": {
                    "count": 1,
                    "recall": 99.8,
                    "min_recall": 99.8,
                    "precision": 100.0,
                    "fp_rate": 0.0,
                    "max_fp_rate": 0.0,
                    "f1": 99.9,
                    "effective_alarms": 0,
                },
                "S3": {
                    "count": 1,
                    "recall": 100.0,
                    "min_recall": 100.0,
                    "precision": 100.0,
                    "fp_rate": 0.0,
                    "max_fp_rate": 0.0,
                    "f1": 100.0,
                    "effective_alarms": 2,
                },
            },
        },
        "paired_synthetic": {
            "classic": {
                "C3": {
                    "count": 1,
                    "recall": 87.0,
                    "min_recall": 87.0,
                    "precision": 99.0,
                    "fp_rate": 0.5,
                    "max_fp_rate": 0.5,
                    "f1": 92.6,
                    "effective_alarms": 1,
                },
            },
            "ml": {},
        },
        "long_quiet": {
            "classic": {
                "C3": {
                    "count": 1,
                    "min_recall": 97.5,
                    "avg_fp_rate": 0.30,
                    "max_fp_rate": 0.42,
                    "effective_alarms": 2,
                },
                "S3": {
                    "count": 1,
                    "min_recall": 94.0,
                    "avg_fp_rate": 1.20,
                    "max_fp_rate": 1.20,
                    "effective_alarms": 1,
                },
            },
            "ml": {
                "C3": {
                    "count": 1,
                    "min_recall": 99.0,
                    "avg_fp_rate": 0.00,
                    "max_fp_rate": 0.00,
                    "effective_alarms": 0,
                },
                "S3": {
                    "count": 1,
                    "min_recall": 98.7,
                    "avg_fp_rate": 0.13,
                    "max_fp_rate": 0.13,
                    "effective_alarms": 0,
                },
            },
        },
    }

    assert cpp_parity.compare_cpp_and_python_report_data(_fake_report_data(), cpp_report_data) == []


def test_compare_cpp_and_python_report_data_reports_drift() -> None:
    cpp_report_data = {
        "paired": {
            "classic": {
                "C3": {
                    "count": 1,
                    "recall": 97.0,
                    "min_recall": 97.0,
                    "precision": 99.3,
                    "fp_rate": 0.3,
                    "max_fp_rate": 0.3,
                    "f1": 98.6,
                    "effective_alarms": 1,
                },
            },
            "ml": {},
        },
        "long_quiet": {
            "classic": {
                "C3": {
                    "count": 1,
                    "min_recall": 97.5,
                    "avg_fp_rate": 0.30,
                    "max_fp_rate": 0.42,
                    "effective_alarms": 3,
                },
            },
            "ml": {},
        },
    }

    mismatches = cpp_parity.compare_cpp_and_python_report_data(_fake_report_data(), cpp_report_data)

    assert any("paired/classic/C3/recall" in mismatch for mismatch in mismatches)
    assert any("long_quiet/classic/C3/effective_alarms" in mismatch for mismatch in mismatches)
    assert any("paired/ml/C3: missing c++ metrics" in mismatch for mismatch in mismatches)


def test_generate_performance_report_main_runs_cpp_parity_before_write(monkeypatch, tmp_path) -> None:
    calls = []
    output_path = tmp_path / "PERFORMANCE.md"
    fake_report = _fake_report_data()

    def _fake_compute(*, progress=None):
        calls.append("compute")
        return fake_report

    def _fake_verify(report_data, *, progress=None, build_dir=None):
        calls.append("verify")
        assert report_data is fake_report
        return {"paired": {}, "long_quiet": {}}

    def _fake_write(path, *, report_data=None, progress=None, execution_info=None):
        calls.append("write")
        assert path == output_path
        assert report_data is fake_report
        assert execution_info["real_paired_dataset_count"] == 2
        assert execution_info["synthetic_paired_dataset_count"] == 0
        assert execution_info["long_quiet_dataset_count"] == 1
        return output_path

    monkeypatch.setattr(generate_report, "compute_performance_report_data", _fake_compute)
    monkeypatch.setattr(generate_report, "verify_cpp_report_parity", _fake_verify)
    monkeypatch.setattr(generate_report, "write_performance_report", _fake_write)
    monkeypatch.setattr(
        generate_report,
        "get_available_paired_datasets",
        lambda *, synthetic=None: [1, 2] if synthetic is False else [],
    )
    monkeypatch.setattr(generate_report, "get_available_long_test_datasets", lambda: [1])
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate_performance_report.py", "--output", str(output_path), "--quiet"],
    )

    assert generate_report.main() == 0
    assert calls == ["compute", "verify", "write"]


def test_generate_performance_report_main_can_skip_cpp_parity(monkeypatch, tmp_path) -> None:
    calls = []
    output_path = tmp_path / "PERFORMANCE.md"
    fake_report = _fake_report_data()

    def _fake_compute(*, progress=None):
        calls.append("compute")
        return fake_report

    def _fake_verify(report_data, *, progress=None, build_dir=None):
        calls.append("verify")
        assert report_data is fake_report
        return {"paired": {}, "long_quiet": {}}

    def _fake_write(path, *, report_data=None, progress=None, execution_info=None):
        calls.append("write")
        assert path == output_path
        assert report_data is fake_report
        return output_path

    monkeypatch.setattr(generate_report, "compute_performance_report_data", _fake_compute)
    monkeypatch.setattr(generate_report, "verify_cpp_report_parity", _fake_verify)
    monkeypatch.setattr(generate_report, "write_performance_report", _fake_write)
    monkeypatch.setattr(
        generate_report,
        "get_available_paired_datasets",
        lambda *, synthetic=None: [1, 2] if synthetic is False else [],
    )
    monkeypatch.setattr(generate_report, "get_available_long_test_datasets", lambda: [1])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_performance_report.py",
            "--output",
            str(output_path),
            "--quiet",
            "--skip-cpp-parity-check",
        ],
    )

    assert generate_report.main() == 0
    assert calls == ["compute", "write"]
