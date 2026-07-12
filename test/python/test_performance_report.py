"""
Tests for the shared performance-report helpers.
"""

from pathlib import Path

from tools.lib import performance_report as report


def _fake_report_data():
    return {
        "paired": {
            "classic": {
                "C3": {"recall": 98.0, "precision": 99.3, "fp_rate": 0.3, "f1": 98.6},
                "C5": {"recall": 99.9, "precision": 100.0, "fp_rate": 0.0, "f1": 100.0},
            },
            "ml": {
                "C3": {"recall": 99.8, "precision": 100.0, "fp_rate": 0.0, "f1": 99.9},
                "S3": {"recall": 100.0, "precision": 100.0, "fp_rate": 0.0, "f1": 100.0},
            },
        },
        "long_quiet": {
            "classic": {
                "C3": {"avg_fp_rate": 0.30, "max_fp_rate": 0.42},
                "S3": {"avg_fp_rate": 1.20, "max_fp_rate": 1.20},
            },
            "ml": {
                "C3": {"avg_fp_rate": 0.00, "max_fp_rate": 0.00},
                "S3": {"avg_fp_rate": 0.13, "max_fp_rate": 0.13},
            },
        },
    }


def test_render_performance_report_markdown_formats_missing_values_as_na() -> None:
    markdown = report.render_performance_report_markdown(_fake_report_data())

    assert markdown.startswith("# Performance Metrics\n")
    assert "| Recall | 98.0% | 99.9% | N/A | N/A |" in markdown
    assert "| Recall | 99.8% | N/A | N/A | 100.0% |" in markdown
    assert "| Avg FP Rate | 0.30% | N/A | N/A | 1.20% |" in markdown
    assert "| Max FP Rate | 0.00% | N/A | N/A | 0.13% |" in markdown


def test_write_performance_report_writes_rendered_markdown(tmp_path, monkeypatch) -> None:
    output_path = tmp_path / "PERFORMANCE.md"
    monkeypatch.setattr(report, "compute_performance_report_data", _fake_report_data)

    written_path = report.write_performance_report(output_path)

    assert written_path == output_path
    assert output_path.read_text(encoding="utf-8") == report.render_performance_report_markdown(_fake_report_data())
    assert output_path.parent == Path(tmp_path)
