"""
ESPectre - Seed Dispersion Analysis Tests

Tests for tools/analyze_seed_dispersion.py.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_SCRIPT = REPO_ROOT / "tools" / "analyze_seed_dispersion.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("analyze_seed_dispersion", TOOL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _run(seed, fp_rate, alarms=0, weak=False):
    return {
        "seed": seed,
        "paired": {
            "by_chip": {
                "C6:selection:normal.npz": {
                    "fp_rate": fp_rate, "recall": 100.0, "effective_alarms": alarms,
                    "static_presence_eval_count": 685, "motion_eval_count": 350,
                },
                "S3:holdout:weak.npz": {
                    "fp_rate": 9.0, "recall": 95.0, "effective_alarms": 0,
                    "static_presence_eval_count": 685, "motion_eval_count": 350,
                    "low_rssi": weak,
                },
                # No reserved pair for this chip: the gate falls back to a
                # chip-level aggregate over training data.
                "ESP32": {
                    "fp_rate": 0.0, "recall": 100.0, "effective_alarms": 0,
                    "static_presence_eval_count": 700, "motion_eval_count": 350,
                },
            }
        },
    }


def test_dispersion_reports_spread_in_evaluations_not_percent():
    """Percentages hide the scale the gate margin is expressed in."""
    module = _load_tool()

    runs = [_run(1, 0.4380), _run(2, 1.0219), _run(3, 0.7299)]
    rows = {row["replay"]: row for row in module.dispersion_rows(runs, "fp_rate")}

    normal = rows["C6:selection:normal.npz"]
    assert normal["seeds"] == 3
    assert normal["min_events"] == 3
    assert normal["max_events"] == 7
    assert normal["min_alarms"] == normal["max_alarms"] == 0


def test_chip_level_fallback_rows_are_not_counted_as_replays():
    """A bare chip row is training data, so it cannot answer a reserved question."""
    module = _load_tool()

    rows = {row["replay"]: row for row in
            module.dispersion_rows([_run(1, 0.5), _run(2, 0.5)], "fp_rate")}

    assert rows["ESP32"]["reserved"] is False
    assert rows["C6:selection:normal.npz"]["reserved"] is True
    assert rows["S3:holdout:weak.npz"]["reserved"] is True


def test_weak_and_normal_links_are_kept_apart():
    """The low_rssi exemption is a separate question and needs separate data."""
    module = _load_tool()

    weak_rows = module.dispersion_rows([_run(1, 0.5, weak=True), _run(2, 0.5, weak=True)],
                                       "fp_rate")
    by_replay = {row["replay"]: row for row in weak_rows}
    assert by_replay["S3:holdout:weak.npz"]["weak"] is True
    assert by_replay["C6:selection:normal.npz"]["weak"] is False


def test_alarm_movement_is_reported_separately_from_rate_spread():
    """Rate jitter and a new alarm are different findings; do not merge them."""
    module = _load_tool()

    runs = [_run(1, 0.5, alarms=0), _run(2, 0.5, alarms=1)]
    normal = next(row for row in module.dispersion_rows(runs, "fp_rate")
                  if row["replay"] == "C6:selection:normal.npz")

    assert normal["min_events"] == normal["max_events"]
    assert (normal["min_alarms"], normal["max_alarms"]) == (0, 1)


def test_seed_search_report_shape_is_understood():
    """The seed-search JSON nests runs differently from the experiment report."""
    module = _load_tool()

    payload = {
        "baseline": {"seed": 100, "selection_paired_metrics": _run(100, 0.5)["paired"]},
        "trials": [
            {"seed": 1, "paired_metrics": _run(1, 0.7)["paired"]},
            {"seed": 2, "status": "export_failed"},
        ],
    }
    groups = list(module.collect_runs(payload))

    assert len(groups) == 1
    label, runs = groups[0]
    assert label == "seed search"
    # The failed trial carries no metrics and must not become a phantom seed.
    assert [run["seed"] for run in runs] == [100, 1]
