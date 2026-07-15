"""
ESPectre - Filter Optimization Tests

Tests for the variance filter optimization tool.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
MODULE_PATH = TOOLS_DIR / "optimize_filter_params.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("optimize_filter_params_tool", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_lowpass_sweep_configs_covers_missing_filter_controls() -> None:
    module = _load_module()

    configs = module.build_lowpass_sweep_configs()
    filter_cfgs = [cfg for _label, cfg in configs]

    assert configs[0][0] == "Production baseline"
    assert len(filter_cfgs) == len(set(filter_cfgs))

    assert any(not cfg.enable_hampel and not cfg.enable_lowpass for cfg in filter_cfgs)
    assert any(cfg.enable_hampel and not cfg.enable_lowpass for cfg in filter_cfgs)

    lowpass_only_cutoffs = sorted(
        cfg.lowpass_cutoff
        for cfg in filter_cfgs
        if not cfg.enable_hampel and cfg.enable_lowpass
    )
    hampel_lowpass_cutoffs = sorted(
        cfg.lowpass_cutoff
        for cfg in filter_cfgs
        if cfg.enable_hampel and cfg.enable_lowpass
    )

    expected_cutoffs = [5.0, 7.0, 9.0, 11.0, 13.0, 15.0]
    assert lowpass_only_cutoffs == expected_cutoffs
    assert hampel_lowpass_cutoffs == expected_cutoffs


def test_select_best_rows_separates_target_winner_from_best_f1() -> None:
    module = _load_module()

    rows = [
        {
            "label": "Higher F1 but misses recall target",
            "summary": {"recall": 94.8, "precision": 95.0, "fp_rate": 2.5, "f1": 94.9},
        },
        {
            "label": "Hits targets with lower F1",
            "summary": {"recall": 95.0, "precision": 92.0, "fp_rate": 4.0, "f1": 93.5},
        },
        {
            "label": "Hits targets but worse F1",
            "summary": {"recall": 95.2, "precision": 91.0, "fp_rate": 4.5, "f1": 93.0},
        },
    ]

    best_by_target, best_by_f1 = module.select_best_rows(rows)

    assert best_by_target["label"] == "Hits targets with lower F1"
    assert best_by_f1["label"] == "Higher F1 but misses recall target"


def test_describe_filter_context_explains_lowpass_base() -> None:
    module = _load_module()

    assert (
        module.describe_filter_context(enable_lowpass=True, lowpass_cutoff=7.0)
        == "base low-pass ON at 7.0 Hz"
    )
    assert (
        module.describe_filter_context(enable_lowpass=False, lowpass_cutoff=11.0)
        == "base low-pass OFF"
    )
