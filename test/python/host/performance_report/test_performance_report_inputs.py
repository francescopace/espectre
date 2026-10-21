# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - Performance Report Inputs Tests

Tests for performance report input collection.

Author: Francesco Pace <francesco.pace@gmail.com>
"""

import json
import sys
from pathlib import Path
from unittest.mock import ANY, Mock

import numpy as np
import pytest

from tools import generate_performance_report
from tools.lib import performance_report, performance_report_inputs
from tools.lib import performance_report_audit as audit


def test_operational_grid_counts_stale_predictions_as_unavailable():
    result = audit.score_timeline(
        np.arange(0, 2_000_001, 250_000),
        [1, 2, 3, 4], [True] * 4, [True] * 4,
    )
    assert result["ticks"] == 8
    # The last prediction covers one more tick, then exceeds the tolerance.
    assert result["available"] == 5
    assert result["raw_positive"] == 5
    # Production holds a confirmed event when later evaluations are unavailable.
    assert result["effective_positive"] == 5
    assert result["alarms"] == 1
    assert result["first_detection_s"] == 1.0


def test_operational_grid_does_not_report_cadence_jitter_as_missing():
    times = np.arange(0, 60_000_001, 10_000)
    indices = np.arange(0, len(times), 26)[1:]
    result = audit.score_timeline(
        times, indices, [False] * len(indices), [True] * len(indices),
    )
    # Only the tick before the first 260 ms evaluation is unavailable.
    assert result["available"] == result["ticks"] - 1


def test_operational_grid_counts_a_real_evaluation_gap_as_unavailable():
    times = np.arange(0, 10_000_001, 250_000)
    indices = [i for i in range(1, len(times)) if not 12 <= i < 20]
    result = audit.score_timeline(
        times, indices, [False] * len(indices), [True] * len(indices),
    )
    # Of eight skipped evaluations, the first tick is still within tolerance.
    assert result["ticks"] - result["available"] == 7


def test_operational_grid_never_uses_a_prediction_before_its_timestamp():
    result = audit.score_timeline(
        np.array([0, 260_000, 510_000, 760_000, 1_010_000, 1_250_000]),
        [1, 2, 3, 4], [True] * 4, [True] * 4,
    )
    assert result["ticks"] == 5
    assert result["available"] == 4
    assert result["raw_positive"] == 4
    assert result["effective_positive"] == 1
    assert result["first_detection_s"] == 1.01


def test_operational_hit_policy_advances_per_evaluation_not_per_tick():
    # Four evaluations inside two ticks still confirm MOTION, as in firmware.
    result = audit.score_timeline(
        np.array([0, 100_000, 200_000, 300_000, 400_000, 500_000]),
        [1, 2, 3, 4], [True] * 4, [True] * 4,
    )
    assert result["alarms"] == 1
    assert result["first_detection_s"] == 0.4


def test_operational_interruptions_exclude_a_trailing_idle_run():
    times = np.arange(0, 6_000_001, 250_000)
    states = [True] * 6 + [False] * 4 + [True] * 6 + [False] * 8
    result = audit.score_timeline(
        times, list(range(1, 25)), states, [True] * 24,
    )
    # MOTION ends at 2.25 s and resumes at 3.5 s; the final IDLE run never resumes.
    assert result["interruptions"] == 1
    assert result["max_interruption_s"] == 1.25


def test_operational_stationary_scoring_excludes_calibration_prefix():
    result = audit.score_timeline(
        np.arange(0, 2_000_001, 250_000),
        list(range(1, 9)), [True] * 4 + [False] * 4, [True] * 8,
        cutoff_us=1_000_000,
    )
    assert result["ticks"] == 4
    assert result["raw_positive"] == 0
    assert result["alarms"] == 0
    assert result["seconds"] == 1.0


def test_operational_summary_retains_a_recording_with_no_available_motion():
    idle = audit.score_timeline(np.array([0, 1_000_000]), [], [], [])
    motion = audit.score_timeline(np.array([0, 1_000_000]), [], [], [])
    summary = audit.summarize([{"static": idle, "motion": motion}])
    assert summary["operational_recall"] == 0.0
    assert summary["conditional_recall"] is None
    assert summary["motion_coverage"] == 0.0
    assert summary["never_detected"] == 1


def test_operational_clock_retains_gaps_and_unwraps_uint32():
    times = audit.packet_times([
        {"device_ticks_us": 2**32 - 10}, {"device_ticks_us": 10},
        {"device_ticks_us": 1_000_010},
    ], 10_000)
    np.testing.assert_array_equal(times, [0, 20, 1_000_020])
    with pytest.raises(ValueError, match="monotonic"):
        audit.packet_times([{"device_ticks_us": 20}, {"device_ticks_us": 10}], 10_000)


def test_calibration_boundary_identifies_only_the_consumed_prefix():
    count = 80
    rows = {
        "X": np.zeros((count, 2)), "ready": np.ones(count, dtype=bool),
        "packet_weight": np.full(count, 25), "reset_index": np.zeros(count),
        "packet_index": np.arange(count) * 25,
    }
    timing = performance_report.derive_detector_timing(10_000)
    result = performance_report._calibrate_classic_replay_rows_with_boundary(rows, timing)
    assert result is not None
    threshold, row = result
    assert 0 < row < count - 1
    assert threshold == performance_report._calibrate_classic_replay_rows(rows, timing)
    rows["ready"][:] = False
    assert performance_report._calibrate_classic_replay_rows_with_boundary(rows, timing) is None


def test_unready_rows_cannot_be_combined_with_row_striding():
    rows = {
        "X": np.zeros((2, 2)), "ready": np.zeros(2, dtype=bool),
        "eligible": np.ones(2, dtype=bool), "reset_index": np.zeros(2),
    }
    detector = performance_report.build_lightweight_detector(
        threshold=1.0, timing=performance_report.derive_detector_timing(10_000),
    )
    with pytest.raises(ValueError, match="striding"):
        performance_report._score_classic_replay_phase_rows(
            rows, detector, row_stride=2, include_unready=True,
        )
    assert performance_report._score_classic_replay_phase_rows(
        rows, detector, include_unready=True,
    ) == [False, False]


def test_source_commit_is_optional_outside_a_git_checkout(monkeypatch):
    def missing_git(*_args, **_kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(generate_performance_report.subprocess, "check_output", missing_git)
    assert generate_performance_report._source_commit() is None


def test_external_catalog_does_not_inherit_first_party_holdout_exposure(monkeypatch, tmp_path):
    monkeypatch.setattr(performance_report, "get_available_paired_datasets", lambda **_: [])
    monkeypatch.setattr(performance_report, "get_available_long_test_dataset_specs", lambda: [])
    monkeypatch.setattr(performance_report, "_load_dataset_info", lambda: {"files": {}})
    monkeypatch.setattr(performance_report, "DATA_DIR", tmp_path)
    assert audit.collect_audit()["exposed_holdout"] is False
    monkeypatch.setattr(performance_report, "DATA_DIR", audit.repo_root() / "data")
    assert audit.collect_audit()["exposed_holdout"] is True


def _stub_report_main_dependencies(monkeypatch, report_data=None):
    """Replace expensive report inputs while preserving main() control flow."""
    data = {} if report_data is None else report_data
    monkeypatch.setattr(
        generate_performance_report.performance_report,
        "configure_dataset_root",
        Mock(),
    )
    monkeypatch.setattr(
        generate_performance_report,
        "collect_extended_report_inputs",
        Mock(return_value=({}, None)),
    )
    monkeypatch.setattr(
        generate_performance_report,
        "_load_cached_report_data",
        Mock(return_value=data),
    )
    monkeypatch.setattr(
        generate_performance_report,
        "dataset_info_revision",
        Mock(return_value="dataset-revision"),
    )
    monkeypatch.setattr(
        generate_performance_report,
        "generated_input_revision",
        Mock(return_value="input-revision"),
    )
    monkeypatch.setattr(
        generate_performance_report,
        "get_available_paired_datasets",
        Mock(return_value=[]),
    )
    monkeypatch.setattr(
        generate_performance_report,
        "get_available_long_test_datasets",
        Mock(return_value=[]),
    )
    monkeypatch.setattr(
        generate_performance_report.performance_report,
        "report_evaluation_view",
        Mock(return_value="HT20/HT-LTF"),
    )
    return data


@pytest.mark.parametrize(("is_current", "expected"), [(True, 0), (False, 1)])
def test_main_check_current_returns_contract_exit_code(
    monkeypatch, tmp_path, capsys, is_current, expected
):
    output = tmp_path / "report.md"
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate_performance_report.py", "--check-current", "--output", str(output)],
    )
    monkeypatch.setattr(
        generate_performance_report.performance_report,
        "configure_dataset_root",
        Mock(),
    )
    monkeypatch.setattr(
        generate_performance_report,
        "generated_report_is_current",
        Mock(return_value=is_current),
    )
    monkeypatch.setattr(
        generate_performance_report,
        "_report_mode_is_current",
        Mock(return_value=True),
    )

    assert generate_performance_report.main() == expected
    stream = capsys.readouterr().out if is_current else capsys.readouterr().err
    assert ("Current:" if is_current else "Stale or missing:") in stream


def test_main_stdout_runs_cpp_parity_for_primary_corpus(monkeypatch, capsys):
    data = _stub_report_main_dependencies(monkeypatch)
    parity = Mock()
    monkeypatch.setattr(generate_performance_report, "verify_cpp_report_parity", parity)
    monkeypatch.setattr(
        generate_performance_report,
        "render_performance_report_markdown",
        Mock(return_value="# generated\n"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate_performance_report.py", "--stdout", "--quiet"],
    )

    assert generate_performance_report.main() == 0
    assert capsys.readouterr().out == "# generated\n"
    parity.assert_called_once_with(data, progress=ANY)


def test_main_external_dataset_uses_external_output_and_skips_cpp_parity(
    monkeypatch, tmp_path, capsys
):
    (tmp_path / "dataset_info.json").write_text('{"files": {}}\n', encoding="utf-8")
    _stub_report_main_dependencies(monkeypatch)
    parity = Mock()
    write_report = Mock(side_effect=lambda output, **_kwargs: Path(output))
    monkeypatch.setattr(generate_performance_report, "verify_cpp_report_parity", parity)
    monkeypatch.setattr(generate_performance_report, "write_performance_report", write_report)
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate_performance_report.py", "--data-dir", str(tmp_path), "--quiet"],
    )

    assert generate_performance_report.main() == 0
    parity.assert_not_called()
    assert write_report.call_args.args[0] == (
        tmp_path / "auto_generated" / "PERFORMANCE_REPORT.md"
    )
    assert "Wrote" in capsys.readouterr().out


def test_main_explicit_skip_bypasses_cpp_parity(monkeypatch):
    _stub_report_main_dependencies(monkeypatch)
    parity = Mock()
    monkeypatch.setattr(generate_performance_report, "verify_cpp_report_parity", parity)
    monkeypatch.setattr(
        generate_performance_report,
        "render_performance_report_markdown",
        Mock(return_value="report"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_performance_report.py",
            "--stdout",
            "--quiet",
            "--skip-cpp-parity-check",
        ],
    )

    assert generate_performance_report.main() == 0
    parity.assert_not_called()


def test_main_propagates_cpp_parity_failure(monkeypatch):
    _stub_report_main_dependencies(monkeypatch)
    monkeypatch.setattr(
        generate_performance_report,
        "verify_cpp_report_parity",
        Mock(side_effect=RuntimeError("parity failed")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate_performance_report.py", "--stdout", "--quiet"],
    )

    with pytest.raises(RuntimeError, match="parity failed"):
        generate_performance_report.main()


def test_external_dataset_root_can_replay_explicit_nonstandard_phy(tmp_path):
    original_root = performance_report.DATA_DIR
    static_dir = tmp_path / "idle"
    motion_dir = tmp_path / "jump"
    static_dir.mkdir()
    motion_dir.mkdir()
    arrays = {
        "csi_data": np.zeros((4, 128), dtype=np.int8),
        "num_subcarriers": np.array(64),
        "label": np.array("static_presence"),
        "phy_mode": np.array(["ht"] * 4),
        "ltf_type": np.array(["lltf"] * 4),
        "channel_width": np.array(["40"] * 4),
    }
    np.savez(static_dir / "static.npz", **arrays)
    arrays["label"] = np.array("motion")
    np.savez(motion_dir / "motion.npz", **arrays)
    (tmp_path / "dataset_info.json").write_text(
        json.dumps(
            {
                "files": {
                    "static_presence": [
                        {
                            "filename": "static.npz",
                            "relative_path": "idle/static.npz",
                            "chip": "ESP32",
                            "subcarriers": 64,
                            "dataset_role": "holdout",
                            "optimal_pair_motion_file": "motion.npz",
                        }
                    ],
                    "motion": [
                        {
                            "filename": "motion.npz",
                            "relative_path": "jump/motion.npz",
                            "chip": "ESP32",
                            "subcarriers": 64,
                            "dataset_role": "holdout",
                            "optimal_pair_static_presence_file": "static.npz",
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    try:
        performance_report.configure_dataset_root(
            tmp_path,
            diagnostic_all_phy=True,
        )
        pairs = performance_report.get_available_paired_datasets(
            roles=performance_report.REPORT_DATASET_ROLES
        )
        static_packets, motion_packets = performance_report.load_real_data_cached(
            pairs[0][0], pairs[0][1]
        )

        assert len(pairs) == 1
        assert len(static_packets) == len(motion_packets) == 4
        assert performance_report.report_evaluation_view() == (
            "all explicit PHY rows (diagnostic)"
        )
    finally:
        performance_report.configure_dataset_root(original_root)


def test_render_includes_synthetic_holdout_and_external_parity_note():
    metrics = {
        "count": 1,
        "recall": 98.0,
        "min_recall": 98.0,
        "precision": 97.0,
        "fp_rate": 2.0,
        "max_fp_rate": 2.0,
        "f1": 97.5,
        "effective_alarms": 1,
    }
    empty = {"classic": {}, "ml": {}}
    report_data = {
        "paired": empty,
        "paired_normal": empty,
        "paired_stress_real": empty,
        "paired_synthetic": {
            "classic": {"ESP32": metrics},
            "ml": {"ESP32": metrics},
        },
        "long_quiet": empty,
        "resources": {},
        "augmentation": None,
    }
    execution_info = {
        "last_update": "2026-08-14",
        "source": "data/untracked/example/dataset_info.json",
        "evaluation_view": "HT20/HT-LTF",
        "dataset_revision": "dataset",
        "input_revision": "inputs",
        "generated_by": "tools/generate_performance_report.py",
        "run_started": "2026-08-14T00:00:00+02:00",
        "run_duration": "1.00s",
        "real_paired_dataset_count": 0,
        "synthetic_paired_dataset_count": 1,
        "long_quiet_dataset_count": 0,
        "cpp_parity_checked": False,
    }

    markdown = performance_report.render_performance_report_markdown(
        report_data,
        execution_info=execution_info,
    )

    assert "Evaluation view: `HT20/HT-LTF`" in markdown
    assert "## Reconstructed or synthetic holdout" in markdown
    assert "C++/Python parity: not run." in markdown


def test_resource_benchmark_executes_cached_binary_each_time(monkeypatch, tmp_path):
    binary = tmp_path / "benchmark"
    binary.write_text("placeholder")
    calls = []

    class Result:
        stdout = 'log line\n{"detectors": {"classic": {}, "ml": {}}}\n'

    monkeypatch.setattr(performance_report_inputs, "_resource_binary", lambda: binary)

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return Result()

    monkeypatch.setattr(performance_report_inputs.subprocess, "run", fake_run)

    first = performance_report_inputs.run_current_resource_benchmark()
    second = performance_report_inputs.run_current_resource_benchmark()

    assert first == second
    assert len(calls) == 2
    assert all(call[0] == [str(binary)] for call in calls)


def test_resource_benchmark_cache_tracks_core_headers():
    core_headers = set(performance_report_inputs.CORE_SOURCE_DIR.glob("*.h"))

    assert core_headers
    assert core_headers.issubset(
        set(performance_report_inputs.RESOURCE_BENCHMARK_DEPENDENCIES)
    )
    assert core_headers.isdisjoint(
        set(performance_report_inputs.RESOURCE_BENCHMARK_SOURCES)
    )


def test_report_cache_tracks_replay_implementation():
    assert (
        performance_report.__file__
        in {str(path) for path in generate_performance_report._replay_input_paths()}
    )


def test_report_pair_filter_keeps_only_selection_and_holdout():
    all_pairs = performance_report.get_available_paired_datasets(synthetic=False)
    reserved_pairs = performance_report.get_available_paired_datasets(
        synthetic=False,
        roles=performance_report.REPORT_DATASET_ROLES,
    )

    assert reserved_pairs
    assert len(reserved_pairs) < len(all_pairs)
    assert all(
        performance_report.get_paired_dataset_role(static_path)
        in performance_report.REPORT_DATASET_ROLES
        for static_path, _motion_path, _num_sc, _chip, _dataset_id in reserved_pairs
    )


def test_report_input_collection_never_starts_training(monkeypatch):
    monkeypatch.setattr(
        performance_report_inputs,
        "run_current_resource_benchmark",
        lambda: {"detectors": {}},
    )
    monkeypatch.setattr(
        performance_report_inputs.npz_cache,
        "load_performance_report_result",
        lambda *_args, **_kwargs: {"recipe": "cached", "rows": []},
    )

    resources, augmentation = performance_report_inputs.collect_extended_report_inputs()

    assert resources == {"detectors": {}}
    assert augmentation["recipe"] == "cached"


def test_reserved_diagnostic_uses_only_reserved_roles_and_two_seed_mix(monkeypatch):
    records = [
        {"role": role, "static_path": f"{role}-static", "motion_path": f"{role}-motion"}
        for role in ("train", "selection", "holdout")
    ]
    classic_calls = []
    ml_paths = []

    monkeypatch.setattr(performance_report_inputs, "_paired_records", lambda: records)
    monkeypatch.setattr("tools.lib.csi_io.load_npz_packet_view", lambda path: [{"path": path}])
    monkeypatch.setattr(
        "tools.lib.ml_training.augmentation.resolve_training_augmentation",
        lambda components: (True, {}, {"components": list(components)}),
    )
    monkeypatch.setattr(
        "tools.lib.ml_training.augmentation.training_packet_augmentation_seeds",
        lambda _config: (101, 202),
    )
    monkeypatch.setattr(
        "tools.lib.ml_training.augmentation._packet_augmentation_stream_provenance",
        lambda _config, seed: {"seed": seed},
    )
    monkeypatch.setattr(
        "tools.lib.ml_training.augmentation._prepare_feature_packets_for_record",
        lambda record, **kwargs: [{"path": record["path"], "seed": kwargs["augmentation_seed"]}],
    )

    def fake_classic_rows(static_path, motion_path, **kwargs):
        classic_calls.append((static_path, motion_path, kwargs["replay_provenance"]["seed"]))
        return {"seed": kwargs["replay_provenance"]["seed"]}

    monkeypatch.setattr(
        "tools.lib.performance_report.load_or_compute_classic_replay_rows",
        fake_classic_rows,
    )
    monkeypatch.setattr(
        "tools.lib.performance_report.compute_classic_row_result",
        lambda _rows, **_kwargs: (0.5, {"tp": 4, "fn": 1, "fp": 1, "tn": 4}),
    )

    def fake_ml_rows(path, **kwargs):
        ml_paths.append(path)
        seed = kwargs["stream_provenance"]["seed"]
        return {
            "X": np.asarray([[seed], [seed + 1]], dtype=np.float32),
            "feature_names": ["turb_autocorr"],
            "packet_index": np.arange(2, dtype=np.int32),
            "evaluation_index": np.arange(2, dtype=np.int32),
            "reset_index": np.zeros(2, dtype=np.int32),
            "evaluation_due": np.ones(2, dtype=bool),
        }

    monkeypatch.setattr(
        "tools.lib.performance_report.load_or_compute_ml_replay_rows", fake_ml_rows
    )
    monkeypatch.setattr(
        "tools.lib.performance_report._compute_ml_row_result",
        lambda static, motion, _threshold: (
            {"tp": len(motion["X"]), "fn": 0, "fp": 0, "tn": len(static["X"])},
            {},
        ),
    )

    result = performance_report_inputs.compute_reserved_augmentation_diagnostic()

    assert result["roles"] == ["selection", "holdout"]
    assert result["seeds"] == [101, 202]
    assert result["pair_count"] == 2
    assert all("train" not in str(call) for call in classic_calls)
    assert all("train" not in str(path) for path in ml_paths)
    assert len(classic_calls) == 4
    assert len(ml_paths) == 8
