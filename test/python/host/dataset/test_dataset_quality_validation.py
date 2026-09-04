# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Tests for dataset-quality primitives called by the supported collect CLI."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from tools.lib.dataset_quality import (
    capture,
    catalog,
    core,
    metrics,
    pairing,
    references,
    rendering,
    replay,
    severity,
)
from tools.lib.repo_paths import repo_root


VALIDATOR_PATH = repo_root() / "tools" / "validate_dataset_quality.py"


def _load_validator_module():
    spec = importlib.util.spec_from_file_location(
        "dataset_quality_validation",
        VALIDATOR_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _restore_validator_context(monkeypatch):
    """Keep shared dataset-quality configuration isolated between tests."""
    monkeypatch.setattr(core, "DATA_DIR", core.DATA_DIR)
    monkeypatch.setattr(core, "DATASET_INFO", core.DATASET_INFO)
    monkeypatch.setattr(core, "REPORT_OUTPUT", core.REPORT_OUTPUT)
    monkeypatch.setattr(core, "DIAGNOSTIC_ALL_PHY", core.DIAGNOSTIC_ALL_PHY)
    monkeypatch.setattr(core.dataset_metadata, "DATA_DIR", core.dataset_metadata.DATA_DIR)
    monkeypatch.setattr(
        core.dataset_metadata,
        "DATASET_INFO_FILE",
        core.dataset_metadata.DATASET_INFO_FILE,
    )
    monkeypatch.setattr(
        core.performance_report,
        "DATA_DIR",
        core.performance_report.DATA_DIR,
    )


class _Capture:
    def __init__(self, **values) -> None:
        self.values = {key: np.asarray(value) for key, value in values.items()}
        self.files = list(self.values)

    def __getitem__(self, key):
        return self.values[key]


def _by_name(results):
    return {result.name: result for result in results}


def test_feature_blocks_preserve_missing_sampler_slots() -> None:
    slot_index = np.arange(0, 1001, 2, dtype=np.int64)
    values = np.where(slot_index < 500, 1.0, 3.0).reshape(-1, 1)
    timing = {
        "slot_index": slot_index,
        "reset_index": np.zeros(len(slot_index), dtype=np.int32),
        "target_pps": 100,
    }

    blocks = metrics._feature_block_medians(values, timing)

    np.testing.assert_array_equal(blocks, np.asarray([[1.0], [3.0]]))
    assert metrics._temporal_coverage_seconds(timing, len(values)) == pytest.approx(10.01)


def test_agnostic_baseline_uses_sampler_grid_for_elapsed_time() -> None:
    slot_index = np.arange(0, 1001, 2, dtype=np.int64)
    evidence = np.linspace(0.0, 1.0, len(slot_index))
    timing = {
        "slot_index": slot_index,
        "reset_index": np.zeros(len(slot_index), dtype=np.int32),
        "target_pps": 100,
    }

    baseline = metrics._agnostic_baseline_stats_from_series(evidence, timing)

    assert baseline is not None
    assert baseline["packet_rate_pps"] == 100.0
    assert baseline["eval_seconds"] == pytest.approx(10.01)
    assert len(baseline["block_margins"]) == 2


def test_temporal_occupancy_uses_complete_production_windows() -> None:
    packets = [
        {"wifi_rx_ts_us": index * 200_000}
        for index in range(11)
    ]

    occupancy = metrics._mean_temporal_occupancy(packets, target_pps=10)

    assert occupancy == pytest.approx(0.5)
    assert rendering._format_occupancy_cell(occupancy, markdown=True) == (
        "**50.0% ❌**"
    )
    assert rendering._format_occupancy_cell(0.7, markdown=True) == "**70.0% ⚠️**"
    assert rendering._format_occupancy_cell(0.85, markdown=True) == "85.0%"


def test_post_collect_temporal_occupancy_uses_recorded_detector_grid(monkeypatch) -> None:
    packets = ({"csi_target_pps": 100},)
    observed = {"occupancy": 0.69}
    monkeypatch.setattr(capture, "_load_validation_packet_view", lambda filepath: packets)
    monkeypatch.setattr(
        capture,
        "_mean_temporal_occupancy",
        lambda values, target_pps: observed["occupancy"],
    )

    result = capture.validate_temporal_occupancy(Path("capture.npz"))[0]

    assert result.name == "temporal_occupancy"
    assert result.status == "FAIL"
    assert result.value == 0.69
    assert "69.0%" in result.message

    observed["occupancy"] = 0.70
    assert capture.validate_temporal_occupancy(Path("capture.npz"))[0].status == "WARN"
    observed["occupancy"] = 0.85
    assert capture.validate_temporal_occupancy(Path("capture.npz"))[0].status == "PASS"


def test_capture_file_centralizes_canonical_admission_checks(monkeypatch) -> None:
    data = {"csi_data": np.zeros((4, 128), dtype=np.int8)}
    calls = []
    monkeypatch.setattr(
        capture,
        "validate_file_integrity",
        lambda filepath: ([core.ValidationResult("file_load", "PASS", "ok")], data),
    )
    monkeypatch.setattr(
        capture,
        "validate_signal_quality",
        lambda csi_data: [core.ValidationResult("signal", "PASS", "ok")],
    )

    def validate_occupancy(filepath, *, target_pps=None):
        calls.append(("occupancy", target_pps))
        return [core.ValidationResult("temporal_occupancy", "PASS", "ok")]

    def validate_continuity(data, csi_data, **kwargs):
        calls.append(("continuity", kwargs))
        return [core.ValidationResult("stream", "PASS", "ok")]

    monkeypatch.setattr(capture, "validate_temporal_occupancy", validate_occupancy)
    monkeypatch.setattr(capture, "validate_capture_continuity", validate_continuity)

    results = capture.validate_capture_file(
        Path("capture.npz"),
        low_rssi=True,
        include_packet_rate=False,
        target_pps=120,
    )

    assert [result.name for result in results] == [
        "file_load",
        "signal",
        "temporal_occupancy",
        "stream",
    ]
    assert calls == [
        ("occupancy", 120),
        ("continuity", {"low_rssi": True, "include_packet_rate": False}),
    ]


def test_occupancy_caps_quality_score() -> None:

    assert metrics.cap_quality_score_by_occupancy(100.0, 0.82) == 82.0
    assert metrics.cap_quality_score_by_occupancy(90.0, 0.95, 0.88) == 88.0
    assert metrics.cap_quality_score_by_occupancy(65.0, 0.95) == 65.0


def test_occupancy_target_prefers_recorded_grid_over_legacy_fallback() -> None:

    assert metrics._resolve_temporal_occupancy_target_pps(
        ({"csi_target_pps": 120},),
        fallback=100,
    ) == 120
    assert metrics._resolve_temporal_occupancy_target_pps(
        ({"csi_target_pps": None},),
        fallback=100,
    ) == 100


def test_pair_score_table_replaces_composite_columns_with_occupancy() -> None:
    row = {
        "chip": "C6",
        "environment": "bedroom",
        "static_presence": "static.npz",
        "motion": "motion.npz",
        "static_date": "2026-08-23 10:00",
        "motion_date": "2026-08-23 10:05",
        "static_rssi_dbm": -50.0,
        "motion_rssi_dbm": -52.0,
        "static_packet_rate_pps": 100.0,
        "motion_packet_rate_pps": 100.0,
        "static_mean_occupancy": 0.8,
        "motion_mean_occupancy": 0.6,
        "motion_coverage": 1.0,
        "pair_separation": 1.0,
        "pair_score": 100.0,
        "reference_cleanliness": {
            "basis": "environment",
            "reference_count": 3,
            "excursion_ratio": 0.0,
            "longest_burst_seconds": 0.0,
            "score": 100.0,
        },
        "feature_score": 60.0,
    }

    table = "\n".join(
        rendering._render_score_table(
            [row],
            rendering._PAIR_SCORE_TABLE,
            markdown=True,
        )
    )

    header = rendering._PAIR_SCORE_TABLE["header"]
    assert "| PPS | Occ | Ref |" in header
    assert "| Pair |" not in header
    assert "| Clean |" not in header
    assert "**80.0% ⚠️** / **60.0% ❌**" in table
    assert "| 60.0 |" in table
    assert "| PPS | Occ | Exc |" in rendering._PRESENCE_SCORE_TABLE["header"]
    assert "| PPS | Occ | Exc |" in rendering._EMPTY_SCORE_TABLE["header"]
    assert "| PPS | Occ | Exc |" in rendering._LONG_TEST_SCORE_TABLE["header"]


def test_classic_replay_consumes_selected_packet_rssi_and_flushes_final_slot() -> None:

    class RecordingDetector:
        def __init__(self):
            self.consumed = []

        def reset(self):
            pass

        def advance_missing_slots(self, _count):
            pass

        def process_packet(self, packet, _subcarriers, rssi_dbm=None, timestamp_us=None):
            self.consumed.append((int(packet[0]), int(rssi_dbm)))

        def update_state(self):
            return {"motion_metric": 0.0}

        def is_ready(self):
            return False

        def get_state(self):
            return core.MotionState.IDLE

        def get_threshold(self):
            return 0.5

    detector = RecordingDetector()
    csi_data = np.stack([
        np.full(128, value, dtype=np.int8) for value in (1, 2, 3)
    ])

    replay._replay_classic_metrics(
        csi_data,
        detector,
        rssi_dbm=np.asarray([-41, -42, -43]),
        wifi_rx_ts_us=np.asarray([0, 40_000, 100_000]),
        target_pps=10,
    )

    assert detector.consumed == [(1, -41), (3, -43)]


def test_configure_dataset_paths_updates_shared_roots(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "external_dataset"
    monkeypatch.setattr(
        core.dataset_metadata, "DATA_DIR", core.dataset_metadata.DATA_DIR
    )
    monkeypatch.setattr(
        core.dataset_metadata,
        "DATASET_INFO_FILE",
        core.dataset_metadata.DATASET_INFO_FILE,
    )
    monkeypatch.setattr(
        core.performance_report, "DATA_DIR", core.performance_report.DATA_DIR
    )

    core.configure_dataset_paths(dataset_root)

    assert core.DATA_DIR == dataset_root
    assert core.DATASET_INFO == dataset_root / "dataset_info.json"
    assert core.REPORT_OUTPUT == (
        dataset_root / "auto_generated" / "DATASET_QUALITY_CHECK.md"
    )
    assert core.dataset_metadata.DATA_DIR == dataset_root
    assert core.performance_report.DATA_DIR == dataset_root
    assert rendering._dataset_file_href("motion", "sample.npz") == (
        "../motion/sample.npz"
    )
    assert rendering._report_source_path() == str(dataset_root / "dataset_info.json")


def test_configure_dataset_paths_accepts_custom_report_output(
    tmp_path, monkeypatch
) -> None:
    dataset_root = tmp_path / "external_dataset"
    report_output = tmp_path / "reports" / "quality.md"
    monkeypatch.setattr(
        core.dataset_metadata, "DATA_DIR", core.dataset_metadata.DATA_DIR
    )
    monkeypatch.setattr(
        core.dataset_metadata,
        "DATASET_INFO_FILE",
        core.dataset_metadata.DATASET_INFO_FILE,
    )
    monkeypatch.setattr(
        core.performance_report, "DATA_DIR", core.performance_report.DATA_DIR
    )

    core.configure_dataset_paths(dataset_root, report_output)

    assert core.REPORT_OUTPUT == report_output
    assert rendering._dataset_file_href("empty", "quiet.npz") == (
        "../external_dataset/empty/quiet.npz"
    )


def test_dataset_file_href_uses_catalog_relative_path(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "external_dataset"
    report_output = dataset_root / "auto_generated" / "quality.md"
    dataset_root.mkdir()
    (dataset_root / "dataset_info.json").write_text(
        '{"files":{"motion":[{"filename":"logical.npz",'
        '"relative_path":"jump/canonical.npz"}]}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        core.dataset_metadata, "DATA_DIR", core.dataset_metadata.DATA_DIR
    )
    monkeypatch.setattr(
        core.dataset_metadata,
        "DATASET_INFO_FILE",
        core.dataset_metadata.DATASET_INFO_FILE,
    )
    monkeypatch.setattr(
        core.performance_report, "DATA_DIR", core.performance_report.DATA_DIR
    )

    core.configure_dataset_paths(dataset_root, report_output)

    assert rendering._dataset_file_href("motion", "logical.npz") == (
        "../jump/canonical.npz"
    )


def test_relative_path_catalog_does_not_scan_source_directory_for_orphans(
    tmp_path, monkeypatch
) -> None:
    dataset_root = tmp_path / "external_dataset"
    source_dir = dataset_root / "empty"
    source_dir.mkdir(parents=True)
    (source_dir / "canonical.npz").touch()
    (source_dir / "uncatalogued_other_view.npz").touch()
    entry = {
        "filename": "logical.npz",
        "relative_path": "empty/canonical.npz",
        "chip": "ESP32",
        "subcarriers": 64,
        "num_packets": 1,
        "collected_at": "2026-08-14T00:00:00+00:00",
        "environment": "external",
        "dataset_role": "holdout",
    }
    monkeypatch.setattr(core, "DATA_DIR", dataset_root)
    monkeypatch.setattr(core.dataset_metadata, "DATA_DIR", dataset_root)

    results = catalog.validate_metadata_completeness(
        {"files": {"empty": [entry], "static_presence": [], "motion": []}}
    )

    assert not any(result.name.startswith("metadata_orphan/") for result in results)


def test_logical_pair_aliases_are_deduplicated_for_per_file_evidence(
    tmp_path, monkeypatch
) -> None:
    dataset_root = tmp_path / "external_dataset"
    monkeypatch.setattr(core.dataset_metadata, "DATA_DIR", dataset_root)
    entries = [
        {"filename": "idle_for_jump.npz", "relative_path": "idle/canonical.npz"},
        {"filename": "idle_for_walk.npz", "relative_path": "idle/canonical.npz"},
    ]

    unique = references._unique_entries_by_resolved_path("static_presence", entries)

    assert unique == entries[:1]


def test_pair_refresh_is_reciprocal_and_keeps_unselected_chips() -> None:
    files = {
        "static_presence": [
            {
                "filename": "static_c6.npz",
                "chip": "C6",
                "subcarriers": 64,
                "collected_at": "2026-08-28T12:00:00+00:00",
                "device_id": "c6-a",
                "environment": "lab",
                "dataset_role": "train",
                "optimal_pair_motion_file": "stale.npz",
            },
            {
                "filename": "static_s3.npz",
                "chip": "S3",
                "subcarriers": 64,
                "collected_at": "2026-08-28T12:00:00+00:00",
                "dataset_role": "train",
                "optimal_pair_motion_file": "motion_s3.npz",
            },
        ],
        "motion": [
            {
                "filename": "wrong_environment.npz",
                "chip": "C6",
                "subcarriers": 64,
                "collected_at": "2026-08-28T12:01:00+00:00",
                "device_id": "c6-a",
                "environment": "hall",
                "dataset_role": "train",
            },
            {
                "filename": "motion_c6.npz",
                "chip": "C6",
                "subcarriers": 64,
                "collected_at": "2026-08-28T12:02:00+00:00",
                "device_id": "c6-a",
                "environment": "lab",
                "dataset_role": "train",
            },
            {
                "filename": "motion_s3.npz",
                "chip": "S3",
                "subcarriers": 64,
                "collected_at": "2026-08-28T12:02:00+00:00",
                "dataset_role": "train",
                "optimal_pair_static_presence_file": "static_s3.npz",
            },
        ],
    }

    rows = pairing.refresh_pair_metadata(files, selected_chips={"C6"})

    assert rows == [
        {
            "static_presence": "static_c6.npz",
            "motion": "motion_c6.npz",
            "delta_seconds": 120.0,
        }
    ]
    assert files["static_presence"][0]["optimal_pair_motion_file"] == "motion_c6.npz"
    assert (
        files["motion"][1]["optimal_pair_static_presence_file"]
        == "static_c6.npz"
    )
    assert files["static_presence"][1]["optimal_pair_motion_file"] == "motion_s3.npz"
    assert (
        files["motion"][2]["optimal_pair_static_presence_file"]
        == "static_s3.npz"
    )


def test_report_current_check_includes_evaluation_view(tmp_path, monkeypatch) -> None:
    report = tmp_path / "quality.md"
    monkeypatch.setattr(core, "DATA_DIR", core.DATA_DIR)
    monkeypatch.setattr(core, "DATASET_INFO", core.DATASET_INFO)
    monkeypatch.setattr(core, "REPORT_OUTPUT", core.REPORT_OUTPUT)
    monkeypatch.setattr(core, "DIAGNOSTIC_ALL_PHY", core.DIAGNOSTIC_ALL_PHY)
    monkeypatch.setattr(core.dataset_metadata, "DATA_DIR", core.dataset_metadata.DATA_DIR)
    monkeypatch.setattr(
        core.dataset_metadata,
        "DATASET_INFO_FILE",
        core.dataset_metadata.DATASET_INFO_FILE,
    )
    monkeypatch.setattr(core.performance_report, "DATA_DIR", core.performance_report.DATA_DIR)
    core.configure_dataset_paths(tmp_path / "dataset", report)
    report.write_text("Evaluation view: `HT20/HT-LTF`\n", encoding="utf-8")

    assert rendering._report_evaluation_view_is_current()

    core.configure_validation_mode(diagnostic_all_phy=True)
    assert not rendering._report_evaluation_view_is_current()


def test_main_forwards_external_dataset_options(tmp_path, monkeypatch) -> None:
    module = _load_validator_module()
    dataset_root = tmp_path / "external_dataset"
    captured = {}
    monkeypatch.setattr(
        core.dataset_metadata, "DATA_DIR", core.dataset_metadata.DATA_DIR
    )
    monkeypatch.setattr(
        core.dataset_metadata,
        "DATASET_INFO_FILE",
        core.dataset_metadata.DATASET_INFO_FILE,
    )
    monkeypatch.setattr(
        core.performance_report, "DATA_DIR", core.performance_report.DATA_DIR
    )
    monkeypatch.setattr(
        module,
        "run_validation",
        lambda **kwargs: captured.update(kwargs) or 0,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_dataset_quality.py",
            "--data-dir",
            str(dataset_root),
            "--chip",
            "ESP32",
            "--preserve-pairs",
            "--diagnostic-all-phy",
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        module.main()

    assert exit_info.value.code == 0
    assert core.DATA_DIR == dataset_root
    assert captured == {
        "chip_filter": "ESP32",
        "generate_report": True,
        "use_cache": True,
        "refresh_pair_metadata": False,
        "diagnostic_all_phy": True,
    }


@pytest.mark.parametrize(("current", "expected"), [(True, 0), (False, 1)])
def test_main_check_current_returns_status(monkeypatch, tmp_path, current, expected) -> None:
    module = _load_validator_module()
    monkeypatch.setattr(module.core, "configure_dataset_paths", lambda *_args: None)
    monkeypatch.setattr(module.core, "configure_validation_mode", lambda **_kwargs: None)
    monkeypatch.setattr(module.core, "REPORT_OUTPUT", tmp_path / "report.md")
    monkeypatch.setattr(module.core, "DATASET_INFO", tmp_path / "dataset_info.json")
    monkeypatch.setattr(module.core, "_report_input_paths", lambda: [])
    monkeypatch.setattr(
        module.dataset_metadata,
        "generated_report_is_current",
        lambda *_args, **_kwargs: current,
    )
    monkeypatch.setattr(module, "_report_evaluation_view_is_current", lambda: True)
    monkeypatch.setattr(
        sys,
        "argv",
        ["validate_dataset_quality.py", "--check-current", "--data-dir", str(tmp_path)],
    )

    with pytest.raises(SystemExit) as exit_info:
        module.main()

    assert exit_info.value.code == expected


def test_main_forwards_no_report_and_no_cache(monkeypatch, tmp_path) -> None:
    module = _load_validator_module()
    captured = {}
    monkeypatch.setattr(module.core, "configure_dataset_paths", lambda *_args: None)
    monkeypatch.setattr(module.core, "configure_validation_mode", lambda **_kwargs: None)
    monkeypatch.setattr(
        module,
        "run_validation",
        lambda **kwargs: captured.update(kwargs) or 4,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_dataset_quality.py",
            "--data-dir",
            str(tmp_path),
            "--no-report",
            "--no-cache",
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        module.main()

    assert exit_info.value.code == 4
    assert captured["generate_report"] is False
    assert captured["use_cache"] is False


def test_validate_file_integrity_rejects_object_arrays(tmp_path) -> None:
    filepath = tmp_path / "malicious_dataset.npz"
    np.savez_compressed(
        filepath,
        csi_data=np.zeros((1, 128), dtype=np.int8),
        num_subcarriers=64,
        chip=np.array("c6", dtype=object),
        label="motion",
    )

    results, data = capture.validate_file_integrity(filepath)

    assert data is None
    assert results[0].name == "file_load"
    assert results[0].status == "FAIL"
    assert "Unsafe NPZ dataset" in results[0].message


def test_file_integrity_rejects_subcarrier_shape_mismatch(tmp_path) -> None:
    path = tmp_path / "bad.npz"
    np.savez(
        path,
        csi_data=np.zeros((10, 128), dtype=np.int8),
        num_subcarriers=np.array(52),
    )

    results, data = capture.validate_file_integrity(path)
    shape = _by_name(results)["csi_shape"]

    assert data is not None
    assert shape.status == "FAIL"
    assert "implies 64 subcarriers" in shape.message


def test_file_integrity_returns_ht20_sensing_view(tmp_path) -> None:
    path = tmp_path / "mixed_phy.npz"
    np.savez(
        path,
        csi_data=np.zeros((5, 128), dtype=np.int8),
        num_subcarriers=np.array(64),
        label=np.array("motion"),
        phy_mode=np.array(["ht", "legacy", "ht", "legacy", "ht"]),
        ltf_type=np.array(["ht-ltf", "lltf", "ht-ltf", "lltf", "ht-ltf"]),
        channel_width=np.array(["20"] * 5),
        stream_seq_num=np.array([1, 2, 3, 4, 5], dtype=np.uint32),
    )

    results, data = capture.validate_file_integrity(path)

    assert _by_name(results)["file_load"].status == "PASS"
    assert data is not None
    assert data["csi_data"].shape[0] == 3
    np.testing.assert_array_equal(
        data["stream_seq_num"],
        np.array([1, 3, 5], dtype=np.uint32),
    )


def test_file_integrity_diagnostic_mode_keeps_nonstandard_phy(tmp_path) -> None:
    path = tmp_path / "lltf_ht40.npz"
    np.savez(
        path,
        csi_data=np.zeros((5, 128), dtype=np.int8),
        num_subcarriers=np.array(64),
        label=np.array("motion"),
        phy_mode=np.array(["ht"] * 5),
        ltf_type=np.array(["lltf"] * 5),
        channel_width=np.array(["40"] * 5),
    )
    core.configure_validation_mode(diagnostic_all_phy=True)

    results, data = capture.validate_file_integrity(path)

    assert _by_name(results)["sensing_contract"].status == "FAIL"
    assert data is not None
    assert data["csi_data"].shape[0] == 5
    assert core._report_evaluation_view() == "all explicit PHY rows (diagnostic)"


def test_signal_quality_checks_packet_count_zero_packets_and_amplitude() -> None:
    strong_packet = np.tile(np.array([30, 40], dtype=np.int8), 64)
    healthy = np.tile(strong_packet, (severity.MIN_PACKETS, 1))

    healthy_results = _by_name(capture.validate_signal_quality(healthy))
    empty_results = _by_name(
        capture.validate_signal_quality(np.zeros((1, 128), dtype=np.int8))
    )

    assert healthy_results["packet_count"].status == "PASS"
    assert healthy_results["zero_packets"].status == "PASS"
    assert healthy_results["signal_level"].status == "PASS"
    assert empty_results["packet_count"].status == "FAIL"
    assert empty_results["zero_packets"].status == "WARN"
    assert empty_results["signal_level"].status == "WARN"


def test_capture_continuity_sees_gaps_after_ht20_filter(tmp_path) -> None:
    path = tmp_path / "legacy_heavy.npz"
    np.savez(
        path,
        csi_data=np.zeros((40, 128), dtype=np.int8),
        num_subcarriers=np.array(64),
        label=np.array("static_presence"),
        duration_ms=np.array(1000.0),
        phy_mode=np.array(["ht", "legacy"] * 20),
        ltf_type=np.array(["ht-ltf", "lltf"] * 20),
        channel_width=np.array(["20"] * 40),
        stream_seq_num=np.arange(1, 41, dtype=np.uint32),
    )

    _, data = capture.validate_file_integrity(path)
    assert data is not None

    continuity = _by_name(
        capture.validate_capture_continuity(data, data["csi_data"])
    )
    assert continuity["stream_seq_gaps"].status in {"WARN", "FAIL"}
    assert continuity["stream_seq_gaps"].value > 0.0


def test_capture_continuity_flags_low_rate_and_stream_gaps() -> None:
    data = _Capture(
        duration_ms=1000.0,
        stream_seq_num=np.array([10, 11, 12, 60], dtype=np.uint32),
    )

    results = _by_name(
        capture.validate_capture_continuity(
            data,
            np.zeros((4, 128), dtype=np.int8),
        )
    )

    assert results["packet_rate"].status == "WARN"
    assert results["stream_seq_gaps"].status == "FAIL"
    assert results["stream_seq_max_gap"].status == "FAIL"
    assert "after packet 2 (seq 12 -> 60)" in results["stream_seq_max_gap"].message


def test_capture_continuity_uses_low_rssi_loss_ceiling() -> None:
    bounded = _Capture(
        duration_ms=960.0,
        stream_seq_num=np.delete(
            np.arange(100, dtype=np.uint32),
            [20, 40, 60, 80],
        ),
    )
    excessive = _Capture(
        duration_ms=940.0,
        stream_seq_num=np.delete(
            np.arange(100, dtype=np.uint32),
            [10, 20, 30, 40, 50, 60],
        ),
    )

    normal = _by_name(
        capture.validate_capture_continuity(
            bounded,
            np.zeros((96, 128), dtype=np.int8),
        )
    )
    low_rssi = _by_name(
        capture.validate_capture_continuity(
            bounded,
            np.zeros((96, 128), dtype=np.int8),
            low_rssi=True,
        )
    )
    rejected = _by_name(
        capture.validate_capture_continuity(
            excessive,
            np.zeros((94, 128), dtype=np.int8),
            low_rssi=True,
        )
    )

    assert normal["stream_seq_gaps"].status == "FAIL"
    assert low_rssi["stream_seq_gaps"].status == "WARN"
    assert low_rssi["stream_seq_gaps"].value == 0.04
    assert rejected["stream_seq_gaps"].status == "FAIL"
    assert rejected["stream_seq_gaps"].value == 0.06


def test_capture_continuity_accepts_supported_boundaries() -> None:
    data = _Capture(
        duration_ms=1000.0,
        stream_seq_num=np.arange(95, dtype=np.uint32),
        device_ticks_us=np.concatenate(
            (
                np.array([0, 150_000], dtype=np.uint64),
                np.arange(2, 95, dtype=np.uint64) * 10_000 + 150_000,
            )
        ),
    )

    results = _by_name(
        capture.validate_capture_continuity(
            data,
            np.zeros((95, 128), dtype=np.int8),
        )
    )

    assert results["packet_rate"].status == "PASS"
    assert results["packet_rate"].value == 95.0
    assert results["inter_packet_gap"].status == "PASS"
    assert results["inter_packet_gap"].value == 150.0


def test_capture_continuity_rejects_large_inter_packet_gap() -> None:
    data = _Capture(
        duration_ms=1000.0,
        stream_seq_num=np.array([1, 2, 3, 4], dtype=np.uint32),
        device_ticks_us=np.array([0, 10_000, 20_000, 2_500_000], dtype=np.uint64),
    )

    results = _by_name(
        capture.validate_capture_continuity(
            data,
            np.zeros((4, 128), dtype=np.int8),
        )
    )

    assert results["inter_packet_gap"].status == "FAIL"
    assert results["inter_packet_gap"].value == 2480.0
    assert "at packet 2->3" in results["inter_packet_gap"].message


def test_excluded_idle_unusable_rows_are_marked_and_listed() -> None:
    rows = [
        {
            "label": "empty",
            "filename": "empty_c6.npz",
            "display_date": "2026-07-04 15:16",
            "chip": "C6",
            "environment": "hobby_room",
            "rssi_dbm": -49.0,
            "packet_rate_pps": 98.7,
            "reference_cleanliness": {
                "basis": "chip",
                "reference_count": 3,
                "excursion_ratio": 0.0,
                "longest_burst_seconds": 0.0,
                "score": 100.0,
            },
            "unusable": False,
        },
        {
            "label": "empty",
            "filename": "empty_c3_zero.npz",
            "display_date": "2026-07-04 16:15",
            "chip": "C3",
            "environment": "hobby_room",
            "rssi_dbm": -48.0,
            "packet_rate_pps": 101.2,
            "reference_cleanliness": None,
            "unusable": True,
        },
    ]

    results = pairing._excluded_idle_unusable_results(rows)
    assert len(results) == 1
    assert results[0].status == "WARN"
    assert results[0].name == "excluded_idle_unusable/empty_c3_zero.npz"

    rendered = rendering._format_excluded_idle_row(rows[1], markdown=True)
    assert rendered.count("**n/a ⚠️**") == 4

    section = "\n".join(rendering._render_unusable_excluded_idle_section(rows))
    assert "## Unscorable excluded idle" in section
    assert "empty_c3_zero.npz" in section
    assert "empty_c6.npz" not in section

    table = "\n".join(
        rendering._render_score_table(
            rows,
            rendering._EXCLUDED_IDLE_SCORE_TABLE,
            markdown=True,
        )
    )
    assert table.index("2026-07-04 16:15") < table.index("2026-07-04 15:16")
