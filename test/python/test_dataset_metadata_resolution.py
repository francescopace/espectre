# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - Dataset Metadata Resolution Tests

Tests for dataset metadata resolution helpers.

Author: Francesco Pace <francesco.pace@gmail.com>
"""

import json
from pathlib import Path

from tools.lib import dataset_metadata, ui


def test_dataset_roles_are_normalized_and_admitted_centrally() -> None:
    assert dataset_metadata.dataset_role({}) == "exclude"
    assert dataset_metadata.dataset_role({"dataset_role": ""}) == "exclude"
    assert dataset_metadata.dataset_role({"dataset_role": " Train "}) == "train"
    assert dataset_metadata.admitted_dataset_role({}) is None
    assert (
        dataset_metadata.admitted_dataset_role({"dataset_role": "selection"})
        == "selection"
    )
    assert (
        dataset_metadata.admitted_dataset_role({"dataset_role": "invalid"})
        is None
    )

    assert (
        dataset_metadata.paired_dataset_role(
            {"dataset_role": "holdout"},
            {"dataset_role": " HOLDOUT "},
        )
        == "holdout"
    )
    assert (
        dataset_metadata.paired_dataset_role(
            {"dataset_role": "train"},
            {"dataset_role": "selection"},
        )
        is None
    )
    assert (
        dataset_metadata.paired_dataset_role(
            {"dataset_role": "train"},
            {},
        )
        is None
    )


def test_generated_report_revision_tracks_exact_dataset_catalog(tmp_path) -> None:
    dataset_info_path = tmp_path / "dataset_info.json"
    report_path = tmp_path / "REPORT.md"
    dataset_info_path.write_text('{"files": {}}\n', encoding="utf-8")

    revision = dataset_metadata.dataset_info_revision(dataset_info_path)
    report_path.write_text(
        f"Dataset revision: `sha256:{revision}`\n",
        encoding="utf-8",
    )

    assert dataset_metadata.generated_report_is_current(
        report_path,
        dataset_info_path,
    )

    dataset_info_path.write_text('{"files": {\"empty\": []}}\n', encoding="utf-8")
    assert not dataset_metadata.generated_report_is_current(
        report_path,
        dataset_info_path,
    )


def test_generated_report_revision_tracks_implementation_inputs(tmp_path) -> None:
    dataset_info_path = tmp_path / "dataset_info.json"
    dependency_path = tmp_path / "detector.py"
    report_path = tmp_path / "REPORT.md"
    dataset_info_path.write_text('{"files": {}}\n', encoding="utf-8")
    dependency_path.write_text("VERSION = 1\n", encoding="utf-8")
    report_path.write_text(
        "\n".join(
            (
                "Dataset revision: "
                f"`sha256:{dataset_metadata.dataset_info_revision(dataset_info_path)}`",
                "Input revision: "
                f"`sha256:{dataset_metadata.generated_input_revision([dependency_path])}`",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    assert dataset_metadata.generated_report_is_current(
        report_path,
        dataset_info_path,
        input_paths=[dependency_path],
    )
    dependency_path.write_text("VERSION = 2\n", encoding="utf-8")
    assert not dataset_metadata.generated_report_is_current(
        report_path,
        dataset_info_path,
        input_paths=[dependency_path],
    )


def _write_dataset_info(tmp_path: Path, payload: dict) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "empty").mkdir(exist_ok=True)
    (data_dir / "static_presence").mkdir(exist_ok=True)
    (data_dir / "motion").mkdir(exist_ok=True)
    (data_dir / "dataset_info.json").write_text(json.dumps(payload), encoding="utf-8")


def test_resolve_explicit_pair_uses_metadata_pair(monkeypatch, tmp_path) -> None:
    info = {
        "files": {
            "static_presence": [
                {
                    "filename": "static_presence_c6_64sc_lab_001.npz",
                    "chip": "C6",
                    "environment": "lab",
                    "subcarriers": 64,
                    "collected_at": "2026-07-05T10:00:00",
                    "optimal_pair_motion_file": "motion_c6_64sc_lab_001.npz",
                }
            ],
            "motion": [
                {
                    "filename": "motion_c6_64sc_lab_001.npz",
                    "chip": "C6",
                    "environment": "lab",
                    "subcarriers": 64,
                    "collected_at": "2026-07-05T10:01:00",
                    "optimal_pair_static_presence_file": "static_presence_c6_64sc_lab_001.npz",
                }
            ],
        }
    }
    _write_dataset_info(tmp_path, info)
    data_dir = tmp_path / "data"
    static_path = data_dir / "static_presence" / "static_presence_c6_64sc_lab_001.npz"
    motion_path = data_dir / "motion" / "motion_c6_64sc_lab_001.npz"
    static_path.write_bytes(b"")
    motion_path.write_bytes(b"")

    monkeypatch.setattr(dataset_metadata, "DATA_DIR", data_dir)
    monkeypatch.setattr(dataset_metadata, "DATASET_INFO_FILE", data_dir / "dataset_info.json")

    pair = dataset_metadata.resolve_explicit_pair(dataset="motion_c6_64sc_lab_001", num_sc=64)

    assert pair.static_presence.path == static_path
    assert pair.motion.path == motion_path
    assert pair.chip == "C6"
    assert pair.num_subcarriers == 64


def test_resolve_dataset_selection_returns_counterpart_for_motion(monkeypatch, tmp_path) -> None:
    info = {
        "files": {
            "static_presence": [
                {
                    "filename": "static_presence_s3_64sc_room_001.npz",
                    "chip": "S3",
                    "environment": "room",
                    "subcarriers": 64,
                    "collected_at": "2026-07-05T12:00:00",
                    "optimal_pair_motion_file": "motion_s3_64sc_room_001.npz",
                }
            ],
            "motion": [
                {
                    "filename": "motion_s3_64sc_room_001.npz",
                    "chip": "S3",
                    "environment": "room",
                    "subcarriers": 64,
                    "collected_at": "2026-07-05T12:01:00",
                    "optimal_pair_static_presence_file": "static_presence_s3_64sc_room_001.npz",
                }
            ],
        }
    }
    _write_dataset_info(tmp_path, info)
    data_dir = tmp_path / "data"
    (data_dir / "static_presence" / "static_presence_s3_64sc_room_001.npz").write_bytes(b"")
    (data_dir / "motion" / "motion_s3_64sc_room_001.npz").write_bytes(b"")

    monkeypatch.setattr(dataset_metadata, "DATA_DIR", data_dir)
    monkeypatch.setattr(dataset_metadata, "DATASET_INFO_FILE", data_dir / "dataset_info.json")

    selected = dataset_metadata.resolve_dataset_selection("motion_s3_64sc_room_001", num_sc=64)

    assert selected.label == "motion"
    assert selected.counterpart_label == "static_presence"
    assert selected.counterpart_entry["filename"] == "static_presence_s3_64sc_room_001.npz"

def test_resolve_dataset_selection_require_pair_excludes_non_pair_labels(monkeypatch, tmp_path) -> None:
    info = {
        "files": {
            "empty": [
                {
                    "filename": "empty_s3_64sc_room_001.npz",
                    "chip": "S3",
                    "environment": "room",
                    "subcarriers": 64,
                    "collected_at": "2026-07-05T12:02:00",
                }
            ],
            "static_presence": [
                {
                    "filename": "static_presence_s3_64sc_room_001.npz",
                    "chip": "S3",
                    "environment": "room",
                    "subcarriers": 64,
                    "collected_at": "2026-07-05T12:00:00",
                    "optimal_pair_motion_file": "motion_s3_64sc_room_001.npz",
                }
            ],
            "motion": [
                {
                    "filename": "motion_s3_64sc_room_001.npz",
                    "chip": "S3",
                    "environment": "room",
                    "subcarriers": 64,
                    "collected_at": "2026-07-05T12:01:00",
                    "optimal_pair_static_presence_file": "static_presence_s3_64sc_room_001.npz",
                }
            ],
        }
    }
    _write_dataset_info(tmp_path, info)
    data_dir = tmp_path / "data"
    (data_dir / "empty").mkdir(exist_ok=True)
    (data_dir / "empty" / "empty_s3_64sc_room_001.npz").write_bytes(b"")
    (data_dir / "static_presence" / "static_presence_s3_64sc_room_001.npz").write_bytes(b"")
    (data_dir / "motion" / "motion_s3_64sc_room_001.npz").write_bytes(b"")

    monkeypatch.setattr(dataset_metadata, "DATA_DIR", data_dir)
    monkeypatch.setattr(dataset_metadata, "DATASET_INFO_FILE", data_dir / "dataset_info.json")

    selected = dataset_metadata.resolve_dataset_selection("static_presence_s3_64sc_room_001", num_sc=64, require_pair=True)

    assert selected.label == "static_presence"

    try:
        dataset_metadata.resolve_dataset_selection("empty_s3_64sc_room_001", num_sc=64, require_pair=True)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("require_pair=True should reject non-pairable labels")


def test_select_dataset_interactively_groups_by_environment_then_chip(monkeypatch, tmp_path, capsys) -> None:
    info = {
        "files": {
            "static_presence": [
                {
                    "filename": "static_presence_s3_64sc_bedroom_001.npz",
                    "chip": "S3",
                    "environment": "bedroom",
                    "subcarriers": 64,
                    "collected_at": "2026-07-05T12:00:00",
                    "optimal_pair_motion_file": "motion_s3_64sc_bedroom_001.npz",
                },
                {
                    "filename": "static_presence_c6_64sc_bedroom_001.npz",
                    "chip": "C6",
                    "environment": "bedroom",
                    "subcarriers": 64,
                    "collected_at": "2026-07-05T11:00:00",
                    "optimal_pair_motion_file": "motion_c6_64sc_bedroom_001.npz",
                },
                {
                    "filename": "static_presence_s3_64sc_living_room_001.npz",
                    "chip": "S3",
                    "environment": "living_room",
                    "subcarriers": 64,
                    "collected_at": "2026-07-05T10:00:00",
                    "optimal_pair_motion_file": "motion_s3_64sc_living_room_001.npz",
                },
            ],
            "motion": [
                {
                    "filename": "motion_s3_64sc_bedroom_001.npz",
                    "chip": "S3",
                    "environment": "bedroom",
                    "subcarriers": 64,
                    "collected_at": "2026-07-05T12:01:00",
                    "optimal_pair_static_presence_file": "static_presence_s3_64sc_bedroom_001.npz",
                },
                {
                    "filename": "motion_c6_64sc_bedroom_001.npz",
                    "chip": "C6",
                    "environment": "bedroom",
                    "subcarriers": 64,
                    "collected_at": "2026-07-05T11:01:00",
                    "optimal_pair_static_presence_file": "static_presence_c6_64sc_bedroom_001.npz",
                },
                {
                    "filename": "motion_s3_64sc_living_room_001.npz",
                    "chip": "S3",
                    "environment": "living_room",
                    "subcarriers": 64,
                    "collected_at": "2026-07-05T10:01:00",
                    "optimal_pair_static_presence_file": "static_presence_s3_64sc_living_room_001.npz",
                },
            ],
        }
    }
    _write_dataset_info(tmp_path, info)
    data_dir = tmp_path / "data"
    for relative_path in [
        "static_presence/static_presence_s3_64sc_bedroom_001.npz",
        "motion/motion_s3_64sc_bedroom_001.npz",
        "static_presence/static_presence_c6_64sc_bedroom_001.npz",
        "motion/motion_c6_64sc_bedroom_001.npz",
        "static_presence/static_presence_s3_64sc_living_room_001.npz",
        "motion/motion_s3_64sc_living_room_001.npz",
    ]:
        (data_dir / relative_path).write_bytes(b"")

    monkeypatch.setattr(dataset_metadata, "DATA_DIR", data_dir)
    monkeypatch.setattr(dataset_metadata, "DATASET_INFO_FILE", data_dir / "dataset_info.json")
    monkeypatch.setattr("builtins.input", lambda _prompt: "1")

    selected = dataset_metadata.select_dataset_interactively(require_pair=True, num_sc=64, prompt="Select grouped dataset")
    captured = capsys.readouterr().out

    assert "Environment: living_room" in captured
    assert "Environment: bedroom" in captured
    assert "  Chip: S3" in captured
    assert "  Chip: C6" in captured
    assert selected.label in {"static_presence", "motion"}


def test_select_dataset_interactively_handles_ctrl_c(monkeypatch, tmp_path, capsys) -> None:
    info = {
        "files": {
            "static_presence": [
                {
                    "filename": "static_presence_s3_64sc_bedroom_001.npz",
                    "chip": "S3",
                    "environment": "bedroom",
                    "subcarriers": 64,
                    "collected_at": "2026-07-05T12:00:00",
                    "optimal_pair_motion_file": "motion_s3_64sc_bedroom_001.npz",
                }
            ],
            "motion": [
                {
                    "filename": "motion_s3_64sc_bedroom_001.npz",
                    "chip": "S3",
                    "environment": "bedroom",
                    "subcarriers": 64,
                    "collected_at": "2026-07-05T12:01:00",
                    "optimal_pair_static_presence_file": "static_presence_s3_64sc_bedroom_001.npz",
                }
            ],
        }
    }
    _write_dataset_info(tmp_path, info)
    data_dir = tmp_path / "data"
    (data_dir / "static_presence" / "static_presence_s3_64sc_bedroom_001.npz").write_bytes(b"")
    (data_dir / "motion" / "motion_s3_64sc_bedroom_001.npz").write_bytes(b"")

    monkeypatch.setattr(dataset_metadata, "DATA_DIR", data_dir)
    monkeypatch.setattr(dataset_metadata, "DATASET_INFO_FILE", data_dir / "dataset_info.json")

    def _raise_interrupt(_prompt):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", _raise_interrupt)

    try:
        dataset_metadata.select_dataset_interactively(require_pair=True, num_sc=64, prompt="Select grouped dataset")
    except SystemExit as exc:
        assert exc.code == 130
    else:
        raise AssertionError("Ctrl-C should exit the selector cleanly")

    captured = capsys.readouterr().out
    assert "Selection cancelled." in captured


def test_show_plot_window_handles_ctrl_c(capsys) -> None:
    class FakePlotModule:
        def __init__(self) -> None:
            self.closed = False

        def show(self) -> None:
            raise KeyboardInterrupt

        def close(self, target) -> None:
            assert target == "all"
            self.closed = True

    fake = FakePlotModule()

    shown = ui.show_plot_window(fake, cancel_message="Plot cancelled.")

    assert shown is False
    assert fake.closed is True
    captured = capsys.readouterr().out
    assert "Plot cancelled." in captured
