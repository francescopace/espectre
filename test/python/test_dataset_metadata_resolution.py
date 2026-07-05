import json
from pathlib import Path

from tools.lib import dataset_metadata, ui


def _write_dataset_info(tmp_path: Path, payload: dict) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "static_presence").mkdir(exist_ok=True)
    (data_dir / "motion").mkdir(exist_ok=True)
    (data_dir / "test").mkdir(exist_ok=True)
    (data_dir / "dataset_info.json").write_text(json.dumps(payload), encoding="utf-8")


def test_resolve_explicit_pair_uses_metadata_pair_and_threshold(monkeypatch, tmp_path) -> None:
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
                    "optimal_threshold_gridsearch": 1.75,
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
    assert pair.threshold == 1.75
    assert pair.threshold_source == "metadata"


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
                    "optimal_threshold_gridsearch": 0.9,
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


def test_resolve_dataset_threshold_falls_back_to_calibration(monkeypatch) -> None:
    monkeypatch.setattr(dataset_metadata, "estimate_runtime_mvs_threshold", lambda packets, threshold_mode=None, selected_subcarriers=None: 1.23)

    threshold, source = dataset_metadata.resolve_dataset_threshold(
        {"filename": "example.npz"},
        packets=[{"csi_data": [1, 2, 3, 4]}],
    )

    assert threshold == 1.23
    assert source == "fallback_calibration"


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
                    "optimal_threshold_gridsearch": 0.9,
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
