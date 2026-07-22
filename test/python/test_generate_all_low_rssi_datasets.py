"""
Tests for tools/generate_all_low_rssi_datasets.py.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import importlib.util
from pathlib import Path
import sys


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "generate_all_low_rssi_datasets.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "generate_all_low_rssi_datasets", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _entry(
    filename,
    chip,
    *,
    environment="lab",
    synthetic=False,
    low_rssi=False,
    pair=None,
):
    entry = {
        "filename": filename,
        "chip": chip,
        "environment": environment,
        "collected_at": "2026-07-22T10:00:00",
        "synthetic": synthetic,
        "low_rssi": low_rssi,
    }
    if pair is not None:
        entry["optimal_pair_motion_file"] = pair
    return entry


def test_collect_jobs_uses_supported_real_chips_and_static_before_motion(monkeypatch, tmp_path):
    module = _load_module()
    monkeypatch.setattr(module.dataset_metadata, "DATA_DIR", tmp_path)
    info = {
        "files": {
            "empty": [
                _entry("empty_c3.npz", "C3"),
                _entry("empty_c6.npz", "C6"),
                _entry("empty_s3.npz", "S3"),
            ],
            "static_presence": [
                _entry("static_c5.npz", "C5"),
                _entry("static_synthetic.npz", "C5", synthetic=True),
            ],
            "motion": [_entry("motion_c5.npz", "C5")],
        }
    }

    jobs = module.collect_jobs(
        info,
        chips=("C3", "C5", "C6", "S3"),
        labels=module.LABEL_ORDER,
    )

    assert [job.source_path.name for job in jobs] == [
        "static_c5.npz",
        "motion_c5.npz",
    ]
    assert [job.profile_name for job in jobs] == [
        "c5_moderate_link",
        "c5_moderate_link",
    ]


def test_collect_jobs_applies_environment_filter(monkeypatch, tmp_path):
    module = _load_module()
    monkeypatch.setattr(module.dataset_metadata, "DATA_DIR", tmp_path)
    info = {
        "files": {
            "static_presence": [
                _entry("bedroom.npz", "C3", environment="bedroom"),
                _entry("lab.npz", "C3", environment="lab"),
            ]
        }
    }

    jobs = module.collect_jobs(
        info,
        chips=("C3",),
        labels=("static_presence",),
        environment="bedroom",
    )

    assert [job.source_path.name for job in jobs] == ["bedroom.npz"]


def test_collect_jobs_skips_groups_covered_by_real_low_rssi_pair(
    monkeypatch, tmp_path
):
    module = _load_module()
    monkeypatch.setattr(module.dataset_metadata, "DATA_DIR", tmp_path)
    info = {
        "files": {
            "empty": [_entry("empty_c3.npz", "C3", environment="bedroom")],
            "static_presence": [
                _entry("static_c3.npz", "C3", environment="bedroom"),
                _entry("static_c3_lab.npz", "C3", environment="lab"),
                _entry("static_c5.npz", "C5", environment="lab"),
                _entry(
                    "static_c3_low.npz",
                    "C3",
                    environment="bedroom",
                    low_rssi=True,
                    pair="motion_c3_low.npz",
                ),
            ],
            "motion": [
                _entry("motion_c3.npz", "C3", environment="bedroom"),
                _entry("motion_c3_lab.npz", "C3", environment="lab"),
                _entry("motion_c5.npz", "C5", environment="lab"),
                _entry(
                    "motion_c3_low.npz",
                    "C3",
                    environment="bedroom",
                    low_rssi=True,
                ),
            ],
        }
    }

    jobs = module.collect_jobs(
        info,
        chips=("C3", "C5"),
        labels=module.LABEL_ORDER,
    )

    assert [job.source_path.name for job in jobs] == [
        "static_c3_lab.npz",
        "static_c5.npz",
        "motion_c3_lab.npz",
        "motion_c5.npz",
    ]


def test_run_jobs_forwards_batch_configuration(monkeypatch, tmp_path):
    module = _load_module()
    calls = []
    jobs = [
        module.GenerationJob(
            label="static_presence",
            chip="C3",
            source_path=tmp_path / "static.npz",
            profile_name="c3_weak_link",
        ),
        module.GenerationJob(
            label="motion",
            chip="C3",
            source_path=tmp_path / "motion.npz",
            profile_name="c3_weak_link",
        ),
    ]

    monkeypatch.setattr(
        module.generator,
        "build_output_path",
        lambda source, label, profile, mode, seed: tmp_path
        / f"{source.stem}_{mode}_{seed}.npz",
    )
    monkeypatch.setattr(
        module.generator,
        "generate_dataset",
        lambda source, **kwargs: calls.append((source, kwargs)),
    )

    result = module.run_jobs(
        jobs,
        mode="shared_session",
        seed=42,
        dry_run=False,
        force=False,
    )

    assert result == (2, 0, 0)
    assert [call[0].name for call in calls] == ["static.npz", "motion.npz"]
    assert all(call[1]["generation_mode"] == "shared_session" for call in calls)
    assert all(call[1]["seed"] == 42 for call in calls)
    assert all(call[1]["register"] is True for call in calls)


def test_default_cli_targets_ml_augmentation():
    module = _load_module()

    args = module.build_argument_parser().parse_args([])

    assert args.mode == "shared_session"
    assert tuple(args.labels) == module.LABEL_ORDER
    assert set(args.chips) == set(module.PROFILE_BY_CHIP)
