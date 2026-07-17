"""
ESPectre - Dataset Metadata Refresh Tests

Tests for dataset metadata refresh helpers in the validator.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
MODULE_PATH = TOOLS_DIR / "validate_dataset_quality.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("validate_dataset_quality", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_refresh_metadata_writes_pair_fields():
    module = _load_module()
    info = {
        "updated_at": "2026-07-04T00:00:00",
        "files": {
            "empty": [
                {
                    "filename": "empty_s3_64sc_dev1_20260704_100000_0001.npz",
                    "chip": "S3",
                    "subcarriers": 64,
                    "collected_at": "2026-07-04T10:00:00.000000"
                }
            ],
            "static_presence": [
                {
                    "filename": "static_presence_s3_64sc_dev1_20260704_113202_0001.npz",
                    "chip": "S3",
                    "subcarriers": 64,
                    "collected_at": "2026-07-04T11:32:02.000000"
                }
            ],
            "motion": [
                {
                    "filename": "motion_s3_64sc_dev1_20260704_113807_0001.npz",
                    "chip": "S3",
                    "subcarriers": 64,
                    "collected_at": "2026-07-04T11:38:07.000000"
                }
            ],
            "test": [],
        },
    }

    refreshed, pair_rows = module.refresh_metadata(info)

    static_entry = refreshed["files"]["static_presence"][0]
    motion_entry = refreshed["files"]["motion"][0]
    empty_entry = refreshed["files"]["empty"][0]

    assert static_entry["optimal_pair_motion_file"] == motion_entry["filename"]
    assert motion_entry["optimal_pair_static_presence_file"] == static_entry["filename"]
    assert pair_rows == [
        {
            "static_presence": static_entry["filename"],
            "motion": motion_entry["filename"],
            "delta_seconds": 365.0,
        }
    ]
    assert empty_entry["filename"] == "empty_s3_64sc_dev1_20260704_100000_0001.npz"
    assert refreshed["updated_at"] == "2026-07-04T00:00:00"
