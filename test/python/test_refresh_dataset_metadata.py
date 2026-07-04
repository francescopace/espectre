"""Tests for dataset metadata refresh tooling."""

from __future__ import annotations

import importlib.util
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
MODULE_PATH = TOOLS_DIR / "3_refresh_dataset_metadata.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("refresh_dataset_metadata", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_refresh_metadata_writes_pair_fields_and_inherits_motion_threshold(monkeypatch):
    module = _load_module()

    def fake_load_packets_for(label, filename):
        return [{"csi_data": [1, 2, 3, 4], "label": label, "filename": filename}]

    monkeypatch.setattr(module, "load_packets_for", fake_load_packets_for)
    monkeypatch.setattr(module, "compute_threshold_info", lambda packets: {"threshold": 0.123456789})

    info = {
        "updated_at": "2026-07-04T00:00:00",
        "files": {
            "empty": [],
            "static_presence": [
                {
                    "filename": "static_presence_s3_64sc_dev1_20260704_113202_0001.npz",
                    "chip": "S3",
                    "subcarriers": 64,
                    "collected_at": "2026-07-04T11:32:02.000000",
                }
            ],
            "motion": [
                {
                    "filename": "motion_s3_64sc_dev1_20260704_113807_0001.npz",
                    "chip": "S3",
                    "subcarriers": 64,
                    "collected_at": "2026-07-04T11:38:07.000000",
                }
            ],
            "test": [],
        },
    }

    refreshed, pair_rows, threshold_rows = module.refresh_metadata(info)

    static_entry = refreshed["files"]["static_presence"][0]
    motion_entry = refreshed["files"]["motion"][0]

    assert static_entry["optimal_pair_motion_file"] == motion_entry["filename"]
    assert motion_entry["optimal_pair_static_presence_file"] == static_entry["filename"]
    assert static_entry["optimal_threshold_gridsearch"] == 0.123456789
    assert motion_entry["optimal_threshold_gridsearch"] == 0.123456789
    assert pair_rows == [
        {
            "static_presence": static_entry["filename"],
            "motion": motion_entry["filename"],
            "delta_seconds": 365.0,
        }
    ]
    assert ("static_presence", static_entry["filename"], 0.123456789, static_entry["filename"]) in threshold_rows
    assert ("motion", motion_entry["filename"], 0.123456789, static_entry["filename"]) in threshold_rows
