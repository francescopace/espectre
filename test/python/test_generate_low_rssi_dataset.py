"""
Tests for tools/generate_low_rssi_dataset.py.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "tools" / "generate_low_rssi_dataset.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "generate_low_rssi_dataset", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_source(path, label, seed):
    rng = np.random.default_rng(seed)
    packet_count = 400
    base = rng.normal(0.0, 22.0, size=(packet_count, 128))
    np.savez_compressed(
        path,
        csi_data=np.clip(np.rint(base), -128, 127).astype(np.int8),
        num_subcarriers=np.asarray(64),
        label=np.asarray(label),
        chip=np.asarray("C3"),
        collected_at=np.asarray("2026-07-22T10:00:00"),
        duration_ms=np.asarray(4000.0),
        format_version=np.asarray("1.2"),
        stream_seq_num=np.arange(packet_count, dtype=np.uint32),
        device_ticks_us=np.arange(packet_count, dtype=np.uint64) * 10000,
        rssi_dbm=np.full(packet_count, -55, dtype=np.int16),
        device_id=np.asarray(1, dtype=np.uint64),
    )


def _source_entry(filename, label, counterpart):
    entry = {
        "filename": filename,
        "chip": "C3",
        "subcarriers": 64,
        "contributor": "tester",
        "collected_at": "2026-07-22T10:00:00",
        "duration_ms": 4000,
        "num_packets": 400,
        "description": f"Real {label}",
        "environment": "lab",
        "device_id": "0x0000000000000001",
    }
    if label == "static_presence":
        entry["optimal_pair_motion_file"] = counterpart
    else:
        entry["optimal_pair_static_presence_file"] = counterpart
    return entry


def test_generation_group_is_shared_by_pair_and_separates_modes():
    module = _load_module()
    static = _source_entry("static.npz", "static_presence", "motion.npz")
    motion = _source_entry("motion.npz", "motion", "static.npz")

    static_group = module.build_generation_group(
        "static_presence", static, "c3_weak_link", 42, "reference_match"
    )
    motion_group = module.build_generation_group(
        "motion", motion, "c3_weak_link", 42, "reference_match"
    )

    assert static_group == motion_group
    assert static_group != module.build_generation_group(
        "static_presence", static, "c3_weak_link", 42, "shared_session"
    )


def test_shared_session_preserves_motion_dynamics():
    module = _load_module()
    parameters = module.ImpairmentParameters(
        source_retention=0.75,
        jitter_sigma=0.2,
        temporal_rho=0.3,
        noise_sigma_abs=0.4,
        spatial_spread=1.2,
        turbulence_retention=0.65,
        turbulence_noise=0.1,
        turbulence_rho=0.4,
        turbulence_skew=0.2,
    )

    effective = module._effective_impairment_parameters(
        parameters,
        label="motion",
        generation_mode="shared_session",
    )

    assert effective.source_retention == 1.0
    assert effective.spatial_spread == pytest.approx(1.32)
    assert effective.turbulence_retention == 1.0
    assert effective.jitter_sigma == parameters.jitter_sigma
    assert effective.noise_sigma_abs == parameters.noise_sigma_abs
    assert module._effective_impairment_parameters(
        parameters,
        label="static_presence",
        generation_mode="shared_session",
    ) == parameters


def test_s3_profile_matches_real_reference_pair():
    module = _load_module()
    profile = module.LOW_RSSI_PROFILES["s3_weak_link"]

    assert profile.reference_chip == "S3"
    assert profile.target_quiet_rssi_dbm == pytest.approx(-77.0)
    assert profile.target_motion_rssi_dbm == pytest.approx(-75.0)
    assert profile.packet_loss == 0.0
    assert profile.reference_datasets == (
        "static_presence_s3_64sc_dev000010b41de8ec00_20260722_172043_630431_0001.npz",
        "motion_s3_64sc_dev000010b41de8ec00_20260722_172305_879358_0001.npz",
    )


def test_c3_profile_matches_real_reference_pair():
    module = _load_module()
    profile = module.LOW_RSSI_PROFILES["c3_weak_link"]

    assert profile.reference_chip == "C3"
    assert profile.target_quiet_rssi_dbm == pytest.approx(-77.0)
    assert profile.target_motion_rssi_dbm == pytest.approx(-77.0)
    assert profile.packet_loss == pytest.approx(0.0010)
    assert profile.reference_datasets == (
        "static_presence_c3_64sc_dev0000acebe64ae708_20260722_210321_712831_0001.npz",
        "motion_c3_64sc_dev0000acebe64ae708_20260722_210523_413343_0001.npz",
    )


def test_c5_profile_matches_real_reference_pair():
    module = _load_module()
    profile = module.LOW_RSSI_PROFILES["c5_moderate_link"]

    assert profile.reference_chip == "C5"
    assert profile.target_quiet_rssi_dbm == pytest.approx(-75.0)
    assert profile.target_motion_rssi_dbm == pytest.approx(-71.0)
    assert profile.packet_loss == 0.0
    assert profile.reference_datasets == (
        "static_presence_c5_64sc_dev000030eda0e46278_20260722_205156_405317_0001.npz",
        "motion_c5_64sc_dev000030eda0e46278_20260722_205350_355335_0001.npz",
    )


def test_c6_profile_matches_real_reference_pair():
    module = _load_module()
    profile = module.LOW_RSSI_PROFILES["c6_moderate_link"]

    assert profile.reference_chip == "C6"
    assert profile.target_quiet_rssi_dbm == pytest.approx(-69.0)
    assert profile.target_motion_rssi_dbm == pytest.approx(-66.0)
    assert profile.packet_loss == 0.0
    assert profile.reference_datasets == (
        "static_presence_c6_64sc_dev00007c2c6742bbac_20260722_191653_148862_0001.npz",
        "motion_c6_64sc_dev00007c2c6742bbac_20260722_191914_560463_0001.npz",
    )


def test_shared_session_reuses_quiet_calibration_and_registers_provenance(
    monkeypatch, tmp_path
):
    module = _load_module()
    data_dir = tmp_path / "data"
    static_dir = data_dir / "static_presence"
    motion_dir = data_dir / "motion"
    static_dir.mkdir(parents=True)
    motion_dir.mkdir(parents=True)
    static_source = static_dir / "static.npz"
    motion_source = motion_dir / "motion.npz"
    _write_source(static_source, "static_presence", 1)
    _write_source(motion_source, "motion", 2)

    info_path = data_dir / "dataset_info.json"
    info = {
        "format_version": "1.2",
        "created_at": "2026-07-22T10:00:00",
        "updated_at": "2026-07-22T10:00:00",
        "labels": {
            "static_presence": {"description": "Static"},
            "motion": {"description": "Motion"},
        },
        "files": {
            "static_presence": [
                _source_entry("static.npz", "static_presence", "motion.npz")
            ],
            "motion": [_source_entry("motion.npz", "motion", "static.npz")],
        },
    }
    module.dataset_metadata.save_dataset_info(info, info_path)
    monkeypatch.setattr(module.dataset_metadata, "DATA_DIR", data_dir)

    static_output = static_dir / "static_syn.npz"
    motion_output = motion_dir / "motion_syn.npz"
    _, static_entry = module.generate_dataset(
        static_source,
        profile_name="c3_weak_link",
        seed=42,
        generation_mode="shared_session",
        output_path=static_output,
        register=True,
        dataset_info_path=info_path,
    )
    _, motion_entry = module.generate_dataset(
        motion_source,
        profile_name="c3_weak_link",
        seed=42,
        generation_mode="shared_session",
        output_path=motion_output,
        register=True,
        dataset_info_path=info_path,
    )

    assert static_entry["low_rssi"] is True
    assert static_entry["synthetic"] is True
    assert "generation" not in static_entry
    assert "training_eligible" not in static_entry
    assert "synthetic_group" not in static_entry
    assert "relative_path" not in static_entry

    refreshed = module.dataset_metadata.load_dataset_info(info_path)
    registered_static = next(
        entry
        for entry in refreshed["files"]["static_presence"]
        if entry["filename"] == "static_syn.npz"
    )
    registered_motion = next(
        entry
        for entry in refreshed["files"]["motion"]
        if entry["filename"] == "motion_syn.npz"
    )
    assert registered_static["optimal_pair_motion_file"] == "motion_syn.npz"
    assert registered_motion["optimal_pair_static_presence_file"] == "static_syn.npz"

    with np.load(static_output, allow_pickle=False) as generated:
        generated_packets = len(generated["csi_data"])
        assert bool(generated["low_rssi"].item()) is True
        assert bool(generated["synthetic"].item()) is True
        assert str(generated["source_dataset"].item()) == "static.npz"
        assert len(generated["stream_seq_num"]) == generated_packets
        assert len(generated["device_ticks_us"]) == generated_packets
        assert len(generated["rssi_dbm"]) == generated_packets
        assert float(np.median(generated["rssi_dbm"])) == pytest.approx(-77.0)
        # Fit metadata covers the intersection of the production feature set
        # with the profile's calibrated reference medians: profiles calibrated
        # under an older production set may lack targets for newer features.
        profile = module.LOW_RSSI_PROFILES["c3_weak_link"]
        fitted_names = [
            name for name in module.FEATURE_NAMES
            if name in profile.reference_feature_medians["static_presence"]
        ]
        assert list(generated["feature_names"]) == fitted_names
        assert generated["source_feature_medians"].shape == (len(fitted_names),)
        assert generated["target_feature_medians"].shape == (len(fitted_names),)
        assert generated["synthetic_feature_medians"].shape == (len(fitted_names),)
        assert generated["feature_relative_errors"].shape == (len(fitted_names),)
        assert float(generated["mean_feature_relative_error"]) >= 0.0
        assert str(generated["deformation_mode"]) == "gain_jitter"
        assert generated["reference_datasets"].shape == (2,)
        assert int(generated["generator_version"]) == module.GENERATOR_VERSION
        with np.load(motion_output, allow_pickle=False) as generated_motion:
            for parameter_name in (
                "jitter_sigma",
                "temporal_rho",
                "noise_sigma_abs",
                "turbulence_noise",
                "turbulence_rho",
                "turbulence_skew",
            ):
                assert float(generated[parameter_name]) == pytest.approx(
                    float(generated_motion[parameter_name])
                )
            assert float(generated_motion["source_retention"]) == 1.0
            assert float(generated_motion["spatial_spread"]) == pytest.approx(
                float(generated["spatial_spread"])
                * module.SHARED_SESSION_MOTION_SPATIAL_BOOST
            )
            assert float(generated_motion["turbulence_retention"]) == 1.0
