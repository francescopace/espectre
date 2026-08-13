# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Tests for dataset-quality primitives called by the supported collect CLI."""

import importlib.util
from pathlib import Path

import numpy as np


VALIDATOR_PATH = Path(__file__).resolve().parents[2] / "tools" / "validate_dataset_quality.py"


def _load_validator_module():
    spec = importlib.util.spec_from_file_location(
        "dataset_quality_validation",
        VALIDATOR_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Capture:
    def __init__(self, **values) -> None:
        self.values = {key: np.asarray(value) for key, value in values.items()}
        self.files = list(self.values)

    def __getitem__(self, key):
        return self.values[key]


def _by_name(results):
    return {result.name: result for result in results}


def test_validate_file_integrity_rejects_object_arrays(tmp_path) -> None:
    module = _load_validator_module()
    filepath = tmp_path / "malicious_dataset.npz"
    np.savez_compressed(
        filepath,
        csi_data=np.zeros((1, 128), dtype=np.int8),
        num_subcarriers=64,
        chip=np.array("c6", dtype=object),
        label="motion",
    )

    results, data = module.validate_file_integrity(filepath)

    assert data is None
    assert results[0].name == "file_load"
    assert results[0].status == "FAIL"
    assert "Unsafe NPZ dataset" in results[0].message


def test_file_integrity_rejects_subcarrier_shape_mismatch(tmp_path) -> None:
    module = _load_validator_module()
    path = tmp_path / "bad.npz"
    np.savez(
        path,
        csi_data=np.zeros((10, 128), dtype=np.int8),
        num_subcarriers=np.array(52),
    )

    results, data = module.validate_file_integrity(path)
    shape = _by_name(results)["csi_shape"]

    assert data is not None
    assert shape.status == "FAIL"
    assert "implies 64 subcarriers" in shape.message


def test_file_integrity_returns_ht20_sensing_view(tmp_path) -> None:
    module = _load_validator_module()
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

    results, data = module.validate_file_integrity(path)

    assert _by_name(results)["file_load"].status == "PASS"
    assert data is not None
    assert data["csi_data"].shape[0] == 3
    np.testing.assert_array_equal(
        data["stream_seq_num"],
        np.array([1, 3, 5], dtype=np.uint32),
    )


def test_signal_quality_checks_packet_count_zero_packets_and_amplitude() -> None:
    module = _load_validator_module()
    strong_packet = np.tile(np.array([30, 40], dtype=np.int8), 64)
    healthy = np.tile(strong_packet, (module.MIN_PACKETS, 1))

    healthy_results = _by_name(module.validate_signal_quality(healthy))
    empty_results = _by_name(
        module.validate_signal_quality(np.zeros((1, 128), dtype=np.int8))
    )

    assert healthy_results["packet_count"].status == "PASS"
    assert healthy_results["zero_packets"].status == "PASS"
    assert healthy_results["signal_level"].status == "PASS"
    assert empty_results["packet_count"].status == "FAIL"
    assert empty_results["zero_packets"].status == "WARN"
    assert empty_results["signal_level"].status == "WARN"


def test_capture_continuity_sees_gaps_after_ht20_filter(tmp_path) -> None:
    module = _load_validator_module()
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

    _, data = module.validate_file_integrity(path)
    assert data is not None

    continuity = _by_name(
        module.validate_capture_continuity(data, data["csi_data"])
    )
    assert continuity["stream_seq_gaps"].status in {"WARN", "FAIL"}
    assert continuity["stream_seq_gaps"].value > 0.0


def test_capture_continuity_flags_low_rate_and_stream_gaps() -> None:
    module = _load_validator_module()
    data = _Capture(
        duration_ms=1000.0,
        stream_seq_num=np.array([10, 11, 12, 60], dtype=np.uint32),
    )

    results = _by_name(
        module.validate_capture_continuity(
            data,
            np.zeros((4, 128), dtype=np.int8),
        )
    )

    assert results["packet_rate"].status == "WARN"
    assert results["stream_seq_gaps"].status == "FAIL"
    assert results["stream_seq_max_gap"].status == "FAIL"
    assert "after packet 2 (seq 12 -> 60)" in results["stream_seq_max_gap"].message


def test_capture_continuity_uses_low_rssi_loss_ceiling() -> None:
    module = _load_validator_module()
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
        module.validate_capture_continuity(
            bounded,
            np.zeros((96, 128), dtype=np.int8),
        )
    )
    low_rssi = _by_name(
        module.validate_capture_continuity(
            bounded,
            np.zeros((96, 128), dtype=np.int8),
            low_rssi=True,
        )
    )
    rejected = _by_name(
        module.validate_capture_continuity(
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
    module = _load_validator_module()
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
        module.validate_capture_continuity(
            data,
            np.zeros((95, 128), dtype=np.int8),
        )
    )

    assert results["packet_rate"].status == "PASS"
    assert results["packet_rate"].value == 95.0
    assert results["inter_packet_gap"].status == "PASS"
    assert results["inter_packet_gap"].value == 150.0


def test_capture_continuity_rejects_large_inter_packet_gap() -> None:
    module = _load_validator_module()
    data = _Capture(
        duration_ms=1000.0,
        stream_seq_num=np.array([1, 2, 3, 4], dtype=np.uint32),
        device_ticks_us=np.array([0, 10_000, 20_000, 2_500_000], dtype=np.uint64),
    )

    results = _by_name(
        module.validate_capture_continuity(
            data,
            np.zeros((4, 128), dtype=np.int8),
        )
    )

    assert results["inter_packet_gap"].status == "FAIL"
    assert results["inter_packet_gap"].value == 2480.0
    assert "at packet 2->3" in results["inter_packet_gap"].message
