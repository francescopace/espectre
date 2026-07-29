import numpy as np
import pytest

from tools.lib import npz_cache
from tools.lib import csi_io


@pytest.fixture(autouse=True)
def isolated_cache_root(tmp_path, monkeypatch):
    """Keep every cache test off the shared workspace cache.

    These tests clear whole artifact trees, so without redirection they delete
    the developer's working cache and force an unrelated cold rebuild.
    """
    root = tmp_path / "npz_cache_root"
    monkeypatch.setenv(npz_cache.NPZ_CACHE_DIR_ENV, str(root))
    npz_cache.clear_runtime_artifacts()
    yield root
    npz_cache.clear_runtime_artifacts()


def _write_source_npz(path, *, values):
    np.savez(path, csi_data=np.asarray([values], dtype=np.int8))


def test_cache_root_follows_the_environment_override(isolated_cache_root):
    assert npz_cache.npz_cache_dir() == isolated_cache_root


def test_runtime_artifact_reuses_value_until_source_changes(tmp_path):
    source_path = tmp_path / "sample.npz"
    _write_source_npz(source_path, values=[1, 2, 3, 4])

    calls = {"count": 0}

    def build():
        calls["count"] += 1
        return {"call": calls["count"]}

    first = npz_cache.get_runtime_artifact(
        source_path,
        artifact_name="unit_runtime",
        artifact_version=1,
        builder=build,
    )
    second = npz_cache.get_runtime_artifact(
        source_path,
        artifact_name="unit_runtime",
        artifact_version=1,
        builder=build,
    )

    assert first is second
    assert calls["count"] == 1

    _write_source_npz(source_path, values=[1, 2, 3, 4, 5, 6])

    third = npz_cache.get_runtime_artifact(
        source_path,
        artifact_name="unit_runtime",
        artifact_version=1,
        builder=build,
    )

    assert third is not first
    assert calls["count"] == 2


def test_runtime_cache_evicts_oldest_entries_when_capacity_is_exceeded(tmp_path):
    calls = {"count": 0}
    source_paths = []
    for index in range(npz_cache.RUNTIME_CACHE_MAX_ENTRIES + 1):
        source_path = tmp_path / f"runtime_{index}.npz"
        _write_source_npz(source_path, values=[index, index + 1])
        source_paths.append(source_path)
        npz_cache.get_runtime_artifact(
            source_path,
            artifact_name="unit_runtime",
            artifact_version=1,
            builder=lambda index=index: {"call": index},
        )

    first_source = source_paths[0]

    def rebuild():
        calls["count"] += 1
        return {"call": "rebuilt"}

    rebuilt = npz_cache.get_runtime_artifact(
        first_source,
        artifact_name="unit_runtime",
        artifact_version=1,
        builder=rebuild,
    )

    assert rebuilt == {"call": "rebuilt"}
    assert calls["count"] == 1


def test_packet_view_cache_does_not_pin_raw_arrays(tmp_path):
    source_path = tmp_path / "packet_only.npz"
    np.savez(
        source_path,
        csi_data=np.zeros((1, 128), dtype=np.int8),
        num_subcarriers=np.array(64),
        label=np.array("motion"),
        chip=np.array("c3"),
    )

    packets = csi_io.load_npz_packet_view(source_path)

    assert len(packets) == 1
    assert all(key[0] != "raw_arrays" for key in npz_cache._RUNTIME_CACHE)


def test_packet_csi_data_accepts_read_only_packet_views(tmp_path):
    import tools.train_ml_model as trainer

    source_path = tmp_path / "packet_view_mapping.npz"
    np.savez(
        source_path,
        csi_data=np.arange(128, dtype=np.int8).reshape(1, 128),
        num_subcarriers=np.array(64),
        label=np.array("motion"),
        chip=np.array("c3"),
    )

    packet_view = csi_io.load_npz_packet_view(source_path)

    csi_data = trainer.packet_csi_data(packet_view[0])

    assert isinstance(csi_data, np.ndarray)
    np.testing.assert_array_equal(csi_data, packet_view[0]["csi_data"])


def test_compare_detection_methods_accepts_read_only_packet_views(tmp_path):
    import tools.compare_detection_methods as compare_methods

    source_path = tmp_path / "compare_packet_view_mapping.npz"
    np.savez(
        source_path,
        csi_data=np.arange(128, dtype=np.int8).reshape(1, 128),
        num_subcarriers=np.array(64),
        label=np.array("motion"),
        chip=np.array("c3"),
    )

    packet_view = csi_io.load_npz_packet_view(source_path)

    csi_data = compare_methods._packet_csi_data(packet_view[0])

    assert isinstance(csi_data, np.ndarray)
    np.testing.assert_array_equal(csi_data, packet_view[0]["csi_data"])


def test_feature_matrix_artifact_roundtrip_tracks_source_identity(tmp_path):
    source_path = tmp_path / "features.npz"
    _write_source_npz(source_path, values=[1, 2, 3, 4])

    params = npz_cache.feature_matrix_parameters(
        feature_names=["f0", "f1"],
        window_size=4,
        subcarriers=(1, 2),
    )
    expected_X = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    npz_cache.save_feature_matrix_artifact(
        source_path,
        parameters=params,
        X=expected_X,
        feature_names=["f0", "f1"],
    )

    cached = npz_cache.load_feature_matrix_artifact(source_path, parameters=params)

    assert cached is not None
    np.testing.assert_allclose(cached["X"], expected_X)
    assert cached["feature_names"] == ["f0", "f1"]

    _write_source_npz(source_path, values=[9, 8, 7, 6, 5, 4])

    assert npz_cache.load_feature_matrix_artifact(source_path, parameters=params) is None


def test_source_identity_survives_a_modification_time_rewrite(tmp_path):
    """A checkout rewrites mtime without changing content; the cache must hit."""
    source_path = tmp_path / "checkout.npz"
    _write_source_npz(source_path, values=[1, 2, 3, 4])

    params = npz_cache.feature_matrix_parameters(
        feature_names=["f0"],
        window_size=4,
        subcarriers=(1,),
    )
    npz_cache.save_feature_matrix_artifact(
        source_path,
        parameters=params,
        X=np.asarray([[1.0]], dtype=np.float32),
        feature_names=["f0"],
    )

    stat = source_path.stat()
    import os

    os.utime(source_path, ns=(stat.st_atime_ns, stat.st_mtime_ns - 5_000_000_000))

    assert npz_cache.load_feature_matrix_artifact(source_path, parameters=params) is not None


def test_artifact_parameters_accept_numpy_and_tuple_values(tmp_path):
    """Parameters must round-trip, or every lookup becomes a silent miss."""
    source_path = tmp_path / "params.npz"
    _write_source_npz(source_path, values=[1, 2, 3, 4])

    params = {
        "threshold": np.float32(0.5),
        "band": (1, 2, 3),
        "count": np.int64(7),
        "flag": np.bool_(True),
    }
    npz_cache.save_npz_artifact(
        source_path,
        artifact_name="unit_params",
        artifact_version=1,
        parameters=params,
        payload={"v": np.arange(3)},
    )

    assert npz_cache.load_npz_artifact(
        source_path,
        artifact_name="unit_params",
        artifact_version=1,
        parameters=params,
    ) is not None


def test_disabled_filters_do_not_fragment_the_feature_key():
    """Two callers that disable a filter must address the same artifact."""
    common = {
        "feature_names": ["f0"],
        "window_size": 100,
        "subcarriers": (1, 2),
    }
    left = npz_cache.feature_matrix_parameters(
        **common, enable_lowpass=False, lowpass_cutoff=0.0
    )
    right = npz_cache.feature_matrix_parameters(
        **common, enable_lowpass=False, lowpass_cutoff=11.0
    )

    assert left == right

    enabled = npz_cache.feature_matrix_parameters(
        **common, enable_lowpass=True, lowpass_cutoff=11.0
    )
    assert enabled != left


def test_trainer_and_validator_address_the_same_feature_artifact(tmp_path):
    """The whole point of the shared cache: one key per reusable feature column.

    The trainer and the dataset-quality validator extract the same features from
    the same capture with the same filter chain. If their keys drift, every
    capture is extracted and stored twice, and a mislabeled key can serve one
    tool's data to the other.
    """
    import tools.train_ml_model as trainer
    import tools.validate_dataset_quality as validator

    feature_names = tuple(validator.VALIDATION_FEATURE_NAMES)
    validator_parameters = validator._validation_feature_cache_parameters(feature_names)
    trainer_parameters = trainer._feature_matrix_cache_parameters(list(feature_names))

    assert validator_parameters == trainer_parameters

    source_path = tmp_path / "shared.npz"
    _write_source_npz(source_path, values=[1, 2, 3, 4])
    _, validator_path = npz_cache.artifact_cache_path(
        source_path,
        artifact_name="feature_column",
        artifact_version=npz_cache.FEATURE_COLUMN_ARTIFACT_VERSION,
        parameters=npz_cache.feature_column_parameters(
            base_parameters=validator_parameters,
            feature_name=feature_names[0],
        ),
    )
    _, trainer_path = npz_cache.artifact_cache_path(
        source_path,
        artifact_name="feature_column",
        artifact_version=npz_cache.FEATURE_COLUMN_ARTIFACT_VERSION,
        parameters=npz_cache.feature_column_parameters(
            base_parameters=trainer_parameters,
            feature_name=feature_names[0],
        ),
    )

    assert validator_path == trainer_path


def test_subset_requests_reuse_cached_feature_columns(monkeypatch, tmp_path):
    import tools.train_ml_model as trainer

    source_path = tmp_path / "subset_source.npz"
    _write_source_npz(source_path, values=[1, 2, 3, 4])
    record = {
        "path": source_path,
        "packets": [{"csi_data": np.zeros(128, dtype=np.int8)}],
        "chip": "C3",
        "lineage_group": "demo",
        "session_group": "demo",
        "environment_group": "lab",
        "pair_id": "pair",
        "day_group": "2026-07-29",
        "dataset_role": "train",
        "synthetic": False,
        "label_name": "motion",
    }
    calls = []

    def fake_extract_features(
        packets,
        window_size,
        feature_names,
        enable_lowpass,
        lowpass_cutoff,
        enable_hampel,
        hampel_window,
        hampel_threshold,
    ):
        del packets, window_size, enable_lowpass, lowpass_cutoff
        del enable_hampel, hampel_window, hampel_threshold
        requested = tuple(feature_names)
        calls.append(requested)
        row_map = {
            ("f0",): np.asarray([[1.0]], dtype=np.float32),
            ("f1",): np.asarray([[2.0]], dtype=np.float32),
        }
        return row_map[requested], None, list(requested), None

    monkeypatch.setattr(trainer, "extract_features", fake_extract_features)

    first, first_cached = trainer._load_or_compute_file_feature_matrix(
        record,
        feature_names=["f0"],
        use_cache=True,
    )
    second, second_cached = trainer._load_or_compute_file_feature_matrix(
        record,
        feature_names=["f0", "f1"],
        use_cache=True,
    )

    assert first_cached is False
    assert second_cached is False
    assert calls == [("f0",), ("f1",)]
    np.testing.assert_allclose(first["X"], np.asarray([[1.0]], dtype=np.float32))
    np.testing.assert_allclose(second["X"], np.asarray([[1.0, 2.0]], dtype=np.float32))
    assert second["feature_names"] == ["f0", "f1"]


def test_clear_persisted_artifacts_removes_selected_artifact_tree(tmp_path):
    source_path = tmp_path / "baseline.npz"
    _write_source_npz(source_path, values=[1, 2, 3, 4])

    params = {"feature_names": ["f0"], "packet_rate_pps": 100.0}
    npz_cache.save_idle_baseline_artifact(
        source_path,
        parameters=params,
        baseline={"score": 95.0, "fp_rate": 0.0},
        median_rssi_dbm=-42.0,
    )

    artifact_root = npz_cache.artifact_dir("idle_baseline")
    assert artifact_root.exists()

    npz_cache.clear_persisted_artifacts("idle_baseline")

    assert not artifact_root.exists()


def test_prune_removes_only_unreachable_artifacts(tmp_path):
    live_source = tmp_path / "live.npz"
    dead_source = tmp_path / "dead.npz"
    _write_source_npz(live_source, values=[1, 2, 3, 4])
    _write_source_npz(dead_source, values=[5, 6, 7, 8])

    params = npz_cache.feature_matrix_parameters(
        feature_names=["f0"],
        window_size=4,
        subcarriers=(1,),
    )
    live_path = npz_cache.save_feature_matrix_artifact(
        live_source,
        parameters=params,
        X=np.asarray([[1.0]], dtype=np.float32),
        feature_names=["f0"],
    )
    dead_path = npz_cache.save_feature_matrix_artifact(
        dead_source,
        parameters=params,
        X=np.asarray([[2.0]], dtype=np.float32),
        feature_names=["f0"],
    )
    dead_source.unlink()

    removed = npz_cache.prune_persisted_artifacts()

    assert removed["missing_source"] == 1
    assert not dead_path.exists()
    assert live_path.exists()
    assert npz_cache.load_feature_matrix_artifact(live_source, parameters=params) is not None


def test_detector_replay_artifact_roundtrip_preserves_secondary_source_identity(tmp_path):
    static_source = tmp_path / "static.npz"
    motion_source = tmp_path / "motion.npz"
    _write_source_npz(static_source, values=[1, 2, 3, 4])
    _write_source_npz(motion_source, values=[5, 6, 7, 8])

    parameters = npz_cache.detector_replay_parameters(
        replay_kind="classic_dataset",
        selected_subcarriers=(1, 2, 3),
        window_size=4,
        secondary_source=motion_source,
    )
    result = {
        "adaptive_threshold": 0.5,
        "metrics": {"recall": 100.0, "fp_rate": 0.0},
    }
    npz_cache.save_detector_replay_artifact(
        static_source,
        parameters=parameters,
        result=result,
    )

    cached = npz_cache.load_detector_replay_artifact(
        static_source,
        parameters=parameters,
    )

    assert cached == result


def test_classic_dataset_result_reuses_persisted_replay(monkeypatch, tmp_path):
    import tools.lib.performance_report as performance_report

    static_source = tmp_path / "static_pair.npz"
    motion_source = tmp_path / "motion_pair.npz"
    _write_source_npz(static_source, values=[1, 2, 3, 4])
    _write_source_npz(motion_source, values=[5, 6, 7, 8])
    calls = {"load": 0, "compute": 0}

    def fake_load_real_data_cached(*_args, **_kwargs):
        calls["load"] += 1
        return (("static",), ("motion",))

    def fake_compute_classic_packet_result(*_args, **_kwargs):
        calls["compute"] += 1
        return 0.75, {"recall": 100.0, "fp_rate": 0.0}

    monkeypatch.setattr(performance_report, "load_real_data_cached", fake_load_real_data_cached)
    monkeypatch.setattr(
        performance_report,
        "compute_classic_packet_result",
        fake_compute_classic_packet_result,
    )
    performance_report.compute_classic_dataset_result.cache_clear()

    first = performance_report.compute_classic_dataset_result(
        static_source,
        motion_source,
        (1, 2, 3),
        4,
    )
    performance_report.compute_classic_dataset_result.cache_clear()
    second = performance_report.compute_classic_dataset_result(
        static_source,
        motion_source,
        (1, 2, 3),
        4,
    )

    assert first == second
    assert calls == {"load": 1, "compute": 1}
