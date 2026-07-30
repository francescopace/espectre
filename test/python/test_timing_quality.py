import numpy as np
import pytest
import tools.train_ml_model as trainer
from tools import replay_classic_candidates
from tools.lib import performance_report
from tools.lib.timing_quality import merge_timing_summaries, summarize_capture_timing


def _timed_packets(
    count=32,
    *,
    gap_index=None,
    gap_us=400_000,
    missing_seq_step=None,
):
    packets = []
    timestamp_us = 0
    seq_num = 100
    base_row = np.tile(np.asarray([20, -12], dtype=np.int8), 64)
    for index in range(count):
        step = 1
        delta_us = 10_000
        if gap_index is not None and index == gap_index:
            delta_us = gap_us
            if missing_seq_step is not None:
                step = missing_seq_step
        timestamp_us += delta_us
        seq_num += step
        packets.append(
            {
                "csi_data": base_row.copy(),
                "device_ticks_us": timestamp_us,
                "stream_seq_num": seq_num,
            }
        )
    return packets


def test_summarize_capture_timing_classifies_clean_stream():
    summary = summarize_capture_timing(_timed_packets())

    assert summary["quality_status"] == "PASS"
    assert summary["quality_bucket"] == "clean"
    assert summary["contaminated_packets"] == 0
    assert summary["packet_rate_pps"] == pytest.approx(100.0, rel=1e-3)
    assert summary["max_gap_ms"] == pytest.approx(10.0)


def test_summarize_capture_timing_classifies_gap_contamination():
    summary = summarize_capture_timing(
        _timed_packets(gap_index=20, gap_us=400_000, missing_seq_step=40)
    )

    assert summary["quality_status"] == "FAIL"
    assert summary["quality_bucket"] == "poor"
    assert summary["contaminated_packets"] >= 1
    assert summary["max_sequence_gap_packets"] >= 20
    assert summary["max_gap_ms"] == pytest.approx(400.0)


def test_merge_timing_summaries_keeps_the_worst_bucket():
    merged = merge_timing_summaries(
        summarize_capture_timing(_timed_packets()),
        summarize_capture_timing(
            _timed_packets(count=64, gap_index=18, gap_us=180_000, missing_seq_step=2)
        ),
    )

    assert merged["quality_status"] == "WARN"
    assert merged["quality_bucket"] == "degraded"
    assert merged["packet_rate_pps"] == pytest.approx(89.38, rel=1e-3)
    assert merged["max_gap_ms"] > 150.0


def test_build_ml_replay_rows_resets_after_contamination_gap():
    rows = performance_report.build_ml_replay_rows(
        _timed_packets(count=256, gap_index=128, gap_us=400_000, missing_seq_step=40),
        trainer.DEFAULT_SUBCARRIERS,
        trainer.SEG_WINDOW_SIZE,
        trainer.EXPORTED_FEATURE_NAMES,
    )

    assert rows["X"].shape[1] == len(trainer.EXPORTED_FEATURE_NAMES)
    assert len(rows["packet_index"]) == len(rows["evaluation_index"]) == len(rows["reset_index"])
    assert np.any(rows["reset_index"] > 0)
    assert rows["evaluation_index"][0] == 0
    assert np.all(np.diff(rows["packet_index"]) > 0)


def test_stream_dense_emits_every_packet_after_warmup_on_clean_stream():
    packets = _timed_packets(count=256)
    replay_rows = performance_report.build_ml_replay_rows(
        packets,
        trainer.DEFAULT_SUBCARRIERS,
        trainer.SEG_WINDOW_SIZE,
        trainer.EXPORTED_FEATURE_NAMES,
        sample_contract="replay_tick",
    )
    stream_dense_rows = performance_report.build_ml_replay_rows(
        packets,
        trainer.DEFAULT_SUBCARRIERS,
        trainer.SEG_WINDOW_SIZE,
        trainer.EXPORTED_FEATURE_NAMES,
        sample_contract="stream_dense",
    )

    assert len(stream_dense_rows["X"]) == len(packets) - trainer.SEG_WINDOW_SIZE + 1
    assert len(stream_dense_rows["X"]) > len(replay_rows["X"])
    assert stream_dense_rows["packet_index"][0] == trainer.SEG_WINDOW_SIZE - 1
    assert np.all(stream_dense_rows["reset_index"] == 0)


def test_load_training_matrix_preserves_timing_context_and_weights(monkeypatch):
    records = [
        {
            "path": trainer.Path("clean.npz"),
            "packets": (),
            "label_name": "empty",
            "is_motion": False,
            "chip": "C6",
            "collected_at": "",
            "day_group": "2026-07-30",
            "pair_id": "pair-a",
            "session_group": "session-a",
            "lineage_group": "lineage-a",
            "dataset_role": "train",
            "synthetic": False,
            "long_recording": False,
            "environment_group": "lab",
            "timing_quality_status": "PASS",
            "timing_quality_bucket": "clean",
            "timing_summary": {},
            "timing_weight": 1.0,
        },
        {
            "path": trainer.Path("warn.npz"),
            "packets": (),
            "label_name": "motion",
            "is_motion": True,
            "chip": "C6",
            "collected_at": "",
            "day_group": "2026-07-30",
            "pair_id": "pair-b",
            "session_group": "session-b",
            "lineage_group": "lineage-b",
            "dataset_role": "train",
            "synthetic": False,
            "long_recording": False,
            "environment_group": "lab",
            "timing_quality_status": "WARN",
            "timing_quality_bucket": "degraded",
            "timing_summary": {},
            "timing_weight": 0.25,
        },
    ]
    stats = {
        "chips": ["C6"],
        "labels": {"empty": 1, "motion": 1},
        "total": 2,
        "files": ["clean.npz", "warn.npz"],
        "excluded_labels": [],
        "excluded_chips": [],
        "excluded_environments": [],
        "excluded_missing_sync_metadata": [],
        "excluded_dataset_roles": [],
        "excluded_long_recordings": [],
        "excluded_timing_quality": [],
        "session_groups": ["session-a", "session-b"],
        "lineage_groups": ["lineage-a", "lineage-b"],
        "environment_groups": ["lab"],
        "sync_metadata_files": [],
        "timing_quality_counts": {
            "clean": 1,
            "degraded": 1,
            "poor": 0,
            "unknown": 0,
        },
    }

    def fake_load_training_file_records(**_kwargs):
        return records, stats

    def fake_load_or_compute_ml_replay_rows(path, **_kwargs):
        record = next(item for item in records if item["path"] == path)
        value = 1.0 if record["label_name"] == "empty" else 2.0
        return {
            "X": np.asarray([[value]], dtype=np.float32),
            "feature_names": [trainer.EXPORTED_FEATURE_NAMES[0]],
            "packet_index": np.asarray([trainer.SEG_WINDOW_SIZE - 1], dtype=np.int32),
            "evaluation_index": np.asarray([0], dtype=np.int32),
            "reset_index": np.asarray([0], dtype=np.int32),
            "cache_hit": True,
        }

    monkeypatch.setattr(trainer, "_load_training_file_records", fake_load_training_file_records)
    monkeypatch.setattr(
        trainer,
        "load_or_compute_ml_replay_rows",
        fake_load_or_compute_ml_replay_rows,
    )

    matrix, _ = trainer.load_training_matrix(
        feature_names=[trainer.EXPORTED_FEATURE_NAMES[0]],
        timing_quality_policy="downweight-warn",
        timing_warn_weight=0.25,
    )

    assert matrix["sample_context"]["timing_quality_bucket"].tolist() == [
        "clean",
        "degraded",
    ]
    np.testing.assert_allclose(
        matrix["sample_weights"],
        np.asarray([1.6, 0.4], dtype=np.float32),
        rtol=1e-6,
    )


def test_load_or_compute_ml_replay_rows_reuses_full_runtime_cache(monkeypatch, tmp_path):
    source_path = tmp_path / "capture.npz"
    packets = _timed_packets(count=256)
    np.savez(
        source_path,
        csi_data=np.asarray([packet["csi_data"] for packet in packets], dtype=np.int8),
        device_ticks_us=np.asarray([packet["device_ticks_us"] for packet in packets], dtype=np.int64),
        stream_seq_num=np.asarray([packet["stream_seq_num"] for packet in packets], dtype=np.int64),
        num_subcarriers=np.asarray(64),
        label=np.asarray("motion"),
        chip=np.asarray("c6"),
    )
    cached_rows = performance_report.build_ml_replay_rows(
        packets,
        trainer.DEFAULT_SUBCARRIERS,
        trainer.SEG_WINDOW_SIZE,
        trainer.EXPORTED_FEATURE_NAMES,
        sample_contract="stream_dense",
    )
    load_calls = []

    def fake_load(source, *, parameters):
        load_calls.append(parameters["feature_names"])
        return cached_rows

    monkeypatch.setattr(
        trainer.npz_cache,
        "load_ml_replay_row_artifact",
        fake_load,
    )
    monkeypatch.setattr(
        performance_report,
        "load_npz_packet_view",
        lambda *_args, **_kwargs: pytest.fail("cache hit must not load packet rows"),
    )

    rows = performance_report.load_or_compute_ml_replay_rows(
        source_path,
        selected_subcarriers=trainer.DEFAULT_SUBCARRIERS,
        window_size=trainer.SEG_WINDOW_SIZE,
        feature_names=trainer.EXPORTED_FEATURE_NAMES[:2],
        sample_contract="stream_dense",
        use_cache=True,
    )
    replay_rows = performance_report.load_or_compute_ml_replay_rows(
        source_path,
        selected_subcarriers=trainer.DEFAULT_SUBCARRIERS,
        window_size=trainer.SEG_WINDOW_SIZE,
        feature_names=trainer.EXPORTED_FEATURE_NAMES[:2],
        sample_contract="replay_tick",
        use_cache=True,
    )

    assert len(load_calls) == 2
    assert tuple(load_calls[0]) == tuple(trainer.EXPORTED_FEATURE_NAMES)
    assert load_calls[0] == load_calls[1]
    assert rows["feature_names"] == list(trainer.EXPORTED_FEATURE_NAMES[:2])
    np.testing.assert_allclose(rows["X"], cached_rows["X"][:, :2])
    evaluation_due = np.asarray(cached_rows["evaluation_due"], dtype=bool)
    np.testing.assert_allclose(
        replay_rows["X"],
        cached_rows["X"][evaluation_due, :2],
    )


def test_augmented_replay_rows_persist_only_for_matching_provenance(
    monkeypatch,
    tmp_path,
):
    cache_root = tmp_path / "cache"
    monkeypatch.setenv(trainer.npz_cache.NPZ_CACHE_DIR_ENV, str(cache_root))
    source_path = tmp_path / "capture.npz"
    packets = _timed_packets(count=128)
    np.savez(
        source_path,
        csi_data=np.asarray(
            [packet["csi_data"] for packet in packets],
            dtype=np.int8,
        ),
        device_ticks_us=np.asarray(
            [packet["device_ticks_us"] for packet in packets],
            dtype=np.int64,
        ),
        stream_seq_num=np.asarray(
            [packet["stream_seq_num"] for packet in packets],
            dtype=np.int64,
        ),
        num_subcarriers=np.asarray(64),
        label=np.asarray("motion"),
        chip=np.asarray("c6"),
    )
    builds = []

    def packet_factory():
        builds.append("built")
        return packets

    provenance = {
        "transform": "training_packet_augmentation_v1",
        "config": {"packet_loss": 0.05},
        "seed": 123,
    }
    first = performance_report.load_or_compute_ml_replay_rows(
        source_path,
        packets_factory=packet_factory,
        selected_subcarriers=trainer.DEFAULT_SUBCARRIERS,
        window_size=trainer.SEG_WINDOW_SIZE,
        feature_names=trainer.EXPORTED_FEATURE_NAMES[:2],
        sample_contract="stream_dense",
        stream_provenance=provenance,
    )
    second = performance_report.load_or_compute_ml_replay_rows(
        source_path,
        packets_factory=lambda: pytest.fail(
            "matching persisted provenance must not rebuild packets"
        ),
        selected_subcarriers=trainer.DEFAULT_SUBCARRIERS,
        window_size=trainer.SEG_WINDOW_SIZE,
        feature_names=trainer.EXPORTED_FEATURE_NAMES[:2],
        sample_contract="stream_dense",
        stream_provenance=provenance,
    )
    different_seed = performance_report.load_or_compute_ml_replay_rows(
        source_path,
        packets_factory=packet_factory,
        selected_subcarriers=trainer.DEFAULT_SUBCARRIERS,
        window_size=trainer.SEG_WINDOW_SIZE,
        feature_names=trainer.EXPORTED_FEATURE_NAMES[:2],
        sample_contract="stream_dense",
        stream_provenance={**provenance, "seed": 124},
    )

    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert different_seed["cache_hit"] is False
    assert builds == ["built", "built"]
    assert len(list((cache_root / "ml_replay_rows").glob("*.npz"))) == 2


def test_load_training_matrix_stream_dense_uses_stream_dense_rows(monkeypatch):
    records = [
        {
            "path": trainer.Path("clean.npz"),
            "packets": _timed_packets(count=96),
            "label_name": "empty",
            "is_motion": False,
            "chip": "C6",
            "collected_at": "",
            "day_group": "2026-07-30",
            "pair_id": "pair-a",
            "session_group": "session-a",
            "lineage_group": "lineage-a",
            "dataset_role": "train",
            "synthetic": False,
            "long_recording": False,
            "environment_group": "lab",
            "timing_quality_status": "PASS",
            "timing_quality_bucket": "clean",
            "timing_summary": {},
            "timing_weight": 1.0,
        }
    ]
    stats = {
        "chips": ["C6"],
        "labels": {"empty": 1},
        "total": 1,
        "files": ["clean.npz"],
        "excluded_labels": [],
        "excluded_chips": [],
        "excluded_environments": [],
        "excluded_missing_sync_metadata": [],
        "excluded_dataset_roles": [],
        "excluded_long_recordings": [],
        "excluded_timing_quality": [],
        "session_groups": ["session-a"],
        "lineage_groups": ["lineage-a"],
        "environment_groups": ["lab"],
        "sync_metadata_files": [],
        "timing_quality_counts": {
            "clean": 1,
            "degraded": 0,
            "poor": 0,
            "unknown": 0,
        },
    }
    replay_rows = {
        "X": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "feature_names": list(trainer.EXPORTED_FEATURE_NAMES[:2]),
        "packet_index": np.asarray([99, 100], dtype=np.int32),
        "evaluation_index": np.asarray([0, 1], dtype=np.int32),
        "reset_index": np.asarray([0, 0], dtype=np.int32),
    }
    seen_calls = []

    def fake_load_training_file_records(**_kwargs):
        return records, stats

    def fake_load_or_compute_ml_replay_rows(*_args, **_kwargs):
        seen_calls.append(_kwargs)
        return replay_rows

    monkeypatch.setattr(trainer, "_load_training_file_records", fake_load_training_file_records)
    monkeypatch.setattr(
        trainer,
        "load_or_compute_ml_replay_rows",
        fake_load_or_compute_ml_replay_rows,
    )
    monkeypatch.setattr(
        trainer.npz_cache,
        "load_ml_replay_row_artifact",
        lambda *_args, **_kwargs: None,
    )

    matrix, _ = trainer.load_training_matrix(
        feature_names=list(trainer.EXPORTED_FEATURE_NAMES[:2]),
    )

    assert [call["sample_contract"] for call in seen_calls] == ["stream_dense"]
    np.testing.assert_allclose(matrix["X"], replay_rows["X"])
    assert matrix["sample_context"]["packet_index"].tolist() == [99, 100]

    trainer.load_training_matrix(
        feature_names=list(trainer.EXPORTED_FEATURE_NAMES[:2]),
        packet_augmentation={"packet_loss": 0.05},
        augmentation_seed=123,
    )

    augmented_call = seen_calls[1]
    assert callable(augmented_call["packets_factory"])
    assert augmented_call["stream_provenance"]["seed"] == 123
    assert augmented_call["stream_provenance"]["config"] == {
        "packet_loss": 0.05,
    }
    assert augmented_call["use_cache"] is True


def test_classic_candidate_replay_reuses_time_aware_runtime_rows(monkeypatch):
    seen = []

    def fake_load_rows(path, **kwargs):
        seen.append((path, kwargs))
        return {
            "X": np.asarray([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32),
            "packet_index": np.asarray([99, 124, 199, 299], dtype=np.int32),
            "reset_index": np.asarray([0, 0, 0, 1], dtype=np.int32),
        }

    monkeypatch.setattr(
        replay_classic_candidates,
        "load_or_compute_ml_replay_rows",
        fake_load_rows,
    )

    cache = replay_classic_candidates.build_replay_cache(
        [trainer.Path("runtime.npz")],
        [trainer.EXPORTED_FEATURE_NAMES[0]],
        quiet=True,
    )

    assert seen[0][1]["sample_contract"] == "replay_tick"
    assert cache["runtime.npz"]["deoverlapped"].tolist() == [
        True,
        False,
        True,
        True,
    ]


def test_timing_audit_rows_aggregate_by_slice_and_bucket():
    rows = performance_report._summarize_timing_audit_rows(
        [
            {
                "slice": "paired_ml_reserved",
                "quality_bucket": "clean",
                "packet_rate_pps": 100.0,
                "contaminated_ratio": 0.0,
                "max_gap_ms": 10.0,
                "ml_metrics": {"recall": 97.0, "precision": 98.0, "fp_rate": 1.0, "f1": 97.5, "effective_alarms": 0},
                "classic_metrics": None,
            },
            {
                "slice": "paired_ml_reserved",
                "quality_bucket": "clean",
                "packet_rate_pps": 98.0,
                "contaminated_ratio": 0.01,
                "max_gap_ms": 20.0,
                "ml_metrics": {"recall": 93.0, "precision": 94.0, "fp_rate": 3.0, "f1": 93.5, "effective_alarms": 1},
                "classic_metrics": None,
            },
        ]
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["slice"] == "paired_ml_reserved"
    assert row["quality_bucket"] == "clean"
    assert row["count"] == 2
    assert row["packet_rate_pps"] == pytest.approx(99.0)
    assert row["contaminated_ratio"] == pytest.approx(0.005)
    assert row["max_gap_ms"] == pytest.approx(20.0)
    assert row["classic_metrics"] is None
    assert row["ml_metrics"]["count"] == 2
    assert row["ml_metrics"]["recall"] == pytest.approx(95.0)
    assert row["ml_metrics"]["min_recall"] == pytest.approx(93.0)
    assert row["ml_metrics"]["precision"] == pytest.approx(96.0)
    assert row["ml_metrics"]["fp_rate"] == pytest.approx(2.0)
    assert row["ml_metrics"]["max_fp_rate"] == pytest.approx(3.0)
    assert row["ml_metrics"]["f1"] == pytest.approx(95.5)
    assert row["ml_metrics"]["effective_alarms"] == 1
