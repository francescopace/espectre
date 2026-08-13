import numpy as np

from tools.lib import performance_report, performance_report_inputs


def test_resource_benchmark_executes_cached_binary_each_time(monkeypatch, tmp_path):
    binary = tmp_path / "benchmark"
    binary.write_text("placeholder")
    calls = []

    class Result:
        stdout = 'log line\n{"detectors": {"classic": {}, "ml": {}}}\n'

    monkeypatch.setattr(performance_report_inputs, "_resource_binary", lambda: binary)

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return Result()

    monkeypatch.setattr(performance_report_inputs.subprocess, "run", fake_run)

    first = performance_report_inputs.run_current_resource_benchmark()
    second = performance_report_inputs.run_current_resource_benchmark()

    assert first == second
    assert len(calls) == 2
    assert all(call[0] == [str(binary)] for call in calls)


def test_report_pair_filter_keeps_only_selection_and_holdout():
    all_pairs = performance_report.get_available_paired_datasets(synthetic=False)
    reserved_pairs = performance_report.get_available_paired_datasets(
        synthetic=False,
        roles=performance_report.REPORT_DATASET_ROLES,
    )

    assert reserved_pairs
    assert len(reserved_pairs) < len(all_pairs)
    assert all(
        performance_report.get_paired_dataset_role(static_path)
        in performance_report.REPORT_DATASET_ROLES
        for static_path, _motion_path, _num_sc, _chip, _dataset_id in reserved_pairs
    )


def test_report_input_collection_never_starts_training(monkeypatch):
    monkeypatch.setattr(
        performance_report_inputs,
        "run_current_resource_benchmark",
        lambda: {"detectors": {}},
    )
    monkeypatch.setattr(
        performance_report_inputs.npz_cache,
        "load_performance_report_result",
        lambda *_args, **_kwargs: {"recipe": "cached", "rows": []},
    )

    resources, augmentation = performance_report_inputs.collect_extended_report_inputs()

    assert resources == {"detectors": {}}
    assert augmentation["recipe"] == "cached"


def test_reserved_diagnostic_uses_only_reserved_roles_and_two_seed_mix(monkeypatch):
    records = [
        {"role": role, "static_path": f"{role}-static", "motion_path": f"{role}-motion"}
        for role in ("train", "selection", "holdout")
    ]
    classic_calls = []
    ml_paths = []

    monkeypatch.setattr(performance_report_inputs, "_paired_records", lambda: records)
    monkeypatch.setattr("tools.lib.csi_io.load_npz_packet_view", lambda path: [{"path": path}])
    monkeypatch.setattr(
        "tools.train_ml_model.resolve_training_augmentation",
        lambda components: (True, {}, {"components": list(components)}),
    )
    monkeypatch.setattr(
        "tools.train_ml_model.training_packet_augmentation_seeds",
        lambda _config: (101, 202),
    )
    monkeypatch.setattr(
        "tools.train_ml_model._packet_augmentation_stream_provenance",
        lambda _config, seed: {"seed": seed},
    )
    monkeypatch.setattr(
        "tools.train_ml_model._prepare_feature_packets_for_record",
        lambda record, **kwargs: [{"path": record["path"], "seed": kwargs["augmentation_seed"]}],
    )

    def fake_classic_rows(static_path, motion_path, **kwargs):
        classic_calls.append((static_path, motion_path, kwargs["replay_provenance"]["seed"]))
        return {"seed": kwargs["replay_provenance"]["seed"]}

    monkeypatch.setattr(
        "tools.lib.performance_report.load_or_compute_classic_replay_rows",
        fake_classic_rows,
    )
    monkeypatch.setattr(
        "tools.lib.performance_report.compute_classic_row_result",
        lambda _rows, **_kwargs: (0.5, {"tp": 4, "fn": 1, "fp": 1, "tn": 4}),
    )

    def fake_ml_rows(path, **kwargs):
        ml_paths.append(path)
        seed = kwargs["stream_provenance"]["seed"]
        return {
            "X": np.asarray([[seed], [seed + 1]], dtype=np.float32),
            "feature_names": ["turb_autocorr"],
            "packet_index": np.arange(2, dtype=np.int32),
            "evaluation_index": np.arange(2, dtype=np.int32),
            "reset_index": np.zeros(2, dtype=np.int32),
            "evaluation_due": np.ones(2, dtype=bool),
        }

    monkeypatch.setattr(
        "tools.lib.performance_report.load_or_compute_ml_replay_rows", fake_ml_rows
    )
    monkeypatch.setattr(
        "tools.lib.performance_report._compute_ml_row_result",
        lambda static, motion, _threshold: (
            {"tp": len(motion["X"]), "fn": 0, "fp": 0, "tn": len(static["X"])},
            {},
        ),
    )

    result = performance_report_inputs.compute_reserved_augmentation_diagnostic()

    assert result["roles"] == ["selection", "holdout"]
    assert result["seeds"] == [101, 202]
    assert result["pair_count"] == 2
    assert all("train" not in str(call) for call in classic_calls)
    assert all("train" not in str(path) for path in ml_paths)
    assert len(classic_calls) == 4
    assert len(ml_paths) == 8
