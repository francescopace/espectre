"""
Room-state multiclass experiment validation.

Exercises the offline-only 3-class prototype path in tools/10_train_ml_model.py
using the real empty/static_presence/motion datasets.
"""

import importlib.util
import sys
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from repo_paths import data_dir, tools_dir
from csi_utils import load_npz_as_packets


TRAIN_ML_MODEL_PATH = tools_dir() / "10_train_ml_model.py"


def _load_train_ml_model_module():
    """Load the training script directly despite its numeric filename."""
    spec = importlib.util.spec_from_file_location("train_ml_model_multiclass", TRAIN_ML_MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestRoomStateMulticlassExperiment:
    """Validate the offline 3-class room-state experiment."""

    def test_npz_loader_preserves_sync_metadata_when_available(self):
        train_ml_model = _load_train_ml_model_module()
        dataset_info = train_ml_model.load_dataset_info()
        sync_file = None
        for entry in dataset_info.get("files", {}).get("empty", []):
            path = data_dir() / "empty" / entry.get("filename", "")
            if not path.exists():
                continue
            packets = load_npz_as_packets(path)
            if packets and packets[0].get("stimulus_id") is not None:
                sync_file = path
                break

        assert sync_file is not None
        packets = load_npz_as_packets(sync_file)

        assert packets
        first = packets[0]
        assert first["stimulus_id"] is not None
        assert first["wifi_rx_start_ts_ns"] is not None or first["device_ticks_us"] is not None
        assert "is_reference" in first

    def test_multiclass_loader_includes_empty_static_presence_and_motion(self):
        train_ml_model = _load_train_ml_model_module()

        _, stats = train_ml_model.load_all_data(
            environment_filter="bedroom",
            excluded_chips="C5,C6,ESP32,S3",
            allowed_labels=train_ml_model.ROOM_STATE_CLASS_NAMES,
        )

        assert stats["labels"]["empty"] > 0
        assert stats["labels"]["static_presence"] > 0
        assert stats["labels"]["motion"] > 0
        assert stats["chips"] == ["C3"]

    def test_multiclass_loader_can_require_sync_metadata(self):
        train_ml_model = _load_train_ml_model_module()

        _, stats = train_ml_model.load_all_data(
            allowed_labels=train_ml_model.ROOM_STATE_CLASS_NAMES,
            require_sync_metadata=True,
        )

        assert stats["labels"]["empty"] > 0
        assert stats["labels"]["static_presence"] > 0
        assert stats["labels"]["motion"] > 0
        assert "C3" in stats["chips"]
        assert len(stats["sync_metadata_files"]) >= 3

    def test_multiclass_experiment_produces_non_random_room_state_separation(self):
        train_ml_model = _load_train_ml_model_module()

        rc, _, metrics = train_ml_model.train_multiclass_experiment(
            seed=42,
            hidden_layers=[16],
            scaler_mode="standard",
            batch_size=32,
            environment_filter="bedroom",
            excluded_chips="C5,C6,ESP32,S3",
            max_epochs=8,
            n_folds=2,
        )

        assert rc == 0
        assert metrics is not None
        assert metrics["class_names"] == train_ml_model.ROOM_STATE_CLASS_NAMES
        assert metrics["oof_confusion_matrix"].shape == (3, 3)
        assert metrics["dense_oof_macro_f1"] > 80.0
        assert metrics["dense_oof_balanced_accuracy"] > 80.0
        assert metrics["dense_oof_recall_empty"] > 95.0
        assert metrics["dense_oof_recall_static_presence"] > 60.0
        assert metrics["dense_oof_recall_motion"] > 80.0

    def test_multiclass_experiment_accepts_amplitude_phase_feature_set(self):
        train_ml_model = _load_train_ml_model_module()

        rc, _, metrics = train_ml_model.train_multiclass_experiment(
            seed=42,
            hidden_layers=[16],
            scaler_mode="standard",
            batch_size=32,
            environment_filter="bedroom",
            excluded_chips="C5,C6,ESP32,S3",
            max_epochs=8,
            n_folds=2,
            feature_set="amplitude_phase",
        )

        assert rc == 0
        assert metrics is not None
        assert metrics["oof_confusion_matrix"].shape == (3, 3)
        assert metrics["dense_oof_confusion_matrix"].shape == (3, 3)
        assert metrics["dense_oof_accuracy"] > 50.0
        assert metrics["dense_oof_support_empty"] > 0
        assert metrics["dense_oof_support_motion"] > 0

    def test_multiclass_experiment_accepts_common_offset_phase_feature_set(self):
        train_ml_model = _load_train_ml_model_module()

        rc, _, metrics = train_ml_model.train_multiclass_experiment(
            seed=42,
            hidden_layers=[16],
            scaler_mode="standard",
            batch_size=32,
            environment_filter="bedroom",
            excluded_chips="C5,C6,ESP32,S3",
            max_epochs=8,
            n_folds=2,
            feature_set="amplitude_common_offset_phase",
        )

        assert rc == 0
        assert metrics is not None
        assert metrics["oof_confusion_matrix"].shape == (3, 3)
        assert metrics["dense_oof_confusion_matrix"].shape == (3, 3)
        assert metrics["dense_oof_accuracy"] > 50.0
        assert metrics["dense_oof_support_empty"] > 0

    def test_multiclass_experiment_accepts_stimulus_phase_feature_set(self):
        train_ml_model = _load_train_ml_model_module()

        rc, _, metrics = train_ml_model.train_multiclass_experiment(
            seed=42,
            hidden_layers=[16],
            scaler_mode="standard",
            batch_size=32,
            max_epochs=8,
            n_folds=2,
            feature_set="amplitude_stimulus_phase",
            require_sync_metadata=True,
        )

        assert rc == 0
        assert metrics is not None
        assert metrics["oof_confusion_matrix"].shape == (3, 3)
