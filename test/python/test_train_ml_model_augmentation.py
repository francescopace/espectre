import argparse

import numpy as np
import pytest

import tools.train_ml_model as trainer


def _synthetic_packets(count=200, *, source="sample.npz", interval_us=10_000):
    packets = []
    base_row = np.tile(np.asarray([20, -12], dtype=np.int8), 64)
    for index in range(count):
        packets.append(
            {
                "csi_data": base_row.copy(),
                "source_file": source,
                "device_ticks_us": index * interval_us,
            }
        )
    return packets


def test_parse_augmentation_components_normalizes_order_and_deduplicates():
    assert trainer.parse_augmentation_components("burst-loss,base,drift,base") == (
        "base",
        "drift",
        "burst-loss",
    )
    assert trainer.parse_augmentation_components(True) == ("base",)
    assert trainer.parse_augmentation_components(None) == tuple()


def test_parse_augmentation_components_rejects_unknown_names():
    with pytest.raises(argparse.ArgumentTypeError):
        trainer.parse_augmentation_components("base,unknown")


def test_resolve_training_augmentation_merges_selected_components():
    components, feature_augmentation, packet_augmentation = trainer.resolve_training_augmentation(
        "base,drift,burst-loss"
    )

    assert components == ("base", "drift", "burst-loss")
    assert feature_augmentation["jitter_sigma"] == pytest.approx(0.10)
    assert packet_augmentation["noise_sigma"] == pytest.approx(0.01)
    assert packet_augmentation["packet_loss"] == pytest.approx(0.05)
    assert packet_augmentation["stutter_probability"] == pytest.approx(0.08)
    assert packet_augmentation["drift_sigma"] > 0.0
    assert packet_augmentation["burst_loss_starts_per_minute"] > 0.0


def test_packet_rate_estimate_uses_effective_throughput_for_bursty_capture():
    packets = _synthetic_packets(count=101)
    timestamp_us = 0
    for index, packet in enumerate(packets):
        if index:
            timestamp_us += 91_000 if index % 10 == 0 else 1_000
        packet["device_ticks_us"] = timestamp_us

    assert trainer._estimate_packet_rate_pps(packets) == pytest.approx(100.0)


def test_drift_augmentation_is_deterministic_and_count_preserving():
    packets = _synthetic_packets()
    config = {
        "drift_sigma": 0.25,
        "drift_episode_count": 1,
        "drift_duration_seconds": (1.0, 1.0),
    }

    first = trainer.augment_csi_packets(packets, config, seed=7)
    second = trainer.augment_csi_packets(packets, config, seed=7)

    assert len(first) == len(packets)
    assert len(second) == len(packets)
    assert any(
        np.any(first_packet["csi_data"] != original_packet["csi_data"])
        for first_packet, original_packet in zip(first, packets)
    )
    for first_packet, second_packet in zip(first, second):
        np.testing.assert_array_equal(first_packet["csi_data"], second_packet["csi_data"])


def test_burst_loss_augmentation_is_deterministic_and_drops_packets():
    packets = _synthetic_packets()
    config = {
        "burst_loss_starts_per_minute": 120.0,
        "burst_length_packets": (2, 2),
    }

    first = trainer.augment_csi_packets(packets, config, seed=11)
    second = trainer.augment_csi_packets(packets, config, seed=11)

    assert 0 < len(first) < len(packets)
    assert len(second) == len(first)
    for first_packet, second_packet in zip(first, second):
        np.testing.assert_array_equal(first_packet["csi_data"], second_packet["csi_data"])
