import numpy as np
import pytest

from tools.lib.host_feature_trackers import (
    CHANNEL_SHAPE_BIN_US,
    HT20_LIVE_BINS,
    ChannelShapeExcessPathTracker,
)


def payload_from_subband_amplitudes(amplitudes, gain=1):
    payload = np.zeros(128, dtype=np.int8)
    for live_index, subcarrier in enumerate(HT20_LIVE_BINS):
        subband = live_index // 7
        payload[2 * subcarrier + 1] = int(amplitudes[subband] * gain)
    return payload


def evaluate_path(amplitude_path, gains=None, duplicate=False):
    tracker = ChannelShapeExcessPathTracker()
    if gains is None:
        gains = [1] * len(amplitude_path)
    for index, (amplitudes, gain) in enumerate(zip(amplitude_path, gains)):
        timestamp_us = index * CHANNEL_SHAPE_BIN_US
        payload = payload_from_subband_amplitudes(amplitudes, gain)
        tracker.process_packet(payload, timestamp_us)
        if duplicate:
            tracker.process_packet(payload.copy(), timestamp_us + 20_000)
    return tracker.excess_path(), tracker


def curved_path(count=12, base=12.0, amplitude=2.5):
    index = np.arange(8, dtype=np.float64)
    first_mode = np.cos(np.pi * (index + 0.5) / 8.0)
    second_mode = np.cos(2.0 * np.pi * (index + 0.5) / 8.0)
    path = []
    for step in range(count):
        angle = 1.5 * np.pi * step / (count - 1)
        values = base + amplitude * (
            np.cos(angle) * first_mode + np.sin(angle) * second_mode
        )
        path.append(np.rint(values).astype(int))
    return path


def test_independent_packet_gain_cancels() -> None:
    path = curved_path()
    baseline, baseline_tracker = evaluate_path(path)
    gained, gained_tracker = evaluate_path(path, gains=[1, 2, 3] * 4)

    assert gained == pytest.approx(baseline, abs=1e-12)
    assert gained_tracker.scale_curvature() == pytest.approx(
        baseline_tracker.scale_curvature(),
        abs=1e-12,
    )
    assert gained_tracker.coherent_innovation_energy() == pytest.approx(
        baseline_tracker.coherent_innovation_energy(),
        abs=1e-12,
    )


def test_exact_stutter_duplicates_do_not_change_the_path() -> None:
    path = curved_path()
    baseline, _ = evaluate_path(path)
    duplicated, _ = evaluate_path(path, duplicate=True)

    assert duplicated == pytest.approx(baseline, abs=1e-12)


def test_missing_bins_are_not_interpolated() -> None:
    path = curved_path(count=5)
    tracker = ChannelShapeExcessPathTracker()
    timestamps = [0, 80_000, 240_000, 320_000, 400_000]
    for amplitudes, timestamp_us in zip(path, timestamps):
        tracker.process_packet(
            payload_from_subband_amplitudes(amplitudes),
            timestamp_us,
        )

    assert len(tracker._profile_path()) == len(path)


def test_curved_low_order_path_exceeds_a_slow_one_direction_path() -> None:
    base = np.arange(10, 18)
    direction = np.asarray([-1, -1, 0, 0, 0, 0, 1, 1])
    straight = [base + direction * (step // 4) for step in range(12)]

    straight_value, _ = evaluate_path(straight)
    curved_value, _ = evaluate_path(curved_path(base=70.0, amplitude=30.0))

    assert curved_value > straight_value
    assert curved_value > 0.0
