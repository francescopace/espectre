# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Contracts for the host-only feature candidates used by ML training."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, "src/python/micro_espectre")

from csi_features import ALL_FEATURES
from tools.lib.ml_training import (
    export,
    feature_cache,
)
from tools.lib.candidate_features import CANDIDATE_FEATURES, candidate_values
from tools.lib.host_feature_trackers import (
    AmplitudeProfileTracker,
    FineChannelShapeTrajectoryTracker,
    HT20_LIVE_BINS,
)


def test_host_candidates_stay_out_of_the_runtime_surface() -> None:
    assert set(CANDIDATE_FEATURES).isdisjoint(ALL_FEATURES)
    for feature_name in (
        "chan_shape_coherent_innovation_contrast",
        "chan_shape_coherent_displacement_ratio",
        "chan_shape_intraband_innovation_energy",
        "chan_shape_subband_rank_gap",
        "chan_shape_scale_curvature",
        "chan_freq_coh_curve_std",
    ):
        with pytest.raises(ValueError, match=r"no C\+\+ extractor id"):
            export.resolve_cpp_feature_ids([feature_name])
    assert export.resolve_cpp_feature_ids(
        ["chan_shape_spread_subband"]
    ) == [48]
    assert export.resolve_cpp_feature_ids(
        ["chan_shape_subband_kendall_lag_excess"]
    ) == [49]


def test_subband_rank_tracking_is_enabled_only_for_the_candidate() -> None:
    production = feature_cache.StreamingFeatureExtractor(
        ["chan_shape_spread_subband"]
    )
    candidate = feature_cache.StreamingFeatureExtractor(
        ["chan_shape_subband_rank_gap"]
    )

    assert production.shape_trajectory_tracker is None
    assert production.production_extractor.shape_trajectory_tracker is not None
    assert candidate.shape_trajectory_tracker.track_subband_rank_gap


def test_promoted_subband_kendall_tracking_uses_production_extractor() -> None:
    production = feature_cache.StreamingFeatureExtractor(
        ["chan_shape_spread_subband"]
    )
    candidate = feature_cache.StreamingFeatureExtractor(
        ["chan_shape_subband_kendall_lag_excess"]
    )

    assert production.shape_trajectory_tracker is None
    assert candidate.shape_trajectory_tracker is None
    assert production.production_extractor.shape_trajectory_tracker is not None
    assert candidate.production_extractor.shape_trajectory_tracker is not None


def test_turbulence_candidates_match_their_definitions() -> None:
    series = np.asarray([1.0, 2.0, 4.0, 8.0, 7.0, 3.0])
    names = [
        "turb_cv",
        "turb_mad_over_mean",
        "turb_p05_over_mean",
        "turb_max_over_mean",
        "turb_min_over_mean",
        "turb_range_over_mean",
        "turb_peak_over_mad",
        "waveform_length_over_mean",
        "turb_skewness",
    ]
    values = candidate_values(names, turbulence_series=series)
    mean = float(np.mean(series))
    median = float(np.median(series))
    mad = float(np.median(np.abs(series - median)))
    std = float(np.std(series))

    assert values["turb_cv"] == pytest.approx(std / mean)
    assert values["turb_mad_over_mean"] == pytest.approx(mad / mean)
    assert values["turb_p05_over_mean"] == pytest.approx(
        np.percentile(series, 5) / mean
    )
    assert values["turb_max_over_mean"] == pytest.approx(np.max(series) / mean)
    assert values["turb_min_over_mean"] == pytest.approx(np.min(series) / mean)
    assert values["turb_range_over_mean"] == pytest.approx(
        (np.max(series) - np.min(series)) / mean
    )
    assert values["turb_peak_over_mad"] == pytest.approx(
        (np.max(series) - mean) / mad
    )
    assert values["waveform_length_over_mean"] == pytest.approx(
        np.mean(np.abs(np.diff(series))) / mean
    )
    assert values["turb_skewness"] == pytest.approx(
        np.mean((series - mean) ** 3) / std**3
    )


def test_amplitude_profile_candidates_ignore_packet_gain() -> None:
    baseline = AmplitudeProfileTracker(window_size=8)
    gained = AmplitudeProfileTracker(window_size=8)
    profiles = [
        np.asarray([2.0 + index + tone for tone in range(12)])
        for index in range(8)
    ]
    for index, profile in enumerate(profiles):
        baseline.process_amplitudes(profile, profile)
        gain = (1.0, 2.0, 0.5, 3.0)[index % 4]
        gained.process_amplitudes(profile * gain, profile * gain)

    assert gained.adjacent_amplitude_correlation() == pytest.approx(
        baseline.adjacent_amplitude_correlation(),
        abs=1e-12,
    )
    assert gained.tone_detrended_aggregated_iqr() == pytest.approx(
        baseline.tone_detrended_aggregated_iqr(),
        abs=1e-12,
    )


def test_coherent_displacement_requires_persistent_low_mode_changes() -> None:
    name = "chan_shape_coherent_displacement_ratio"

    def evaluate(positions, mode=1, indices=None):
        modes = np.zeros((len(positions), 8))
        modes[:, mode] = positions
        indices = range(len(positions)) if indices is None else indices
        path = list(zip(indices, modes, strict=True))
        tracker = SimpleNamespace(_binned_path=lambda: path)
        return candidate_values([name], shape_trajectory_tracker=tracker)[name]

    assert evaluate([0, 1, 2, 3]) == pytest.approx(1.0)
    assert evaluate([0, 1, 0, 1]) == 0.0
    assert evaluate([0, 1, 2, 3], mode=5) == 0.0
    assert evaluate([1, 1, 1, 1]) == 0.0
    assert evaluate([0, 1, 2, 3], indices=[0, 1, 3, 4]) == 0.0


def test_fine_trajectory_ignores_packet_gain_and_resets() -> None:
    name = "chan_shape_intraband_innovation_energy"
    baseline = FineChannelShapeTrajectoryTracker()
    gained = FineChannelShapeTrajectoryTracker()
    observed = []
    for index in range(24):
        raw = np.zeros(128, dtype=np.int8)
        for tone, subcarrier in enumerate(HT20_LIVE_BINS):
            raw[2 * subcarrier + 1] = round(
                20 + 6 * np.sin(index * .5) * np.cos((tone % 7 + .5) * np.pi / 7)
            )
        baseline.process_packet(raw, index * 80_000)
        gained.process_packet(raw * (2 if index % 2 else 1), index * 80_000)
        left = candidate_values([name], fine_shape_trajectory_tracker=baseline)[name]
        right = candidate_values([name], fine_shape_trajectory_tracker=gained)[name]
        assert right == pytest.approx(left, abs=1e-12)
        observed.append(left)
    assert max(observed) > 0
    baseline.reset()
    assert candidate_values([name], fine_shape_trajectory_tracker=baseline)[name] == 0


def test_streaming_extractor_evaluates_every_host_candidate() -> None:
    extractor = feature_cache.StreamingFeatureExtractor(
        CANDIDATE_FEATURES,
        window_packets=12,
        packet_interval_us=80_000,
    )
    values = None
    for packet_index in range(16):
        raw = np.zeros(128, dtype=np.int8)
        for subcarrier in range(64):
            raw[2 * subcarrier] = (
                (3 * subcarrier + packet_index) % 41
            ) - 20
            raw[2 * subcarrier + 1] = (
                (5 * subcarrier + 2 * packet_index) % 47
            ) - 23
        packet = {
            "csi_data": raw,
            "seq_num": packet_index,
            "device_ticks_us": packet_index * 80_000,
        }
        values = extractor.process_packet(raw, packet=packet)

    assert values is not None
    assert len(values) == len(CANDIDATE_FEATURES)
    assert np.all(np.isfinite(np.asarray(values, dtype=np.float64)))
