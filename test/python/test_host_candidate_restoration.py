# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
import sys

import numpy as np
import pytest

sys.path.insert(0, 'src/python/micro_espectre')

from csi_features import ALL_FEATURES, DEFAULT_FEATURES
from tools import train_ml_model
from tools.lib.candidate_features import CANDIDATE_FEATURES, candidate_values
from tools.lib.host_feature_trackers import AmplitudeProfileTracker


RESTORED_CANDIDATES = {
    'turb_mad_over_mean',
    'l1_delta_autocorr',
    'chan_freq_coh_curve_std',
    'chan_freq_coh_cv',
    'chan_coh_gap',
    'chan_coh_subband_gap_median',
    'chan_coh_mean',
    'chan_coh_gap_low_frac',
    'chan_coh_gap_q20',
    'chan_coh_subband_median_gap',
    'turb_band_power_ratio',
    'turb_cv',
    'turb_p05_over_mean',
    'turb_max_over_mean',
    'turb_min_over_mean',
    'turb_range_over_mean',
    'turb_peak_over_mad',
    'waveform_length_over_mean',
    'turb_skewness',
    'turb_iqr_over_mean_aggr_detrended',
    'turb_iqr_over_mean_aggr_tone_detrended',
    'chan_shape_lag_ratio',
    'chan_rank_gap',
    'chan_ratio_gap',
    'chan_shape_scale_curvature',
    'corr_amp_d1',
    'chan_coh_gap_spread',
}


def test_restored_candidates_stay_out_of_the_runtime_surface() -> None:
    assert RESTORED_CANDIDATES <= set(CANDIDATE_FEATURES)
    assert set(CANDIDATE_FEATURES).isdisjoint(ALL_FEATURES)
    assert DEFAULT_FEATURES == [
        'turb_iqr_over_mean_aggr',
        'turb_autocorr',
        'turb_zcr',
        'l1_delta_lag_ratio',
        'chan_shape_spread',
        'chan_shape_coherent_innovation_energy',
        'chan_shape_excess_path',
    ]
    with pytest.raises(ValueError, match=r'no C\+\+ extractor id'):
        train_ml_model.resolve_cpp_feature_ids(['chan_shape_scale_curvature'])
    with pytest.raises(ValueError, match=r'no C\+\+ extractor id'):
        train_ml_model.resolve_cpp_feature_ids(['chan_freq_coh_curve_std'])


def test_restored_turbulence_statistics_match_their_definitions() -> None:
    series = np.asarray([1.0, 2.0, 4.0, 8.0, 7.0, 3.0])
    names = [
        'turb_cv',
        'turb_mad_over_mean',
        'turb_p05_over_mean',
        'turb_max_over_mean',
        'turb_min_over_mean',
        'turb_range_over_mean',
        'turb_peak_over_mad',
        'waveform_length_over_mean',
        'turb_skewness',
    ]
    values = candidate_values(names, turbulence_series=series)
    mean = float(np.mean(series))
    median = float(np.median(series))
    mad = float(np.median(np.abs(series - median)))
    std = float(np.std(series))

    assert values['turb_cv'] == pytest.approx(std / mean)
    assert values['turb_mad_over_mean'] == pytest.approx(mad / mean)
    assert values['turb_p05_over_mean'] == pytest.approx(
        np.percentile(series, 5) / mean
    )
    assert values['turb_max_over_mean'] == pytest.approx(np.max(series) / mean)
    assert values['turb_min_over_mean'] == pytest.approx(np.min(series) / mean)
    assert values['turb_range_over_mean'] == pytest.approx(
        (np.max(series) - np.min(series)) / mean
    )
    assert values['turb_peak_over_mad'] == pytest.approx(
        (np.max(series) - mean) / mad
    )
    assert values['waveform_length_over_mean'] == pytest.approx(
        np.mean(np.abs(np.diff(series))) / mean
    )
    assert values['turb_skewness'] == pytest.approx(
        np.mean((series - mean) ** 3) / std**3
    )


def test_retired_l1_autocorrelation_remains_host_evaluable() -> None:
    series = np.asarray(
        [0.1, 0.12, 0.11, 0.3, 0.32, 0.31, 0.1, 0.12, 0.11, 0.3]
    )

    value = candidate_values(
        ['l1_delta_autocorr'],
        l1_series=series,
    )['l1_delta_autocorr']

    centered = series - np.mean(series)
    expected = np.mean(centered[:-1] * centered[1:]) / np.var(series)
    assert value == pytest.approx(expected)


def test_linear_aggregated_drift_is_removed() -> None:
    series = np.linspace(0.1, 0.9, 25)

    value = candidate_values(
        ['turb_iqr_over_mean_aggr_detrended'],
        aggregated_turbulence_series=series,
    )['turb_iqr_over_mean_aggr_detrended']

    assert value == pytest.approx(0.0, abs=1e-12)


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


def test_streaming_extractor_evaluates_every_host_candidate() -> None:
    extractor = train_ml_model.StreamingFeatureExtractor(
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
            'csi_data': raw,
            'seq_num': packet_index,
            'device_ticks_us': packet_index * 80_000,
        }
        values = extractor.process_packet(raw, packet=packet)

    assert values is not None
    assert len(values) == len(CANDIDATE_FEATURES)
    assert np.all(np.isfinite(np.asarray(values, dtype=np.float64)))
