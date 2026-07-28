"""
ESPectre - Feature Extraction Tests

Unit tests for shared feature extraction helpers.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import pytest
import math
import numpy as np
from csi_features import (
    calc_autocorrelation,
    calc_mad,
    calc_zero_crossing_rate,
    extract_features_by_name,
    ALL_FEATURES,
    INVARIANT5_FEATURES,
    DEFAULT_FEATURES,
    FEATURE_NAMES,
)


def _stats(values, count=None):
    """Helper: compute (count, mean, std) for a list of values."""
    if count is None:
        count = len(values)
    if count == 0:
        return count, 0.0, 0.0
    mean = sum(values[:count]) / count
    var = sum((values[i] - mean) ** 2 for i in range(count)) / count
    std = math.sqrt(var) if var > 0 else 0.0
    return count, mean, std


class TestCalcAutocorrelation:
    """Test lag-1 autocorrelation calculation"""
    
    def test_empty_buffer(self):
        """Test autocorrelation of empty buffer"""
        assert calc_autocorrelation([], 0) == 0.0
    
    def test_two_values(self):
        """Test autocorrelation of two values (needs 3+)"""
        assert calc_autocorrelation([1.0, 2.0], 2) == 0.0
    
    def test_constant_values(self):
        """Test autocorrelation of constant values"""
        buffer = [5.0] * 10
        ac = calc_autocorrelation(buffer, 10)
        assert ac == 0.0  # Variance is 0
    
    def test_highly_correlated_signal(self):
        """Test that smooth signal has high autocorrelation"""
        # Slow sinusoid -> high autocorrelation
        buffer = [math.sin(i * 0.1) for i in range(50)]
        ac = calc_autocorrelation(buffer, 50)
        assert ac > 0.9  # Very high correlation
    
    def test_random_signal_low_autocorrelation(self):
        """Test that random signal has low autocorrelation"""
        np.random.seed(42)
        buffer = list(np.random.normal(0, 1, 100))
        ac = calc_autocorrelation(buffer, 100)
        # Random noise should have low autocorrelation
        assert abs(ac) < 0.3
    
    def test_output_range(self):
        """Test that autocorrelation is in [-1, 1]"""
        np.random.seed(42)
        buffer = list(np.random.normal(5, 2, 50))
        ac = calc_autocorrelation(buffer, 50)
        assert -1.0 <= ac <= 1.0


class TestCalcMAD:
    """Test Median Absolute Deviation calculation"""
    
    def test_empty_buffer(self):
        """Test MAD of empty buffer"""
        assert calc_mad([], 0) == 0.0
    
    def test_single_value(self):
        """Test MAD of single value"""
        assert calc_mad([5.0], 1) == 0.0
    
    def test_constant_values(self):
        """Test MAD of constant values"""
        buffer = [5.0] * 10
        mad = calc_mad(buffer, 10)
        assert mad == 0.0
    
    def test_symmetric_distribution(self):
        """Test MAD of symmetric values"""
        # Values: [1, 2, 3, 4, 5], median = 3
        # |1-3|=2, |2-3|=1, |3-3|=0, |4-3|=1, |5-3|=2
        # Sorted abs devs: [0, 1, 1, 2, 2], median = 1
        buffer = [1.0, 2.0, 3.0, 4.0, 5.0]
        mad = calc_mad(buffer, 5)
        assert mad == pytest.approx(1.0, rel=1e-6)
    
    def test_with_outlier(self):
        """Test MAD robustness to outliers"""
        # MAD should be robust to outliers
        buffer_no_outlier = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        buffer_with_outlier = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0]
        
        mad_clean = calc_mad(buffer_no_outlier, 10)
        mad_outlier = calc_mad(buffer_with_outlier, 10)
        
        # MAD should not change dramatically with one outlier
        # (unlike std which would increase a lot)
        assert mad_outlier < 3 * mad_clean
    
    def test_positive_result(self):
        """Test that MAD is non-negative"""
        np.random.seed(42)
        buffer = list(np.random.normal(5, 2, 50))
        mad = calc_mad(buffer, 50)
        assert mad >= 0


class TestExtractAllFeatures:
    """Test full feature extraction"""
    
    def test_returns_default_feature_count(self):
        """Test that the default feature count is returned"""
        buffer = [float(i) for i in range(50)]
        features = extract_features_by_name(
            buffer, 50, feature_names=DEFAULT_FEATURES, l1_series=buffer,
            l1_delta_lag_ratio=1.0,
        )
        assert len(features) == len(DEFAULT_FEATURES)
    
    def test_empty_buffer_returns_zeros(self):
        """Test that empty buffer returns zeros"""
        features = extract_features_by_name(
            [], 0, feature_names=DEFAULT_FEATURES, l1_delta_lag_ratio=0.0
        )
        assert features == [0.0] * len(DEFAULT_FEATURES)
    
    def test_single_value_returns_zeros(self):
        """Test that single-value buffer returns zeros"""
        features = extract_features_by_name(
            [5.0], 1, feature_names=DEFAULT_FEATURES, l1_delta_lag_ratio=0.0
        )
        assert features == [0.0] * len(DEFAULT_FEATURES)
    
    def test_feature_names_match(self):
        """Test that FEATURE_NAMES matches DEFAULT_FEATURES (production = Coherence-7)"""
        assert len(FEATURE_NAMES) == len(DEFAULT_FEATURES)
        assert FEATURE_NAMES == DEFAULT_FEATURES
        assert DEFAULT_FEATURES == INVARIANT5_FEATURES

    def test_production_set_is_the_only_feature_surface(self):
        """The production feature registry contains only exported features."""
        assert ALL_FEATURES == tuple(DEFAULT_FEATURES)
        assert len(DEFAULT_FEATURES) == 5

    def test_unknown_feature_raises(self):
        """Unknown feature names are rejected."""
        buffer = [float(i) for i in range(50)]
        with pytest.raises(ValueError, match="Unknown feature"):
            extract_features_by_name(buffer, 50, feature_names=['not_a_feature'])
    
    def test_all_features_are_float(self):
        """Test that all features are floats"""
        np.random.seed(42)
        buffer = list(np.random.normal(5, 2, 50))
        features = extract_features_by_name(
            buffer, 50, feature_names=DEFAULT_FEATURES, l1_series=buffer,
            l1_delta_lag_ratio=1.0,
        )
        for i, f in enumerate(features):
            assert isinstance(f, (int, float)), f"Feature {i} ({FEATURE_NAMES[i]}) is {type(f)}"
    
    def test_motion_vs_idle_features_differ(self):
        """Test that motion-like and idle-like buffers produce different features"""
        # Idle-like: low variance, stable signal
        idle_buffer = [5.0 + 0.01 * (i % 3) for i in range(50)]
        # Motion-like: high variance, turbulent signal
        np.random.seed(42)
        motion_buffer = list(np.random.normal(5, 3, 50))
        
        idle_features = extract_features_by_name(
            idle_buffer, 50, feature_names=DEFAULT_FEATURES, l1_series=idle_buffer,
            l1_delta_lag_ratio=1.0,
        )
        motion_features = extract_features_by_name(
            motion_buffer, 50, feature_names=DEFAULT_FEATURES, l1_series=motion_buffer,
            l1_delta_lag_ratio=2.0,
        )

        # turb_mad_over_mean is part of the production Core-6 set and rises with
        # turbulence, so motion must exceed idle. The vectors must also differ.
        mad_idx = FEATURE_NAMES.index('turb_mad_over_mean')
        assert motion_features[mad_idx] > idle_features[mad_idx]
        assert motion_features != idle_features

class TestCalcZeroCrossingRate:
    """Test the median-crossing rate helper"""

    def test_short_buffer_returns_zero(self):
        assert calc_zero_crossing_rate([1.0], 1, 0.0) == 0.0

    def test_alternating_signal_crosses_every_sample(self):
        values = [1.0, -1.0] * 25
        assert calc_zero_crossing_rate(values, len(values), 0.0) == 1.0

    def test_single_excursion_crosses_twice(self):
        values = [0.0] * 20 + [5.0] * 10 + [0.0] * 20
        rate = calc_zero_crossing_rate(values, len(values), 2.5)
        assert rate == pytest.approx(2 / 49)

    def test_shift_and_scale_invariance_with_median_center(self):
        np.random.seed(7)
        base = list(np.random.normal(0.0, 1.0, 60))
        transformed = [0.02 * v + 10.0 for v in base]
        base_median = sorted(base)[30]
        transformed_median = sorted(transformed)[30]
        assert calc_zero_crossing_rate(base, 60, base_median) == pytest.approx(
            calc_zero_crossing_rate(transformed, 60, transformed_median)
        )


class TestCandidateFeatures:
    """Test the weak-link candidate features"""

    def test_turb_zcr_separates_noise_from_coherent_excursions(self):
        np.random.seed(11)
        noise = list(np.random.normal(5.0, 1.0, 50))
        excursion = [5.0] * 20 + [9.0 + 0.01 * i for i in range(15)] + [5.0] * 15
        noise_zcr = extract_features_by_name(noise, 50, feature_names=['turb_zcr'])[0]
        excursion_zcr = extract_features_by_name(excursion, 50, feature_names=['turb_zcr'])[0]
        assert noise_zcr > excursion_zcr

    def test_turb_zcr_survives_reused_buffer_sort(self):
        """The mad feature sorts the reused buffer; zcr must see time order."""
        values = [1.0, -1.0] * 25
        combined = extract_features_by_name(
            list(values), 50,
            feature_names=['turb_zcr', 'turb_mad_over_mean'],
            reuse_turbulence_buffer=True,
        )
        alone = extract_features_by_name(
            list(values), 50, feature_names=['turb_zcr']
        )
        assert combined[0] == pytest.approx(alone[0])
        assert combined[0] == pytest.approx(1.0)

    def test_every_production_feature_is_scale_invariant(self):
        """The reason the production set is what it is.

        The per-packet CSI scaling factor is never recorded, so a feature that
        carries absolute magnitude carries the link's noise floor with it. On
        weak links that floor can exceed the motion it is meant to measure,
        which is how l1_delta and l1_delta_std took a weak pair from 0% to
        100% false positives on 2026-07-27.
        """
        np.random.seed(13)
        turb = [5.0 + 0.1 * (i % 5) for i in range(50)]
        series = [abs(v) + 0.05 for v in np.random.normal(0.1, 0.03, 40)]

        base = extract_features_by_name(
            turb, 50, feature_names=DEFAULT_FEATURES,
            l1_series=series, l1_delta_lag_ratio=1.4)
        boosted = extract_features_by_name(
            [v * 10.0 for v in turb], 50, feature_names=DEFAULT_FEATURES,
            l1_series=[v * 10.0 for v in series], l1_delta_lag_ratio=1.4)

        for name, before, after in zip(DEFAULT_FEATURES, base, boosted):
            assert after == pytest.approx(before, abs=1e-9), (
                f"{name} moved when both streams were scaled by 10")

    def test_l1_delta_autocorr_matches_direct_computation(self):
        turb = [5.0 + 0.1 * (i % 5) for i in range(50)]
        series = [0.1, 0.12, 0.11, 0.3, 0.32, 0.31, 0.1, 0.12, 0.11, 0.3]
        value = extract_features_by_name(
            turb, 50, feature_names=['l1_delta_autocorr'], l1_series=series
        )[0]
        assert value == pytest.approx(calc_autocorrelation(series, len(series)))

    def test_l1_delta_lag_ratio_uses_preprocessed_tracker_metric(self):
        turb = [5.0 + 0.1 * (i % 5) for i in range(50)]
        series = [0.1 + 0.01 * (i % 3) for i in range(40)]

        value = extract_features_by_name(
            turb,
            50,
            feature_names=['l1_delta_lag_ratio'],
            l1_series=series,
            l1_delta_lag_ratio=1.75,
        )[0]

        assert value == pytest.approx(1.75)

    def test_l1_delta_lag_ratio_requires_preprocessed_tracker_metric(self):
        turb = [5.0 + 0.1 * (i % 5) for i in range(50)]

        with pytest.raises(ValueError, match="l1_delta_lag_ratio is required"):
            extract_features_by_name(
                turb,
                50,
                feature_names=['l1_delta_lag_ratio'],
                l1_series=[0.1] * 40,
            )

    def test_l1_candidates_return_zero_without_series_samples(self):
        turb = [5.0 + 0.1 * (i % 5) for i in range(50)]
        values = extract_features_by_name(
            turb, 50,
            feature_names=['l1_delta_autocorr'],
            l1_series=[],
        )
        assert values == [0.0]
