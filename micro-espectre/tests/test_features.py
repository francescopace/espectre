"""
Micro-ESPectre - Feature Extraction Unit Tests

Tests for feature functions and classes in src/features.py.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import pytest
import math
import numpy as np
from features import (
    calc_skewness,
    calc_iqr,
    calc_autocorrelation,
    calc_mad,
    extract_features_by_name,
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


class TestCalcSkewness:
    """Test skewness calculation"""
    
    def test_empty_list(self):
        """Test skewness of empty list"""
        assert calc_skewness([], 0, 0.0, 0.0) == 0.0
    
    def test_single_value(self):
        """Test skewness of single value"""
        assert calc_skewness([5.0], 1, 5.0, 0.0) == 0.0
    
    def test_two_values(self):
        """Test skewness of two values (needs 3+)"""
        n, m, s = _stats([1.0, 2.0])
        assert calc_skewness([1.0, 2.0], n, m, s) == 0.0
    
    def test_symmetric_distribution(self):
        """Test skewness of symmetric distribution (should be ~0)"""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        n, m, s = _stats(values)
        skew = calc_skewness(values, n, m, s)
        assert abs(skew) < 0.1  # Should be close to 0
    
    def test_right_skewed(self):
        """Test skewness of right-skewed distribution"""
        # Most values low, one high -> positive skew
        values = [1.0, 1.0, 1.0, 1.0, 10.0]
        n, m, s = _stats(values)
        skew = calc_skewness(values, n, m, s)
        assert skew > 0
    
    def test_left_skewed(self):
        """Test skewness of left-skewed distribution"""
        # Most values high, one low -> negative skew
        values = [10.0, 10.0, 10.0, 10.0, 1.0]
        n, m, s = _stats(values)
        skew = calc_skewness(values, n, m, s)
        assert skew < 0
    
    def test_constant_values(self):
        """Test skewness of constant values (std=0)"""
        values = [5.0] * 10
        n, m, s = _stats(values)
        skew = calc_skewness(values, n, m, s)
        assert skew == 0.0
    
    def test_matches_scipy(self):
        """Test that result approximately matches scipy"""
        np.random.seed(42)
        values = list(np.random.exponential(2.0, 100))
        n, m, s = _stats(values)
        
        our_skew = calc_skewness(values, n, m, s)
        
        # Exponential distribution should have positive skew
        assert our_skew > 0
    
    def test_with_count_parameter(self):
        """Test that count parameter limits values used"""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]  # Last value is outlier
        n_all, m_all, s_all = _stats(values)
        n_part, m_part, s_part = _stats(values, count=5)
        skew_all = calc_skewness(values, n_all, m_all, s_all)
        skew_partial = calc_skewness(values, n_part, m_part, s_part)
        # Skewness without outlier should be different
        assert abs(skew_all) != abs(skew_partial)


class TestCalcIQR:
    """Test interquartile range calculation."""

    def test_empty_buffer(self):
        """Test IQR of empty buffer."""
        assert calc_iqr([], 0) == 0.0

    def test_single_value(self):
        """Test IQR of single value."""
        assert calc_iqr([5.0], 1) == 0.0

    def test_constant_values(self):
        """Test IQR of constant values."""
        buffer = [5.0] * 10
        assert calc_iqr(buffer, 10) == 0.0

    def test_monotonic_values(self):
        """Test IQR on a simple increasing sequence."""
        buffer = [float(i) for i in range(8)]
        iqr = calc_iqr(buffer, 8)
        assert iqr == pytest.approx(3.5, rel=1e-6)

    def test_outlier_robustness(self):
        """Test that a single outlier has limited impact on IQR."""
        buffer = [1.0] * 9 + [100.0]
        iqr = calc_iqr(buffer, 10)
        assert iqr == 0.0

    def test_positive_for_spread_distribution(self):
        """Test that wider distributions yield positive IQR."""
        np.random.seed(42)
        buffer = list(np.random.normal(5, 2, 100))
        iqr = calc_iqr(buffer, 100)
        assert iqr > 0.0


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
        features = extract_features_by_name(buffer, 50, feature_names=DEFAULT_FEATURES)
        assert len(features) == len(DEFAULT_FEATURES)
    
    def test_empty_buffer_returns_zeros(self):
        """Test that empty buffer returns zeros"""
        features = extract_features_by_name([], 0, feature_names=DEFAULT_FEATURES)
        assert features == [0.0] * len(DEFAULT_FEATURES)
    
    def test_single_value_returns_zeros(self):
        """Test that single-value buffer returns zeros"""
        features = extract_features_by_name([5.0], 1, feature_names=DEFAULT_FEATURES)
        assert features == [0.0] * len(DEFAULT_FEATURES)
    
    def test_feature_names_match(self):
        """Test that FEATURE_NAMES matches DEFAULT_FEATURES"""
        assert len(FEATURE_NAMES) == len(DEFAULT_FEATURES)

    def test_unknown_feature_raises(self):
        """Removed legacy features are no longer accepted."""
        buffer = [float(i) for i in range(50)]
        with pytest.raises(ValueError, match="Unknown feature"):
            extract_features_by_name(buffer, 50, feature_names=['turb_kurtosis'])
    
    def test_amplitudes_parameter_ignored(self):
        """Test that amplitudes parameter does not affect output"""
        buffer = [float(i) for i in range(50)]
        features_no_amp = extract_features_by_name(buffer, 50, feature_names=DEFAULT_FEATURES)
        features_with_amp = extract_features_by_name(
            buffer, 50, amplitudes=[1.0] * 12, feature_names=DEFAULT_FEATURES
        )
        assert features_no_amp == features_with_amp
    
    def test_all_features_are_float(self):
        """Test that all features are floats"""
        np.random.seed(42)
        buffer = list(np.random.normal(5, 2, 50))
        features = extract_features_by_name(buffer, 50, feature_names=DEFAULT_FEATURES)
        for i, f in enumerate(features):
            assert isinstance(f, (int, float)), f"Feature {i} ({FEATURE_NAMES[i]}) is {type(f)}"
    
    def test_motion_vs_idle_features_differ(self):
        """Test that motion-like and idle-like buffers produce different features"""
        # Idle-like: low variance, stable signal
        idle_buffer = [5.0 + 0.01 * (i % 3) for i in range(50)]
        # Motion-like: high variance, turbulent signal
        np.random.seed(42)
        motion_buffer = list(np.random.normal(5, 3, 50))
        
        idle_features = extract_features_by_name(idle_buffer, 50, feature_names=DEFAULT_FEATURES)
        motion_features = extract_features_by_name(motion_buffer, 50, feature_names=DEFAULT_FEATURES)
        
        # Std should be higher for motion
        assert motion_features[1] > idle_features[1]
        # MAD should be higher for motion
        mad_idx = FEATURE_NAMES.index('turb_mad')
        assert motion_features[mad_idx] > idle_features[mad_idx]
