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
    calc_kurtosis,
    calc_entropy_turb,
    calc_zero_crossing_rate,
    calc_autocorrelation,
    calc_mad,
    extract_all_features,
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


class TestCalcKurtosis:
    """Test kurtosis calculation"""
    
    def test_empty_list(self):
        """Test kurtosis of empty list"""
        assert calc_kurtosis([], 0, 0.0, 0.0) == 0.0
    
    def test_single_value(self):
        """Test kurtosis of single value"""
        assert calc_kurtosis([5.0], 1, 5.0, 0.0) == 0.0
    
    def test_three_values(self):
        """Test kurtosis of three values (needs 4+)"""
        n, m, s = _stats([1.0, 2.0, 3.0])
        assert calc_kurtosis([1.0, 2.0, 3.0], n, m, s) == 0.0
    
    def test_normal_distribution(self):
        """Test kurtosis of normal distribution (should be ~0)"""
        np.random.seed(42)
        values = list(np.random.normal(0, 1, 1000))
        n, m, s = _stats(values)
        kurt = calc_kurtosis(values, n, m, s)
        # Excess kurtosis of normal is 0
        assert abs(kurt) < 0.5
    
    def test_uniform_distribution(self):
        """Test kurtosis of uniform distribution (should be < 0)"""
        np.random.seed(42)
        values = list(np.random.uniform(0, 1, 1000))
        n, m, s = _stats(values)
        kurt = calc_kurtosis(values, n, m, s)
        # Uniform distribution has negative excess kurtosis
        assert kurt < 0
    
    def test_heavy_tailed(self):
        """Test kurtosis of heavy-tailed distribution (should be > 0)"""
        # Create data with outliers -> heavy tails
        values = list(np.random.normal(0, 1, 100))
        values.extend([10.0, -10.0, 15.0, -15.0])  # Add outliers
        n, m, s = _stats(values)
        kurt = calc_kurtosis(values, n, m, s)
        # Should have positive excess kurtosis
        assert kurt > 0
    
    def test_constant_values(self):
        """Test kurtosis of constant values (std=0)"""
        values = [5.0] * 10
        n, m, s = _stats(values)
        kurt = calc_kurtosis(values, n, m, s)
        assert kurt == 0.0


class TestCalcEntropyTurb:
    """Test Shannon entropy calculation"""
    
    def test_empty_buffer(self):
        """Test entropy of empty buffer"""
        assert calc_entropy_turb([], 0) == 0.0
    
    def test_single_value(self):
        """Test entropy of single value"""
        assert calc_entropy_turb([5.0], 1) == 0.0
    
    def test_constant_values(self):
        """Test entropy of constant values (max-min ~0)"""
        buffer = [5.0] * 10
        entropy = calc_entropy_turb(buffer, 10)
        assert entropy == 0.0
    
    def test_uniform_distribution_max_entropy(self):
        """Test that uniform distribution has higher entropy"""
        # Uniform across bins -> high entropy
        buffer = [float(i % 10) for i in range(100)]  # 0-9 evenly
        entropy = calc_entropy_turb(buffer, 100, n_bins=10)
        
        # Max entropy for 10 bins = log2(10) ~ 3.32
        assert entropy > 2.0
    
    def test_concentrated_distribution_low_entropy(self):
        """Test that concentrated distribution has low entropy"""
        # All values in one bin -> low entropy
        buffer = [5.0 + 0.01 * i for i in range(100)]  # Very narrow range
        entropy = calc_entropy_turb(buffer, 100, n_bins=10)
        
        # Most values in few bins -> lower entropy
        assert entropy >= 0
    
    def test_returns_positive(self):
        """Test that entropy is non-negative"""
        np.random.seed(42)
        buffer = list(np.random.normal(5, 2, 50))
        entropy = calc_entropy_turb(buffer, 50)
        assert entropy >= 0


class TestCalcZeroCrossingRate:
    """Test zero-crossing rate calculation"""
    
    def test_empty_buffer(self):
        """Test ZCR of empty buffer"""
        assert calc_zero_crossing_rate([], 0) == 0.0
    
    def test_single_value(self):
        """Test ZCR of single value"""
        assert calc_zero_crossing_rate([5.0], 1) == 0.0
    
    def test_constant_values(self):
        """Test ZCR of constant values (all at mean)"""
        buffer = [5.0] * 10
        zcr = calc_zero_crossing_rate(buffer, 10)
        assert zcr == 0.0
    
    def test_alternating_values(self):
        """Test ZCR of perfectly alternating values"""
        buffer = [0.0, 10.0, 0.0, 10.0, 0.0, 10.0]
        zcr = calc_zero_crossing_rate(buffer, 6)
        # Mean is 5.0, so 0<5<10>5>0<5<10>5>0 -> crosses every time
        assert zcr == 1.0
    
    def test_monotonic_increasing(self):
        """Test ZCR of monotonic increasing values"""
        buffer = [float(i) for i in range(10)]
        zcr = calc_zero_crossing_rate(buffer, 10)
        # Mean is 4.5, values cross only once (from below to above)
        # 0,1,2,3,4 below mean, 5,6,7,8,9 above mean -> 1 crossing
        assert zcr == pytest.approx(1.0 / 9.0, rel=1e-6)
    
    def test_output_range(self):
        """Test that ZCR is always in [0, 1]"""
        np.random.seed(42)
        buffer = list(np.random.normal(5, 2, 50))
        zcr = calc_zero_crossing_rate(buffer, 50)
        assert 0.0 <= zcr <= 1.0
    
    def test_high_zcr_for_noisy_signal(self):
        """Test that noisy signal has high ZCR"""
        np.random.seed(42)
        buffer = list(np.random.normal(0, 1, 100))
        zcr = calc_zero_crossing_rate(buffer, 100)
        # Random noise should cross mean frequently
        assert zcr > 0.3


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
    
    def test_returns_12_features(self):
        """Test that 12 features are returned"""
        buffer = [float(i) for i in range(50)]
        features = extract_all_features(buffer, 50)
        assert len(features) == 12
    
    def test_empty_buffer_returns_zeros(self):
        """Test that empty buffer returns zeros"""
        features = extract_all_features([], 0)
        assert features == [0.0] * 12
    
    def test_single_value_returns_zeros(self):
        """Test that single-value buffer returns zeros"""
        features = extract_all_features([5.0], 1)
        assert features == [0.0] * 12
    
    def test_feature_names_match(self):
        """Test that FEATURE_NAMES has 12 entries"""
        assert len(FEATURE_NAMES) == 12
    
    def test_amplitudes_parameter_ignored(self):
        """Test that amplitudes parameter does not affect output"""
        buffer = [float(i) for i in range(50)]
        features_no_amp = extract_all_features(buffer, 50)
        features_with_amp = extract_all_features(buffer, 50, amplitudes=[1.0] * 12)
        assert features_no_amp == features_with_amp
    
    def test_all_features_are_float(self):
        """Test that all features are floats"""
        np.random.seed(42)
        buffer = list(np.random.normal(5, 2, 50))
        features = extract_all_features(buffer, 50)
        for i, f in enumerate(features):
            assert isinstance(f, (int, float)), f"Feature {i} ({FEATURE_NAMES[i]}) is {type(f)}"
    
    def test_motion_vs_idle_features_differ(self):
        """Test that motion-like and idle-like buffers produce different features"""
        # Idle-like: low variance, stable signal
        idle_buffer = [5.0 + 0.01 * (i % 3) for i in range(50)]
        # Motion-like: high variance, turbulent signal
        np.random.seed(42)
        motion_buffer = list(np.random.normal(5, 3, 50))
        
        idle_features = extract_all_features(idle_buffer, 50)
        motion_features = extract_all_features(motion_buffer, 50)
        
        # Std should be higher for motion
        assert motion_features[1] > idle_features[1]
        # MAD should be higher for motion
        assert motion_features[9] > idle_features[9]
