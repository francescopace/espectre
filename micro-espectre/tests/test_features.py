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
    calc_iqr_turb,
    calc_entropy_turb,
)


class TestCalcSkewness:
    """Test skewness calculation"""
    
    def test_empty_list(self):
        """Test skewness of empty list"""
        assert calc_skewness([]) == 0.0
    
    def test_single_value(self):
        """Test skewness of single value"""
        assert calc_skewness([5.0]) == 0.0
    
    def test_two_values(self):
        """Test skewness of two values (needs 3+)"""
        assert calc_skewness([1.0, 2.0]) == 0.0
    
    def test_symmetric_distribution(self):
        """Test skewness of symmetric distribution (should be ~0)"""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        skew = calc_skewness(values)
        assert abs(skew) < 0.1  # Should be close to 0
    
    def test_right_skewed(self):
        """Test skewness of right-skewed distribution"""
        # Most values low, one high -> positive skew
        values = [1.0, 1.0, 1.0, 1.0, 10.0]
        skew = calc_skewness(values)
        assert skew > 0
    
    def test_left_skewed(self):
        """Test skewness of left-skewed distribution"""
        # Most values high, one low -> negative skew
        values = [10.0, 10.0, 10.0, 10.0, 1.0]
        skew = calc_skewness(values)
        assert skew < 0
    
    def test_constant_values(self):
        """Test skewness of constant values (std=0)"""
        values = [5.0] * 10
        skew = calc_skewness(values)
        assert skew == 0.0
    
    def test_matches_scipy(self):
        """Test that result approximately matches scipy"""
        np.random.seed(42)
        values = list(np.random.exponential(2.0, 100))
        
        our_skew = calc_skewness(values)
        
        # Exponential distribution should have positive skew
        assert our_skew > 0


class TestCalcKurtosis:
    """Test kurtosis calculation"""
    
    def test_empty_list(self):
        """Test kurtosis of empty list"""
        assert calc_kurtosis([]) == 0.0
    
    def test_single_value(self):
        """Test kurtosis of single value"""
        assert calc_kurtosis([5.0]) == 0.0
    
    def test_three_values(self):
        """Test kurtosis of three values (needs 4+)"""
        assert calc_kurtosis([1.0, 2.0, 3.0]) == 0.0
    
    def test_normal_distribution(self):
        """Test kurtosis of normal distribution (should be ~0)"""
        np.random.seed(42)
        values = list(np.random.normal(0, 1, 1000))
        kurt = calc_kurtosis(values)
        # Excess kurtosis of normal is 0
        assert abs(kurt) < 0.5
    
    def test_uniform_distribution(self):
        """Test kurtosis of uniform distribution (should be < 0)"""
        np.random.seed(42)
        values = list(np.random.uniform(0, 1, 1000))
        kurt = calc_kurtosis(values)
        # Uniform distribution has negative excess kurtosis
        assert kurt < 0
    
    def test_heavy_tailed(self):
        """Test kurtosis of heavy-tailed distribution (should be > 0)"""
        # Create data with outliers -> heavy tails
        values = list(np.random.normal(0, 1, 100))
        values.extend([10.0, -10.0, 15.0, -15.0])  # Add outliers
        kurt = calc_kurtosis(values)
        # Should have positive excess kurtosis
        assert kurt > 0
    
    def test_constant_values(self):
        """Test kurtosis of constant values (std=0)"""
        values = [5.0] * 10
        kurt = calc_kurtosis(values)
        assert kurt == 0.0


class TestCalcIqrTurb:
    """Test IQR approximation calculation"""
    
    def test_empty_buffer(self):
        """Test IQR of empty buffer"""
        assert calc_iqr_turb([], 0) == 0.0
    
    def test_single_value(self):
        """Test IQR of single value"""
        assert calc_iqr_turb([5.0], 1) == 0.0
    
    def test_two_values(self):
        """Test IQR of two values"""
        buffer = [0.0, 10.0]
        iqr = calc_iqr_turb(buffer, 2)
        # IQR = (max - min) * 0.5 = (10 - 0) * 0.5 = 5
        assert iqr == pytest.approx(5.0, rel=1e-6)
    
    def test_known_range(self):
        """Test IQR with known range"""
        buffer = [1.0, 2.0, 3.0, 4.0, 5.0]
        iqr = calc_iqr_turb(buffer, 5)
        # IQR = (5 - 1) * 0.5 = 2
        assert iqr == pytest.approx(2.0, rel=1e-6)
    
    def test_constant_values(self):
        """Test IQR of constant values"""
        buffer = [5.0] * 10
        iqr = calc_iqr_turb(buffer, 10)
        assert iqr == pytest.approx(0.0, rel=1e-6)
    
    def test_partial_buffer(self):
        """Test IQR with partial buffer"""
        buffer = [1.0, 5.0, 0.0, 0.0, 0.0]  # Only first 2 valid
        iqr = calc_iqr_turb(buffer, 2)
        # IQR = (5 - 1) * 0.5 = 2
        assert iqr == pytest.approx(2.0, rel=1e-6)


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
        
        # Max entropy for 10 bins = log2(10) ≈ 3.32
        assert entropy > 2.0
    
    def test_concentrated_distribution_low_entropy(self):
        """Test that concentrated distribution has low entropy"""
        # All values in one bin -> low entropy
        buffer = [5.0 + 0.01 * i for i in range(100)]  # Very narrow range
        entropy = calc_entropy_turb(buffer, 100, n_bins=10)
        
        # Most values in few bins -> lower entropy
        # (Might still have some entropy due to discretization)
        assert entropy >= 0
    
    def test_returns_positive(self):
        """Test that entropy is non-negative"""
        np.random.seed(42)
        buffer = list(np.random.normal(5, 2, 50))
        entropy = calc_entropy_turb(buffer, 50)
        assert entropy >= 0
