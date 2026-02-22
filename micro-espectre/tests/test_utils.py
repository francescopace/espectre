"""
Micro-ESPectre - Utility Functions Tests

Tests for utility functions in src/utils.py.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import pytest
import math
import numpy as np
from utils import (
    to_signed_int8,
    calculate_median,
    insertion_sort,
    calculate_percentile,
    calculate_variance,
    calculate_std,
    calculate_magnitude,
    calculate_spatial_turbulence,
    calculate_moving_variance,
    extract_amplitude,
    extract_amplitudes,
    extract_all_magnitudes,
    extract_phase,
    extract_phases,
    extract_amplitudes_and_phases,
)


# ============================================================================
# Signed Integer Conversion Tests
# ============================================================================

class TestToSignedInt8:
    """Test unsigned to signed int8 conversion"""
    
    def test_positive_values(self):
        """Test values < 128 remain positive"""
        assert to_signed_int8(0) == 0
        assert to_signed_int8(1) == 1
        assert to_signed_int8(127) == 127
    
    def test_negative_values(self):
        """Test values >= 128 become negative"""
        assert to_signed_int8(128) == -128
        assert to_signed_int8(255) == -1
        assert to_signed_int8(200) == 200 - 256  # -56


# ============================================================================
# Median Calculation Tests
# ============================================================================

class TestCalculateMedian:
    """Test median calculation"""
    
    def test_empty_list(self):
        """Test median of empty list"""
        assert calculate_median([]) == 0
    
    def test_single_value(self):
        """Test median of single value"""
        assert calculate_median([5]) == 5
    
    def test_odd_count(self):
        """Test median of odd-length list"""
        values = [3, 1, 2]
        assert calculate_median(values) == 2
    
    def test_even_count(self):
        """Test median of even-length list (integer average)"""
        values = [1, 2, 3, 4]
        assert calculate_median(values) == 2  # (2 + 3) // 2 = 2
    
    def test_with_floats(self):
        """Test median with float values"""
        values = [1.0, 5.0, 3.0, 2.0, 4.0]
        assert calculate_median(values) == 3.0


# ============================================================================
# Insertion Sort Tests
# ============================================================================

class TestInsertionSort:
    """Test insertion sort implementation"""
    
    def test_empty_array(self):
        """Test sorting empty array"""
        arr = []
        insertion_sort(arr, 0)
        assert arr == []
    
    def test_single_element(self):
        """Test sorting single element"""
        arr = [5]
        insertion_sort(arr, 1)
        assert arr == [5]
    
    def test_sorted_array(self):
        """Test already sorted array"""
        arr = [1, 2, 3, 4, 5]
        insertion_sort(arr, 5)
        assert arr == [1, 2, 3, 4, 5]
    
    def test_reverse_sorted(self):
        """Test reverse sorted array"""
        arr = [5, 4, 3, 2, 1]
        insertion_sort(arr, 5)
        assert arr == [1, 2, 3, 4, 5]
    
    def test_random_array(self):
        """Test random array"""
        arr = [3, 1, 4, 1, 5, 9, 2, 6]
        insertion_sort(arr, 8)
        assert arr == [1, 1, 2, 3, 4, 5, 6, 9]
    
    def test_partial_sort(self):
        """Test sorting only first n elements"""
        arr = [5, 3, 1, 9, 7]
        insertion_sort(arr, 3)  # Only sort first 3
        assert arr[:3] == [1, 3, 5]
        assert arr[3:] == [9, 7]  # Unchanged


# ============================================================================
# Percentile Calculation Tests
# ============================================================================

class TestCalculatePercentile:
    """Test percentile calculation"""
    
    def test_empty_list(self):
        """Test percentile of empty list"""
        assert calculate_percentile([], 50) == 0.0
    
    def test_single_value(self):
        """Test percentile of single value"""
        assert calculate_percentile([5.0], 50) == 5.0
    
    def test_p0(self):
        """Test 0th percentile (minimum)"""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert calculate_percentile(values, 0) == 1.0
    
    def test_p100(self):
        """Test 100th percentile (maximum)"""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert calculate_percentile(values, 100) == 5.0
    
    def test_p50(self):
        """Test 50th percentile (median)"""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = calculate_percentile(values, 50)
        assert result == pytest.approx(3.0, rel=1e-6)
    
    def test_interpolation(self):
        """Test linear interpolation between values"""
        values = [0.0, 10.0]
        assert calculate_percentile(values, 50) == pytest.approx(5.0, rel=1e-6)


# ============================================================================
# Variance and Std Tests
# ============================================================================

class TestCalculateVariance:
    """Test variance calculation"""
    
    def test_empty_list(self):
        """Test variance of empty list"""
        assert calculate_variance([]) == 0.0
    
    def test_single_value(self):
        """Test variance of single value (should be 0)"""
        assert calculate_variance([5.0]) == 0.0
    
    def test_constant_values(self):
        """Test variance of constant values"""
        values = [5.0] * 10
        assert calculate_variance(values) == 0.0
    
    def test_known_variance(self):
        """Test with known variance"""
        # Values: [1, 3, 5] -> mean = 3
        # Var = ((1-3)^2 + (3-3)^2 + (5-3)^2) / 3 = (4 + 0 + 4) / 3 = 8/3
        values = [1.0, 3.0, 5.0]
        result = calculate_variance(values)
        assert result == pytest.approx(8.0 / 3.0, rel=1e-6)


class TestCalculateStd:
    """Test standard deviation calculation"""
    
    def test_empty_list(self):
        """Test std of empty list"""
        assert calculate_std([]) == 0.0
    
    def test_constant_values(self):
        """Test std of constant values"""
        values = [5.0] * 10
        assert calculate_std(values) == 0.0
    
    def test_known_std(self):
        """Test with known std"""
        values = [1.0, 3.0, 5.0]
        result = calculate_std(values)
        expected = math.sqrt(8.0 / 3.0)
        assert result == pytest.approx(expected, rel=1e-6)


# ============================================================================
# Magnitude Calculation Tests
# ============================================================================

class TestCalculateMagnitude:
    """Test I/Q magnitude calculation"""
    
    def test_zero_components(self):
        """Test magnitude of zero"""
        assert calculate_magnitude(0, 0) == 0.0
    
    def test_real_only(self):
        """Test magnitude with only real component"""
        assert calculate_magnitude(3, 0) == 3.0
    
    def test_imag_only(self):
        """Test magnitude with only imaginary component"""
        assert calculate_magnitude(0, 4) == 4.0
    
    def test_known_magnitude(self):
        """Test 3-4-5 triangle"""
        assert calculate_magnitude(3, 4) == 5.0
    
    def test_negative_components(self):
        """Test with negative components"""
        assert calculate_magnitude(-3, -4) == 5.0


# ============================================================================
# Spatial Turbulence Tests
# ============================================================================

class TestCalculateSpatialTurbulence:
    """Test spatial turbulence calculation"""
    
    def test_empty_band(self):
        """Test with empty band"""
        magnitudes = [10.0] * 64
        assert calculate_spatial_turbulence(magnitudes, []) == 0.0
    
    def test_constant_magnitudes_cv(self):
        """Test with constant magnitudes (CV mode)"""
        magnitudes = [10.0] * 64
        band = [0, 1, 2, 3, 4]
        result = calculate_spatial_turbulence(magnitudes, band, use_cv_normalization=True)
        assert result == 0.0  # std = 0, so std/mean = 0
    
    def test_constant_magnitudes_raw(self):
        """Test with constant magnitudes (raw std mode)"""
        magnitudes = [10.0] * 64
        band = [0, 1, 2, 3, 4]
        result = calculate_spatial_turbulence(magnitudes, band, use_cv_normalization=False)
        assert result == 0.0  # std = 0
    
    def test_varied_magnitudes(self):
        """Test with varied magnitudes"""
        magnitudes = [float(i) for i in range(64)]
        band = [10, 20, 30, 40, 50]
        
        result = calculate_spatial_turbulence(magnitudes, band, use_cv_normalization=True)
        assert result > 0  # std/mean > 0 for non-constant values


# ============================================================================
# Moving Variance Tests
# ============================================================================

class TestCalculateMovingVariance:
    """Test moving variance calculation"""
    
    def test_short_series(self):
        """Test with series shorter than window"""
        values = [1.0, 2.0, 3.0]
        result = calculate_moving_variance(values, window_size=10)
        assert result == []
    
    def test_output_length(self):
        """Test output length"""
        values = [float(i) for i in range(100)]
        result = calculate_moving_variance(values, window_size=20)
        assert len(result) == 80  # 100 - 20
    
    def test_constant_series_zero_variance(self):
        """Test that constant series gives zero variance"""
        values = [5.0] * 50
        result = calculate_moving_variance(values, window_size=10)
        
        assert all(v == 0.0 for v in result)
    
    def test_variance_values_positive(self):
        """Test that variances are non-negative"""
        np.random.seed(42)
        values = list(np.random.normal(5, 2, 100))
        result = calculate_moving_variance(values, window_size=20)
        
        assert all(v >= 0 for v in result)


# ============================================================================
# CSI I/Q Parsing Tests
# ============================================================================

class TestExtractAmplitude:
    """Test single subcarrier amplitude extraction"""
    
    def test_invalid_index(self):
        """Test with out-of-bounds index"""
        csi_data = [0] * 10
        assert extract_amplitude(csi_data, 100) == 0.0
    
    def test_zero_iq(self):
        """Test with zero I/Q values"""
        # Each subcarrier is 2 bytes: [Q, I]
        csi_data = [0] * 128
        assert extract_amplitude(csi_data, 0) == 0.0
    
    def test_known_amplitude(self):
        """Test with known I/Q values (3, 4 -> magnitude 5)"""
        csi_data = [0] * 128
        # Subcarrier 5: bytes 10, 11
        # Format: [Q, I] = [4, 3]
        csi_data[10] = 4  # Q
        csi_data[11] = 3  # I
        
        result = extract_amplitude(csi_data, 5)
        assert result == 5.0


class TestExtractAmplitudes:
    """Test multiple subcarrier amplitude extraction"""
    
    def test_default_all_subcarriers(self):
        """Test extracting all available subcarriers"""
        csi_data = [1] * 128  # 64 subcarriers
        result = extract_amplitudes(csi_data)
        
        assert len(result) == 64
    
    def test_specific_subcarriers(self):
        """Test extracting specific subcarriers"""
        csi_data = [0] * 128
        result = extract_amplitudes(csi_data, subcarriers=[0, 10, 20])
        
        assert len(result) == 3


class TestExtractAllMagnitudes:
    """Test extracting all magnitudes"""
    
    def test_returns_indexed_list(self):
        """Test that result is indexed by subcarrier"""
        csi_data = [0] * 128
        result = extract_all_magnitudes(csi_data)
        
        assert len(result) == 64
        assert all(isinstance(x, float) for x in result)


class TestExtractPhase:
    """Test single subcarrier phase extraction"""
    
    def test_invalid_index(self):
        """Test with out-of-bounds index"""
        csi_data = [0] * 10
        assert extract_phase(csi_data, 100) == 0.0
    
    def test_zero_iq(self):
        """Test with zero I/Q values"""
        csi_data = [0] * 128
        assert extract_phase(csi_data, 0) == 0.0
    
    def test_known_phase(self):
        """Test with known I/Q values"""
        csi_data = [0] * 128
        # I = 1, Q = 0 -> phase = atan2(0, 1) = 0
        csi_data[0] = 0  # Q
        csi_data[1] = 1  # I
        
        result = extract_phase(csi_data, 0)
        assert result == pytest.approx(0.0, abs=1e-6)
    
    def test_phase_range(self):
        """Test that phase is in [-pi, pi]"""
        csi_data = [0] * 128
        csi_data[0] = 127  # Q = 127
        csi_data[1] = 128  # I = -128 (signed)
        
        result = extract_phase(csi_data, 0)
        assert -math.pi <= result <= math.pi


class TestExtractPhases:
    """Test multiple subcarrier phase extraction"""
    
    def test_default_all_subcarriers(self):
        """Test extracting all phases"""
        csi_data = [1] * 128
        result = extract_phases(csi_data)
        
        assert len(result) == 64
    
    def test_specific_subcarriers(self):
        """Test extracting specific subcarrier phases"""
        csi_data = [1] * 128
        result = extract_phases(csi_data, subcarriers=[0, 10, 20])
        
        assert len(result) == 3


class TestExtractAmplitudesAndPhases:
    """Test combined amplitude and phase extraction"""
    
    def test_returns_two_lists(self):
        """Test that two lists are returned"""
        csi_data = [1] * 128
        amplitudes, phases = extract_amplitudes_and_phases(csi_data)
        
        assert len(amplitudes) == 64
        assert len(phases) == 64
    
    def test_specific_subcarriers(self):
        """Test with specific subcarriers"""
        csi_data = [1] * 128
        amplitudes, phases = extract_amplitudes_and_phases(csi_data, subcarriers=[0, 10, 20])
        
        assert len(amplitudes) == 3
        assert len(phases) == 3
    
    def test_consistency_with_separate_calls(self):
        """Test that results match separate extraction calls"""
        np.random.seed(42)
        csi_data = list(np.random.randint(0, 256, 128))
        
        amps_combined, phases_combined = extract_amplitudes_and_phases(csi_data)
        amps_separate = extract_amplitudes(csi_data)
        phases_separate = extract_phases(csi_data)
        
        for i in range(len(amps_combined)):
            assert amps_combined[i] == pytest.approx(amps_separate[i], rel=1e-6)
            assert phases_combined[i] == pytest.approx(phases_separate[i], rel=1e-6)
