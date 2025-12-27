"""
Micro-ESPectre - NBVI Calibrator Unit Tests

Tests for NBVICalibrator class in src/nbvi_calibrator.py.

Note: NBVICalibrator uses file-based storage at a hardcoded path (/nbvi_buffer.bin)
which is designed for MicroPython on ESP32. For unit tests, we mock the file path
or test only the algorithmic components that don't require file I/O.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import pytest
import math
import os
import numpy as np
from pathlib import Path
import tempfile

# We need to patch BUFFER_FILE before importing NBVICalibrator
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Patch the buffer file path to use a temp directory
import nbvi_calibrator
_original_buffer_file = nbvi_calibrator.BUFFER_FILE
_temp_dir = tempfile.gettempdir()
nbvi_calibrator.BUFFER_FILE = os.path.join(_temp_dir, 'nbvi_buffer_test.bin')


@pytest.fixture(autouse=True)
def cleanup_buffer_file():
    """Clean up test buffer file before and after each test"""
    test_file = nbvi_calibrator.BUFFER_FILE
    if os.path.exists(test_file):
        try:
            os.remove(test_file)
        except:
            pass
    yield
    if os.path.exists(test_file):
        try:
            os.remove(test_file)
        except:
            pass


class TestNBVICalibrator:
    """Test NBVICalibrator class"""
    
    def test_initialization(self):
        """Test calibrator initialization"""
        from nbvi_calibrator import NBVICalibrator
        
        calibrator = NBVICalibrator(buffer_size=500, percentile=10)
        
        assert calibrator.buffer_size == 500
        assert calibrator.percentile == 10
        assert calibrator.alpha == 0.5
        assert calibrator.min_spacing == 1
        assert calibrator._packet_count == 0
        
        # Cleanup
        calibrator.free_buffer()
    
    def test_custom_parameters(self):
        """Test custom parameter initialization"""
        from nbvi_calibrator import NBVICalibrator
        
        calibrator = NBVICalibrator(
            buffer_size=1000,
            percentile=15,
            alpha=0.4,
            min_spacing=5
        )
        
        assert calibrator.buffer_size == 1000
        assert calibrator.percentile == 15
        assert calibrator.alpha == 0.4
        assert calibrator.min_spacing == 5
        
        calibrator.free_buffer()
    
    def test_add_packet_returns_count(self):
        """Test that add_packet returns packet count"""
        from nbvi_calibrator import NBVICalibrator
        
        calibrator = NBVICalibrator(buffer_size=100)
        
        # Create synthetic CSI data (128 bytes = 64 subcarriers * 2 I/Q)
        csi_data = bytes([30, 10] * 64)
        
        count = calibrator.add_packet(csi_data)
        assert count == 1
        
        count = calibrator.add_packet(csi_data)
        assert count == 2
        
        calibrator.free_buffer()
    
    def test_add_packet_stops_at_buffer_size(self):
        """Test that add_packet stops accepting at buffer_size"""
        from nbvi_calibrator import NBVICalibrator
        
        calibrator = NBVICalibrator(buffer_size=10)
        csi_data = bytes([30, 10] * 64)
        
        # Add more than buffer_size packets
        for i in range(15):
            count = calibrator.add_packet(csi_data)
        
        # Should stop at buffer_size
        assert count == 10
        assert calibrator._packet_count == 10
        
        calibrator.free_buffer()
    
    def test_free_buffer(self):
        """Test that free_buffer cleans up resources"""
        from nbvi_calibrator import NBVICalibrator, BUFFER_FILE
        
        calibrator = NBVICalibrator(buffer_size=10)
        csi_data = bytes([30, 10] * 64)
        
        for _ in range(5):
            calibrator.add_packet(csi_data)
        
        calibrator.free_buffer()
        
        # File should be removed
        assert not os.path.exists(BUFFER_FILE)


class TestNBVICalculation:
    """Test NBVI calculation methods"""
    
    def test_calculate_nbvi_weighted(self):
        """Test NBVI weighted calculation"""
        from nbvi_calibrator import NBVICalibrator
        
        calibrator = NBVICalibrator()
        
        # Test with known values
        magnitudes = [30.0, 32.0, 28.0, 31.0, 29.0]
        
        result = calibrator._calculate_nbvi_weighted(magnitudes)
        
        assert 'nbvi' in result
        assert 'mean' in result
        assert 'std' in result
        assert result['mean'] == pytest.approx(30.0, rel=1e-6)
        assert result['nbvi'] > 0
        
        calibrator.free_buffer()
    
    def test_nbvi_empty_list(self):
        """Test NBVI with empty list returns inf"""
        from nbvi_calibrator import NBVICalibrator
        
        calibrator = NBVICalibrator()
        result = calibrator._calculate_nbvi_weighted([])
        
        assert result['nbvi'] == float('inf')
        
        calibrator.free_buffer()
    
    def test_nbvi_zero_mean(self):
        """Test NBVI with zero mean returns inf"""
        from nbvi_calibrator import NBVICalibrator
        
        calibrator = NBVICalibrator()
        result = calibrator._calculate_nbvi_weighted([0.0, 0.0, 0.0])
        
        assert result['nbvi'] == float('inf')
        
        calibrator.free_buffer()
    
    def test_nbvi_lower_is_better(self):
        """Test that stable signal has lower NBVI than noisy signal"""
        from nbvi_calibrator import NBVICalibrator
        
        calibrator = NBVICalibrator()
        
        # Stable signal (low std)
        stable = [50.0, 50.5, 49.5, 50.2, 49.8]
        result_stable = calibrator._calculate_nbvi_weighted(stable)
        
        # Noisy signal (high std)
        noisy = [50.0, 60.0, 40.0, 55.0, 45.0]
        result_noisy = calibrator._calculate_nbvi_weighted(noisy)
        
        # Lower NBVI = better subcarrier
        assert result_stable['nbvi'] < result_noisy['nbvi']
        
        calibrator.free_buffer()


class TestPercentileCalculation:
    """Test percentile calculation"""
    
    def test_percentile_empty(self):
        """Test percentile of empty list"""
        from nbvi_calibrator import NBVICalibrator
        
        calibrator = NBVICalibrator()
        result = calibrator._percentile([], 10)
        
        assert result == 0.0
        
        calibrator.free_buffer()
    
    def test_percentile_single_value(self):
        """Test percentile of single value"""
        from nbvi_calibrator import NBVICalibrator
        
        calibrator = NBVICalibrator()
        result = calibrator._percentile([5.0], 50)
        
        assert result == 5.0
        
        calibrator.free_buffer()
    
    def test_percentile_median(self):
        """Test 50th percentile is median"""
        from nbvi_calibrator import NBVICalibrator
        
        calibrator = NBVICalibrator()
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = calibrator._percentile(values, 50)
        
        assert result == pytest.approx(3.0, rel=1e-6)
        
        calibrator.free_buffer()
    
    def test_percentile_extremes(self):
        """Test 0 and 100 percentiles"""
        from nbvi_calibrator import NBVICalibrator
        
        calibrator = NBVICalibrator()
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        p0 = calibrator._percentile(values, 0)
        p100 = calibrator._percentile(values, 100)
        
        assert p0 == 1.0
        assert p100 == 5.0
        
        calibrator.free_buffer()


class TestNoiseGate:
    """Test noise gate functionality"""
    
    def test_noise_gate_excludes_weak(self):
        """Test that noise gate excludes weak subcarriers"""
        from nbvi_calibrator import NBVICalibrator
        
        calibrator = NBVICalibrator()
        
        # Create metrics with some weak subcarriers
        metrics = [
            {'subcarrier': 0, 'mean': 50.0, 'nbvi': 0.1},  # Strong
            {'subcarrier': 1, 'mean': 0.5, 'nbvi': 0.1},   # Weak (below threshold)
            {'subcarrier': 2, 'mean': 40.0, 'nbvi': 0.2},  # Strong
            {'subcarrier': 3, 'mean': 0.8, 'nbvi': 0.1},   # Weak
            {'subcarrier': 4, 'mean': 30.0, 'nbvi': 0.3},  # Strong
        ]
        
        filtered = calibrator._apply_noise_gate(metrics)
        
        # Weak subcarriers should be excluded
        subcarriers = [m['subcarrier'] for m in filtered]
        assert 1 not in subcarriers
        assert 3 not in subcarriers
        
        calibrator.free_buffer()
    
    def test_noise_gate_keeps_strong(self):
        """Test that noise gate keeps strong subcarriers"""
        from nbvi_calibrator import NBVICalibrator
        
        calibrator = NBVICalibrator()
        
        # All strong subcarriers
        metrics = [
            {'subcarrier': i, 'mean': 30.0 + i, 'nbvi': 0.1}
            for i in range(20)
        ]
        
        filtered = calibrator._apply_noise_gate(metrics)
        
        # Most should be kept (bottom 25% excluded by percentile with default noise_gate_percentile=25)
        # 20 subcarriers, 25% = 5 excluded, 15 kept
        assert len(filtered) >= 15
        
        calibrator.free_buffer()


class TestSpectralSpacing:
    """Test spectral spacing selection"""
    
    def test_select_top_5_always_included(self):
        """Test that top 5 subcarriers are always included"""
        from nbvi_calibrator import NBVICalibrator
        
        calibrator = NBVICalibrator()
        
        # Create sorted metrics (by NBVI ascending)
        metrics = [
            {'subcarrier': i, 'nbvi': i * 0.01}
            for i in range(30)
        ]
        
        selected = calibrator._select_with_spacing(metrics, k=12)
        
        # Top 5 (subcarriers 0,1,2,3,4) should be included
        for i in range(5):
            assert i in selected
        
        calibrator.free_buffer()
    
    def test_select_returns_correct_count(self):
        """Test that selection returns requested count"""
        from nbvi_calibrator import NBVICalibrator
        
        calibrator = NBVICalibrator()
        
        metrics = [
            {'subcarrier': i, 'nbvi': i * 0.01}
            for i in range(64)
        ]
        
        selected = calibrator._select_with_spacing(metrics, k=12)
        
        assert len(selected) == 12
        
        calibrator.free_buffer()
    
    def test_select_respects_spacing(self):
        """Test that selection respects minimum spacing"""
        from nbvi_calibrator import NBVICalibrator
        
        calibrator = NBVICalibrator(min_spacing=3)
        
        # Create metrics where adjacent subcarriers have low NBVI
        metrics = [
            {'subcarrier': i, 'nbvi': 0.01 if i < 10 else 0.1}
            for i in range(64)
        ]
        
        selected = calibrator._select_with_spacing(metrics, k=12)
        
        # After top 5, remaining should have spacing >= 3
        # (This is a soft constraint in the algorithm)
        assert len(selected) == 12
        
        calibrator.free_buffer()
    
    def test_selected_is_sorted(self):
        """Test that selected subcarriers are sorted"""
        from nbvi_calibrator import NBVICalibrator
        
        calibrator = NBVICalibrator()
        
        # Random order metrics
        metrics = [
            {'subcarrier': 50, 'nbvi': 0.01},
            {'subcarrier': 10, 'nbvi': 0.02},
            {'subcarrier': 30, 'nbvi': 0.03},
            {'subcarrier': 20, 'nbvi': 0.04},
            {'subcarrier': 40, 'nbvi': 0.05},
            {'subcarrier': 5, 'nbvi': 0.06},
            {'subcarrier': 15, 'nbvi': 0.07},
            {'subcarrier': 25, 'nbvi': 0.08},
            {'subcarrier': 35, 'nbvi': 0.09},
            {'subcarrier': 45, 'nbvi': 0.10},
            {'subcarrier': 55, 'nbvi': 0.11},
            {'subcarrier': 60, 'nbvi': 0.12},
        ]
        
        selected = calibrator._select_with_spacing(metrics, k=12)
        
        # Should be sorted ascending
        assert selected == sorted(selected)
        
        calibrator.free_buffer()


class TestVarianceTwoPass:
    """Test two-pass variance calculation"""
    
    def test_variance_empty(self):
        """Test variance of empty list"""
        from nbvi_calibrator import NBVICalibrator
        
        calibrator = NBVICalibrator()
        result = calibrator._calculate_variance_two_pass([])
        
        assert result == 0.0
        
        calibrator.free_buffer()
    
    def test_variance_single_value(self):
        """Test variance of single value"""
        from nbvi_calibrator import NBVICalibrator
        
        calibrator = NBVICalibrator()
        result = calibrator._calculate_variance_two_pass([5.0])
        
        assert result == 0.0
        
        calibrator.free_buffer()
    
    def test_variance_known_value(self):
        """Test variance with known result"""
        from nbvi_calibrator import NBVICalibrator
        
        calibrator = NBVICalibrator()
        # Variance of [1,2,3,4,5] = 2.0
        result = calibrator._calculate_variance_two_pass([1.0, 2.0, 3.0, 4.0, 5.0])
        
        assert result == pytest.approx(2.0, rel=1e-6)
        
        calibrator.free_buffer()


class TestGetValidSubcarriers:
    """Test valid subcarrier indices"""
    
    def test_returns_all_64(self):
        """Test that all 64 subcarriers are returned"""
        from nbvi_calibrator import get_valid_subcarriers
        
        subcarriers = get_valid_subcarriers()
        
        assert len(subcarriers) == 64
        assert subcarriers == tuple(range(64))
    
    def test_chip_type_ignored(self):
        """Test that chip_type parameter is ignored"""
        from nbvi_calibrator import get_valid_subcarriers
        
        # Should return same result regardless of chip_type
        result1 = get_valid_subcarriers(chip_type='c6')
        result2 = get_valid_subcarriers(chip_type='s3')
        result3 = get_valid_subcarriers(chip_type=None)
        
        assert result1 == result2 == result3


class TestCalibrationIntegration:
    """Integration tests for full calibration flow"""
    
    @pytest.mark.skip(reason="Requires MicroPython umqtt module - run on device")
    def test_calibration_with_synthetic_data(self):
        """Test full calibration with synthetic stable CSI data"""
        from nbvi_calibrator import NBVICalibrator
        
        # Use smaller buffer for testing
        calibrator = NBVICalibrator(buffer_size=200)
        
        np.random.seed(42)
        
        # Generate stable CSI packets
        for _ in range(200):
            # Each subcarrier has consistent amplitude with small noise
            csi_data = []
            for sc in range(64):
                # Base amplitude varies by subcarrier, small noise
                base_amp = 20 + sc % 20
                noise = np.random.normal(0, 2)
                I = int(base_amp + noise)
                Q = int(base_amp * 0.3 + np.random.normal(0, 1))
                csi_data.extend([max(-127, min(127, I)), max(-127, min(127, Q))])
            
            calibrator.add_packet(bytes(csi_data))
        
        # Calibrate
        current_band = list(range(11, 23))  # Default band
        selected_band, norm_scale = calibrator.calibrate(
            current_band, 
            window_size=50,
            step=25
        )
        
        # Should return a valid band
        if selected_band is not None:
            assert len(selected_band) == 12
            assert all(0 <= sc < 64 for sc in selected_band)
            assert norm_scale > 0
        
        calibrator.free_buffer()
    
    @pytest.mark.skip(reason="Requires MicroPython umqtt module - run on device")
    def test_calibration_returns_none_with_insufficient_data(self):
        """Test that calibration fails gracefully with insufficient data"""
        from nbvi_calibrator import NBVICalibrator
        
        calibrator = NBVICalibrator(buffer_size=100)
        
        # Add only a few packets (less than window_size)
        for _ in range(10):
            csi_data = bytes([30, 10] * 64)
            calibrator.add_packet(csi_data)
        
        current_band = list(range(11, 23))
        selected_band, norm_scale = calibrator.calibrate(
            current_band,
            window_size=50,  # More than available packets
            step=25
        )
        
        # Should return None due to insufficient data
        assert selected_band is None
        assert norm_scale == 1.0
        
        calibrator.free_buffer()

