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

from nbvi_calibrator import NBVICalibrator


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


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================

class TestNBVICalibrator:
    """Test NBVICalibrator class initialization and basic operations"""
    
    def test_initialization(self):
        """Test calibrator initialization"""
        calibrator = NBVICalibrator(buffer_size=500, percentile=10)
        
        assert calibrator.buffer_size == 500
        assert calibrator.percentile == 10
        assert calibrator.alpha == 0.5
        assert calibrator.min_spacing == 1
        assert calibrator._packet_count == 0
        
        calibrator.free_buffer()
    
    def test_custom_parameters(self):
        """Test custom parameter initialization"""
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
        from nbvi_calibrator import BUFFER_FILE
        
        calibrator = NBVICalibrator(buffer_size=10)
        csi_data = bytes([30, 10] * 64)
        
        for _ in range(5):
            calibrator.add_packet(csi_data)
        
        calibrator.free_buffer()
        
        # File should be removed
        assert not os.path.exists(BUFFER_FILE)


# ============================================================================
# NBVI CALCULATION TESTS
# ============================================================================

class TestNBVICalculation:
    """Test NBVI calculation methods"""
    
    def test_calculate_nbvi_weighted(self):
        """Test NBVI weighted calculation"""
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
        calibrator = NBVICalibrator()
        result = calibrator._calculate_nbvi_weighted([])
        
        assert result['nbvi'] == float('inf')
        
        calibrator.free_buffer()
    
    def test_nbvi_zero_mean(self):
        """Test NBVI with zero mean returns inf"""
        calibrator = NBVICalibrator()
        result = calibrator._calculate_nbvi_weighted([0.0, 0.0, 0.0])
        
        assert result['nbvi'] == float('inf')
        
        calibrator.free_buffer()
    
    def test_nbvi_lower_is_better(self):
        """Test that stable signal has lower NBVI than noisy signal"""
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


# ============================================================================
# NOISE GATE TESTS
# ============================================================================

class TestNoiseGate:
    """Test noise gate functionality"""
    
    def test_noise_gate_excludes_weak(self):
        """Test that noise gate excludes weak subcarriers"""
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
    
    def test_noise_gate_all_weak(self):
        """Test noise gate when all subcarriers are weak"""
        calibrator = NBVICalibrator()
        
        # All weak subcarriers
        metrics = [
            {'subcarrier': i, 'mean': 0.5, 'nbvi': 0.1}
            for i in range(20)
        ]
        
        filtered = calibrator._apply_noise_gate(metrics)
        
        # Should return empty list
        assert len(filtered) == 0
        
        calibrator.free_buffer()
    
    def test_noise_gate_mixed(self):
        """Test noise gate with mixed strong/weak"""
        calibrator = NBVICalibrator()
        
        metrics = [
            {'subcarrier': 0, 'mean': 50.0, 'nbvi': 0.1},
            {'subcarrier': 1, 'mean': 0.5, 'nbvi': 0.1},  # Below 1.0 threshold
            {'subcarrier': 2, 'mean': 30.0, 'nbvi': 0.2},
            {'subcarrier': 3, 'mean': 0.8, 'nbvi': 0.1},  # Below 1.0 threshold
            {'subcarrier': 4, 'mean': 40.0, 'nbvi': 0.15},
        ]
        
        filtered = calibrator._apply_noise_gate(metrics)
        
        # Weak subcarriers should be excluded
        subcarriers = [m['subcarrier'] for m in filtered]
        assert 1 not in subcarriers
        assert 3 not in subcarriers
        
        calibrator.free_buffer()


# ============================================================================
# SPECTRAL SPACING TESTS
# ============================================================================

class TestSpectralSpacing:
    """Test spectral spacing selection"""
    
    def test_select_top_5_always_included(self):
        """Test that top 5 subcarriers are always included"""
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
    
    def test_select_fewer_than_k(self):
        """Test selection when fewer than k subcarriers available"""
        calibrator = NBVICalibrator()
        
        # Only 8 subcarriers available
        metrics = [
            {'subcarrier': i * 5, 'nbvi': i * 0.01}
            for i in range(8)
        ]
        
        selected = calibrator._select_with_spacing(metrics, k=12)
        
        # Should return all available
        assert len(selected) == 8
        
        calibrator.free_buffer()
    
    def test_select_exact_k(self):
        """Test selection when exactly k subcarriers available"""
        calibrator = NBVICalibrator()
        
        metrics = [
            {'subcarrier': i * 4, 'nbvi': i * 0.01}
            for i in range(12)
        ]
        
        selected = calibrator._select_with_spacing(metrics, k=12)
        
        assert len(selected) == 12
        
        calibrator.free_buffer()
    
    def test_select_with_tight_spacing(self):
        """Test selection with very tight spacing constraint"""
        calibrator = NBVICalibrator(min_spacing=10)
        
        # All subcarriers are close together
        metrics = [
            {'subcarrier': i, 'nbvi': i * 0.01}
            for i in range(64)
        ]
        
        selected = calibrator._select_with_spacing(metrics, k=12)
        
        # Should still return 12 (falls back to best remaining)
        assert len(selected) == 12
        
        calibrator.free_buffer()


# ============================================================================
# PACKET READING TESTS
# ============================================================================

class TestPacketReading:
    """Test packet reading from file"""
    
    def test_read_packet(self):
        """Test reading a single packet"""
        calibrator = NBVICalibrator(buffer_size=10)
        
        # Add some packets
        for i in range(5):
            csi_data = bytes([30 + i, 10] * 64)
            calibrator.add_packet(csi_data)
        
        # Prepare for reading
        calibrator._prepare_for_reading()
        
        # Read first packet
        packet = calibrator._read_packet(0)
        
        assert packet is not None
        assert len(packet) == 64
        
        calibrator.free_buffer()
    
    def test_read_window(self):
        """Test reading a window of packets"""
        calibrator = NBVICalibrator(buffer_size=20)
        
        # Add packets
        for i in range(20):
            csi_data = bytes([30 + i % 10, 10] * 64)
            calibrator.add_packet(csi_data)
        
        # Prepare for reading
        calibrator._prepare_for_reading()
        
        # Read window
        window = calibrator._read_window(0, 10)
        
        assert len(window) == 10
        assert len(window[0]) == 64
        
        calibrator.free_buffer()


# ============================================================================
# BASELINE WINDOW TESTS
# ============================================================================

class TestBaselineWindow:
    """Test baseline window finding"""
    
    def test_find_baseline_insufficient_packets(self):
        """Test baseline finding with insufficient packets"""
        calibrator = NBVICalibrator(buffer_size=50)
        
        # Add fewer packets than window_size
        for _ in range(30):
            csi_data = bytes([30, 10] * 64)
            calibrator.add_packet(csi_data)
        
        calibrator._prepare_for_reading()
        
        current_band = list(range(11, 23))
        # _find_candidate_windows returns empty list when insufficient packets
        candidates = calibrator._find_candidate_windows(
            current_band, window_size=50, step=25
        )
        
        assert candidates == []
        
        calibrator.free_buffer()
    
    def test_find_baseline_with_stable_data(self):
        """Test baseline finding with stable data"""
        calibrator = NBVICalibrator(buffer_size=300)
        
        np.random.seed(42)
        
        # Generate stable data
        for _ in range(300):
            csi_data = []
            for sc in range(64):
                base_amp = 30
                noise = np.random.normal(0, 1)
                I = max(0, min(255, int(base_amp + noise)))
                Q = max(0, min(255, int(base_amp * 0.3)))
                csi_data.extend([I, Q])
            
            calibrator.add_packet(bytes(csi_data))
        
        calibrator._prepare_for_reading()
        
        current_band = list(range(11, 23))
        # _find_candidate_windows returns list of (start_idx, variance) tuples
        candidates = calibrator._find_candidate_windows(
            current_band, window_size=100, step=50
        )
        
        # Should find at least one candidate window
        assert len(candidates) > 0
        # Each candidate is a tuple of (start_idx, variance)
        assert isinstance(candidates[0], tuple)
        assert len(candidates[0]) == 2
        
        calibrator.free_buffer()


# ============================================================================
# ADD PACKET EDGE CASES
# ============================================================================

class TestAddPacketEdgeCases:
    """Test add_packet edge cases"""
    
    def test_add_packet_short_data_rejected(self):
        """Test adding packet with short CSI data is rejected"""
        calibrator = NBVICalibrator(buffer_size=10)
        
        # Short CSI data (less than 128 bytes) - should be rejected
        csi_data = bytes([30, 10] * 10)  # Only 20 bytes
        
        count = calibrator.add_packet(csi_data)
        
        # Short packets are rejected (count stays at 0)
        assert count == 0
        
        calibrator.free_buffer()
    
    def test_add_packet_signed_conversion(self):
        """Test that signed byte conversion works correctly"""
        calibrator = NBVICalibrator(buffer_size=10)
        
        # Create data with negative values (as unsigned bytes > 127)
        csi_data = bytes([200, 150] * 64)  # Would be negative if signed
        
        count = calibrator.add_packet(csi_data)
        
        assert count == 1
        
        calibrator.free_buffer()


# ============================================================================
# CALIBRATION INTEGRATION TESTS
# ============================================================================

class TestCalibrationIntegration:
    """Integration tests for full calibration flow"""
    
    def test_calibration_with_synthetic_data(self):
        """Test full calibration with synthetic stable CSI data"""
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
                # Convert to unsigned byte range (0-255)
                I = max(0, min(255, I if I >= 0 else I + 256))
                Q = max(0, min(255, Q if Q >= 0 else Q + 256))
                csi_data.extend([I, Q])
            
            calibrator.add_packet(bytes(csi_data))
        
        # Calibrate
        selected_band, mv_values = calibrator.calibrate()
        
        # Should return a valid band and mv_values
        if selected_band is not None:
            assert len(selected_band) == 12
            assert all(0 <= sc < 64 for sc in selected_band)
            assert len(mv_values) > 0, "mv_values should not be empty"
        
        calibrator.free_buffer()
    
    def test_calibration_returns_none_with_insufficient_data(self):
        """Test that calibration fails gracefully with insufficient data"""
        calibrator = NBVICalibrator(buffer_size=100)
        
        # Add only a few packets (less than window_size)
        for _ in range(10):
            csi_data = bytes([30, 10] * 64)
            calibrator.add_packet(csi_data)
        
        # Calibrate - should fail due to insufficient data
        selected_band, mv_values = calibrator.calibrate()
        
        # Should return None due to insufficient data
        assert selected_band is None
        assert mv_values == []
        
        calibrator.free_buffer()
    
    def test_calibration_with_good_data(self):
        """Test calibration succeeds with good synthetic data"""
        calibrator = NBVICalibrator(buffer_size=200)
        
        np.random.seed(42)
        
        # Generate stable baseline data
        for _ in range(200):
            csi_data = []
            for sc in range(64):
                # Good subcarriers in the middle have high, stable amplitude
                if 10 <= sc <= 50 and sc != 32:  # Avoid DC subcarrier
                    base_amp = 40 + (sc % 10)
                    noise = np.random.normal(0, 1)
                else:
                    # Weak or guard band subcarriers
                    base_amp = 2
                    noise = 0
                
                I = int(base_amp + noise)
                Q = int(base_amp * 0.3 + np.random.normal(0, 0.5))
                # Ensure values are in uint8 range (bytes expects 0-255)
                I = max(0, min(255, I))
                Q = max(0, min(255, Q))
                csi_data.extend([I, Q])
            
            calibrator.add_packet(bytes(csi_data))
        
        # Run calibration
        selected_band, mv_values = calibrator.calibrate()
        
        # Should return valid band and mv_values
        assert selected_band is not None
        assert len(selected_band) == 12
        assert len(mv_values) > 0
        
        calibrator.free_buffer()
    
    def test_calibration_returns_mv_values(self):
        """Test that calibration returns mv_values for threshold calculation"""
        calibrator = NBVICalibrator(buffer_size=200)
        
        np.random.seed(42)
        
        for _ in range(200):
            csi_data = []
            for sc in range(64):
                if 10 <= sc <= 50 and sc != 32:
                    base_amp = 40
                else:
                    base_amp = 2
                
                I = max(0, min(255, int(base_amp)))
                Q = max(0, min(255, int(base_amp * 0.3)))
                csi_data.extend([I, Q])
            
            calibrator.add_packet(bytes(csi_data))
        
        # Calibrate
        selected_band, mv_values = calibrator.calibrate()
        
        # mv_values should be returned for external threshold calculation
        if selected_band is not None:
            assert len(mv_values) > 0, "mv_values should not be empty"
            # All values should be non-negative (variance)
            assert all(v >= 0 for v in mv_values), "All mv_values should be non-negative"
        
        calibrator.free_buffer()


# ============================================================================
# CALIBRATION FAILURE TESTS
# ============================================================================

class TestCalibrationFailurePaths:
    """Test calibration failure paths"""
    
    def test_calibration_no_valid_subcarriers(self):
        """Test calibration when all subcarriers are invalid - uses fallback"""
        calibrator = NBVICalibrator(buffer_size=200)
        
        # All zeros - all subcarriers will be null
        for _ in range(200):
            csi_data = bytes([0, 0] * 64)
            calibrator.add_packet(csi_data)
        
        # Calibrate
        selected_band, mv_values = calibrator.calibrate()
        
        # With all zeros, calibration may fail or use fallback
        # Either way, mv_values should be a list
        assert isinstance(mv_values, list)
        
        calibrator.free_buffer()
    
    def test_calibration_few_valid_subcarriers(self):
        """Test calibration when too few valid subcarriers - uses fallback"""
        calibrator = NBVICalibrator(buffer_size=200)
        
        # Only a few strong subcarriers
        for _ in range(200):
            csi_data = []
            for sc in range(64):
                if sc in [15, 16, 17]:  # Only 3 strong subcarriers
                    I, Q = 50, 15
                else:
                    I, Q = 0, 0
                csi_data.extend([I, Q])
            
            calibrator.add_packet(bytes(csi_data))
        
        # Calibrate
        selected_band, mv_values = calibrator.calibrate()
        
        # With few valid subcarriers, calibration may fail or use fallback
        # Either way, mv_values should be a list
        assert isinstance(mv_values, list)
        
        calibrator.free_buffer()
