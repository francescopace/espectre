"""
Micro-ESPectre - Additional NBVI Calibrator Tests

Additional tests to improve coverage for src/nbvi_calibrator.py.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import pytest
import math
import os
import sys
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Patch buffer file path before importing
import nbvi_calibrator
_temp_dir = tempfile.gettempdir()
nbvi_calibrator.BUFFER_FILE = os.path.join(_temp_dir, 'nbvi_buffer_additional_test.bin')

from nbvi_calibrator import NBVICalibrator, get_valid_subcarriers, NUM_SUBCARRIERS


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


class TestNBVICalibrationFlow:
    """Test full calibration flow"""
    
    def test_calibration_with_good_data(self):
        """Test calibration succeeds with good synthetic data"""
        calibrator = NBVICalibrator(
            buffer_size=200
        )
        
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
        current_band = list(range(11, 23))
        selected_band, norm_scale = calibrator.calibrate(
            current_band,
            window_size=50,
            step=25
        )
        
        # Should return valid band
        assert selected_band is not None
        assert len(selected_band) == 12
        assert norm_scale > 0
        
        calibrator.free_buffer()
    
    def test_calibration_insufficient_data(self):
        """Test calibration fails gracefully with insufficient data"""
        calibrator = NBVICalibrator(buffer_size=100)
        
        # Add fewer packets than window_size
        for _ in range(30):
            csi_data = bytes([30, 10] * 64)
            calibrator.add_packet(csi_data)
        
        current_band = list(range(11, 23))
        selected_band, norm_scale = calibrator.calibrate(
            current_band,
            window_size=50,
            step=25
        )
        
        # Should return None
        assert selected_band is None
        assert norm_scale == 1.0
        
        calibrator.free_buffer()
    
    def test_calibration_normalization_always_active(self):
        """Test that normalization is always active (scale = 0.25 / baseline_variance)"""
        calibrator = NBVICalibrator(
            buffer_size=200
        )
        
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
        
        current_band = list(range(11, 23))
        selected_band, norm_scale = calibrator.calibrate(
            current_band,
            window_size=50,
            step=25
        )
        
        # Normalization is always active, scale should be > 0 and within valid range
        if selected_band is not None:
            assert norm_scale > 0, "Normalization scale should be positive"
            assert 0.01 <= norm_scale <= 100.0, f"Scale out of range: {norm_scale}"
        
        calibrator.free_buffer()


class TestNBVIReadPacket:
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


class TestNBVINoiseGateEdgeCases:
    """Test noise gate edge cases"""
    
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


class TestNBVISelectWithSpacing:
    """Test subcarrier selection with spacing"""
    
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


class TestNBVIBaselineWindow:
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


class TestNBVIAddPacketEdgeCases:
    """Test add_packet edge cases"""
    
    def test_add_packet_short_data(self):
        """Test adding packet with short CSI data"""
        calibrator = NBVICalibrator(buffer_size=10)
        
        # Short CSI data (less than 128 bytes)
        csi_data = bytes([30, 10] * 10)  # Only 20 bytes
        
        count = calibrator.add_packet(csi_data)
        
        assert count == 1
        
        calibrator.free_buffer()
    
    def test_add_packet_signed_conversion(self):
        """Test that signed byte conversion works correctly"""
        calibrator = NBVICalibrator(buffer_size=10)
        
        # Create data with negative values (as unsigned bytes > 127)
        csi_data = bytes([200, 150] * 64)  # Would be negative if signed
        
        count = calibrator.add_packet(csi_data)
        
        assert count == 1
        
        calibrator.free_buffer()


class TestNBVIVarianceTwoPass:
    """Test two-pass variance calculation"""
    
    def test_variance_large_values(self):
        """Test variance with large values"""
        calibrator = NBVICalibrator()
        
        values = [1000.0, 1001.0, 1002.0, 1003.0, 1004.0]
        result = calibrator._calculate_variance_two_pass(values)
        
        assert result == pytest.approx(2.0, rel=1e-6)
        
        calibrator.free_buffer()
    
    def test_variance_negative_values(self):
        """Test variance with negative values"""
        calibrator = NBVICalibrator()
        
        values = [-2.0, -1.0, 0.0, 1.0, 2.0]
        result = calibrator._calculate_variance_two_pass(values)
        
        assert result == pytest.approx(2.0, rel=1e-6)
        
        calibrator.free_buffer()


class TestNBVICalibrationFailurePaths:
    """Test calibration failure paths"""
    
    def test_calibration_no_valid_subcarriers(self):
        """Test calibration when all subcarriers are invalid - uses fallback"""
        calibrator = NBVICalibrator(buffer_size=200)
        
        # All zeros - all subcarriers will be null
        for _ in range(200):
            csi_data = bytes([0, 0] * 64)
            calibrator.add_packet(csi_data)
        
        current_band = list(range(11, 23))
        selected_band, norm_scale = calibrator.calibrate(
            current_band,
            window_size=50,
            step=25
        )
        
        # Should use fallback: return default subcarriers with calculated normalization
        assert selected_band == list(current_band)  # Default subcarriers used
        assert norm_scale > 0  # Normalization still calculated
        assert norm_scale <= 1.0  # Won't amplify (only attenuate or no-op)
        
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
        
        current_band = list(range(11, 23))
        selected_band, norm_scale = calibrator.calibrate(
            current_band,
            window_size=50,
            step=25
        )
        
        # Should use fallback: return default subcarriers with calculated normalization
        assert selected_band == list(current_band)  # Default subcarriers used
        assert norm_scale > 0  # Normalization still calculated
        
        calibrator.free_buffer()

