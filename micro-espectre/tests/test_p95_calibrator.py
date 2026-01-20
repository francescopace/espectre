"""
Tests for P95 Calibrator - P95 Moving Variance Optimization

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import pytest
import math
import os
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Patch buffer file for testing
import p95_calibrator
p95_calibrator.BUFFER_FILE = os.path.join(tempfile.gettempdir(), 'p95_test.bin')

from p95_calibrator import (
    P95Calibrator,
    BandCalibrator,  # Backward compat alias
    BAND_SIZE, MVS_WINDOW_SIZE, MVS_THRESHOLD, BAND_SELECTION_PERCENTILE
)
from src.config import GUARD_BAND_LOW, GUARD_BAND_HIGH, DC_SUBCARRIER


class TestHT20Constants:
    """Test HT20-only constants (64 subcarriers)"""
    
    def test_guard_bands(self):
        """Test HT20 guard band constants"""
        assert GUARD_BAND_LOW == 11
        assert GUARD_BAND_HIGH == 52
        assert DC_SUBCARRIER == 32


class TestBandSelectionConstants:
    """Test band selection constants"""
    
    def test_band_selection_percentile(self):
        """Test P95 is used for band selection (fixed algorithm)"""
        assert BAND_SELECTION_PERCENTILE == 95


class TestP95CalibratorInit:
    """Test P95Calibrator initialization"""
    
    def test_default_init(self):
        """Test default initialization"""
        cal = P95Calibrator()
        
        assert cal.buffer_size == 700
        assert cal._packet_count == 0
        assert cal._filtered_count == 0
        
        cal.free_buffer()
    
    def test_custom_buffer_size(self):
        """Test custom buffer size"""
        cal = P95Calibrator(buffer_size=500)
        
        assert cal.buffer_size == 500
        
        cal.free_buffer()


class TestP95CalibratorAddPacket:
    """Test packet addition"""
    
    def test_add_packet_returns_count(self):
        """Test add_packet returns progress count"""
        cal = P95Calibrator(buffer_size=10)
        
        # Create fake 64 SC packet (128 bytes I/Q)
        packet = bytes([50, 50] * 64)
        
        count = cal.add_packet(packet)
        assert count == 1
        
        count = cal.add_packet(packet)
        assert count == 2
        
        cal.free_buffer()
    
    def test_add_packet_stops_at_buffer_size(self):
        """Test buffer stops at max size"""
        cal = P95Calibrator(buffer_size=5)
        
        packet = bytes([50, 50] * 64)
        
        for i in range(10):
            count = cal.add_packet(packet)
        
        assert count == 5  # Stopped at buffer_size
        assert cal._packet_count == 5
        
        cal.free_buffer()
    
    def test_filter_wrong_sc_count(self):
        """Test packets with wrong SC count are filtered (HT20: 64 SC only)"""
        cal = P95Calibrator(buffer_size=10)
        
        # 64 SC packet (128 bytes) - HT20 standard
        packet_64 = bytes([50, 50] * 64)
        
        # 256 SC packet (512 bytes) - not supported
        packet_256 = bytes([50, 50] * 256)
        
        # Add 64 SC - should be accepted
        count = cal.add_packet(packet_64)
        assert count == 1
        
        # Add 256 SC - should be filtered
        count = cal.add_packet(packet_256)
        assert count == 1  # Still 1
        assert cal._filtered_count == 1
        
        # Add another 64 SC
        count = cal.add_packet(packet_64)
        assert count == 2
        
        cal.free_buffer()
    
    def test_accepts_ht20_packets(self):
        """Test HT20 packets (64 SC) are accepted"""
        cal = P95Calibrator(buffer_size=10)
        
        packet = bytes([50, 50] * 64)
        count = cal.add_packet(packet)
        assert count == 1
        
        cal.free_buffer()


class TestUtilityFunctions:
    """Test utility functions from utils.py"""
    
    def test_calculate_spatial_turbulence(self):
        """Test spatial turbulence calculation"""
        from utils import calculate_spatial_turbulence
        
        # Constant magnitudes -> 0 turbulence
        packet = [50] * 64
        turb = calculate_spatial_turbulence(packet, list(range(12)))
        assert turb == 0.0
        
        # Variable magnitudes -> non-zero turbulence
        packet = list(range(64))
        turb = calculate_spatial_turbulence(packet, list(range(12)))
        assert turb > 0
    
    def test_calculate_moving_variance(self):
        """Test moving variance calculation"""
        from utils import calculate_moving_variance
        
        # Constant values -> 0 variance
        values = [1.0] * 100
        mv = calculate_moving_variance(values, window_size=50)
        assert len(mv) == 50  # 100 - 50
        assert all(v == 0.0 for v in mv)
        
        # Variable values -> non-zero variance
        values = list(range(100))
        mv = calculate_moving_variance(values, window_size=50)
        assert len(mv) == 50
        assert all(v > 0 for v in mv)
    
    def test_calculate_percentile(self):
        """Test percentile calculation"""
        from utils import calculate_percentile
        
        # Known values
        values = list(range(100))  # 0-99
        pxx = calculate_percentile(values, 95)
        # P95 of 0-99 should be around 94-95
        assert 94 <= pxx <= 95
        
        # Empty list
        pxx_empty = calculate_percentile([], 95)
        assert pxx_empty == 0.0  # Default for empty list (matches C++ NBVICalibrator)
    
    def test_get_candidate_bands(self):
        """Test candidate band generation for HT20 (64 SC)"""
        cal = P95Calibrator(buffer_size=10)
        
        candidates = cal._get_candidate_bands()
        
        # Should have bands before and after DC
        assert len(candidates) > 0
        
        # Each band should have 12 subcarriers
        for band in candidates:
            assert len(band) == BAND_SIZE
        
        # No band should include DC (subcarrier 32)
        for band in candidates:
            assert DC_SUBCARRIER not in band
        
        # All subcarriers should be in valid range
        for band in candidates:
            for sc in band:
                assert GUARD_BAND_LOW <= sc <= GUARD_BAND_HIGH
        
        cal.free_buffer()
    


class TestP95CalibratorCalibration:
    """Test calibration process"""
    
    def test_calibration_insufficient_packets(self):
        """Test calibration fails with insufficient packets"""
        cal = P95Calibrator(buffer_size=100)
        
        # Add only a few packets
        packet = bytes([50, 50] * 64)
        for _ in range(10):
            cal.add_packet(packet)
        
        band, mv_values = cal.calibrate()
        
        assert band is None
        assert mv_values == []
        
        cal.free_buffer()
    
    def test_calibration_returns_valid_band(self):
        """Test calibration returns 12 valid subcarriers"""
        cal = P95Calibrator(buffer_size=100)
        
        # Add enough packets with some variation
        import random
        random.seed(42)
        
        for _ in range(100):
            # Random I/Q values
            packet = bytes([random.randint(30, 80) for _ in range(128)])
            cal.add_packet(packet)
        
        band, mv_values = cal.calibrate()
        
        if band is not None:
            assert len(band) == 12
            # All subcarriers should be in valid range
            for sc in band:
                assert 11 <= sc <= 52
        
        cal.free_buffer()
    
    def test_calibration_returns_mv_values(self):
        """Test calibration returns MV values for threshold calculation"""
        cal = P95Calibrator(buffer_size=100)
        
        import random
        random.seed(42)
        
        for _ in range(100):
            packet = bytes([random.randint(30, 80) for _ in range(128)])
            cal.add_packet(packet)
        
        band, mv_values = cal.calibrate()
        
        # MV values should be a non-empty list when calibration succeeds
        if band is not None:
            assert len(mv_values) > 0
            assert all(v >= 0 for v in mv_values)
        
        cal.free_buffer()


class TestP95CalibratorEvaluateBand:
    """Test band evaluation"""
    
    def test_evaluate_band_returns_metrics(self):
        """Test band evaluation returns expected metrics"""
        cal = P95Calibrator(buffer_size=100)
        
        # Create synthetic packets
        import random
        random.seed(42)
        
        for _ in range(100):
            packet = bytes([random.randint(40, 60) for _ in range(128)])
            cal.add_packet(packet)
        
        cal._prepare_for_reading()
        packets = cal._read_all_packets()
        
        band = list(range(11, 23))  # Default band
        result = cal._evaluate_band(packets, band)
        
        assert 'p95' in result
        assert 'mv_values' in result
        assert 'fp_estimate' in result
        
        assert result['p95'] >= 0
        assert len(result['mv_values']) >= 0
        assert 0 <= result['fp_estimate'] <= 1
        
        cal.free_buffer()


class TestP95CalibratorFreeBuffer:
    """Test buffer cleanup"""
    
    def test_free_buffer_closes_file(self):
        """Test free_buffer closes file handle"""
        cal = P95Calibrator(buffer_size=10)
        
        packet = bytes([50, 50] * 64)
        cal.add_packet(packet)
        
        assert cal._file is not None
        
        cal.free_buffer()
        
        assert cal._file is None
    
    def test_free_buffer_removes_file(self):
        """Test free_buffer removes buffer file"""
        cal = P95Calibrator(buffer_size=10)
        
        packet = bytes([50, 50] * 64)
        cal.add_packet(packet)
        cal._file.flush()
        
        # File should exist
        assert os.path.exists(p95_calibrator.BUFFER_FILE)
        
        cal.free_buffer()
        
        # File should be removed
        assert not os.path.exists(p95_calibrator.BUFFER_FILE)


class TestConstants:
    """Test module constants"""
    
    def test_band_size(self):
        """Test band size constant"""
        assert BAND_SIZE == 12
    
    def test_mvs_window_size(self):
        """Test MVS window size"""
        assert MVS_WINDOW_SIZE == 50
    
    def test_mvs_threshold(self):
        """Test MVS threshold"""
        assert MVS_THRESHOLD == 1.0
