"""
Tests for Band Calibrator - P95 Moving Variance Optimization

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
import band_calibrator
band_calibrator.BUFFER_FILE = os.path.join(tempfile.gettempdir(), 'band_test.bin')

from band_calibrator import (
    BandCalibrator,
    ADAPTIVE_THRESHOLD_FACTOR,
    BAND_SIZE, MVS_WINDOW_SIZE, MVS_THRESHOLD
)
from src.config import GUARD_BAND_LOW, GUARD_BAND_HIGH, DC_SUBCARRIER


class TestHT20Constants:
    """Test HT20-only constants (64 subcarriers)"""
    
    def test_guard_bands(self):
        """Test HT20 guard band constants"""
        assert GUARD_BAND_LOW == 11
        assert GUARD_BAND_HIGH == 52
        assert DC_SUBCARRIER == 32


class TestAdaptiveThresholdConstants:
    """Test adaptive threshold constants"""
    
    def test_factor_value(self):
        """Test default factor for P95 × factor formula"""
        assert ADAPTIVE_THRESHOLD_FACTOR == 1.4


class TestBandCalibratorInit:
    """Test BandCalibrator initialization"""
    
    def test_default_init(self):
        """Test default initialization"""
        cal = BandCalibrator()
        
        assert cal.buffer_size == 700
        assert cal._packet_count == 0
        assert cal._filtered_count == 0
        
        cal.free_buffer()
    
    def test_custom_buffer_size(self):
        """Test custom buffer size"""
        cal = BandCalibrator(buffer_size=500)
        
        assert cal.buffer_size == 500
        
        cal.free_buffer()


class TestBandCalibratorAddPacket:
    """Test packet addition"""
    
    def test_add_packet_returns_count(self):
        """Test add_packet returns progress count"""
        cal = BandCalibrator(buffer_size=10)
        
        # Create fake 64 SC packet (128 bytes I/Q)
        packet = bytes([50, 50] * 64)
        
        count = cal.add_packet(packet)
        assert count == 1
        
        count = cal.add_packet(packet)
        assert count == 2
        
        cal.free_buffer()
    
    def test_add_packet_stops_at_buffer_size(self):
        """Test buffer stops at max size"""
        cal = BandCalibrator(buffer_size=5)
        
        packet = bytes([50, 50] * 64)
        
        for i in range(10):
            count = cal.add_packet(packet)
        
        assert count == 5  # Stopped at buffer_size
        assert cal._packet_count == 5
        
        cal.free_buffer()
    
    def test_filter_wrong_sc_count(self):
        """Test packets with wrong SC count are filtered (HT20: 64 SC only)"""
        cal = BandCalibrator(buffer_size=10)
        
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
        cal = BandCalibrator(buffer_size=10)
        
        packet = bytes([50, 50] * 64)
        count = cal.add_packet(packet)
        assert count == 1
        
        cal.free_buffer()


class TestBandCalibratorHelpers:
    """Test helper methods"""
    
    def test_calculate_spatial_turbulence(self):
        """Test spatial turbulence calculation"""
        cal = BandCalibrator(buffer_size=10)
        
        # Constant magnitudes -> 0 turbulence
        packet = [50] * 64
        turb = cal._calculate_spatial_turbulence(packet, list(range(12)))
        assert turb == 0.0
        
        # Variable magnitudes -> non-zero turbulence
        packet = list(range(64))
        turb = cal._calculate_spatial_turbulence(packet, list(range(12)))
        assert turb > 0
        
        cal.free_buffer()
    
    def test_calculate_moving_variance(self):
        """Test moving variance calculation"""
        cal = BandCalibrator(buffer_size=10)
        
        # Constant values -> 0 variance
        values = [1.0] * 100
        mv = cal._calculate_moving_variance(values, window_size=50)
        assert len(mv) == 50  # 100 - 50
        assert all(v == 0.0 for v in mv)
        
        # Variable values -> non-zero variance
        values = list(range(100))
        mv = cal._calculate_moving_variance(values, window_size=50)
        assert len(mv) == 50
        assert all(v > 0 for v in mv)
        
        cal.free_buffer()
    
    def test_calculate_p95(self):
        """Test P95 calculation"""
        cal = BandCalibrator(buffer_size=10)
        
        # Known values
        values = list(range(100))  # 0-99
        p95 = cal._calculate_p95(values)
        # P95 of 0-99 should be around 94-95
        assert 94 <= p95 <= 95
        
        # Empty list
        p95_empty = cal._calculate_p95([])
        assert p95_empty == float('inf')
        
        cal.free_buffer()
    
    def test_get_candidate_bands(self):
        """Test candidate band generation for HT20 (64 SC)"""
        cal = BandCalibrator(buffer_size=10)
        
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
    


class TestBandCalibratorCalibration:
    """Test calibration process"""
    
    def test_calibration_insufficient_packets(self):
        """Test calibration fails with insufficient packets"""
        cal = BandCalibrator(buffer_size=100)
        
        # Add only a few packets
        packet = bytes([50, 50] * 64)
        for _ in range(10):
            cal.add_packet(packet)
        
        band, scale = cal.calibrate()
        
        assert band is None
        assert scale == 1.0
        
        cal.free_buffer()
    
    def test_calibration_returns_valid_band(self):
        """Test calibration returns 12 valid subcarriers"""
        cal = BandCalibrator(buffer_size=100)
        
        # Add enough packets with some variation
        import random
        random.seed(42)
        
        for _ in range(100):
            # Random I/Q values
            packet = bytes([random.randint(30, 80) for _ in range(128)])
            cal.add_packet(packet)
        
        band, scale = cal.calibrate()
        
        if band is not None:
            assert len(band) == 12
            # All subcarriers should be in valid range
            for sc in band:
                assert 11 <= sc <= 52
        
        cal.free_buffer()
    
    def test_calibration_adaptive_threshold(self):
        """Test adaptive threshold is valid"""
        cal = BandCalibrator(buffer_size=100)
        
        import random
        random.seed(42)
        
        for _ in range(100):
            packet = bytes([random.randint(30, 80) for _ in range(128)])
            cal.add_packet(packet)
        
        band, adaptive_threshold = cal.calibrate()
        
        # Adaptive threshold should be positive and reasonable
        assert adaptive_threshold > 0
        assert adaptive_threshold <= 10.0
        
        cal.free_buffer()


class TestBandCalibratorEvaluateBand:
    """Test band evaluation"""
    
    def test_evaluate_band_returns_metrics(self):
        """Test band evaluation returns expected metrics"""
        cal = BandCalibrator(buffer_size=100)
        
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
        assert 'mean_mv' in result
        assert 'fp_estimate' in result
        
        assert result['p95'] >= 0
        assert result['mean_mv'] >= 0
        assert 0 <= result['fp_estimate'] <= 1
        
        cal.free_buffer()


class TestBandCalibratorFreeBuffer:
    """Test buffer cleanup"""
    
    def test_free_buffer_closes_file(self):
        """Test free_buffer closes file handle"""
        cal = BandCalibrator(buffer_size=10)
        
        packet = bytes([50, 50] * 64)
        cal.add_packet(packet)
        
        assert cal._file is not None
        
        cal.free_buffer()
        
        assert cal._file is None
    
    def test_free_buffer_removes_file(self):
        """Test free_buffer removes buffer file"""
        cal = BandCalibrator(buffer_size=10)
        
        packet = bytes([50, 50] * 64)
        cal.add_packet(packet)
        cal._file.flush()
        
        # File should exist
        assert os.path.exists(band_calibrator.BUFFER_FILE)
        
        cal.free_buffer()
        
        # File should be removed
        assert not os.path.exists(band_calibrator.BUFFER_FILE)


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
    
    def test_adaptive_threshold_factor(self):
        """Test adaptive threshold factor (P95 × factor)"""
        assert ADAPTIVE_THRESHOLD_FACTOR == 1.4
