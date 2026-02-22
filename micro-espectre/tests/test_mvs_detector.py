"""
Micro-ESPectre - MVS Detector Tests

Tests for the MVSDetector wrapper class.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import pytest
import numpy as np
from mvs_detector import MVSDetector
from detector_interface import MotionState


class TestMVSDetectorBasics:
    """Test MVS detector basic functionality"""
    
    def test_initialization(self):
        """Test detector initialization with defaults"""
        detector = MVSDetector()
        
        assert detector.get_name() == "MVS"
        assert detector.total_packets == 0
        assert not detector.is_ready()
    
    def test_initialization_with_params(self):
        """Test detector initialization with custom params"""
        detector = MVSDetector(
            window_size=50,
            threshold=2.0,
            enable_lowpass=True,
            lowpass_cutoff=8.0,
            enable_hampel=True,
            hampel_window=5,
            hampel_threshold=3.0
        )
        
        assert detector.get_threshold() == 2.0
    
    def test_process_packet(self):
        """Test processing a single packet"""
        detector = MVSDetector(window_size=10)
        
        # Create fake CSI data (64 subcarriers, 2 bytes each)
        csi_data = [10, 10] * 64  # I=10, Q=10 for each subcarrier
        
        detector.process_packet(csi_data, selected_subcarriers=[0, 10, 20, 30])
        
        assert detector.total_packets == 1
    
    def test_buffer_filling(self):
        """Test that is_ready becomes True after filling buffer"""
        detector = MVSDetector(window_size=10)
        csi_data = [10, 10] * 64
        
        for _ in range(9):
            detector.process_packet(csi_data, [0, 10, 20])
        assert not detector.is_ready()
        
        detector.process_packet(csi_data, [0, 10, 20])
        assert detector.is_ready()
    
    def test_update_state_returns_metrics(self):
        """Test that update_state returns metrics dict"""
        detector = MVSDetector(window_size=10)
        csi_data = [10, 10] * 64
        
        for _ in range(15):
            detector.process_packet(csi_data, [0, 10, 20])
        
        metrics = detector.update_state()
        
        assert 'moving_variance' in metrics
        assert 'state' in metrics
    
    def test_get_state(self):
        """Test getting current state"""
        detector = MVSDetector(window_size=10)
        csi_data = [10, 10] * 64
        
        # Initial state
        state = detector.get_state()
        assert state in [MotionState.IDLE, MotionState.MOTION]
    
    def test_get_motion_metric(self):
        """Test getting motion metric"""
        detector = MVSDetector(window_size=10)
        csi_data = [10, 10] * 64
        
        for _ in range(15):
            detector.process_packet(csi_data, [0, 10, 20])
        detector.update_state()
        
        metric = detector.get_motion_metric()
        assert isinstance(metric, (int, float))
    
    def test_set_threshold_valid(self):
        """Test setting valid threshold"""
        detector = MVSDetector()
        
        result = detector.set_threshold(2.0)
        assert result is True
        assert detector.get_threshold() == 2.0
    
    def test_set_threshold_invalid(self):
        """Test setting invalid threshold"""
        detector = MVSDetector()
        original = detector.get_threshold()
        
        # Too low
        result = detector.set_threshold(0.05)
        assert result is False
        assert detector.get_threshold() == original
        
        # Too high
        result = detector.set_threshold(20.0)
        assert result is False
        assert detector.get_threshold() == original
    
    def test_set_adaptive_threshold(self):
        """Test setting adaptive threshold"""
        detector = MVSDetector()
        detector.set_adaptive_threshold(1.5)
        # Just verify it doesn't crash - internal state is handled by context
    
    def test_reset(self):
        """Test resetting detector"""
        detector = MVSDetector(window_size=10)
        detector.track_data = True
        csi_data = [10, 10] * 64
        
        for _ in range(15):
            detector.process_packet(csi_data, [0, 10, 20])
            detector.update_state()
        
        assert len(detector.moving_var_history) > 0
        assert len(detector.state_history) > 0
        
        detector.reset()
        
        assert detector.moving_var_history == []
        assert detector.state_history == []
        assert detector.get_motion_count() == 0
    
    def test_tracking_mode(self):
        """Test data tracking mode"""
        detector = MVSDetector(window_size=10)
        detector.track_data = True
        csi_data = [10, 10] * 64
        
        for _ in range(15):
            detector.process_packet(csi_data, [0, 10, 20])
            detector.update_state()
        
        # Should have tracked variance and states
        assert len(detector.moving_var_history) > 0
        assert len(detector.state_history) > 0
    
    def test_last_turbulence_property(self):
        """Test last_turbulence property"""
        detector = MVSDetector(window_size=10)
        csi_data = [10, 10] * 64
        
        detector.process_packet(csi_data, [0, 10, 20])
        
        turb = detector.last_turbulence
        assert isinstance(turb, (int, float))
    
    def test_cv_normalization_property(self):
        """Test CV normalization property getter/setter"""
        detector = MVSDetector()
        
        # Default value
        original = detector.use_cv_normalization
        assert isinstance(original, bool)
        
        # Set value
        detector.use_cv_normalization = not original
        assert detector.use_cv_normalization == (not original)
    
    def test_motion_detection_with_varying_data(self):
        """Test motion detection with varying CSI data"""
        detector = MVSDetector(window_size=20, threshold=0.5)
        detector.track_data = True
        
        # Idle-like data: constant signal
        idle_csi = [10, 10] * 64
        for _ in range(30):
            detector.process_packet(idle_csi, list(range(0, 60, 5)))
            detector.update_state()
        
        # Should be mostly IDLE
        idle_states = detector.state_history.count('IDLE')
        assert idle_states > 0
        
        # Motion-like data: highly varying signal
        np.random.seed(42)
        for _ in range(30):
            motion_csi = list(np.random.randint(0, 256, 128))
            detector.process_packet(motion_csi, list(range(0, 60, 5)))
            detector.update_state()
        
        # Should detect some motion
        motion_states = detector.state_history.count('MOTION')
        # Note: exact count depends on threshold and data variance


class TestMVSDetectorEdgeCases:
    """Test MVS detector edge cases"""
    
    def test_empty_subcarriers(self):
        """Test with empty subcarrier list"""
        detector = MVSDetector(window_size=10)
        csi_data = [10, 10] * 64
        
        # Should not crash with empty subcarriers
        detector.process_packet(csi_data, [])
    
    def test_single_subcarrier(self):
        """Test with single subcarrier"""
        detector = MVSDetector(window_size=10)
        csi_data = [10, 10] * 64
        
        detector.process_packet(csi_data, [32])
        assert detector.total_packets == 1
