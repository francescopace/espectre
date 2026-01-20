"""
Micro-ESPectre - Segmentation Unit Tests

Tests for the SegmentationContext class in src/segmentation.py.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import pytest
import math
import numpy as np
from segmentation import SegmentationContext


class TestSegmentationContextInit:
    """Test SegmentationContext initialization"""
    
    def test_default_parameters(self):
        """Test default parameters"""
        ctx = SegmentationContext()
        assert ctx.window_size == 50
        assert ctx.threshold == 1.0
        assert ctx.state == SegmentationContext.STATE_IDLE
        assert ctx.buffer_count == 0
    
    def test_custom_parameters(self):
        """Test custom parameters"""
        ctx = SegmentationContext(
            window_size=100,
            threshold=2.5,
            enable_hampel=False
        )
        assert ctx.window_size == 100
        assert ctx.threshold == 2.5
    
    def test_buffer_pre_allocation(self):
        """Test that turbulence buffer is pre-allocated"""
        ctx = SegmentationContext(window_size=75)
        assert len(ctx.turbulence_buffer) == 75
    
    def test_hampel_disabled_by_default(self):
        """Test that Hampel filter is disabled by default"""
        ctx = SegmentationContext()
        assert ctx.hampel_filter is None
    
    def test_hampel_enabled(self):
        """Test Hampel filter initialization when enabled"""
        ctx = SegmentationContext(
            enable_hampel=True,
            hampel_window=5,
            hampel_threshold=3.0
        )
        assert ctx.hampel_filter is not None
    
    def test_lowpass_disabled_by_default(self):
        """Test that low-pass filter is disabled by default"""
        ctx = SegmentationContext()
        assert ctx.lowpass_filter is None
    
    def test_lowpass_enabled(self):
        """Test low-pass filter initialization when enabled"""
        ctx = SegmentationContext(
            enable_lowpass=True,
            lowpass_cutoff=11.5
        )
        assert ctx.lowpass_filter is not None
        assert ctx.lowpass_filter.cutoff_hz == 11.5


class TestComputeVarianceTwoPass:
    """Test the static two-pass variance calculation"""
    
    def test_empty_list(self):
        """Test variance of empty list"""
        result = SegmentationContext.compute_variance_two_pass([])
        assert result == 0.0
    
    def test_single_value(self):
        """Test variance of single value"""
        result = SegmentationContext.compute_variance_two_pass([5.0])
        assert result == 0.0
    
    def test_constant_values(self):
        """Test variance of constant values"""
        result = SegmentationContext.compute_variance_two_pass([10.0] * 100)
        assert result == pytest.approx(0.0, abs=1e-10)
    
    def test_known_variance(self):
        """Test with known variance"""
        # Values 1, 2, 3, 4, 5 have variance = 2.0
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = SegmentationContext.compute_variance_two_pass(values)
        assert result == pytest.approx(2.0, rel=1e-6)
    
    def test_matches_numpy(self):
        """Test that result matches numpy variance"""
        np.random.seed(42)
        values = list(np.random.normal(50, 15, 100))
        
        result = SegmentationContext.compute_variance_two_pass(values)
        expected = np.var(values)
        
        assert result == pytest.approx(expected, rel=1e-6)


class TestComputeSpatialTurbulence:
    """Test the static spatial turbulence calculation"""
    
    def test_empty_data(self):
        """Test with empty CSI data"""
        turb, amps = SegmentationContext.compute_spatial_turbulence([])
        assert turb == 0.0
        assert amps == []
    
    def test_minimal_data(self):
        """Test with minimal CSI data"""
        turb, amps = SegmentationContext.compute_spatial_turbulence([0])
        assert turb == 0.0
    
    def test_single_subcarrier(self):
        """Test with single subcarrier (I, Q)"""
        # I=3, Q=4 -> amplitude = 5
        turb, amps = SegmentationContext.compute_spatial_turbulence([3, 4])
        assert len(amps) == 1
        assert amps[0] == pytest.approx(5.0, rel=1e-6)
    
    def test_multiple_subcarriers(self):
        """Test with multiple subcarriers"""
        # 4 subcarriers with I/Q pairs
        csi_data = [3, 4, 6, 8, 5, 12, 8, 15]  # Amplitudes: 5, 10, 13, 17
        turb, amps = SegmentationContext.compute_spatial_turbulence(csi_data)
        
        assert len(amps) == 4
        assert amps[0] == pytest.approx(5.0, rel=1e-6)
        assert amps[1] == pytest.approx(10.0, rel=1e-6)
    
    def test_selected_subcarriers(self):
        """Test with selected subcarriers only"""
        # 8 subcarriers, select only indices 0, 2, 3
        csi_data = [3, 4, 6, 8, 5, 12, 8, 15, 0, 0, 0, 0, 0, 0, 0, 0]
        selected = [0, 2, 3]
        
        turb, amps = SegmentationContext.compute_spatial_turbulence(csi_data, selected)
        
        assert len(amps) == 3
    
    def test_turbulence_is_std(self):
        """Test that turbulence equals standard deviation of amplitudes"""
        # Create data with known std
        # I=10, Q=0 for all -> all amplitudes = 10
        csi_data = [10, 0] * 10
        turb, amps = SegmentationContext.compute_spatial_turbulence(csi_data)
        
        # All amplitudes equal -> std = 0
        assert turb == pytest.approx(0.0, abs=1e-6)


class TestAddTurbulence:
    """Test the add_turbulence method and state machine"""
    
    def test_buffer_filling(self):
        """Test that buffer fills correctly"""
        ctx = SegmentationContext(window_size=10)
        
        for i in range(10):
            ctx.add_turbulence(float(i))
        
        assert ctx.buffer_count == 10
    
    def test_variance_zero_before_full(self):
        """Test variance is 0 before buffer is full"""
        ctx = SegmentationContext(window_size=10)
        
        for i in range(5):
            ctx.add_turbulence(float(i))
        
        assert ctx.current_moving_variance == 0.0
    
    def test_variance_after_full(self):
        """Test variance is calculated after buffer is full"""
        ctx = SegmentationContext(window_size=5)
        
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            ctx.add_turbulence(v)
        
        # Lazy evaluation: must call update_state() to calculate variance
        ctx.update_state()
        
        # Variance should be 2.0 for [1,2,3,4,5]
        assert ctx.current_moving_variance == pytest.approx(2.0, rel=1e-6)
    
    def test_circular_buffer(self):
        """Test circular buffer behavior"""
        ctx = SegmentationContext(window_size=5)
        
        # Fill with initial values
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            ctx.add_turbulence(v)
        
        # Add more values - should overwrite oldest
        for v in [6.0, 7.0]:
            ctx.add_turbulence(v)
        
        # Buffer should now contain [6, 7, 3, 4, 5] in some order
        assert ctx.buffer_count == 5


class TestStateMachine:
    """Test motion detection state machine"""
    
    def test_initial_state_idle(self):
        """Test initial state is IDLE"""
        ctx = SegmentationContext()
        assert ctx.get_state() == SegmentationContext.STATE_IDLE
    
    def test_transition_to_motion(self):
        """Test transition from IDLE to MOTION"""
        ctx = SegmentationContext(window_size=5, threshold=1.0)
        
        # Add high-variance values
        for v in [1.0, 10.0, 1.0, 10.0, 1.0]:
            ctx.add_turbulence(v)
        
        # Lazy evaluation: must call update_state() to calculate variance and update state
        ctx.update_state()
        
        # Variance should be high -> MOTION
        assert ctx.current_moving_variance > 1.0
        assert ctx.get_state() == SegmentationContext.STATE_MOTION
    
    def test_transition_to_idle(self):
        """Test transition from MOTION to IDLE"""
        ctx = SegmentationContext(window_size=5, threshold=1.0)
        
        # First create MOTION state
        for v in [1.0, 10.0, 1.0, 10.0, 1.0]:
            ctx.add_turbulence(v)
        
        ctx.update_state()
        assert ctx.get_state() == SegmentationContext.STATE_MOTION
        
        # Now add low-variance values
        for v in [5.0, 5.0, 5.0, 5.0, 5.0]:
            ctx.add_turbulence(v)
        
        ctx.update_state()
        # Variance should be low -> IDLE
        assert ctx.get_state() == SegmentationContext.STATE_IDLE
    
    def test_stays_idle_with_low_variance(self):
        """Test that state stays IDLE with low variance"""
        ctx = SegmentationContext(window_size=5, threshold=1.0)
        
        # Add constant values
        for _ in range(20):
            ctx.add_turbulence(5.0)
        
        ctx.update_state()  # Lazy evaluation: must call to update state
        assert ctx.get_state() == SegmentationContext.STATE_IDLE


class TestGetMetrics:
    """Test get_metrics method"""
    
    def test_metrics_structure(self):
        """Test that metrics dict has expected keys"""
        ctx = SegmentationContext()
        metrics = ctx.get_metrics()
        
        assert 'moving_variance' in metrics
        assert 'threshold' in metrics
        assert 'turbulence' in metrics
        assert 'state' in metrics
    
    def test_metrics_values(self):
        """Test that metrics reflect current state"""
        ctx = SegmentationContext(threshold=2.5)
        ctx.add_turbulence(10.0)
        
        metrics = ctx.get_metrics()
        
        assert metrics['threshold'] == 2.5
        assert metrics['turbulence'] == 10.0


class TestReset:
    """Test reset functionality"""
    
    def test_soft_reset(self):
        """Test soft reset (keep buffer)"""
        ctx = SegmentationContext(window_size=5)
        
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            ctx.add_turbulence(v)
        
        ctx.reset(full=False)
        
        assert ctx.state == SegmentationContext.STATE_IDLE
        assert ctx.packet_index == 0
        # Buffer should still have data
        assert ctx.buffer_count == 5
    
    def test_full_reset(self):
        """Test full reset (clear buffer)"""
        ctx = SegmentationContext(window_size=5)
        
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            ctx.add_turbulence(v)
        
        ctx.reset(full=True)
        
        assert ctx.state == SegmentationContext.STATE_IDLE
        assert ctx.packet_index == 0
        assert ctx.buffer_count == 0
        assert ctx.current_moving_variance == 0.0


class TestAdaptiveThreshold:
    """Test adaptive threshold functionality"""
    
    def test_set_adaptive_threshold(self):
        """Test setting adaptive threshold"""
        ctx = SegmentationContext()
        ctx.set_adaptive_threshold(2.0)
        
        assert ctx.threshold == 2.0
    
    def test_adaptive_threshold_clamping(self):
        """Test that adaptive threshold is clamped"""
        ctx = SegmentationContext()
        
        ctx.set_adaptive_threshold(0.01)  # Too low
        assert ctx.threshold == 0.1
        
        ctx.set_adaptive_threshold(100.0)  # Too high
        assert ctx.threshold == 10.0
    
    def test_no_normalization_applied(self):
        """Test that turbulence is NOT normalized (adaptive threshold approach)"""
        ctx = SegmentationContext(window_size=5, threshold=2.0)
        
        # Add turbulence - should NOT be scaled (adaptive threshold doesn't normalize)
        ctx.add_turbulence(5.0)
        
        # last_turbulence should be 5.0 (no normalization)
        assert ctx.last_turbulence == pytest.approx(5.0, rel=1e-6)
    
class TestHampelIntegration:
    """Test integration with Hampel filter"""
    
    def test_hampel_filters_outliers(self):
        """Test that Hampel filter removes outliers"""
        ctx = SegmentationContext(
            window_size=10,
            enable_hampel=True,
            hampel_window=5,
            hampel_threshold=3.0
        )
        
        # Add values with some variance (needed for MAD calculation)
        for v in [5.0, 5.5, 4.5, 5.2, 4.8, 5.1, 4.9]:
            ctx.add_turbulence(v)
        
        # Add extreme outlier
        ctx.add_turbulence(100.0)
        
        # Outlier should be filtered (replaced with median ~5.0)
        assert ctx.last_turbulence < 100.0


class TestLowPassIntegration:
    """Test integration with low-pass filter"""
    
    def test_lowpass_smooths_signal(self):
        """Test that low-pass filter smooths high-frequency noise"""
        import numpy as np
        
        ctx = SegmentationContext(
            window_size=50,
            enable_lowpass=True,
            lowpass_cutoff=10.0
        )
        
        # Generate noisy signal: base + high-freq noise
        np.random.seed(42)
        baseline = 5.0
        noise = np.random.randn(50) * 2.0
        signal = baseline + noise
        
        for v in signal:
            ctx.add_turbulence(v)
        
        # Variance of last_turbulence should be lower than raw input variance
        # due to smoothing (we can only check it doesn't follow noise exactly)
        # The filtered value should be closer to baseline than the noisy input
        assert 3.0 < ctx.last_turbulence < 7.0  # Should be smoothed toward baseline
    
    def test_lowpass_preserves_dc(self):
        """Test that low-pass filter preserves DC component"""
        ctx = SegmentationContext(
            window_size=50,
            enable_lowpass=True,
            lowpass_cutoff=10.0
        )
        
        # Feed constant value
        for _ in range(30):
            ctx.add_turbulence(5.0)
        
        # Should pass through unchanged
        assert ctx.last_turbulence == pytest.approx(5.0, rel=0.01)
    
    def test_filter_chain_order(self):
        """Test that filter chain applies: hampel → lowpass (no normalization)"""
        ctx = SegmentationContext(
            window_size=10,
            enable_lowpass=True,
            lowpass_cutoff=10.0,
            enable_hampel=True,
            hampel_window=5,
            hampel_threshold=3.0
        )
        
        # Feed values to initialize filter
        for v in [3.0, 3.1, 2.9, 3.0, 3.2]:
            ctx.add_turbulence(v)
        
        # Feed normal value (no normalization, just filtering)
        ctx.add_turbulence(3.0)
        
        # Output should be around 3.0 (slightly smoothed by lowpass)
        assert 2.5 < ctx.last_turbulence < 3.5


class TestCalculateSpatialTurbulence:
    """Test the instance method calculate_spatial_turbulence"""
    
    def test_stores_amplitudes(self, synthetic_csi_packet, default_subcarriers):
        """Test that amplitudes are stored for feature calculation"""
        ctx = SegmentationContext()
        
        turb = ctx.calculate_spatial_turbulence(synthetic_csi_packet, default_subcarriers)
        
        assert ctx.last_amplitudes is not None
        assert len(ctx.last_amplitudes) == len(default_subcarriers)


class TestEndToEnd:
    """End-to-end integration tests"""
    
    def test_baseline_detection(self, synthetic_csi_baseline_packets, default_subcarriers):
        """Test that baseline packets produce IDLE state"""
        ctx = SegmentationContext(window_size=50, threshold=1.0)
        
        motion_count = 0
        for pkt in synthetic_csi_baseline_packets:
            turb = ctx.calculate_spatial_turbulence(pkt['csi_data'], default_subcarriers)
            ctx.add_turbulence(turb)
            ctx.update_state()  # Lazy evaluation: must call to update state
            if ctx.get_state() == SegmentationContext.STATE_MOTION:
                motion_count += 1
        
        # Most packets should be IDLE for baseline
        motion_rate = motion_count / len(synthetic_csi_baseline_packets)
        assert motion_rate < 0.5  # Less than 50% should be motion
    
    def test_movement_detection(self, synthetic_csi_movement_packets, default_subcarriers):
        """Test that movement packets produce MOTION state"""
        ctx = SegmentationContext(window_size=50, threshold=1.0)
        
        motion_count = 0
        for pkt in synthetic_csi_movement_packets:
            turb = ctx.calculate_spatial_turbulence(pkt['csi_data'], default_subcarriers)
            ctx.add_turbulence(turb)
            ctx.update_state()  # Lazy evaluation: must call to update state
            if ctx.get_state() == SegmentationContext.STATE_MOTION:
                motion_count += 1
        
        # After warmup, many packets should be MOTION for movement
        # (Allow warmup period of window_size packets)
        warmup = 50
        if len(synthetic_csi_movement_packets) > warmup:
            motion_rate = motion_count / (len(synthetic_csi_movement_packets) - warmup)
            # Should detect some motion
            assert motion_rate > 0.1  # At least 10% motion

