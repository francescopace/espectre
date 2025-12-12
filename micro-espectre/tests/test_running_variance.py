"""
Micro-ESPectre - Running Variance Tests

Converted from tools/13_test_running_variance.py.
Compares two-pass variance with Welford's running variance algorithm.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import pytest
import math
import numpy as np
from csi_utils import calculate_variance_two_pass, calculate_spatial_turbulence


class RunningVariance:
    """
    Welford's online algorithm for running variance on a sliding window.
    
    Maintains a circular buffer and calculates variance incrementally
    when a new value is added and an old value is removed.
    
    Time complexity: O(1) per update (vs O(N) for two-pass)
    """
    
    def __init__(self, window_size):
        self.window_size = window_size
        self.buffer = [0.0] * window_size
        self.buffer_index = 0
        self.count = 0
        self.sum = 0.0
        self.sum_sq = 0.0
    
    def add(self, value):
        """Add a new value to the window"""
        if self.count < self.window_size:
            self.buffer[self.buffer_index] = value
            self.sum += value
            self.sum_sq += value * value
            self.count += 1
        else:
            old_value = self.buffer[self.buffer_index]
            self.sum -= old_value
            self.sum_sq -= old_value * old_value
            self.buffer[self.buffer_index] = value
            self.sum += value
            self.sum_sq += value * value
        
        self.buffer_index = (self.buffer_index + 1) % self.window_size
    
    def get_variance(self):
        """Calculate variance from running sums: Var(X) = E[X²] - E[X]²"""
        if self.count == 0:
            return 0.0
        
        n = self.count
        mean = self.sum / n
        mean_sq = self.sum_sq / n
        variance = mean_sq - mean * mean
        return max(0.0, variance)
    
    def get_values(self):
        """Get current buffer values"""
        if self.count < self.window_size:
            return self.buffer[:self.count]
        else:
            result = []
            idx = self.buffer_index
            for _ in range(self.window_size):
                result.append(self.buffer[idx])
                idx = (idx + 1) % self.window_size
            return result


class TwoPassVariance:
    """Two-pass variance on a sliding window"""
    
    def __init__(self, window_size):
        self.window_size = window_size
        self.buffer = []
    
    def add(self, value):
        """Add a new value to the window"""
        self.buffer.append(value)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
    
    def get_variance(self):
        """Calculate variance using two-pass algorithm"""
        return calculate_variance_two_pass(self.buffer)
    
    def get_values(self):
        """Get current buffer values"""
        return list(self.buffer)


class TestSyntheticData:
    """Test with synthetic data patterns"""
    
    @pytest.fixture
    def window_size(self):
        return 100
    
    def test_constant_values(self, window_size, constant_values):
        """Test with constant values"""
        two_pass = TwoPassVariance(window_size)
        running = RunningVariance(window_size)
        
        max_diff = 0.0
        for value in constant_values:
            two_pass.add(value)
            running.add(value)
            diff = abs(two_pass.get_variance() - running.get_variance())
            max_diff = max(max_diff, diff)
        
        assert max_diff < 1e-6
    
    def test_linear_ramp(self, window_size, linear_ramp):
        """Test with linear ramp"""
        two_pass = TwoPassVariance(window_size)
        running = RunningVariance(window_size)
        
        max_diff = 0.0
        for value in linear_ramp:
            two_pass.add(value)
            running.add(value)
            diff = abs(two_pass.get_variance() - running.get_variance())
            max_diff = max(max_diff, diff)
        
        assert max_diff < 1e-6
    
    def test_sine_wave(self, window_size, sine_wave):
        """Test with sine wave"""
        two_pass = TwoPassVariance(window_size)
        running = RunningVariance(window_size)
        
        max_diff = 0.0
        for value in sine_wave:
            two_pass.add(value)
            running.add(value)
            diff = abs(two_pass.get_variance() - running.get_variance())
            max_diff = max(max_diff, diff)
        
        assert max_diff < 1e-6
    
    def test_random_uniform(self, window_size, random_uniform):
        """Test with random uniform distribution"""
        two_pass = TwoPassVariance(window_size)
        running = RunningVariance(window_size)
        
        max_diff = 0.0
        for value in random_uniform:
            two_pass.add(value)
            running.add(value)
            diff = abs(two_pass.get_variance() - running.get_variance())
            max_diff = max(max_diff, diff)
        
        assert max_diff < 1e-6
    
    def test_random_normal(self, window_size, random_normal):
        """Test with random normal distribution"""
        two_pass = TwoPassVariance(window_size)
        running = RunningVariance(window_size)
        
        max_diff = 0.0
        for value in random_normal:
            two_pass.add(value)
            running.add(value)
            diff = abs(two_pass.get_variance() - running.get_variance())
            max_diff = max(max_diff, diff)
        
        assert max_diff < 1e-6
    
    def test_step_function(self, window_size, step_function):
        """Test with step function"""
        two_pass = TwoPassVariance(window_size)
        running = RunningVariance(window_size)
        
        max_diff = 0.0
        for value in step_function:
            two_pass.add(value)
            running.add(value)
            diff = abs(two_pass.get_variance() - running.get_variance())
            max_diff = max(max_diff, diff)
        
        assert max_diff < 1e-6
    
    def test_impulse(self, window_size, impulse_data):
        """Test with impulse/spike data"""
        two_pass = TwoPassVariance(window_size)
        running = RunningVariance(window_size)
        
        max_diff = 0.0
        for value in impulse_data:
            two_pass.add(value)
            running.add(value)
            diff = abs(two_pass.get_variance() - running.get_variance())
            max_diff = max(max_diff, diff)
        
        assert max_diff < 1e-6


class TestRealCSIData:
    """Test with real CSI data (skip if not available)"""
    
    def test_real_csi_variance_equivalence(self, real_csi_data_available, 
                                            real_baseline_packets, 
                                            real_movement_packets,
                                            default_subcarriers):
        """Test variance equivalence on real CSI data"""
        if not real_csi_data_available:
            pytest.skip("Real CSI data not available")
        
        window_size = 100
        all_packets = real_baseline_packets + real_movement_packets
        
        two_pass = TwoPassVariance(window_size)
        running = RunningVariance(window_size)
        
        max_diff = 0.0
        
        for pkt in all_packets:
            turb = calculate_spatial_turbulence(pkt['csi_data'], default_subcarriers)
            two_pass.add(turb)
            running.add(turb)
            diff = abs(two_pass.get_variance() - running.get_variance())
            max_diff = max(max_diff, diff)
        
        assert max_diff < 1e-6
    
    def test_buffer_contents_match(self, real_csi_data_available,
                                    real_baseline_packets,
                                    real_movement_packets,
                                    default_subcarriers):
        """Test that buffer contents match at the end"""
        if not real_csi_data_available:
            pytest.skip("Real CSI data not available")
        
        window_size = 100
        all_packets = real_baseline_packets + real_movement_packets
        
        two_pass = TwoPassVariance(window_size)
        running = RunningVariance(window_size)
        
        for pkt in all_packets:
            turb = calculate_spatial_turbulence(pkt['csi_data'], default_subcarriers)
            two_pass.add(turb)
            running.add(turb)
        
        tp_vals = two_pass.get_values()
        run_vals = running.get_values()
        
        assert len(tp_vals) == len(run_vals)
        for a, b in zip(tp_vals, run_vals):
            assert abs(a - b) < 1e-10


class TestDetectionEquivalence:
    """Test that both methods produce identical detection results"""
    
    def test_state_machine_equivalence(self, real_csi_data_available,
                                        real_baseline_packets,
                                        real_movement_packets,
                                        default_subcarriers):
        """Test that state transitions are identical"""
        if not real_csi_data_available:
            pytest.skip("Real CSI data not available")
        
        window_size = 100
        threshold = 0.5
        all_packets = real_baseline_packets + real_movement_packets
        
        two_pass = TwoPassVariance(window_size)
        running = RunningVariance(window_size)
        
        state_mismatches = 0
        
        for pkt in all_packets:
            turb = calculate_spatial_turbulence(pkt['csi_data'], default_subcarriers)
            two_pass.add(turb)
            running.add(turb)
            
            state_tp = 'MOTION' if two_pass.get_variance() > threshold else 'IDLE'
            state_run = 'MOTION' if running.get_variance() > threshold else 'IDLE'
            
            if state_tp != state_run:
                state_mismatches += 1
        
        assert state_mismatches == 0


class TestNumericalStability:
    """Test numerical stability with extreme values"""
    
    @pytest.fixture
    def window_size(self):
        return 100
    
    def test_very_small_values(self, window_size):
        """Test with very small values"""
        data = [1e-8 + i * 1e-10 for i in range(500)]
        
        two_pass = TwoPassVariance(window_size)
        running = RunningVariance(window_size)
        
        for value in data:
            two_pass.add(value)
            running.add(value)
            
            assert math.isfinite(running.get_variance())
    
    def test_very_large_values(self, window_size):
        """Test with very large values"""
        data = [1e8 + i * 1e6 for i in range(500)]
        
        two_pass = TwoPassVariance(window_size)
        running = RunningVariance(window_size)
        
        for value in data:
            two_pass.add(value)
            running.add(value)
            
            assert math.isfinite(running.get_variance())
    
    def test_mixed_scale(self, window_size):
        """Test with mixed scale values"""
        data = [1e-5] * 100 + [1e5] * 100 + [1e-5] * 300
        
        two_pass = TwoPassVariance(window_size)
        running = RunningVariance(window_size)
        
        for value in data:
            two_pass.add(value)
            running.add(value)
            
            # Running variance might have numerical issues here
            # but should still be finite
            assert math.isfinite(running.get_variance())
    
    def test_near_constant(self, window_size):
        """Test with near-constant values"""
        data = [100.0 + i * 1e-12 for i in range(500)]
        
        two_pass = TwoPassVariance(window_size)
        running = RunningVariance(window_size)
        
        for value in data:
            two_pass.add(value)
            running.add(value)
            
            # Should not produce negative variance
            assert running.get_variance() >= 0


class TestPerformance:
    """Performance comparison tests"""
    
    def test_running_faster_than_two_pass(self):
        """Verify running variance is faster than two-pass"""
        import time
        
        window_size = 100
        num_values = 10000
        np.random.seed(42)
        data = list(np.random.normal(50, 15, num_values))
        
        # Two-pass timing
        two_pass = TwoPassVariance(window_size)
        start = time.perf_counter()
        for value in data:
            two_pass.add(value)
            _ = two_pass.get_variance()
        time_tp = time.perf_counter() - start
        
        # Running variance timing
        running = RunningVariance(window_size)
        start = time.perf_counter()
        for value in data:
            running.add(value)
            _ = running.get_variance()
        time_run = time.perf_counter() - start
        
        # Running should be faster (O(1) vs O(N))
        # Allow some margin for test environment variability
        speedup = time_tp / time_run
        assert speedup > 1.0  # At least not slower

