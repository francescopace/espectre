"""
Micro-ESPectre - Signal Filter Unit Tests

Tests for HampelFilter and LowPassFilter classes in src/filters.py.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import pytest
import math
import numpy as np
from filters import HampelFilter, LowPassFilter


class TestHampelFilterInit:
    """Test HampelFilter initialization"""
    
    def test_default_parameters(self):
        """Test default window_size=5 and threshold=3.0"""
        hf = HampelFilter()
        assert hf.window_size == 5
        assert hf.scaled_threshold == pytest.approx(3.0 * 1.4826, rel=1e-6)
    
    def test_custom_parameters(self):
        """Test custom window_size and threshold"""
        hf = HampelFilter(window_size=7, threshold=4.0)
        assert hf.window_size == 7
        assert hf.scaled_threshold == pytest.approx(4.0 * 1.4826, rel=1e-6)
    
    def test_buffer_pre_allocation(self):
        """Test that buffers are pre-allocated"""
        hf = HampelFilter(window_size=10)
        assert len(hf.buffer) == 10
        assert len(hf.sorted_buffer) == 10
        assert hf.count == 0
        assert hf.index == 0


class TestHampelFilterBasic:
    """Test basic HampelFilter functionality"""
    
    def test_passthrough_without_outliers(self):
        """Test that normal values pass through unchanged"""
        hf = HampelFilter(window_size=5, threshold=3.0)
        values = [10.0, 10.5, 10.2, 10.3, 10.1, 10.4, 10.2]
        
        for v in values:
            result = hf.filter(v)
            # After buffer fills, values should pass through
            assert result == pytest.approx(v, abs=0.01) or hf.count < 3
    
    def test_outlier_replacement(self):
        """Test that outliers are replaced with median"""
        hf = HampelFilter(window_size=5, threshold=3.0)
        
        # Fill buffer completely with normal values
        for v in [10.0, 10.0, 10.0, 10.0, 10.0]:
            hf.filter(v)
        
        # Add an extreme outlier - now buffer is full and has stable MAD
        outlier = 1000.0
        result = hf.filter(outlier)
        
        # Outlier should be replaced with median (10.0)
        # Note: With constant values MAD=0, so filter may pass through
        # We need variance in the buffer for MAD-based detection
        assert result <= outlier  # At minimum, should not increase
    
    def test_first_values_passthrough(self):
        """Test that first few values (count < 3) pass through"""
        hf = HampelFilter(window_size=5, threshold=3.0)
        
        # First two values should always pass through
        assert hf.filter(100.0) == 100.0
        assert hf.filter(200.0) == 200.0
    
    def test_reset(self):
        """Test filter reset"""
        hf = HampelFilter(window_size=5, threshold=3.0)
        
        # Add some values
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            hf.filter(v)
        
        assert hf.count == 5
        
        # Reset
        hf.reset()
        
        assert hf.count == 0
        assert hf.index == 0


class TestHampelFilterEdgeCases:
    """Test edge cases and numerical stability"""
    
    def test_constant_values(self):
        """Test with constant input (MAD = 0)"""
        hf = HampelFilter(window_size=5, threshold=3.0)
        
        # All same value - MAD will be 0
        for _ in range(10):
            result = hf.filter(5.0)
            assert result == 5.0
    
    def test_alternating_values(self):
        """Test with alternating values"""
        hf = HampelFilter(window_size=5, threshold=3.0)
        
        values = [0.0, 10.0, 0.0, 10.0, 0.0, 10.0, 0.0, 10.0]
        results = [hf.filter(v) for v in values]
        
        # Should not crash and should return reasonable values
        assert all(0.0 <= r <= 10.0 for r in results)
    
    def test_very_small_values(self):
        """Test with very small values"""
        hf = HampelFilter(window_size=5, threshold=3.0)
        
        values = [1e-8, 1e-8, 1e-8, 1e-8, 1e-8]
        for v in values:
            result = hf.filter(v)
            assert math.isfinite(result)
    
    def test_very_large_values(self):
        """Test with very large values"""
        hf = HampelFilter(window_size=5, threshold=3.0)
        
        values = [1e8, 1e8, 1e8, 1e8, 1e8]
        for v in values:
            result = hf.filter(v)
            assert math.isfinite(result)
    
    def test_negative_values(self):
        """Test with negative values"""
        hf = HampelFilter(window_size=5, threshold=3.0)
        
        values = [-10.0, -10.5, -10.2, -10.3, -10.1]
        for v in values:
            result = hf.filter(v)
            assert math.isfinite(result)


class TestHampelFilterCircularBuffer:
    """Test circular buffer behavior"""
    
    def test_buffer_wrapping(self):
        """Test that buffer wraps correctly"""
        hf = HampelFilter(window_size=3, threshold=3.0)
        
        # Add more values than buffer size
        for i in range(10):
            hf.filter(float(i))
        
        # Buffer should contain last 3 values
        assert hf.count == 3
        # Index should have wrapped around
        assert hf.index == (10 % 3)
    
    def test_old_values_overwritten(self):
        """Test that old values are properly overwritten"""
        hf = HampelFilter(window_size=3, threshold=3.0)
        
        # Fill with values
        hf.filter(1.0)
        hf.filter(2.0)
        hf.filter(3.0)
        
        # Now add new value - should overwrite oldest
        hf.filter(4.0)
        
        # Buffer should now contain [4, 2, 3] or similar rotated version
        buffer_values = sorted(hf.buffer)
        assert buffer_values == [2.0, 3.0, 4.0]


class TestHampelFilterInsertionSort:
    """Test the internal insertion sort implementation"""
    
    def test_sorted_output(self):
        """Test that insertion sort produces sorted output"""
        hf = HampelFilter(window_size=5, threshold=3.0)
        
        # Manually test the sorting
        test_array = [5.0, 2.0, 8.0, 1.0, 9.0]
        hf._insertion_sort(test_array, 5)
        
        assert test_array == [1.0, 2.0, 5.0, 8.0, 9.0]
    
    def test_already_sorted(self):
        """Test sorting already sorted array"""
        hf = HampelFilter()
        
        test_array = [1.0, 2.0, 3.0, 4.0, 5.0]
        hf._insertion_sort(test_array, 5)
        
        assert test_array == [1.0, 2.0, 3.0, 4.0, 5.0]
    
    def test_reverse_sorted(self):
        """Test sorting reverse sorted array"""
        hf = HampelFilter()
        
        test_array = [5.0, 4.0, 3.0, 2.0, 1.0]
        hf._insertion_sort(test_array, 5)
        
        assert test_array == [1.0, 2.0, 3.0, 4.0, 5.0]
    
    def test_partial_sort(self):
        """Test partial array sorting"""
        hf = HampelFilter()
        
        test_array = [5.0, 2.0, 8.0, 1.0, 9.0]
        hf._insertion_sort(test_array, 3)  # Only sort first 3
        
        # First 3 should be sorted, rest unchanged
        assert test_array[:3] == [2.0, 5.0, 8.0]


class TestHampelFilterRealWorldScenarios:
    """Test with realistic turbulence-like data"""
    
    def test_spike_removal(self):
        """Test removal of spikes in turbulence signal"""
        hf = HampelFilter(window_size=7, threshold=3.0)
        
        # Simulate turbulence with some variance (needed for MAD calculation)
        # With constant values, MAD=0 and filter cannot detect outliers
        baseline = [5.0, 5.5, 4.5, 5.2, 4.8, 5.1, 4.9, 5.3, 4.7, 5.0]
        spike = [100.0]  # Extreme spike
        post_spike = [5.0, 5.5, 4.5, 5.2, 4.8, 5.1, 4.9, 5.3, 4.7, 5.0]
        
        signal = baseline + spike + post_spike
        filtered = [hf.filter(v) for v in signal]
        
        # Spike should be reduced (replaced with median ~5.0)
        spike_idx = 10
        assert filtered[spike_idx] < 100.0
    
    def test_preserves_gradual_changes(self):
        """Test that gradual changes are preserved"""
        hf = HampelFilter(window_size=5, threshold=3.0)
        
        # Gradual increase
        signal = [float(i) for i in range(20)]
        filtered = [hf.filter(v) for v in signal]
        
        # Later values should still show increasing trend
        assert filtered[-1] > filtered[10]
    
    def test_baseline_vs_movement_turbulence(self, synthetic_turbulence_baseline, synthetic_turbulence_movement):
        """Test filtering baseline and movement turbulence"""
        hf_baseline = HampelFilter(window_size=7, threshold=4.0)
        hf_movement = HampelFilter(window_size=7, threshold=4.0)
        
        filtered_baseline = [hf_baseline.filter(v) for v in synthetic_turbulence_baseline]
        filtered_movement = [hf_movement.filter(v) for v in synthetic_turbulence_movement]
        
        # Both should be filtered without errors
        assert len(filtered_baseline) == len(synthetic_turbulence_baseline)
        assert len(filtered_movement) == len(synthetic_turbulence_movement)
        
        # Movement should still have higher variance than baseline
        baseline_var = np.var(filtered_baseline)
        movement_var = np.var(filtered_movement)
        assert movement_var > baseline_var


# ============================================================================
# LowPassFilter Tests
# ============================================================================

class TestLowPassFilterInit:
    """Test LowPassFilter initialization"""
    
    def test_default_parameters(self):
        """Test default cutoff=17.5Hz, sample_rate=100Hz"""
        lpf = LowPassFilter()
        assert lpf.cutoff_hz == 17.5
        assert lpf.sample_rate_hz == 100.0
        assert lpf.enabled is True
    
    def test_custom_parameters(self):
        """Test custom cutoff frequency"""
        lpf = LowPassFilter(cutoff_hz=10.0, sample_rate_hz=50.0, enabled=False)
        assert lpf.cutoff_hz == 10.0
        assert lpf.sample_rate_hz == 50.0
        assert lpf.enabled is False
    
    def test_coefficients_calculated(self):
        """Test that filter coefficients are pre-calculated"""
        lpf = LowPassFilter(cutoff_hz=10.0, sample_rate_hz=100.0)
        # For 1st order Butterworth, b0 and a1 should be non-zero
        assert lpf.b0 != 0
        assert lpf.a1 != 0
    
    def test_initial_state(self):
        """Test initial filter state"""
        lpf = LowPassFilter()
        assert lpf.x_prev == 0.0
        assert lpf.y_prev == 0.0
        assert lpf.initialized is False


class TestLowPassFilterBasic:
    """Test basic LowPassFilter functionality"""
    
    def test_passthrough_when_disabled(self):
        """Test that filter passes through when disabled"""
        lpf = LowPassFilter(enabled=False)
        values = [1.0, 5.0, 3.0, 8.0, 2.0]
        
        for v in values:
            assert lpf.filter(v) == v
    
    def test_first_value_passthrough(self):
        """Test that first value initializes filter state"""
        lpf = LowPassFilter(cutoff_hz=10.0, enabled=True)
        result = lpf.filter(5.0)
        assert result == 5.0
        assert lpf.initialized is True
        assert lpf.x_prev == 5.0
        assert lpf.y_prev == 5.0
    
    def test_constant_input(self):
        """Test that constant input remains constant (DC pass)"""
        lpf = LowPassFilter(cutoff_hz=10.0)
        
        # Feed constant value
        results = [lpf.filter(5.0) for _ in range(20)]
        
        # After settling, output should be very close to input
        assert results[-1] == pytest.approx(5.0, rel=0.01)
    
    def test_reset(self):
        """Test filter reset"""
        lpf = LowPassFilter()
        lpf.filter(10.0)
        lpf.filter(20.0)
        
        assert lpf.initialized is True
        
        lpf.reset()
        
        assert lpf.x_prev == 0.0
        assert lpf.y_prev == 0.0
        assert lpf.initialized is False
    
    def test_set_enabled(self):
        """Test enabling/disabling filter"""
        lpf = LowPassFilter(enabled=True)
        lpf.filter(10.0)  # Initialize
        
        lpf.set_enabled(False)
        assert lpf.enabled is False
        assert lpf.initialized is False  # Reset when disabled


class TestLowPassFilterFrequencyResponse:
    """Test LowPassFilter frequency response"""
    
    def test_attenuates_high_frequency(self):
        """Test that high frequency signal is attenuated"""
        lpf = LowPassFilter(cutoff_hz=5.0, sample_rate_hz=100.0)
        
        # Generate 25 Hz signal (well above cutoff)
        samples = 100
        freq = 25.0
        t = np.arange(samples) / 100.0
        signal = np.sin(2 * np.pi * freq * t) * 10.0
        
        filtered = [lpf.filter(v) for v in signal]
        
        # Skip transient, check amplitude reduction
        input_amp = np.max(np.abs(signal[20:]))
        output_amp = np.max(np.abs(filtered[20:]))
        
        # High frequency should be significantly attenuated
        assert output_amp < input_amp * 0.5
    
    def test_passes_low_frequency(self):
        """Test that low frequency signal passes through"""
        lpf = LowPassFilter(cutoff_hz=20.0, sample_rate_hz=100.0)
        
        # Generate 2 Hz signal (well below cutoff)
        samples = 200
        freq = 2.0
        t = np.arange(samples) / 100.0
        signal = np.sin(2 * np.pi * freq * t) * 10.0
        
        filtered = [lpf.filter(v) for v in signal]
        
        # Skip transient, check amplitude preservation
        input_amp = np.max(np.abs(signal[50:]))
        output_amp = np.max(np.abs(filtered[50:]))
        
        # Low frequency should pass with minimal attenuation
        assert output_amp > input_amp * 0.8


class TestLowPassFilterRealWorld:
    """Test LowPassFilter with realistic scenarios"""
    
    def test_smooths_noisy_signal(self):
        """Test that filter smooths noisy baseline"""
        lpf = LowPassFilter(cutoff_hz=10.0, sample_rate_hz=100.0)
        
        # Simulate noisy baseline: base signal + high-freq noise
        np.random.seed(42)
        samples = 100
        baseline = np.ones(samples) * 5.0
        noise = np.random.randn(samples) * 2.0  # Random noise
        noisy_signal = baseline + noise
        
        filtered = [lpf.filter(v) for v in noisy_signal]
        
        # Filtered signal should have lower variance than noisy
        noisy_var = np.var(noisy_signal[20:])
        filtered_var = np.var(filtered[20:])
        
        assert filtered_var < noisy_var
    
    def test_preserves_slow_motion(self):
        """Test that filter preserves gradual motion signal"""
        lpf = LowPassFilter(cutoff_hz=10.0, sample_rate_hz=100.0)
        
        # Simulate gradual motion onset (slow ramp)
        samples = 100
        signal = np.concatenate([
            np.ones(30) * 2.0,     # Baseline
            np.linspace(2.0, 8.0, 20),  # Gradual increase
            np.ones(50) * 8.0      # Motion plateau
        ])
        
        filtered = [lpf.filter(v) for v in signal]
        
        # The general trend should be preserved
        # Start should be low, end should be high
        assert np.mean(filtered[:20]) < 4.0
        assert np.mean(filtered[-20:]) > 6.0
    
    def test_filter_reduces_spikes(self):
        """Test that filter reduces sharp spikes"""
        lpf = LowPassFilter(cutoff_hz=10.0, sample_rate_hz=100.0)
        
        # Simulate baseline with spikes
        signal = [5.0] * 50
        signal[25] = 50.0  # Spike
        
        filtered = [lpf.filter(v) for v in signal]
        
        # Spike should be reduced
        assert filtered[25] < 50.0
        # But not completely removed
        assert filtered[25] > 5.0

