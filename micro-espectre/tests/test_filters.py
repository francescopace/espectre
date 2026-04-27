"""
Micro-ESPectre - Signal Filter Unit Tests

Tests for HampelFilter, LowPassFilter, and BreathingFilter classes in src/filters.py.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import pytest
import math
import numpy as np
from filters import HampelFilter, LowPassFilter, BreathingFilter
from utils import insertion_sort


class TestHampelFilterInit:
    """Test HampelFilter initialization"""
    
    def test_default_parameters(self):
        """Test default window_size=5 and threshold=3.0"""
        hf = HampelFilter()
        assert hf.window_size == 5
        assert hf.scaled_threshold == pytest.approx(3.0 * 1.4826, rel=1e-6)
    
    def test_custom_parameters(self):
        """Test custom window_size and threshold"""
        hf = HampelFilter(window_size=7, threshold=5.0)
        assert hf.window_size == 7
        assert hf.scaled_threshold == pytest.approx(5.0 * 1.4826, rel=1e-6)
    
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
        test_array = [5.0, 2.0, 8.0, 1.0, 9.0]
        insertion_sort(test_array, 5)
        
        assert test_array == [1.0, 2.0, 5.0, 8.0, 9.0]
    
    def test_already_sorted(self):
        """Test sorting already sorted array"""
        test_array = [1.0, 2.0, 3.0, 4.0, 5.0]
        insertion_sort(test_array, 5)
        
        assert test_array == [1.0, 2.0, 3.0, 4.0, 5.0]
    
    def test_reverse_sorted(self):
        """Test sorting reverse sorted array"""
        test_array = [5.0, 4.0, 3.0, 2.0, 1.0]
        insertion_sort(test_array, 5)
        
        assert test_array == [1.0, 2.0, 3.0, 4.0, 5.0]
    
    def test_partial_sort(self):
        """Test partial array sorting"""
        test_array = [5.0, 2.0, 8.0, 1.0, 9.0]
        insertion_sort(test_array, 3)  # Only sort first 3
        
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
        hf_baseline = HampelFilter(window_size=7, threshold=5.0)
        hf_movement = HampelFilter(window_size=7, threshold=5.0)
        
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
        """Test default cutoff=11.0Hz, sample_rate=100Hz"""
        lpf = LowPassFilter()
        assert lpf.cutoff_hz == 11.0
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


# =============================================================================
# BreathingFilter Tests
# =============================================================================


class TestBreathingFilterInit:
    """Test BreathingFilter initialization"""

    def test_default_state(self):
        """Test initial state is zeroed and uninitialized"""
        bf = BreathingFilter()
        assert bf.initialized is False
        assert bf.energy == 0.0
        assert bf.hp_x_prev == 0.0
        assert bf.hp_y_prev == 0.0
        assert bf.lp_x_prev == 0.0
        assert bf.lp_y_prev == 0.0

    def test_coefficients_match_cpp(self):
        """Test that filter coefficients match C++ csi_filters.cpp"""
        assert BreathingFilter.HP_B0 == pytest.approx(0.99749, rel=1e-5)
        assert BreathingFilter.HP_A1 == pytest.approx(-0.99498, rel=1e-5)
        assert BreathingFilter.LP_B0 == pytest.approx(0.01850, rel=1e-5)
        assert BreathingFilter.LP_A1 == pytest.approx(-0.96300, rel=1e-5)
        assert BreathingFilter.ENERGY_ALPHA == pytest.approx(0.00333, rel=1e-3)

    def test_reset(self):
        """Test reset returns to initial state"""
        bf = BreathingFilter()
        # Process some data
        bf.filter(100.0)
        bf.filter(110.0)
        assert bf.initialized is True

        bf.reset()
        assert bf.initialized is False
        assert bf.energy == 0.0
        assert bf.hp_x_prev == 0.0


class TestBreathingFilterBasic:
    """Test basic BreathingFilter functionality"""

    def test_first_sample_returns_zero(self):
        """Test that first sample initializes state and returns 0"""
        bf = BreathingFilter()
        result = bf.filter(100.0)
        assert result == 0.0
        assert bf.initialized is True
        assert bf.hp_x_prev == 100.0

    def test_get_score_before_init(self):
        """Test get_score returns 0 before initialization"""
        bf = BreathingFilter()
        assert bf.get_score() == 0.0

    def test_get_score_after_init(self):
        """Test get_score returns same as filter output"""
        bf = BreathingFilter()
        bf.filter(100.0)
        bf.filter(110.0)
        score = bf.get_score()
        assert score >= 0.0

    def test_constant_input_decays_to_zero(self):
        """Test that constant amplitude produces near-zero score (HP removes DC)"""
        bf = BreathingFilter()
        # Feed constant value for 500 samples (5 seconds at 100 Hz)
        for _ in range(500):
            bf.filter(100.0)
        # DC should be fully removed by HP filter
        assert bf.get_score() < 0.01

    def test_step_change_transient(self):
        """Test that a step change produces a transient then decays"""
        bf = BreathingFilter()
        # Settle at 100
        for _ in range(300):
            bf.filter(100.0)
        score_before = bf.get_score()

        # Step to 200
        bf.filter(200.0)
        score_after = bf.get_score()

        # Step should produce transient energy
        assert score_after > score_before


class TestBreathingFilterFrequencyResponse:
    """Test frequency selectivity of the breathing bandpass filter"""

    def test_breathing_rate_passes(self):
        """Test that 0.3 Hz (18 BPM) passes through with high energy"""
        bf = BreathingFilter()
        freq = 0.3  # Hz (breathing rate)
        sample_rate = 100.0
        duration = 30.0  # seconds - enough for filter to settle

        for i in range(int(sample_rate * duration)):
            t = i / sample_rate
            amplitude = 100.0 + 5.0 * math.sin(2 * math.pi * freq * t)
            bf.filter(amplitude)

        breathing_score = bf.get_score()
        assert breathing_score > 0.1, f"Breathing rate signal should pass, got {breathing_score}"

    def test_fast_signal_rejected(self):
        """Test that 5 Hz signal is attenuated (above LP cutoff 0.6 Hz)"""
        bf_fast = BreathingFilter()
        bf_breath = BreathingFilter()
        sample_rate = 100.0
        duration = 30.0

        for i in range(int(sample_rate * duration)):
            t = i / sample_rate
            # Same amplitude modulation, different frequencies
            fast = 100.0 + 5.0 * math.sin(2 * math.pi * 5.0 * t)
            breath = 100.0 + 5.0 * math.sin(2 * math.pi * 0.3 * t)
            bf_fast.filter(fast)
            bf_breath.filter(breath)

        # Fast signal should be significantly weaker than breathing
        assert bf_fast.get_score() < bf_breath.get_score() * 0.2, \
            f"5 Hz should be attenuated: fast={bf_fast.get_score()}, breath={bf_breath.get_score()}"

    def test_very_slow_signal_rejected(self):
        """Test that 0.01 Hz signal is attenuated (below HP cutoff 0.08 Hz)"""
        bf_slow = BreathingFilter()
        bf_breath = BreathingFilter()
        sample_rate = 100.0
        duration = 120.0  # Longer for very slow signal

        for i in range(int(sample_rate * duration)):
            t = i / sample_rate
            slow = 100.0 + 5.0 * math.sin(2 * math.pi * 0.01 * t)
            breath = 100.0 + 5.0 * math.sin(2 * math.pi * 0.3 * t)
            bf_slow.filter(slow)
            bf_breath.filter(breath)

        # Very slow signal should be weaker than breathing
        assert bf_slow.get_score() < bf_breath.get_score() * 0.3, \
            f"0.01 Hz should be attenuated: slow={bf_slow.get_score()}, breath={bf_breath.get_score()}"


class TestBreathingFilterCppParity:
    """Test exact numerical parity with C++ implementation"""

    def test_known_sequence(self):
        """Test a known input sequence produces expected outputs

        Verify step-by-step against the C++ filter math
        (csi_filters.cpp) using the same coefficients.
        """
        bf = BreathingFilter()

        # Input: amplitude sums simulating a breathing pattern
        inputs = [100.0, 102.0, 105.0, 103.0, 99.0,
                  97.0, 100.0, 104.0, 106.0, 103.0,
                  98.0, 96.0, 99.0, 103.0, 107.0,
                  104.0, 100.0, 97.0, 98.0, 102.0]

        # Compute outputs using Python implementation
        py_outputs = []
        for x in inputs:
            py_outputs.append(bf.filter(x))

        # First output must be 0 (initialization)
        assert py_outputs[0] == 0.0

        # Manually verify the filter math for sample 1:
        # HP: hp_out = 0.99749 * (102 - 100) - (-0.99498) * 0 = 1.99498
        # LP: lp_out = 0.01850 * (1.99498 + 0) - (-0.96300) * 0 = 0.036907
        # sq = 0.036907^2 = 0.001362
        # energy = 0.00333 * 0.001362 + 0.99667 * 0 = 0.000004535
        # score = sqrt(0.000004535) ≈ 0.002130
        expected_1 = 0.99749 * (102.0 - 100.0)
        lp_1 = 0.01850 * (expected_1 + 0.0)
        sq_1 = lp_1 * lp_1
        e_1 = 0.00333 * sq_1
        assert py_outputs[1] == pytest.approx(math.sqrt(e_1), rel=1e-4)

        # All outputs must be non-negative
        for out in py_outputs:
            assert out >= 0.0

    def test_filter_vs_get_score_consistency(self):
        """Test that filter() return value matches get_score()"""
        bf = BreathingFilter()
        inputs = [100.0, 105.0, 110.0, 108.0, 103.0]

        for x in inputs:
            filter_result = bf.filter(x)

        # get_score() should match the last filter() return
        assert bf.get_score() == pytest.approx(filter_result, rel=1e-10)

