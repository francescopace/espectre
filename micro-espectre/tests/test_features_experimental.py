"""
Micro-ESPectre - Experimental Feature Extraction Tests

Tests for FFT, spectral, energy, multi-lag autocorrelation, and phase features.
These are experimental features from CSI-F, CSI-HC, and Wi-Limb papers.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""
# ruff: noqa: E402

import pytest
import math
import numpy as np
from features import (
    _fft_real,
    calc_energy_features,
    calc_spectral_centroid,
    calc_spectral_flatness,
    calc_spectral_rolloff,
    calc_autocorrelation_lag,
    calc_periodicity_strength,
    calc_subcarrier_correlation,
    calc_phase_diff_variance,
    calc_phase_std,
    calc_phase_entropy,
    calc_phase_range,
    calc_amp_entropy,
    extract_features_by_name,
    ALL_AVAILABLE_FEATURES,
)


# ============================================================================
# FFT Implementation Tests
# ============================================================================

class TestFFTReal:
    """Test pure Python FFT implementation"""
    
    def test_empty_input(self):
        """Test FFT of empty input"""
        result = _fft_real([])
        assert result == [0.0]
    
    def test_single_value(self):
        """Test FFT of single value"""
        result = _fft_real([5.0])
        assert result == [0.0]
    
    def test_two_values(self):
        """Test FFT of two values"""
        result = _fft_real([1.0, 2.0])
        assert len(result) >= 1
        assert all(isinstance(x, float) for x in result)
    
    def test_power_of_two_length(self):
        """Test FFT with power-of-2 length input"""
        values = [math.sin(2 * math.pi * i / 16) for i in range(16)]
        result = _fft_real(values)
        assert len(result) == 8  # n_fft/2 positive frequencies
    
    def test_non_power_of_two_length(self):
        """Test FFT with non-power-of-2 length (zero-pads internally)"""
        values = [1.0] * 10  # Will be padded to 16
        result = _fft_real(values)
        assert len(result) == 8  # 16/2
    
    def test_sinusoid_peak_detection(self):
        """Test that FFT correctly detects sinusoid frequency"""
        # Create a sinusoid at bin 4 of 32-point FFT
        n = 32
        freq_bin = 4
        values = [math.sin(2 * math.pi * freq_bin * i / n) for i in range(n)]
        result = _fft_real(values)
        
        # Find peak (skip DC)
        peak_idx = max(range(1, len(result)), key=lambda i: result[i])
        assert peak_idx == freq_bin
    
    def test_dc_component(self):
        """Test DC component detection"""
        # Constant signal = all energy at DC
        values = [5.0] * 16
        result = _fft_real(values)
        
        # DC should be dominant
        assert result[0] > sum(result[1:]) * 0.9
    
    def test_magnitudes_positive(self):
        """Test that all magnitudes are non-negative"""
        np.random.seed(42)
        values = list(np.random.normal(0, 1, 32))
        result = _fft_real(values)
        
        assert all(x >= 0 for x in result)


# ============================================================================
# Energy Features Tests (CSI-F paper)
# ============================================================================

class TestEnergyFeatures:
    """Test FFT-based energy feature extraction"""
    
    def test_short_buffer(self):
        """Test energy features with too-short buffer"""
        result = calc_energy_features([1.0, 2.0, 3.0], 3)
        assert result == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    def test_returns_six_values(self):
        """Test that 6 energy features are returned"""
        values = [float(i) for i in range(32)]
        result = calc_energy_features(values, 32)
        assert len(result) == 6
    
    def test_total_energy_positive(self):
        """Test that total energy is positive for non-zero signal"""
        np.random.seed(42)
        values = list(np.random.normal(5, 2, 50))
        result = calc_energy_features(values, 50)
        
        total_energy = result[0]
        assert total_energy > 0
    
    def test_energy_ratio_range(self):
        """Test that low frequency energy ratio is in [0, 1]"""
        np.random.seed(42)
        values = list(np.random.normal(5, 2, 50))
        result = calc_energy_features(values, 50)
        
        energy_ratio_low = result[4]
        assert 0.0 <= energy_ratio_low <= 1.0
    
    def test_energy_sum_consistency(self):
        """Test that band energies sum to total energy"""
        np.random.seed(42)
        values = list(np.random.normal(5, 2, 50))
        result = calc_energy_features(values, 50)
        
        total, low, mid, high = result[0], result[1], result[2], result[3]
        band_sum = low + mid + high
        
        # Allow small numerical error
        assert abs(total - band_sum) < total * 0.01 or total < 1e-10
    
    def test_low_freq_signal(self):
        """Test that slow signal has high low-frequency energy"""
        # Very slow sinusoid (0.5 Hz equivalent)
        values = [math.sin(2 * math.pi * i / 100) for i in range(100)]
        result = calc_energy_features(values, 100, sample_rate=100.0)
        
        energy_ratio_low = result[4]
        assert energy_ratio_low > 0.5  # Mostly low frequency
    
    def test_dominant_frequency_nonnegative(self):
        """Test that dominant frequency is non-negative"""
        np.random.seed(42)
        values = list(np.random.normal(5, 2, 50))
        result = calc_energy_features(values, 50)
        
        dominant_freq = result[5]
        assert dominant_freq >= 0


# ============================================================================
# Spectral Features Tests (CSI-HC paper)
# ============================================================================

class TestSpectralCentroid:
    """Test spectral centroid calculation"""
    
    def test_short_buffer(self):
        """Test with too-short buffer"""
        assert calc_spectral_centroid([1.0, 2.0, 3.0], 3) == 0.0
    
    def test_positive_result(self):
        """Test that centroid is positive for non-zero signal"""
        np.random.seed(42)
        values = list(np.random.normal(5, 2, 50))
        result = calc_spectral_centroid(values, 50)
        
        assert result >= 0
    
    def test_low_freq_signal_low_centroid(self):
        """Test that slow signal has lower centroid"""
        # Slow sinusoid
        slow = [math.sin(2 * math.pi * i / 50) for i in range(100)]
        # Fast sinusoid
        fast = [math.sin(2 * math.pi * i / 5) for i in range(100)]
        
        centroid_slow = calc_spectral_centroid(slow, 100)
        centroid_fast = calc_spectral_centroid(fast, 100)
        
        assert centroid_slow < centroid_fast


class TestSpectralFlatness:
    """Test spectral flatness (Wiener entropy) calculation"""
    
    def test_short_buffer(self):
        """Test with too-short buffer"""
        assert calc_spectral_flatness([1.0, 2.0, 3.0], 3) == 0.0
    
    def test_result_range(self):
        """Test that flatness is in [0, 1]"""
        np.random.seed(42)
        values = list(np.random.normal(5, 2, 50))
        result = calc_spectral_flatness(values, 50)
        
        assert 0.0 <= result <= 1.0
    
    def test_white_noise_high_flatness(self):
        """Test that white noise has high flatness"""
        np.random.seed(42)
        noise = list(np.random.normal(0, 1, 128))
        result = calc_spectral_flatness(noise, 128)
        
        # White noise should have relatively flat spectrum
        assert result > 0.1
    
    def test_pure_sinusoid_lower_flatness(self):
        """Test that pure sinusoid has lower flatness than noise"""
        # Pure sinusoid (peaked spectrum)
        sinusoid = [math.sin(2 * math.pi * 4 * i / 64) for i in range(64)]
        # White noise (flat spectrum)
        np.random.seed(42)
        noise = list(np.random.normal(0, 1, 64))
        
        flat_sin = calc_spectral_flatness(sinusoid, 64)
        flat_noise = calc_spectral_flatness(noise, 64)
        
        assert flat_sin < flat_noise


class TestSpectralRolloff:
    """Test spectral rolloff calculation"""
    
    def test_short_buffer(self):
        """Test with too-short buffer"""
        assert calc_spectral_rolloff([1.0, 2.0, 3.0], 3) == 0.0
    
    def test_result_nonnegative(self):
        """Test that rolloff is non-negative"""
        np.random.seed(42)
        values = list(np.random.normal(5, 2, 50))
        result = calc_spectral_rolloff(values, 50)
        
        assert result >= 0
    
    def test_low_freq_signal_low_rolloff(self):
        """Test that slow signal has lower rolloff"""
        # Slow sinusoid
        slow = [math.sin(2 * math.pi * i / 50) for i in range(100)]
        # Fast sinusoid
        fast = [math.sin(2 * math.pi * i / 5) for i in range(100)]
        
        rolloff_slow = calc_spectral_rolloff(slow, 100)
        rolloff_fast = calc_spectral_rolloff(fast, 100)
        
        assert rolloff_slow < rolloff_fast


# ============================================================================
# Multi-Lag Autocorrelation Tests (Wi-Limb paper)
# ============================================================================

class TestAutocorrelationLag:
    """Test multi-lag autocorrelation"""
    
    def test_short_buffer(self):
        """Test with too-short buffer for given lag"""
        assert calc_autocorrelation_lag([1.0, 2.0], 2, lag=5) == 0.0
    
    def test_lag1_matches_standard(self):
        """Test that lag-1 matches standard autocorrelation"""
        np.random.seed(42)
        values = list(np.random.normal(5, 2, 50))
        
        from features import calc_autocorrelation
        standard = calc_autocorrelation(values, 50)
        lag1 = calc_autocorrelation_lag(values, 50, lag=1)
        
        # Should be very close
        assert abs(standard - lag1) < 0.01
    
    def test_result_range(self):
        """Test that result is in [-1, 1]"""
        np.random.seed(42)
        values = list(np.random.normal(5, 2, 50))
        
        for lag in [1, 2, 5, 10]:
            result = calc_autocorrelation_lag(values, 50, lag=lag)
            assert -1.0 <= result <= 1.0
    
    def test_smooth_signal_high_autocorr(self):
        """Test that smooth signal has high autocorrelation at small lags"""
        # Slow sinusoid
        values = [math.sin(0.1 * i) for i in range(50)]
        
        ac1 = calc_autocorrelation_lag(values, 50, lag=1)
        ac2 = calc_autocorrelation_lag(values, 50, lag=2)
        
        assert ac1 > 0.9
        assert ac2 > 0.8


class TestPeriodicityStrength:
    """Test periodicity strength calculation"""
    
    def test_short_buffer(self):
        """Test with too-short buffer"""
        assert calc_periodicity_strength([1.0] * 5, 5) == 0.0
    
    def test_result_range(self):
        """Test that result is in [0, 1]"""
        np.random.seed(42)
        values = list(np.random.normal(5, 2, 50))
        result = calc_periodicity_strength(values, 50)
        
        assert 0.0 <= result <= 1.0
    
    def test_periodic_signal_high_strength(self):
        """Test that periodic signal has high periodicity strength"""
        # Periodic signal with period ~10 samples
        values = [math.sin(2 * math.pi * i / 10) for i in range(100)]
        result = calc_periodicity_strength(values, 100)
        
        assert result > 0.5
    
    def test_random_signal_low_strength(self):
        """Test that random signal has low periodicity strength"""
        np.random.seed(42)
        values = list(np.random.normal(0, 1, 100))
        result = calc_periodicity_strength(values, 100)
        
        assert result < 0.5


# ============================================================================
# Cross-Subcarrier Correlation Tests
# ============================================================================

class TestSubcarrierCorrelation:
    """Test subcarrier correlation calculation"""
    
    def test_empty_buffer(self):
        """Test with empty or None buffer"""
        assert calc_subcarrier_correlation(None) == (0.0, 0.0)
        assert calc_subcarrier_correlation([]) == (0.0, 0.0)
    
    def test_single_packet(self):
        """Test with single packet (needs at least 2)"""
        assert calc_subcarrier_correlation([[1.0, 2.0, 3.0]]) == (0.0, 0.0)
    
    def test_identical_packets_high_correlation(self):
        """Test that identical packets have correlation ~1"""
        packet = [1.0, 2.0, 3.0, 4.0, 5.0]
        buffer = [packet[:] for _ in range(5)]
        
        mean_corr, std_corr = calc_subcarrier_correlation(buffer)
        assert mean_corr > 0.99
    
    def test_random_packets_lower_correlation(self):
        """Test that random packets have lower correlation"""
        np.random.seed(42)
        buffer = [list(np.random.normal(5, 2, 10)) for _ in range(10)]
        
        mean_corr, std_corr = calc_subcarrier_correlation(buffer)
        assert mean_corr < 0.9


# ============================================================================
# Phase Features Tests
# ============================================================================

class TestPhaseDiffVariance:
    """Test phase difference variance calculation"""
    
    def test_short_input(self):
        """Test with too few phases"""
        assert calc_phase_diff_variance(None) == 0.0
        assert calc_phase_diff_variance([0.0, 1.0]) == 0.0
    
    def test_constant_phases_zero_variance(self):
        """Test that constant phases give zero variance"""
        phases = [0.5] * 10
        result = calc_phase_diff_variance(phases)
        assert result < 1e-10
    
    def test_linear_phases_low_variance(self):
        """Test that linear phases have low variance"""
        phases = [0.1 * i for i in range(10)]
        result = calc_phase_diff_variance(phases)
        assert result < 1e-10  # All diffs are equal
    
    def test_random_phases_higher_variance(self):
        """Test that random phases have higher variance"""
        np.random.seed(42)
        phases = list(np.random.uniform(-math.pi, math.pi, 20))
        result = calc_phase_diff_variance(phases)
        
        assert result > 0


class TestPhaseStd:
    """Test phase standard deviation calculation"""
    
    def test_short_input(self):
        """Test with too few phases"""
        assert calc_phase_std(None) == 0.0
        assert calc_phase_std([0.5]) == 0.0
    
    def test_constant_phases_zero_std(self):
        """Test that constant phases give zero std"""
        phases = [0.5] * 10
        result = calc_phase_std(phases)
        assert result < 1e-10
    
    def test_varied_phases_positive_std(self):
        """Test that varied phases have positive std"""
        phases = [0.0, 0.5, 1.0, -0.5, 0.3]
        result = calc_phase_std(phases)
        
        assert result > 0
    
    def test_result_reasonable_range(self):
        """Test that std is reasonable for [-pi, pi] phases"""
        np.random.seed(42)
        phases = list(np.random.uniform(-math.pi, math.pi, 20))
        result = calc_phase_std(phases)
        
        # Max std for uniform [-pi, pi] is about 1.81
        assert result < 2.0


class TestPhaseEntropy:
    """Test phase entropy calculation"""
    
    def test_short_input(self):
        """Test with too few phases"""
        assert calc_phase_entropy(None) == 0.0
        assert calc_phase_entropy([0.5]) == 0.0
    
    def test_constant_phases_zero_entropy(self):
        """Test that constant phases give zero entropy"""
        phases = [0.5] * 10
        result = calc_phase_entropy(phases)
        assert result == 0.0
    
    def test_varied_phases_positive_entropy(self):
        """Test that varied phases have positive entropy"""
        phases = [0.0, 0.5, 1.0, 1.5, 2.0, -0.5, -1.0, -1.5]
        result = calc_phase_entropy(phases)
        
        assert result > 0
    
    def test_uniform_distribution_higher_entropy(self):
        """Test that uniform distribution has higher entropy"""
        # Concentrated phases
        concentrated = [0.1 * i for i in range(10)]
        # Spread phases
        spread = [math.pi * 2 * i / 10 - math.pi for i in range(10)]
        
        ent_conc = calc_phase_entropy(concentrated)
        ent_spread = calc_phase_entropy(spread)
        
        # Both should be positive, spread should be >= concentrated
        assert ent_spread >= ent_conc or abs(ent_spread - ent_conc) < 0.1


class TestPhaseRange:
    """Test phase range calculation"""
    
    def test_short_input(self):
        """Test with too few phases"""
        assert calc_phase_range(None) == 0.0
        assert calc_phase_range([0.5]) == 0.0
    
    def test_constant_phases_zero_range(self):
        """Test that constant phases give zero range"""
        phases = [0.5] * 10
        result = calc_phase_range(phases)
        assert result == 0.0
    
    def test_known_range(self):
        """Test with known range"""
        phases = [0.0, 1.0, 2.0, -1.0, 0.5]
        result = calc_phase_range(phases)
        
        assert result == pytest.approx(3.0, rel=1e-6)  # 2.0 - (-1.0) = 3.0
    
    def test_range_nonnegative(self):
        """Test that range is non-negative"""
        np.random.seed(42)
        phases = list(np.random.uniform(-math.pi, math.pi, 20))
        result = calc_phase_range(phases)
        
        assert result >= 0


# ============================================================================
# Amplitude Entropy Tests
# ============================================================================

class TestAmpEntropy:
    """Test amplitude entropy calculation"""
    
    def test_empty_input(self):
        """Test with empty or None input"""
        assert calc_amp_entropy(None) == 0.0
        assert calc_amp_entropy([5.0]) == 0.0
    
    def test_constant_amplitudes_zero_entropy(self):
        """Test that constant amplitudes give zero entropy"""
        amplitudes = [10.0] * 12
        result = calc_amp_entropy(amplitudes)
        assert result == 0.0
    
    def test_varied_amplitudes_positive_entropy(self):
        """Test that varied amplitudes have positive entropy"""
        amplitudes = [1.0, 2.0, 5.0, 10.0, 15.0, 3.0, 8.0, 12.0, 6.0, 4.0, 9.0, 11.0]
        result = calc_amp_entropy(amplitudes)
        
        assert result > 0
    
    def test_entropy_nonnegative(self):
        """Test that entropy is non-negative"""
        np.random.seed(42)
        amplitudes = list(np.random.uniform(1, 20, 12))
        result = calc_amp_entropy(amplitudes)
        
        assert result >= 0


# ============================================================================
# Extended Feature Extraction Tests
# ============================================================================

class TestExtractExperimentalFeatures:
    """Test extraction of experimental features"""
    
    def test_fft_features_extraction(self):
        """Test extraction of FFT-based features"""
        np.random.seed(42)
        buffer = list(np.random.normal(5, 2, 50))
        
        fft_features = [
            'fft_total_energy', 'fft_low_energy', 'fft_mid_energy',
            'fft_high_energy', 'fft_energy_ratio_low', 'fft_dominant_freq'
        ]
        
        result = extract_features_by_name(buffer, 50, feature_names=fft_features)
        
        assert len(result) == 6
        assert all(isinstance(x, float) for x in result)
    
    def test_spectral_features_extraction(self):
        """Test extraction of spectral features"""
        np.random.seed(42)
        buffer = list(np.random.normal(5, 2, 50))
        
        spectral_features = ['spectral_centroid', 'spectral_flatness', 'spectral_rolloff']
        
        result = extract_features_by_name(buffer, 50, feature_names=spectral_features)
        
        assert len(result) == 3
        assert all(isinstance(x, float) for x in result)
    
    def test_multilag_autocorr_features(self):
        """Test extraction of multi-lag autocorrelation features"""
        buffer = [math.sin(0.1 * i) for i in range(50)]
        
        autocorr_features = ['turb_autocorr', 'turb_autocorr_lag2', 
                             'turb_autocorr_lag5', 'turb_periodicity']
        
        result = extract_features_by_name(buffer, 50, feature_names=autocorr_features)
        
        assert len(result) == 4
        assert all(-1.0 <= x <= 1.0 for x in result[:3])
        assert 0.0 <= result[3] <= 1.0  # periodicity
    
    def test_phase_features_extraction(self):
        """Test extraction of phase features"""
        buffer = [float(i) for i in range(50)]
        phases = [0.1 * i for i in range(12)]
        
        phase_features = ['phase_diff_var', 'phase_std', 'phase_entropy', 'phase_range']
        
        result = extract_features_by_name(
            buffer, 50, phases=phases, feature_names=phase_features
        )
        
        assert len(result) == 4
        assert all(isinstance(x, float) for x in result)
    
    def test_unknown_feature_raises(self):
        """Test that unknown feature raises ValueError"""
        buffer = [float(i) for i in range(50)]
        
        with pytest.raises(ValueError, match="Unknown feature"):
            extract_features_by_name(buffer, 50, feature_names=['nonexistent_feature'])
    
    def test_all_available_features(self):
        """Test that all features in ALL_AVAILABLE_FEATURES can be extracted"""
        np.random.seed(42)
        buffer = list(np.random.normal(5, 2, 50))
        amplitudes = list(np.random.uniform(1, 20, 12))
        phases = list(np.random.uniform(-math.pi, math.pi, 12))
        
        result = extract_features_by_name(
            buffer, 50, 
            amplitudes=amplitudes, 
            phases=phases,
            feature_names=ALL_AVAILABLE_FEATURES
        )
        
        assert len(result) == len(ALL_AVAILABLE_FEATURES)
        assert all(isinstance(x, (int, float)) for x in result)
