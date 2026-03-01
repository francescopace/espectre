"""
Micro-ESPectre - Gesture Feature Extraction Tests

Unit tests for gesture_features.py.
Verifies that each morphology and phase feature:
  - Returns correct values for synthetic inputs
  - Is within expected range
  - Handles edge cases (empty input, constant signal, short events)

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import math
import sys
from pathlib import Path

import pytest

src_path = str(Path(__file__).parent.parent / 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from gesture_features import (
    calc_event_duration,
    calc_peak_position,
    calc_peak_to_mean_ratio,
    calc_rise_fall_asymmetry,
    calc_pre_post_energy_ratio,
    calc_n_local_peaks,
    calc_peak_fwhm,
    calc_turb_mad,
    calc_turb_iqr,
    calc_phase_diff_var,
    calc_phase_entropy,
    calc_phase_circular_variance,
    extract_gesture_features,
    GESTURE_FEATURES,
    NUM_GESTURE_FEATURES,
)


# ============================================================================
# Helpers
# ============================================================================

def gaussian(n, peak_pos=None, sigma=None, baseline=1.0, amplitude=10.0):
    """Generate a Gaussian turbulence sequence of length n."""
    if peak_pos is None:
        peak_pos = n // 2
    if sigma is None:
        sigma = n / 6.0
    return [baseline + amplitude * math.exp(-((i - peak_pos) ** 2) / (2 * sigma * sigma))
            for i in range(n)]


def constant(n, value=5.0):
    return [value] * n


def ramp(n, start=1.0, end=10.0):
    if n < 2:
        return [start]
    return [start + (end - start) * i / (n - 1) for i in range(n)]


def flat_phases(n, n_phases=5, value=0.0):
    return [[value] * n_phases for _ in range(n)]


# ============================================================================
# event_duration
# ============================================================================

class TestEventDuration:
    def test_normalization(self):
        assert calc_event_duration([1.0] * 200) == pytest.approx(1.0)

    def test_half_reference(self):
        # Log-compressed and quantized in 0.1 bins.
        assert calc_event_duration([1.0] * 100) == pytest.approx(0.9)

    def test_single_packet(self):
        assert calc_event_duration([1.0]) == pytest.approx(0.1)

    def test_empty(self):
        assert calc_event_duration([]) == pytest.approx(0.0)


# ============================================================================
# peak_position
# ============================================================================

class TestPeakPosition:
    def test_peak_at_center(self):
        t = gaussian(100, peak_pos=50)
        pos = calc_peak_position(t)
        assert 0.45 < pos < 0.55

    def test_peak_at_start(self):
        t = [10.0] + [1.0] * 99
        assert calc_peak_position(t) == pytest.approx(0.0)

    def test_peak_at_end(self):
        t = [1.0] * 99 + [10.0]
        assert calc_peak_position(t) == pytest.approx(1.0)

    def test_range(self):
        for _ in range(5):
            import random
            t = [random.random() for _ in range(50)]
            pos = calc_peak_position(t)
            assert 0.0 <= pos <= 1.0


# ============================================================================
# peak_to_mean_ratio
# ============================================================================

class TestPeakToMeanRatio:
    def test_constant_signal(self):
        assert calc_peak_to_mean_ratio(constant(50, 5.0)) == pytest.approx(1.0)

    def test_spike(self):
        t = [1.0] * 49 + [100.0]
        ratio = calc_peak_to_mean_ratio(t)
        assert ratio > 1.0

    def test_clamped_at_10(self):
        t = [0.001] * 99 + [1000.0]
        assert calc_peak_to_mean_ratio(t) == pytest.approx(10.0)

    def test_empty(self):
        assert calc_peak_to_mean_ratio([]) == pytest.approx(1.0)


# ============================================================================
# rise_fall_asymmetry
# ============================================================================

class TestRiseFallAsymmetry:
    def test_symmetric_peak(self):
        t = gaussian(100)
        asym = calc_rise_fall_asymmetry(t)
        assert asym == pytest.approx(0.0, abs=0.05)

    def test_early_peak(self):
        t = gaussian(100, peak_pos=10)
        assert calc_rise_fall_asymmetry(t) < 0.0

    def test_range(self):
        t = gaussian(100, peak_pos=30)
        assert -1.0 <= calc_rise_fall_asymmetry(t) <= 1.0


# ============================================================================
# pre_post_energy_ratio
# ============================================================================

class TestPrePostEnergyRatio:
    def test_symmetric(self):
        t = gaussian(100)  # symmetric Gaussian → ratio ≈ 1.0
        ratio = calc_pre_post_energy_ratio(t)
        assert ratio == pytest.approx(1.0, abs=0.3)

    def test_front_heavy(self):
        t = [10.0] * 50 + [1.0] * 50
        ratio = calc_pre_post_energy_ratio(t)
        assert ratio > 1.0

    def test_back_heavy(self):
        t = [1.0] * 50 + [10.0] * 50
        ratio = calc_pre_post_energy_ratio(t)
        assert ratio < 1.0

    def test_clamped(self):
        t = [100.0] * 50 + [0.001] * 50
        ratio = calc_pre_post_energy_ratio(t)
        assert ratio == pytest.approx(10.0)


# ============================================================================
# n_local_peaks
# ============================================================================

class TestNLocalPeaks:
    def test_constant(self):
        assert calc_n_local_peaks(constant(100)) == pytest.approx(0.0)

    def test_single_peak(self):
        # Simple triangle spike: surpasses prominence threshold at the apex
        # The apex value is 10x the baseline, so prominence >> 10% of range
        t = [1.0] * 100
        t[50] = 100.0  # spike: prominence = 99, sig_range = 99, threshold = 9.9
        count = calc_n_local_peaks(t)
        assert count == pytest.approx(1 / 10.0, abs=0.05)

    def test_multiple_peaks(self):
        # Two well-separated spikes with high prominence
        t = [1.0] * 100
        t[25] = 100.0
        t[75] = 100.0
        count = calc_n_local_peaks(t)
        assert count >= 2 / 10.0

    def test_minimum_input(self):
        assert calc_n_local_peaks([]) == 0.0
        assert calc_n_local_peaks([1.0]) == 0.0
        assert calc_n_local_peaks([1.0, 2.0]) == 0.0


# ============================================================================
# peak_fwhm
# ============================================================================

class TestPeakFWHM:
    def test_narrow_peak(self):
        t = gaussian(100, sigma=5.0)
        fwhm = calc_peak_fwhm(t)
        assert 0.0 < fwhm < 0.4  # Narrow peak → small FWHM

    def test_wide_peak(self):
        t = gaussian(100, sigma=30.0)
        fwhm = calc_peak_fwhm(t)
        assert fwhm > 0.3  # Wide peak → large FWHM

    def test_range(self):
        t = gaussian(100)
        fwhm = calc_peak_fwhm(t)
        assert 0.0 <= fwhm <= 1.0

    def test_too_short(self):
        assert calc_peak_fwhm([1.0, 2.0]) == pytest.approx(0.0)


# ============================================================================
# Robust turbulence features
# ============================================================================

class TestRobustTurbulenceFeatures:
    def test_mad_constant(self):
        assert calc_turb_mad(constant(50, 3.0)) == pytest.approx(0.0)

    def test_iqr_constant(self):
        assert calc_turb_iqr(constant(50, 3.0)) == pytest.approx(0.0)

    def test_mad_positive(self):
        t = [1.0, 1.0, 2.0, 3.0, 9.0, 2.0, 1.0]
        assert calc_turb_mad(t) > 0.0

    def test_iqr_positive(self):
        t = list(range(40))
        assert calc_turb_iqr(t) > 0.0


# ============================================================================
# Phase features
# ============================================================================

class TestPhaseFeatures:
    def test_constant_phase_zero_variance(self):
        phases = [[0.0, 0.0, 0.0, 0.0]] * 20
        assert calc_phase_diff_var(phases) == pytest.approx(0.0, abs=1e-10)
        assert calc_phase_circular_variance(phases) == pytest.approx(0.0, abs=1e-10)

    def test_varying_phase(self):
        phases = [[math.sin(i * j * 0.1) for j in range(6)] for i in range(30)]
        assert calc_phase_circular_variance(phases) > 0.0

    def test_entropy_max_uniform(self):
        # Uniform distribution across bins → maximum entropy
        phases = [[i / 10.0 for _ in range(5)] for i in range(50)]
        entropy = calc_phase_entropy(phases)
        assert entropy >= 0.0

    def test_empty_phases(self):
        assert calc_phase_diff_var([]) == pytest.approx(0.0)
        assert calc_phase_entropy([]) == pytest.approx(0.0)
        assert calc_phase_circular_variance([]) == pytest.approx(0.0)


# ============================================================================
# extract_gesture_features (combined)
# ============================================================================

class TestExtractGestureFeatures:
    def test_returns_correct_length(self):
        event = [{'turbulence': 1.0, 'phases': [0.0, 0.5, 1.0]} for _ in range(50)]
        features = extract_gesture_features(event)
        assert len(features) == NUM_GESTURE_FEATURES

    def test_feature_names_count(self):
        assert len(GESTURE_FEATURES) == NUM_GESTURE_FEATURES

    def test_empty_event(self):
        features = extract_gesture_features([])
        assert len(features) == NUM_GESTURE_FEATURES
        assert all(f == 0.0 for f in features)

    def test_event_without_phases(self):
        event = [{'turbulence': 1.0 + i * 0.1} for i in range(50)]
        features = extract_gesture_features(event)
        assert len(features) == NUM_GESTURE_FEATURES
        # Morphology features should be non-zero
        assert features[0] > 0.0   # event_duration
        assert features[2] > 0.0   # peak_to_mean_ratio
        # Phase features should be 0.0 (no phases provided)
        assert features[9] == pytest.approx(0.0)
        assert features[10] == pytest.approx(0.0)
        assert features[11] == pytest.approx(0.0)

    def test_gaussian_event(self):
        n = 100
        turb = gaussian(n, peak_pos=50, sigma=10.0)
        phases = [[math.sin(j * 0.1) for j in range(6)] for _ in range(n)]
        event = [{'turbulence': turb[i], 'phases': phases[i]} for i in range(n)]

        features = extract_gesture_features(event)

        assert len(features) == NUM_GESTURE_FEATURES

        # Peak position near center
        assert 0.4 < features[1] < 0.6

        # Peak-to-mean > 1 for a spike
        assert features[2] > 1.0

        # Phase features finite and non-negative
        for f in features[8:]:
            assert f >= 0.0
            assert math.isfinite(f)

    def test_all_features_finite(self):
        import random
        random.seed(42)
        n = 80
        event = [
            {
                'turbulence': 1.0 + random.random() * 5.0,
                'phases': [random.uniform(-3.14, 3.14) for _ in range(8)]
            }
            for _ in range(n)
        ]
        features = extract_gesture_features(event)
        for i, (name, val) in enumerate(zip(GESTURE_FEATURES, features)):
            assert math.isfinite(val), f"Feature '{name}' (index {i}) is not finite: {val}"


# ============================================================================
# Cross-platform validation (synthetic data for C++ comparison)
# ============================================================================

class TestCrossPlatformFeatures:
    """
    Tests with known synthetic data that can be replicated in C++ tests
    to ensure Python and C++ implementations produce identical results.
    """

    def test_synthetic_triangular_peak(self):
        """Triangular peak at center: deterministic input for cross-platform validation."""
        n = 100
        turb_list = [1.0] * n
        mid = n // 2
        turb_list[mid - 1] = 3.0
        turb_list[mid] = 10.0
        turb_list[mid + 1] = 3.0

        # Expected values (computed analytically)
        assert calc_event_duration(turb_list) == pytest.approx(0.9)

        # Peak at index 50 in a 100-element list → 50/99 ≈ 0.505
        assert calc_peak_position(turb_list) == pytest.approx(50.0 / 99.0, rel=1e-4)

        # Peak=10, mean=(97*1 + 3 + 10 + 3)/100 = 113/100 = 1.13 → ratio ≈ 8.85
        expected_mean = (97 * 1.0 + 3.0 + 10.0 + 3.0) / 100
        assert calc_peak_to_mean_ratio(turb_list) == pytest.approx(10.0 / expected_mean, rel=1e-3)

        # Symmetric triangular peak -> asymmetry ~ 0
        assert calc_rise_fall_asymmetry(turb_list) == pytest.approx(0.0, abs=0.05)

        # n_local_peaks = 1 (single spike) → 0.1
        assert calc_n_local_peaks(turb_list) == pytest.approx(0.1, abs=0.05)

    def test_synthetic_constant_signal(self):
        """Constant signal: all features should have predictable values."""
        n = 200
        turb_list = [5.0] * n

        assert calc_event_duration(turb_list) == pytest.approx(1.0)
        assert calc_peak_to_mean_ratio(turb_list) == pytest.approx(1.0)
        assert calc_pre_post_energy_ratio(turb_list) == pytest.approx(1.0, abs=0.01)
        assert calc_n_local_peaks(turb_list) == pytest.approx(0.0)

    def test_synthetic_ramp_signal(self):
        """Linear ramp: peak at end, asymmetry near +1."""
        n = 100
        turb_list = [float(i) for i in range(n)]

        # Peak at index 99 → peak_position = 99/99 = 1.0
        assert calc_peak_position(turb_list) == pytest.approx(1.0)

        # For peak at end: rise >> fall => asymmetry near +1
        assert calc_rise_fall_asymmetry(turb_list) > 0.95

        # Pre-post energy ratio: first half energy < second half → ratio < 1
        assert calc_pre_post_energy_ratio(turb_list) < 1.0

    def test_synthetic_phases_constant(self):
        """Constant phases: all phase features should be zero or near-zero."""
        n = 50
        phases_list = [[0.5, 0.5, 0.5, 0.5]] * n

        assert calc_phase_diff_var(phases_list) == pytest.approx(0.0)
        assert calc_phase_circular_variance(phases_list) == pytest.approx(0.0)

    def test_synthetic_phases_linear_ramp(self):
        """Linear phase ramp: known variance and range."""
        n = 20
        # Each packet has phases [0, 1, 2, 3] → std, range, diff_var all deterministic
        phases_list = [[0.0, 1.0, 2.0, 3.0]] * n

        # Circular variance > 0 for spread phases.
        assert calc_phase_circular_variance(phases_list) > 0.0

        # Phase diff var: diffs = [1, 1, 1], mean_diff = 1, var = 0
        assert calc_phase_diff_var(phases_list) == pytest.approx(0.0)
