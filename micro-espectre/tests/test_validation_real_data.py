"""
Micro-ESPectre - Validation Tests with Real CSI Data

Tests that validate algorithm performance using real CSI data from data/.
These tests verify that algorithms produce expected results on actual captured data.

Configuration is aligned with C++ tests (test_motion_detection.cpp):
- window_size = DETECTOR_DEFAULT_WINDOW_SIZE (75)
- warmup = DETECTOR_DEFAULT_WINDOW_SIZE (buffer must be full before detection)
- adaptive_factor = 1.0 (DEFAULT_ADAPTIVE_FACTOR)
- enable_hampel = false
- CV normalization for ESP32 and C3 (needs_cv_normalization())
- MVS targets: 97% recall, 10% FP rate
- ML targets: 93% recall, 10% FP rate
- Baseline packets: first 300 skipped (GAIN_LOCK_SKIP)

Converted from:
- tools/11_test_band_selection.py (algorithm validation)
- tools/12_test_csi_features.py (Feature extraction validation)
- tools/14_test_publish_time_features.py (Publish-time features)
- tools/10_test_retroactive_calibration.py (Calibration validation)

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import pytest

# ============================================================================
# Detector Constants (match C++ defaults from base_detector.h)
# ============================================================================

# Default window size for motion detection (matches C++ DETECTOR_DEFAULT_WINDOW_SIZE)
# This is also used as warmup: we skip the first window_size packets in evaluation
# because the detector needs a full buffer before it can produce valid predictions.
DETECTOR_DEFAULT_WINDOW_SIZE = 75
import numpy as np
import math
import os
import tempfile
from pathlib import Path

# Patch buffer file path BEFORE importing calibrators
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
import nbvi_calibrator
nbvi_calibrator.BUFFER_FILE = os.path.join(tempfile.gettempdir(), 'nbvi_buffer_validation_test.bin')

# Import from src and tools
from segmentation import SegmentationContext
from features import (
    calc_skewness, calc_kurtosis, calc_entropy_turb,
    calc_zero_crossing_rate, calc_mad,
)
from filters import HampelFilter
from csi_utils import (
    load_baseline_and_movement, calculate_spatial_turbulence,
    calculate_variance_two_pass, MVSDetector
)


# ============================================================================
# Data Directory
# ============================================================================

DATA_DIR = Path(__file__).parent.parent / 'data'


# ============================================================================
# Dataset Configuration
# ============================================================================

def get_available_datasets():
    """Get list of available datasets (HT20: 64 SC only)"""
    from csi_utils import find_dataset
    datasets = []
    
    # C3 64 SC dataset (HT20) - uses high-sensitivity band [18-29]
    try:
        baseline_c3, movement_c3, _ = find_dataset(chip='C3', num_sc=64)
        datasets.append(pytest.param(
            (baseline_c3, movement_c3, 64, 'C3'),
            id="c3_64sc"
        ))
    except FileNotFoundError:
        pass
    
    # C6 64 SC dataset (HT20)
    try:
        baseline_c6, movement_c6, _ = find_dataset(chip='C6', num_sc=64)
        datasets.append(pytest.param(
            (baseline_c6, movement_c6, 64, 'C6'),
            id="c6_64sc"
        ))
    except FileNotFoundError:
        pass
    
    # ESP32 64 SC dataset (HT20)
    try:
        baseline_esp32, movement_esp32, _ = find_dataset(chip='ESP32', num_sc=64)
        datasets.append(pytest.param(
            (baseline_esp32, movement_esp32, 64, 'ESP32'),
            id="esp32_64sc"
        ))
    except FileNotFoundError:
        pass
    
    # S3 64 SC dataset (HT20)
    try:
        baseline_s3, movement_s3, _ = find_dataset(chip='S3', num_sc=64)
        datasets.append(pytest.param(
            (baseline_s3, movement_s3, 64, 'S3'),
            id="s3_64sc"
        ))
    except FileNotFoundError:
        pass
    
    return datasets


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(params=get_available_datasets())
def dataset_config(request):
    """
    Parametrized fixture that provides dataset configuration.
    Tests using this fixture will run once per available dataset.
    
    Returns:
        tuple: (baseline_path, movement_path, num_subcarriers, chip)
    """
    return request.param


@pytest.fixture
def real_data(dataset_config):
    """Load real CSI data from the current dataset.
    
    Matches C++ behavior (csi_test_data.h):
    - Baseline: first 300 packets skipped (GAIN_LOCK_SKIP) for radio warm-up
    - Movement: all packets loaded
    """
    from csi_utils import load_npz_as_packets
    baseline_path, movement_path, num_sc, chip = dataset_config
    
    # Match C++ GAIN_LOCK_SKIP = 300 (skip radio warm-up noise in baseline)
    GAIN_LOCK_SKIP = 300
    
    baseline_packets = load_npz_as_packets(baseline_path)
    movement_packets = load_npz_as_packets(movement_path)
    
    # Skip first GAIN_LOCK_SKIP baseline packets (matches C++ behavior)
    baseline_packets = baseline_packets[GAIN_LOCK_SKIP:]
    
    return baseline_packets, movement_packets


@pytest.fixture
def num_subcarriers(dataset_config):
    """Get number of subcarriers for current dataset"""
    _, _, num_sc, _ = dataset_config
    return num_sc


@pytest.fixture
def chip_type(dataset_config):
    """Get chip type for current dataset"""
    _, _, _, chip = dataset_config
    return chip


@pytest.fixture
def window_size(chip_type):
    """Get optimal window size for chip type.
    
    All chips use the same window size for consistent behavior.
    This matches the production default DETECTOR_DEFAULT_WINDOW_SIZE.
    """
    return DETECTOR_DEFAULT_WINDOW_SIZE


@pytest.fixture(params=["nbvi"])
def calibration_algorithm(request, chip_type):
    """
    Parametrized fixture for calibration algorithm.
    Tests using this fixture will run once per algorithm.
    NBVI is the sole calibration algorithm.
    """
    algo = request.param
    return algo


@pytest.fixture
def use_cv_normalization(chip_type):
    """Get CV normalization setting for chip type.
    
    Matches C++ needs_cv_normalization():
    - ESP32: hardware doesn't support gain lock
    - C3: current dataset was collected without gain lock (signal too strong)
    """
    return chip_type in ('ESP32', 'C3')


@pytest.fixture
def fp_rate_target(chip_type):
    """Get MVS target FP rate for chip type.
    
    Matches C++ get_fp_rate_target(): 10.0f for all chips.
    """
    return 10.0


@pytest.fixture
def recall_target(chip_type):
    """Get recall target for chip type.
    
    Matches C++ get_recall_target(): 97.0f for all chips.
    """
    return 97.0


@pytest.fixture
def ml_fp_rate_target(chip_type):
    """Get ML-specific FP rate target for chip type.
    
    Matches C++ get_ml_fp_rate_target(): 10.0f for all chips.
    """
    return 10.0


@pytest.fixture
def ml_recall_target(chip_type):
    """Get ML-specific recall target for chip type.
    
    Matches C++ get_ml_recall_target(): 93.0f for all chips.
    """
    return 93.0


@pytest.fixture
def enable_hampel(chip_type):
    """Enable Hampel filter for chip type.
    
    Matches C++ get_enable_hampel(): false for all chips.
    """
    return False


@pytest.fixture
def baseline_amplitudes(real_data, default_subcarriers):
    """Extract amplitudes from baseline packets"""
    baseline_packets, _ = real_data
    
    all_amplitudes = []
    for pkt in baseline_packets:
        csi_data = pkt['csi_data']
        amps = []
        for sc_idx in default_subcarriers:
            # Espressif CSI format: [Imaginary, Real, ...] per subcarrier
            q_idx = sc_idx * 2      # Imaginary first
            i_idx = sc_idx * 2 + 1  # Real second
            if i_idx < len(csi_data):
                I = float(csi_data[i_idx])
                Q = float(csi_data[q_idx])
                amps.append(math.sqrt(I**2 + Q**2))
        all_amplitudes.append(amps)
    
    return np.array(all_amplitudes)


@pytest.fixture
def movement_amplitudes(real_data, default_subcarriers):
    """Extract amplitudes from movement packets"""
    _, movement_packets = real_data
    
    all_amplitudes = []
    for pkt in movement_packets:
        csi_data = pkt['csi_data']
        amps = []
        for sc_idx in default_subcarriers:
            # Espressif CSI format: [Imaginary, Real, ...] per subcarrier
            q_idx = sc_idx * 2      # Imaginary first
            i_idx = sc_idx * 2 + 1  # Real second
            if i_idx < len(csi_data):
                I = float(csi_data[i_idx])
                Q = float(csi_data[q_idx])
                amps.append(math.sqrt(I**2 + Q**2))
        all_amplitudes.append(amps)
    
    return np.array(all_amplitudes)


# ============================================================================
# MVS Detection Tests
# ============================================================================

def run_nbvi_calibration(baseline_packets, num_subcarriers, hint_band=None):
    """
    Run NBVI calibration exactly as in production.
    
    Note: baseline_packets is assumed to already have GAIN_LOCK_SKIP packets
    removed (done in real_data fixture to match C++ csi_test_data.h behavior).
    
    Args:
        baseline_packets: List of baseline CSI packets (already gain-lock skipped)
        num_subcarriers: Number of subcarriers
        hint_band: Optional subcarrier band to use when searching for baseline
                   candidate windows. Matches C++ start_calibration(current_band).
    
    Returns:
        tuple: (selected_band, adaptive_threshold)
    """
    from nbvi_calibrator import NBVICalibrator
    from threshold import calculate_adaptive_threshold
    
    # Use first 700 packets for calibration (gain lock skip already done in fixture)
    buffer_size = min(700, len(baseline_packets))
    
    calibrator = NBVICalibrator(buffer_size=buffer_size)
    
    # Feed baseline packets for calibration
    for pkt in baseline_packets[:buffer_size]:
        csi_bytes = bytes(int(x) & 0xFF for x in pkt['csi_data'])
        calibrator.add_packet(csi_bytes)
    
    # Run calibration (NBVI-based algorithm)
    # Pass hint_band to match C++ behavior where start_calibration() receives current_band
    selected_band, mv_values = calibrator.calibrate(hint_band=hint_band)
    calibrator.free_buffer()
    
    # Calculate adaptive threshold from mv_values
    if selected_band is not None and len(mv_values) > 0:
        adaptive_threshold, _ = calculate_adaptive_threshold(mv_values, "auto")
    else:
        adaptive_threshold = 1.0
    
    return selected_band, adaptive_threshold


def run_calibration(baseline_packets, num_subcarriers, algorithm="nbvi", hint_band=None):
    """
    Run calibration using NBVI algorithm.
    
    Args:
        baseline_packets: List of baseline CSI packets
        num_subcarriers: Number of subcarriers
        algorithm: Calibration algorithm (only "nbvi" supported)
        hint_band: Optional subcarrier band to use as hint for calibration.
                   Matches C++ start_calibration(current_band) behavior.
    
    Returns:
        tuple: (selected_band, adaptive_threshold)
    """
    return run_nbvi_calibration(baseline_packets, num_subcarriers, hint_band=hint_band)


def run_calibration_with_cv(baseline_packets, window_size, selected_band, use_cv_normalization):
    """
    Calculate adaptive threshold with optional CV normalization.
    
    For chips that need CV normalization (ESP32, C3), we must calculate
    the threshold manually since NBVI calibrator doesn't support CV normalization.
    
    Note: baseline_packets is assumed to already have GAIN_LOCK_SKIP packets
    removed (done in real_data fixture to match C++ csi_test_data.h behavior).
    
    Args:
        baseline_packets: List of baseline CSI packets (already gain-lock skipped)
        window_size: Window size for moving variance
        selected_band: Pre-selected subcarriers (optimal for chip)
        use_cv_normalization: Whether to use CV normalization
    
    Returns:
        float: Adaptive threshold
    """
    import numpy as np
    
    # Use first 700 packets for calibration (gain lock skip already done in fixture)
    turbs = []
    for pkt in baseline_packets[:700]:
        turb, _ = SegmentationContext.compute_spatial_turbulence(
            pkt['csi_data'], selected_band, use_cv_normalization=use_cv_normalization
        )
        turbs.append(turb)
    
    # Calculate moving variance
    mv_values = []
    for i in range(window_size, len(turbs)):
        window = turbs[i-window_size:i]
        mv = sum((x - sum(window)/len(window))**2 for x in window) / len(window)
        mv_values.append(mv)
    
    # Use P95 percentile (matches DEFAULT_ADAPTIVE_FACTOR = 1.0)
    p95 = np.percentile(mv_values, 95)
    return p95


class TestMVSDetectionRealData:
    """Test MVS motion detection with real CSI data using NBVI calibration"""
    
    def test_baseline_low_motion_rate(self, real_data, num_subcarriers, window_size, fp_rate_target, enable_hampel, calibration_algorithm, chip_type, use_cv_normalization, default_subcarriers):
        """Test that baseline data produces low motion detection rate"""
        
        baseline_packets, _ = real_data
        
        # Use appropriate calibration based on CV normalization need
        if use_cv_normalization:
            selected_band = default_subcarriers
            adaptive_threshold = run_calibration_with_cv(baseline_packets, window_size, selected_band, use_cv_normalization)
        else:
            # Pass default_subcarriers as hint_band (matches C++ start_calibration behavior)
            selected_band, adaptive_threshold = run_calibration(baseline_packets, num_subcarriers, calibration_algorithm, hint_band=default_subcarriers)
        
        ctx = SegmentationContext(window_size=window_size, threshold=adaptive_threshold, enable_hampel=enable_hampel)
        ctx.use_cv_normalization = use_cv_normalization
        
        motion_count = 0
        for pkt in baseline_packets:
            turb = ctx.calculate_spatial_turbulence(pkt['csi_data'], selected_band)
            ctx.add_turbulence(turb)
            ctx.update_state()  # Lazy evaluation: must call to update state
            if ctx.get_state() == SegmentationContext.STATE_MOTION:
                motion_count += 1
        
        # Skip warmup period
        effective_packets = len(baseline_packets) - DETECTOR_DEFAULT_WINDOW_SIZE
        motion_rate = motion_count / effective_packets if effective_packets > 0 else 0
        
        # Target: < fp_rate_target% FP rate (chip-specific)
        target_rate = fp_rate_target / 100.0
        assert motion_rate < target_rate, f"[{calibration_algorithm}] Baseline motion rate too high: {motion_rate:.1%} (target: <{fp_rate_target}%)"
    
    def test_movement_high_motion_rate(self, real_data, num_subcarriers, window_size, enable_hampel, calibration_algorithm, chip_type, use_cv_normalization, default_subcarriers):
        """Test that movement data produces high motion detection rate"""
        
        baseline_packets, movement_packets = real_data
        
        # Use appropriate calibration based on CV normalization need
        if use_cv_normalization:
            selected_band = default_subcarriers
            adaptive_threshold = run_calibration_with_cv(baseline_packets, window_size, selected_band, use_cv_normalization)
        else:
            # Pass default_subcarriers as hint_band (matches C++ start_calibration behavior)
            selected_band, adaptive_threshold = run_calibration(baseline_packets, num_subcarriers, calibration_algorithm, hint_band=default_subcarriers)
        
        ctx = SegmentationContext(window_size=window_size, threshold=adaptive_threshold, enable_hampel=enable_hampel)
        ctx.use_cv_normalization = use_cv_normalization
        
        motion_count = 0
        for pkt in movement_packets:
            turb = ctx.calculate_spatial_turbulence(pkt['csi_data'], selected_band)
            ctx.add_turbulence(turb)
            ctx.update_state()  # Lazy evaluation: must call to update state
            if ctx.get_state() == SegmentationContext.STATE_MOTION:
                motion_count += 1
        
        # Skip warmup period
        effective_packets = len(movement_packets) - DETECTOR_DEFAULT_WINDOW_SIZE
        motion_rate = motion_count / effective_packets if effective_packets > 0 else 0
        
        # Target: > 97% recall (matches C++ get_recall_target())
        assert motion_rate > 0.97, f"[{calibration_algorithm}] Movement motion rate too low: {motion_rate:.1%} (target: >97%)"
    
    def test_mvs_detector_wrapper(self, real_data, num_subcarriers, window_size, calibration_algorithm, chip_type, use_cv_normalization, default_subcarriers):
        """Test MVSDetector wrapper class with calibration"""
        
        baseline_packets, movement_packets = real_data
        
        # Use appropriate calibration based on CV normalization need
        if use_cv_normalization:
            selected_band = default_subcarriers
            adaptive_threshold = run_calibration_with_cv(baseline_packets, window_size, selected_band, use_cv_normalization)
        else:
            # Pass default_subcarriers as hint_band (matches C++ start_calibration behavior)
            selected_band, adaptive_threshold = run_calibration(baseline_packets, num_subcarriers, calibration_algorithm, hint_band=default_subcarriers)
        
        # Test with the calibrated band and adaptive threshold
        # Note: csi_utils.MVSDetector has different signature than src.mvs_detector.MVSDetector
        detector = MVSDetector(
            window_size=window_size,
            threshold=adaptive_threshold,
            selected_subcarriers=selected_band,
            track_data=True
        )
        # csi_utils.MVSDetector internally uses SegmentationContext
        detector._context.use_cv_normalization = use_cv_normalization
        
        for pkt in baseline_packets:
            detector.process_packet(pkt['csi_data'])
        
        baseline_motion = detector.get_motion_count()
        
        # Reset and test on movement
        detector.reset()
        
        for pkt in movement_packets:
            detector.process_packet(pkt['csi_data'])
        
        movement_motion = detector.get_motion_count()
        
        # Movement should have significantly more motion packets
        assert movement_motion > baseline_motion * 2


# ============================================================================
# Feature Separation Tests
# ============================================================================

def fishers_criterion(values_class1, values_class2):
    """
    Calculate Fisher's criterion for class separability.
    
    J = (μ₁ - μ₂)² / (σ₁² + σ₂²)
    
    Higher J = better separation between classes.
    """
    mu1 = np.mean(values_class1)
    mu2 = np.mean(values_class2)
    var1 = np.var(values_class1)
    var2 = np.var(values_class2)
    
    # Use very small epsilon to handle near-zero variances
    # CV-normalized turbulence produces very small variance values (1e-14 to 1e-11)
    # but can still show good separation (Fisher J > 1.0)
    if var1 + var2 < 1e-20:
        return 0.0
    
    return (mu1 - mu2) ** 2 / (var1 + var2)


class TestFeatureSeparationRealData:
    """Test feature separation between baseline and movement"""
    
    def test_skewness_separation(self, baseline_amplitudes, movement_amplitudes):
        """Test that skewness shows separation between baseline and movement"""
        baseline_skew = [calc_skewness(list(r), len(r), float(np.mean(r)), float(np.std(r))) for r in baseline_amplitudes]
        movement_skew = [calc_skewness(list(r), len(r), float(np.mean(r)), float(np.std(r))) for r in movement_amplitudes]
        
        J = fishers_criterion(baseline_skew, movement_skew)
        
        # Should have some separation
        # Note: Skewness is not the primary detection method (MVS is)
        # so we only require minimal separation to confirm the feature works
        assert J > 0.0001, f"Skewness Fisher's J too low: {J:.6f}"
    
    def test_kurtosis_separation(self, baseline_amplitudes, movement_amplitudes):
        """Test that kurtosis shows separation between baseline and movement"""
        baseline_kurt = [calc_kurtosis(list(r), len(r), float(np.mean(r)), float(np.std(r))) for r in baseline_amplitudes]
        movement_kurt = [calc_kurtosis(list(r), len(r), float(np.mean(r)), float(np.std(r))) for r in movement_amplitudes]
        
        J = fishers_criterion(baseline_kurt, movement_kurt)
        
        # Should have some separation
        # Note: Kurtosis is not the primary detection method (MVS is)
        # so we only require minimal separation to confirm the feature works
        assert J > 0.0001, f"Kurtosis Fisher's J too low: {J:.6f}"
    
    def test_turbulence_variance_separation(self, real_data, default_subcarriers, chip_type, use_cv_normalization, window_size):
        """Test that turbulence variance separates baseline from movement.
        
        Uses CV normalization for chips that need it (ESP32, C3),
        matching C++ needs_cv_normalization() behavior.
        """
        baseline_packets, movement_packets = real_data
        
        # Calculate turbulence for each packet using CV normalization where needed
        baseline_turb = []
        for pkt in baseline_packets:
            turb, _ = SegmentationContext.compute_spatial_turbulence(
                pkt['csi_data'], default_subcarriers, use_cv_normalization=use_cv_normalization
            )
            baseline_turb.append(turb)
        
        movement_turb = []
        for pkt in movement_packets:
            turb, _ = SegmentationContext.compute_spatial_turbulence(
                pkt['csi_data'], default_subcarriers, use_cv_normalization=use_cv_normalization
            )
            movement_turb.append(turb)
        
        # Calculate variance of turbulence over windows (use window_size from C++ config)
        analysis_window = window_size
        
        def window_variances(values, ws):
            variances = []
            for i in range(0, len(values) - ws, ws // 2):
                window = values[i:i + ws]
                variances.append(calculate_variance_two_pass(window))
            return variances
        
        baseline_vars = window_variances(baseline_turb, analysis_window)
        movement_vars = window_variances(movement_turb, analysis_window)
        
        if len(baseline_vars) > 0 and len(movement_vars) > 0:
            J = fishers_criterion(baseline_vars, movement_vars)
            
            # Variance should show good separation (this is the core of MVS)
            assert J > 0.5, f"Turbulence variance Fisher's J too low: {J:.3f}"


# ============================================================================
# Publish-Time Features Tests
# ============================================================================

class TestPublishTimeFeaturesRealData:
    """Test publish-time feature extraction with real data"""
    
    def test_mad_turb_separation(self, real_data, default_subcarriers, window_size, chip_type, use_cv_normalization):
        """Test MAD of turbulence buffer separates baseline from movement"""
        
        baseline_packets, movement_packets = real_data
        ws = window_size
        
        def calculate_mad_values(packets):
            ctx = SegmentationContext(window_size=ws, threshold=1.0)
            ctx.use_cv_normalization = use_cv_normalization
            mad_values = []
            
            for pkt in packets:
                turb = ctx.calculate_spatial_turbulence(pkt['csi_data'], default_subcarriers)
                ctx.add_turbulence(turb)
                
                if ctx.buffer_count >= ws:
                    mad = calc_mad(ctx.turbulence_buffer, ctx.buffer_count)
                    mad_values.append(mad)
            
            return mad_values
        
        baseline_mad = calculate_mad_values(baseline_packets)
        movement_mad = calculate_mad_values(movement_packets)
        
        if len(baseline_mad) > 0 and len(movement_mad) > 0:
            J = fishers_criterion(baseline_mad, movement_mad)
            
            # MAD should show good separation (S3 has lower separation due to noisier baseline)
            min_j = 0.3 if chip_type == 'S3' else 0.5
            assert J > min_j, f"MAD Fisher's J too low: {J:.3f} (target: >{min_j})"
    
    def test_entropy_turb_separation(self, real_data, default_subcarriers, window_size):
        """Test entropy of turbulence buffer separates baseline from movement"""
        baseline_packets, movement_packets = real_data
        ws = window_size
        
        def calculate_entropy_values(packets):
            ctx = SegmentationContext(window_size=ws, threshold=1.0)
            entropy_values = []
            
            for pkt in packets:
                turb = ctx.calculate_spatial_turbulence(pkt['csi_data'], default_subcarriers)
                ctx.add_turbulence(turb)
                
                if ctx.buffer_count >= ws:
                    entropy = calc_entropy_turb(ctx.turbulence_buffer, ctx.buffer_count)
                    entropy_values.append(entropy)
            
            return entropy_values
        
        baseline_entropy = calculate_entropy_values(baseline_packets)
        movement_entropy = calculate_entropy_values(movement_packets)
        
        if len(baseline_entropy) > 0 and len(movement_entropy) > 0:
            J = fishers_criterion(baseline_entropy, movement_entropy)
            
            # Entropy should show some separation
            assert J > 0.1, f"Entropy Fisher's J too low: {J:.3f}"


# ============================================================================
# Hampel Filter Tests with Real Data
# ============================================================================

class TestHampelFilterRealData:
    """Test Hampel filter with real CSI turbulence data"""
    
    def test_hampel_reduces_spikes(self, real_data, default_subcarriers):
        """Test that Hampel filter reduces turbulence spikes"""
        baseline_packets, movement_packets = real_data
        all_packets = baseline_packets + movement_packets
        
        # Calculate raw turbulence
        raw_turbulence = []
        for pkt in all_packets:
            turb = calculate_spatial_turbulence(pkt['csi_data'], default_subcarriers)
            raw_turbulence.append(turb)
        
        # Apply Hampel filter
        hf = HampelFilter(window_size=7, threshold=4.0)
        filtered_turbulence = [hf.filter(t) for t in raw_turbulence]
        
        # Filtered should have lower max (spikes reduced)
        raw_max = max(raw_turbulence)
        filtered_max = max(filtered_turbulence)
        
        # If there were spikes, they should be reduced
        if raw_max > np.mean(raw_turbulence) * 3:
            assert filtered_max <= raw_max, "Hampel should not increase max value"
    
    def test_hampel_preserves_variance_separation(self, real_data, default_subcarriers):
        """Test that Hampel filter preserves baseline/movement separation"""
        baseline_packets, movement_packets = real_data
        
        # Calculate filtered turbulence for baseline
        hf_baseline = HampelFilter(window_size=7, threshold=4.0)
        baseline_turb = []
        for pkt in baseline_packets:
            turb = calculate_spatial_turbulence(pkt['csi_data'], default_subcarriers)
            filtered = hf_baseline.filter(turb)
            baseline_turb.append(filtered)
        
        # Calculate filtered turbulence for movement
        hf_movement = HampelFilter(window_size=7, threshold=4.0)
        movement_turb = []
        for pkt in movement_packets:
            turb = calculate_spatial_turbulence(pkt['csi_data'], default_subcarriers)
            filtered = hf_movement.filter(turb)
            movement_turb.append(filtered)
        
        # Movement should still have higher variance
        baseline_var = np.var(baseline_turb)
        movement_var = np.var(movement_turb)
        
        assert movement_var > baseline_var, \
            f"Movement variance ({movement_var:.3f}) should be > baseline ({baseline_var:.3f})"


# ============================================================================
# Performance Metrics Tests
# ============================================================================

class TestPerformanceMetrics:
    """Test that we achieve expected performance metrics with NBVI calibration"""
    
    def test_mvs_optimal_subcarriers(self, real_data, window_size, fp_rate_target, recall_target, enable_hampel, chip_type, default_subcarriers, use_cv_normalization):
        """
        Test MVS motion detection with optimal (offline-tuned) subcarriers.
        
        This is the "best case" reference test - uses pre-calculated optimal
        subcarriers for each chip (matches C++ test_mvs_optimal_subcarriers).
        
        No NBVI calibration is used - subcarriers are fixed from conftest.py.
        
        Target: >97% Recall, <10% FP Rate for all chips.
        """
        import numpy as np
        from threshold import calculate_adaptive_threshold
        
        baseline_packets, movement_packets = real_data
        
        # Use optimal subcarriers (from conftest.py, matches C++ configuration)
        selected_band = default_subcarriers
        
        # Calculate adaptive threshold using a calibration detector (matches C++)
        # C++ uses a separate cal_detector to collect MV values
        cal_ctx = SegmentationContext(
            window_size=window_size, threshold=1.0, enable_hampel=enable_hampel
        )
        cal_ctx.use_cv_normalization = use_cv_normalization
        
        mv_values = []
        calibration_packets = min(len(baseline_packets), 700)
        for pkt in baseline_packets[:calibration_packets]:
            turb = cal_ctx.calculate_spatial_turbulence(pkt['csi_data'], selected_band)
            cal_ctx.add_turbulence(turb)
            cal_ctx.update_state()
            # Check if buffer is ready (buffer_count >= window_size)
            if cal_ctx.buffer_count >= cal_ctx.window_size:
                mv_values.append(cal_ctx.current_moving_variance)
        
        # Use P95 percentile (matches C++ calculate_adaptive_threshold)
        adaptive_threshold, _ = calculate_adaptive_threshold(mv_values, threshold_mode="auto")
        
        # Initialize with adaptive threshold (new detector, matches C++)
        ctx = SegmentationContext(
            window_size=window_size, threshold=adaptive_threshold, enable_hampel=enable_hampel
        )
        ctx.use_cv_normalization = use_cv_normalization
        
        num_baseline = len(baseline_packets)
        num_movement = len(movement_packets)
        
        # Process baseline (expecting IDLE)
        baseline_motion_packets = 0
        for pkt in baseline_packets:
            turb = ctx.calculate_spatial_turbulence(pkt['csi_data'], selected_band)
            ctx.add_turbulence(turb)
            ctx.update_state()
            if ctx.get_state() == SegmentationContext.STATE_MOTION:
                baseline_motion_packets += 1
        
        # Process movement (expecting MOTION, continue in same context)
        movement_with_motion = 0
        movement_without_motion = 0
        for pkt in movement_packets:
            turb = ctx.calculate_spatial_turbulence(pkt['csi_data'], selected_band)
            ctx.add_turbulence(turb)
            ctx.update_state()
            if ctx.get_state() == SegmentationContext.STATE_MOTION:
                movement_with_motion += 1
            else:
                movement_without_motion += 1
        
        # Calculate metrics
        pkt_tp = movement_with_motion
        pkt_fn = movement_without_motion
        pkt_tn = num_baseline - baseline_motion_packets
        pkt_fp = baseline_motion_packets
        
        pkt_recall = pkt_tp / (pkt_tp + pkt_fn) * 100.0 if (pkt_tp + pkt_fn) > 0 else 0
        pkt_precision = pkt_tp / (pkt_tp + pkt_fp) * 100.0 if (pkt_tp + pkt_fp) > 0 else 0
        pkt_fp_rate = pkt_fp / num_baseline * 100.0 if num_baseline > 0 else 0
        pkt_f1 = 2 * (pkt_precision / 100) * (pkt_recall / 100) / ((pkt_precision + pkt_recall) / 100) * 100 if (pkt_precision + pkt_recall) > 0 else 0
        
        print(f"\n  * Recall:     {pkt_recall:.1f}% (target: >{recall_target}%)")
        print(f"  * Precision:  {pkt_precision:.1f}%")
        print(f"  * FP Rate:    {pkt_fp_rate:.1f}% (target: <{fp_rate_target}%)")
        print(f"  * F1-Score:   {pkt_f1:.1f}%")
        
        # Record results for summary table
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        from conftest import record_performance
        record_performance(chip_type, 'mvs_optimal', pkt_recall, pkt_fp_rate, pkt_precision, pkt_f1)
        
        # Assertions
        assert pkt_recall > recall_target, f"Recall too low: {pkt_recall:.1f}% (target: >{recall_target}%)"
        assert pkt_fp_rate < fp_rate_target, f"FP Rate too high: {pkt_fp_rate:.1f}% (target: <{fp_rate_target}%)"

    def test_mvs_detection_accuracy(self, real_data, num_subcarriers, window_size, fp_rate_target, recall_target, enable_hampel, calibration_algorithm, chip_type, default_subcarriers, use_cv_normalization):
        """
        Test MVS motion detection accuracy with real CSI data.
        
        This test uses auto-calibration exactly as in production:
        - Band selection from baseline data (NBVI) for chips with gain lock
        - Optimal subcarriers for chips without gain lock (ESP32, C3)
        - Adaptive threshold from calibration
        - Process ALL packets (no warmup skip)
        - Process baseline first, then movement (continuous context)
        - Unified window_size (75) and adaptive threshold (P95 percentile)
        - CV normalization for ESP32 and C3 (no gain lock / skipped gain lock)
        
        Target: >97% Recall, <10% FP Rate for all chips.
        """
        import numpy as np
        from threshold import calculate_adaptive_threshold
        
        baseline_packets, movement_packets = real_data
        
        # Chips with CV normalization need optimal subcarriers since NBVI calibrator
        # doesn't support CV normalization. For chips with gain lock, use NBVI.
        if use_cv_normalization:
            # Use optimal subcarriers (from conftest.py, matches C++ configuration)
            selected_band = default_subcarriers
            
            # Calculate adaptive threshold using a calibration detector (matches C++)
            cal_ctx = SegmentationContext(
                window_size=window_size, threshold=1.0, enable_hampel=enable_hampel
            )
            cal_ctx.use_cv_normalization = use_cv_normalization
            
            mv_values = []
            calibration_packets = min(len(baseline_packets), 700)
            for pkt in baseline_packets[:calibration_packets]:
                turb = cal_ctx.calculate_spatial_turbulence(pkt['csi_data'], selected_band)
                cal_ctx.add_turbulence(turb)
                cal_ctx.update_state()
                # Check if buffer is ready (buffer_count >= window_size)
                if cal_ctx.buffer_count >= cal_ctx.window_size:
                    mv_values.append(cal_ctx.current_moving_variance)
            
            # Use P95 percentile (matches C++ calculate_adaptive_threshold)
            adaptive_threshold, _ = calculate_adaptive_threshold(mv_values, threshold_mode="auto")
        else:
            # Use NBVI calibration for chips with gain lock (C6, S3)
            # Pass default_subcarriers as hint_band (matches C++ start_calibration behavior)
            selected_band, adaptive_threshold = run_calibration(baseline_packets, num_subcarriers, calibration_algorithm, hint_band=default_subcarriers)
        
        # Initialize with adaptive threshold from calibration
        ctx = SegmentationContext(
            window_size=window_size, threshold=adaptive_threshold, enable_hampel=enable_hampel
        )
        # Set CV normalization based on chip
        ctx.use_cv_normalization = use_cv_normalization
        
        num_baseline = len(baseline_packets)
        num_movement = len(movement_packets)
        
        # ========================================
        # Process baseline (expecting IDLE)
        # ========================================
        baseline_motion_packets = 0
        
        for pkt in baseline_packets:
            turb = ctx.calculate_spatial_turbulence(pkt['csi_data'], selected_band)
            ctx.add_turbulence(turb)
            ctx.update_state()  # Lazy evaluation: must call to update state
            if ctx.get_state() == SegmentationContext.STATE_MOTION:
                baseline_motion_packets += 1
        
        # ========================================
        # Process movement (expecting MOTION)
        # Continue with same context (no reset)
        # ========================================
        movement_with_motion = 0
        movement_without_motion = 0
        
        for pkt in movement_packets:
            turb = ctx.calculate_spatial_turbulence(pkt['csi_data'], selected_band)
            ctx.add_turbulence(turb)
            ctx.update_state()  # Lazy evaluation: must call to update state
            if ctx.get_state() == SegmentationContext.STATE_MOTION:
                movement_with_motion += 1
            else:
                movement_without_motion += 1
        
        # ========================================
        # Calculate metrics (same as C++)
        # ========================================
        pkt_tp = movement_with_motion
        pkt_fn = movement_without_motion
        pkt_tn = num_baseline - baseline_motion_packets
        pkt_fp = baseline_motion_packets
        
        pkt_recall = pkt_tp / (pkt_tp + pkt_fn) * 100.0 if (pkt_tp + pkt_fn) > 0 else 0
        pkt_precision = pkt_tp / (pkt_tp + pkt_fp) * 100.0 if (pkt_tp + pkt_fp) > 0 else 0
        pkt_fp_rate = pkt_fp / num_baseline * 100.0 if num_baseline > 0 else 0
        pkt_f1 = 2 * (pkt_precision / 100) * (pkt_recall / 100) / ((pkt_precision + pkt_recall) / 100) * 100 if (pkt_precision + pkt_recall) > 0 else 0
        
        # ========================================
        # Print results (same format as C++)
        # ========================================
        print("\n")
        print("=" * 70)
        print("                         TEST SUMMARY")
        print("=" * 70)
        print()
        print(f"CONFUSION MATRIX ({num_baseline} baseline + {num_movement} movement packets):")
        print("                    Predicted")
        print("                IDLE      MOTION")
        print(f"Actual IDLE     {pkt_tn:4d} (TN)  {pkt_fp:4d} (FP)")
        print(f"    MOTION      {pkt_fn:4d} (FN)  {pkt_tp:4d} (TP)")
        print()
        print("MOTION DETECTION METRICS:")
        print(f"  * True Positives (TP):   {pkt_tp}")
        print(f"  * True Negatives (TN):   {pkt_tn}")
        print(f"  * False Positives (FP):  {pkt_fp}")
        print(f"  * False Negatives (FN):  {pkt_fn}")
        print(f"  * Recall:     {pkt_recall:.1f}% (target: >{recall_target}%)")
        print(f"  * Precision:  {pkt_precision:.1f}%")
        print(f"  * FP Rate:    {pkt_fp_rate:.1f}% (target: <{fp_rate_target}%)")
        print(f"  * F1-Score:   {pkt_f1:.1f}%")
        print()
        print("=" * 70)
        
        # Record results for summary table
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        from conftest import record_performance
        record_performance(chip_type, 'mvs_nbvi', pkt_recall, pkt_fp_rate, pkt_precision, pkt_f1)
        
        # ========================================
        # Assertions (chip-specific thresholds)
        # ========================================
        assert pkt_recall > recall_target, f"Recall too low: {pkt_recall:.1f}% (target: >{recall_target}%)"
        assert pkt_fp_rate < fp_rate_target, f"FP Rate too high: {pkt_fp_rate:.1f}% (target: <{fp_rate_target}%)"

    def test_ml_detection_accuracy(self, real_data, num_subcarriers, ml_fp_rate_target, ml_recall_target, chip_type, use_cv_normalization):
        """
        Test ML (Neural Network) motion detection accuracy with real CSI data.
        
        ML uses a pre-trained MLP model for motion classification.
        No calibration needed - uses pre-trained weights.
        
        Note: ML model uses fixed subcarriers [11-22] regardless of chip type.
        CV normalization is enabled for chips without gain lock (ESP32).
        
        Target: >90% Recall, <fp_rate_target% FP Rate
        """
        from ml_detector import MLDetector, ML_SUBCARRIERS
        from detector_interface import MotionState

        baseline_packets, movement_packets = real_data
        
        num_baseline = len(baseline_packets)
        num_movement = len(movement_packets)
        
        # ML model uses fixed subcarriers (must match training)
        ml_subcarriers = ML_SUBCARRIERS
        
        # ========================================
        # Initialize ML Detector (no calibration needed)
        # CV normalization only for chips without gain lock
        # ========================================
        detector = MLDetector(
            threshold=0.5,  # Default probability threshold
            window_size=DETECTOR_DEFAULT_WINDOW_SIZE,
            use_cv_normalization=use_cv_normalization
        )
        
        print(f"\nML Detector initialized")
        print(f"  Threshold: 0.5")
        print(f"  Window size: {DETECTOR_DEFAULT_WINDOW_SIZE} (DETECTOR_DEFAULT_WINDOW_SIZE)")
        print(f"  Subcarriers: {ml_subcarriers} (fixed for ML)")
        
        # ========================================
        # Process ALL baseline packets (first window_size packets are warmup)
        # ========================================
        warmup = DETECTOR_DEFAULT_WINDOW_SIZE
        baseline_motion_packets = 0
        baseline_eval_count = num_baseline - warmup
        
        for i, pkt in enumerate(baseline_packets):
            detector.process_packet(pkt['csi_data'], ml_subcarriers)
            detector.update_state()
            # Only count after warmup
            if i >= warmup and detector.get_state() == MotionState.MOTION:
                baseline_motion_packets += 1
        
        # ========================================
        # Process movement packets (continue without reset, first window_size packets are warmup)
        # ========================================
        movement_warmup = DETECTOR_DEFAULT_WINDOW_SIZE
        movement_with_motion = 0
        movement_without_motion = 0
        movement_eval_count = num_movement - movement_warmup
        
        for i, pkt in enumerate(movement_packets):
            detector.process_packet(pkt['csi_data'], ml_subcarriers)
            detector.update_state()
            # Only count after warmup
            if i >= movement_warmup:
                if detector.get_state() == MotionState.MOTION:
                    movement_with_motion += 1
                else:
                    movement_without_motion += 1
        
        # ========================================
        # Calculate metrics
        # ========================================
        pkt_tp = movement_with_motion
        pkt_fn = movement_without_motion
        pkt_tn = baseline_eval_count - baseline_motion_packets if baseline_eval_count > 0 else 0
        pkt_fp = baseline_motion_packets
        
        pkt_recall = pkt_tp / (pkt_tp + pkt_fn) * 100.0 if (pkt_tp + pkt_fn) > 0 else 0
        pkt_precision = pkt_tp / (pkt_tp + pkt_fp) * 100.0 if (pkt_tp + pkt_fp) > 0 else 0
        pkt_fp_rate = pkt_fp / baseline_eval_count * 100.0 if baseline_eval_count > 0 else 0
        pkt_f1 = 2 * (pkt_precision / 100) * (pkt_recall / 100) / ((pkt_precision + pkt_recall) / 100) * 100 if (pkt_precision + pkt_recall) > 0 else 0
        
        # ========================================
        # Print results
        # ========================================
        print("\n")
        print("=" * 70)
        print("                     ML DETECTION TEST SUMMARY")
        print("=" * 70)
        print()
        print(f"CONFUSION MATRIX ({baseline_eval_count} baseline + {movement_eval_count} movement packets):")
        print("                    Predicted")
        print("                IDLE      MOTION")
        print(f"Actual IDLE     {pkt_tn:4d} (TN)  {pkt_fp:4d} (FP)")
        print(f"    MOTION      {pkt_fn:4d} (FN)  {pkt_tp:4d} (TP)")
        print()
        print("METRICS:")
        print(f"  * Recall:     {pkt_recall:.1f}% (target: >{ml_recall_target}%)")
        print(f"  * Precision:  {pkt_precision:.1f}%")
        print(f"  * FP Rate:    {pkt_fp_rate:.1f}% (target: <{ml_fp_rate_target}%)")
        print(f"  * F1-Score:   {pkt_f1:.1f}%")
        print()
        print("=" * 70)
        
        # Record results for summary table
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        from conftest import record_performance
        record_performance(chip_type, 'ml', pkt_recall, pkt_fp_rate, pkt_precision, pkt_f1)
        
        # ========================================
        # Assertions
        # ========================================
        assert pkt_recall > ml_recall_target, f"ML Recall too low: {pkt_recall:.1f}% (target: >{ml_recall_target}%)"
        if baseline_eval_count > 0:
            assert pkt_fp_rate < ml_fp_rate_target, f"ML FP Rate too high: {pkt_fp_rate:.1f}% (target: <{ml_fp_rate_target}%)"


# ============================================================================
# Float32 Stability Tests (ESP32 Simulation)
# ============================================================================

class TestFloat32Stability:
    """
    Test numerical stability with float32 precision.
    These tests simulate ESP32 behavior where calculations use 32-bit floats.
    """
    
    def test_turbulence_float32_accuracy(self, real_data, default_subcarriers):
        """Test that float32 turbulence calculation is accurate"""
        baseline_packets, _ = real_data
        
        max_rel_error = 0.0
        
        for pkt in baseline_packets[:200]:
            csi_data = pkt['csi_data']
            
            # Float64 reference (Python default)
            # Espressif CSI format: [Imaginary, Real, ...] per subcarrier
            amplitudes_f64 = []
            for sc_idx in default_subcarriers:
                q_idx = sc_idx * 2      # Imaginary first
                i_idx = sc_idx * 2 + 1  # Real second
                I = float(csi_data[i_idx])
                Q = float(csi_data[q_idx])
                amplitudes_f64.append(math.sqrt(I*I + Q*Q))
            turb_f64 = np.std(amplitudes_f64)
            
            # Float32 simulation (ESP32)
            amplitudes_f32 = []
            for sc_idx in default_subcarriers:
                q_idx = sc_idx * 2      # Imaginary first
                i_idx = sc_idx * 2 + 1  # Real second
                I = np.float32(float(csi_data[i_idx]))
                Q = np.float32(float(csi_data[q_idx]))
                amp = np.sqrt(I*I + Q*Q)
                amplitudes_f32.append(float(amp))
            turb_f32 = np.std(np.array(amplitudes_f32, dtype=np.float32))
            
            if turb_f64 > 0.01:  # Avoid division by near-zero
                rel_error = abs(turb_f32 - turb_f64) / turb_f64
                max_rel_error = max(max_rel_error, rel_error)
        
        # Float32 should be accurate within 0.1% for typical CSI values
        assert max_rel_error < 0.001, \
            f"Float32 turbulence error too high: {max_rel_error:.4%}"
    
    def test_variance_two_pass_vs_single_pass_float32(self, real_data, default_subcarriers):
        """Test that two-pass variance is more stable than single-pass with float32"""
        baseline_packets, _ = real_data
        
        # Generate turbulence values
        turbulences = []
        for pkt in baseline_packets[:100]:
            turb = calculate_spatial_turbulence(pkt['csi_data'], default_subcarriers)
            turbulences.append(turb)
        
        window = turbulences[:50]
        
        # Reference (float64)
        var_ref = np.var(window)
        
        # Two-pass with float32
        window_f32 = np.array(window, dtype=np.float32)
        mean_f32 = np.mean(window_f32)
        var_two_pass = np.mean((window_f32 - mean_f32) ** 2)
        
        # Single-pass with float32 (E[X²] - E[X]²)
        sum_x = np.float32(0.0)
        sum_sq = np.float32(0.0)
        for x in window_f32:
            sum_x += x
            sum_sq += x * x
        n = np.float32(len(window_f32))
        mean_single = sum_x / n
        var_single_pass = (sum_sq / n) - (mean_single * mean_single)
        
        # Both should be close to reference for normal CSI values
        error_two_pass = abs(var_two_pass - var_ref)
        error_single_pass = abs(var_single_pass - var_ref)
        
        # For normal CSI data, both methods should work
        assert error_two_pass < 0.01, f"Two-pass error too high: {error_two_pass}"
        assert error_single_pass < 0.01, f"Single-pass error too high: {error_single_pass}"
    
    def test_csi_range_values_float32_stable(self):
        """Test that float32 is stable within CSI amplitude range (0-200)"""
        # CSI amplitudes are typically 0-200 range - well within float32 precision
        csi_like_values = [30.0 + i * 0.1 for i in range(50)]  # Typical CSI turbulence
        
        # Reference (float64)
        var_ref = np.var(csi_like_values)
        
        # Two-pass with float32
        values_f32 = np.array(csi_like_values, dtype=np.float32)
        mean_f32 = np.mean(values_f32)
        var_two_pass = float(np.mean((values_f32 - mean_f32) ** 2))
        
        # Single-pass with float32
        sum_x = np.float32(0.0)
        sum_sq = np.float32(0.0)
        for x in values_f32:
            sum_x += x
            sum_sq += x * x
        n = np.float32(len(values_f32))
        mean_single = sum_x / n
        var_single_pass = float((sum_sq / n) - (mean_single * mean_single))
        
        # For CSI-range values, both methods should be accurate
        error_two_pass = abs(var_two_pass - var_ref) / var_ref if var_ref > 0 else 0
        error_single_pass = abs(var_single_pass - var_ref) / var_ref if var_ref > 0 else 0
        
        # Both should work for normal CSI values
        assert error_two_pass < 0.001, \
            f"Two-pass error too high for CSI range: {error_two_pass:.4%}"
        assert error_single_pass < 0.001, \
            f"Single-pass error too high for CSI range: {error_single_pass:.4%}"


# ============================================================================
# End-to-End Tests with Band Calibration and Normalization
# ============================================================================

class TestEndToEndWithCalibration:
    """
    Test complete pipeline: Band Calibration → Normalization → MVS Detection
    
    These tests verify that the system works end-to-end with:
    - Auto-calibration selecting subcarriers from real data (NBVI)
    - Adaptive threshold applied to turbulence values
    - MVS motion detection achieving target performance
    """
    
    def test_band_calibration_produces_valid_band(self, real_data, num_subcarriers, calibration_algorithm, chip_type, default_subcarriers):
        """Test that band calibration produces valid subcarrier selection"""
        
        from threshold import calculate_adaptive_threshold
        from src.config import GUARD_BAND_LOW, GUARD_BAND_HIGH, DC_SUBCARRIER
        
        baseline_packets, _ = real_data
        
        # HT20 fixed guard bands (64 SC)
        guard_low = GUARD_BAND_LOW
        guard_high = GUARD_BAND_HIGH
        
        # Run calibration with selected algorithm
        # Pass default_subcarriers as hint_band (matches C++ start_calibration behavior)
        selected_band, adaptive_threshold = run_calibration(baseline_packets, num_subcarriers, calibration_algorithm, hint_band=default_subcarriers)
        
        # Verify calibration results
        assert selected_band is not None, f"[{calibration_algorithm}] Band calibration failed"
        assert len(selected_band) == 12, f"[{calibration_algorithm}] Expected 12 subcarriers, got {len(selected_band)}"
        
        # All subcarriers should be valid (within valid range for this SC count)
        for sc in selected_band:
            assert guard_low <= sc <= guard_high, \
                f"[{calibration_algorithm}] Subcarrier {sc} outside valid range [{guard_low}-{guard_high}]"
        
        # Adaptive threshold should be valid
        assert adaptive_threshold > 0.0, f"[{calibration_algorithm}] Invalid adaptive threshold: {adaptive_threshold}"
        assert 0.1 <= adaptive_threshold <= 10.0, \
            f"[{calibration_algorithm}] Adaptive threshold out of range: {adaptive_threshold}"
        
        print(f"\n[{calibration_algorithm.upper()}] Band Calibration Results:")
        print(f"  Selected band: {selected_band}")
        print(f"  Adaptive threshold: {adaptive_threshold:.4f}")
    
    def test_end_to_end_with_band_calibration_and_mvs(self, real_data, num_subcarriers, window_size, fp_rate_target, recall_target, enable_hampel, calibration_algorithm, chip_type, use_cv_normalization, default_subcarriers):
        """
        Test complete end-to-end flow: Band Calibration → MVS → Detection
        
        This test verifies that the system achieves target performance (>97% Recall, <10% FP)
        when using automatic band selection for optimal subcarrier bands.
        """
        baseline_packets, movement_packets = real_data
        
        # ========================================
        # Step 1: Band Calibration
        # ========================================
        print("\n" + "=" * 70)
        print(f"  END-TO-END TEST: Band Calibration + MVS ({num_subcarriers} SC, {calibration_algorithm.upper()})")
        print("=" * 70)
        
        # Use appropriate calibration based on CV normalization need
        print(f"\nStep 1: {calibration_algorithm.upper()} Band Calibration...")
        if use_cv_normalization:
            selected_band = default_subcarriers
            adaptive_threshold = run_calibration_with_cv(baseline_packets, window_size, selected_band, use_cv_normalization)
            print(f"  Using optimal subcarriers (CV normalization): {selected_band}")
        else:
            # Pass default_subcarriers as hint_band (matches C++ start_calibration behavior)
            selected_band, adaptive_threshold = run_calibration(baseline_packets, num_subcarriers, calibration_algorithm, hint_band=default_subcarriers)
            print(f"  Selected band: {selected_band}")
        
        assert selected_band is not None, f"[{calibration_algorithm}] Band calibration failed for {num_subcarriers} SC"
        print(f"  Adaptive threshold: {adaptive_threshold:.4f}")
        
        # ========================================
        # Step 2: Initialize MVS with calibration results
        # ========================================
        # Initialize MVS with calibration-selected subcarriers AND adaptive threshold
        # This tests the complete production pipeline
        print(f"\nStep 2: Initialize MVS with calibration results (Hampel: {enable_hampel}, CV norm: {use_cv_normalization})...")
        ctx = SegmentationContext(
            window_size=window_size,
            threshold=adaptive_threshold,  # Apply calibration adaptive threshold
            enable_hampel=enable_hampel
        )
        ctx.use_cv_normalization = use_cv_normalization
        
        # ========================================
        # Step 3: Process baseline (expecting IDLE)
        # ========================================
        print("\nStep 3: Process baseline packets (expecting IDLE)...")
        baseline_motion = 0
        
        for pkt in baseline_packets:
            turb = ctx.calculate_spatial_turbulence(pkt['csi_data'], selected_band)
            ctx.add_turbulence(turb)
            ctx.update_state()  # Lazy evaluation: must call to update state
            if ctx.get_state() == SegmentationContext.STATE_MOTION:
                baseline_motion += 1
        
        # ========================================
        # Step 4: Process movement (expecting MOTION)
        # ========================================
        print("Step 4: Process movement packets (expecting MOTION)...")
        movement_motion = 0
        
        for pkt in movement_packets:
            turb = ctx.calculate_spatial_turbulence(pkt['csi_data'], selected_band)
            ctx.add_turbulence(turb)
            ctx.update_state()  # Lazy evaluation: must call to update state
            if ctx.get_state() == SegmentationContext.STATE_MOTION:
                movement_motion += 1
        
        # ========================================
        # Step 5: Calculate metrics
        # ========================================
        num_baseline = len(baseline_packets)
        num_movement = len(movement_packets)
        
        pkt_tp = movement_motion
        pkt_fn = num_movement - movement_motion
        pkt_tn = num_baseline - baseline_motion
        pkt_fp = baseline_motion
        
        recall = pkt_tp / (pkt_tp + pkt_fn) * 100.0 if (pkt_tp + pkt_fn) > 0 else 0
        precision = pkt_tp / (pkt_tp + pkt_fp) * 100.0 if (pkt_tp + pkt_fp) > 0 else 0
        fp_rate = pkt_fp / num_baseline * 100.0 if num_baseline > 0 else 0
        f1 = 2 * (precision / 100) * (recall / 100) / ((precision + recall) / 100) * 100 \
            if (precision + recall) > 0 else 0
        
        print()
        print("=" * 70)
        print("  END-TO-END RESULTS (Band Calibration + MVS)")
        print("=" * 70)
        print()
        print(f"CONFUSION MATRIX ({num_baseline} baseline + {num_movement} movement packets):")
        print("                    Predicted")
        print("                IDLE      MOTION")
        print(f"Actual IDLE     {pkt_tn:4d} (TN)  {pkt_fp:4d} (FP)")
        print(f"    MOTION      {pkt_fn:4d} (FN)  {pkt_tp:4d} (TP)")
        print()
        print("METRICS:")
        print(f"  * Recall:     {recall:.1f}% (target: >{recall_target}%)")
        print(f"  * Precision:  {precision:.1f}%")
        print(f"  * FP Rate:    {fp_rate:.1f}% (target: <{fp_rate_target}%)")
        print(f"  * F1-Score:   {f1:.1f}%")
        print()
        print("=" * 70)
        
        # ========================================
        # Assertions (chip-specific thresholds)
        # ========================================
        # Band calibrator auto-selects subcarriers using NBVI algorithm.
        # This achieves excellent performance with spectral diversity.
        assert recall > recall_target, f"End-to-end Recall too low ({num_subcarriers} SC): {recall:.1f}% (target: >{recall_target}%)"
        assert fp_rate < fp_rate_target, f"End-to-end FP Rate too high ({num_subcarriers} SC): {fp_rate:.1f}% (target: <{fp_rate_target}%)"

