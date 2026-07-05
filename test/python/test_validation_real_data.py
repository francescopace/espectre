"""
Micro-ESPectre - Validation Tests with Real CSI Data

Tests that validate algorithm performance using real CSI data from data/.
These tests verify that algorithms produce expected results on actual captured data.

Configuration is aligned with C++ tests (test_motion_detection.cpp):
- window_size = DETECTOR_DEFAULT_WINDOW_SIZE (100)
- warmup = DETECTOR_DEFAULT_WINDOW_SIZE (buffer must be full before detection)
- adaptive_factor = 1.3 (DEFAULT_ADAPTIVE_FACTOR)
- enable_hampel = true
- CV normalization always enabled
- Targets come from getter fixtures aligned with C++ target functions
- Baseline packets: no startup packets skipped; threshold calibration starts at packet 0

Converted from:
- tools/11_test_band_selection.py (algorithm validation)
- tools/12_test_csi_features.py (Feature extraction validation)
- tools/14_test_publish_time_features.py (Publish-time features)
- tools/10_test_retroactive_calibration.py (Calibration validation)

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import pytest
import json

# ============================================================================
# Detector Constants (imported from config.py, matches C++ base_detector.h)
# ============================================================================
import numpy as np
import math
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from repo_paths import data_dir, python_src_dir

sys.path.insert(0, str(python_src_dir()))

# Import from src and tools
from segmentation import SegmentationContext
from features import (
    calc_skewness, calc_mad,
)
from filters import HampelFilter
from csi_utils import (
    load_static_presence_and_motion, calculate_spatial_turbulence,
    calculate_variance_two_pass, MVSDetector
)
from config import (
    SEG_WINDOW_SIZE as DETECTOR_DEFAULT_WINDOW_SIZE,
    CALIBRATION_BUFFER_SIZE,
    HAMPEL_WINDOW,
    HAMPEL_THRESHOLD,
)


# ============================================================================
# Data Directory
# ============================================================================

DATA_DIR = data_dir()


# ============================================================================
# Dataset Configuration
# ============================================================================

def get_available_datasets():
    """Get explicit static-presence/motion pairs (HT20: 64 SC only)."""
    datasets = []

    dataset_info_path = DATA_DIR / "dataset_info.json"
    if not dataset_info_path.exists():
        return datasets

    with dataset_info_path.open("r") as f:
        dataset_info = json.load(f)

    files = dataset_info.get("files", {})
    motion_by_filename = {
        entry.get("filename"): entry
        for entry in files.get("motion", [])
        if entry.get("filename")
    }

    pair_entries = []
    for static_entry in files.get("static_presence", []):
        if static_entry.get("subcarriers") != 64:
            continue
        motion_filename = static_entry.get("optimal_pair_motion_file")
        motion_entry = motion_by_filename.get(motion_filename)
        if not motion_entry or motion_entry.get("subcarriers") != 64:
            continue

        chip = static_entry.get("chip")
        static_path = DATA_DIR / "static_presence" / static_entry["filename"]
        motion_path = DATA_DIR / "motion" / motion_filename
        if not chip or not static_path.exists() or not motion_path.exists():
            continue

        environment = static_entry.get("environment") or "unknown"
        dataset_id = f"{chip.lower()}_{environment}_{static_path.stem}"
        pair_entries.append((chip, environment, static_path.name, static_path, motion_path, dataset_id))

    for chip, _environment, _filename, static_path, motion_path, dataset_id in sorted(pair_entries):
        datasets.append(pytest.param(
            (static_path, motion_path, 64, chip, dataset_id),
            id=dataset_id
        ))

    return datasets


def get_available_empty_datasets():
    """Get empty-room recordings for ML false-positive gates."""
    empty_dir = DATA_DIR / "empty"
    datasets = []
    for path in sorted(empty_dir.glob("empty_*_64sc_*.npz")):
        datasets.append(pytest.param(path, id=path.stem))
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
        tuple: (static_presence_path, motion_path, num_subcarriers, chip, dataset_id)
    """
    return request.param


@pytest.fixture
def real_data(dataset_config):
    """Load real CSI data from the current dataset.
    
    Matches C++ behavior (csi_test_data.h):
    - Baseline: all packets loaded, starting from packet 0
    - Movement: all packets loaded
    """
    from csi_utils import load_npz_as_packets
    static_presence_path, motion_path, num_sc, chip, dataset_id = dataset_config

    static_presence_packets = load_npz_as_packets(static_presence_path)
    motion_packets = load_npz_as_packets(motion_path)

    return static_presence_packets, motion_packets


@pytest.fixture
def num_subcarriers(dataset_config):
    """Get number of subcarriers for current dataset"""
    _, _, num_sc, _, _ = dataset_config
    return num_sc


@pytest.fixture
def chip_type(dataset_config):
    """Get chip type for current dataset"""
    _, _, _, chip, _ = dataset_config
    return chip


@pytest.fixture
def dataset_id(dataset_config):
    """Get the stable dataset id for current static-presence/motion pair."""
    _, _, _, _, dataset_id_value = dataset_config
    return dataset_id_value


@pytest.fixture
def window_size(chip_type):
    """Get optimal window size for chip type.
    
    All chips use the same window size for consistent behavior.
    This matches the production default DETECTOR_DEFAULT_WINDOW_SIZE.
    """
    return DETECTOR_DEFAULT_WINDOW_SIZE


@pytest.fixture(params=["fixed_default"])
def calibration_algorithm(request, chip_type):
    """
    Parametrized fixture for the fixed-subcarrier startup calibration path.
    """
    algo = request.param
    return algo


@pytest.fixture
def enable_hampel(chip_type):
    """Enable Hampel filter for chip type.
    
    Matches C++ get_enable_hampel(): true for all chips.
    """
    return True


@pytest.fixture
def static_presence_amplitudes(real_data, default_subcarriers):
    """Extract amplitudes from baseline packets"""
    static_presence_packets, _ = real_data
    
    all_amplitudes = []
    for pkt in static_presence_packets:
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
def motion_amplitudes(real_data, default_subcarriers):
    """Extract amplitudes from movement packets"""
    _, motion_packets = real_data
    
    all_amplitudes = []
    for pkt in motion_packets:
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

def run_fixed_subcarrier_calibration(static_presence_packets, num_subcarriers, hint_band=None, mvs_window_size=None):
    """
    Run fixed-subcarrier threshold bootstrap exactly as in production.
    
    Calibration starts from packet 0 and uses the first CALIBRATION_BUFFER_SIZE
    packets, matching live startup behavior.
    
    Args:
        static_presence_packets: List of baseline CSI packets
        num_subcarriers: Number of subcarriers
        hint_band: Optional subcarrier band override (defaults to fixed defaults).
        mvs_window_size: MVS window size for validation
    
    Returns:
        tuple: (selected_band, adaptive_threshold)
    """
    from threshold import get_threshold_factor

    selected_band = hint_band
    max_moving_variance = None
    window_size = mvs_window_size or DETECTOR_DEFAULT_WINDOW_SIZE
    cal_ctx = SegmentationContext(window_size=window_size, threshold=1.0, enable_hampel=True)

    buffer_size = min(CALIBRATION_BUFFER_SIZE, len(static_presence_packets))
    for pkt in static_presence_packets[:buffer_size]:
        turb = cal_ctx.calculate_spatial_turbulence(pkt['csi_data'], selected_band)
        cal_ctx.add_turbulence(turb)
        cal_ctx.update_state()
        if cal_ctx.buffer_count >= cal_ctx.window_size:
            current_moving_variance = float(cal_ctx.current_moving_variance)
            if max_moving_variance is None or current_moving_variance > max_moving_variance:
                max_moving_variance = current_moving_variance

    if max_moving_variance is not None:
        adaptive_threshold = max_moving_variance * get_threshold_factor("auto")
    else:
        adaptive_threshold = 1.0
    return selected_band, adaptive_threshold


def run_calibration(static_presence_packets, num_subcarriers, algorithm="fixed_default", hint_band=None,
                    mvs_window_size=None):
    """
    Run startup calibration using fixed subcarriers.
    
    Args:
        static_presence_packets: List of baseline CSI packets
        num_subcarriers: Number of subcarriers
        algorithm: Calibration variant name (only "fixed_default" supported)
        hint_band: Optional fixed subcarrier band to use
        mvs_window_size: MVS window size for validation
    
    Returns:
        tuple: (selected_band, adaptive_threshold)
    """
    return run_fixed_subcarrier_calibration(
        static_presence_packets,
        num_subcarriers,
        hint_band=hint_band,
        mvs_window_size=mvs_window_size,
    )


class TestMVSDetectionRealData:
    """Test MVS motion detection with real CSI data using fixed subcarriers."""
    
    def test_static_presence_low_motion_rate(self, real_data, num_subcarriers, window_size, fp_rate_target, enable_hampel, calibration_algorithm, chip_type, default_subcarriers):
        """Test that baseline data produces low motion detection rate"""
        
        static_presence_packets, _ = real_data
        
        selected_band, adaptive_threshold = run_calibration(
            static_presence_packets,
            num_subcarriers,
            calibration_algorithm,
            hint_band=default_subcarriers,
            mvs_window_size=window_size,
        )
        
        ctx = SegmentationContext(window_size=window_size, threshold=adaptive_threshold, enable_hampel=enable_hampel)
        
        motion_count = 0
        for pkt in static_presence_packets:
            turb = ctx.calculate_spatial_turbulence(pkt['csi_data'], selected_band)
            ctx.add_turbulence(turb)
            ctx.update_state()  # Lazy evaluation: must call to update state
            if ctx.get_state() == SegmentationContext.STATE_MOTION:
                motion_count += 1
        
        # Skip warmup period
        effective_packets = len(static_presence_packets) - DETECTOR_DEFAULT_WINDOW_SIZE
        motion_rate = motion_count / effective_packets if effective_packets > 0 else 0
        
        # Target: < fp_rate_target% FP rate (chip-specific)
        target_rate = fp_rate_target / 100.0
        assert motion_rate < target_rate, f"[{calibration_algorithm}] Baseline motion rate too high: {motion_rate:.1%} (target: <{fp_rate_target}%)"
    
    def test_motion_high_motion_rate(self, real_data, num_subcarriers, window_size, recall_target, enable_hampel, calibration_algorithm, chip_type, default_subcarriers):
        """Test that movement data produces high motion detection rate"""
        
        static_presence_packets, motion_packets = real_data
        
        selected_band, adaptive_threshold = run_calibration(
            static_presence_packets,
            num_subcarriers,
            calibration_algorithm,
            hint_band=default_subcarriers,
            mvs_window_size=window_size,
        )
        
        ctx = SegmentationContext(window_size=window_size, threshold=adaptive_threshold, enable_hampel=enable_hampel)
        
        motion_count = 0
        for pkt in motion_packets:
            turb = ctx.calculate_spatial_turbulence(pkt['csi_data'], selected_band)
            ctx.add_turbulence(turb)
            ctx.update_state()  # Lazy evaluation: must call to update state
            if ctx.get_state() == SegmentationContext.STATE_MOTION:
                motion_count += 1
        
        # Skip warmup period
        effective_packets = len(motion_packets) - DETECTOR_DEFAULT_WINDOW_SIZE
        motion_rate = motion_count / effective_packets if effective_packets > 0 else 0
        
        # Target from recall_target fixture (matches C++ get_recall_target()).
        min_recall_rate = recall_target / 100.0
        assert motion_rate > min_recall_rate, (
            f"[{calibration_algorithm}] Movement motion rate too low: "
            f"{motion_rate:.1%} (target: >{recall_target}%)"
        )
    
    def test_mvs_detector_wrapper(self, real_data, num_subcarriers, window_size, calibration_algorithm, chip_type, default_subcarriers):
        """Test MVSDetector wrapper class with calibration"""
        
        static_presence_packets, motion_packets = real_data
        
        selected_band, adaptive_threshold = run_calibration(
            static_presence_packets,
            num_subcarriers,
            calibration_algorithm,
            hint_band=default_subcarriers,
            mvs_window_size=window_size,
        )
        
        # Test with the calibrated band and adaptive threshold
        # Note: csi_utils.MVSDetector has different signature than src.mvs_detector.MVSDetector
        detector = MVSDetector(
            window_size=window_size,
            threshold=adaptive_threshold,
            selected_subcarriers=selected_band,
            track_data=True
        )
        # csi_utils.MVSDetector internally uses SegmentationContext
        
        for pkt in static_presence_packets:
            detector.process_packet(pkt)
        
        static_presence_motion = detector.get_motion_count()
        
        # Reset and test on movement
        detector.reset()
        
        for pkt in motion_packets:
            detector.process_packet(pkt)
        
        motion_motion = detector.get_motion_count()
        
        # Movement should have significantly more motion packets
        assert motion_motion > static_presence_motion * 2


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
    
    def test_skewness_separation(self, static_presence_amplitudes, motion_amplitudes):
        """Test that skewness shows separation between baseline and movement"""
        static_presence_skew = [calc_skewness(list(r), len(r), float(np.mean(r)), float(np.std(r))) for r in static_presence_amplitudes]
        motion_skew = [calc_skewness(list(r), len(r), float(np.mean(r)), float(np.std(r))) for r in motion_amplitudes]
        
        J = fishers_criterion(static_presence_skew, motion_skew)
        
        # Should have some separation
        # Note: Skewness is not the primary detection method (MVS is)
        # so we only require minimal separation to confirm the feature works
        assert J > 0.0001, f"Skewness Fisher's J too low: {J:.6f}"
    
    def test_turbulence_variance_separation(self, real_data, default_subcarriers, chip_type, window_size):
        """Test that turbulence variance separates baseline from movement.
        
        Uses the shared CV-normalized turbulence path, matching production.
        """
        static_presence_packets, motion_packets = real_data
        
        # Calculate turbulence for each packet using the shared runtime path.
        static_presence_turb = []
        for pkt in static_presence_packets:
            turb, _ = SegmentationContext.compute_spatial_turbulence(pkt['csi_data'], default_subcarriers)
            static_presence_turb.append(turb)
        
        motion_turb = []
        for pkt in motion_packets:
            turb, _ = SegmentationContext.compute_spatial_turbulence(pkt['csi_data'], default_subcarriers)
            motion_turb.append(turb)
        
        # Calculate variance of turbulence over windows (use window_size from C++ config)
        analysis_window = window_size
        
        def window_variances(values, ws):
            variances = []
            for i in range(0, len(values) - ws, ws // 2):
                window = values[i:i + ws]
                variances.append(calculate_variance_two_pass(window))
            return variances
        
        static_presence_vars = window_variances(static_presence_turb, analysis_window)
        motion_vars = window_variances(motion_turb, analysis_window)
        
        if len(static_presence_vars) > 0 and len(motion_vars) > 0:
            J = fishers_criterion(static_presence_vars, motion_vars)
            
            # Variance should show good separation (this is the core of MVS)
            assert J > 0.5, f"Turbulence variance Fisher's J too low: {J:.3f}"


# ============================================================================
# Publish-Time Features Tests
# ============================================================================

class TestPublishTimeFeaturesRealData:
    """Test publish-time feature extraction with real data"""
    
    def test_mad_turb_separation(self, real_data, default_subcarriers, window_size, chip_type):
        """Test MAD of turbulence buffer separates baseline from movement"""
        
        static_presence_packets, motion_packets = real_data
        ws = window_size
        
        def calculate_mad_values(packets):
            ctx = SegmentationContext(window_size=ws, threshold=1.0)
            mad_values = []
            
            for pkt in packets:
                turb = ctx.calculate_spatial_turbulence(pkt['csi_data'], default_subcarriers)
                ctx.add_turbulence(turb)
                
                if ctx.buffer_count >= ws:
                    mad = calc_mad(ctx.turbulence_buffer, ctx.buffer_count)
                    mad_values.append(mad)
            
            return mad_values
        
        static_presence_mad = calculate_mad_values(static_presence_packets)
        motion_mad = calculate_mad_values(motion_packets)
        
        if len(static_presence_mad) > 0 and len(motion_mad) > 0:
            J = fishers_criterion(static_presence_mad, motion_mad)
            
            # MAD should show good separation (S3 has lower separation due to noisier baseline)
            min_j = 0.3 if chip_type == 'S3' else 0.5
            assert J > min_j, f"MAD Fisher's J too low: {J:.3f} (target: >{min_j})"
    
# ============================================================================
# Hampel Filter Tests with Real Data
# ============================================================================

class TestHampelFilterRealData:
    """Test Hampel filter with real CSI turbulence data"""
    
    def test_hampel_reduces_spikes(self, real_data, default_subcarriers):
        """Test that Hampel filter reduces turbulence spikes"""
        static_presence_packets, motion_packets = real_data
        all_packets = static_presence_packets + motion_packets
        
        # Calculate raw turbulence
        raw_turbulence = []
        for pkt in all_packets:
            turb = calculate_spatial_turbulence(
                pkt['csi_data'],
                default_subcarriers,
            )
            raw_turbulence.append(turb)
        
        # Apply Hampel filter
        hf = HampelFilter(window_size=HAMPEL_WINDOW, threshold=HAMPEL_THRESHOLD)
        filtered_turbulence = [hf.filter(t) for t in raw_turbulence]
        
        # Filtered should have lower max (spikes reduced)
        raw_max = max(raw_turbulence)
        filtered_max = max(filtered_turbulence)
        
        # If there were spikes, they should be reduced
        if raw_max > np.mean(raw_turbulence) * 3:
            assert filtered_max <= raw_max, "Hampel should not increase max value"
    
    def test_hampel_preserves_variance_separation(self, real_data, default_subcarriers):
        """Test that Hampel filter preserves baseline/movement separation"""
        static_presence_packets, motion_packets = real_data
        
        # Calculate filtered turbulence for baseline
        hf_baseline = HampelFilter(window_size=HAMPEL_WINDOW, threshold=HAMPEL_THRESHOLD)
        static_presence_turb = []
        for pkt in static_presence_packets:
            turb = calculate_spatial_turbulence(
                pkt['csi_data'],
                default_subcarriers,
            )
            filtered = hf_baseline.filter(turb)
            static_presence_turb.append(filtered)
        
        # Calculate filtered turbulence for movement
        hf_movement = HampelFilter(window_size=HAMPEL_WINDOW, threshold=HAMPEL_THRESHOLD)
        motion_turb = []
        for pkt in motion_packets:
            turb = calculate_spatial_turbulence(
                pkt['csi_data'],
                default_subcarriers,
            )
            filtered = hf_movement.filter(turb)
            motion_turb.append(filtered)
        
        # Movement should still have higher variance
        static_presence_var = np.var(static_presence_turb)
        motion_var = np.var(motion_turb)
        
        assert motion_var > static_presence_var, \
            f"Movement variance ({motion_var:.3f}) should be > baseline ({static_presence_var:.3f})"


# ============================================================================
# Performance Metrics Tests
# ============================================================================

class TestPerformanceMetrics:
    """Test that we achieve expected performance metrics with fixed subcarriers."""
    
    def test_mvs_fixed_subcarriers(self, real_data, window_size, fp_rate_target,
                                   recall_target, enable_hampel, chip_type,
                                   default_subcarriers, dataset_id):
        """
        Test MVS motion detection with fixed production subcarriers.
        
        This is a fixed-band regression test that uses the shared production
        subcarriers for each chip (matches C++ test_mvs_fixed_subcarriers).
        
        Startup uses fixed subcarriers from conftest.py.
        """
        import numpy as np
        from threshold import get_threshold_factor
        static_presence_packets, motion_packets = real_data
        
        # Context-aware subcarriers from dataset_info metadata.
        selected_band = default_subcarriers
        
        # Keep threshold calibration aligned with runtime test pipeline.
        cal_ctx = SegmentationContext(
            window_size=window_size, threshold=1.0, enable_hampel=enable_hampel
        )
        max_moving_variance = None
        calibration_packets = min(len(static_presence_packets), CALIBRATION_BUFFER_SIZE)
        for pkt in static_presence_packets[:calibration_packets]:
            turb = cal_ctx.calculate_spatial_turbulence(pkt['csi_data'], selected_band)
            cal_ctx.add_turbulence(turb)
            cal_ctx.update_state()
            if cal_ctx.buffer_count >= cal_ctx.window_size:
                current_moving_variance = float(cal_ctx.current_moving_variance)
                if max_moving_variance is None or current_moving_variance > max_moving_variance:
                    max_moving_variance = current_moving_variance
        adaptive_threshold = (
            max_moving_variance * get_threshold_factor("auto")
            if max_moving_variance is not None
            else 1.0
        )
        
        # Initialize with adaptive threshold (new detector, matches C++)
        ctx = SegmentationContext(
            window_size=window_size, threshold=adaptive_threshold, enable_hampel=enable_hampel
        )
        
        num_baseline = len(static_presence_packets)
        num_movement = len(motion_packets)
        
        # Process baseline (expecting IDLE)
        static_presence_motion_packets = 0
        for pkt in static_presence_packets:
            turb = ctx.calculate_spatial_turbulence(pkt['csi_data'], selected_band)
            ctx.add_turbulence(turb)
            ctx.update_state()
            if ctx.get_state() == SegmentationContext.STATE_MOTION:
                static_presence_motion_packets += 1
        
        # Process movement (expecting MOTION, continue in same context)
        motion_with_motion = 0
        motion_without_motion = 0
        for pkt in motion_packets:
            turb = ctx.calculate_spatial_turbulence(pkt['csi_data'], selected_band)
            ctx.add_turbulence(turb)
            ctx.update_state()
            if ctx.get_state() == SegmentationContext.STATE_MOTION:
                motion_with_motion += 1
            else:
                motion_without_motion += 1
        
        # Calculate metrics
        pkt_tp = motion_with_motion
        pkt_fn = motion_without_motion
        pkt_tn = num_baseline - static_presence_motion_packets
        pkt_fp = static_presence_motion_packets
        
        pkt_recall = pkt_tp / (pkt_tp + pkt_fn) * 100.0 if (pkt_tp + pkt_fn) > 0 else 0
        pkt_precision = pkt_tp / (pkt_tp + pkt_fp) * 100.0 if (pkt_tp + pkt_fp) > 0 else 0
        pkt_fp_rate = pkt_fp / num_baseline * 100.0 if num_baseline > 0 else 0
        pkt_f1 = 2 * (pkt_precision / 100) * (pkt_recall / 100) / ((pkt_precision + pkt_recall) / 100) * 100 if (pkt_precision + pkt_recall) > 0 else 0
        
        print(f"\n  * Dataset pair: {dataset_id}")
        print(f"  * Subcarriers: {selected_band}")
        print(f"  * Threshold:  {adaptive_threshold:.3f}")
        print(f"  * Recall:     {pkt_recall:.1f}% (target: >{recall_target}%)")
        print(f"  * Precision:  {pkt_precision:.1f}%")
        print(f"  * FP Rate:    {pkt_fp_rate:.1f}% (target: <{fp_rate_target}%)")
        print(f"  * F1-Score:   {pkt_f1:.1f}%")

        # Assertions
        assert pkt_recall > recall_target, (
            f"Recall too low: {pkt_recall:.1f}% (target: >{recall_target}%)"
        )
        assert pkt_fp_rate < fp_rate_target, (
            f"FP Rate too high: {pkt_fp_rate:.1f}% (target: <{fp_rate_target}%)"
        )

    def test_mvs_detection_accuracy(self, real_data, num_subcarriers, window_size, fp_rate_target,
                                    recall_target, enable_hampel, calibration_algorithm, chip_type,
                                    default_subcarriers, dataset_id):
        """
        Test MVS motion detection accuracy with real CSI data.
        
        This test uses the current production startup path:
        - Fixed default subcarriers for all chips
        - Adaptive threshold from baseline calibration
        - Process ALL packets (no warmup skip)
        - Process baseline first, then movement (continuous context)
        - Unified window_size (100) and adaptive threshold (max x 1.3)
        - CV normalization for all chips
        
        Targets: >recall_target% Recall, <fp_rate_target% FP Rate.
        """
        static_presence_packets, motion_packets = real_data

        selected_band, adaptive_threshold = run_calibration(
            static_presence_packets,
            num_subcarriers,
            calibration_algorithm,
            hint_band=default_subcarriers,
            mvs_window_size=window_size,
        )
        
        # Initialize with adaptive threshold from calibration
        ctx = SegmentationContext(
            window_size=window_size, threshold=adaptive_threshold, enable_hampel=enable_hampel
        )
        
        num_baseline = len(static_presence_packets)
        num_movement = len(motion_packets)
        
        # ========================================
        # Process baseline (expecting IDLE)
        # ========================================
        static_presence_motion_packets = 0
        
        for pkt in static_presence_packets:
            turb = ctx.calculate_spatial_turbulence(pkt['csi_data'], selected_band)
            ctx.add_turbulence(turb)
            ctx.update_state()  # Lazy evaluation: must call to update state
            if ctx.get_state() == SegmentationContext.STATE_MOTION:
                static_presence_motion_packets += 1
        
        # ========================================
        # Process movement (expecting MOTION)
        # Continue with same context (no reset)
        # ========================================
        motion_with_motion = 0
        motion_without_motion = 0
        
        for pkt in motion_packets:
            turb = ctx.calculate_spatial_turbulence(pkt['csi_data'], selected_band)
            ctx.add_turbulence(turb)
            ctx.update_state()  # Lazy evaluation: must call to update state
            if ctx.get_state() == SegmentationContext.STATE_MOTION:
                motion_with_motion += 1
            else:
                motion_without_motion += 1
        
        # ========================================
        # Calculate metrics (same as C++)
        # ========================================
        pkt_tp = motion_with_motion
        pkt_fn = motion_without_motion
        pkt_tn = num_baseline - static_presence_motion_packets
        pkt_fp = static_presence_motion_packets
        
        pkt_recall = pkt_tp / (pkt_tp + pkt_fn) * 100.0 if (pkt_tp + pkt_fn) > 0 else 0
        pkt_precision = pkt_tp / (pkt_tp + pkt_fp) * 100.0 if (pkt_tp + pkt_fp) > 0 else 0
        pkt_fp_rate = pkt_fp / num_baseline * 100.0 if num_baseline > 0 else 0
        pkt_f1 = 2 * (pkt_precision / 100) * (pkt_recall / 100) / ((pkt_precision + pkt_recall) / 100) * 100 if (pkt_precision + pkt_recall) > 0 else 0
        
        # ========================================
        # Print results (same format as C++)
        # ========================================
        print("\n")
        print("=" * 70)
        print("                   TEST SUMMARY (Context-aware)")
        print("=" * 70)
        print(f"Dataset pair: {dataset_id}")
        print(f"Subcarriers: {selected_band}")
        print(f"Threshold:   {adaptive_threshold:.3f}")
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
        record_performance(chip_type, 'mvs', pkt_recall, pkt_fp_rate, pkt_precision, pkt_f1,
                           dataset_id=dataset_id)
        
        # ========================================
        # Assertions (chip-specific thresholds)
        # ========================================
        assert pkt_recall > recall_target, f"Recall too low: {pkt_recall:.1f}% (target: >{recall_target}%)"
        assert pkt_fp_rate < fp_rate_target, f"FP Rate too high: {pkt_fp_rate:.1f}% (target: <{fp_rate_target}%)"

    def test_ml_detection_accuracy(self, real_data, num_subcarriers, ml_fp_rate_target, ml_recall_target,
                                   chip_type, dataset_id):
        """
        Test ML (Neural Network) motion detection accuracy with real CSI data.
        
        ML uses a pre-trained MLP model for motion classification.
        No calibration needed - uses pre-trained weights.
        
        Note: ML model uses fixed subcarriers from config.DEFAULT_SUBCARRIERS regardless of chip type.
        ML uses the shared CV-normalized turbulence path.
        
        Targets: >ml_recall_target% Recall, <ml_fp_rate_target% FP Rate.
        """
        from ml_detector import MLDetector
        from config import DEFAULT_SUBCARRIERS
        from detector_interface import MotionState

        static_presence_packets, motion_packets = real_data
        
        num_baseline = len(static_presence_packets)
        num_movement = len(motion_packets)
        
        # ML model uses fixed subcarriers (must match training)
        ml_subcarriers = DEFAULT_SUBCARRIERS
        # ========================================
        # Initialize ML Detector (no calibration needed)
        # ========================================
        detector = MLDetector(
            threshold=5.0,  # Default scaled threshold (0.1-10.0)
            window_size=DETECTOR_DEFAULT_WINDOW_SIZE,
        )
        
        print(f"\nML Detector initialized")
        print(f"  Threshold: 5.0")
        print(f"  Window size: {DETECTOR_DEFAULT_WINDOW_SIZE} (DETECTOR_DEFAULT_WINDOW_SIZE)")
        print(f"  Subcarriers: {ml_subcarriers} (fixed for ML)")
        print("  Turbulence: normalized runtime path")
        
        # ========================================
        # Process ALL baseline packets (first window_size packets are warmup)
        # ========================================
        warmup = DETECTOR_DEFAULT_WINDOW_SIZE
        static_presence_motion_packets = 0
        static_presence_eval_count = num_baseline - warmup
        
        for i, pkt in enumerate(static_presence_packets):
            detector.process_packet(pkt['csi_data'], ml_subcarriers)
            detector.update_state()
            # Only count after warmup
            if i >= warmup and detector.get_state() == MotionState.MOTION:
                static_presence_motion_packets += 1
        
        # ========================================
        # Process movement packets (continue without reset, first window_size packets are warmup)
        # ========================================
        motion_warmup = DETECTOR_DEFAULT_WINDOW_SIZE
        motion_with_motion = 0
        motion_without_motion = 0
        motion_eval_count = num_movement - motion_warmup
        
        for i, pkt in enumerate(motion_packets):
            detector.process_packet(pkt['csi_data'], ml_subcarriers)
            detector.update_state()
            # Only count after warmup
            if i >= motion_warmup:
                if detector.get_state() == MotionState.MOTION:
                    motion_with_motion += 1
                else:
                    motion_without_motion += 1
        
        # ========================================
        # Calculate metrics
        # ========================================
        pkt_tp = motion_with_motion
        pkt_fn = motion_without_motion
        pkt_tn = static_presence_eval_count - static_presence_motion_packets if static_presence_eval_count > 0 else 0
        pkt_fp = static_presence_motion_packets
        
        pkt_recall = pkt_tp / (pkt_tp + pkt_fn) * 100.0 if (pkt_tp + pkt_fn) > 0 else 0
        pkt_precision = pkt_tp / (pkt_tp + pkt_fp) * 100.0 if (pkt_tp + pkt_fp) > 0 else 0
        pkt_fp_rate = pkt_fp / static_presence_eval_count * 100.0 if static_presence_eval_count > 0 else 0
        pkt_f1 = 2 * (pkt_precision / 100) * (pkt_recall / 100) / ((pkt_precision + pkt_recall) / 100) * 100 if (pkt_precision + pkt_recall) > 0 else 0
        
        # ========================================
        # Print results
        # ========================================
        print("\n")
        print("=" * 70)
        print("                     ML DETECTION TEST SUMMARY")
        print("=" * 70)
        print()
        print(f"CONFUSION MATRIX ({static_presence_eval_count} baseline + {motion_eval_count} movement packets):")
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
        record_performance(chip_type, 'ml', pkt_recall, pkt_fp_rate, pkt_precision, pkt_f1,
                           dataset_id=dataset_id)
        
        # ========================================
        # Assertions
        # ========================================
        assert pkt_recall > ml_recall_target, f"ML Recall too low: {pkt_recall:.1f}% (target: >{ml_recall_target}%)"
        if static_presence_eval_count > 0:
            assert pkt_fp_rate < ml_fp_rate_target, f"ML FP Rate too high: {pkt_fp_rate:.1f}% (target: <{ml_fp_rate_target}%)"

    @pytest.mark.parametrize("empty_dataset_path", get_available_empty_datasets())
    def test_ml_empty_false_positive_rate(self, empty_dataset_path):
        """Validate that empty-room recordings stay below the ML FP target."""
        from csi_utils import load_npz_as_packets
        from ml_detector import MLDetector
        from config import DEFAULT_SUBCARRIERS
        from detector_interface import MotionState

        packets = load_npz_as_packets(empty_dataset_path)
        detector = MLDetector(
            threshold=5.0,
            window_size=DETECTOR_DEFAULT_WINDOW_SIZE,
        )

        warmup = DETECTOR_DEFAULT_WINDOW_SIZE
        eval_count = max(len(packets) - warmup, 0)
        motion_packets = 0

        for i, pkt in enumerate(packets):
            detector.process_packet(pkt["csi_data"], DEFAULT_SUBCARRIERS)
            detector.update_state()
            if i >= warmup and detector.get_state() == MotionState.MOTION:
                motion_packets += 1

        fp_rate = motion_packets / eval_count * 100.0 if eval_count > 0 else 0.0
        assert eval_count > 0
        fp_rate_target = 5.0
        assert fp_rate < fp_rate_target, (
            f"ML empty-room FP Rate too high for {empty_dataset_path.name}: "
            f"{fp_rate:.1f}% (target: <{fp_rate_target}%)"
        )


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
        static_presence_packets, _ = real_data
        
        max_rel_error = 0.0
        
        for pkt in static_presence_packets[:200]:
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
        static_presence_packets, _ = real_data
        
        # Generate turbulence values
        turbulences = []
        for pkt in static_presence_packets[:100]:
            turb = calculate_spatial_turbulence(
                pkt['csi_data'],
                default_subcarriers,
            )
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
# End-to-End Tests with Startup Calibration and Normalization
# ============================================================================

class TestEndToEndWithCalibration:
    """
    Test complete pipeline: Startup Calibration → Normalization → MVS Detection
    
    These tests verify that the system works end-to-end with:
    - Fixed default subcarriers shared by MVS and ML
    - Adaptive threshold applied to turbulence values
    - MVS motion detection achieving target performance
    """
    
    def test_band_calibration_produces_valid_band(self, real_data, num_subcarriers, calibration_algorithm, chip_type, default_subcarriers):
        """Test that startup calibration produces a valid fixed band and threshold."""
        
        from threshold import calculate_adaptive_threshold
        from config import GUARD_BAND_LOW, GUARD_BAND_HIGH, DC_SUBCARRIER
        
        static_presence_packets, _ = real_data
        
        # HT20 fixed guard bands (64 SC)
        guard_low = GUARD_BAND_LOW
        guard_high = GUARD_BAND_HIGH
        
        # Run calibration with the fixed default subcarriers.
        selected_band, adaptive_threshold = run_calibration(static_presence_packets, num_subcarriers, calibration_algorithm, hint_band=default_subcarriers)
        
        # Verify calibration results
        assert selected_band is not None, f"[{calibration_algorithm}] Band calibration failed"
        assert len(selected_band) == 12, f"[{calibration_algorithm}] Expected 12 subcarriers, got {len(selected_band)}"
        
        # All subcarriers should be valid (within valid range for this SC count)
        for sc in selected_band:
            assert guard_low <= sc <= guard_high, \
                f"[{calibration_algorithm}] Subcarrier {sc} outside valid range [{guard_low}-{guard_high}]"
        
        # Adaptive threshold should be valid
        assert adaptive_threshold > 0.0, f"[{calibration_algorithm}] Invalid adaptive threshold: {adaptive_threshold}"
        assert 0.0 <= adaptive_threshold <= 10.0, \
            f"[{calibration_algorithm}] Adaptive threshold out of range: {adaptive_threshold}"
        
        print(f"\n[{calibration_algorithm.upper()}] Startup Calibration Results:")
        print(f"  Selected band: {selected_band}")
        print(f"  Adaptive threshold: {adaptive_threshold:.4f}")
    
    def test_end_to_end_with_band_calibration_and_mvs(self, real_data, num_subcarriers, window_size, fp_rate_target, recall_target, enable_hampel, calibration_algorithm, chip_type, default_subcarriers):
        """
        Test complete end-to-end flow: Startup Calibration → MVS → Detection
        
        This test verifies that the system achieves target performance using
        recall_target/fp_rate_target fixtures.
        when using the fixed default subcarrier set.
        """
        static_presence_packets, motion_packets = real_data
        
        # ========================================
        # Step 1: Startup Calibration
        # ========================================
        print("\n" + "=" * 70)
        print(f"  END-TO-END TEST: Startup Calibration + MVS ({num_subcarriers} SC, {calibration_algorithm.upper()})")
        print("=" * 70)
        
        print(f"\nStep 1: {calibration_algorithm.upper()} Startup Calibration...")
        selected_band, adaptive_threshold = run_calibration(
            static_presence_packets,
            num_subcarriers,
            calibration_algorithm,
            hint_band=default_subcarriers,
            mvs_window_size=window_size,
        )
        print(f"  Selected band: {selected_band}")
        
        assert selected_band is not None, f"[{calibration_algorithm}] Startup calibration failed for {num_subcarriers} SC"
        print(f"  Adaptive threshold: {adaptive_threshold:.4f}")
        
        # ========================================
        # Step 2: Initialize MVS with calibration results
        # ========================================
        # Initialize MVS with calibration-selected subcarriers AND adaptive threshold
        # This tests the complete production pipeline
        print(f"\nStep 2: Initialize MVS with calibration results (Hampel: {enable_hampel})...")
        ctx = SegmentationContext(
            window_size=window_size,
            threshold=adaptive_threshold,  # Apply calibration adaptive threshold
            enable_hampel=enable_hampel
        )
        
        # ========================================
        # Step 3: Process baseline (expecting IDLE)
        # ========================================
        print("\nStep 3: Process baseline packets (expecting IDLE)...")
        static_presence_motion = 0
        
        for pkt in static_presence_packets:
            turb = ctx.calculate_spatial_turbulence(pkt['csi_data'], selected_band)
            ctx.add_turbulence(turb)
            ctx.update_state()  # Lazy evaluation: must call to update state
            if ctx.get_state() == SegmentationContext.STATE_MOTION:
                static_presence_motion += 1
        
        # ========================================
        # Step 4: Process movement (expecting MOTION)
        # ========================================
        print("Step 4: Process movement packets (expecting MOTION)...")
        motion_motion = 0
        
        for pkt in motion_packets:
            turb = ctx.calculate_spatial_turbulence(pkt['csi_data'], selected_band)
            ctx.add_turbulence(turb)
            ctx.update_state()  # Lazy evaluation: must call to update state
            if ctx.get_state() == SegmentationContext.STATE_MOTION:
                motion_motion += 1
        
        # ========================================
        # Step 5: Calculate metrics
        # ========================================
        num_baseline = len(static_presence_packets)
        num_movement = len(motion_packets)
        
        pkt_tp = motion_motion
        pkt_fn = num_movement - motion_motion
        pkt_tn = num_baseline - static_presence_motion
        pkt_fp = static_presence_motion
        
        recall = pkt_tp / (pkt_tp + pkt_fn) * 100.0 if (pkt_tp + pkt_fn) > 0 else 0
        precision = pkt_tp / (pkt_tp + pkt_fp) * 100.0 if (pkt_tp + pkt_fp) > 0 else 0
        fp_rate = pkt_fp / num_baseline * 100.0 if num_baseline > 0 else 0
        f1 = 2 * (precision / 100) * (recall / 100) / ((precision + recall) / 100) * 100 \
            if (precision + recall) > 0 else 0
        
        print()
        print("=" * 70)
        print("  END-TO-END RESULTS (Startup Calibration + MVS)")
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
        # Startup calibration keeps the fixed default band and tunes only the threshold.
        assert recall > recall_target, f"End-to-end Recall too low ({num_subcarriers} SC): {recall:.1f}% (target: >{recall_target}%)"
        assert fp_rate < fp_rate_target, f"End-to-end FP Rate too high ({num_subcarriers} SC): {fp_rate:.1f}% (target: <{fp_rate_target}%)"
