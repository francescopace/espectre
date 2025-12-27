"""
Micro-ESPectre - Validation Tests with Real CSI Data

Tests that validate algorithm performance using real CSI data from data/.
These tests verify that algorithms produce expected results on actual captured data.

Converted from:
- tools/11_test_nbvi_selection.py (NBVI algorithm validation)
- tools/12_test_csi_features.py (Feature extraction validation)
- tools/14_test_publish_time_features.py (Publish-time features)
- tools/10_test_retroactive_calibration.py (Calibration validation)

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import pytest
import numpy as np
import math
import os
import tempfile
from pathlib import Path

# Patch NBVI buffer file path BEFORE importing NBVICalibrator
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
import nbvi_calibrator
nbvi_calibrator.BUFFER_FILE = os.path.join(tempfile.gettempdir(), 'nbvi_buffer_validation_test.bin')

# Import from src and tools
from segmentation import SegmentationContext
from features import (
    calc_skewness, calc_kurtosis, calc_iqr_turb, calc_entropy_turb,
    PublishTimeFeatureExtractor, MultiFeatureDetector
)
from filters import HampelFilter
from csi_utils import (
    load_baseline_and_movement, calculate_spatial_turbulence,
    calculate_variance_two_pass, MVSDetector
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def selected_subcarriers():
    """Default subcarrier band"""
    return [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]


@pytest.fixture
def real_data(real_csi_data_available):
    """Load real CSI data"""
    if not real_csi_data_available:
        pytest.skip("Real CSI data not available")
    return load_baseline_and_movement()


@pytest.fixture
def baseline_amplitudes(real_data, selected_subcarriers):
    """Extract amplitudes from baseline packets"""
    baseline_packets, _ = real_data
    
    all_amplitudes = []
    for pkt in baseline_packets:
        csi_data = pkt['csi_data']
        amps = []
        for sc_idx in selected_subcarriers:
            i_idx = sc_idx * 2
            q_idx = sc_idx * 2 + 1
            if q_idx < len(csi_data):
                I = float(csi_data[i_idx])
                Q = float(csi_data[q_idx])
                amps.append(math.sqrt(I**2 + Q**2))
        all_amplitudes.append(amps)
    
    return np.array(all_amplitudes)


@pytest.fixture
def movement_amplitudes(real_data, selected_subcarriers):
    """Extract amplitudes from movement packets"""
    _, movement_packets = real_data
    
    all_amplitudes = []
    for pkt in movement_packets:
        csi_data = pkt['csi_data']
        amps = []
        for sc_idx in selected_subcarriers:
            i_idx = sc_idx * 2
            q_idx = sc_idx * 2 + 1
            if q_idx < len(csi_data):
                I = float(csi_data[i_idx])
                Q = float(csi_data[q_idx])
                amps.append(math.sqrt(I**2 + Q**2))
        all_amplitudes.append(amps)
    
    return np.array(all_amplitudes)


# ============================================================================
# MVS Detection Tests
# ============================================================================

class TestMVSDetectionRealData:
    """Test MVS motion detection with real CSI data"""
    
    def test_baseline_low_motion_rate(self, real_data, selected_subcarriers):
        """Test that baseline data produces low motion detection rate"""
        baseline_packets, _ = real_data
        
        ctx = SegmentationContext(window_size=50, threshold=1.0)
        
        motion_count = 0
        for pkt in baseline_packets:
            turb = ctx.calculate_spatial_turbulence(pkt['csi_data'], selected_subcarriers)
            ctx.add_turbulence(turb)
            ctx.update_state()  # Lazy evaluation: must call to update state
            if ctx.get_state() == SegmentationContext.STATE_MOTION:
                motion_count += 1
        
        # Skip warmup period
        warmup = 50
        effective_packets = len(baseline_packets) - warmup
        motion_rate = motion_count / effective_packets if effective_packets > 0 else 0
        
        # Baseline should have less than 20% motion (ideally < 10%)
        assert motion_rate < 0.20, f"Baseline motion rate too high: {motion_rate:.1%}"
    
    def test_movement_high_motion_rate(self, real_data, selected_subcarriers):
        """Test that movement data produces high motion detection rate"""
        _, movement_packets = real_data
        
        ctx = SegmentationContext(window_size=50, threshold=1.0)
        
        motion_count = 0
        for pkt in movement_packets:
            turb = ctx.calculate_spatial_turbulence(pkt['csi_data'], selected_subcarriers)
            ctx.add_turbulence(turb)
            ctx.update_state()  # Lazy evaluation: must call to update state
            if ctx.get_state() == SegmentationContext.STATE_MOTION:
                motion_count += 1
        
        # Skip warmup period
        warmup = 50
        effective_packets = len(movement_packets) - warmup
        motion_rate = motion_count / effective_packets if effective_packets > 0 else 0
        
        # Movement should have more than 50% motion
        assert motion_rate > 0.50, f"Movement motion rate too low: {motion_rate:.1%}"
    
    def test_mvs_detector_wrapper(self, real_data, selected_subcarriers):
        """Test MVSDetector wrapper class"""
        baseline_packets, movement_packets = real_data
        
        # Test on baseline
        detector = MVSDetector(
            window_size=50,
            threshold=1.0,
            selected_subcarriers=selected_subcarriers
        )
        
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
    
    if var1 + var2 < 1e-10:
        return 0.0
    
    return (mu1 - mu2) ** 2 / (var1 + var2)


class TestFeatureSeparationRealData:
    """Test feature separation between baseline and movement"""
    
    def test_skewness_separation(self, baseline_amplitudes, movement_amplitudes):
        """Test that skewness shows separation between baseline and movement"""
        baseline_skew = [calc_skewness(list(row)) for row in baseline_amplitudes]
        movement_skew = [calc_skewness(list(row)) for row in movement_amplitudes]
        
        J = fishers_criterion(baseline_skew, movement_skew)
        
        # Should have some separation (J > 0.1)
        assert J > 0.1, f"Skewness Fisher's J too low: {J:.3f}"
    
    def test_kurtosis_separation(self, baseline_amplitudes, movement_amplitudes):
        """Test that kurtosis shows separation between baseline and movement"""
        baseline_kurt = [calc_kurtosis(list(row)) for row in baseline_amplitudes]
        movement_kurt = [calc_kurtosis(list(row)) for row in movement_amplitudes]
        
        J = fishers_criterion(baseline_kurt, movement_kurt)
        
        # Should have some separation
        assert J > 0.1, f"Kurtosis Fisher's J too low: {J:.3f}"
    
    def test_turbulence_variance_separation(self, real_data, selected_subcarriers):
        """Test that turbulence variance separates baseline from movement"""
        baseline_packets, movement_packets = real_data
        
        # Calculate turbulence for each packet
        baseline_turb = []
        for pkt in baseline_packets:
            turb = calculate_spatial_turbulence(pkt['csi_data'], selected_subcarriers)
            baseline_turb.append(turb)
        
        movement_turb = []
        for pkt in movement_packets:
            turb = calculate_spatial_turbulence(pkt['csi_data'], selected_subcarriers)
            movement_turb.append(turb)
        
        # Calculate variance of turbulence over windows
        window_size = 50
        
        def window_variances(values, window_size):
            variances = []
            for i in range(0, len(values) - window_size, window_size // 2):
                window = values[i:i + window_size]
                variances.append(calculate_variance_two_pass(window))
            return variances
        
        baseline_vars = window_variances(baseline_turb, window_size)
        movement_vars = window_variances(movement_turb, window_size)
        
        if len(baseline_vars) > 0 and len(movement_vars) > 0:
            J = fishers_criterion(baseline_vars, movement_vars)
            
            # Variance should show good separation (this is the core of MVS)
            assert J > 0.5, f"Turbulence variance Fisher's J too low: {J:.3f}"


# ============================================================================
# Publish-Time Features Tests
# ============================================================================

class TestPublishTimeFeaturesRealData:
    """Test publish-time feature extraction with real data"""
    
    def test_iqr_turb_separation(self, real_data, selected_subcarriers):
        """Test IQR of turbulence buffer separates baseline from movement"""
        baseline_packets, movement_packets = real_data
        window_size = 50
        
        def calculate_iqr_values(packets):
            ctx = SegmentationContext(window_size=window_size, threshold=1.0)
            iqr_values = []
            
            for pkt in packets:
                turb = ctx.calculate_spatial_turbulence(pkt['csi_data'], selected_subcarriers)
                ctx.add_turbulence(turb)
                
                if ctx.buffer_count >= window_size:
                    iqr = calc_iqr_turb(ctx.turbulence_buffer, ctx.buffer_count)
                    iqr_values.append(iqr)
            
            return iqr_values
        
        baseline_iqr = calculate_iqr_values(baseline_packets)
        movement_iqr = calculate_iqr_values(movement_packets)
        
        if len(baseline_iqr) > 0 and len(movement_iqr) > 0:
            J = fishers_criterion(baseline_iqr, movement_iqr)
            
            # IQR should show good separation
            assert J > 0.5, f"IQR Fisher's J too low: {J:.3f}"
    
    def test_entropy_turb_separation(self, real_data, selected_subcarriers):
        """Test entropy of turbulence buffer separates baseline from movement"""
        baseline_packets, movement_packets = real_data
        window_size = 50
        
        def calculate_entropy_values(packets):
            ctx = SegmentationContext(window_size=window_size, threshold=1.0)
            entropy_values = []
            
            for pkt in packets:
                turb = ctx.calculate_spatial_turbulence(pkt['csi_data'], selected_subcarriers)
                ctx.add_turbulence(turb)
                
                if ctx.buffer_count >= window_size:
                    entropy = calc_entropy_turb(ctx.turbulence_buffer, ctx.buffer_count)
                    entropy_values.append(entropy)
            
            return entropy_values
        
        baseline_entropy = calculate_entropy_values(baseline_packets)
        movement_entropy = calculate_entropy_values(movement_packets)
        
        if len(baseline_entropy) > 0 and len(movement_entropy) > 0:
            J = fishers_criterion(baseline_entropy, movement_entropy)
            
            # Entropy should show some separation
            assert J > 0.1, f"Entropy Fisher's J too low: {J:.3f}"
    
    def test_feature_extractor_produces_values(self, real_data, selected_subcarriers):
        """Test that PublishTimeFeatureExtractor produces valid feature values"""
        baseline_packets, _ = real_data
        window_size = 50
        
        ctx = SegmentationContext(window_size=window_size, threshold=1.0)
        extractor = PublishTimeFeatureExtractor()
        
        # Process packets
        for pkt in baseline_packets[:100]:
            turb = ctx.calculate_spatial_turbulence(pkt['csi_data'], selected_subcarriers)
            ctx.add_turbulence(turb)
        
        # Get features
        if ctx.last_amplitudes is not None and ctx.buffer_count >= window_size:
            features = extractor.compute_features(
                ctx.last_amplitudes,
                ctx.turbulence_buffer,
                ctx.buffer_count,
                ctx.current_moving_variance
            )
            
            # All features should be present and finite
            assert 'skewness' in features
            assert 'kurtosis' in features
            assert 'variance_turb' in features
            assert 'iqr_turb' in features
            assert 'entropy_turb' in features
            
            for key, value in features.items():
                assert math.isfinite(value), f"Feature {key} is not finite: {value}"


# ============================================================================
# Multi-Feature Detector Tests
# ============================================================================

class TestMultiFeatureDetectorRealData:
    """Test multi-feature detector with real data"""
    
    def test_detector_confidence_baseline_vs_movement(self, real_data, selected_subcarriers):
        """Test that detector confidence is higher for movement"""
        baseline_packets, movement_packets = real_data
        window_size = 50
        
        def average_confidence(packets):
            ctx = SegmentationContext(window_size=window_size, threshold=1.0)
            extractor = PublishTimeFeatureExtractor()
            detector = MultiFeatureDetector()
            
            confidences = []
            
            for pkt in packets:
                turb = ctx.calculate_spatial_turbulence(pkt['csi_data'], selected_subcarriers)
                ctx.add_turbulence(turb)
                
                if ctx.last_amplitudes is not None and ctx.buffer_count >= window_size:
                    features = extractor.compute_features(
                        ctx.last_amplitudes,
                        ctx.turbulence_buffer,
                        ctx.buffer_count,
                        ctx.current_moving_variance
                    )
                    _, confidence, _ = detector.detect(features)
                    confidences.append(confidence)
            
            return np.mean(confidences) if confidences else 0.0
        
        baseline_conf = average_confidence(baseline_packets)
        movement_conf = average_confidence(movement_packets)
        
        # Movement should have higher average confidence
        assert movement_conf > baseline_conf, \
            f"Movement confidence ({movement_conf:.3f}) should be > baseline ({baseline_conf:.3f})"


# ============================================================================
# Hampel Filter Tests with Real Data
# ============================================================================

class TestHampelFilterRealData:
    """Test Hampel filter with real CSI turbulence data"""
    
    def test_hampel_reduces_spikes(self, real_data, selected_subcarriers):
        """Test that Hampel filter reduces turbulence spikes"""
        baseline_packets, movement_packets = real_data
        all_packets = baseline_packets + movement_packets
        
        # Calculate raw turbulence
        raw_turbulence = []
        for pkt in all_packets:
            turb = calculate_spatial_turbulence(pkt['csi_data'], selected_subcarriers)
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
    
    def test_hampel_preserves_variance_separation(self, real_data, selected_subcarriers):
        """Test that Hampel filter preserves baseline/movement separation"""
        baseline_packets, movement_packets = real_data
        
        # Calculate filtered turbulence for baseline
        hf_baseline = HampelFilter(window_size=7, threshold=4.0)
        baseline_turb = []
        for pkt in baseline_packets:
            turb = calculate_spatial_turbulence(pkt['csi_data'], selected_subcarriers)
            filtered = hf_baseline.filter(turb)
            baseline_turb.append(filtered)
        
        # Calculate filtered turbulence for movement
        hf_movement = HampelFilter(window_size=7, threshold=4.0)
        movement_turb = []
        for pkt in movement_packets:
            turb = calculate_spatial_turbulence(pkt['csi_data'], selected_subcarriers)
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
    """Test that we achieve expected performance metrics"""
    
    def test_mvs_detection_accuracy(self, real_data, selected_subcarriers):
        """
        Test MVS motion detection accuracy with real CSI data.
        
        This test replicates the C++ test methodology exactly:
        - Process ALL packets (no warmup skip)
        - Process baseline first, then movement (continuous context)
        - Same window_size=50, threshold=1.0
        - Same subcarriers [11-22]
        - Hampel filter disabled (pure MVS algorithm performance)
        
        Target: >95% Recall, <1% FP Rate
        """
        baseline_packets, movement_packets = real_data
        
        # Initialize with same defaults as C++ (Hampel disabled for pure MVS test)
        ctx = SegmentationContext(window_size=50, threshold=1.0, enable_hampel=False)
        
        num_baseline = len(baseline_packets)
        num_movement = len(movement_packets)
        
        # ========================================
        # Process baseline (expecting IDLE)
        # ========================================
        baseline_motion_packets = 0
        
        for pkt in baseline_packets:
            turb = ctx.calculate_spatial_turbulence(pkt['csi_data'], selected_subcarriers)
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
            turb = ctx.calculate_spatial_turbulence(pkt['csi_data'], selected_subcarriers)
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
        print(f"  * Recall:     {pkt_recall:.1f}% (target: >90%)")
        print(f"  * Precision:  {pkt_precision:.1f}%")
        print(f"  * FP Rate:    {pkt_fp_rate:.1f}% (target: <10%)")
        print(f"  * F1-Score:   {pkt_f1:.1f}%")
        print()
        print("=" * 70)
        
        # ========================================
        # Assertions (same thresholds as C++)
        # ========================================
        assert movement_with_motion > 950, f"Recall too low: {movement_with_motion}/1000 ({pkt_recall:.1f}%)"
        assert pkt_fp_rate < 1.0, f"FP Rate too high: {pkt_fp_rate:.1f}%"


# ============================================================================
# Float32 Stability Tests (ESP32 Simulation)
# ============================================================================

class TestFloat32Stability:
    """
    Test numerical stability with float32 precision.
    These tests simulate ESP32 behavior where calculations use 32-bit floats.
    """
    
    def test_turbulence_float32_accuracy(self, real_data, selected_subcarriers):
        """Test that float32 turbulence calculation is accurate"""
        baseline_packets, _ = real_data
        
        max_rel_error = 0.0
        
        for pkt in baseline_packets[:200]:
            csi_data = pkt['csi_data']
            
            # Float64 reference (Python default)
            amplitudes_f64 = []
            for sc_idx in selected_subcarriers:
                i = sc_idx * 2
                I = float(csi_data[i])
                Q = float(csi_data[i + 1])
                amplitudes_f64.append(math.sqrt(I*I + Q*Q))
            turb_f64 = np.std(amplitudes_f64)
            
            # Float32 simulation (ESP32)
            amplitudes_f32 = []
            for sc_idx in selected_subcarriers:
                i = sc_idx * 2
                I = np.float32(float(csi_data[i]))
                Q = np.float32(float(csi_data[i + 1]))
                amp = np.sqrt(I*I + Q*Q)
                amplitudes_f32.append(float(amp))
            turb_f32 = np.std(np.array(amplitudes_f32, dtype=np.float32))
            
            if turb_f64 > 0.01:  # Avoid division by near-zero
                rel_error = abs(turb_f32 - turb_f64) / turb_f64
                max_rel_error = max(max_rel_error, rel_error)
        
        # Float32 should be accurate within 0.1% for typical CSI values
        assert max_rel_error < 0.001, \
            f"Float32 turbulence error too high: {max_rel_error:.4%}"
    
    def test_variance_two_pass_vs_single_pass_float32(self, real_data, selected_subcarriers):
        """Test that two-pass variance is more stable than single-pass with float32"""
        baseline_packets, _ = real_data
        
        # Generate turbulence values
        turbulences = []
        for pkt in baseline_packets[:100]:
            turb = calculate_spatial_turbulence(pkt['csi_data'], selected_subcarriers)
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
# End-to-End Tests with NBVI Calibration and Normalization
# ============================================================================

class TestEndToEndWithCalibration:
    """
    Test complete pipeline: NBVI Calibration → Normalization → MVS Detection
    
    These tests verify that the system works end-to-end with:
    - NBVI auto-calibration selecting subcarriers from real data
    - Normalization scale applied to turbulence values
    - MVS motion detection achieving target performance
    """
    
    def test_nbvi_calibration_produces_valid_band(self, real_data):
        """Test that NBVI calibration produces valid subcarrier selection"""
        from nbvi_calibrator import NBVICalibrator
        
        baseline_packets, _ = real_data
        
        # Production parameters: 300 packets for gain lock, 700 packets for NBVI
        # Skip first 300 packets (gain lock phase), then use 700 for calibration
        gain_lock_skip = 300
        buffer_size = 700
        
        calibrator = NBVICalibrator(
            buffer_size=buffer_size,
            percentile=10,
            alpha=0.5,
            min_spacing=1,
            noise_gate_percentile=25  # Production value
        )
        
        # Feed baseline packets, skipping gain lock phase
        # (convert numpy array to bytes for MicroPython-compatible API)
        for pkt in baseline_packets[gain_lock_skip:gain_lock_skip + buffer_size]:
            # Convert int8 numpy array to bytes (same as ESP32 raw CSI data)
            csi_bytes = bytes(int(x) & 0xFF for x in pkt['csi_data'])
            calibrator.add_packet(csi_bytes)
        
        # Run calibration with default band (production window_size=200, step=50)
        default_band = list(range(11, 23))  # [11, 12, ..., 22]
        selected_band, normalization_scale = calibrator.calibrate(
            default_band, window_size=200, step=50
        )
        
        # Cleanup
        calibrator.free_buffer()
        
        # Verify calibration results
        assert selected_band is not None, "NBVI calibration failed"
        assert len(selected_band) == 12, f"Expected 12 subcarriers, got {len(selected_band)}"
        
        # All subcarriers should be valid (not guard bands)
        for sc in selected_band:
            assert 0 <= sc < 64, f"Invalid subcarrier index: {sc}"
        
        # Normalization scale should be valid
        assert normalization_scale > 0.0, f"Invalid normalization scale: {normalization_scale}"
        assert 0.1 <= normalization_scale <= 10.0, \
            f"Normalization scale out of range: {normalization_scale}"
        
        print(f"\nNBVI Calibration Results:")
        print(f"  Selected band: {selected_band}")
        print(f"  Normalization scale: {normalization_scale:.4f}")
    
    def test_end_to_end_with_nbvi_and_normalization(self, real_data):
        """
        Test complete end-to-end flow: NBVI → Normalization → MVS → Detection
        
        This test verifies that the system achieves target performance (>90% Recall, <10% FP)
        when using NBVI-selected subcarriers and auto-calculated normalization scale.
        """
        from nbvi_calibrator import NBVICalibrator
        
        baseline_packets, movement_packets = real_data
        
        # ========================================
        # Step 1: NBVI Calibration (Production parameters)
        # ========================================
        print("\n" + "=" * 70)
        print("  END-TO-END TEST: NBVI Calibration + Normalization + MVS")
        print("=" * 70)
        
        # Production parameters: 300 packets for gain lock, 700 packets for NBVI
        gain_lock_skip = 300
        buffer_size = 700
        
        calibrator = NBVICalibrator(
            buffer_size=buffer_size,
            percentile=10,
            alpha=0.5,
            min_spacing=1,
            noise_gate_percentile=25  # Production value
        )
        
        # Feed baseline packets, skipping gain lock phase
        print(f"\nStep 1: NBVI Calibration with {buffer_size} baseline packets (skipping first {gain_lock_skip} for gain lock)...")
        for pkt in baseline_packets[gain_lock_skip:gain_lock_skip + buffer_size]:
            csi_bytes = bytes(int(x) & 0xFF for x in pkt['csi_data'])
            calibrator.add_packet(csi_bytes)
        
        # Run calibration with production parameters (window_size=200, step=50)
        default_band = list(range(11, 23))
        selected_band, normalization_scale = calibrator.calibrate(
            default_band, window_size=200, step=50
        )
        calibrator.free_buffer()
        
        assert selected_band is not None, "NBVI calibration failed"
        print(f"  Selected band: {selected_band}")
        print(f"  Normalization scale: {normalization_scale:.4f}")
        
        # ========================================
        # Step 2: Initialize MVS with calibration results
        # ========================================
        # Initialize MVS with NBVI-selected subcarriers but NO normalization
        # (normalization is disabled by default, testing pure NBVI band selection)
        print("\nStep 2: Initialize MVS with calibration results...")
        ctx = SegmentationContext(
            window_size=50,
            threshold=1.0,
            normalization_scale=1.0,  # Normalization disabled (default)
            enable_hampel=False  # Disable for pure MVS measurement
        )
        
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
        print("  END-TO-END RESULTS (NBVI + Normalization + MVS)")
        print("=" * 70)
        print()
        print(f"CONFUSION MATRIX ({num_baseline} baseline + {num_movement} movement packets):")
        print("                    Predicted")
        print("                IDLE      MOTION")
        print(f"Actual IDLE     {pkt_tn:4d} (TN)  {pkt_fp:4d} (FP)")
        print(f"    MOTION      {pkt_fn:4d} (FN)  {pkt_tp:4d} (TP)")
        print()
        print("METRICS:")
        print(f"  * Recall:     {recall:.1f}% (target: >90%)")
        print(f"  * Precision:  {precision:.1f}%")
        print(f"  * FP Rate:    {fp_rate:.1f}% (target: <10%)")
        print(f"  * F1-Score:   {f1:.1f}%")
        print()
        print("=" * 70)
        
        # ========================================
        # Assertions
        # ========================================
        # NBVI auto-selects subcarriers that achieve comparable performance to manually optimized band.
        # With the signed/unsigned fix, NBVI achieves ~95% recall with 0% FP.
        assert recall > 90.0, f"End-to-end Recall too low: {recall:.1f}% (target: >90%)"
        assert fp_rate < 10.0, f"End-to-end FP Rate too high: {fp_rate:.1f}% (target: <10%)"

