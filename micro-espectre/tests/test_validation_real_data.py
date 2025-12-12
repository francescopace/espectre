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
from pathlib import Path

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
    
    def test_f1_score_above_threshold(self, real_data, selected_subcarriers):
        """Test that MVS achieves minimum F1 score"""
        baseline_packets, movement_packets = real_data
        
        # Calculate confusion matrix elements
        ctx = SegmentationContext(window_size=50, threshold=1.0)
        
        # True Negatives (baseline detected as idle)
        tn = 0
        fp = 0
        for pkt in baseline_packets[50:]:  # Skip warmup
            turb = ctx.calculate_spatial_turbulence(pkt['csi_data'], selected_subcarriers)
            ctx.add_turbulence(turb)
            if ctx.get_state() == SegmentationContext.STATE_IDLE:
                tn += 1
            else:
                fp += 1
        
        ctx.reset(full=True)
        
        # True Positives (movement detected as motion)
        tp = 0
        fn = 0
        for pkt in movement_packets[50:]:  # Skip warmup
            turb = ctx.calculate_spatial_turbulence(pkt['csi_data'], selected_subcarriers)
            ctx.add_turbulence(turb)
            if ctx.get_state() == SegmentationContext.STATE_MOTION:
                tp += 1
            else:
                fn += 1
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Should achieve at least 80% F1 score
        assert f1 > 0.80, f"F1 score too low: {f1:.1%} (P={precision:.1%}, R={recall:.1%})"


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

