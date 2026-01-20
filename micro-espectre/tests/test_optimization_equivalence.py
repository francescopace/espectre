"""
Micro-ESPectre - Optimization Equivalence Tests

Converted from tools/16_test_optimization_equivalence.py.
Tests that optimizations produce identical results to original implementations.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import pytest
import numpy as np
from segmentation import SegmentationContext
from csi_utils import calculate_spatial_turbulence

# Test configuration
WINDOW_SIZE = 50
HAMPEL_WINDOW = 7
HAMPEL_THRESHOLD = 4.0
TOLERANCE = 1e-6


class OriginalRunningVariance:
    """Original O(1) running variance"""
    
    def __init__(self, window_size):
        self.window_size = window_size
        self.buffer = [0.0] * window_size
        self.buffer_index = 0
        self.buffer_count = 0
        self.sum = 0.0
        self.sum_sq = 0.0
    
    def add(self, value):
        if self.buffer_count < self.window_size:
            self.sum += value
            self.sum_sq += value * value
            self.buffer_count += 1
        else:
            old_value = self.buffer[self.buffer_index]
            self.sum -= old_value
            self.sum_sq -= old_value * old_value
            self.sum += value
            self.sum_sq += value * value
        
        self.buffer[self.buffer_index] = value
        self.buffer_index = (self.buffer_index + 1) % self.window_size
        return self._get_variance()
    
    def _get_variance(self):
        if self.buffer_count < self.window_size:
            return 0.0
        n = self.buffer_count
        mean = self.sum / n
        mean_sq = self.sum_sq / n
        return max(0.0, mean_sq - mean * mean)


class OriginalHampelFilter:
    """Original Hampel filter (before optimization)"""
    
    def __init__(self, window_size=5, threshold=3.0):
        self.window_size = window_size
        self.threshold = threshold
        self.buffer = []
    
    def filter(self, value):
        self.buffer.append(value)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        
        if len(self.buffer) < 3:
            return value
        
        sorted_buffer = list(self.buffer)
        sorted_buffer.sort()
        
        n = len(sorted_buffer)
        median = sorted_buffer[n // 2]
        
        deviations = [abs(v - median) for v in self.buffer]
        deviations.sort()
        mad = deviations[n // 2]
        
        if mad > 1e-6:
            deviation = abs(value - median) / (1.4826 * mad)
            if deviation > self.threshold:
                return median
        
        return value


class OptimizedTwoPassVariance:
    """Two-pass variance (like C++ implementation)"""
    
    def __init__(self, window_size):
        self.window_size = window_size
        self.buffer = [0.0] * window_size
        self.buffer_index = 0
        self.buffer_count = 0
    
    def add(self, value):
        self.buffer[self.buffer_index] = value
        self.buffer_index = (self.buffer_index + 1) % self.window_size
        if self.buffer_count < self.window_size:
            self.buffer_count += 1
        return self._calculate_variance_two_pass()
    
    def _calculate_variance_two_pass(self):
        if self.buffer_count < self.window_size:
            return 0.0
        return SegmentationContext.compute_variance_two_pass(self.buffer[:self.buffer_count])


class OptimizedHampelFilter:
    """Hampel filter with pre-allocated buffers and insertion sort"""
    
    def __init__(self, window_size=5, threshold=3.0):
        self.window_size = window_size
        self.threshold = threshold
        self.buffer = [0.0] * window_size
        self.sorted_buffer = [0.0] * window_size
        self.deviations = [0.0] * window_size
        self.count = 0
        self.index = 0
    
    def _insertion_sort(self, arr, n):
        for i in range(1, n):
            key = arr[i]
            j = i - 1
            while j >= 0 and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
    
    def filter(self, value):
        self.buffer[self.index] = value
        self.index = (self.index + 1) % self.window_size
        if self.count < self.window_size:
            self.count += 1
        
        if self.count < 3:
            return value
        
        n = self.count
        
        for i in range(n):
            self.sorted_buffer[i] = self.buffer[i]
        
        self._insertion_sort(self.sorted_buffer, n)
        median = self.sorted_buffer[n // 2]
        
        for i in range(n):
            self.deviations[i] = abs(self.buffer[i] - median)
        
        self._insertion_sort(self.deviations, n)
        mad = self.deviations[n // 2]
        
        if mad > 1e-6:
            deviation = abs(value - median) / (1.4826 * mad)
            if deviation > self.threshold:
                return median
        
        return value


@pytest.fixture
def turbulence_values(real_csi_data_available, real_baseline_packets, 
                      real_movement_packets, default_subcarriers):
    """Load turbulence values from real CSI data"""
    if not real_csi_data_available:
        # Fall back to synthetic data
        np.random.seed(42)
        baseline = list(np.random.normal(5.0, 0.5, 500))
        movement = list(np.random.normal(10.0, 3.0, 500))
        return baseline + movement
    
    values = []
    for packet in real_baseline_packets:
        turb = calculate_spatial_turbulence(packet['csi_data'], default_subcarriers)
        values.append(float(turb))
    
    for packet in real_movement_packets:
        turb = calculate_spatial_turbulence(packet['csi_data'], default_subcarriers)
        values.append(float(turb))
    
    return values


class TestVarianceEquivalence:
    """Test variance algorithm equivalence"""
    
    def test_variance_within_tolerance(self, turbulence_values):
        """Test that two-pass and running variance produce similar results"""
        original = OriginalRunningVariance(WINDOW_SIZE)
        optimized = OptimizedTwoPassVariance(WINDOW_SIZE)
        
        max_diff = 0.0
        mismatches = []
        
        for i, value in enumerate(turbulence_values):
            orig_var = original.add(value)
            opt_var = optimized.add(value)
            
            diff = abs(orig_var - opt_var)
            max_diff = max(max_diff, diff)
            
            if diff > TOLERANCE and (orig_var > 0 or opt_var > 0):
                mismatches.append({
                    'index': i,
                    'original': orig_var,
                    'optimized': opt_var,
                    'diff': diff
                })
        
        assert len(mismatches) == 0, f"Found {len(mismatches)} mismatches, max_diff={max_diff}"


class TestHampelEquivalence:
    """Test Hampel filter equivalence"""
    
    def test_hampel_within_tolerance(self, turbulence_values):
        """Test that optimized Hampel produces same results"""
        original = OriginalHampelFilter(HAMPEL_WINDOW, HAMPEL_THRESHOLD)
        optimized = OptimizedHampelFilter(HAMPEL_WINDOW, HAMPEL_THRESHOLD)
        
        max_diff = 0.0
        mismatches = []
        
        for i, value in enumerate(turbulence_values):
            orig_filtered = original.filter(value)
            opt_filtered = optimized.filter(value)
            
            diff = abs(orig_filtered - opt_filtered)
            max_diff = max(max_diff, diff)
            
            if diff > TOLERANCE:
                mismatches.append({
                    'index': i,
                    'input': value,
                    'original': orig_filtered,
                    'optimized': opt_filtered
                })
        
        assert len(mismatches) == 0, f"Found {len(mismatches)} mismatches"
    
    def test_outlier_detection_count_matches(self, turbulence_values):
        """Test that outlier detection counts match"""
        original = OriginalHampelFilter(HAMPEL_WINDOW, HAMPEL_THRESHOLD)
        optimized = OptimizedHampelFilter(HAMPEL_WINDOW, HAMPEL_THRESHOLD)
        
        outliers_orig = 0
        outliers_opt = 0
        
        for value in turbulence_values:
            orig_filtered = original.filter(value)
            opt_filtered = optimized.filter(value)
            
            if orig_filtered != value:
                outliers_orig += 1
            if opt_filtered != value:
                outliers_opt += 1
        
        assert outliers_orig == outliers_opt


class TestFullPipelineEquivalence:
    """Test full pipeline: Hampel + Variance"""
    
    def test_pipeline_variance_matches(self, turbulence_values):
        """Test full pipeline produces identical variance"""
        # Original pipeline
        orig_hampel = OriginalHampelFilter(HAMPEL_WINDOW, HAMPEL_THRESHOLD)
        orig_variance = OriginalRunningVariance(WINDOW_SIZE)
        
        # Optimized pipeline
        opt_hampel = OptimizedHampelFilter(HAMPEL_WINDOW, HAMPEL_THRESHOLD)
        opt_variance = OptimizedTwoPassVariance(WINDOW_SIZE)
        
        max_diff = 0.0
        mismatches = []
        
        for i, value in enumerate(turbulence_values):
            orig_filtered = orig_hampel.filter(value)
            orig_var = orig_variance.add(orig_filtered)
            
            opt_filtered = opt_hampel.filter(value)
            opt_var = opt_variance.add(opt_filtered)
            
            diff = abs(orig_var - opt_var)
            max_diff = max(max_diff, diff)
            
            if diff > TOLERANCE and (orig_var > 0 or opt_var > 0):
                mismatches.append({'index': i, 'diff': diff})
        
        assert len(mismatches) == 0


class TestMotionDetectionEquivalence:
    """Test motion detection state equivalence"""
    
    def test_motion_states_match(self, turbulence_values):
        """Test that motion detection states match"""
        threshold = 1.0
        
        # Original pipeline
        orig_hampel = OriginalHampelFilter(HAMPEL_WINDOW, HAMPEL_THRESHOLD)
        orig_variance = OriginalRunningVariance(WINDOW_SIZE)
        
        # Optimized pipeline
        opt_hampel = OptimizedHampelFilter(HAMPEL_WINDOW, HAMPEL_THRESHOLD)
        opt_variance = OptimizedTwoPassVariance(WINDOW_SIZE)
        
        state_mismatches = 0
        
        for value in turbulence_values:
            orig_filtered = orig_hampel.filter(value)
            orig_var = orig_variance.add(orig_filtered)
            orig_motion = orig_var > threshold
            
            opt_filtered = opt_hampel.filter(value)
            opt_var = opt_variance.add(opt_filtered)
            opt_motion = opt_var > threshold
            
            if orig_motion != opt_motion:
                state_mismatches += 1
        
        # Allow small number of mismatches at threshold boundary
        mismatch_rate = state_mismatches / len(turbulence_values) * 100
        assert mismatch_rate < 0.1, f"Mismatch rate: {mismatch_rate:.4f}%"
    
    def test_motion_counts_match(self, turbulence_values):
        """Test that motion packet counts match"""
        threshold = 1.0
        
        # Original pipeline
        orig_hampel = OriginalHampelFilter(HAMPEL_WINDOW, HAMPEL_THRESHOLD)
        orig_variance = OriginalRunningVariance(WINDOW_SIZE)
        
        # Optimized pipeline
        opt_hampel = OptimizedHampelFilter(HAMPEL_WINDOW, HAMPEL_THRESHOLD)
        opt_variance = OptimizedTwoPassVariance(WINDOW_SIZE)
        
        orig_motion_count = 0
        opt_motion_count = 0
        
        for value in turbulence_values:
            orig_filtered = orig_hampel.filter(value)
            orig_var = orig_variance.add(orig_filtered)
            if orig_var > threshold:
                orig_motion_count += 1
            
            opt_filtered = opt_hampel.filter(value)
            opt_var = opt_variance.add(opt_filtered)
            if opt_var > threshold:
                opt_motion_count += 1
        
        # Counts should be very close
        count_diff = abs(orig_motion_count - opt_motion_count)
        assert count_diff < 5  # Allow small difference

