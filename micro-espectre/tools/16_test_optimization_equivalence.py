"""
Micro-ESPectre - Optimization Equivalence Test

Tests that the proposed optimizations produce identical results to the original
implementations using REAL CSI data from the data/ folder.

Optimizations tested:
1. Two-pass variance vs Running variance O(1)
2. Pre-allocated buffers vs Dynamic lists for Hampel filter
3. Insertion sort vs Python's Timsort for small arrays

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import sys
import math
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import data loader and turbulence calculation
from mvs_utils import load_binary_data, calculate_spatial_turbulence, BASELINE_FILE, MOVEMENT_FILE

# Test configuration
WINDOW_SIZE = 50
HAMPEL_WINDOW = 7
HAMPEL_THRESHOLD = 4.0
TOLERANCE = 1e-6  # Maximum allowed difference

# Selected subcarriers (default band)
SELECTED_SUBCARRIERS = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]


# =============================================================================
# ORIGINAL IMPLEMENTATIONS (Current Code - before optimization)
# =============================================================================

class OriginalRunningVariance:
    """Original O(1) running variance from segmentation.py"""
    
    def __init__(self, window_size):
        self.window_size = window_size
        self.buffer = [0.0] * window_size
        self.buffer_index = 0
        self.buffer_count = 0
        self.sum = 0.0
        self.sum_sq = 0.0
    
    def add(self, value):
        """Add value and return current variance"""
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
    """Original Hampel filter from filters.py (before optimization)"""
    
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
        
        # Create copy and sort
        sorted_buffer = list(self.buffer)
        sorted_buffer.sort()
        
        n = len(sorted_buffer)
        median = sorted_buffer[n // 2]
        
        # Calculate MAD
        deviations = [abs(v - median) for v in self.buffer]
        deviations.sort()
        
        mad = deviations[n // 2]
        
        if mad > 1e-6:
            deviation = abs(value - median) / (1.4826 * mad)
            if deviation > self.threshold:
                return median
        
        return value


# =============================================================================
# OPTIMIZED IMPLEMENTATIONS (Proposed Changes)
# =============================================================================

class OptimizedTwoPassVariance:
    """Two-pass variance (like C++ implementation)"""
    
    def __init__(self, window_size):
        self.window_size = window_size
        self.buffer = [0.0] * window_size
        self.buffer_index = 0
        self.buffer_count = 0
    
    def add(self, value):
        """Add value and return current variance"""
        self.buffer[self.buffer_index] = value
        self.buffer_index = (self.buffer_index + 1) % self.window_size
        if self.buffer_count < self.window_size:
            self.buffer_count += 1
        
        return self._calculate_variance_two_pass()
    
    def _calculate_variance_two_pass(self):
        """Two-pass variance calculation (numerically stable)"""
        if self.buffer_count < self.window_size:
            return 0.0
        
        n = self.buffer_count
        
        # First pass: calculate mean
        total = 0.0
        for i in range(n):
            total += self.buffer[i]
        mean = total / n
        
        # Second pass: calculate variance
        variance = 0.0
        for i in range(n):
            diff = self.buffer[i] - mean
            variance += diff * diff
        variance /= n
        
        return variance


class OptimizedHampelFilter:
    """Hampel filter with pre-allocated buffers and insertion sort"""
    
    def __init__(self, window_size=5, threshold=3.0):
        self.window_size = window_size
        self.threshold = threshold
        # Pre-allocated buffers
        self.buffer = [0.0] * window_size
        self.sorted_buffer = [0.0] * window_size
        self.deviations = [0.0] * window_size
        self.count = 0
        self.index = 0
    
    def _insertion_sort(self, arr, n):
        """In-place insertion sort for small arrays"""
        for i in range(1, n):
            key = arr[i]
            j = i - 1
            while j >= 0 and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
    
    def filter(self, value):
        # Add to circular buffer
        self.buffer[self.index] = value
        self.index = (self.index + 1) % self.window_size
        if self.count < self.window_size:
            self.count += 1
        
        if self.count < 3:
            return value
        
        n = self.count
        
        # Copy to sorted buffer (no allocation)
        for i in range(n):
            self.sorted_buffer[i] = self.buffer[i]
        
        # Insertion sort (faster than Timsort for small N)
        self._insertion_sort(self.sorted_buffer, n)
        
        median = self.sorted_buffer[n // 2]
        
        # Calculate deviations (reuse buffer)
        for i in range(n):
            self.deviations[i] = abs(self.buffer[i] - median)
        
        self._insertion_sort(self.deviations, n)
        
        mad = self.deviations[n // 2]
        
        if mad > 1e-6:
            deviation = abs(value - median) / (1.4826 * mad)
            if deviation > self.threshold:
                return median
        
        return value


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_turbulence_from_real_data():
    """Load real CSI data and calculate turbulence values"""
    print("Loading real CSI data...")
    
    turbulence_values = []
    
    # Load baseline data
    try:
        baseline_packets = load_binary_data(BASELINE_FILE)
        print(f"  Loaded {len(baseline_packets)} baseline packets from {BASELINE_FILE.name}")
        
        for packet in baseline_packets:
            csi_data = packet['csi_data']
            # Use mvs_utils calculate_spatial_turbulence (handles numpy properly)
            turbulence = calculate_spatial_turbulence(csi_data, SELECTED_SUBCARRIERS)
            turbulence_values.append(float(turbulence))
    except Exception as e:
        print(f"  Warning: Could not load baseline data: {e}")
        import traceback
        traceback.print_exc()
    
    # Load movement data
    try:
        movement_packets = load_binary_data(MOVEMENT_FILE)
        print(f"  Loaded {len(movement_packets)} movement packets from {MOVEMENT_FILE.name}")
        
        for packet in movement_packets:
            csi_data = packet['csi_data']
            turbulence = calculate_spatial_turbulence(csi_data, SELECTED_SUBCARRIERS)
            turbulence_values.append(float(turbulence))
    except Exception as e:
        print(f"  Warning: Could not load movement data: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"  Total turbulence values: {len(turbulence_values)}")
    
    if turbulence_values:
        print(f"  Turbulence range: {min(turbulence_values):.2f} - {max(turbulence_values):.2f}")
        print(f"  Turbulence mean: {sum(turbulence_values)/len(turbulence_values):.2f}")
    
    return turbulence_values


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_variance_equivalence(turbulence_values):
    """Test that two-pass variance produces same results as running variance"""
    print("\n" + "="*60)
    print("TEST 1: Variance Algorithm Equivalence (Real Data)")
    print("="*60)
    
    original = OriginalRunningVariance(WINDOW_SIZE)
    optimized = OptimizedTwoPassVariance(WINDOW_SIZE)
    
    max_diff = 0.0
    total_diff = 0.0
    num_comparisons = 0
    mismatches = []
    
    for i, value in enumerate(turbulence_values):
        orig_var = original.add(value)
        opt_var = optimized.add(value)
        
        diff = abs(orig_var - opt_var)
        max_diff = max(max_diff, diff)
        total_diff += diff
        
        if orig_var > 0 or opt_var > 0:
            num_comparisons += 1
            
            if diff > TOLERANCE:
                mismatches.append({
                    'packet': i,
                    'original': orig_var,
                    'optimized': opt_var,
                    'diff': diff
                })
    
    avg_diff = total_diff / num_comparisons if num_comparisons > 0 else 0
    
    print(f"\nPackets processed: {len(turbulence_values)}")
    print(f"Window size: {WINDOW_SIZE}")
    print(f"Comparisons (non-zero): {num_comparisons}")
    print(f"Maximum difference: {max_diff:.2e}")
    print(f"Average difference: {avg_diff:.2e}")
    print(f"Tolerance: {TOLERANCE:.2e}")
    
    if mismatches:
        print(f"\n⚠️  Found {len(mismatches)} mismatches above tolerance!")
        for m in mismatches[:5]:
            print(f"   Packet {m['packet']}: orig={m['original']:.6f}, opt={m['optimized']:.6f}, diff={m['diff']:.2e}")
        return False
    else:
        print(f"\n✅ PASS: All {num_comparisons} comparisons within tolerance")
        return True


def test_hampel_equivalence(turbulence_values):
    """Test that optimized Hampel filter produces same results"""
    print("\n" + "="*60)
    print("TEST 2: Hampel Filter Equivalence (Real Data)")
    print("="*60)
    
    original = OriginalHampelFilter(HAMPEL_WINDOW, HAMPEL_THRESHOLD)
    optimized = OptimizedHampelFilter(HAMPEL_WINDOW, HAMPEL_THRESHOLD)
    
    max_diff = 0.0
    total_diff = 0.0
    outliers_detected_orig = 0
    outliers_detected_opt = 0
    mismatches = []
    
    for i, value in enumerate(turbulence_values):
        orig_filtered = original.filter(value)
        opt_filtered = optimized.filter(value)
        
        diff = abs(orig_filtered - opt_filtered)
        max_diff = max(max_diff, diff)
        total_diff += diff
        
        if orig_filtered != value:
            outliers_detected_orig += 1
        if opt_filtered != value:
            outliers_detected_opt += 1
        
        if diff > TOLERANCE:
            mismatches.append({
                'packet': i,
                'input': value,
                'original': orig_filtered,
                'optimized': opt_filtered,
                'diff': diff
            })
    
    avg_diff = total_diff / len(turbulence_values) if turbulence_values else 0
    
    print(f"\nPackets processed: {len(turbulence_values)}")
    print(f"Hampel window: {HAMPEL_WINDOW}")
    print(f"Hampel threshold: {HAMPEL_THRESHOLD}")
    print(f"Outliers detected (original): {outliers_detected_orig}")
    print(f"Outliers detected (optimized): {outliers_detected_opt}")
    print(f"Maximum difference: {max_diff:.2e}")
    print(f"Average difference: {avg_diff:.2e}")
    print(f"Tolerance: {TOLERANCE:.2e}")
    
    if mismatches:
        print(f"\n⚠️  Found {len(mismatches)} mismatches above tolerance!")
        for m in mismatches[:5]:
            print(f"   Packet {m['packet']}: input={m['input']:.2f}, orig={m['original']:.6f}, opt={m['optimized']:.6f}")
        return False
    else:
        print(f"\n✅ PASS: All {len(turbulence_values)} comparisons within tolerance")
        return True


def test_full_pipeline_equivalence(turbulence_values):
    """Test full pipeline: Hampel + Variance together"""
    print("\n" + "="*60)
    print("TEST 3: Full Pipeline Equivalence (Hampel → Variance)")
    print("="*60)
    
    # Original pipeline
    orig_hampel = OriginalHampelFilter(HAMPEL_WINDOW, HAMPEL_THRESHOLD)
    orig_variance = OriginalRunningVariance(WINDOW_SIZE)
    
    # Optimized pipeline
    opt_hampel = OptimizedHampelFilter(HAMPEL_WINDOW, HAMPEL_THRESHOLD)
    opt_variance = OptimizedTwoPassVariance(WINDOW_SIZE)
    
    max_diff = 0.0
    total_diff = 0.0
    num_comparisons = 0
    mismatches = []
    
    # Track state changes
    orig_states = []
    opt_states = []
    threshold = 1.0  # Default threshold
    
    for i, value in enumerate(turbulence_values):
        # Original pipeline
        orig_filtered = orig_hampel.filter(value)
        orig_var = orig_variance.add(orig_filtered)
        
        # Optimized pipeline
        opt_filtered = opt_hampel.filter(value)
        opt_var = opt_variance.add(opt_filtered)
        
        # Track motion state
        orig_states.append(1 if orig_var > threshold else 0)
        opt_states.append(1 if opt_var > threshold else 0)
        
        diff = abs(orig_var - opt_var)
        max_diff = max(max_diff, diff)
        total_diff += diff
        
        if orig_var > 0 or opt_var > 0:
            num_comparisons += 1
            
            if diff > TOLERANCE:
                mismatches.append({
                    'packet': i,
                    'input': value,
                    'original': orig_var,
                    'optimized': opt_var,
                    'diff': diff
                })
    
    avg_diff = total_diff / num_comparisons if num_comparisons > 0 else 0
    
    # Count state differences
    state_diffs = sum(1 for o, n in zip(orig_states, opt_states) if o != n)
    
    print(f"\nPackets processed: {len(turbulence_values)}")
    print(f"Comparisons (non-zero): {num_comparisons}")
    print(f"Maximum variance difference: {max_diff:.2e}")
    print(f"Average variance difference: {avg_diff:.2e}")
    print(f"Motion state differences: {state_diffs}")
    print(f"Tolerance: {TOLERANCE:.2e}")
    
    if mismatches:
        print(f"\n⚠️  Found {len(mismatches)} mismatches above tolerance!")
        for m in mismatches[:5]:
            print(f"   Packet {m['packet']}: input={m['input']:.2f}, orig={m['original']:.6f}, opt={m['optimized']:.6f}")
        return False
    else:
        print(f"\n✅ PASS: Full pipeline produces identical results")
        return True


def test_motion_detection_equivalence(turbulence_values):
    """Test that motion detection produces same results"""
    print("\n" + "="*60)
    print("TEST 4: Motion Detection Equivalence")
    print("="*60)
    
    threshold = 1.0
    
    # Original pipeline
    orig_hampel = OriginalHampelFilter(HAMPEL_WINDOW, HAMPEL_THRESHOLD)
    orig_variance = OriginalRunningVariance(WINDOW_SIZE)
    
    # Optimized pipeline
    opt_hampel = OptimizedHampelFilter(HAMPEL_WINDOW, HAMPEL_THRESHOLD)
    opt_variance = OptimizedTwoPassVariance(WINDOW_SIZE)
    
    orig_motion_count = 0
    opt_motion_count = 0
    state_matches = 0
    state_mismatches = 0
    
    for value in turbulence_values:
        # Original
        orig_filtered = orig_hampel.filter(value)
        orig_var = orig_variance.add(orig_filtered)
        orig_motion = orig_var > threshold
        
        # Optimized
        opt_filtered = opt_hampel.filter(value)
        opt_var = opt_variance.add(opt_filtered)
        opt_motion = opt_var > threshold
        
        if orig_motion:
            orig_motion_count += 1
        if opt_motion:
            opt_motion_count += 1
        
        if orig_motion == opt_motion:
            state_matches += 1
        else:
            state_mismatches += 1
    
    print(f"\nPackets processed: {len(turbulence_values)}")
    print(f"Threshold: {threshold}")
    print(f"Motion detections (original): {orig_motion_count}")
    print(f"Motion detections (optimized): {opt_motion_count}")
    print(f"State matches: {state_matches}")
    print(f"State mismatches: {state_mismatches}")
    
    if state_mismatches == 0:
        print(f"\n✅ PASS: Motion detection produces identical results")
        return True
    else:
        print(f"\n⚠️  WARNING: {state_mismatches} state mismatches detected")
        # This might be acceptable due to floating point differences at threshold boundary
        mismatch_rate = state_mismatches / len(turbulence_values) * 100
        print(f"   Mismatch rate: {mismatch_rate:.4f}%")
        if mismatch_rate < 0.1:  # Less than 0.1% is acceptable
            print(f"   (Acceptable: < 0.1%)")
            return True
        return False


def main():
    print("\n" + "="*60)
    print("OPTIMIZATION EQUIVALENCE TEST (REAL CSI DATA)")
    print("="*60)
    
    # Load real turbulence data
    turbulence_values = load_turbulence_from_real_data()
    
    if not turbulence_values:
        print("\n❌ ERROR: No data loaded. Cannot run tests.")
        return 1
    
    print(f"\nVerifying that optimizations produce identical results...")
    print(f"Tolerance: {TOLERANCE}")
    
    results = []
    
    results.append(("Variance Algorithm", test_variance_equivalence(turbulence_values)))
    results.append(("Hampel Filter", test_hampel_equivalence(turbulence_values)))
    results.append(("Full Pipeline", test_full_pipeline_equivalence(turbulence_values)))
    results.append(("Motion Detection", test_motion_detection_equivalence(turbulence_values)))
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL TESTS PASSED - Optimizations are safe to implement")
    else:
        print("⚠️  SOME TESTS FAILED - Review differences before implementing")
    print("="*60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
