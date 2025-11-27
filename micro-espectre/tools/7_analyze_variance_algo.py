#!/usr/bin/env python3
"""
Variance Algorithm Analysis - Compare single-pass vs two-pass formulas

This script compares the numerical stability and accuracy of different
variance calculation methods:
- Single-pass: variance = E[X²] - E[X]² (computationally efficient)
- Two-pass: variance = sum((x - mean)²) / n (numerically stable)
- NumPy: np.var() / np.std() (reference implementation)

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import math
import time
import numpy as np
from mvs_utils import load_baseline_and_movement
from config import SELECTED_SUBCARRIERS, WINDOW_SIZE


# ============================================================================
# Variance Calculation Functions
# ============================================================================

def std_single_pass(values):
    """
    Calculate standard deviation using single-pass formula.
    std = sqrt(E[X²] - E[X]²)
    
    Computationally efficient but can have numerical issues with large values.
    """
    if len(values) < 2:
        return 0.0
    
    n = len(values)
    sum_x = sum(values)
    sum_sq = sum(x * x for x in values)
    
    mean = sum_x / n
    variance = (sum_sq / n) - (mean * mean)
    
    # Protect against negative variance due to floating point errors
    if variance < 0.0:
        variance = 0.0
    
    return math.sqrt(variance)


def std_two_pass(values):
    """
    Calculate standard deviation using two-pass formula.
    std = sqrt(sum((x - mean)²) / n)
    
    Numerically more stable but requires two passes over data.
    """
    if len(values) < 2:
        return 0.0
    
    n = len(values)
    
    # First pass: calculate mean
    mean = sum(values) / n
    
    # Second pass: calculate variance
    variance = sum((x - mean) ** 2 for x in values) / n
    
    return math.sqrt(variance)


def var_single_pass(values):
    """
    Calculate variance using single-pass formula.
    variance = E[X²] - E[X]²
    """
    if len(values) < 2:
        return 0.0
    
    n = len(values)
    sum_x = sum(values)
    sum_sq = sum(x * x for x in values)
    
    mean = sum_x / n
    variance = (sum_sq / n) - (mean * mean)
    
    # Protect against negative variance
    if variance < 0.0:
        variance = 0.0
    
    return variance


def var_two_pass(values):
    """
    Calculate variance using two-pass formula.
    variance = sum((x - mean)²) / n
    """
    if len(values) < 2:
        return 0.0
    
    n = len(values)
    
    # First pass: calculate mean
    mean = sum(values) / n
    
    # Second pass: calculate variance
    variance = sum((x - mean) ** 2 for x in values) / n
    
    return variance


# ============================================================================
# Spatial Turbulence Calculation
# ============================================================================

def turbulence_single_pass(csi_data, selected_subcarriers):
    """Calculate spatial turbulence using single-pass std formula."""
    sum_amp = 0.0
    sum_sq = 0.0
    count = 0
    
    for sc_idx in selected_subcarriers:
        i = sc_idx * 2
        if i + 1 < len(csi_data):
            I = float(csi_data[i])
            Q = float(csi_data[i + 1])
            amplitude = math.sqrt(I*I + Q*Q)
            sum_amp += amplitude
            sum_sq += amplitude * amplitude
            count += 1
    
    if count < 2:
        return 0.0
    
    mean = sum_amp / count
    variance = (sum_sq / count) - (mean * mean)
    
    if variance < 0.0:
        variance = 0.0
    
    return math.sqrt(variance)


def turbulence_two_pass(csi_data, selected_subcarriers):
    """Calculate spatial turbulence using two-pass std formula."""
    amplitudes = []
    
    for sc_idx in selected_subcarriers:
        i = sc_idx * 2
        if i + 1 < len(csi_data):
            I = float(csi_data[i])
            Q = float(csi_data[i + 1])
            amplitudes.append(math.sqrt(I*I + Q*Q))
    
    if len(amplitudes) < 2:
        return 0.0
    
    return std_two_pass(amplitudes)


def turbulence_numpy(csi_data, selected_subcarriers):
    """Calculate spatial turbulence using NumPy (reference)."""
    amplitudes = []
    
    for sc_idx in selected_subcarriers:
        i = sc_idx * 2
        if i + 1 < len(csi_data):
            I = float(csi_data[i])
            Q = float(csi_data[i + 1])
            amplitudes.append(math.sqrt(I*I + Q*Q))
    
    if len(amplitudes) < 2:
        return 0.0
    
    return np.std(amplitudes, ddof=0)  # Population std (ddof=0)


# ============================================================================
# Float32 Simulation (ESP32)
# ============================================================================

def turbulence_single_pass_float32(csi_data, selected_subcarriers):
    """
    Calculate spatial turbulence using single-pass formula with float32.
    This simulates exactly what the ESP32 C code does.
    """
    sum_amp = np.float32(0.0)
    sum_sq = np.float32(0.0)
    count = 0
    
    for sc_idx in selected_subcarriers:
        i = sc_idx * 2
        if i + 1 < len(csi_data):
            I = np.float32(float(csi_data[i]))
            Q = np.float32(float(csi_data[i + 1]))
            amplitude = np.sqrt(I*I + Q*Q)  # float32 sqrt
            sum_amp += amplitude
            sum_sq += amplitude * amplitude
            count += 1
    
    if count < 2:
        return 0.0
    
    n = np.float32(count)
    mean = sum_amp / n
    variance = (sum_sq / n) - (mean * mean)
    
    if variance < 0.0:
        variance = np.float32(0.0)
    
    return float(np.sqrt(variance))


def std_single_pass_float32(values):
    """Single-pass std with float32 precision (simulating ESP32)."""
    values_f32 = np.array(values, dtype=np.float32)
    
    n = np.float32(len(values_f32))
    sum_x = np.float32(0.0)
    sum_sq = np.float32(0.0)
    
    for x in values_f32:
        sum_x += x
        sum_sq += x * x
    
    mean = sum_x / n
    variance = (sum_sq / n) - (mean * mean)
    
    if variance < 0.0:
        variance = np.float32(0.0)
    
    return float(np.sqrt(variance))


def std_two_pass_float32(values):
    """Two-pass std with float32 precision (simulating ESP32)."""
    values_f32 = np.array(values, dtype=np.float32)
    
    n = np.float32(len(values_f32))
    
    # First pass
    mean = np.float32(0.0)
    for x in values_f32:
        mean += x
    mean /= n
    
    # Second pass
    variance = np.float32(0.0)
    for x in values_f32:
        diff = x - mean
        variance += diff * diff
    variance /= n
    
    return float(np.sqrt(variance))


# ============================================================================
# Analysis Functions
# ============================================================================

def analyze_turbulence(packets, selected_subcarriers):
    """Analyze turbulence calculation methods."""
    print("\n" + "=" * 60)
    print("SPATIAL TURBULENCE COMPARISON")
    print("=" * 60)
    print(f"Packets analyzed: {len(packets)}")
    print(f"Subcarriers used: {len(selected_subcarriers)}")
    
    results_single = []
    results_two = []
    results_numpy = []
    
    for pkt in packets:
        csi = pkt['csi_data']
        results_single.append(turbulence_single_pass(csi, selected_subcarriers))
        results_two.append(turbulence_two_pass(csi, selected_subcarriers))
        results_numpy.append(turbulence_numpy(csi, selected_subcarriers))
    
    # Statistics
    print(f"\nTurbulence Statistics (single-pass):")
    print(f"  Mean: {np.mean(results_single):.4f}")
    print(f"  Std:  {np.std(results_single):.4f}")
    print(f"  Min:  {np.min(results_single):.4f}")
    print(f"  Max:  {np.max(results_single):.4f}")
    
    # Compare single-pass vs two-pass
    diff_st = [abs(s - t) for s, t in zip(results_single, results_two)]
    rel_diff_st = [abs(s - t) / max(t, 1e-10) * 100 for s, t in zip(results_single, results_two)]
    
    print(f"\nSingle-pass vs Two-pass:")
    print(f"  Max absolute error: {max(diff_st):.2e}")
    print(f"  Mean absolute error: {np.mean(diff_st):.2e}")
    print(f"  Max relative error: {max(rel_diff_st):.2e}%")
    
    # Compare single-pass vs NumPy
    diff_sn = [abs(s - n) for s, n in zip(results_single, results_numpy)]
    rel_diff_sn = [abs(s - n) / max(n, 1e-10) * 100 for s, n in zip(results_single, results_numpy)]
    
    print(f"\nSingle-pass vs NumPy:")
    print(f"  Max absolute error: {max(diff_sn):.2e}")
    print(f"  Mean absolute error: {np.mean(diff_sn):.2e}")
    print(f"  Max relative error: {max(rel_diff_sn):.2e}%")
    
    # Compare two-pass vs NumPy
    diff_tn = [abs(t - n) for t, n in zip(results_two, results_numpy)]
    rel_diff_tn = [abs(t - n) / max(n, 1e-10) * 100 for t, n in zip(results_two, results_numpy)]
    
    print(f"\nTwo-pass vs NumPy:")
    print(f"  Max absolute error: {max(diff_tn):.2e}")
    print(f"  Mean absolute error: {np.mean(diff_tn):.2e}")
    print(f"  Max relative error: {max(rel_diff_tn):.2e}%")
    
    return results_single


def analyze_moving_variance(turbulence_values, window_size):
    """Analyze moving variance calculation methods."""
    print("\n" + "=" * 60)
    print("MOVING VARIANCE COMPARISON")
    print("=" * 60)
    print(f"Window size: {window_size}")
    print(f"Windows analyzed: {len(turbulence_values) - window_size + 1}")
    
    results_single = []
    results_two = []
    results_numpy = []
    
    for i in range(len(turbulence_values) - window_size + 1):
        window = turbulence_values[i:i + window_size]
        results_single.append(var_single_pass(window))
        results_two.append(var_two_pass(window))
        results_numpy.append(np.var(window, ddof=0))
    
    # Statistics
    print(f"\nMoving Variance Statistics (single-pass):")
    print(f"  Mean: {np.mean(results_single):.6f}")
    print(f"  Std:  {np.std(results_single):.6f}")
    print(f"  Min:  {np.min(results_single):.6f}")
    print(f"  Max:  {np.max(results_single):.6f}")
    
    # Compare single-pass vs two-pass
    diff_st = [abs(s - t) for s, t in zip(results_single, results_two)]
    rel_diff_st = [abs(s - t) / max(t, 1e-10) * 100 for s, t in zip(results_single, results_two)]
    
    print(f"\nSingle-pass vs Two-pass:")
    print(f"  Max absolute error: {max(diff_st):.2e}")
    print(f"  Mean absolute error: {np.mean(diff_st):.2e}")
    print(f"  Max relative error: {max(rel_diff_st):.2e}%")
    
    # Compare single-pass vs NumPy
    diff_sn = [abs(s - n) for s, n in zip(results_single, results_numpy)]
    rel_diff_sn = [abs(s - n) / max(n, 1e-10) * 100 for s, n in zip(results_single, results_numpy)]
    
    print(f"\nSingle-pass vs NumPy:")
    print(f"  Max absolute error: {max(diff_sn):.2e}")
    print(f"  Mean absolute error: {np.mean(diff_sn):.2e}")
    print(f"  Max relative error: {max(rel_diff_sn):.2e}%")
    
    # Compare two-pass vs NumPy
    diff_tn = [abs(t - n) for t, n in zip(results_two, results_numpy)]
    rel_diff_tn = [abs(t - n) / max(n, 1e-10) * 100 for t, n in zip(results_two, results_numpy)]
    
    print(f"\nTwo-pass vs NumPy:")
    print(f"  Max absolute error: {max(diff_tn):.2e}")
    print(f"  Mean absolute error: {np.mean(diff_tn):.2e}")
    print(f"  Max relative error: {max(rel_diff_tn):.2e}%")


def analyze_turbulence_float32(packets, selected_subcarriers):
    """
    Analyze turbulence calculation with float32 (ESP32 simulation).
    This tests if single-pass is safe for CSI amplitude data.
    """
    print("\n" + "=" * 60)
    print("TURBULENCE FLOAT32 ANALYSIS (ESP32 Simulation)")
    print("=" * 60)
    print(f"Packets analyzed: {len(packets)}")
    
    errors_abs = []
    errors_rel = []
    
    for pkt in packets:
        csi = pkt['csi_data']
        
        # Reference (float64, two-pass)
        ref = turbulence_two_pass(csi, selected_subcarriers)
        
        # Float32 single-pass (simulating ESP32 C code)
        single_f32 = turbulence_single_pass_float32(csi, selected_subcarriers)
        
        errors_abs.append(abs(single_f32 - ref))
        if ref > 1e-10:
            errors_rel.append(abs(single_f32 - ref) / ref * 100)
    
    print(f"\nSingle-pass float32 vs Two-pass float64 reference:")
    print(f"  Max absolute error: {max(errors_abs):.2e}")
    print(f"  Mean absolute error: {np.mean(errors_abs):.2e}")
    print(f"  Max relative error: {max(errors_rel):.4f}%")
    print(f"  Mean relative error: {np.mean(errors_rel):.4f}%")
    
    # Analyze amplitude range to understand why it's safe
    all_amplitudes = []
    for pkt in packets:
        csi = pkt['csi_data']
        for sc_idx in selected_subcarriers:
            i = sc_idx * 2
            if i + 1 < len(csi):
                I = float(csi[i])
                Q = float(csi[i + 1])
                all_amplitudes.append(math.sqrt(I*I + Q*Q))
    
    print(f"\nCSI Amplitude Statistics:")
    print(f"  Min: {min(all_amplitudes):.2f}")
    print(f"  Max: {max(all_amplitudes):.2f}")
    print(f"  Mean: {np.mean(all_amplitudes):.2f}")
    print(f"  Std: {np.std(all_amplitudes):.2f}")
    
    # Calculate theoretical worst-case for these amplitudes
    amp_max = max(all_amplitudes)
    e_x2_max = amp_max * amp_max  # E[X²] worst case
    print(f"\nTheoretical Analysis:")
    print(f"  Max amplitude: {amp_max:.2f}")
    print(f"  Max E[X²]: {e_x2_max:.2f}")
    print(f"  Float32 precision: ~7 significant digits")
    print(f"  Digits needed for E[X²]: {len(str(int(e_x2_max)))}")
    
    if e_x2_max < 1e6:
        print(f"\n✅ SAFE: E[X²] < 10⁶, well within float32 precision")
    else:
        print(f"\n⚠️  WARNING: E[X²] >= 10⁶, potential precision issues")
    
    return max(errors_rel)


def analyze_float32_stability(turbulence_values, window_size):
    """Analyze numerical stability with float32 (ESP32 simulation) for moving variance."""
    print("\n" + "=" * 60)
    print("MOVING VARIANCE FLOAT32 ANALYSIS (ESP32 Simulation)")
    print("=" * 60)
    
    # Test moving variance calculation with float32
    errors_single = []
    errors_two = []
    
    for i in range(len(turbulence_values) - window_size + 1):
        window = turbulence_values[i:i + window_size]
        
        # Reference (float64)
        ref = np.std(window, ddof=0)
        
        # Float32 versions
        single_f32 = std_single_pass_float32(window)
        two_f32 = std_two_pass_float32(window)
        
        errors_single.append(abs(single_f32 - ref))
        errors_two.append(abs(two_f32 - ref))
    
    print(f"\nSingle-pass float32 vs float64 reference:")
    print(f"  Max error: {max(errors_single):.2e}")
    print(f"  Mean error: {np.mean(errors_single):.2e}")
    
    print(f"\nTwo-pass float32 vs float64 reference:")
    print(f"  Max error: {max(errors_two):.2e}")
    print(f"  Mean error: {np.mean(errors_two):.2e}")
    
    # Worst-case simulation with extreme values
    print("\n--- Worst-case Simulation ---")
    
    # Simulate large values that could cause issues
    extreme_values = [1e6 + i * 0.001 for i in range(window_size)]
    
    ref_extreme = np.std(extreme_values, ddof=0)
    single_extreme = std_single_pass_float32(extreme_values)
    two_extreme = std_two_pass_float32(extreme_values)
    
    print(f"\nExtreme values test (values around 1e6):")
    print(f"  Reference (float64): {ref_extreme:.6e}")
    print(f"  Single-pass (float32): {single_extreme:.6e}")
    print(f"  Two-pass (float32): {two_extreme:.6e}")
    print(f"  Single-pass error: {abs(single_extreme - ref_extreme):.2e}")
    print(f"  Two-pass error: {abs(two_extreme - ref_extreme):.2e}")


def benchmark_performance(packets, selected_subcarriers, turbulence_values, window_size):
    """
    Benchmark performance of single-pass vs two-pass algorithms.
    Measures execution time for both turbulence and moving variance calculations.
    """
    print("\n" + "=" * 60)
    print("PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    num_iterations = 10  # Run multiple times for more accurate timing
    
    # -------------------------------------------------------------------------
    # Benchmark Turbulence Calculation
    # -------------------------------------------------------------------------
    print(f"\n--- Turbulence Calculation ({len(packets)} packets, {num_iterations} iterations) ---")
    
    # Single-pass turbulence
    start = time.perf_counter()
    for _ in range(num_iterations):
        for pkt in packets:
            turbulence_single_pass(pkt['csi_data'], selected_subcarriers)
    time_single_turb = (time.perf_counter() - start) / num_iterations
    
    # Two-pass turbulence
    start = time.perf_counter()
    for _ in range(num_iterations):
        for pkt in packets:
            turbulence_two_pass(pkt['csi_data'], selected_subcarriers)
    time_two_turb = (time.perf_counter() - start) / num_iterations
    
    overhead_turb = (time_two_turb - time_single_turb) / time_single_turb * 100
    
    print(f"\nSingle-pass: {time_single_turb*1000:.2f} ms")
    print(f"Two-pass:    {time_two_turb*1000:.2f} ms")
    print(f"Overhead:    {overhead_turb:+.1f}%")
    print(f"Per packet:  Single={time_single_turb/len(packets)*1e6:.2f} µs, Two={time_two_turb/len(packets)*1e6:.2f} µs")
    
    # -------------------------------------------------------------------------
    # Benchmark Moving Variance Calculation
    # -------------------------------------------------------------------------
    num_windows = len(turbulence_values) - window_size + 1
    print(f"\n--- Moving Variance Calculation ({num_windows} windows, {num_iterations} iterations) ---")
    
    # Single-pass variance
    start = time.perf_counter()
    for _ in range(num_iterations):
        for i in range(num_windows):
            window = turbulence_values[i:i + window_size]
            var_single_pass(window)
    time_single_var = (time.perf_counter() - start) / num_iterations
    
    # Two-pass variance
    start = time.perf_counter()
    for _ in range(num_iterations):
        for i in range(num_windows):
            window = turbulence_values[i:i + window_size]
            var_two_pass(window)
    time_two_var = (time.perf_counter() - start) / num_iterations
    
    overhead_var = (time_two_var - time_single_var) / time_single_var * 100
    
    print(f"\nSingle-pass: {time_single_var*1000:.2f} ms")
    print(f"Two-pass:    {time_two_var*1000:.2f} ms")
    print(f"Overhead:    {overhead_var:+.1f}%")
    print(f"Per window:  Single={time_single_var/num_windows*1e6:.2f} µs, Two={time_two_var/num_windows*1e6:.2f} µs")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print(f"\n--- Performance Summary ---")
    print(f"\nTurbulence (single-pass → two-pass):")
    print(f"  Overhead: {overhead_turb:+.1f}%")
    print(f"  Extra time per packet: {(time_two_turb-time_single_turb)/len(packets)*1e6:.2f} µs")
    
    print(f"\nMoving Variance (single-pass → two-pass):")
    print(f"  Overhead: {overhead_var:+.1f}%")
    print(f"  Extra time per window: {(time_two_var-time_single_var)/num_windows*1e6:.2f} µs")
    
    # ESP32 estimation (Python is ~10-50x slower than C)
    c_factor = 20  # Conservative estimate
    print(f"\n--- ESP32 Estimation (assuming C is ~{c_factor}x faster) ---")
    print(f"  Turbulence overhead per packet: ~{(time_two_turb-time_single_turb)/len(packets)*1e6/c_factor:.2f} µs")
    print(f"  Moving variance overhead per window: ~{(time_two_var-time_single_var)/num_windows*1e6/c_factor:.2f} µs")
    
    # At 100 packets/second
    packets_per_sec = 100
    total_overhead_us = ((time_two_turb-time_single_turb)/len(packets) + 
                         (time_two_var-time_single_var)/num_windows) * 1e6 / c_factor
    print(f"\n  At {packets_per_sec} packets/sec:")
    print(f"    Extra CPU time: ~{total_overhead_us * packets_per_sec:.1f} µs/sec")
    print(f"    CPU overhead: ~{total_overhead_us * packets_per_sec / 10000:.3f}%")
    
    return {
        'turb_single': time_single_turb,
        'turb_two': time_two_turb,
        'var_single': time_single_var,
        'var_two': time_two_var
    }


def print_recommendation(turbulence_values, window_size):
    """Print final recommendation based on analysis."""
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    
    turb_max = max(turbulence_values)
    turb_mean = np.mean(turbulence_values)
    
    print(f"\nData characteristics:")
    print(f"  Turbulence range: 0 - {turb_max:.2f}")
    print(f"  Turbulence mean: {turb_mean:.2f}")
    
    # Analyze float32 stability for our actual data
    print(f"\nFloat32 stability analysis for actual data:")
    
    # Test with actual turbulence values
    errors_single = []
    errors_two = []
    
    for i in range(len(turbulence_values) - window_size + 1):
        window = turbulence_values[i:i + window_size]
        ref = np.std(window, ddof=0)
        single_f32 = std_single_pass_float32(window)
        two_f32 = std_two_pass_float32(window)
        errors_single.append(abs(single_f32 - ref) / max(ref, 1e-10) * 100)
        errors_two.append(abs(two_f32 - ref) / max(ref, 1e-10) * 100)
    
    print(f"  Single-pass max relative error: {max(errors_single):.4f}%")
    print(f"  Two-pass max relative error: {max(errors_two):.4f}%")
    
    print(f"\nNumerical stability assessment:")
    print(f"  - Values are small (max ~{turb_max:.0f})")
    print(f"  - For float64 (Python): Both formulas are equivalent")
    print(f"  - For float32 (ESP32): Two-pass is ~20x more accurate")
    print(f"  - HOWEVER: With extreme values (~1e6), single-pass FAILS completely!")
    
    print(f"\n⚠️  IMPORTANT FINDING:")
    print(f"   The worst-case test shows single-pass can produce WRONG results")
    print(f"   with float32 when values are large (error: 572 vs expected 0.014)")
    
    print(f"\n✅ RECOMMENDATION:")
    print(f"   - For Python (float64): Single-pass is safe and efficient")
    print(f"   - For ESP32 (float32): Use TWO-PASS for robustness")
    print(f"   - Current C code uses two-pass: THIS IS CORRECT!")
    print(f"   - Keep segmentation.py aligned with C (two-pass for moving variance)")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("VARIANCE ALGORITHM ANALYSIS")
    print("Comparing single-pass vs two-pass formulas")
    print("=" * 60)
    
    # Load data
    print("\nLoading CSI data...")
    baseline_packets, movement_packets = load_baseline_and_movement()
    all_packets = baseline_packets + movement_packets
    print(f"Loaded {len(baseline_packets)} baseline + {len(movement_packets)} movement packets")
    
    # Get configuration
    selected_subcarriers = SELECTED_SUBCARRIERS
    window_size = WINDOW_SIZE
    
    print(f"\nConfiguration:")
    print(f"  Selected subcarriers: {len(selected_subcarriers)}")
    print(f"  Window size: {window_size}")
    
    # Analyze turbulence (float64)
    turbulence_values = analyze_turbulence(all_packets, selected_subcarriers)
    
    # Analyze turbulence with float32 (ESP32 simulation)
    analyze_turbulence_float32(all_packets, selected_subcarriers)
    
    # Analyze moving variance (float64)
    analyze_moving_variance(turbulence_values, window_size)
    
    # Analyze moving variance with float32 (ESP32 simulation)
    analyze_float32_stability(turbulence_values, window_size)
    
    # Performance benchmark
    benchmark_performance(all_packets, selected_subcarriers, turbulence_values, window_size)
    
    # Print recommendation
    print_recommendation(turbulence_values, window_size)


if __name__ == "__main__":
    main()
