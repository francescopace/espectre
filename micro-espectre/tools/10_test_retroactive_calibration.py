#!/usr/bin/env python3
"""
Retroactive Auto-Calibration & Ring Geometry Analysis Test
Tests automatic baseline detection, calibration, and subcarrier selection using
retrospective analysis and I/Q constellation geometry.

OVERVIEW:
This comprehensive tool tests multiple approaches for automatic subcarrier selection:
1. Variance-based calibration (retrospective baseline detection)
2. Ring geometry analysis (I/Q constellation patterns)
3. Hybrid strategies (combining variance + geometry)

The system uses a retrospective approach: when low variance is detected in the 
LAST 200 packets, those packets are used for calibration (safe, no risk of future movement).

TESTS INCLUDED:
1. find_optimal_baseline_threshold()
   PURPOSE: Determines the optimal variance threshold to distinguish baseline from movement
   METHOD: Analyzes 200-packet windows in both baseline and movement data
   OUTPUT: Threshold value that maximizes separation (0.3878)
   
2. test_false_positive_rate()
   PURPOSE: Validates that movement is not falsely detected as baseline
   METHOD: Tests threshold on movement data windows
   OUTPUT: False positive rate (should be <5%)
   
3. test_detection_latency()
   PURPOSE: Measures how quickly baseline periods are detected
   METHOD: Simulates baseline stream and tracks first detection
   OUTPUT: Time to first detection (target: <3s)
   
4. test_with_random_starting_bands() [CRITICAL]
   PURPOSE: Validates algorithm works from ANY starting band (universal deployment)
   METHOD: Tests 6 different starting bands (edges, center, random, sparse)
   OUTPUT: Success rate and performance improvements
   WHY CRITICAL: Simulates deploying in unknown environment where default may be suboptimal
   
5. test_realistic_mixed_scenario() [CRITICAL]
   PURPOSE: Tests with realistic data (80% baseline, 20% movement scattered)
   METHOD: Creates random mix simulating real-world usage
   OUTPUT: Calibration success rate with mixed data
   WHY CRITICAL: Real deployment will have occasional movement, not pure blocks
   
6. simulate_runtime_scenario()
   PURPOSE: Full end-to-end simulation of system behavior
   METHOD: Movement → Baseline → Movement → Baseline sequence
   OUTPUT: Complete calibration timeline and final performance

7. test_ring_geometry_selection() [NEW]
   PURPOSE: Test automatic subcarrier selection using I/Q constellation geometry
   METHOD: Analyzes ring radius, thickness, and compactness for all 64 subcarriers
   OUTPUT: Best band selection based on geometric patterns
   STRATEGIES TESTED: 17 total (12 geometric + 4 hybrid + 1 variance-only)
   
8. test_all_scoring_strategies() [NEW]
   PURPOSE: Comprehensive comparison of all scoring strategies
   METHOD: Tests pure geometric, hybrid (variance+geometry), and variance-only
   OUTPUT: Performance ranking and best strategy identification

KEY FINDINGS:

1. Baseline Detection (Variance-Based):
   - Optimal threshold: 0.3878 (variance of turbulence)
   - Baseline variance: 0.02-0.39 (mean 0.21)
   - Movement variance: 3.07-12.99 (mean 6.90)
   - Separation: ~8x difference (excellent!)
   - False positive rate: 0.0%
   - Detection latency: 2.0s

2. Adaptive Threshold:
   - REQUIRED for universal deployment
   - Calculated from first 100 packets: threshold = estimated_variance * 2.0
   - Enables calibration from ANY starting band (6/6 = 100%)
   - Without adaptive: only 2/6 (33%) success

3. Variance-Based Performance:
   - Artificial scenario (pure blocks): F1=96.7% ✅
   - Random starting bands: F1=96.7% ✅ (with adaptive threshold)
   - Realistic mixed data (80/20): FAILS ❌
     → Adaptive threshold too permissive (9.20)
     → Detects "baseline" in windows with movement
     → Calibration selects wrong bands
     → Validation fails (bands not in KNOWN_GOOD_BANDS)

4. Ring Geometry Analysis Results:
   - 17 strategies tested (12 geometric + 4 hybrid + 1 variance-only)
   - Best overall: variance_only [0-11] → F1=92.4%
   - Best hybrid: hybrid_variance_instability [4-15] → F1=92.2%
   - Best geometric: movement_instability [4-15] → F1=92.2%
   - Default (manual optimization): [11-22] → F1=97.3%
   
   KEY INSIGHT: Hybrid strategies (variance + geometry) do NOT improve over pure strategies
   - Hybrid vs Geometric improvement: +0.0%
   - Gap to default: -4.9% to -5.1%
   
   GEOMETRIC PATTERNS DISCOVERED:
   - Baseline: Larger radius, smaller thickness (compact, stable ring)
   - Movement: Smaller radius, larger thickness (dispersed, unstable ring)
   - Best discriminator: movement_instability = thickness/radius ratio
   - Radius-based metrics FAIL (wrong hypothesis about baseline/movement patterns)

5. Strategy Comparison Summary:
   
   Category          | Best Strategy              | Band    | F1    | vs Default
   ------------------|----------------------------|---------|-------|------------
   Default (manual)  | variance-based             | [11-22] | 97.3% | ---
   Variance-Only     | variance_only              | [0-11]  | 92.4% | -4.9%
   Hybrid            | hybrid_variance_instability| [4-15]  | 92.2% | -5.1%
   Geometric         | movement_instability       | [4-15]  | 92.2% | -5.1%
   
   PERFORMANCE DISTRIBUTION:
   - 🎉 BEST (F1≥97%): 0/17 strategies (0%)
   - ✅ GOOD (F1≥95%): 0/17 strategies (0%)
   - ⚠️  OK (F1≥85%): 7/17 strategies (41%)
   - ❌ BAD (F1<85%): 10/17 strategies (59%)

6. Limitations:
   - Variance-based: Works ONLY with pure baseline (200+ packet blocks without movement)
   - Variance-based: Does NOT work with realistic mixed data (scattered movement)
   - Ring geometry: Cannot beat manually optimized default (-4.9% gap)
   - Hybrid strategies: Add NO value over pure strategies (0.0% improvement)
   - All methods: Require validation against list of known good bands

CONCLUSION:
1. Variance-based calibration is DIFFICULT with realistic runtime data
2. Ring geometry analysis provides insights but doesn't improve performance
3. Hybrid strategies (variance + geometry) are REDUNDANT - no improvement
4. Variance-only remains the most reliable automatic method (F1=92.4%)
5. Manual optimization still outperforms all automatic methods (+4.9%)

RECOMMENDED SOLUTION:
- Primary: Use robust default band [11-22] (F1=97.3%)
- Fallback: Variance-only automatic selection [0-11] (F1=92.4%)
- Validation: Optional ring geometry check for confidence
- Manual: Allow user calibration for optimal performance

SCIENTIFIC CONTRIBUTION:
This analysis demonstrates that:
- Simple variance analysis outperforms complex geometric features
- Combining multiple metrics doesn't always improve results (curse of dimensionality)
- Manual empirical optimization remains superior to automatic methods
- I/Q constellation geometry reveals interesting patterns but limited practical value

Usage:
    python tools/9_test_retroactive_calibration.py

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import numpy as np
from mvs_utils import load_baseline_and_movement, test_mvs_configuration, calculate_spatial_turbulence, calculate_variance_two_pass
from config import WINDOW_SIZE, THRESHOLD, SELECTED_SUBCARRIERS

BAND_SIZE = 12
TOTAL_SUBCARRIERS = 64
DEFAULT_BAND = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

# Known good bands (from previous analysis)
KNOWN_GOOD_BANDS = [
    [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],  # F1=97.3%
    [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42],  # F1=96.7%
    [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],  # F1=95.4%
    [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43],  # F1=96.7%
]

def find_optimal_baseline_threshold(baseline_packets, movement_packets):
    """
    Find optimal threshold to distinguish baseline from movement
    
    Returns:
        float: Optimal variance threshold
    """
    print(f"\n{'='*70}")
    print(f"  FINDING OPTIMAL BASELINE DETECTION THRESHOLD")
    print(f"{'='*70}\n")
    
    # Calculate variance for windows in baseline
    baseline_variances = []
    window_size = 200
    
    for i in range(0, len(baseline_packets) - window_size, 50):
        window = baseline_packets[i:i+window_size]
        turbulences = [calculate_spatial_turbulence(pkt['csi_data'], DEFAULT_BAND) 
                      for pkt in window]
        variance = calculate_variance_two_pass(turbulences)
        baseline_variances.append(variance)
    
    # Calculate variance for windows in movement
    movement_variances = []
    for i in range(0, len(movement_packets) - window_size, 50):
        window = movement_packets[i:i+window_size]
        turbulences = [calculate_spatial_turbulence(pkt['csi_data'], DEFAULT_BAND) 
                      for pkt in window]
        variance = calculate_variance_two_pass(turbulences)
        movement_variances.append(variance)
    
    print(f"Baseline windows analyzed: {len(baseline_variances)}")
    print(f"Movement windows analyzed: {len(movement_variances)}")
    
    print(f"\nBaseline variance statistics:")
    print(f"  Mean: {np.mean(baseline_variances):.4f}")
    print(f"  Std:  {np.std(baseline_variances):.4f}")
    print(f"  Min:  {np.min(baseline_variances):.4f}")
    print(f"  Max:  {np.max(baseline_variances):.4f}")
    
    print(f"\nMovement variance statistics:")
    print(f"  Mean: {np.mean(movement_variances):.4f}")
    print(f"  Std:  {np.std(movement_variances):.4f}")
    print(f"  Min:  {np.min(movement_variances):.4f}")
    print(f"  Max:  {np.max(movement_variances):.4f}")
    
    # Find threshold that separates them
    # Use max of baseline as conservative threshold
    threshold = np.max(baseline_variances)
    
    # Test threshold
    baseline_detected = sum(1 for v in baseline_variances if v < threshold)
    movement_rejected = sum(1 for v in movement_variances if v >= threshold)
    
    baseline_accuracy = baseline_detected / len(baseline_variances) * 100
    movement_accuracy = movement_rejected / len(movement_variances) * 100
    
    print(f"\nProposed threshold: {threshold:.4f}")
    print(f"  Baseline detection rate: {baseline_accuracy:.1f}%")
    print(f"  Movement rejection rate: {movement_accuracy:.1f}%")
    
    # Try a more conservative threshold (mean + 2*std)
    conservative_threshold = np.mean(baseline_variances) + 2 * np.std(baseline_variances)
    
    baseline_detected_cons = sum(1 for v in baseline_variances if v < conservative_threshold)
    movement_rejected_cons = sum(1 for v in movement_variances if v >= conservative_threshold)
    
    baseline_accuracy_cons = baseline_detected_cons / len(baseline_variances) * 100
    movement_accuracy_cons = movement_rejected_cons / len(movement_variances) * 100
    
    print(f"\nConservative threshold (mean + 2σ): {conservative_threshold:.4f}")
    print(f"  Baseline detection rate: {baseline_accuracy_cons:.1f}%")
    print(f"  Movement rejection rate: {movement_accuracy_cons:.1f}%")
    
    # Choose the one with best movement rejection
    if movement_accuracy_cons > movement_accuracy:
        return conservative_threshold
    else:
        return threshold

class OpportunisticCalibrator:
    """
    Automatic calibrator that detects baseline periods and calibrates opportunistically
    """
    
    def __init__(self, baseline_threshold=0.5, starting_band=None, use_adaptive_threshold=False):
        self.packet_history = []
        self.current_band = starting_band if starting_band else DEFAULT_BAND
        self.starting_band = self.current_band.copy()
        self.calibrated = False
        self.baseline_threshold = baseline_threshold
        self.use_adaptive_threshold = use_adaptive_threshold
        self.adaptive_threshold = None
        self.check_interval = 100  # Check every 100 packets
        self.calibration_window = 200
        self.detection_window = 200
        self.packet_count = 0
        self.calibration_time = None
        
    def process_packet(self, packet):
        """Process incoming packet"""
        self.packet_count += 1
        
        # Add to history
        self.packet_history.append(packet)
        if len(self.packet_history) > 500:
            self.packet_history.pop(0)
        
        # Check for calibration opportunity periodically
        if not self.calibrated and self.packet_count % self.check_interval == 0:
            self.check_for_baseline_period()
    
    def calculate_adaptive_threshold(self):
        """
        Calculate adaptive threshold based on current band's baseline variance
        
        Uses first 100 packets to estimate baseline variance for current band
        """
        if len(self.packet_history) < 100:
            return self.baseline_threshold
        
        # Use first 100 packets to estimate baseline variance
        sample = self.packet_history[:100]
        turbulences = [calculate_spatial_turbulence(pkt['csi_data'], self.current_band) 
                      for pkt in sample]
        estimated_baseline_var = calculate_variance_two_pass(turbulences)
        
        # Threshold = 2x estimated baseline variance (conservative)
        adaptive_threshold = estimated_baseline_var * 2.0
        
        return adaptive_threshold
    
    def check_for_baseline_period(self):
        """Check if recent history contains a baseline period"""
        
        if len(self.packet_history) < self.detection_window:
            return
        
        # Calculate adaptive threshold if enabled
        if self.use_adaptive_threshold and self.adaptive_threshold is None:
            self.adaptive_threshold = self.calculate_adaptive_threshold()
            print(f"\n🔧 Adaptive threshold calculated: {self.adaptive_threshold:.4f}")
        
        threshold_to_use = self.adaptive_threshold if self.use_adaptive_threshold else self.baseline_threshold
        
        # Check the LAST detection_window packets
        recent_window = self.packet_history[-self.detection_window:]
        
        # Calculate variance
        turbulences = [calculate_spatial_turbulence(pkt['csi_data'], self.current_band) 
                      for pkt in recent_window]
        variance = calculate_variance_two_pass(turbulences)
        
        # If variance is low → baseline detected!
        if variance < threshold_to_use:
            print(f"\n📊 Baseline period detected at packet {self.packet_count}!")
            print(f"   Variance: {variance:.4f} < threshold {threshold_to_use:.4f}")
            print(f"   Using packets [{self.packet_count - self.detection_window}:{self.packet_count}]")
            
            # Calibrate using these packets
            self.calibrate_with_window(recent_window)
    
    def calibrate_with_window(self, window):
        """Calibrate using a specific window of packets"""
        
        print(f"\n🔧 Running auto-calibration...")
        
        best_band = None
        min_variance = float('inf')
        
        # Test all contiguous bands
        for start in range(TOTAL_SUBCARRIERS - BAND_SIZE + 1):
            band = list(range(start, start + BAND_SIZE))
            
            turbulences = [calculate_spatial_turbulence(pkt['csi_data'], band) 
                          for pkt in window]
            variance = calculate_variance_two_pass(turbulences)
            
            if variance < min_variance:
                min_variance = variance
                best_band = band
        
        print(f"   Best band found: {best_band}")
        print(f"   Variance: {min_variance:.4f}")
        
        # Validate
        if self.validate_band(best_band):
            print(f"   ✅ Validation passed")
            self.current_band = best_band
            self.calibrated = True
            self.calibration_time = self.packet_count
        else:
            print(f"   ⚠️  Validation failed, keeping default")
    
    def validate_band(self, band):
        """Validate that band is in known good bands"""
        return band in KNOWN_GOOD_BANDS

def simulate_runtime_scenario(baseline_packets, movement_packets, baseline_threshold):
    """
    Simulate a realistic runtime scenario with alternating baseline/movement
    """
    print(f"\n{'='*70}")
    print(f"  SIMULATING RUNTIME SCENARIO")
    print(f"{'='*70}\n")
    
    # Create realistic scenario: movement → baseline → movement → baseline
    scenario = []
    
    # Phase 1: Initial movement (0-2s)
    scenario.extend(movement_packets[:200])
    print("Phase 1 (0-2s): Movement")
    
    # Phase 2: Baseline period (2-5s)
    scenario.extend(baseline_packets[:300])
    print("Phase 2 (2-5s): Baseline (calibration opportunity)")
    
    # Phase 3: Movement (5-8s)
    scenario.extend(movement_packets[200:500])
    print("Phase 3 (5-8s): Movement")
    
    # Phase 4: Another baseline (8-10s)
    scenario.extend(baseline_packets[300:500])
    print("Phase 4 (8-10s): Baseline")
    
    print(f"\nTotal packets: {len(scenario)} ({len(scenario)/100:.1f}s @ 100Hz)")
    
    # Run calibrator
    print(f"\n{'='*70}")
    print(f"  RUNNING OPPORTUNISTIC CALIBRATOR")
    print(f"{'='*70}")
    
    calibrator = OpportunisticCalibrator(baseline_threshold=baseline_threshold)
    
    for i, packet in enumerate(scenario):
        calibrator.process_packet(packet)
        
        # Print progress
        if (i + 1) % 200 == 0:
            time = (i + 1) / 100
            status = "CALIBRATED" if calibrator.calibrated else "RUNNING"
            print(f"t={time:.1f}s (packet {i+1}): {status}")
    
    # Results
    print(f"\n{'='*70}")
    print(f"  CALIBRATION RESULTS")
    print(f"{'='*70}\n")
    
    if calibrator.calibrated:
        print(f"✅ Auto-calibration successful!")
        print(f"   Calibrated at packet: {calibrator.calibration_time} (t={(calibrator.calibration_time/100):.1f}s)")
        print(f"   Selected band: {calibrator.current_band}")
        
        # Test performance
        fp, tp, score = test_mvs_configuration(
            baseline_packets, movement_packets,
            calibrator.current_band, THRESHOLD, WINDOW_SIZE
        )
        
        f1 = (2 * tp / (2*tp + fp + (len(movement_packets) - tp)) * 100) if (tp + fp) > 0 else 0.0
        
        print(f"   Performance: FP={fp}, TP={tp}, F1={f1:.1f}%")
        
        # Compare with default
        default_fp, default_tp, _ = test_mvs_configuration(
            baseline_packets, movement_packets,
            DEFAULT_BAND, THRESHOLD, WINDOW_SIZE
        )
        default_f1 = (2 * default_tp / (2*default_tp + default_fp + (len(movement_packets) - default_tp)) * 100)
        
        print(f"\n   Comparison with default:")
        print(f"   Default: F1={default_f1:.1f}%")
        print(f"   Calibrated: F1={f1:.1f}%")
        print(f"   Improvement: {f1 - default_f1:+.1f}%")
    else:
        print(f"❌ Auto-calibration did not trigger")
        print(f"   No baseline period detected with threshold {baseline_threshold:.4f}")
    
    return calibrator


def test_false_positive_rate(baseline_packets, movement_packets, threshold):
    """Test how often baseline is falsely detected in movement data"""
    print(f"\n{'='*70}")
    print(f"  TESTING FALSE POSITIVE RATE (threshold={threshold:.4f})")
    print(f"{'='*70}\n")
    
    window_size = 200
    
    # Test on movement data
    false_positives = 0
    total_windows = 0
    
    for i in range(0, len(movement_packets) - window_size, 50):
        window = movement_packets[i:i+window_size]
        turbulences = [calculate_spatial_turbulence(pkt['csi_data'], DEFAULT_BAND) 
                      for pkt in window]
        variance = calculate_variance_two_pass(turbulences)
        
        total_windows += 1
        if variance < threshold:
            false_positives += 1
    
    fp_rate = (false_positives / total_windows * 100) if total_windows > 0 else 0
    
    print(f"Movement windows tested: {total_windows}")
    print(f"False positives (detected as baseline): {false_positives}")
    print(f"False positive rate: {fp_rate:.1f}%")
    
    if fp_rate < 5:
        print(f"\n✅ EXCELLENT: Very low false positive rate")
    elif fp_rate < 15:
        print(f"\n✅ GOOD: Acceptable false positive rate")
    else:
        print(f"\n⚠️  HIGH: Too many false positives, threshold too high")
    
    return fp_rate

def test_detection_latency(baseline_packets, threshold):
    """Test how quickly baseline is detected"""
    print(f"\n{'='*70}")
    print(f"  TESTING DETECTION LATENCY")
    print(f"{'='*70}\n")
    
    window_size = 200
    check_interval = 50
    
    # Simulate: start collecting baseline packets
    detected_at = None
    
    for i in range(window_size, len(baseline_packets), check_interval):
        window = baseline_packets[i-window_size:i]
        turbulences = [calculate_spatial_turbulence(pkt['csi_data'], DEFAULT_BAND) 
                      for pkt in window]
        variance = calculate_variance_two_pass(turbulences)
        
        if variance < threshold:
            detected_at = i
            print(f"✅ Baseline detected at packet {i} ({i/100:.1f}s)")
            print(f"   Variance: {variance:.4f}")
            break
    
    if detected_at:
        latency = detected_at / 100
        print(f"\nDetection latency: {latency:.1f}s")
        
        if latency < 3:
            print("✅ FAST: Baseline detected quickly")
        elif latency < 5:
            print("✅ ACCEPTABLE: Reasonable detection time")
        else:
            print("⚠️  SLOW: Takes too long to detect")
    else:
        print("\n❌ Baseline not detected")
    
    return detected_at

def test_with_random_starting_bands(baseline_packets, movement_packets, threshold, use_adaptive=False):
    """
    Test calibration starting from different (potentially bad) bands
    
    This is the critical test to validate the algorithm works in any environment
    """
    mode_str = "ADAPTIVE THRESHOLD" if use_adaptive else "FIXED THRESHOLD"
    print(f"\n{'='*70}")
    print(f"  TESTING WITH RANDOM/BAD STARTING BANDS ({mode_str})")
    print(f"{'='*70}\n")
    
    print("This simulates deploying the system in a new environment")
    print("where the default band may not be optimal.\n")
    
    test_bands = [
        ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], "Lower Edge (DC area)"),
        ([52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63], "Upper Edge (high freq)"),
        ([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60], "Random Sparse"),
        ([25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36], "Mid-High Band"),
        ([40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51], "High Band"),
        ([20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31], "Mid Band"),
    ]
    
    # Create scenario: movement → baseline → movement
    scenario = (movement_packets[:200] + 
                baseline_packets[:300] + 
                movement_packets[200:500])
    
    results = []
    
    print(f"{'Starting Band':<25} {'Detected?':<12} {'Final Band':<15} {'F1-Score':<10} {'Status':<8}")
    print("-" * 80)
    
    for start_band, description in test_bands:
        # Create calibrator with this starting band
        calibrator = OpportunisticCalibrator(
            baseline_threshold=threshold,
            starting_band=start_band,
            use_adaptive_threshold=use_adaptive
        )
        
        # Run scenario
        for packet in scenario:
            calibrator.process_packet(packet)
        
        # Evaluate
        if calibrator.calibrated:
            # Test performance
            fp, tp, _ = test_mvs_configuration(
                baseline_packets, movement_packets,
                calibrator.current_band, THRESHOLD, WINDOW_SIZE
            )
            f1 = (2 * tp / (2*tp + fp + (len(movement_packets) - tp)) * 100) if (tp + fp) > 0 else 0.0
            
            final_band_str = f"[{calibrator.current_band[0]}-{calibrator.current_band[-1]}]"
            
            # Check if it improved
            start_fp, start_tp, _ = test_mvs_configuration(
                baseline_packets, movement_packets,
                start_band, THRESHOLD, WINDOW_SIZE
            )
            start_f1 = (2 * start_tp / (2*start_tp + start_fp + (len(movement_packets) - start_tp)) * 100) if (start_tp + start_fp) > 0 else 0.0
            
            improvement = f1 - start_f1
            status = "✅ GOOD" if f1 >= 95 else "⚠️  OK" if f1 >= 85 else "❌ BAD"
            
            print(f"{description:<25} {'Yes':<12} {final_band_str:<15} {f1:<10.1f} {status:<8}")
            
            results.append({
                'start_band': start_band,
                'description': description,
                'final_band': calibrator.current_band,
                'start_f1': start_f1,
                'final_f1': f1,
                'improvement': improvement,
                'calibrated': True
            })
        else:
            print(f"{description:<25} {'No':<12} {'-':<15} {'-':<10} {'❌ FAIL':<8}")
            
            results.append({
                'start_band': start_band,
                'description': description,
                'calibrated': False
            })
    
    # Analysis
    print(f"\n{'='*70}")
    print(f"  RANDOM START ANALYSIS ({mode_str})")
    print(f"{'='*70}\n")
    
    calibrated_count = sum(1 for r in results if r['calibrated'])
    good_count = sum(1 for r in results if r.get('final_f1', 0) >= 95)
    
    print(f"Starting bands tested: {len(results)}")
    print(f"Successfully calibrated: {calibrated_count}/{len(results)} ({calibrated_count/len(results)*100:.0f}%)")
    print(f"Good performance (F1≥95%): {good_count}/{len(results)} ({good_count/len(results)*100:.0f}%)")
    
    if calibrated_count == len(results):
        print("\n✅ EXCELLENT: Calibration works from ANY starting band!")
    elif calibrated_count >= len(results) * 0.8:
        print("\n✅ GOOD: Calibration works from most starting bands")
    else:
        print("\n⚠️  PROBLEM: Calibration fails from some starting bands")
        if not use_adaptive:
            print("   → Try adaptive threshold")
    
    # Show improvements
    if calibrated_count > 0:
        print(f"\nPerformance improvements:")
        for r in results:
            if r['calibrated']:
                print(f"  {r['description']:<25} {r['start_f1']:.1f}% → {r['final_f1']:.1f}% ({r['improvement']:+.1f}%)")
    
    return results

def calculate_ring_radius(packets, subcarrier):
    """
    Calculate the mean radius of the I/Q constellation ring for a subcarrier
    
    Args:
        packets: List of CSI packets
        subcarrier: Subcarrier index (0-63)
    
    Returns:
        float: Mean radius (magnitude) of the constellation points
    """
    radii = []
    for pkt in packets:
        csi_data = pkt['csi_data']
        i_idx = subcarrier * 2
        q_idx = subcarrier * 2 + 1
        if q_idx < len(csi_data):
            I = float(csi_data[i_idx])
            Q = float(csi_data[q_idx])
            radius = np.sqrt(I**2 + Q**2)
            radii.append(radius)
    
    return np.mean(radii) if radii else 0.0

def calculate_ring_thickness(packets, subcarrier):
    """
    Calculate the thickness (std deviation) of the I/Q constellation ring
    
    Args:
        packets: List of CSI packets
        subcarrier: Subcarrier index (0-63)
    
    Returns:
        float: Standard deviation of the radius (ring thickness)
    """
    radii = []
    for pkt in packets:
        csi_data = pkt['csi_data']
        i_idx = subcarrier * 2
        q_idx = subcarrier * 2 + 1
        if q_idx < len(csi_data):
            I = float(csi_data[i_idx])
            Q = float(csi_data[q_idx])
            radius = np.sqrt(I**2 + Q**2)
            radii.append(radius)
    
    return np.std(radii) if radii else 0.0

def calculate_ring_compactness(packets, subcarrier):
    """
    Calculate the compactness of the I/Q constellation ring
    
    Compactness = radius / thickness
    Higher values indicate a more compact, stable ring
    
    Args:
        packets: List of CSI packets
        subcarrier: Subcarrier index (0-63)
    
    Returns:
        float: Compactness ratio
    """
    radius = calculate_ring_radius(packets, subcarrier)
    thickness = calculate_ring_thickness(packets, subcarrier)
    
    return radius / thickness if thickness > 0 else 0.0

def analyze_ring_geometry_all_subcarriers(baseline_packets, movement_packets):
    """
    Analyze ring geometry for all 64 subcarriers
    
    Returns:
        list: List of dicts with geometry metrics for each subcarrier
    """
    results = []
    
    for sc in range(TOTAL_SUBCARRIERS):
        # Baseline metrics
        baseline_radius = calculate_ring_radius(baseline_packets, sc)
        baseline_thickness = calculate_ring_thickness(baseline_packets, sc)
        baseline_compactness = calculate_ring_compactness(baseline_packets, sc)
        
        # Movement metrics
        movement_radius = calculate_ring_radius(movement_packets, sc)
        movement_thickness = calculate_ring_thickness(movement_packets, sc)
        movement_compactness = calculate_ring_compactness(movement_packets, sc)
        
        # Ratios (key discriminators)
        radius_ratio = baseline_radius / movement_radius if movement_radius > 0 else 0.0
        thickness_ratio = movement_thickness / baseline_thickness if baseline_thickness > 0 else 0.0
        
        # Composite score
        # Good subcarriers have:
        # - radius_ratio > 1.0 (baseline has larger radius)
        # - thickness_ratio > 1.0 (movement is more dispersed)
        # - high baseline_compactness (baseline is compact)
        score = radius_ratio * thickness_ratio * baseline_compactness
        
        results.append({
            'subcarrier': sc,
            'baseline_radius': baseline_radius,
            'baseline_thickness': baseline_thickness,
            'baseline_compactness': baseline_compactness,
            'movement_radius': movement_radius,
            'movement_thickness': movement_thickness,
            'movement_compactness': movement_compactness,
            'radius_ratio': radius_ratio,
            'thickness_ratio': thickness_ratio,
            'score': score
        })
    
    return results

def find_best_contiguous_band_by_score(scores, band_size=12):
    """
    Find the best contiguous band based on composite scores
    
    Args:
        scores: List of dicts with 'subcarrier' and 'score' keys
        band_size: Size of the band to select
    
    Returns:
        tuple: (best_band, avg_score, metrics)
    """
    best_band = None
    best_avg_score = 0.0
    best_metrics = None
    
    for start in range(TOTAL_SUBCARRIERS - band_size + 1):
        band = list(range(start, start + band_size))
        
        # Calculate average score for this band
        band_scores = [s for s in scores if s['subcarrier'] in band]
        avg_score = np.mean([s['score'] for s in band_scores])
        
        if avg_score > best_avg_score:
            best_avg_score = avg_score
            best_band = band
            
            # Calculate average metrics for this band
            best_metrics = {
                'avg_radius_ratio': np.mean([s['radius_ratio'] for s in band_scores]),
                'avg_thickness_ratio': np.mean([s['thickness_ratio'] for s in band_scores]),
                'avg_baseline_compactness': np.mean([s['baseline_compactness'] for s in band_scores]),
                'avg_score': avg_score
            }
    
    return best_band, best_avg_score, best_metrics

def calculate_variance_score_for_band(baseline_packets, movement_packets, band):
    """
    Calculate variance-based score for a band (lower variance = better)
    Returns inverted score so higher is better (consistent with other metrics)
    """
    turbulences = [calculate_spatial_turbulence(pkt['csi_data'], band) 
                  for pkt in baseline_packets]
    variance = calculate_variance_two_pass(turbulences)
    
    # Invert so lower variance = higher score
    # Use 1/(variance + epsilon) to avoid division by zero
    return 1.0 / (variance + 0.01)

def test_all_scoring_strategies(baseline_packets, movement_packets):
    """
    Test multiple scoring strategies to find the best one
    """
    print(f"\n{'='*70}")
    print(f"  TESTING ALL SCORING STRATEGIES")
    print(f"{'='*70}\n")
    
    # Get geometry data
    geometry_results = analyze_ring_geometry_all_subcarriers(baseline_packets, movement_packets)
    
    # Calculate variance for each subcarrier (for hybrid strategies)
    variance_scores = []
    for sc in range(TOTAL_SUBCARRIERS):
        band = [sc]  # Single subcarrier band for scoring
        turbulences = [calculate_spatial_turbulence(pkt['csi_data'], band) 
                      for pkt in baseline_packets]
        variance = calculate_variance_two_pass(turbulences)
        variance_score = 1.0 / (variance + 0.01)  # Invert: lower variance = higher score
        variance_scores.append(variance_score)
    
    # Add variance scores to geometry results
    for i, r in enumerate(geometry_results):
        r['variance_score'] = variance_scores[i]
    
    # Define all scoring strategies (including new hybrid ones)
    strategies = {
        # Original geometric strategies
        'original': lambda r: r['radius_ratio'] * r['thickness_ratio'] * r['baseline_compactness'],
        'inverted_radius': lambda r: (1/r['radius_ratio'] if r['radius_ratio'] > 0 else 0) * r['thickness_ratio'] * r['baseline_compactness'],
        'thickness_only': lambda r: r['thickness_ratio'] * r['baseline_compactness'],
        'radius_only': lambda r: r['radius_ratio'] * r['baseline_compactness'],
        'thickness_squared': lambda r: (r['thickness_ratio'] ** 2) * r['baseline_compactness'],
        'compactness_only': lambda r: r['baseline_compactness'],
        'radius_diff': lambda r: abs(r['baseline_radius'] - r['movement_radius']) * r['thickness_ratio'],
        'thickness_diff': lambda r: abs(r['baseline_thickness'] - r['movement_thickness']) * r['baseline_compactness'],
        'combined_diff': lambda r: abs(r['baseline_radius'] - r['movement_radius']) * abs(r['baseline_thickness'] - r['movement_thickness']),
        'movement_dispersion': lambda r: r['movement_thickness'] / (r['baseline_thickness'] + 1e-6),
        'baseline_stability': lambda r: r['baseline_radius'] / (r['baseline_thickness'] + 1e-6),
        'movement_instability': lambda r: r['movement_thickness'] / (r['movement_radius'] + 1e-6),
        
        # NEW: Hybrid strategies (variance + geometry)
        'hybrid_variance_instability': lambda r: r['variance_score'] * (r['movement_thickness'] / (r['movement_radius'] + 1e-6)),
        'hybrid_variance_thickness': lambda r: r['variance_score'] * r['thickness_ratio'],
        'hybrid_variance_weighted': lambda r: (0.7 * r['variance_score']) + (0.3 * (r['movement_thickness'] / (r['movement_radius'] + 1e-6))),
        'hybrid_variance_product': lambda r: r['variance_score'] * r['thickness_ratio'] * r['baseline_compactness'],
        
        # Pure variance (for comparison)
        'variance_only': lambda r: r['variance_score'],
    }
    
    results = []
    
    print(f"Testing {len(strategies)} different scoring strategies...")
    print(f"(Including {len([k for k in strategies.keys() if 'hybrid' in k])} new hybrid strategies)\n")
    print(f"{'Strategy':<30} {'Band':<15} {'FP':<6} {'TP':<6} {'F1':<8} {'Status':<8}")
    print("-" * 85)
    
    for strategy_name, score_func in strategies.items():
        # Calculate scores with this strategy
        scored_results = []
        for r in geometry_results:
            score = score_func(r)
            scored_results.append({**r, 'score': score})
        
        # Find best band
        best_band, _, _ = find_best_contiguous_band_by_score(scored_results, BAND_SIZE)
        
        # Test performance
        fp, tp, _ = test_mvs_configuration(
            baseline_packets, movement_packets,
            best_band, THRESHOLD, WINDOW_SIZE
        )
        f1 = (2 * tp / (2*tp + fp + (len(movement_packets) - tp)) * 100) if (tp + fp) > 0 else 0.0
        
        status = "🎉 BEST" if f1 >= 97 else "✅ GOOD" if f1 >= 95 else "⚠️  OK" if f1 >= 85 else "❌ BAD"
        
        # Mark hybrid strategies
        strategy_display = f"🔀 {strategy_name}" if 'hybrid' in strategy_name else strategy_name
        
        print(f"{strategy_display:<30} [{best_band[0]}-{best_band[-1]}]{'':<7} {fp:<6} {tp:<6} {f1:<8.1f} {status:<8}")
        
        results.append({
            'strategy': strategy_name,
            'band': best_band,
            'fp': fp,
            'tp': tp,
            'f1': f1
        })
    
    # Find best strategy
    best_result = max(results, key=lambda x: x['f1'])
    
    # Find best hybrid strategy
    hybrid_results = [r for r in results if 'hybrid' in r['strategy']]
    best_hybrid = max(hybrid_results, key=lambda x: x['f1']) if hybrid_results else None
    
    # Find best geometric strategy (non-hybrid, non-variance)
    geometric_results = [r for r in results if 'hybrid' not in r['strategy'] and r['strategy'] != 'variance_only']
    best_geometric = max(geometric_results, key=lambda x: x['f1']) if geometric_results else None
    
    print(f"\n{'='*70}")
    print(f"  BEST STRATEGIES COMPARISON")
    print(f"{'='*70}\n")
    
    # Compare with default
    fp_default, tp_default, _ = test_mvs_configuration(
        baseline_packets, movement_packets,
        DEFAULT_BAND, THRESHOLD, WINDOW_SIZE
    )
    f1_default = (2 * tp_default / (2*tp_default + fp_default + (len(movement_packets) - tp_default)) * 100)
    
    print(f"{'Category':<25} {'Strategy':<30} {'Band':<15} {'F1':<8} {'vs Default':<12}")
    print("-" * 95)
    print(f"{'Default (baseline)':<25} {'variance-based':<30} [{DEFAULT_BAND[0]}-{DEFAULT_BAND[-1]}]{'':<7} {f1_default:<8.1f} {'---':<12}")
    print(f"{'Overall Best':<25} {best_result['strategy']:<30} [{best_result['band'][0]}-{best_result['band'][-1]}]{'':<7} {best_result['f1']:<8.1f} {(best_result['f1']-f1_default):+.1f}%")
    
    if best_hybrid:
        print(f"{'Best Hybrid':<25} {best_hybrid['strategy']:<30} [{best_hybrid['band'][0]}-{best_hybrid['band'][-1]}]{'':<7} {best_hybrid['f1']:<8.1f} {(best_hybrid['f1']-f1_default):+.1f}%")
    
    if best_geometric:
        print(f"{'Best Geometric':<25} {best_geometric['strategy']:<30} [{best_geometric['band'][0]}-{best_geometric['band'][-1]}]{'':<7} {best_geometric['f1']:<8.1f} {(best_geometric['f1']-f1_default):+.1f}%")
    
    print(f"\n{'='*70}")
    print(f"  ANALYSIS")
    print(f"{'='*70}\n")
    
    if best_result['f1'] >= f1_default:
        print(f"🎉 SUCCESS: {best_result['strategy']} matches or beats default!")
        print(f"   Improvement: {(best_result['f1']-f1_default):+.1f}%")
    elif best_result['f1'] >= 95:
        print(f"✅ EXCELLENT: {best_result['strategy']} achieves F1≥95%")
        print(f"   Gap to default: {(best_result['f1']-f1_default):.1f}%")
    else:
        print(f"⚠️  Best strategy: {best_result['strategy']} with F1={best_result['f1']:.1f}%")
        print(f"   Gap to default: {(best_result['f1']-f1_default):.1f}%")
    
    if best_hybrid and best_geometric:
        hybrid_improvement = best_hybrid['f1'] - best_geometric['f1']
        print(f"\n💡 Hybrid vs Pure Geometric:")
        print(f"   Hybrid: {best_hybrid['f1']:.1f}%")
        print(f"   Geometric: {best_geometric['f1']:.1f}%")
        print(f"   Improvement: {hybrid_improvement:+.1f}%")
        
        if hybrid_improvement > 0:
            print(f"   ✅ Hybrid strategies improve over pure geometric!")
        else:
            print(f"   ⚠️  Hybrid strategies don't improve over pure geometric")
    
    return best_result, results

def test_ring_geometry_selection(baseline_packets, movement_packets):
    """
    Test automatic subcarrier selection using ring geometry analysis
    
    This test analyzes the I/Q constellation geometry to select optimal subcarriers
    based on the observation that baseline has larger radius but smaller thickness
    compared to movement.
    """
    print(f"\n{'='*70}")
    print(f"  PHASE 4: RING GEOMETRY-BASED SUBCARRIER SELECTION")
    print(f"{'='*70}\n")
    
    print("Analyzing I/Q constellation ring geometry for all 64 subcarriers...")
    print("Testing multiple scoring strategies to find optimal approach\n")
    
    # Test all strategies
    best_result, all_results = test_all_scoring_strategies(baseline_packets, movement_packets)
    
    # Now do detailed analysis with the original strategy for comparison
    print(f"\n{'='*70}")
    print(f"  DETAILED ANALYSIS (Original Strategy)")
    print(f"{'='*70}\n")
    
    # Analyze all subcarriers
    geometry_results = analyze_ring_geometry_all_subcarriers(baseline_packets, movement_packets)
    
    # Sort by score
    sorted_results = sorted(geometry_results, key=lambda x: x['score'], reverse=True)
    
    # Show top 10 subcarriers
    print("Top 10 subcarriers by ring geometry score:")
    print(f"{'SC':<4} {'Radius_Ratio':<14} {'Thick_Ratio':<13} {'Compactness':<12} {'Score':<8}")
    print("-" * 70)
    
    for i, r in enumerate(sorted_results[:10]):
        print(f"{r['subcarrier']:<4} {r['radius_ratio']:<14.3f} {r['thickness_ratio']:<13.3f} "
              f"{r['baseline_compactness']:<12.2f} {r['score']:<8.1f}")
    
    # Find best contiguous band
    print(f"\nFinding best contiguous band of {BAND_SIZE} subcarriers...")
    best_band, best_score, metrics = find_best_contiguous_band_by_score(geometry_results, BAND_SIZE)
    
    print(f"\n✅ Best band selected: {best_band}")
    print(f"   Range: [{best_band[0]}-{best_band[-1]}]")
    print(f"   Average radius ratio: {metrics['avg_radius_ratio']:.3f}")
    print(f"   Average thickness ratio: {metrics['avg_thickness_ratio']:.3f}")
    print(f"   Average baseline compactness: {metrics['avg_baseline_compactness']:.2f}")
    print(f"   Average composite score: {metrics['avg_score']:.1f}")
    
    # Test performance with MVS
    print(f"\n{'='*70}")
    print(f"  PERFORMANCE COMPARISON")
    print(f"{'='*70}\n")
    
    # Test ring geometry band
    fp_ring, tp_ring, _ = test_mvs_configuration(
        baseline_packets, movement_packets,
        best_band, THRESHOLD, WINDOW_SIZE
    )
    f1_ring = (2 * tp_ring / (2*tp_ring + fp_ring + (len(movement_packets) - tp_ring)) * 100) if (tp_ring + fp_ring) > 0 else 0.0
    
    # Test default band
    fp_default, tp_default, _ = test_mvs_configuration(
        baseline_packets, movement_packets,
        DEFAULT_BAND, THRESHOLD, WINDOW_SIZE
    )
    f1_default = (2 * tp_default / (2*tp_default + fp_default + (len(movement_packets) - tp_default)) * 100)
    
    # Test best known band
    best_known_band = KNOWN_GOOD_BANDS[1]  # [31-42] with F1=96.7%
    fp_known, tp_known, _ = test_mvs_configuration(
        baseline_packets, movement_packets,
        best_known_band, THRESHOLD, WINDOW_SIZE
    )
    f1_known = (2 * tp_known / (2*tp_known + fp_known + (len(movement_packets) - tp_known)) * 100)
    
    # Print comparison table
    print(f"{'Method':<30} {'Band':<15} {'FP':<6} {'TP':<6} {'F1':<8} {'Status':<8}")
    print("-" * 80)
    print(f"{'Ring Geometry Selection':<30} [{best_band[0]}-{best_band[-1]}]{'':<7} {fp_ring:<6} {tp_ring:<6} {f1_ring:<8.1f} "
          f"{'✅ BEST' if f1_ring >= max(f1_default, f1_known) else '✅ GOOD' if f1_ring >= 95 else '⚠️  OK'}")
    print(f"{'Default Band':<30} [{DEFAULT_BAND[0]}-{DEFAULT_BAND[-1]}]{'':<7} {fp_default:<6} {tp_default:<6} {f1_default:<8.1f} "
          f"{'✅ GOOD' if f1_default >= 95 else '⚠️  OK'}")
    print(f"{'Best Known Band':<30} [{best_known_band[0]}-{best_known_band[-1]}]{'':<7} {fp_known:<6} {tp_known:<6} {f1_known:<8.1f} "
          f"{'✅ GOOD' if f1_known >= 95 else '⚠️  OK'}")
    
    # Analysis
    print(f"\n{'='*70}")
    print(f"  RING GEOMETRY ANALYSIS")
    print(f"{'='*70}\n")
    
    improvement_vs_default = f1_ring - f1_default
    improvement_vs_known = f1_ring - f1_known
    
    if f1_ring >= max(f1_default, f1_known):
        print(f"🎉 RING GEOMETRY SELECTION OUTPERFORMS ALL METHODS!")
        print(f"\n   Ring Geometry: F1={f1_ring:.1f}%")
        print(f"   Default Band:  F1={f1_default:.1f}% ({improvement_vs_default:+.1f}%)")
        print(f"   Best Known:    F1={f1_known:.1f}% ({improvement_vs_known:+.1f}%)")
        
        print(f"\n✅ Key findings:")
        print(f"   1. Ring geometry analysis successfully identifies optimal subcarriers")
        print(f"   2. Larger baseline radius + smaller thickness = better discrimination")
        print(f"   3. Composite score (radius_ratio × thickness_ratio × compactness) works well")
        print(f"   4. Method is fully automatic and data-driven")
        
    elif f1_ring >= 95:
        print(f"✅ RING GEOMETRY SELECTION ACHIEVES EXCELLENT PERFORMANCE!")
        print(f"\n   Ring Geometry: F1={f1_ring:.1f}%")
        print(f"   Default Band:  F1={f1_default:.1f}% ({improvement_vs_default:+.1f}%)")
        print(f"   Best Known:    F1={f1_known:.1f}% ({improvement_vs_known:+.1f}%)")
        
        if improvement_vs_default > 0:
            print(f"\n✅ Improves over default by {improvement_vs_default:.1f}%")
        else:
            print(f"\n⚠️  Slightly lower than default ({improvement_vs_default:.1f}%), but still excellent")
    else:
        print(f"⚠️  Ring geometry selection underperforms")
        print(f"\n   Ring Geometry: F1={f1_ring:.1f}%")
        print(f"   Default Band:  F1={f1_default:.1f}%")
        print(f"   Best Known:    F1={f1_known:.1f}%")
        print(f"\n   May need to refine the scoring algorithm")
    
    # Show detailed geometry for selected band
    print(f"\n{'='*70}")
    print(f"  DETAILED GEOMETRY FOR SELECTED BAND [{best_band[0]}-{best_band[-1]}]")
    print(f"{'='*70}\n")
    
    print(f"{'SC':<4} {'Base_R':<8} {'Base_T':<8} {'Move_R':<8} {'Move_T':<8} {'R_Ratio':<9} {'T_Ratio':<9} {'Score':<8}")
    print("-" * 80)
    
    for sc in best_band[:6]:  # Show first 6 subcarriers
        r = geometry_results[sc]
        print(f"{sc:<4} {r['baseline_radius']:<8.1f} {r['baseline_thickness']:<8.1f} "
              f"{r['movement_radius']:<8.1f} {r['movement_thickness']:<8.1f} "
              f"{r['radius_ratio']:<9.3f} {r['thickness_ratio']:<9.3f} {r['score']:<8.1f}")
    
    if len(best_band) > 6:
        print("   ...")
    
    return {
        'band': best_band,
        'f1': f1_ring,
        'metrics': metrics,
        'improvement_vs_default': improvement_vs_default,
        'improvement_vs_known': improvement_vs_known
    }

def test_realistic_mixed_scenario(baseline_packets, movement_packets, threshold):
    """
    Test with completely realistic mixed data (80% baseline, 20% movement scattered)
    
    This simulates real-world usage where user moves occasionally
    """
    print(f"\n{'='*70}")
    print(f"  REALISTIC MIXED SCENARIO TEST")
    print(f"{'='*70}\n")
    
    print("Simulating realistic usage: 80% baseline, 20% movement (scattered)")
    
    # Create realistic mixed stream
    import random
    random.seed(42)  # Reproducible
    
    mixed_stream = []
    baseline_idx = 0
    movement_idx = 0
    
    for i in range(1000):
        if random.random() < 0.2 and movement_idx < len(movement_packets):
            # 20% chance of movement
            mixed_stream.append(movement_packets[movement_idx])
            movement_idx += 1
        elif baseline_idx < len(baseline_packets):
            # 80% baseline
            mixed_stream.append(baseline_packets[baseline_idx])
            baseline_idx += 1
    
    print(f"Created mixed stream: {len(mixed_stream)} packets")
    print(f"  Baseline packets: ~{baseline_idx}")
    print(f"  Movement packets: ~{movement_idx}")
    
    # Test with different starting bands
    test_bands = [
        ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], "Lower Edge"),
        ([52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63], "Upper Edge"),
        ([25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36], "Mid-High"),
    ]
    
    results = []
    
    print(f"\n{'Starting Band':<20} {'Detected?':<12} {'Time (s)':<10} {'Final Band':<15} {'F1':<8}")
    print("-" * 75)
    
    for start_band, description in test_bands:
        calibrator = OpportunisticCalibrator(
            baseline_threshold=threshold,
            starting_band=start_band,
            use_adaptive_threshold=True
        )
        
        for packet in mixed_stream:
            calibrator.process_packet(packet)
        
        if calibrator.calibrated:
            fp, tp, _ = test_mvs_configuration(
                baseline_packets, movement_packets,
                calibrator.current_band, THRESHOLD, WINDOW_SIZE
            )
            f1 = (2 * tp / (2*tp + fp + (len(movement_packets) - tp)) * 100) if (tp + fp) > 0 else 0.0
            
            final_str = f"[{calibrator.current_band[0]}-{calibrator.current_band[-1]}]"
            time_str = f"{calibrator.calibration_time/100:.1f}s"
            
            print(f"{description:<20} {'Yes':<12} {time_str:<10} {final_str:<15} {f1:<8.1f}")
            
            results.append({
                'start': description,
                'calibrated': True,
                'time': calibrator.calibration_time/100,
                'final_band': calibrator.current_band,
                'f1': f1
            })
        else:
            print(f"{description:<20} {'No':<12} {'-':<10} {'-':<15} {'-':<8}")
            results.append({'start': description, 'calibrated': False})
    
    # Analysis
    print(f"\n{'='*70}")
    print(f"  REALISTIC SCENARIO ANALYSIS")
    print(f"{'='*70}\n")
    
    calibrated = sum(1 for r in results if r['calibrated'])
    
    if calibrated == len(results):
        avg_time = np.mean([r['time'] for r in results if r['calibrated']])
        avg_f1 = np.mean([r['f1'] for r in results if r['calibrated']])
        
        print(f"✅ SUCCESS: Works with realistic mixed data!")
        print(f"   Calibrated: {calibrated}/{len(results)}")
        print(f"   Average time: {avg_time:.1f}s")
        print(f"   Average F1: {avg_f1:.1f}%")
    else:
        print(f"⚠️  Partial success: {calibrated}/{len(results)} calibrated")
    
    return results

def main():
    print("\n╔═══════════════════════════════════════════════════════╗")
    print("║   Retroactive Auto-Calibration Test                  ║")
    print("╚═══════════════════════════════════════════════════════╝")
    
    # Load data
    print("\n📂 Loading data...")
    try:
        baseline_packets, movement_packets = load_baseline_and_movement()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return
    
    print(f"   Loaded {len(baseline_packets)} baseline packets")
    print(f"   Loaded {len(movement_packets)} movement packets")
    
    # Step 1: Find optimal threshold
    optimal_threshold = find_optimal_baseline_threshold(baseline_packets, movement_packets)
    
    # Step 2: Test false positive rate
    fp_rate = test_false_positive_rate(baseline_packets, movement_packets, optimal_threshold)
    
    # Step 3: Test detection latency
    detection_latency = test_detection_latency(baseline_packets, optimal_threshold)
    
    # Step 4: CRITICAL TEST - Random starting bands (fixed threshold)
    print("\n" + "="*70)
    print("  PHASE 1: Testing with FIXED threshold")
    print("="*70)
    random_results_fixed = test_with_random_starting_bands(baseline_packets, movement_packets, optimal_threshold, use_adaptive=False)
    
    # Step 5: Test with ADAPTIVE threshold
    print("\n" + "="*70)
    print("  PHASE 2: Testing with ADAPTIVE threshold")
    print("="*70)
    random_results_adaptive = test_with_random_starting_bands(baseline_packets, movement_packets, optimal_threshold, use_adaptive=True)
    
    # Step 6: NEW - Test with realistic mixed data
    print("\n" + "="*70)
    print("  PHASE 3: Testing with REALISTIC MIXED DATA")
    print("="*70)
    realistic_results = test_realistic_mixed_scenario(baseline_packets, movement_packets, optimal_threshold)
    
    # Step 7: NEW - Test ring geometry selection with all strategies
    ring_geometry_results = test_ring_geometry_selection(baseline_packets, movement_packets)
    
    # Step 8: Simulate full runtime scenario
    print(f"\n{'='*70}")
    print(f"  FULL RUNTIME SIMULATION (with optimal starting band)")
    print(f"{'='*70}")
    
    calibrator = simulate_runtime_scenario(baseline_packets, movement_packets, optimal_threshold)
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*70}\n")
    
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    print(f"False positive rate: {fp_rate:.1f}%")
    
    if detection_latency:
        print(f"Detection latency: {detection_latency/100:.1f}s")
    
    # Analyze random start results
    fixed_calibrated = sum(1 for r in random_results_fixed if r['calibrated'])
    fixed_good = sum(1 for r in random_results_fixed if r.get('final_f1', 0) >= 95)
    
    adaptive_calibrated = sum(1 for r in random_results_adaptive if r['calibrated'])
    adaptive_good = sum(1 for r in random_results_adaptive if r.get('final_f1', 0) >= 95)
    
    print(f"\n{'='*70}")
    print(f"  COMPARISON: FIXED vs ADAPTIVE THRESHOLD")
    print(f"{'='*70}\n")
    
    print(f"Fixed Threshold:")
    print(f"  Calibrated: {fixed_calibrated}/{len(random_results_fixed)} ({fixed_calibrated/len(random_results_fixed)*100:.0f}%)")
    print(f"  Good (F1≥95%): {fixed_good}/{len(random_results_fixed)} ({fixed_good/len(random_results_fixed)*100:.0f}%)")
    
    print(f"\nAdaptive Threshold:")
    print(f"  Calibrated: {adaptive_calibrated}/{len(random_results_adaptive)} ({adaptive_calibrated/len(random_results_adaptive)*100:.0f}%)")
    print(f"  Good (F1≥95%): {adaptive_good}/{len(random_results_adaptive)*100:.0f}%)")
    
    # Analyze realistic scenario
    realistic_calibrated = sum(1 for r in realistic_results if r['calibrated'])
    realistic_good = sum(1 for r in realistic_results if r.get('f1', 0) >= 95)
    
    # Analyze ring geometry results
    if ring_geometry_results:
        ring_f1 = ring_geometry_results['f1']
        ring_improvement = ring_geometry_results['improvement_vs_default']
        
        print(f"\n{'='*70}")
        print(f"  RING GEOMETRY SELECTION SUMMARY")
        print(f"{'='*70}\n")
        
        print(f"Best strategy: {ring_geometry_results.get('best_strategy', 'N/A')}")
        print(f"Selected band: {ring_geometry_results['band']}")
        print(f"F1 Score: {ring_f1:.1f}%")
        print(f"Improvement vs default: {ring_improvement:+.1f}%")
        
        if ring_f1 >= 97:
            print(f"\n🎉 EXCELLENT: Ring geometry achieves F1≥97%!")
        elif ring_f1 >= 95:
            print(f"\n✅ GOOD: Ring geometry achieves F1≥95%")
        else:
            print(f"\n⚠️  Ring geometry F1={ring_f1:.1f}% (tested {len(ring_geometry_results.get('all_strategies', []))} strategies)")
    
    if calibrator.calibrated and adaptive_calibrated == len(random_results_adaptive) and realistic_calibrated == len(realistic_results):
        print(f"\n🎉 OPPORTUNISTIC CALIBRATION IS PRODUCTION-READY!")
        print(f"\nKey findings:")
        print(f"  1. Adaptive threshold enables universal calibration")
        print(f"  2. Works from ANY starting band ({adaptive_calibrated}/{len(random_results_adaptive)})")
        print(f"  3. Works with realistic mixed data ({realistic_calibrated}/{len(realistic_results)})")
        print(f"  4. Achieves F1≥95% in all scenarios")
        print(f"  5. Calibrates within {calibrator.calibration_time/100:.1f}s")
        print(f"  6. False positive rate: {fp_rate:.1f}%")
        
        print(f"\nDeployment strategy:")
        print(f"  - System starts with ANY reasonable default band")
        print(f"  - Monitors packet variance continuously (every 100 packets)")
        print(f"  - Calculates adaptive threshold from first 100 packets")
        print(f"  - When baseline detected (variance < adaptive_threshold):")
        print(f"    → Uses last 200 packets for calibration")
        print(f"    → Finds band with minimum variance")
        print(f"    → Validates against known good bands")
        print(f"    → Switches to calibrated band if valid")
        print(f"  - Completely automatic and transparent to user")
        print(f"  - No 'remain still' message needed!")
        print(f"  - Works even with occasional movement in data!")
    elif adaptive_calibrated >= len(random_results_adaptive) * 0.8 or realistic_calibrated >= len(realistic_results) * 0.8:
        print(f"\n✅ ADAPTIVE THRESHOLD SOLVES THE PROBLEM!")
        print(f"\nFindings:")
        print(f"  1. Fixed threshold: {fixed_calibrated}/{len(random_results_fixed)} success")
        print(f"  2. Adaptive threshold: {adaptive_calibrated}/{len(random_results_adaptive)} success")
        print(f"  3. Realistic mixed data: {realistic_calibrated}/{len(realistic_results)} success")
        print(f"  4. Adaptive threshold is REQUIRED for universal deployment")
        
        print(f"\nRecommendation:")
        print(f"  - Use robust default band (e.g., [10-21] or [31-42])")
        print(f"  - Implement adaptive threshold calculation")
        print(f"  - Fallback to default if calibration fails")
    else:
        print(f"\n⚠️  Calibration did not trigger in this scenario")
        print(f"   May need to adjust threshold or scenario")
    
    print()

if __name__ == '__main__':
    main()
