#!/usr/bin/env python3
"""
NBVI (Normalized Baseline Variability Index) Subcarrier Selection Test

Implements and validates the NBVI algorithm for automatic subcarrier selection
in CSI-based motion detection systems on ESP32-C6.

NBVI FORMULA (Optimized):
    NBVI_weighted = 0.3 × (σ/μ²) + 0.7 × (σ/μ)
    
    Where:
    - σ: Standard deviation of subcarrier magnitude in baseline
    - μ: Mean magnitude of subcarrier in baseline
    - 0.3: Energy normalization weight (penalizes weak subcarriers)
    - 0.7: Stability weight (rewards low variance)

Original formula (σ/μ²) was too aggressive - weighting is essential.

DETECTION MODES:
1. Percentile-based (RECOMMENDED): NO threshold configuration needed
   - Analyzes sliding windows over buffer (800-1000 packets)
   - Finds quietest windows using 10th percentile
   - Adapts automatically to any environment
   - Performance: F1=91.2% on mixed data (4/4 success)

2. Threshold-based: Uses adaptive threshold (like script 9)
   - Faster (2s vs 8s) but less robust
   - Performance: F1=86.9% on mixed data (3/3 success)
   - Requires threshold tuning

KEY FEATURES:
- Calibration-free (automatic baseline detection)
- Noise Gate (excludes weak subcarriers below 10th percentile)
- Spectral De-correlation (spacing Δf≥3 for diversity)
- Linear complexity O(N·L) - suitable for ESP32-C6
- Threshold-free (percentile mode) - zero configuration

PERFORMANCE RESULTS:

Phase 1 - Pure Data (Baseline + Movement Separate):
  Best: NBVI Weighted α=0.3 + Spacing
  Band: [6, 9, 10, 13, 21, 24, 27, 35, 36, 37, 40, 54]
  F1: 97.1% (gap to manual: -0.2%)
  
Phase 2 - Mixed Data Threshold-Based (80% baseline + 20% movement):
  Best: NBVI Weighted α=0.3
  F1: 86.9% (3/3 calibrated)
  vs Variance-only: ∞ (variance-only failed 0/3)
  
Phase 3 - Mixed Data Percentile-Based:
  Best: NBVI α=0.3 Percentile p10
  F1: 91.2% (4/4 calibrated)
  vs Threshold: +4.3%
  Average: 89.9% (+3.0% vs threshold)

COMPARISON WITH ALL TESTED STRATEGIES (23+ total):

Rank | Strategy              | Pure  | Mixed | Threshold? | Overall
-----|----------------------|-------|-------|------------|----------
🥇   | NBVI Percentile p10  | 97.1% | 91.2% | NO ✅      | WINNER
🥈   | NBVI Threshold       | 97.1% | 86.9% | Yes ⚠️     | Good
🥉   | Default (manual)     | 97.3% | N/A   | N/A        | Baseline
4    | Variance-only        | 92.4% | FAIL  | Yes ⚠️     | Fails
5-23 | Other strategies     | 63-92%| N/A   | Various    | Inferior

KEY FINDINGS:

1. NBVI Weighted α=0.3 Percentile p10 is PRODUCTION-READY
   - Pure data: F1=97.1% (gap -0.2% vs manual optimization)
   - Mixed data: F1=91.2% (best automatic method ever tested)
   - Success rate: 100% (4/4 on percentile, 3/3 on threshold)
   - Zero configuration: NO threshold needed

2. Percentile-Based Superiority
   - Finds better baseline windows (variance 0.26 vs 0.58)
   - +3.0% average improvement vs threshold
   - +4.3% best case improvement
   - Adapts to any environment automatically

3. Validation of Technical Report
   - NBVI concept: ✅ Validated (with α=0.3 weighting)
   - Noise Gate: ✅ Critical (excludes 7 weak subcarriers)
   - Spacing Δf≥3: ✅ Essential for diversity
   - Calibration-free: ✅ Works automatically
   - Percentile approach: ✅ Superior to threshold

4. Trade-offs
   - Latency: 8s (percentile) vs 2s (threshold)
   - Buffer: 1000 packets vs 200
   - Worth it: +4.3% performance for 6s extra

RECOMMENDED DEPLOYMENT CONFIGURATION:

```c
nbvi_config_t config = {
    .use_percentile = true,          // ← RECOMMENDED
    .percentile = 10,                // 10th percentile
    .analysis_buffer_size = 1000,    // 10s @ 100Hz
    .alpha = 0.3f,                   // NBVI weighting
    .min_spacing = 3,                // Spectral de-correlation
    .noise_gate_percentile = 10      // Exclude weak subcarriers
};
```

CONCLUSION:
NBVI Weighted α=0.3 with Percentile-based detection (p10) is the BEST
automatic subcarrier selection method tested, achieving near-optimal
performance (97.1% pure, 91.2% mixed) with ZERO configuration required.

Usage:
    python tools/11_test_nbvi_selection.py

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import numpy as np
from mvs_utils import load_baseline_and_movement, test_mvs_configuration, calculate_spatial_turbulence
from config import WINDOW_SIZE, THRESHOLD

BAND_SIZE = 12
TOTAL_SUBCARRIERS = 64
DEFAULT_BAND = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

# Test Configuration
ESP32_BUFFER_SIZE = 500      # Packets to collect for calibration - 1000 on ESP32-C6 is too large
ESP32_WINDOW_SIZE = 100      # Window size for baseline detection - 200 on ESP32-C6 is too large
ESP32_WINDOW_STEP = 50       # Step size for sliding window analysis

def calculate_magnitude(csi_data, subcarrier):
    """
    Calculate magnitude |H| from I/Q components
    
    Args:
        csi_data: CSI data array
        subcarrier: Subcarrier index (0-63)
    
    Returns:
        float: Magnitude |H| = sqrt(I² + Q²)
    """
    i_idx = subcarrier * 2
    q_idx = subcarrier * 2 + 1
    
    if q_idx < len(csi_data):
        I = float(csi_data[i_idx])
        Q = float(csi_data[q_idx])
        return np.sqrt(I**2 + Q**2)
    return 0.0

def extract_magnitude_series(packets, subcarrier):
    """
    Extract magnitude time series for a subcarrier
    
    Args:
        packets: List of CSI packets
        subcarrier: Subcarrier index
    
    Returns:
        np.array: Array of magnitudes
    """
    magnitudes = []
    for pkt in packets:
        mag = calculate_magnitude(pkt['csi_data'], subcarrier)
        magnitudes.append(mag)
    return np.array(magnitudes)

def calculate_nbvi(packets, subcarrier):
    """
    Calculate NBVI (Normalized Baseline Variability Index)
    
    NBVI = σ / μ²
    
    Lower NBVI = Better subcarrier (stable + strong)
    
    Args:
        packets: Baseline packets
        subcarrier: Subcarrier index
    
    Returns:
        dict: {
            'nbvi': NBVI value,
            'mean': μ,
            'std': σ,
            'cv': Coefficient of Variation (σ/μ)
        }
    """
    magnitudes = extract_magnitude_series(packets, subcarrier)
    
    mean = np.mean(magnitudes)
    std = np.std(magnitudes)
    
    # Avoid division by zero
    if mean < 1e-6:
        return {
            'nbvi': float('inf'),
            'mean': mean,
            'std': std,
            'cv': float('inf')
        }
    
    cv = std / mean
    nbvi = std / (mean ** 2)
    
    return {
        'nbvi': nbvi,
        'mean': mean,
        'std': std,
        'cv': cv
    }

def apply_noise_gate(subcarrier_metrics, percentile=10):
    """
    Apply Noise Gate: exclude subcarriers with weak signal
    
    Excludes subcarriers with mean magnitude below the specified percentile
    
    Args:
        subcarrier_metrics: List of dicts with 'subcarrier' and 'mean' keys
        percentile: Percentile threshold (default: 10)
    
    Returns:
        list: Filtered subcarrier metrics
    """
    means = [m['mean'] for m in subcarrier_metrics]
    threshold = np.percentile(means, percentile)
    
    filtered = [m for m in subcarrier_metrics if m['mean'] >= threshold]
    
    print(f"Noise Gate: {len(subcarrier_metrics) - len(filtered)} subcarriers excluded")
    print(f"  Threshold: {threshold:.2f} (mean magnitude at {percentile}th percentile)")
    
    return filtered

def select_with_spectral_spacing(sorted_metrics, k=12, min_spacing=3):
    """
    Select subcarriers with spectral de-correlation strategy
    
    Strategy:
    1. Top 5: Always include (highest priority)
    2. Remaining 7: Select with minimum spacing Δf ≥ min_spacing
    
    Args:
        sorted_metrics: Subcarriers sorted by NBVI (ascending)
        k: Number of subcarriers to select (default: 12)
        min_spacing: Minimum spacing between subcarriers (default: 3)
    
    Returns:
        list: Selected subcarrier indices
    """
    # Phase 1: Top 5 absolute best
    top_5 = [m['subcarrier'] for m in sorted_metrics[:5]]
    selected = top_5.copy()
    
    print(f"\nSpectral De-correlation Strategy:")
    print(f"  Top 5 (absolute priority): {top_5}")
    
    # Phase 2: Remaining 7 with spacing
    remaining_needed = k - 5
    candidates = sorted_metrics[5:]
    
    for candidate in candidates:
        if len(selected) >= k:
            break
        
        sc = candidate['subcarrier']
        
        # Check spacing with already selected
        min_dist = min([abs(sc - s) for s in selected])
        
        if min_dist >= min_spacing:
            selected.append(sc)
        elif len(selected) < k and candidate == candidates[-1]:
            # Last resort: if we can't fill k with spacing, add anyway
            selected.append(sc)
    
    # If still not enough, add best remaining regardless of spacing
    if len(selected) < k:
        for candidate in candidates:
            if len(selected) >= k:
                break
            sc = candidate['subcarrier']
            if sc not in selected:
                selected.append(sc)
    
    selected.sort()
    print(f"  Remaining 7 (with spacing Δf≥{min_spacing}): {selected[5:]}")
    
    return selected

def calculate_nbvi_variants(packets, subcarrier):
    """
    Calculate all NBVI variants for comparison
    
    Returns:
        dict: All variant scores
    """
    magnitudes = extract_magnitude_series(packets, subcarrier)
    
    mean = np.mean(magnitudes)
    std = np.std(magnitudes)
    
    if mean < 1e-6:
        return {
            'nbvi_original': float('inf'),
            'nbvi_1_5': float('inf'),
            'nbvi_weighted_03': float('inf'),
            'nbvi_weighted_05': float('inf'),
            'cv': float('inf'),
            'mean': mean,
            'std': std
        }
    
    cv = std / mean
    
    return {
        'nbvi_original': std / (mean ** 2),      # σ/μ²
        'nbvi_1_5': std / (mean ** 1.5),         # σ/μ^1.5
        'nbvi_weighted_03': 0.3 * (std / (mean ** 2)) + 0.7 * cv,  # Weighted α=0.3
        'nbvi_weighted_05': 0.5 * (std / (mean ** 2)) + 0.5 * cv,  # Weighted α=0.5
        'cv': cv,
        'mean': mean,
        'std': std
    }

class OpportunisticNBVICalibrator:
    """
    Automatic calibrator using NBVI for subcarrier selection
    
    Supports two modes:
    1. Threshold-based: Uses adaptive threshold (like script 9)
    2. Percentile-based: NO threshold - finds quietest windows automatically
    """
    
    def __init__(self, variant='weighted_03', use_percentile=True, percentile=10, 
                 baseline_threshold=0.5, starting_band=None, analysis_buffer_size=ESP32_BUFFER_SIZE):
        self.variant = variant
        self.use_percentile = use_percentile
        self.percentile_threshold = percentile
        self.analysis_buffer_size = analysis_buffer_size  # For percentile analysis
        self.packet_history = []
        self.current_band = starting_band if starting_band else DEFAULT_BAND
        self.starting_band = self.current_band.copy()
        self.calibrated = False
        self.baseline_threshold = baseline_threshold
        self.adaptive_threshold = None
        self.check_interval = 100
        self.detection_window = ESP32_WINDOW_SIZE 
        self.packet_count = 0
        self.calibration_time = None
        
    def process_packet(self, packet):
        """Process incoming packet"""
        self.packet_count += 1
        
        # Add to history (keep enough for percentile analysis)
        max_history = max(self.analysis_buffer_size + 500, 1500) if self.use_percentile else 500
        self.packet_history.append(packet)
        if len(self.packet_history) > max_history:
            self.packet_history.pop(0)
        
        # Check for calibration opportunity periodically
        if not self.calibrated and self.packet_count % self.check_interval == 0:
            self.check_for_baseline_period()
    
    def calculate_adaptive_threshold(self):
        """Calculate adaptive threshold based on current band's baseline variance"""
        if len(self.packet_history) < 100:
            return self.baseline_threshold
        
        sample = self.packet_history[:100]
        turbulences = [calculate_spatial_turbulence(pkt['csi_data'], self.current_band) 
                      for pkt in sample]
        from mvs_utils import calculate_variance_two_pass
        estimated_baseline_var = calculate_variance_two_pass(turbulences)
        
        return estimated_baseline_var * 2.0
    
    def check_for_baseline_period(self):
        """Check if recent history contains a baseline period"""
        if self.use_percentile:
            return self.check_for_baseline_period_percentile()
        else:
            return self.check_for_baseline_period_threshold()
    
    def check_for_baseline_period_threshold(self):
        """Original threshold-based approach"""
        if len(self.packet_history) < self.detection_window:
            return
        
        # Calculate adaptive threshold if not set
        if self.adaptive_threshold is None:
            self.adaptive_threshold = self.calculate_adaptive_threshold()
        
        # Check the LAST detection_window packets
        recent_window = self.packet_history[-self.detection_window:]
        
        # Calculate variance
        turbulences = [calculate_spatial_turbulence(pkt['csi_data'], self.current_band) 
                      for pkt in recent_window]
        from mvs_utils import calculate_variance_two_pass
        variance = calculate_variance_two_pass(turbulences)
        
        # If variance is low → baseline detected!
        if variance < self.adaptive_threshold:
            print(f"\n📊 Baseline detected (threshold) at packet {self.packet_count}")
            print(f"   Variance: {variance:.4f} < threshold {self.adaptive_threshold:.4f}")
            self.calibrate_with_nbvi(recent_window)
    
    def check_for_baseline_period_percentile(self):
        """
        Percentile-based approach: NO absolute threshold!
        
        Finds the quietest window in the last N packets using percentile analysis.
        This adapts automatically to any environment.
        """
        # Need enough data for percentile analysis
        if len(self.packet_history) < self.analysis_buffer_size:
            return
        
        # Analyze last analysis_buffer_size packets
        buffer = self.packet_history[-self.analysis_buffer_size:]
        
        # Calculate variance for sliding windows
        window_variances = []
        from mvs_utils import calculate_variance_two_pass
        
        for i in range(0, len(buffer) - self.detection_window, ESP32_WINDOW_STEP):
            window = buffer[i:i+self.detection_window]
            turbulences = [calculate_spatial_turbulence(pkt['csi_data'], self.current_band) 
                          for pkt in window]
            variance = calculate_variance_two_pass(turbulences)
            
            window_variances.append({
                'start': i,
                'variance': variance,
                'window': window
            })
        
        if not window_variances:
            return
        
        # Calculate percentile threshold (adaptive!)
        variances = [w['variance'] for w in window_variances]
        p_threshold = np.percentile(variances, self.percentile_threshold)
        
        # Find windows below percentile
        baseline_candidates = [w for w in window_variances if w['variance'] <= p_threshold]
        
        if baseline_candidates:
            # Use window with minimum variance
            best_window = min(baseline_candidates, key=lambda x: x['variance'])
            
            print(f"\n📊 Baseline detected (percentile) at packet {self.packet_count}")
            print(f"   Variance: {best_window['variance']:.4f}")
            print(f"   p{self.percentile_threshold} threshold: {p_threshold:.4f} (adaptive)")
            print(f"   Windows analyzed: {len(window_variances)}")
            print(f"   Baseline candidates: {len(baseline_candidates)}")
            
            self.calibrate_with_nbvi(best_window['window'])
    
    def calibrate_with_nbvi(self, window):
        """Calibrate using NBVI on the detected baseline window"""
        print(f"🔧 Running NBVI calibration (variant: {self.variant})...")
        
        # Calculate NBVI for all subcarriers
        all_metrics = []
        for sc in range(TOTAL_SUBCARRIERS):
            metrics = calculate_nbvi_variants(window, sc)
            metrics['subcarrier'] = sc
            all_metrics.append(metrics)
        
        # Apply Noise Gate
        means = [m['mean'] for m in all_metrics]
        threshold = np.percentile(means, 10)
        filtered_metrics = [m for m in all_metrics if m['mean'] >= threshold]
        
        # Sort by selected variant
        variant_key_map = {
            'original': 'nbvi_original',
            'weighted_03': 'nbvi_weighted_03',
            'weighted_05': 'nbvi_weighted_05',
            'nbvi_1_5': 'nbvi_1_5'
        }
        sort_key = variant_key_map.get(self.variant, 'nbvi_weighted_03')
        sorted_metrics = sorted(filtered_metrics, key=lambda x: x[sort_key])
        
        # Select with spacing
        selected_band = select_with_spectral_spacing(sorted_metrics, k=BAND_SIZE, min_spacing=3)
        
        print(f"   Selected band: {selected_band}")
        
        self.current_band = selected_band
        self.calibrated = True
        self.calibration_time = self.packet_count

def test_nbvi_percentile_vs_threshold(baseline_packets, movement_packets):
    """
    Compare percentile-based vs threshold-based NBVI calibration
    
    Tests both approaches on the same mixed data stream to evaluate
    which is more robust and reliable.
    """
    print(f"\n{'='*70}")
    print(f"  PERCENTILE vs THRESHOLD COMPARISON")
    print(f"{'='*70}\n")
    
    print("Testing both approaches on realistic mixed data")
    print("(80% baseline + 20% movement scattered)\n")
    
    # Create mixed stream
    import random
    random.seed(42)
    
    mixed_stream = []
    baseline_idx = 0
    movement_idx = 0
    
    for i in range(1000):
        if random.random() < 0.2 and movement_idx < len(movement_packets):
            mixed_stream.append(movement_packets[movement_idx])
            movement_idx += 1
        elif baseline_idx < len(baseline_packets):
            mixed_stream.append(baseline_packets[baseline_idx])
            baseline_idx += 1
    
    print(f"Mixed stream: {len(mixed_stream)} packets")
    print(f"  Baseline: ~{baseline_idx}, Movement: ~{movement_idx}")
    
    # Test configurations
    test_configs = [
        ('weighted_03', True, 10, 'NBVI α=0.3 Percentile p10'),
        ('weighted_03', True, 20, 'NBVI α=0.3 Percentile p20'),
        ('weighted_03', False, None, 'NBVI α=0.3 Threshold'),
        ('weighted_05', True, 10, 'NBVI α=0.5 Percentile p10'),
        ('nbvi_1_5', True, 10, 'NBVI^1.5 Percentile p10'),
    ]
    
    results = []
    
    print(f"\n{'Method':<30} {'Calibrated?':<12} {'Time':<8} {'Band':<20} {'F1':<8}")
    print("-" * 90)
    
    for variant, use_perc, perc_val, name in test_configs:
        calibrator = OpportunisticNBVICalibrator(
            variant=variant,
            use_percentile=use_perc,
            percentile=perc_val if perc_val else 10,
            baseline_threshold=0.5,
            analysis_buffer_size=ESP32_BUFFER_SIZE  # ESP32-C6 memory constraint
        )
        
        for packet in mixed_stream:
            calibrator.process_packet(packet)
        
        if calibrator.calibrated:
            fp, tp, _ = test_mvs_configuration(
                baseline_packets, movement_packets,
                calibrator.current_band, THRESHOLD, WINDOW_SIZE
            )
            f1 = (2 * tp / (2*tp + fp + (len(movement_packets) - tp)) * 100) if (tp + fp) > 0 else 0.0
            
            band_str = f"[{calibrator.current_band[0]}...{calibrator.current_band[-1]}]"
            time_str = f"{calibrator.calibration_time/100:.1f}s"
            
            print(f"{name:<30} {'Yes':<12} {time_str:<8} {band_str:<20} {f1:<8.1f}")
            
            results.append({
                'method': name,
                'calibrated': True,
                'time': calibrator.calibration_time/100,
                'band': calibrator.current_band,
                'f1': f1,
                'use_percentile': use_perc
            })
        else:
            print(f"{name:<30} {'No':<12} {'-':<8} {'-':<20} {'-':<8}")
            results.append({'method': name, 'calibrated': False})
    
    # Analysis
    print(f"\n{'='*70}")
    print(f"  PERCENTILE vs THRESHOLD ANALYSIS")
    print(f"{'='*70}\n")
    
    percentile_results = [r for r in results if r.get('use_percentile') and r['calibrated']]
    threshold_results = [r for r in results if not r.get('use_percentile') and r['calibrated']]
    
    if percentile_results and threshold_results:
        avg_f1_perc = np.mean([r['f1'] for r in percentile_results])
        avg_f1_thresh = np.mean([r['f1'] for r in threshold_results])
        
        print(f"Percentile-based: {len(percentile_results)}/{len([r for r in results if r.get('use_percentile')])} calibrated")
        print(f"  Average F1: {avg_f1_perc:.1f}%")
        
        print(f"\nThreshold-based: {len(threshold_results)}/{len([r for r in results if not r.get('use_percentile')])} calibrated")
        print(f"  Average F1: {avg_f1_thresh:.1f}%")
        
        improvement = avg_f1_perc - avg_f1_thresh
        print(f"\nPercentile improvement: {improvement:+.1f}%")
        
        if improvement > 0:
            print(f"✅ Percentile-based is MORE robust!")
        elif improvement > -1:
            print(f"✅ Both approaches perform similarly")
        else:
            print(f"⚠️  Threshold-based performs better")
    
    elif percentile_results:
        print(f"✅ Percentile-based: {len(percentile_results)} calibrated")
        print(f"❌ Threshold-based: 0 calibrated")
        print(f"\n🎉 Percentile-based is SIGNIFICANTLY more robust!")
    
    return results

def test_nbvi_realistic_mixed_scenario(baseline_packets, movement_packets, baseline_threshold=0.5):
    """
    Test NBVI calibration on realistic mixed data (80% baseline, 20% movement)
    
    Replicates test_realistic_mixed_scenario from script 9, but uses NBVI
    for subcarrier selection instead of variance.
    """
    print(f"\n{'='*70}")
    print(f"  NBVI ON REALISTIC MIXED SCENARIO (Threshold-Based)")
    print(f"{'='*70}\n")
    
    print("Simulating realistic usage: 80% baseline, 20% movement (scattered)")
    
    # Create realistic mixed stream (same as script 9)
    import random
    random.seed(42)
    
    mixed_stream = []
    baseline_idx = 0
    movement_idx = 0
    
    for i in range(1000):
        if random.random() < 0.2 and movement_idx < len(movement_packets):
            mixed_stream.append(movement_packets[movement_idx])
            movement_idx += 1
        elif baseline_idx < len(baseline_packets):
            mixed_stream.append(baseline_packets[baseline_idx])
            baseline_idx += 1
    
    print(f"Created mixed stream: {len(mixed_stream)} packets")
    print(f"  Baseline packets: ~{baseline_idx}")
    print(f"  Movement packets: ~{movement_idx}")
    
    # Test different NBVI variants
    test_variants = [
        ('weighted_03', 'NBVI Weighted α=0.3'),
        ('weighted_05', 'NBVI Weighted α=0.5'),
        ('nbvi_1_5', 'NBVI^1.5'),
    ]
    
    results = []
    
    print(f"\n{'Variant':<25} {'Calibrated?':<12} {'Time (s)':<10} {'Band':<20} {'F1':<8}")
    print("-" * 85)
    
    for variant_id, variant_name in test_variants:
        calibrator = OpportunisticNBVICalibrator(
            variant=variant_id,
            baseline_threshold=baseline_threshold
        )
        
        for packet in mixed_stream:
            calibrator.process_packet(packet)
        
        if calibrator.calibrated:
            # Test performance
            fp, tp, _ = test_mvs_configuration(
                baseline_packets, movement_packets,
                calibrator.current_band, THRESHOLD, WINDOW_SIZE
            )
            f1 = (2 * tp / (2*tp + fp + (len(movement_packets) - tp)) * 100) if (tp + fp) > 0 else 0.0
            
            band_str = f"[{calibrator.current_band[0]}...{calibrator.current_band[-1]}]"
            time_str = f"{calibrator.calibration_time/100:.1f}s"
            
            print(f"{variant_name:<25} {'Yes':<12} {time_str:<10} {band_str:<20} {f1:<8.1f}")
            
            results.append({
                'variant': variant_name,
                'calibrated': True,
                'time': calibrator.calibration_time/100,
                'band': calibrator.current_band,
                'f1': f1
            })
        else:
            print(f"{variant_name:<25} {'No':<12} {'-':<10} {'-':<20} {'-':<8}")
            results.append({'variant': variant_name, 'calibrated': False})
    
    # Add baseline comparison (variance-only from script 9)
    print(f"{'─'*85}")
    print(f"{'Variance-only (script 9)':<25} {'No*':<12} {'-':<10} {'-':<20} {'-':<8}")
    print(f"  * Script 9 showed 0/3 success on mixed data")
    
    # Analysis
    print(f"\n{'='*70}")
    print(f"  MIXED SCENARIO ANALYSIS")
    print(f"{'='*70}\n")
    
    calibrated_count = sum(1 for r in results if r['calibrated'])
    
    if calibrated_count == len(results):
        avg_time = np.mean([r['time'] for r in results if r['calibrated']])
        avg_f1 = np.mean([r['f1'] for r in results if r['calibrated']])
        
        print(f"🎉 SUCCESS: NBVI works with realistic mixed data!")
        print(f"   Calibrated: {calibrated_count}/{len(results)} (100%)")
        print(f"   Average time: {avg_time:.1f}s")
        print(f"   Average F1: {avg_f1:.1f}%")
        print(f"\n   ✅ NBVI outperforms variance-only (0/3) on mixed data!")
    elif calibrated_count > 0:
        print(f"⚠️  Partial success: {calibrated_count}/{len(results)} calibrated")
        
        for r in results:
            if r['calibrated']:
                print(f"   ✅ {r['variant']}: F1={r['f1']:.1f}%")
    else:
        print(f"❌ NBVI calibration failed on mixed data")
        print(f"   Same issue as variance-only (adaptive threshold too permissive)")
    
    return results

def test_nbvi_selection(baseline_packets, movement_packets):
    """
    Test NBVI-based subcarrier selection with all variants
    
    Returns:
        dict: Results including selected band and performance metrics
    """
    print(f"\n{'='*70}")
    print(f"  NBVI VARIANTS COMPARISON TEST")
    print(f"{'='*70}\n")
    
    print(f"Testing 6 NBVI variants:")
    print(f"  1. NBVI Original (σ/μ²)")
    print(f"  2. NBVI^1.5 (σ/μ^1.5)")
    print(f"  3. NBVI Weighted α=0.3")
    print(f"  4. NBVI Weighted α=0.5")
    print(f"  5. NBVI No Spacing (top 12 contiguous)")
    print(f"  6. NBVI Centered (12 contiguous around best)")
    
    print(f"\nAnalyzing {TOTAL_SUBCARRIERS} subcarriers...")
    print(f"Baseline packets: {len(baseline_packets)}")
    
    # Step 1: Calculate all NBVI variants for all subcarriers
    all_metrics = []
    
    for sc in range(TOTAL_SUBCARRIERS):
        metrics = calculate_nbvi_variants(baseline_packets, sc)
        metrics['subcarrier'] = sc
        all_metrics.append(metrics)
    
    # Step 2: Apply Noise Gate (exclude weak subcarriers)
    print(f"\nNoise Gate (exclude weak subcarriers)")
    filtered_metrics = apply_noise_gate(all_metrics, percentile=10)
    
    # Step 3: Test all variants
    variants = {
        'nbvi_original_spacing': {
            'name': 'NBVI Original (σ/μ²) + Spacing',
            'sort_key': 'nbvi_original',
            'use_spacing': True
        },
        'nbvi_1_5_spacing': {
            'name': 'NBVI^1.5 (σ/μ^1.5) + Spacing',
            'sort_key': 'nbvi_1_5',
            'use_spacing': True
        },
        'nbvi_weighted_03': {
            'name': 'NBVI Weighted α=0.3 + Spacing',
            'sort_key': 'nbvi_weighted_03',
            'use_spacing': True
        },
        'nbvi_weighted_05': {
            'name': 'NBVI Weighted α=0.5 + Spacing',
            'sort_key': 'nbvi_weighted_05',
            'use_spacing': True
        },
        'nbvi_original_no_spacing': {
            'name': 'NBVI Original (σ/μ²) No Spacing',
            'sort_key': 'nbvi_original',
            'use_spacing': False
        },
        'nbvi_original_centered': {
            'name': 'NBVI Original (σ/μ²) Centered',
            'sort_key': 'nbvi_original',
            'use_spacing': 'centered'
        }
    }
    
    results = []
    
    print(f"\n{'='*70}")
    print(f"  TESTING ALL VARIANTS")
    print(f"{'='*70}\n")
    
    print(f"{'Variant':<40} {'Band':<25} {'FP':<6} {'TP':<6} {'F1':<8}")
    print("-" * 95)
    
    for variant_id, variant_config in variants.items():
        # Sort by this variant's key
        sorted_metrics = sorted(filtered_metrics, key=lambda x: x[variant_config['sort_key']])
        
        # Select band based on strategy
        if variant_config['use_spacing'] == True:
            selected_band = select_with_spectral_spacing(sorted_metrics, k=BAND_SIZE, min_spacing=3)
        elif variant_config['use_spacing'] == 'centered':
            # Centered strategy: find best, then select 12 contiguous around it
            best_sc = sorted_metrics[0]['subcarrier']
            start = max(0, best_sc - 5)
            end = min(TOTAL_SUBCARRIERS, start + BAND_SIZE)
            if end - start < BAND_SIZE:
                start = max(0, end - BAND_SIZE)
            selected_band = list(range(start, end))
        else:  # No spacing
            selected_band = sorted([m['subcarrier'] for m in sorted_metrics[:BAND_SIZE]])
        
        # Test performance
        fp, tp, _ = test_mvs_configuration(
            baseline_packets, movement_packets,
            selected_band, THRESHOLD, WINDOW_SIZE
        )
        f1 = (2 * tp / (2*tp + fp + (len(movement_packets) - tp)) * 100) if (tp + fp) > 0 else 0.0
        
        band_str = f"[{selected_band[0]}-{selected_band[-1]}]" if len(selected_band) > 3 else str(selected_band)
        
        print(f"{variant_config['name']:<40} {band_str:<25} {fp:<6} {tp:<6} {f1:<8.1f}")
        
        results.append({
            'variant': variant_id,
            'name': variant_config['name'],
            'band': selected_band,
            'fp': fp,
            'tp': tp,
            'f1': f1
        })
    
    
    # Add baseline comparisons
    print(f"{'─'*95}")
    
    # Test default band
    fp_default, tp_default, _ = test_mvs_configuration(
        baseline_packets, movement_packets,
        DEFAULT_BAND, THRESHOLD, WINDOW_SIZE
    )
    f1_default = (2 * tp_default / (2*tp_default + fp_default + (len(movement_packets) - tp_default)) * 100)
    
    print(f"{'Default (manual) [11-22]':<40} {'[11-22]':<25} {fp_default:<6} {tp_default:<6} {f1_default:<8.1f}")
    
    # Test variance_only band [0-11]
    variance_band = list(range(0, 12))
    fp_var, tp_var, _ = test_mvs_configuration(
        baseline_packets, movement_packets,
        variance_band, THRESHOLD, WINDOW_SIZE
    )
    f1_var = (2 * tp_var / (2*tp_var + fp_var + (len(movement_packets) - tp_var)) * 100)
    
    print(f"{'Variance-only [0-11]':<40} {'[0-11]':<25} {fp_var:<6} {tp_var:<6} {f1_var:<8.1f}")
    
    # Find best variant
    best_variant = max(results, key=lambda x: x['f1'])
    
    # Analysis
    print(f"\n{'='*70}")
    print(f"  ANALYSIS")
    print(f"{'='*70}\n")
    
    print(f"Best NBVI Variant: {best_variant['name']}")
    print(f"  Band: {best_variant['band']}")
    print(f"  F1: {best_variant['f1']:.1f}%")
    print(f"  vs Default: {(best_variant['f1'] - f1_default):+.1f}%")
    print(f"  vs Variance-only: {(best_variant['f1'] - f1_var):+.1f}%")
    
    # Rank all variants
    print(f"\nVariant Ranking:")
    sorted_results = sorted(results, key=lambda x: x['f1'], reverse=True)
    for i, r in enumerate(sorted_results):
        status = "🎉" if r['f1'] >= f1_default else "✅" if r['f1'] >= 95 else "⚠️"
        print(f"  {i+1}. {status} {r['name']}: F1={r['f1']:.1f}%")
    
    if best_variant['f1'] >= f1_default:
        print(f"\n🎉 SUCCESS: Best NBVI variant matches or beats manual optimization!")
    elif best_variant['f1'] >= 95:
        print(f"\n✅ EXCELLENT: Best NBVI variant achieves F1≥95%")
    elif best_variant['f1'] >= f1_var:
        print(f"\n✅ GOOD: Best NBVI variant improves over variance-only")
    else:
        print(f"\n⚠️  All NBVI variants underperform variance-only")
    
    return {
        'best_variant': best_variant,
        'all_results': results,
        'f1_default': f1_default,
        'f1_variance': f1_var
    }

def main():
    print("\n╔═══════════════════════════════════════════════════════╗")
    print("║   NBVI Subcarrier Selection Test                     ║")
    print("╚═══════════════════════════════════════════════════════╝")
    
    # Load data
    print("\n📂 Loading data...")
    try:
        baseline_packets, movement_packets = load_baseline_and_movement()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return
    
    print(f"   Baseline: {len(baseline_packets)} packets")
    print(f"   Movement: {len(movement_packets)} packets")
    
    # Test NBVI variants on pure data
    print("\n" + "="*70)
    print("  PHASE 1: NBVI on PURE DATA (baseline + movement separate)")
    print("="*70)
    results_pure = test_nbvi_selection(baseline_packets, movement_packets)
    
    # Test NBVI on realistic mixed scenario (threshold-based)
    print("\n" + "="*70)
    print("  PHASE 2: NBVI on MIXED DATA - Threshold-Based")
    print("="*70)
    results_mixed_threshold = test_nbvi_realistic_mixed_scenario(baseline_packets, movement_packets, baseline_threshold=0.5)
    
    # Test percentile vs threshold comparison
    print("\n" + "="*70)
    print("  PHASE 3: PERCENTILE vs THRESHOLD Comparison")
    print("="*70)
    results_percentile = test_nbvi_percentile_vs_threshold(baseline_packets, movement_packets)
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*70}\n")
    
    # Pure data results
    best_pure = results_pure['best_variant']
    
    print(f"📊 PHASE 1 - Pure Data Results:")
    print(f"   Best Variant: {best_pure['name']}")
    print(f"   Band: {best_pure['band']}")
    print(f"   F1: {best_pure['f1']:.1f}%")
    print(f"   vs Default: {(best_pure['f1'] - results_pure['f1_default']):+.1f}%")
    
    # Mixed data results (threshold-based)
    mixed_calibrated_thresh = sum(1 for r in results_mixed_threshold if r['calibrated'])
    
    print(f"\n📊 PHASE 2 - Mixed Data (Threshold-Based):")
    print(f"   Calibrated: {mixed_calibrated_thresh}/{len(results_mixed_threshold)}")
    
    if mixed_calibrated_thresh > 0:
        best_mixed_thresh = max([r for r in results_mixed_threshold if r['calibrated']], key=lambda x: x['f1'])
        print(f"   Best: {best_mixed_thresh['variant']}, F1={best_mixed_thresh['f1']:.1f}%")
    
    # Percentile results
    percentile_calibrated = sum(1 for r in results_percentile if r['calibrated'])
    
    print(f"\n📊 PHASE 3 - Percentile vs Threshold:")
    print(f"   Percentile calibrated: {len([r for r in results_percentile if r.get('use_percentile') and r['calibrated']])}")
    print(f"   Threshold calibrated: {len([r for r in results_percentile if not r.get('use_percentile') and r['calibrated']])}")
    
    if percentile_calibrated > 0:
        best_percentile = max([r for r in results_percentile if r['calibrated']], key=lambda x: x['f1'])
        print(f"   Best: {best_percentile['method']}, F1={best_percentile['f1']:.1f}%")
    
    # Overall conclusion
    print(f"\n{'='*70}")
    print(f"  OVERALL CONCLUSION")
    print(f"{'='*70}\n")
    
    # Determine best overall approach
    percentile_success = len([r for r in results_percentile if r.get('use_percentile') and r['calibrated']])
    threshold_success = mixed_calibrated_thresh
    
    if best_pure['f1'] >= 97 and percentile_success > threshold_success:
        print(f"🎉 NBVI PERCENTILE-BASED IS PRODUCTION-READY!")
        print(f"\n   ✅ Pure data: F1={best_pure['f1']:.1f}% (excellent)")
        print(f"   ✅ Percentile: {percentile_success} calibrated (robust)")
        print(f"   ✅ Threshold: {threshold_success} calibrated")
        print(f"   ✅ Gap to manual: {(best_pure['f1'] - results_pure['f1_default']):+.1f}% (minimal)")
        print(f"\n   🎯 Percentile approach eliminates threshold configuration!")
    elif best_pure['f1'] >= 97:
        print(f"🎉 NBVI IS PRODUCTION-READY!")
        print(f"\n   ✅ Pure data: F1={best_pure['f1']:.1f}% (excellent)")
        print(f"   ✅ Mixed data: Works with both approaches")
        print(f"   ✅ Gap to manual: {(best_pure['f1'] - results_pure['f1_default']):+.1f}% (minimal)")
    elif best_pure['f1'] >= 95:
        print(f"✅ NBVI ACHIEVES EXCELLENT PERFORMANCE")
        print(f"\n   Pure data: F1={best_pure['f1']:.1f}%")
        print(f"   Percentile: {percentile_success} calibrated")
        print(f"   Threshold: {threshold_success} calibrated")
    
    print(f"\n💡 Key Insights:")
    print(f"  1. NBVI Weighted α=0.3: Best on pure data (F1={best_pure['f1']:.1f}%)")
    print(f"  2. Percentile-based: {percentile_success} calibrated (NO threshold needed!)")
    print(f"  3. Threshold-based: {threshold_success} calibrated")
    print(f"  4. Noise Gate + Spacing: Critical for performance")
    print(f"  5. Weighting (α=0.3-0.5) essential vs pure NBVI (σ/μ²)")
    
    print()

if __name__ == '__main__':
    main()
