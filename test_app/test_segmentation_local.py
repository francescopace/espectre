#!/usr/bin/env python3
"""
ESPectre - Local Segmentation Test (Streaming Mode)
Replicates the ESP32 C test flow exactly: packet-by-packet processing.

This script implements temporal segmentation using spatial turbulence and moving 
variance for motion detection, processing packets one-by-one like the C implementation.

Usage:
    # Run with default parameters
    python test_segmentation_local.py
    
    # Show visualization (2 plots: baseline and movement moving variance)
    python test_segmentation_local.py --plot
    
    # Optimize parameters (grid search)
    python test_segmentation_local.py --optimize
    
    # Analyze subcarrier importance
    python test_segmentation_local.py --analyze-subcarriers
    
    # Show help
    python test_segmentation_local.py --help

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import numpy as np
import matplotlib.pyplot as plt
import re
import argparse

# ============================================================================
# CONFIGURATION - Default Parameters
# ============================================================================

FILE_NAME = 'main/real_csi_data_esp32_c6.h'
K_FACTOR = 2.0
WINDOW_SIZE = 10
MIN_SEGMENT = 10
MAX_SEGMENT = 40
SELECTED_SUBCARRIERS = [53, 21, 52, 20, 58, 54, 22, 45, 46, 51, 19, 57]

# ============================================================================
# DATA LOADING
# ============================================================================

def extract_all_packets_from_h(file_path, base_array_name):
    """Extract CSI packets from .h file"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        packet_regex = re.compile(
            r'static\s+const\s+int8_t\s+' + re.escape(base_array_name) + r'\d+\[128\]\s*=\s*{(.*?)}', 
            re.DOTALL
        )
        all_packets = []
        matches = packet_regex.finditer(content)
        for match in matches:
            data_str = match.group(1)
            data_str = re.sub(r'/\*.*?\*/', '', data_str, flags=re.DOTALL)
            data_str = re.sub(r'//.*?\n', '', data_str)
            values = re.findall(r'(-?\d+)', data_str)
            if len(values) == 128:
                all_packets.append(np.array(values, dtype=np.int8))
        if not all_packets:
            return None
        return np.array(all_packets)
    except Exception as e:
        print(f"Error: {e}")
        return None

# ============================================================================
# SPATIAL TURBULENCE CALCULATION
# ============================================================================

def calculate_spatial_turbulence(csi_packet, selected_subcarriers=None):
    """Calculate spatial turbulence (std of subcarrier amplitudes)"""
    sc_list = selected_subcarriers if selected_subcarriers is not None else SELECTED_SUBCARRIERS
    
    amplitudes = []
    for sc_idx in sc_list:
        I = float(csi_packet[sc_idx * 2])
        Q = float(csi_packet[sc_idx * 2 + 1])
        amplitudes.append(np.sqrt(I*I + Q*Q))
    
    return np.std(amplitudes)

# ============================================================================
# STREAMING SEGMENTATION CLASS (Replicates C implementation)
# ============================================================================

class StreamingSegmentation:
    """
    Streaming segmentation that processes packets one-by-one,
    maintaining internal state like the C implementation.
    """
    
    def __init__(self, window_size=10, K=1.5, min_length=20, max_length=60, track_data=False):
        self.window_size = window_size
        self.K = K
        self.min_length = min_length
        self.max_length = max_length
        self.track_data = track_data
        
        # Circular buffer for turbulence values
        self.turbulence_buffer = np.zeros(window_size)
        self.buffer_index = 0
        self.buffer_count = 0
        
        # State machine
        self.state = 'IDLE'
        self.motion_start = 0
        self.motion_length = 0
        self.packet_index = 0
        
        # Calibration
        self.calibrating = False
        self.calibration_variances = []
        self.calibration_target = 0
        self.threshold = 0.0
        self.threshold_calibrated = False
        
        # Statistics
        self.segments_detected = 0
        self.motion_packets = 0
        
        # Data tracking for visualization
        if track_data:
            self.turbulence_history = []
            self.moving_var_history = []
            self.state_history = []
        
    def start_calibration(self, num_samples):
        """Start calibration phase"""
        self.calibrating = True
        self.calibration_variances = []
        self.calibration_target = num_samples
        self.threshold_calibrated = False
        
        # Reset buffers
        self.turbulence_buffer = np.zeros(self.window_size)
        self.buffer_index = 0
        self.buffer_count = 0
        self.packet_index = 0
    
    def add_turbulence(self, turbulence):
        """Add one turbulence value and update state"""
        # Add to circular buffer
        self.turbulence_buffer[self.buffer_index] = turbulence
        self.buffer_index = (self.buffer_index + 1) % self.window_size
        if self.buffer_count < self.window_size:
            self.buffer_count += 1
        
        # Calculate moving variance
        moving_var = self._calculate_moving_variance()
        
        # Track data for visualization
        if self.track_data and not self.calibrating:
            self.turbulence_history.append(turbulence)
            self.moving_var_history.append(moving_var)
            self.state_history.append(self.state)
        
        # During calibration: collect variance values
        if self.calibrating:
            if self.buffer_count >= self.window_size:
                if len(self.calibration_variances) < self.calibration_target:
                    self.calibration_variances.append(moving_var)
            self.packet_index += 1
            return False
        
        # Normal operation: segmentation
        if not self.threshold_calibrated:
            self.packet_index += 1
            return False
        
        segment_completed = False
        
        # State machine
        if self.state == 'IDLE':
            if moving_var > self.threshold:
                self.state = 'MOTION'
                self.motion_start = self.packet_index
                self.motion_length = 1
                self.motion_packets += 1
        else:  # MOTION
            self.motion_length += 1
            self.motion_packets += 1
            
            # Check for motion end or max length
            if moving_var < self.threshold or self.motion_length >= self.max_length:
                # Validate segment
                if self.motion_length >= self.min_length:
                    segment_completed = True
                    self.segments_detected += 1
                
                self.state = 'IDLE'
                self.motion_length = 0
        
        self.packet_index += 1
        return segment_completed
    
    def finalize_calibration(self):
        """Finalize calibration and calculate threshold"""
        if not self.calibrating or len(self.calibration_variances) < 100:
            return False
        
        variances = np.array(self.calibration_variances)
        mean_var = np.mean(variances)
        std_var = np.std(variances)
        
        self.threshold = mean_var + self.K * std_var
        self.threshold_calibrated = True
        self.calibrating = False
        
        return True
    
    def reset(self):
        """Reset state machine but preserve buffer and threshold"""
        self.state = 'IDLE'
        self.motion_start = 0
        self.motion_length = 0
        self.packet_index = 0
        self.segments_detected = 0
        self.motion_packets = 0
        
        # Reset tracking data for new phase
        if self.track_data:
            self.turbulence_history = []
            self.moving_var_history = []
            self.state_history = []
    
    def _calculate_moving_variance(self):
        """Calculate moving variance from buffer"""
        if self.buffer_count < self.window_size:
            return 0.0
        
        mean = np.mean(self.turbulence_buffer)
        variance = np.mean((self.turbulence_buffer - mean) ** 2)
        
        return variance

# ============================================================================
# PARAMETER OPTIMIZATION
# ============================================================================

def optimize_parameters_streaming(baseline_packets, movement_packets):
    """
    Grid search to find optimal parameters using streaming segmentation.
    
    Returns:
        dict: Best configuration and results
    """
    print("\n" + "="*60)
    print("  PARAMETER OPTIMIZATION (Streaming)")
    print("="*60 + "\n")
    
    # Parameter ranges to test
    k_values = [1.0, 1.5, 2.0, 2.5, 3.0]
    window_sizes = [5, 10, 15, 20, 25]
    min_lengths = [10, 15, 20, 25]
    max_lengths = [40, 60, 80]
    
    best_config = None
    best_score = -1000
    all_results = []
    
    total_combinations = len(k_values) * len(window_sizes) * len(min_lengths) * len(max_lengths)
    print(f"Testing {total_combinations} parameter combinations...\n")
    
    tested = 0
    for K in k_values:
        for window_size in window_sizes:
            for min_length in min_lengths:
                for max_length in max_lengths:
                    tested += 1
                    
                    # Test configuration
                    seg = StreamingSegmentation(window_size, K, min_length, max_length)
                    
                    # Calibrate
                    seg.start_calibration(500)
                    for pkt in baseline_packets[:500]:
                        seg.add_turbulence(calculate_spatial_turbulence(pkt))
                    seg.finalize_calibration()
                    
                    # Test baseline
                    seg.reset()
                    for pkt in baseline_packets[:500]:
                        seg.add_turbulence(calculate_spatial_turbulence(pkt))
                    baseline_fp = seg.segments_detected
                    
                    # Test movement
                    seg.reset()
                    for pkt in movement_packets[:500]:
                        seg.add_turbulence(calculate_spatial_turbulence(pkt))
                    movement_tp = seg.segments_detected
                    
                    # Score: penalize FP heavily, reward TP
                    score = movement_tp - baseline_fp * 10
                    
                    result = {
                        'K': K,
                        'window_size': window_size,
                        'min_length': min_length,
                        'max_length': max_length,
                        'false_positives': baseline_fp,
                        'true_positives': movement_tp,
                        'score': score
                    }
                    
                    all_results.append(result)
                    
                    if score > best_score:
                        best_score = score
                        best_config = result
                    
                    if tested % 50 == 0:
                        print(f"  Progress: {tested}/{total_combinations} ({tested*100//total_combinations}%)")
    
    print(f"\n✅ Optimization complete!\n")
    
    # Print top 5 configurations
    sorted_results = sorted(all_results, key=lambda x: x['score'], reverse=True)
    
    print("Top 5 Configurations:")
    print("-" * 80)
    print(f"{'Rank':<6} {'K':<6} {'Window':<8} {'MinLen':<8} {'MaxLen':<8} {'FP':<5} {'TP':<5} {'Score':<8}")
    print("-" * 80)
    
    for i, result in enumerate(sorted_results[:5]):
        print(f"{i+1:<6} {result['K']:<6.1f} {result['window_size']:<8} {result['min_length']:<8} "
              f"{result['max_length']:<8} {result['false_positives']:<5} {result['true_positives']:<5} {result['score']:<8.2f}")
    
    print("-" * 80)
    print()
    
    return best_config

def analyze_subcarrier_importance(baseline_packets, movement_packets):
    """
    Analyze which subcarriers are most informative.
    
    Returns:
        list: Sorted subcarrier indices by importance
    """
    print("\n" + "="*60)
    print("  SUBCARRIER IMPORTANCE ANALYSIS")
    print("="*60 + "\n")
    
    num_subcarriers = 64
    
    # Calculate per-subcarrier statistics
    baseline_means = np.zeros(num_subcarriers)
    baseline_vars = np.zeros(num_subcarriers)
    movement_means = np.zeros(num_subcarriers)
    movement_vars = np.zeros(num_subcarriers)
    
    for sc in range(num_subcarriers):
        baseline_amps = []
        movement_amps = []
        
        for pkt in baseline_packets[:500]:
            I = float(pkt[sc * 2])
            Q = float(pkt[sc * 2 + 1])
            baseline_amps.append(np.sqrt(I*I + Q*Q))
        
        for pkt in movement_packets[:500]:
            I = float(pkt[sc * 2])
            Q = float(pkt[sc * 2 + 1])
            movement_amps.append(np.sqrt(I*I + Q*Q))
        
        baseline_means[sc] = np.mean(baseline_amps)
        baseline_vars[sc] = np.var(baseline_amps)
        movement_means[sc] = np.mean(movement_amps)
        movement_vars[sc] = np.var(movement_amps)
    
    # Calculate Fisher score
    fisher_scores = np.zeros(num_subcarriers)
    for sc in range(num_subcarriers):
        mean_diff = abs(movement_means[sc] - baseline_means[sc])
        var_sum = baseline_vars[sc] + movement_vars[sc]
        fisher_scores[sc] = (mean_diff ** 2) / (var_sum + 1e-6)
    
    # Sort by Fisher score
    sorted_indices = np.argsort(fisher_scores)[::-1]
    
    print("Top 20 Most Informative Subcarriers:")
    print("-" * 70)
    print(f"{'Rank':<6} {'SC':<6} {'Fisher':<12} {'VarRatio':<12} {'MeanDiff':<12}")
    print("-" * 70)
    
    for i in range(min(20, num_subcarriers)):
        sc = sorted_indices[i]
        var_ratio = movement_vars[sc] / (baseline_vars[sc] + 1e-6)
        mean_diff = abs(movement_means[sc] - baseline_means[sc])
        print(f"{i+1:<6} {sc:<6} {fisher_scores[sc]:<12.4f} {var_ratio:<12.2f} {mean_diff:<12.2f}")
    
    print("-" * 70)
    print()
    
    # Test different subcarrier selections
    test_configs = [
        ('ALL [0-63]', list(range(64))),
        ('Top 8', sorted_indices[:8].tolist()),
        ('Top 12', sorted_indices[:12].tolist()),
        ('Top 16', sorted_indices[:16].tolist()),
        ('Current', SELECTED_SUBCARRIERS),
    ]
    
    print("Testing subcarrier configurations:")
    print("-" * 70)
    print(f"{'Config':<20} {'#SC':<6} {'FP':<5} {'TP':<5} {'Score':<8}")
    print("-" * 70)
    
    for name, sc_list in test_configs:
        seg = StreamingSegmentation(WINDOW_SIZE, K_FACTOR, MIN_SEGMENT, MAX_SEGMENT)
        
        # Calibrate
        seg.start_calibration(500)
        for pkt in baseline_packets[:500]:
            turb = calculate_spatial_turbulence(pkt, sc_list)
            seg.add_turbulence(turb)
        seg.finalize_calibration()
        
        # Test baseline
        seg.reset()
        for pkt in baseline_packets[:500]:
            seg.add_turbulence(calculate_spatial_turbulence(pkt, sc_list))
        fp = seg.segments_detected
        
        # Test movement
        seg.reset()
        for pkt in movement_packets[:500]:
            seg.add_turbulence(calculate_spatial_turbulence(pkt, sc_list))
        tp = seg.segments_detected
        
        score = tp - fp * 10
        print(f"{name:<20} {len(sc_list):<6} {fp:<5} {tp:<5} {score:<8.2f}")
    
    print("-" * 70)
    print()
    
    return sorted_indices

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_streaming_results(baseline_data, movement_data, threshold):
    """
    Visualize streaming segmentation results.
    
    Args:
        baseline_data: dict with 'moving_var', 'segments', 'motion_state'
        movement_data: dict with 'moving_var', 'segments', 'motion_state'
        threshold: adaptive threshold value
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle('ESPectre - Segmentation Analysis (Moving Variance)', fontsize=14, fontweight='bold')
    
    # Time axis (in seconds @ 20Hz)
    time_baseline = np.arange(len(baseline_data['moving_var'])) / 20.0
    time_movement = np.arange(len(movement_data['moving_var'])) / 20.0
    
    # Plot 1: Baseline Moving Variance
    axes[0].plot(time_baseline, baseline_data['moving_var'], 'g-', alpha=0.7, linewidth=0.8, label='Moving Variance')
    axes[0].axhline(y=threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold = {threshold:.4f}')
    
    # Highlight motion state
    for i, state in enumerate(baseline_data['motion_state']):
        if state == 'MOTION':
            axes[0].axvspan(i/20.0, (i+1)/20.0, alpha=0.2, color='red')
    
    axes[0].set_ylabel('Moving Variance', fontsize=10)
    axes[0].set_title(f'Baseline - Moving Variance (FP Segments: {baseline_data["segments"]})', fontsize=11, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right')
    
    # Plot 2: Movement Moving Variance
    axes[1].plot(time_movement, movement_data['moving_var'], 'g-', alpha=0.7, linewidth=0.8, label='Moving Variance')
    axes[1].axhline(y=threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold = {threshold:.4f}')
    
    # Highlight motion state
    for i, state in enumerate(movement_data['motion_state']):
        if state == 'MOTION':
            axes[1].axvspan(i/20.0, (i+1)/20.0, alpha=0.2, color='green')
    
    axes[1].set_xlabel('Time (seconds)', fontsize=10)
    axes[1].set_ylabel('Moving Variance', fontsize=10)
    axes[1].set_title(f'Movement - Moving Variance (TP Segments: {movement_data["segments"]})', fontsize=11, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# MAIN TEST
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='ESPectre - Local Segmentation Test (Streaming Mode)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--plot', action='store_true',
                       help='Show visualization plots')
    parser.add_argument('--optimize', action='store_true',
                       help='Run parameter optimization (grid search)')
    parser.add_argument('--analyze-subcarriers', action='store_true',
                       help='Analyze subcarrier importance')
    
    args = parser.parse_args()
    
    print("\n╔═══════════════════════════════════════════════════════╗")
    print("║   SEGMENTATION TEST (Streaming Mode)                  ║")
    print("║   Replicates ESP32 C code exactly                     ║")
    print("╚═══════════════════════════════════════════════════════╝\n")
    
    print("Configuration:")
    print(f"  K Factor: {K_FACTOR}")
    print(f"  Window Size: {WINDOW_SIZE} packets ({WINDOW_SIZE/20.0:.2f}s)")
    print(f"  Min Segment: {MIN_SEGMENT} packets ({MIN_SEGMENT/20.0:.2f}s)")
    print(f"  Max Segment: {MAX_SEGMENT} packets ({MAX_SEGMENT/20.0:.2f}s)")
    print(f"  Subcarriers: {SELECTED_SUBCARRIERS}\n")
    
    # Load CSI data
    print("Loading CSI data...")
    baseline_packets = extract_all_packets_from_h(FILE_NAME, 'real_baseline_')
    movement_packets = extract_all_packets_from_h(FILE_NAME, 'real_movement_')
    
    if baseline_packets is None or movement_packets is None:
        print("ERROR: Failed to load CSI data")
        return
    
    print(f"  Loaded {len(baseline_packets)} baseline packets")
    print(f"  Loaded {len(movement_packets)} movement packets\n")
    
    # ========================================================================
    # SPECIAL MODES
    # ========================================================================
    
    if args.optimize:
        best_config = optimize_parameters_streaming(baseline_packets, movement_packets)
        if best_config:
            print(f"Best Configuration:")
            print(f"  K={best_config['K']}, window={best_config['window_size']}, "
                  f"min={best_config['min_length']}, max={best_config['max_length']}")
            print(f"  FP={best_config['false_positives']}, TP={best_config['true_positives']}, "
                  f"Score={best_config['score']:.2f}\n")
        return
    
    if args.analyze_subcarriers:
        sorted_indices = analyze_subcarrier_importance(baseline_packets, movement_packets)
        print(f"✅ Top 8 subcarriers: {sorted_indices[:8].tolist()}\n")
        return
    
    # ========================================================================
    # PHASE 1: CALIBRATION
    # ========================================================================
    
    print("="*60)
    print("  PHASE 1: CALIBRATION")
    print("="*60 + "\n")
    
    seg = StreamingSegmentation(
        window_size=WINDOW_SIZE,
        K=K_FACTOR,
        min_length=MIN_SEGMENT,
        max_length=MAX_SEGMENT,
        track_data=args.plot
    )
    
    seg.start_calibration(1000)
    
    for pkt in baseline_packets[:1000]:
        turbulence = calculate_spatial_turbulence(pkt)
        seg.add_turbulence(turbulence)
    
    seg.finalize_calibration()
    
    print(f"Calibration complete:")
    print(f"  Samples: {len(seg.calibration_variances)}")
    print(f"  Threshold: {seg.threshold:.4f}\n")
    
    # ========================================================================
    # PHASE 2: TEST ON BASELINE
    # ========================================================================
    
    print("="*60)
    print("  PHASE 2: TEST ON BASELINE")
    print("="*60 + "\n")
    
    seg.reset()
    
    for pkt in baseline_packets[:1000]:
        turbulence = calculate_spatial_turbulence(pkt)
        seg.add_turbulence(turbulence)
    
    baseline_fp = seg.segments_detected
    baseline_motion = seg.motion_packets
    
    print(f"Results:")
    print(f"  Packets: 1000")
    print(f"  Motion packets: {baseline_motion} ({baseline_motion/10.0:.1f}%)")
    print(f"  Segments (FP): {baseline_fp}")
    print(f"  FP Rate: {baseline_fp/10.0:.1f}%\n")
    
    # ========================================================================
    # PHASE 3: TEST ON MOVEMENT
    # ========================================================================
    
    print("="*60)
    print("  PHASE 3: TEST ON MOVEMENT")
    print("="*60 + "\n")
    
    seg.reset()
    
    for pkt in movement_packets[:1000]:
        turbulence = calculate_spatial_turbulence(pkt)
        seg.add_turbulence(turbulence)
    
    movement_tp = seg.segments_detected
    movement_motion = seg.motion_packets
    
    print(f"Results:")
    print(f"  Packets: 1000")
    print(f"  Motion packets: {movement_motion} ({movement_motion/10.0:.1f}%)")
    print(f"  Segments (TP): {movement_tp}")
    print(f"  Recall: {movement_motion/10.0:.1f}%\n")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("="*60)
    print("  SUMMARY")
    print("="*60 + "\n")
    
    print(f"Baseline:")
    print(f"  FP Rate: {baseline_fp/10.0:.1f}% ({baseline_fp} segments)")
    print(f"  Motion packets: {baseline_motion} ({baseline_motion/10.0:.1f}%)\n")
    
    print(f"Movement:")
    print(f"  Recall: {movement_motion/10.0:.1f}% ({movement_motion}/1000 packets)")
    print(f"  Segments: {movement_tp}\n")
    
    if baseline_fp == 0 and movement_tp >= 15:
        print("✅ EXCELLENT: Perfect segmentation!")
        print(f"   0 false positives, {movement_tp} segments detected")
    elif baseline_fp == 0:
        print("✅ GOOD: No false positives")
        print(f"   {movement_tp} segments detected (expected ~17)")
    else:
        print(f"⚠️  WARNING: {baseline_fp} false positive segments")
    
    print()
    
    # ========================================================================
    # VISUALIZATION (if --plot enabled)
    # ========================================================================
    
    if args.plot and seg.track_data:
        print("Generating visualization...\n")
        
        # Re-run with data tracking enabled
        seg_plot = StreamingSegmentation(
            window_size=WINDOW_SIZE,
            K=K_FACTOR,
            min_length=MIN_SEGMENT,
            max_length=MAX_SEGMENT,
            track_data=True
        )
        
        # Calibrate
        seg_plot.start_calibration(1000)
        for pkt in baseline_packets[:1000]:
            seg_plot.add_turbulence(calculate_spatial_turbulence(pkt))
        seg_plot.finalize_calibration()
        
        # Test baseline
        seg_plot.reset()
        for pkt in baseline_packets[:1000]:
            seg_plot.add_turbulence(calculate_spatial_turbulence(pkt))
        
        baseline_data = {
            'turbulence': np.array(seg_plot.turbulence_history),
            'moving_var': np.array(seg_plot.moving_var_history),
            'motion_state': seg_plot.state_history,
            'segments': seg_plot.segments_detected
        }
        
        # Test movement
        seg_plot.reset()
        for pkt in movement_packets[:1000]:
            seg_plot.add_turbulence(calculate_spatial_turbulence(pkt))
        
        movement_data = {
            'turbulence': np.array(seg_plot.turbulence_history),
            'moving_var': np.array(seg_plot.moving_var_history),
            'motion_state': seg_plot.state_history,
            'segments': seg_plot.segments_detected
        }
        
        plot_streaming_results(baseline_data, movement_data, seg_plot.threshold)

if __name__ == "__main__":
    main()
