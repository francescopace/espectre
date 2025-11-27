#!/usr/bin/env python3
"""
ESPectre - Filtered Segmentation Test
Tests the effect of applying filters BEFORE calculating moving variance.

This script compares different filtering strategies:
1. No filtering (baseline)
2. Butterworth only (low-pass 8Hz)
3. Butterworth + Hampel (outlier removal)
4. Full pipeline (Butterworth + Hampel + Savitzky-Golay)

Usage:
    # Run comparison test
    python test_segmentation_filtered.py
    
    # Show visualization
    python test_segmentation_filtered.py --plot
    
    # Optimize filter parameters
    python test_segmentation_filtered.py --optimize-filters

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy import signal
import pywt
from mvs_utils import load_baseline_and_movement
from config import WINDOW_SIZE, THRESHOLD, SELECTED_SUBCARRIERS

# ============================================================================
# CONFIGURATION
# ============================================================================

BUTTERWORTH_ORDER = 4
BUTTERWORTH_CUTOFF = 4.0  # Hz (optimal for threshold < 2.0, was 8.0)
SAMPLING_RATE = 100.0     # Hz (assumed)

HAMPEL_WINDOW = 5
HAMPEL_THRESHOLD = 3.0

SAVGOL_WINDOW = 5
SAVGOL_POLYORDER = 2

WAVELET_TYPE = 'db4'          # Daubechies 4 (same as C implementation)
WAVELET_LEVEL = 3             # Decomposition level (1-3)
WAVELET_THRESHOLD = 1.0       # Noise threshold
WAVELET_MODE = 'soft'         # 'soft' or 'hard' thresholding

# ============================================================================


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
# FILTER IMPLEMENTATIONS
# ============================================================================

class ButterworthFilter:
    """Butterworth IIR low-pass filter (4th order, 8Hz cutoff @ 100Hz sampling)"""
    
    def __init__(self, order=BUTTERWORTH_ORDER, cutoff=BUTTERWORTH_CUTOFF, fs=SAMPLING_RATE):
        self.order = order
        self.cutoff = cutoff
        self.fs = fs
        
        # Design filter
        nyquist = fs / 2.0
        normal_cutoff = cutoff / nyquist
        self.b, self.a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        
        # Initialize state
        self.zi = signal.lfilter_zi(self.b, self.a)
        self.initialized = False
    
    def filter(self, value):
        """Apply filter to single value"""
        if not self.initialized:
            # Initialize state with first value
            self.zi = self.zi * value
            self.initialized = True
        
        # Apply filter
        filtered, self.zi = signal.lfilter(self.b, self.a, [value], zi=self.zi)
        return filtered[0]
    
    def reset(self):
        """Reset filter state"""
        self.zi = signal.lfilter_zi(self.b, self.a)
        self.initialized = False


class HampelFilter:
    """Hampel filter for outlier detection and removal"""
    
    def __init__(self, window_size=HAMPEL_WINDOW, threshold=HAMPEL_THRESHOLD):
        self.window_size = window_size
        self.threshold = threshold
        self.buffer = []
    
    def filter(self, value):
        """Apply Hampel filter to single value"""
        self.buffer.append(value)
        
        # Keep only window_size values
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        
        # Need at least 3 values for MAD calculation
        if len(self.buffer) < 3:
            return value
        
        # Calculate median and MAD
        median = np.median(self.buffer)
        mad = np.median(np.abs(np.array(self.buffer) - median))
        
        # Check if current value is outlier
        if mad > 1e-6:  # Avoid division by zero
            deviation = abs(value - median) / (1.4826 * mad)
            if deviation > self.threshold:
                return median  # Replace outlier with median
        
        return value
    
    def reset(self):
        """Reset filter state"""
        self.buffer = []


class SavitzkyGolayFilter:
    """Savitzky-Golay filter for smoothing"""
    
    def __init__(self, window_size=SAVGOL_WINDOW, polyorder=SAVGOL_POLYORDER):
        self.window_size = window_size
        self.polyorder = polyorder
        self.buffer = []
    
    def filter(self, value):
        """Apply Savitzky-Golay filter to single value"""
        self.buffer.append(value)
        
        # Keep only window_size values
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        
        # Need full window for filtering
        if len(self.buffer) < self.window_size:
            return value
        
        # Apply Savitzky-Golay filter
        filtered = signal.savgol_filter(self.buffer, self.window_size, self.polyorder)
        
        # Return center value (most recent after filtering)
        return filtered[-1]
    
    def reset(self):
        """Reset filter state"""
        self.buffer = []


class WaveletFilter:
    """Wavelet denoising filter using PyWavelets (Daubechies db4)"""
    
    def __init__(self, wavelet=WAVELET_TYPE, level=WAVELET_LEVEL, 
                 threshold=WAVELET_THRESHOLD, mode=WAVELET_MODE):
        self.wavelet = wavelet
        self.level = level
        self.threshold = threshold
        self.mode = mode
        self.buffer = []
        # Buffer size must be power of 2 for wavelet transform
        self.buffer_size = 64  # Same as C implementation (WAVELET_BUFFER_SIZE)
    
    def filter(self, value):
        """Apply wavelet denoising to single value"""
        self.buffer.append(value)
        
        # Keep only buffer_size values
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        
        # Need full buffer for wavelet transform
        if len(self.buffer) < self.buffer_size:
            return value
        
        # Apply wavelet decomposition
        coeffs = pywt.wavedec(self.buffer, self.wavelet, level=self.level)
        
        # Apply thresholding to detail coefficients (keep approximation unchanged)
        coeffs_thresh = [coeffs[0]]  # Keep approximation
        for detail in coeffs[1:]:
            coeffs_thresh.append(pywt.threshold(detail, self.threshold, mode=self.mode))
        
        # Reconstruct signal
        denoised = pywt.waverec(coeffs_thresh, self.wavelet)
        
        # Handle potential length mismatch due to wavelet reconstruction
        if len(denoised) > self.buffer_size:
            denoised = denoised[:self.buffer_size]
        elif len(denoised) < self.buffer_size:
            # Pad with last value if needed
            denoised = np.pad(denoised, (0, self.buffer_size - len(denoised)), 
                            mode='edge')
        
        # Return middle sample (to minimize edge effects, same as C implementation)
        return denoised[self.buffer_size // 2]
    
    def reset(self):
        """Reset filter state"""
        self.buffer = []


class FilterPipeline:
    """Complete filter pipeline"""
    
    def __init__(self, config):
        """
        config: dict with keys:
            - butterworth: bool
            - hampel: bool
            - savgol: bool
            - wavelet: bool
        """
        self.config = config
        
        # Initialize filters
        self.butterworth = ButterworthFilter() if config.get('butterworth', False) else None
        self.hampel = HampelFilter() if config.get('hampel', False) else None
        self.savgol = SavitzkyGolayFilter() if config.get('savgol', False) else None
        self.wavelet = WaveletFilter() if config.get('wavelet', False) else None
    
    def filter(self, value):
        """Apply filter pipeline to single value"""
        filtered = value
        
        # Apply filters in sequence
        if self.butterworth:
            filtered = self.butterworth.filter(filtered)
        
        if self.hampel:
            filtered = self.hampel.filter(filtered)
        
        if self.savgol:
            filtered = self.savgol.filter(filtered)
        
        if self.wavelet:
            filtered = self.wavelet.filter(filtered)
        
        return filtered
    
    def reset(self):
        """Reset all filters"""
        if self.butterworth:
            self.butterworth.reset()
        if self.hampel:
            self.hampel.reset()
        if self.savgol:
            self.savgol.reset()
        if self.wavelet:
            self.wavelet.reset()

# ============================================================================
# FILTERED STREAMING SEGMENTATION
# ============================================================================

class FilteredStreamingSegmentation:
    """
    Streaming segmentation with optional filtering of turbulence values
    BEFORE adding to moving variance buffer.
    """
    
    def __init__(self, window_size=50, threshold=3.0, filter_config=None, track_data=False):
        self.window_size = window_size
        self.threshold = threshold
        self.track_data = track_data
        
        # Initialize filter pipeline
        if filter_config is None:
            filter_config = {'butterworth': False, 'hampel': False, 'savgol': False}
        self.filter_pipeline = FilterPipeline(filter_config)
        self.filter_config = filter_config
        
        # Circular buffer for turbulence values
        self.turbulence_buffer = np.zeros(window_size)
        self.buffer_index = 0
        self.buffer_count = 0
        
        # State machine
        self.state = 'IDLE'
        self.motion_start = 0
        self.motion_length = 0
        self.packet_index = 0
        
        # Statistics
        self.segments_detected = 0
        self.motion_packets = 0
        
        # Data tracking for visualization
        if track_data:
            self.raw_turbulence_history = []
            self.filtered_turbulence_history = []
            self.moving_var_history = []
            self.state_history = []
    
    def add_turbulence(self, turbulence):
        """Add one turbulence value (with optional filtering) and update state"""
        # FILTER FIRST (if enabled)
        filtered_turbulence = self.filter_pipeline.filter(turbulence)
        
        # Track data for visualization
        if self.track_data:
            self.raw_turbulence_history.append(turbulence)
            self.filtered_turbulence_history.append(filtered_turbulence)
        
        # Add FILTERED value to circular buffer
        self.turbulence_buffer[self.buffer_index] = filtered_turbulence
        self.buffer_index = (self.buffer_index + 1) % self.window_size
        if self.buffer_count < self.window_size:
            self.buffer_count += 1
        
        # Calculate moving variance
        moving_var = self._calculate_moving_variance()
        
        # Track data for visualization
        if self.track_data:
            self.moving_var_history.append(moving_var)
            self.state_history.append(self.state)
        
        segment_completed = False
        
        # State machine
        if self.state == 'IDLE':
            if moving_var > self.threshold:
                self.state = 'MOTION'
                self.motion_start = self.packet_index
                self.motion_length = 1
        else:  # MOTION
            self.motion_length += 1
            
            # Check for motion end
            if moving_var < self.threshold:
                segment_completed = True
                self.segments_detected += 1
                
                self.state = 'IDLE'
                self.motion_length = 0
        
        # Count packets in MOTION state
        if self.state == 'MOTION':
            self.motion_packets += 1
        
        self.packet_index += 1
        return segment_completed
    
    def reset(self):
        """Reset state machine and filters"""
        self.state = 'IDLE'
        self.motion_start = 0
        self.motion_length = 0
        self.packet_index = 0
        self.segments_detected = 0
        self.motion_packets = 0
        
        # Reset filters
        self.filter_pipeline.reset()
        
        # Reset tracking data
        if self.track_data:
            self.raw_turbulence_history = []
            self.filtered_turbulence_history = []
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
# COMPARISON TEST
# ============================================================================

def run_comparison_test(baseline_packets, movement_packets, num_packets=1000, track_data=False):
    """
    Run comparison test with different filter configurations.
    
    Returns:
        dict: Results for each configuration
    """
    configs = {
        'No Filter': {'butterworth': False, 'hampel': False, 'savgol': False, 'wavelet': False},
        'Butterworth': {'butterworth': True, 'hampel': False, 'savgol': False, 'wavelet': False},
        'Hampel': {'butterworth': False, 'hampel': True, 'savgol': False, 'wavelet': False},
        'Savitzky-Golay': {'butterworth': False, 'hampel': False, 'savgol': True, 'wavelet': False},
        'Wavelet': {'butterworth': False, 'hampel': False, 'savgol': False, 'wavelet': True},
        'Butter+Hampel': {'butterworth': True, 'hampel': True, 'savgol': False, 'wavelet': False},
        'Butter+SavGol': {'butterworth': True, 'hampel': False, 'savgol': True, 'wavelet': False},
        'Butter+Wavelet': {'butterworth': True, 'hampel': False, 'savgol': False, 'wavelet': True},
        'Hampel+SavGol': {'butterworth': False, 'hampel': True, 'savgol': True, 'wavelet': False},
        'Wavelet+Hampel': {'butterworth': False, 'hampel': True, 'savgol': False, 'wavelet': True},
        'Full Pipeline': {'butterworth': True, 'hampel': True, 'savgol': True, 'wavelet': False},
        'Full+Wavelet': {'butterworth': True, 'hampel': True, 'savgol': True, 'wavelet': True},
    }
    
    results = {}
    
    for name, config in configs.items():
        seg = FilteredStreamingSegmentation(
            window_size=WINDOW_SIZE,
            threshold=THRESHOLD,
            filter_config=config,
            track_data=track_data
        )
        
        # Test baseline
        seg.reset()
        for pkt in baseline_packets[:num_packets]:
            turbulence = calculate_spatial_turbulence(pkt)
            seg.add_turbulence(turbulence)
        
        baseline_fp = seg.motion_packets
        baseline_motion = seg.motion_packets
        
        # Save baseline data for visualization
        baseline_data = None
        if track_data:
            baseline_data = {
                'raw_turbulence': np.array(seg.raw_turbulence_history),
                'filtered_turbulence': np.array(seg.filtered_turbulence_history),
                'moving_var': np.array(seg.moving_var_history),
                'motion_state': seg.state_history,
                'segments': seg.segments_detected
            }
        
        # Test movement
        seg.reset()
        for pkt in movement_packets[:num_packets]:
            turbulence = calculate_spatial_turbulence(pkt)
            seg.add_turbulence(turbulence)
        
        movement_tp = seg.motion_packets
        movement_motion = seg.motion_packets
        
        # Save movement data for visualization
        movement_data = None
        if track_data:
            movement_data = {
                'raw_turbulence': np.array(seg.raw_turbulence_history),
                'filtered_turbulence': np.array(seg.filtered_turbulence_history),
                'moving_var': np.array(seg.moving_var_history),
                'motion_state': seg.state_history,
                'segments': seg.segments_detected
            }
        
        # Calculate metrics
        fp_rate = baseline_fp / (num_packets / 100.0)
        recall = movement_motion / (num_packets / 100.0)
        score = movement_tp - baseline_fp * 10
        
        results[name] = {
            'config': config,
            'baseline_fp': baseline_fp,
            'baseline_motion': baseline_motion,
            'movement_tp': movement_tp,
            'movement_motion': movement_motion,
            'fp_rate': fp_rate,
            'recall': recall,
            'score': score,
            'baseline_data': baseline_data,
            'movement_data': movement_data
        }
    
    return results

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_comparison(results, threshold):
    """
    Visualize comparison: No Filter baseline + top 3 filter configurations.
    """
    # Always include "No Filter" as baseline
    no_filter = ('No Filter', results['No Filter'])
    
    # Sort other configurations by score (descending) and select top 3
    other_configs = [(name, res) for name, res in results.items() if name != 'No Filter']
    sorted_configs = sorted(other_configs, key=lambda x: x[1]['score'], reverse=True)
    top_3_filters = sorted_configs[:3]
    
    # Combine: No Filter + Top 3
    configs_to_plot = [no_filter] + top_3_filters
    
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
    
    fig.suptitle('ESPectre - No Filter vs Top 3 Filters (by Score)', 
                 fontsize=14, fontweight='bold')
    
    for i, (config_name, result) in enumerate(configs_to_plot):
        # Skip if no data
        if result['baseline_data'] is None:
            continue
        
        baseline_data = result['baseline_data']
        movement_data = result['movement_data']
        
        # Time axis (in seconds @ 20Hz)
        time = np.arange(len(baseline_data['moving_var'])) / 20.0
        
        # Plot baseline
        ax1 = fig.add_subplot(gs[i, 0])
        ax1.plot(time, baseline_data['moving_var'], 'g-', alpha=0.7, linewidth=0.8)
        ax1.axhline(y=threshold, color='r', linestyle='--', linewidth=2)
        
        # Highlight motion state
        for j, state in enumerate(baseline_data['motion_state']):
            if state == 'MOTION':
                ax1.axvspan(j/20.0, (j+1)/20.0, alpha=0.2, color='red')
        
        ax1.set_ylabel('Moving Variance', fontsize=9)
        # Special title for No Filter (baseline)
        if i == 0:
            ax1.set_title(f'Baseline: {config_name}\nBaseline (FP: {result["baseline_fp"]}, Score: {result["score"]:.0f})', 
                         fontsize=10, fontweight='bold')
        else:
            ax1.set_title(f'#{i}: {config_name}\nBaseline (FP: {result["baseline_fp"]}, Score: {result["score"]:.0f})', 
                         fontsize=10, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Plot movement
        ax2 = fig.add_subplot(gs[i, 1])
        ax2.plot(time, movement_data['moving_var'], 'g-', alpha=0.7, linewidth=0.8)
        ax2.axhline(y=threshold, color='r', linestyle='--', linewidth=2)
        
        # Highlight motion state
        for j, state in enumerate(movement_data['motion_state']):
            if state == 'MOTION':
                ax2.axvspan(j/20.0, (j+1)/20.0, alpha=0.2, color='green')
        
        ax2.set_ylabel('Moving Variance', fontsize=9)
        # Special title for No Filter (baseline)
        if i == 0:
            ax2.set_title(f'Baseline: {config_name}\nMovement (TP: {result["movement_tp"]}, Recall: {result["recall"]:.1f}%)', 
                         fontsize=10, fontweight='bold')
        else:
            ax2.set_title(f'#{i}: {config_name}\nMovement (TP: {result["movement_tp"]}, Recall: {result["recall"]:.1f}%)', 
                         fontsize=10, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add x-label only to bottom plots
        if i == 3:
            ax1.set_xlabel('Time (seconds)', fontsize=9)
            ax2.set_xlabel('Time (seconds)', fontsize=9)
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# FILTER PARAMETER OPTIMIZATION
# ============================================================================

def optimize_filter_parameters(baseline_packets, movement_packets):
    """
    Optimize filter parameters using grid search.
    """
    print("\n" + "="*60)
    print("  FILTER PARAMETER OPTIMIZATION")
    print("="*60 + "\n")
    
    # Test different Butterworth cutoff frequencies
    cutoff_values = [4.0, 6.0, 8.0, 10.0, 12.0]
    
    print("Testing Butterworth cutoff frequencies:")
    print("-" * 70)
    print(f"{'Cutoff (Hz)':<15} {'FP':<5} {'TP':<5} {'Score':<8}")
    print("-" * 70)
    
    best_cutoff = BUTTERWORTH_CUTOFF
    best_score = -1000
    
    for cutoff in cutoff_values:
        # Create custom filter
        class CustomButterworthFilter(ButterworthFilter):
            def __init__(self):
                super().__init__(cutoff=cutoff)
        
        # Monkey patch the filter class temporarily
        original_filter = ButterworthFilter
        globals()['ButterworthFilter'] = CustomButterworthFilter
        
        # Test configuration
        seg = FilteredStreamingSegmentation(
            window_size=WINDOW_SIZE,
            threshold=THRESHOLD,
            filter_config={'butterworth': True, 'hampel': False, 'savgol': False}
        )
        
        # Test baseline
        seg.reset()
        for pkt in baseline_packets[:500]:
            seg.add_turbulence(calculate_spatial_turbulence(pkt))
        fp = seg.motion_packets
        
        # Test movement
        seg.reset()
        for pkt in movement_packets[:500]:
            seg.add_turbulence(calculate_spatial_turbulence(pkt))
        tp = seg.motion_packets
        
        score = tp - fp * 10
        
        print(f"{cutoff:<15.1f} {fp:<5} {tp:<5} {score:<8.2f}")
        
        if score > best_score:
            best_score = score
            best_cutoff = cutoff
        
        # Restore original filter
        globals()['ButterworthFilter'] = original_filter
    
    print("-" * 70)
    print(f"\n✅ Best cutoff: {best_cutoff} Hz (score: {best_score:.2f})\n")

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='ESPectre - Filtered Segmentation Test',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--plot', action='store_true',
                       help='Show visualization plots')
    parser.add_argument('--optimize-filters', action='store_true',
                       help='Optimize filter parameters')
    
    args = parser.parse_args()
    
    print("\n╔═══════════════════════════════════════════════════════╗")
    print("║   FILTERED SEGMENTATION TEST                          ║")
    print("║   Testing filters BEFORE moving variance              ║")
    print("╚═══════════════════════════════════════════════════════╝\n")
    
    print("Configuration:")
    print(f"  Window Size: {WINDOW_SIZE} packets")
    print(f"  Threshold: {THRESHOLD}")
    print(f"  Butterworth: {BUTTERWORTH_ORDER}th order, {BUTTERWORTH_CUTOFF}Hz cutoff")
    print(f"  Hampel: window={HAMPEL_WINDOW}, threshold={HAMPEL_THRESHOLD}")
    print(f"  Savitzky-Golay: window={SAVGOL_WINDOW}, polyorder={SAVGOL_POLYORDER}")
    print(f"  Wavelet: {WAVELET_TYPE}, level={WAVELET_LEVEL}, threshold={WAVELET_THRESHOLD}, mode={WAVELET_MODE}\n")
    
    # Load CSI data
    print("Loading CSI data...")
    try:
        baseline_data, movement_data = load_baseline_and_movement()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return
    
    # Extract only CSI data
    baseline_packets = np.array([p['csi_data'] for p in baseline_data])
    movement_packets = np.array([p['csi_data'] for p in movement_data])
    
    print(f"  Loaded {len(baseline_packets)} baseline packets")
    print(f"  Loaded {len(movement_packets)} movement packets\n")
    
    # ========================================================================
    # FILTER OPTIMIZATION MODE
    # ========================================================================
    
    if args.optimize_filters:
        optimize_filter_parameters(baseline_packets, movement_packets)
        return
    
    # ========================================================================
    # COMPARISON TEST
    # ========================================================================
    
    print("="*60)
    print("  RUNNING COMPARISON TEST")
    print("="*60 + "\n")
    
    results = run_comparison_test(baseline_packets, movement_packets, 
                                  num_packets=1000, track_data=args.plot)
    
    # Print results table
    print("Results:")
    print("-" * 90)
    print(f"{'Configuration':<20} {'FP':<5} {'FP%':<8} {'TP':<5} {'Recall%':<10} {'Score':<8}")
    print("-" * 90)
    
    # Print all configurations
    config_order = [
        'No Filter',
        'Butterworth',
        'Hampel',
        'Savitzky-Golay',
        'Wavelet',
        'Butter+Hampel',
        'Butter+SavGol',
        'Butter+Wavelet',
        'Hampel+SavGol',
        'Wavelet+Hampel',
        'Full Pipeline',
        'Full+Wavelet'
    ]
    
    for name in config_order:
        if name in results:
            result = results[name]
            print(f"{name:<20} {result['baseline_fp']:<5} {result['fp_rate']:<8.1f} "
                  f"{result['movement_tp']:<5} {result['recall']:<10.1f} {result['score']:<8.2f}")
    
    print("-" * 90)
    print()
    
    # ========================================================================
    # ANALYSIS
    # ========================================================================
    
    print("="*60)
    print("  ANALYSIS")
    print("="*60 + "\n")
    
    no_filter = results['No Filter']
    butterworth = results['Butterworth']
    butter_hampel = results['Butter+Hampel']
    full_pipeline = results['Full Pipeline']
    
    print("False Positive Reduction:")
    if no_filter['baseline_fp'] > 0:
        butter_reduction = (1 - butterworth['baseline_fp'] / no_filter['baseline_fp']) * 100
        bh_reduction = (1 - butter_hampel['baseline_fp'] / no_filter['baseline_fp']) * 100
        full_reduction = (1 - full_pipeline['baseline_fp'] / no_filter['baseline_fp']) * 100
        
        print(f"  Butterworth:      {butter_reduction:>6.1f}% reduction")
        print(f"  Butter+Hampel:    {bh_reduction:>6.1f}% reduction")
        print(f"  Full Pipeline:    {full_reduction:>6.1f}% reduction")
    else:
        print("  No false positives in baseline (already perfect!)")
    
    print()
    
    # Find best configuration
    best_config = max(results.items(), key=lambda x: x[1]['score'])
    print(f"✅ Best Configuration: {best_config[0]}")
    print(f"   Score: {best_config[1]['score']:.2f}")
    print(f"   FP: {best_config[1]['baseline_fp']}, TP: {best_config[1]['movement_tp']}")
    print()
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    
    if args.plot:
        print("Generating visualizations...\n")
        plot_comparison(results, THRESHOLD)

if __name__ == "__main__":
    main()
