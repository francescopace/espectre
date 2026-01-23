#!/usr/bin/env python3
"""
ESPectre - Filter Comparison Tool

Visualizes how different filters affect the turbulence signal:
1. No Filter (raw signal)
2. Hampel Only (outlier removal)
3. Lowpass Only (frequency smoothing)
4. Hampel + Lowpass (combined)

The key insight is that these filters have DIFFERENT purposes:
- Hampel: Removes spikes/outliers without smoothing the signal
- Lowpass: Smooths high-frequency noise but introduces lag
- Combined: Best of both - spike removal + noise smoothing

Usage:
    python 5_analyze_filter_turbulence.py              # Use C6 dataset
    python 5_analyze_filter_turbulence.py --chip S3    # Use S3 dataset
    python 5_analyze_filter_turbulence.py --plot       # Show visualization
    python 5_analyze_filter_turbulence.py --optimize-filters

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path
from scipy import signal
import pywt

# Add src to path for config and segmentation import
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from config import (SEG_WINDOW_SIZE, SEG_THRESHOLD,
                    HAMPEL_WINDOW, HAMPEL_THRESHOLD, LOWPASS_CUTOFF)
from segmentation import SegmentationContext

from csi_utils import calculate_spatial_turbulence, HampelFilter, find_dataset, DEFAULT_SUBCARRIERS
from csi_utils import load_baseline_and_movement

# Alias for backward compatibility
SELECTED_SUBCARRIERS = DEFAULT_SUBCARRIERS

# Alias for backward compatibility
WINDOW_SIZE = SEG_WINDOW_SIZE
THRESHOLD = 1.0 if SEG_THRESHOLD == "auto" else SEG_THRESHOLD

# ============================================================================
# CONFIGURATION
# ============================================================================

# Sampling rate
SAMPLING_RATE = 100.0       # - SAMPLING_RATE: Data acquisition rate in Hz (must match your sensor)

# EMA (Exponential Moving Average) filter parameters
# fc ≈ Fs × α / (2π) → for fc=20Hz @ 100Hz: α ≈ 0.8 (actually ~12.7 Hz)
# For fc=25Hz: α ≈ 0.95
EMA_ALPHA = 0.95            # - ALPHA: Very light smoothing, ~24 Hz cutoff

# SMA (Simple Moving Average) filter parameters
# fc ≈ 0.44 × Fs / N → for fc=20Hz: N ≈ 2.2 → use 2
SMA_WINDOW = 2              # - WINDOW: Minimal averaging for ~22 Hz cutoff

# Butterworth low-pass filter parameters
BUTTERWORTH_ORDER = 2       # - ORDER: Lower order = gentler slope
BUTTERWORTH_CUTOFF = 17.0   # - CUTOFF: 17 Hz preserves movement (5-20 Hz has 20-25x contrast)

# Chebyshev low-pass filter parameters
CHEBYSHEV_ORDER = 2         # - ORDER: Filter steepness
CHEBYSHEV_CUTOFF = 20.0     # - CUTOFF: 20 Hz cutoff
CHEBYSHEV_RIPPLE = 0.5      # - RIPPLE: Lower ripple = flatter passband

# Bessel low-pass filter parameters (best for preserving transients)
BESSEL_ORDER = 2            # - ORDER: Filter steepness
BESSEL_CUTOFF = 20.0        # - CUTOFF: 20 Hz cutoff

# Hampel filter parameters - imported from src/config.py:
# HAMPEL_WINDOW, HAMPEL_THRESHOLD

# Savitzky-Golay filter parameters (smoothing while preserving peaks)
SAVGOL_WINDOW = 5          # - WINDOW: Number of points used for polynomial fitting (must be odd)
SAVGOL_POLYORDER = 2       # - POLYORDER: Degree of the fitting polynomial (must be < WINDOW)

# Wavelet filter parameters (denoising)
WAVELET_TYPE = 'db4'          # Daubechies 4 (same as C implementation)
WAVELET_LEVEL = 3             # - LEVEL: Decomposition level (1-3)
WAVELET_THRESHOLD = 1.0       # - THRESHOLD: Noise threshold
WAVELET_MODE = 'soft'         # - MODE: 'soft' or 'hard' thresholding

# ============================================================================


# ============================================================================
# FILTER IMPLEMENTATIONS
# ============================================================================

class EMAFilter:
    """Exponential Moving Average (EMA) low-pass filter.
    
    Simplest IIR filter with minimal memory (O(1)) and low latency.
    Formula: y[n] = α × x[n] + (1-α) × y[n-1]
    Cutoff frequency: fc ≈ Fs × α / (2π)
    
    Pro: Minimal memory, low latency, simple
    Con: Slow roll-off (-20 dB/decade), phase distortion
    """
    
    def __init__(self, alpha=EMA_ALPHA):
        self.alpha = alpha
        self.last_output = None
    
    def filter(self, value):
        """Apply EMA filter to single value"""
        if self.last_output is None:
            self.last_output = value
            return value
        
        self.last_output = self.alpha * value + (1 - self.alpha) * self.last_output
        return self.last_output
    
    def reset(self):
        """Reset filter state"""
        self.last_output = None


class SMAFilter:
    """Simple Moving Average (SMA) low-pass filter.
    
    FIR filter with linear phase (no distortion).
    Cutoff frequency: fc ≈ 0.44 × Fs / N
    
    Pro: Linear phase, always stable, simple
    Con: High latency (N/2 samples), requires buffer O(N)
    """
    
    def __init__(self, window_size=SMA_WINDOW):
        self.window_size = window_size
        self.buffer = []
        self.sum = 0.0
    
    def filter(self, value):
        """Apply SMA filter to single value"""
        self.buffer.append(value)
        self.sum += value
        
        if len(self.buffer) > self.window_size:
            self.sum -= self.buffer.pop(0)
        
        return self.sum / len(self.buffer)
    
    def reset(self):
        """Reset filter state"""
        self.buffer = []
        self.sum = 0.0


class ButterworthFilter:
    """Butterworth IIR low-pass filter.
    
    Maximally flat frequency response in passband.
    Roll-off: -20N dB/decade (N = order)
    
    Pro: Flat passband, precise cutoff, efficient
    Con: Phase distortion, potential overshoot
    """
    
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


class ChebyshevFilter:
    """Chebyshev Type I IIR low-pass filter.
    
    Steeper roll-off than Butterworth, but with ripple in passband.
    
    Pro: Steepest roll-off for given order
    Con: Ripple in passband, more phase distortion
    """
    
    def __init__(self, order=CHEBYSHEV_ORDER, cutoff=CHEBYSHEV_CUTOFF, 
                 ripple=CHEBYSHEV_RIPPLE, fs=SAMPLING_RATE):
        self.order = order
        self.cutoff = cutoff
        self.ripple = ripple
        self.fs = fs
        
        # Design filter
        nyquist = fs / 2.0
        normal_cutoff = cutoff / nyquist
        self.b, self.a = signal.cheby1(order, ripple, normal_cutoff, btype='low', analog=False)
        
        # Initialize state
        self.zi = signal.lfilter_zi(self.b, self.a)
        self.initialized = False
    
    def filter(self, value):
        """Apply filter to single value"""
        if not self.initialized:
            self.zi = self.zi * value
            self.initialized = True
        
        filtered, self.zi = signal.lfilter(self.b, self.a, [value], zi=self.zi)
        return filtered[0]
    
    def reset(self):
        """Reset filter state"""
        self.zi = signal.lfilter_zi(self.b, self.a)
        self.initialized = False


class BesselFilter:
    """Bessel IIR low-pass filter.
    
    Optimized for linear phase (constant group delay).
    Preserves signal shape better than Butterworth.
    
    Pro: Near-linear phase, preserves transients, no overshoot
    Con: Slower roll-off than Butterworth
    """
    
    def __init__(self, order=BESSEL_ORDER, cutoff=BESSEL_CUTOFF, fs=SAMPLING_RATE):
        self.order = order
        self.cutoff = cutoff
        self.fs = fs
        
        # Design filter (bessel uses 'norm' parameter)
        nyquist = fs / 2.0
        normal_cutoff = cutoff / nyquist
        self.b, self.a = signal.bessel(order, normal_cutoff, btype='low', analog=False, norm='phase')
        
        # Initialize state
        self.zi = signal.lfilter_zi(self.b, self.a)
        self.initialized = False
    
    def filter(self, value):
        """Apply filter to single value"""
        if not self.initialized:
            self.zi = self.zi * value
            self.initialized = True
        
        filtered, self.zi = signal.lfilter(self.b, self.a, [value], zi=self.zi)
        return filtered[0]
    
    def reset(self):
        """Reset filter state"""
        self.zi = signal.lfilter_zi(self.b, self.a)
        self.initialized = False


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
            - ema: bool
            - sma: bool
            - butterworth: bool
            - chebyshev: bool
            - bessel: bool
            - hampel: bool
            - savgol: bool
            - wavelet: bool
        """
        self.config = config
        
        # Initialize filters with parameters matching production configuration
        self.ema = EMAFilter() if config.get('ema', False) else None
        self.sma = SMAFilter() if config.get('sma', False) else None
        self.butterworth = ButterworthFilter() if config.get('butterworth', False) else None
        self.chebyshev = ChebyshevFilter() if config.get('chebyshev', False) else None
        self.bessel = BesselFilter() if config.get('bessel', False) else None
        self.hampel = HampelFilter(window_size=HAMPEL_WINDOW, threshold=HAMPEL_THRESHOLD) if config.get('hampel', False) else None
        self.savgol = SavitzkyGolayFilter() if config.get('savgol', False) else None
        self.wavelet = WaveletFilter() if config.get('wavelet', False) else None
    
    def filter(self, value):
        """Apply filter pipeline to single value"""
        filtered = value
        
        # Apply low-pass filters first (in order of complexity)
        if self.ema:
            filtered = self.ema.filter(filtered)
        
        if self.sma:
            filtered = self.sma.filter(filtered)
        
        if self.butterworth:
            filtered = self.butterworth.filter(filtered)
        
        if self.chebyshev:
            filtered = self.chebyshev.filter(filtered)
        
        if self.bessel:
            filtered = self.bessel.filter(filtered)
        
        # Then apply outlier/smoothing filters
        if self.hampel:
            filtered = self.hampel.filter(filtered)
        
        if self.savgol:
            filtered = self.savgol.filter(filtered)
        
        if self.wavelet:
            filtered = self.wavelet.filter(filtered)
        
        return filtered
    
    def reset(self):
        """Reset all filters"""
        if self.ema:
            self.ema.reset()
        if self.sma:
            self.sma.reset()
        if self.butterworth:
            self.butterworth.reset()
        if self.chebyshev:
            self.chebyshev.reset()
        if self.bessel:
            self.bessel.reset()
        if self.hampel:
            self.hampel.reset()
        if self.savgol:
            self.savgol.reset()
        if self.wavelet:
            self.wavelet.reset()

# ============================================================================
# FILTERED STREAMING SEGMENTATION (uses production SegmentationContext)
# ============================================================================

class FilteredStreamingSegmentation:
    """
    Wrapper around SegmentationContext that applies external filters
    BEFORE passing turbulence values to the production MVS implementation.
    
    This ensures consistent results with the production code while allowing
    experimentation with different filter configurations.
    """
    
    def __init__(self, window_size=50, threshold=3.0, filter_config=None, track_data=False):
        self.window_size = window_size
        self.threshold = threshold
        self.track_data = track_data
        
        # Initialize external filter pipeline
        if filter_config is None:
            filter_config = {}
        self.filter_pipeline = FilterPipeline(filter_config)
        self.filter_config = filter_config
        
        # Use production SegmentationContext with built-in filters DISABLED
        # (we apply filters externally via FilterPipeline)
        self._context = SegmentationContext(
            window_size=window_size,
            threshold=threshold,
            enable_hampel=False,  # Disabled - we apply filters externally
            enable_lowpass=False  # Disabled - we apply filters externally
        )
        
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
        # FILTER FIRST using external pipeline
        filtered_turbulence = self.filter_pipeline.filter(turbulence)
        
        # Track data for visualization
        if self.track_data:
            self.raw_turbulence_history.append(turbulence)
            self.filtered_turbulence_history.append(filtered_turbulence)
        
        # Pass FILTERED value to production SegmentationContext
        self._context.add_turbulence(filtered_turbulence)
        self._context.update_state()  # Calculate variance and update state machine
        
        # Track moving variance
        if self.track_data:
            self.moving_var_history.append(self._context.current_moving_variance)
            state_str = 'MOTION' if self._context.get_state() == SegmentationContext.STATE_MOTION else 'IDLE'
            self.state_history.append(state_str)
        
        # Count packets in MOTION state
        if self._context.get_state() == SegmentationContext.STATE_MOTION:
            self.motion_packets += 1
        
        return False  # Segment completion not tracked in this wrapper
    
    def reset(self):
        """Reset state machine and filters"""
        self._context.reset(full=True)
        self.filter_pipeline.reset()
        self.segments_detected = 0
        self.motion_packets = 0
        
        # Reset tracking data
        if self.track_data:
            self.raw_turbulence_history = []
            self.filtered_turbulence_history = []
            self.moving_var_history = []
            self.state_history = []

# ============================================================================
# COMPARISON TEST
# ============================================================================

def run_comparison_test(baseline_packets, movement_packets, num_packets=None, track_data=False):
    """
    Run comparison test with different filter configurations.
    
    Args:
        baseline_packets: List of baseline CSI packets
        movement_packets: List of movement CSI packets
        num_packets: Max packets to process (None = all packets)
        track_data: Whether to track data for visualization
    
    Returns:
        dict: Results for each configuration
    """
    # Use all packets if not specified
    if num_packets is None:
        num_packets = max(len(baseline_packets), len(movement_packets))
    configs = {
        # Baseline
        'No Filter': {},
        
        # Single low-pass filters
        'EMA': {'ema': True},
        'SMA': {'sma': True},
        'Butterworth': {'butterworth': True},
        'Chebyshev': {'chebyshev': True},
        'Bessel': {'bessel': True},
        
        # Single other filters
        'Hampel': {'hampel': True},
        'Savitzky-Golay': {'savgol': True},
        'Wavelet': {'wavelet': True},
        
        # Combinations with Butterworth
        'Butter+Hampel': {'butterworth': True, 'hampel': True},
        'Butter+SavGol': {'butterworth': True, 'savgol': True},
        
        # Combinations with other low-pass
        'EMA+Hampel': {'ema': True, 'hampel': True},
        'Bessel+Hampel': {'bessel': True, 'hampel': True},
        'Chebyshev+Hampel': {'chebyshev': True, 'hampel': True},
        
        # Full pipelines
        'Full Pipeline': {'butterworth': True, 'hampel': True, 'savgol': True},
        'Bessel+Full': {'bessel': True, 'hampel': True, 'savgol': True},
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
        baseline_to_process = baseline_packets[:num_packets]
        for csi_data in baseline_to_process:
            turbulence = calculate_spatial_turbulence(csi_data, SELECTED_SUBCARRIERS)
            seg.add_turbulence(turbulence)
        
        baseline_fp = seg.motion_packets
        baseline_motion = seg.motion_packets
        baseline_count = len(baseline_to_process)
        
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
        movement_to_process = movement_packets[:num_packets]
        for csi_data in movement_to_process:
            turbulence = calculate_spatial_turbulence(csi_data, SELECTED_SUBCARRIERS)
            seg.add_turbulence(turbulence)
        
        movement_tp = seg.motion_packets
        movement_motion = seg.motion_packets
        movement_count = len(movement_to_process)
        
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
        
        # Calculate metrics using actual packet counts
        fp_rate = baseline_fp / baseline_count * 100 if baseline_count > 0 else 0
        recall = movement_motion / movement_count * 100 if movement_count > 0 else 0
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

def plot_filter_effect(baseline_packets, movement_packets, num_packets=500):
    """
    Visualize the effect of different filters on moving variance.
    
    Shows 4 filter configurations:
    1. No Filter (raw signal)
    2. Hampel Only (outlier removal)
    3. Lowpass Only (frequency smoothing)  
    4. Hampel + Lowpass (combined)
    
    For each configuration, shows:
    - Left column: Moving Variance on BASELINE (should stay below threshold)
    - Right column: Moving Variance on MOVEMENT (should exceed threshold)
    
    Uses production SegmentationContext for consistent results.
    
    Args:
        baseline_packets: numpy array of CSI data (shape: num_packets x num_subcarriers*2)
        movement_packets: numpy array of CSI data (shape: num_packets x num_subcarriers*2)
        num_packets: Number of packets to process
    """
    # Configuration for the 4 filter setups (using production SegmentationContext options)
    filter_configs = [
        ('No Filter', {'hampel': False, 'lowpass': False}),
        ('Hampel Only', {'hampel': True, 'lowpass': False}),
        ('Lowpass Only', {'hampel': False, 'lowpass': True}),
        ('Hampel + Lowpass', {'hampel': True, 'lowpass': True}),
    ]
    
    # Process both baseline and movement data with each filter configuration
    results = {}
    
    for name, config in filter_configs:
        # Create SegmentationContext with specific filter configuration
        ctx_baseline = SegmentationContext(
            window_size=WINDOW_SIZE,
            threshold=THRESHOLD,
            enable_hampel=config['hampel'],
            enable_lowpass=config['lowpass']
        )
        
        # Process BASELINE
        baseline_mv = []
        for i in range(min(num_packets, len(baseline_packets))):
            turb = calculate_spatial_turbulence(baseline_packets[i], SELECTED_SUBCARRIERS)
            ctx_baseline.add_turbulence(turb)
            ctx_baseline.update_state()
            baseline_mv.append(ctx_baseline.current_moving_variance)
        
        baseline_fp = sum(1 for v in baseline_mv if v > THRESHOLD)
        
        # Create new context for movement (fresh state)
        ctx_movement = SegmentationContext(
            window_size=WINDOW_SIZE,
            threshold=THRESHOLD,
            enable_hampel=config['hampel'],
            enable_lowpass=config['lowpass']
        )
        
        # Process MOVEMENT
        movement_mv = []
        for i in range(min(num_packets, len(movement_packets))):
            turb = calculate_spatial_turbulence(movement_packets[i], SELECTED_SUBCARRIERS)
            ctx_movement.add_turbulence(turb)
            ctx_movement.update_state()
            movement_mv.append(ctx_movement.current_moving_variance)
        
        movement_tp = sum(1 for v in movement_mv if v > THRESHOLD)
        
        results[name] = {
            'baseline_mv': np.array(baseline_mv),
            'movement_mv': np.array(movement_mv),
            'baseline_fp': baseline_fp,
            'movement_tp': movement_tp,
            'config': config
        }
    
    # Create visualization
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('ESPectre - Filter Effect on Moving Variance\n'
                 'Left: Baseline (FP should be 0) | Right: Movement (TP should be high)',
                 fontsize=13, fontweight='bold')
    
    # Maximize window
    try:
        mng = plt.get_current_fig_manager()
        if hasattr(mng, 'window'):
            if hasattr(mng.window, 'showMaximized'):
                mng.window.showMaximized()
            elif hasattr(mng.window, 'state'):
                mng.window.state('zoomed')
        elif hasattr(mng, 'full_screen_toggle'):
            mng.full_screen_toggle()
    except Exception:
        pass
    
    # Colors for each filter type
    colors = {
        'No Filter': '#666666',
        'Hampel Only': '#e74c3c',
        'Lowpass Only': '#3498db',
        'Hampel + Lowpass': '#27ae60'
    }
    
    for i, (name, data) in enumerate(results.items()):
        # Time axes
        time_baseline = np.arange(len(data['baseline_mv'])) / SAMPLING_RATE
        time_movement = np.arange(len(data['movement_mv'])) / SAMPLING_RATE
        
        # Left plot: Baseline Moving Variance
        ax1 = fig.add_subplot(4, 2, i*2 + 1)
        ax1.plot(time_baseline, data['baseline_mv'], color=colors[name], 
                linewidth=1.0, alpha=0.8)
        ax1.axhline(y=THRESHOLD, color='red', linestyle='--', linewidth=2, 
                   label=f'Threshold ({THRESHOLD})')
        
        # Highlight false positives
        for j, var in enumerate(data['baseline_mv']):
            if var > THRESHOLD:
                ax1.axvspan(j/SAMPLING_RATE, (j+1)/SAMPLING_RATE, 
                           alpha=0.3, color='red')
        
        ax1.set_ylabel('Moving Variance', fontsize=9)
        fp_pct = data['baseline_fp'] / len(data['baseline_mv']) * 100 if len(data['baseline_mv']) > 0 else 0
        ax1.set_title(f'{name}\nBaseline (FP: {data["baseline_fp"]}, {fp_pct:.1f}%)', 
                     fontsize=10, fontweight='bold', color=colors[name])
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        if i == 3:
            ax1.set_xlabel('Time (seconds)', fontsize=9)
        
        # Right plot: Movement Moving Variance
        ax2 = fig.add_subplot(4, 2, i*2 + 2)
        ax2.plot(time_movement, data['movement_mv'], color=colors[name], 
                linewidth=1.0, alpha=0.8)
        ax2.axhline(y=THRESHOLD, color='red', linestyle='--', linewidth=2, 
                   label=f'Threshold ({THRESHOLD})')
        
        # Highlight true positives
        for j, var in enumerate(data['movement_mv']):
            if var > THRESHOLD:
                ax2.axvspan(j/SAMPLING_RATE, (j+1)/SAMPLING_RATE, 
                           alpha=0.2, color='green')
        
        ax2.set_ylabel('Moving Variance', fontsize=9)
        tp_pct = data['movement_tp'] / len(data['movement_mv']) * 100 if len(data['movement_mv']) > 0 else 0
        ax2.set_title(f'{name}\nMovement (TP: {data["movement_tp"]}, {tp_pct:.1f}%)', 
                     fontsize=10, fontweight='bold', color=colors[name])
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        if i == 3:
            ax2.set_xlabel('Time (seconds)', fontsize=9)
    
    plt.tight_layout()
    
    # Add explanation text
    fig.text(0.5, 0.01, 
             'Hampel: Removes spikes/outliers (preserves signal shape) | '
             'Lowpass: Smooths high-frequency noise (introduces lag) | '
             'Combined: Best of both',
             ha='center', fontsize=9, style='italic', color='#666666')
    
    plt.subplots_adjust(bottom=0.05)
    plt.show()
    
    return results


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
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
    
    fig.suptitle('ESPectre - No Filter vs Top 3 Filters (by Score)', 
                 fontsize=14, fontweight='bold')
    
    # Maximize window
    try:
        mng = plt.get_current_fig_manager()
        if hasattr(mng, 'window'):
            if hasattr(mng.window, 'showMaximized'):
                mng.window.showMaximized()
            elif hasattr(mng.window, 'state'):
                mng.window.state('zoomed')
        elif hasattr(mng, 'full_screen_toggle'):
            mng.full_screen_toggle()
    except Exception:
        pass
    
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
        for csi_data in baseline_packets[:500]:
            seg.add_turbulence(calculate_spatial_turbulence(csi_data, SELECTED_SUBCARRIERS))
        fp = seg.motion_packets
        
        # Test movement
        seg.reset()
        for csi_data in movement_packets[:500]:
            seg.add_turbulence(calculate_spatial_turbulence(csi_data, SELECTED_SUBCARRIERS))
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
    parser.add_argument('--chip', type=str, default='C6',
                       help='Chip type to use: C6, S3, etc. (default: C6)')
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
    print(f"  Sampling Rate: {SAMPLING_RATE} Hz")
    print(f"  EMA: α={EMA_ALPHA} (fc≈{SAMPLING_RATE * EMA_ALPHA / (2*3.14159):.1f} Hz)")
    print(f"  SMA: window={SMA_WINDOW} (fc≈{0.44 * SAMPLING_RATE / SMA_WINDOW:.1f} Hz)")
    print(f"  Butterworth: {BUTTERWORTH_ORDER}th order, {BUTTERWORTH_CUTOFF}Hz cutoff")
    print(f"  Chebyshev: {CHEBYSHEV_ORDER}th order, {CHEBYSHEV_CUTOFF}Hz cutoff, {CHEBYSHEV_RIPPLE}dB ripple")
    print(f"  Bessel: {BESSEL_ORDER}th order, {BESSEL_CUTOFF}Hz cutoff")
    print(f"  Hampel: window={HAMPEL_WINDOW}, threshold={HAMPEL_THRESHOLD}")
    print(f"  Savitzky-Golay: window={SAVGOL_WINDOW}, polyorder={SAVGOL_POLYORDER}")
    print(f"  Wavelet: {WAVELET_TYPE}, level={WAVELET_LEVEL}, threshold={WAVELET_THRESHOLD}, mode={WAVELET_MODE}\n")
    
    # Load CSI data
    chip = args.chip.upper()
    print(f"Loading CSI data for {chip}...")
    try:
        baseline_path, movement_path, chip_name = find_dataset(chip=chip)
        baseline_data, movement_data = load_baseline_and_movement(chip=chip)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return
    
    # Extract only CSI data
    baseline_packets = np.array([p['csi_data'] for p in baseline_data])
    movement_packets = np.array([p['csi_data'] for p in movement_data])
    
    print(f"  Chip: {chip_name}")
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
                                  num_packets=None, track_data=args.plot)
    
    # Print results table
    print("Results:")
    print("-" * 90)
    print(f"{'Configuration':<20} {'FP':<5} {'FP%':<8} {'TP':<5} {'Recall%':<10} {'Score':<8}")
    print("-" * 90)
    
    # Print all configurations
    config_order = [
        'No Filter',
        # Single low-pass
        'EMA',
        'SMA',
        'Butterworth',
        'Chebyshev',
        'Bessel',
        # Single other
        'Hampel',
        'Savitzky-Golay',
        'Wavelet',
        # Combinations
        'Butter+Hampel',
        'Butter+SavGol',
        'EMA+Hampel',
        'Bessel+Hampel',
        'Chebyshev+Hampel',
        'Full Pipeline',
        'Bessel+Full',
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
    
    print("Low-Pass Filter Comparison:")
    print("-" * 50)
    lowpass_filters = ['EMA', 'SMA', 'Butterworth', 'Chebyshev', 'Bessel']
    for name in lowpass_filters:
        if name in results:
            r = results[name]
            if no_filter['baseline_fp'] > 0:
                fp_reduction = (1 - r['baseline_fp'] / no_filter['baseline_fp']) * 100
            else:
                fp_reduction = 0
            recall_diff = r['recall'] - no_filter['recall']
            print(f"  {name:<12}: FP {fp_reduction:>+6.1f}%, Recall {recall_diff:>+5.1f}%, Score {r['score']:>6.0f}")
    print()
    
    # Find best overall configuration
    best_config = max(results.items(), key=lambda x: x[1]['score'])
    print(f"✅ Best Configuration: {best_config[0]}")
    print(f"   Score: {best_config[1]['score']:.2f}")
    print(f"   FP: {best_config[1]['baseline_fp']}, TP: {best_config[1]['movement_tp']}")
    print(f"   Recall: {best_config[1]['recall']:.1f}%, FP Rate: {best_config[1]['fp_rate']:.1f}%")
    print()
    
    # Find best low-pass only
    best_lowpass = max([(n, r) for n, r in results.items() if n in lowpass_filters], 
                       key=lambda x: x[1]['score'])
    print(f"🔽 Best Low-Pass Filter: {best_lowpass[0]}")
    print(f"   Score: {best_lowpass[1]['score']:.2f}")
    print()
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    
    if args.plot:
        print("Generating filter comparison visualization...\n")
        plot_filter_effect(baseline_packets, movement_packets, num_packets=500)

if __name__ == "__main__":
    main()
