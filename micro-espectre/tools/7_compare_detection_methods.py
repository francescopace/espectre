#!/usr/bin/env python3
"""
Detection Methods Comparison
Compares RSSI, Mean Amplitude, Turbulence, Espressif's esp_radar, and MVS algorithms

Usage:
    python tools/7_compare_detection_methods.py              # Use C6 dataset
    python tools/7_compare_detection_methods.py --chip S3    # Use S3 dataset
    python tools/7_compare_detection_methods.py --plot       # Show visualization

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# Add src to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from config import SEG_WINDOW_SIZE, SEG_THRESHOLD

from csi_utils import load_baseline_and_movement, MVSDetector, calculate_spatial_turbulence, find_dataset, DEFAULT_SUBCARRIERS

# Alias for backward compatibility
SELECTED_SUBCARRIERS = DEFAULT_SUBCARRIERS

# Alias for backward compatibility
WINDOW_SIZE = SEG_WINDOW_SIZE
THRESHOLD = 1.0 if SEG_THRESHOLD == "auto" else SEG_THRESHOLD

def calculate_rssi(csi_packet):
    """Calculate RSSI (mean of all subcarrier amplitudes)"""
    amplitudes = []
    for sc_idx in range(64):
        # Espressif CSI format: [Imaginary, Real, ...] per subcarrier
        Q = float(csi_packet[sc_idx * 2])      # Imaginary first
        I = float(csi_packet[sc_idx * 2 + 1])  # Real second
        amplitude = np.sqrt(I*I + Q*Q)
        amplitudes.append(amplitude)
    return np.mean(amplitudes)


def trimmean(arr, percent):
    """
    Trimmed mean - removes percent/2 from each end before calculating mean.
    Replicates Espressif's trimmean function.
    """
    if len(arr) == 0:
        return 0.0
    sorted_arr = np.sort(arr)
    n = len(sorted_arr)
    trim_count = int(n * percent / 2)
    if trim_count * 2 >= n:
        return np.median(arr)
    trimmed = sorted_arr[trim_count:n - trim_count]
    return np.mean(trimmed) if len(trimmed) > 0 else np.mean(arr)


def pearson_correlation(a, b):
    """
    Calculate Pearson correlation coefficient between two vectors.
    Replicates Espressif's corr() function from utils.c
    
    Returns value in range [-1, 1], where:
    - 1 = perfectly correlated (identical signal shape)
    - 0 = no correlation
    - -1 = perfectly anti-correlated
    """
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    
    mean_a = np.mean(a)
    mean_b = np.mean(b)
    
    cov_sum = 0.0
    var_sum_a = 0.0
    var_sum_b = 0.0
    
    for i in range(len(a)):
        diff_a = a[i] - mean_a
        diff_b = b[i] - mean_b
        cov_sum += diff_a * diff_b
        var_sum_a += diff_a ** 2
        var_sum_b += diff_b ** 2
    
    denominator = np.sqrt(var_sum_a * var_sum_b)
    if denominator < 1e-10:
        return 0.0
    
    return cov_sum / denominator


def pca_power_method(data_matrix, max_iters=30, precision=0.0001):
    """
    PCA using power method to find principal eigenvector.
    Replicates Espressif's pca() function from pca.c
    
    Args:
        data_matrix: 2D array of shape (num_packets, num_subcarriers)
        max_iters: Maximum iterations for power method
        precision: Convergence threshold
    
    Returns:
        Principal component vector of shape (num_subcarriers,)
    """
    if len(data_matrix) == 0:
        return None
    
    # Transpose: we want subcarriers as rows, packets as columns
    # This matches Espressif's matrix layout
    matrix = np.array(data_matrix).T  # (num_subcarriers, num_packets)
    rows, cols = matrix.shape
    
    if cols == 0:
        return None
    
    # Compute covariance matrix (cols x cols)
    # Espressif divides by (rows * cols) for normalization
    zoom_out = rows * cols
    cov_matrix = np.zeros((cols, cols))
    
    for i in range(cols):
        for j in range(i + 1):
            cov_sum = 0.0
            for k in range(rows):
                cov_sum += matrix[k, i] * matrix[k, j]
            cov_matrix[i, j] = cov_sum / zoom_out
            if i != j:
                cov_matrix[j, i] = cov_matrix[i, j]
    
    # Power method to find principal eigenvector
    eigenvector = np.ones(cols)
    eigenvalue = 1.0
    eigenvalue_last = 0.0
    
    for iteration in range(max_iters):
        if abs(eigenvalue - eigenvalue_last) <= precision:
            break
        
        eigenvalue_last = eigenvalue
        eigenvalue = 0.0
        
        # Multiply: eigenvalue_list = cov_matrix @ eigenvector
        eigenvalue_list = np.zeros(cols)
        for i in range(cols):
            for j in range(cols):
                eigenvalue_list[i] += cov_matrix[i, j] * eigenvector[j]
            if eigenvalue_list[i] > eigenvalue:
                eigenvalue = eigenvalue_list[i]
        
        # Normalize
        if eigenvalue > 1e-10:
            eigenvector = eigenvalue_list / eigenvalue
    
    # Project data onto eigenvector to get principal component
    # output[i] = sum(matrix[i, j] * eigenvector[j]) / cols
    output = np.zeros(rows)
    for i in range(rows):
        for j in range(cols):
            output[i] += matrix[i, j] * eigenvector[j]
        output[i] /= cols
    
    return output


class EspressifDetector:
    """
    Implementation of Espressif's esp_radar detection algorithm.
    
    Based on esp-radar component v0.3.1 (Apache-2.0 license).
    Source: https://github.com/espressif/esp-csi
    
    Algorithm:
    1. Collect CSI amplitude data into a sliding window
    2. Apply PCA (power method) to extract principal component vector
    3. Calculate Pearson correlation between current and past PCA vectors:
       - waveform_jitter = max(|corr(pca_current, pca_past[i])|)
       - High correlation (≈1) = similar signal = NO movement
       - Low correlation (≈0) = different signal = movement
    4. Invert values: output = 1 - correlation (so high = movement)
    5. Apply count-based detection: if N violations in window, declare motion
    
    Key difference from MVS:
    - Espressif: PCA + correlation + count-based detection
    - ESPectre MVS: turbulence + moving variance + threshold
    """
    
    # Parameters from Espressif code
    DEFAULT_PCA_WINDOW_SIZE = 10   # Number of packets for PCA computation
    DEFAULT_MOVE_BUFFER_SIZE = 5   # Buffer for jitter values
    DEFAULT_OUTLIERS_NUM = 2       # Need 2/5 violations to trigger
    DEFAULT_MOVE_THRESHOLD = 0.03  # Threshold on inverted jitter (1 - corr)
    BUFF_MAX_LEN = 25
    CSI_CORR_THRESHOLD = 0.98      # Threshold for training sample collection
    
    def __init__(self, 
                 pca_window_size=DEFAULT_PCA_WINDOW_SIZE,
                 move_buffer_size=DEFAULT_MOVE_BUFFER_SIZE,
                 outliers_num=DEFAULT_OUTLIERS_NUM,
                 move_threshold=DEFAULT_MOVE_THRESHOLD,
                 subcarriers=None,
                 track_data=False):
        self.pca_window_size = pca_window_size
        self.move_buffer_size = move_buffer_size
        self.outliers_num = outliers_num
        self.move_threshold = move_threshold
        self.subcarriers = subcarriers
        self.track_data = track_data
        
        # CSI data buffer for PCA (sliding window of amplitude vectors)
        self.csi_buffer = []
        
        # PCA output buffer (stores PCA vectors for correlation)
        self.pca_buffer = []
        self.pca_count = 0
        
        # Calibration data (baseline PCA vectors for wander calculation)
        self.calibration_data = []
        self.is_calibrating = False
        
        # Jitter buffer for smoothing
        self.jitter_buffer = []
        
        # Tracking
        self.jitter_history = []
        self.wander_history = []
        self.state_history = []
        self.move_count = 0
        self.packet_count = 0
    
    def _extract_amplitudes(self, csi_packet):
        """Extract amplitudes from CSI packet"""
        amplitudes = []
        indices = self.subcarriers if self.subcarriers is not None else range(64)
        for sc_idx in indices:
            # Espressif CSI format: [Imaginary, Real, ...] per subcarrier
            Q = float(csi_packet[sc_idx * 2])
            I = float(csi_packet[sc_idx * 2 + 1])
            amplitude = np.sqrt(I*I + Q*Q)
            amplitudes.append(amplitude)
        return np.array(amplitudes)
    
    def _compute_waveform_jitter(self, pca_current):
        """
        Compute waveform_jitter as max correlation with CALIBRATION vectors.
        
        NOTE: Original esp_radar uses correlation with past PCA vectors,
        but this doesn't provide good separation for continuous movement.
        
        Alternative: compare with calibration (baseline) samples.
        - High correlation with baseline = idle
        - Low correlation with baseline = movement (different from baseline)
        
        This provides better separation because movement causes the signal
        to differ from the baseline pattern, not just from the previous packet.
        """
        # If we have calibration data, compare with it
        if len(self.calibration_data) > 0:
            max_corr = 0.0
            for calib_pca in self.calibration_data:
                corr_val = abs(pearson_correlation(pca_current, calib_pca))
                if corr_val > max_corr:
                    max_corr = corr_val
            return max_corr
        
        # Fallback: compare with past PCA vectors (original algorithm)
        if len(self.pca_buffer) < 2:
            return 1.0
        
        max_corr = 0.0
        for i in range(min(self.move_buffer_size - 1, len(self.pca_buffer) - 1)):
            past_idx = len(self.pca_buffer) - 2 - i
            if past_idx < 0:
                break
            pca_past = self.pca_buffer[past_idx]
            corr_val = abs(pearson_correlation(pca_current, pca_past))
            if corr_val > max_corr:
                max_corr = corr_val
        
        return max_corr
    
    def _compute_waveform_wander(self, pca_current):
        """
        Compute waveform_wander as max correlation with calibration samples.
        
        From esp_radar.c line 760-777:
        - Iterate over calibration (baseline) PCA vectors
        - Calculate |corr(current, calibration[i])|
        - Take maximum correlation
        - High correlation = similar to baseline = no presence change
        """
        if len(self.calibration_data) == 0:
            return 1.0  # No calibration data
        
        max_corr = 0.0
        for calib_pca in self.calibration_data:
            corr_val = abs(pearson_correlation(pca_current, calib_pca))
            if corr_val > max_corr:
                max_corr = corr_val
        
        return max_corr
    
    def process_packet(self, csi_packet):
        """
        Process a CSI packet using Espressif's PCA + correlation algorithm.
        Returns (move_status, room_status)
        """
        self.packet_count += 1
        
        # Extract amplitudes
        amplitudes = self._extract_amplitudes(csi_packet)
        
        # Add to CSI buffer (sliding window for PCA)
        self.csi_buffer.append(amplitudes)
        if len(self.csi_buffer) > self.pca_window_size:
            self.csi_buffer.pop(0)
        
        # Need enough data for PCA
        if len(self.csi_buffer) < self.pca_window_size:
            if self.track_data:
                self.jitter_history.append(0.0)
                self.wander_history.append(0.0)
                self.state_history.append('IDLE')
            return False, False
        
        # Compute PCA on current window
        pca_current = pca_power_method(self.csi_buffer)
        if pca_current is None:
            if self.track_data:
                self.jitter_history.append(0.0)
                self.wander_history.append(0.0)
                self.state_history.append('IDLE')
            return False, False
        
        # Calculate waveform metrics (as correlations, 0-1 range)
        jitter_corr = self._compute_waveform_jitter(pca_current)
        wander_corr = self._compute_waveform_wander(pca_current)
        
        # Invert: 1 - correlation, so high value = movement/change
        # This matches esp_radar.c lines 898-899
        jitter_inverted = 1.0 - jitter_corr
        wander_inverted = 1.0 - wander_corr
        
        # Store PCA vector for future comparisons
        self.pca_buffer.append(pca_current.copy())
        if len(self.pca_buffer) > self.BUFF_MAX_LEN:
            self.pca_buffer.pop(0)
        self.pca_count += 1
        
        # Collect calibration samples during initial phase
        # Collect every Nth sample to get diverse baseline patterns
        if len(self.calibration_data) < 20 and self.pca_count % 5 == 0:
            self.calibration_data.append(pca_current.copy())
        
        # Add to jitter buffer for smoothing
        self.jitter_buffer.append(jitter_inverted)
        if len(self.jitter_buffer) > self.BUFF_MAX_LEN:
            self.jitter_buffer.pop(0)
        
        if self.track_data:
            self.jitter_history.append(jitter_inverted)
            self.wander_history.append(wander_inverted)
        
        # Detection logic: count threshold violations
        if len(self.jitter_buffer) < self.move_buffer_size:
            if self.track_data:
                self.state_history.append('IDLE')
            return False, False
        
        # Count how many recent jitter values exceed threshold
        move_count = 0
        jitter_median = np.median(self.jitter_buffer)
        
        for i in range(self.move_buffer_size):
            idx = -(i + 1)
            jitter_val = self.jitter_buffer[idx]
            
            # Espressif's dual condition (adapted for inverted values)
            if (jitter_val > self.move_threshold or 
                (jitter_val > jitter_median and jitter_val > 0.01)):
                move_count += 1
        
        # Final decision
        move_status = move_count >= self.outliers_num
        room_status = wander_inverted > 0.1  # Presence if wander changed significantly
        
        if self.track_data:
            self.state_history.append('MOTION' if move_status else 'IDLE')
            if move_status:
                self.move_count += 1
        
        return move_status, room_status
    
    def get_motion_count(self):
        """Return number of motion detections"""
        return self.move_count
    
    def reset(self):
        """Reset detector state"""
        self.csi_buffer = []
        self.pca_buffer = []
        self.pca_count = 0
        self.calibration_data = []
        self.jitter_buffer = []
        self.jitter_history = []
        self.wander_history = []
        self.state_history = []
        self.move_count = 0
        self.packet_count = 0


def calculate_mean_amplitude(csi_packet, selected_subcarriers):
    """Calculate mean amplitude of selected subcarriers"""
    amplitudes = []
    for sc_idx in selected_subcarriers:
        # Espressif CSI format: [Imaginary, Real, ...] per subcarrier
        Q = float(csi_packet[sc_idx * 2])      # Imaginary first
        I = float(csi_packet[sc_idx * 2 + 1])  # Real second
        amplitude = np.sqrt(I*I + Q*Q)
        amplitudes.append(amplitude)
    return np.mean(amplitudes)

def compare_detection_methods(baseline_packets, movement_packets, subcarriers, window_size, threshold):
    """
    Compare different detection methods on same data
    Returns metrics for each method
    """
    import time
    
    # Calculate metrics for each method
    methods = {
        'RSSI': {'baseline': [], 'movement': []},
        'Mean Amplitude': {'baseline': [], 'movement': []},
        'Turbulence': {'baseline': [], 'movement': []},
        'MVS': {'baseline': [], 'movement': []},
        'Espressif': {'baseline': [], 'movement': []}
    }
    
    # Timing dictionary (in microseconds per packet)
    timing = {}
    all_packets = list(baseline_packets) + list(movement_packets)
    num_packets = len(all_packets)
    
    # Process baseline
    rssi_values = []
    mean_amp_values = []
    turbulence_values = []
    
    for pkt in baseline_packets:
        rssi_values.append(calculate_rssi(pkt['csi_data']))
        mean_amp_values.append(calculate_mean_amplitude(pkt['csi_data'], subcarriers))
        turbulence_values.append(calculate_spatial_turbulence(pkt['csi_data'], subcarriers))
    
    methods['RSSI']['baseline'] = np.array(rssi_values)
    methods['Mean Amplitude']['baseline'] = np.array(mean_amp_values)
    methods['Turbulence']['baseline'] = np.array(turbulence_values)
    
    # Time MVS
    start = time.perf_counter()
    detector_baseline = MVSDetector(window_size, threshold, subcarriers, track_data=True)
    for pkt in baseline_packets:
        detector_baseline.process_packet(pkt['csi_data'])
    methods['MVS']['baseline'] = np.array(detector_baseline.moving_var_history)
    
    # Calculate Espressif method for baseline
    # Espressif uses ALL subcarriers with step_size (default: every 4th = 16 SC)
    # We use all 64 subcarriers to match original behavior
    esp_subcarriers = list(range(0, 64, 4))  # Every 4th subcarrier = 16 total (Espressif default)
    esp_detector_baseline = EspressifDetector(subcarriers=esp_subcarriers, track_data=True)
    for pkt in baseline_packets:
        esp_detector_baseline.process_packet(pkt['csi_data'])
    
    # Calibrate threshold using ESPectre's adaptive method: P95 × 1.4
    # This provides a fair comparison using the same calibration strategy
    baseline_jitter = np.array(esp_detector_baseline.jitter_history)
    valid_jitter = baseline_jitter[baseline_jitter > 0]
    if len(valid_jitter) > 0:
        # ESPectre method: P95 × 1.4 (same as MVS adaptive threshold)
        p95_jitter = np.percentile(valid_jitter, 95)
        calibrated_threshold = p95_jitter * 1.4
    else:
        calibrated_threshold = 0.01  # Default threshold
    
    # Reset and re-run with calibrated threshold
    esp_detector_baseline = EspressifDetector(
        move_threshold=calibrated_threshold,
        subcarriers=esp_subcarriers,
        track_data=True
    )
    for pkt in baseline_packets:
        esp_detector_baseline.process_packet(pkt['csi_data'])
    methods['Espressif']['baseline'] = np.array(esp_detector_baseline.jitter_history)
    
    # Process movement
    rssi_values = []
    mean_amp_values = []
    turbulence_values = []
    
    for pkt in movement_packets:
        rssi_values.append(calculate_rssi(pkt['csi_data']))
        mean_amp_values.append(calculate_mean_amplitude(pkt['csi_data'], subcarriers))
        turbulence_values.append(calculate_spatial_turbulence(pkt['csi_data'], subcarriers))
    
    methods['RSSI']['movement'] = np.array(rssi_values)
    methods['Mean Amplitude']['movement'] = np.array(mean_amp_values)
    methods['Turbulence']['movement'] = np.array(turbulence_values)
    
    # Calculate MVS for movement (continue timing)
    detector_movement = MVSDetector(window_size, threshold, subcarriers, track_data=True)
    for pkt in movement_packets:
        detector_movement.process_packet(pkt['csi_data'])
    mvs_time = time.perf_counter() - start
    timing['MVS'] = (mvs_time / num_packets) * 1e6  # microseconds per packet
    methods['MVS']['movement'] = np.array(detector_movement.moving_var_history)
    
    # Time Espressif (including calibration run)
    start = time.perf_counter()
    esp_detector_movement = EspressifDetector(
        move_threshold=calibrated_threshold,
        subcarriers=esp_subcarriers,
        track_data=True
    )
    for pkt in movement_packets:
        esp_detector_movement.process_packet(pkt['csi_data'])
    # Add baseline time (already ran once for calibration, run again for timing)
    esp_detector_timing = EspressifDetector(
        move_threshold=calibrated_threshold,
        subcarriers=esp_subcarriers,
        track_data=False
    )
    for pkt in baseline_packets:
        esp_detector_timing.process_packet(pkt['csi_data'])
    esp_time = time.perf_counter() - start
    timing['Espressif'] = (esp_time / num_packets) * 1e6  # microseconds per packet
    methods['Espressif']['movement'] = np.array(esp_detector_movement.jitter_history)
    
    # Time simple methods
    start = time.perf_counter()
    for pkt in all_packets:
        calculate_rssi(pkt['csi_data'])
    timing['RSSI'] = ((time.perf_counter() - start) / num_packets) * 1e6
    
    start = time.perf_counter()
    for pkt in all_packets:
        calculate_mean_amplitude(pkt['csi_data'], subcarriers)
    timing['Mean Amplitude'] = ((time.perf_counter() - start) / num_packets) * 1e6
    
    start = time.perf_counter()
    for pkt in all_packets:
        calculate_spatial_turbulence(pkt['csi_data'], subcarriers)
    timing['Turbulence'] = ((time.perf_counter() - start) / num_packets) * 1e6
    
    # Store calibrated threshold for display
    esp_detector_baseline.calibrated_threshold = calibrated_threshold
    esp_detector_movement.calibrated_threshold = calibrated_threshold
    
    return methods, detector_baseline, detector_movement, esp_detector_baseline, esp_detector_movement, timing

def plot_comparison(methods, detector_baseline, detector_movement, 
                   esp_detector_baseline, esp_detector_movement,
                   threshold, subcarriers, esp_subcarriers, timing):
    """
    Plot comparison of detection methods
    """
    fig, axes = plt.subplots(5, 2, figsize=(20, 12))
    fig.suptitle('Detection Methods Comparison', 
                 fontsize=14, fontweight='bold')
    
    # Maximize window
    try:
        mng = plt.get_current_fig_manager()
        # Try different backends
        if hasattr(mng, 'window'):
            if hasattr(mng.window, 'showMaximized'):
                mng.window.showMaximized()  # Qt backend
            elif hasattr(mng.window, 'state'):
                mng.window.state('zoomed')  # TkAgg backend
        elif hasattr(mng, 'full_screen_toggle'):
            mng.full_screen_toggle()  # Some backends
    except Exception:
        pass  # Ignore if maximization fails
    
    method_names = ['RSSI', 'Mean Amplitude', 'Turbulence', 'Espressif', 'MVS']
    
    for row, method_name in enumerate(method_names):
        baseline_data = methods[method_name]['baseline']
        movement_data = methods[method_name]['movement']
        
        # Calculate threshold based on method
        if method_name == 'MVS':
            simple_threshold = threshold
        elif method_name == 'Espressif':
            # Espressif uses calibrated threshold (auto-calibrated from baseline)
            simple_threshold = esp_detector_baseline.move_threshold if esp_detector_baseline else EspressifDetector.DEFAULT_MOVE_THRESHOLD
        else:
            # Simple threshold for RSSI, Mean Amplitude, Turbulence: mean + 2*std of baseline
            simple_threshold = np.mean(baseline_data) + 2 * np.std(baseline_data)
        
        # Time axis
        time_baseline = np.arange(len(baseline_data)) / 100.0
        time_movement = np.arange(len(movement_data)) / 100.0
        
        # ====================================================================
        # LEFT: Baseline
        # ====================================================================
        ax_baseline = axes[row, 0]
        
        # Plot data
        if method_name == 'MVS':
            color = 'blue'
            linewidth = 1.5
        elif method_name == 'Espressif':
            color = 'purple'
            linewidth = 1.5
        else:
            color = 'green'
            linewidth = 1.0
            
        ax_baseline.plot(time_baseline, baseline_data, color=color, alpha=0.7, 
                        linewidth=linewidth, label=method_name)
        ax_baseline.axhline(y=simple_threshold, color='r', linestyle='--', 
                          linewidth=2, label=f'Threshold={simple_threshold:.4f}')
        
        # Highlight false positives
        if method_name == 'MVS':
            for i, state in enumerate(detector_baseline.state_history):
                if state == 'MOTION':
                    ax_baseline.axvspan(i/100.0, (i+1)/100.0, alpha=0.3, color='red')
            fp = detector_baseline.get_motion_count()
        elif method_name == 'Espressif':
            for i, state in enumerate(esp_detector_baseline.state_history):
                if state == 'MOTION':
                    ax_baseline.axvspan(i/100.0, (i+1)/100.0, alpha=0.3, color='red')
            fp = esp_detector_baseline.get_motion_count()
        else:
            fp = np.sum(baseline_data > simple_threshold)
            for i, val in enumerate(baseline_data):
                if val > simple_threshold:
                    ax_baseline.axvspan(i/100.0, (i+1)/100.0, alpha=0.3, color='red')
        
        ax_baseline.set_ylabel('Value', fontsize=10)
        if method_name == 'MVS':
            title_prefix = '[BEST] '
        elif method_name == 'Espressif':
            title_prefix = '[PCA] '
        else:
            title_prefix = ''
        # Show subcarriers and timing in baseline title
        if method_name == 'Espressif':
            sc_info = f"SC: {len(esp_subcarriers)} (step 4)"
        elif method_name in ['MVS', 'Turbulence', 'Mean Amplitude']:
            sc_info = f"SC: {subcarriers[0]}-{subcarriers[-1]}"
        elif method_name == 'RSSI':
            sc_info = "SC: all"
        else:
            sc_info = ""
        time_us = timing.get(method_name, 0)
        time_info = f"{time_us:.0f}us/pkt" if time_us > 0 else ""
        sc_suffix = f" [{sc_info}, {time_info}]" if sc_info else ""
        ax_baseline.set_title(f'{title_prefix}{method_name} - Baseline (FP={fp}){sc_suffix}', 
                            fontsize=11, fontweight='bold')
        ax_baseline.grid(True, alpha=0.3)
        ax_baseline.legend(fontsize=9)
        
        # Add colored border for special methods
        if method_name == 'MVS':
            for spine in ax_baseline.spines.values():
                spine.set_edgecolor('green')
                spine.set_linewidth(3)
        elif method_name == 'Espressif':
            for spine in ax_baseline.spines.values():
                spine.set_edgecolor('purple')
                spine.set_linewidth(3)
        
        if row == 4:  # Bottom row
            ax_baseline.set_xlabel('Time (seconds)', fontsize=10)
        
        # ====================================================================
        # RIGHT: Movement
        # ====================================================================
        ax_movement = axes[row, 1]
        
        # Plot data
        ax_movement.plot(time_movement, movement_data, color=color, alpha=0.7, 
                        linewidth=linewidth, label=method_name)
        ax_movement.axhline(y=simple_threshold, color='r', linestyle='--', 
                          linewidth=2, label=f'Threshold={simple_threshold:.4f}')
        
        # Highlight true positives
        if method_name == 'MVS':
            for i, state in enumerate(detector_movement.state_history):
                if state == 'MOTION':
                    ax_movement.axvspan(i/100.0, (i+1)/100.0, alpha=0.3, color='green')
                else:
                    ax_movement.axvspan(i/100.0, (i+1)/100.0, alpha=0.2, color='red')
            tp = detector_movement.get_motion_count()
            fn = len(movement_data) - tp
        elif method_name == 'Espressif':
            for i, state in enumerate(esp_detector_movement.state_history):
                if state == 'MOTION':
                    ax_movement.axvspan(i/100.0, (i+1)/100.0, alpha=0.3, color='green')
                else:
                    ax_movement.axvspan(i/100.0, (i+1)/100.0, alpha=0.2, color='red')
            tp = esp_detector_movement.get_motion_count()
            fn = len(movement_data) - tp
        else:
            tp = np.sum(movement_data > simple_threshold)
            fn = len(movement_data) - tp
            for i, val in enumerate(movement_data):
                if val > simple_threshold:
                    ax_movement.axvspan(i/100.0, (i+1)/100.0, alpha=0.3, color='green')
                else:
                    ax_movement.axvspan(i/100.0, (i+1)/100.0, alpha=0.2, color='red')
        
        recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
        precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
        
        ax_movement.set_ylabel('Value', fontsize=10)
        ax_movement.set_title(f'{title_prefix}{method_name} - Movement (TP={tp}, R={recall:.0f}%, P={precision:.0f}%)', 
                            fontsize=11, fontweight='bold')
        ax_movement.grid(True, alpha=0.3)
        ax_movement.legend(fontsize=9)
        
        # Add colored border for special methods
        if method_name == 'MVS':
            for spine in ax_movement.spines.values():
                spine.set_edgecolor('green')
                spine.set_linewidth(3)
        elif method_name == 'Espressif':
            for spine in ax_movement.spines.values():
                spine.set_edgecolor('purple')
                spine.set_linewidth(3)
        
        if row == 4:  # Bottom row
            ax_movement.set_xlabel('Time (seconds)', fontsize=10)
    
    plt.tight_layout()
    plt.show()

def print_comparison_summary(methods, detector_baseline, detector_movement, 
                           esp_detector_baseline, esp_detector_movement,
                           threshold, subcarriers, timing):
    """Print comparison summary"""
    print("\n" + "="*80)
    print("  DETECTION METHODS COMPARISON SUMMARY")
    print("="*80)
    print("  Espressif: Full PCA + Pearson correlation algorithm (from esp-radar v0.3.1)")
    print("="*80 + "\n")
    
    print(f"Configuration:")
    print(f"  Subcarriers: {subcarriers}")
    print(f"  MVS Window Size: {WINDOW_SIZE}")
    print(f"  MVS Threshold: {threshold}")
    esp_threshold = getattr(esp_detector_baseline, 'move_threshold', EspressifDetector.DEFAULT_MOVE_THRESHOLD)
    print(f"  Espressif PCA window: {EspressifDetector.DEFAULT_PCA_WINDOW_SIZE}")
    print(f"  Espressif move_threshold: {esp_threshold:.4f} (P95 × 1.4)")
    print(f"  Espressif outliers_num: {EspressifDetector.DEFAULT_OUTLIERS_NUM}/{EspressifDetector.DEFAULT_MOVE_BUFFER_SIZE}")
    print()
    
    # Calculate metrics for each method
    results = []
    for method_name in ['RSSI', 'Mean Amplitude', 'Turbulence', 'Espressif', 'MVS']:
        baseline_data = methods[method_name]['baseline']
        movement_data = methods[method_name]['movement']
        
        if method_name == 'MVS':
            fp = detector_baseline.get_motion_count()
            tp = detector_movement.get_motion_count()
        elif method_name == 'Espressif':
            fp = esp_detector_baseline.get_motion_count()
            tp = esp_detector_movement.get_motion_count()
        else:
            # Simple threshold: mean + 2*std
            simple_threshold = np.mean(baseline_data) + 2 * np.std(baseline_data)
            fp = np.sum(baseline_data > simple_threshold)
            tp = np.sum(movement_data > simple_threshold)
        
        fn = len(movement_data) - tp
        recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
        precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
        results.append({
            'name': method_name,
            'fp': fp,
            'tp': tp,
            'fn': fn,
            'recall': recall,
            'precision': precision,
            'f1': f1
        })
    
    # Find best method by F1 (balances precision and recall)
    best_by_f1 = max(results, key=lambda r: r['f1'])
    best_by_precision = max(results, key=lambda r: r['precision'])
    
    print(f"{'Method':<15} {'FP':<8} {'TP':<8} {'FN':<8} {'Recall':<10} {'Precision':<12} {'F1':<10} {'Time':<10}")
    print("-" * 90)
    
    for r in results:
        marker = " *" if r['name'] == best_by_f1['name'] else "  "
        time_us = timing.get(r['name'], 0)
        time_str = f"{time_us:.0f}us" if time_us > 0 else "-"
        print(f"{marker} {r['name']:<13} {r['fp']:<8} {r['tp']:<8} {r['fn']:<8} {r['recall']:<10.1f} {r['precision']:<12.1f} {r['f1']:<10.1f} {time_str:<10}")
    
    print("-" * 80)
    print(f"\n* Best method by F1 Score: {best_by_f1['name']}")
    print(f"   - F1: {best_by_f1['f1']:.1f}%")
    print(f"   - Recall: {best_by_f1['recall']:.1f}%")
    print(f"   - Precision: {best_by_f1['precision']:.1f}%")
    print(f"   - FP: {best_by_f1['fp']}, FN: {best_by_f1['fn']}")
    
    # Compare MVS vs Espressif specifically
    mvs_result = next(r for r in results if r['name'] == 'MVS')
    esp_result = next(r for r in results if r['name'] == 'Espressif')
    
    print("\n" + "-"*80)
    print("  MVS vs Espressif Comparison")
    print("-"*80)
    print(f"  {'Metric':<15} {'MVS':<15} {'Espressif':<15} {'Winner':<15}")
    print(f"  {'-'*60}")
    
    metrics = [
        ('Recall', mvs_result['recall'], esp_result['recall']),
        ('Precision', mvs_result['precision'], esp_result['precision']),
        ('F1 Score', mvs_result['f1'], esp_result['f1']),
        ('False Pos.', -mvs_result['fp'], -esp_result['fp']),  # Negative because lower is better
    ]
    
    mvs_wins = 0
    esp_wins = 0
    for name, mvs_val, esp_val in metrics:
        if name == 'False Pos.':
            mvs_display = -mvs_val
            esp_display = -esp_val
            winner = 'MVS' if mvs_val > esp_val else ('Espressif' if esp_val > mvs_val else 'Tie')
        else:
            mvs_display = mvs_val
            esp_display = esp_val
            winner = 'MVS' if mvs_val > esp_val else ('Espressif' if esp_val > mvs_val else 'Tie')
        
        if winner == 'MVS':
            mvs_wins += 1
        elif winner == 'Espressif':
            esp_wins += 1
            
        print(f"  {name:<15} {mvs_display:<15.1f} {esp_display:<15.1f} {winner:<15}")
    
    print(f"\n  Overall: MVS wins {mvs_wins}/4, Espressif wins {esp_wins}/4\n")

def main():
    parser = argparse.ArgumentParser(description='Compare detection methods (RSSI, Mean Amplitude, Turbulence, Espressif, MVS)')
    parser.add_argument('--chip', type=str, default='C6',
                        help='Chip type to use: C6, S3, etc. (default: C6)')
    parser.add_argument('--plot', action='store_true', help='Show visualization plots')
    
    args = parser.parse_args()
    
    print("\n╔═══════════════════════════════════════════════════════════╗")
    print("║       Detection Methods Comparison (incl. Espressif)     ║")
    print("╚═══════════════════════════════════════════════════════════╝\n")
    
    # Load data
    chip = args.chip.upper()
    print(f"Loading {chip} data...")
    try:
        baseline_path, movement_path, chip_name = find_dataset(chip=chip)
        baseline_packets, movement_packets = load_baseline_and_movement(chip=chip)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    print(f"   Chip: {chip_name}")
    print(f"   Loaded {len(baseline_packets)} baseline packets")
    print(f"   Loaded {len(movement_packets)} movement packets\n")
    
    # Compare methods
    methods, detector_baseline, detector_movement, esp_detector_baseline, esp_detector_movement, timing = compare_detection_methods(
        baseline_packets, movement_packets, SELECTED_SUBCARRIERS, WINDOW_SIZE, THRESHOLD
    )
    
    # Print summary
    print_comparison_summary(methods, detector_baseline, detector_movement, 
                            esp_detector_baseline, esp_detector_movement,
                            THRESHOLD, SELECTED_SUBCARRIERS, timing)
    
    # Show plot if requested
    if args.plot:
        print("Generating comparison visualization...\n")
        esp_subcarriers = list(range(0, 64, 4))  # Every 4th subcarrier = 16 total
        plot_comparison(methods, detector_baseline, detector_movement, 
                       esp_detector_baseline, esp_detector_movement,
                       THRESHOLD, SELECTED_SUBCARRIERS, esp_subcarriers, timing)

if __name__ == '__main__':
    main()
