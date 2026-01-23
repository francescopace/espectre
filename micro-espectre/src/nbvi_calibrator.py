"""
NBVI (Normalized Baseline Variability Index) Calibrator

Automatic subcarrier selection based on baseline variability analysis.
Identifies optimal subcarriers for motion detection using statistical analysis.

Algorithm:
1. Collect baseline CSI packets (quiet room)
2. Find candidate baseline windows using percentile-based detection
3. For each candidate, calculate NBVI for all subcarriers
4. Select 12 subcarriers with lowest NBVI and spectral spacing
5. Validate using MVS false positive rate

Output: (selected_band, mv_values)
- selected_band: List of 12 optimal subcarrier indices
- mv_values: Moving variance values for adaptive threshold calculation

Adaptive threshold is calculated externally using threshold.py.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import math
import gc
import os

# Import HT20 constants from config
from src.config import (
    NUM_SUBCARRIERS, EXPECTED_CSI_LEN,
    GUARD_BAND_LOW, GUARD_BAND_HIGH, DC_SUBCARRIER, BAND_SIZE
)

# Import utility functions
from src.utils import (
    calculate_percentile,
    calculate_variance,
    calculate_std,
    calculate_moving_variance
)

# Import interface
from src.calibrator_interface import ICalibrator

# Constants
BUFFER_FILE = '/nbvi_buffer.bin'

# Threshold for null subcarrier detection (mean amplitude below this = null)
NULL_SUBCARRIER_THRESHOLD = 1.0

# MVS parameters for validation
MVS_WINDOW_SIZE = 50
MVS_THRESHOLD = 1.0


def cleanup_buffer_file():
    """Remove any leftover buffer file from previous interrupted runs."""
    try:
        os.remove(BUFFER_FILE)
        print("NBVI: Cleaned up leftover buffer file")
    except OSError:
        pass


class NBVICalibrator(ICalibrator):
    """
    Automatic NBVI calibrator with percentile-based baseline detection
    
    Collects CSI packets at boot and automatically selects optimal subcarriers
    using NBVI Weighted alpha=0.5 algorithm with percentile-based detection.
    
    Uses file-based storage to avoid RAM limitations. Magnitudes stored as
    uint8 (max CSI magnitude ~181 fits in 1 byte).
    
    After subcarrier selection, calculates adaptive threshold using Pxx * factor.
    """
    
    def __init__(self, buffer_size=700,
                 percentile=10, alpha=0.5, min_spacing=1, noise_gate_percentile=25):
        """
        Initialize NBVI calibrator
        
        Args:
            buffer_size: Number of packets to collect (default: 700)
            percentile: Percentile for baseline window detection (default: 10)
            alpha: NBVI weighting factor (default: 0.5)
            min_spacing: Minimum spacing between subcarriers (default: 1)
            noise_gate_percentile: Percentile for noise gate (default: 25)
        """
        self.buffer_size = buffer_size
        self.percentile = percentile
        self.alpha = alpha
        self.min_spacing = min_spacing
        self.noise_gate_percentile = noise_gate_percentile
        self._packet_count = 0
        self._filtered_count = 0
        self._file = None
        self._initialized = False
        
        # Remove old buffer file if exists
        try:
            os.remove(BUFFER_FILE)
        except OSError:
            pass
        
        # Open file for writing
        self._file = open(BUFFER_FILE, 'wb')
        
    def add_packet(self, csi_data):
        """
        Add CSI packet to calibration buffer (file-based)
        
        HT20 only: expects 128 bytes (64 subcarriers x 2 I/Q).
        
        Args:
            csi_data: CSI data array (128 bytes for HT20)
        
        Returns:
            int: Current buffer size (progress indicator)
        """
        if self._packet_count >= self.buffer_size:
            return self.buffer_size
        
        # Filter by expected CSI length (HT20: 128 bytes)
        if len(csi_data) != EXPECTED_CSI_LEN:
            self._filtered_count += 1
            if self._filtered_count % 50 == 1:
                print(f'[WARN] Filtered {self._filtered_count} packets with wrong SC count (got {len(csi_data)} bytes)')
            return self._packet_count
        
        # Initialize on first packet
        if not self._initialized:
            self._initialized = True
            print(f'NBVI: HT20 mode, {NUM_SUBCARRIERS} SC, guard [{GUARD_BAND_LOW}-{GUARD_BAND_HIGH}], DC={DC_SUBCARRIER}')
        
        # Extract magnitudes and write directly to file
        magnitudes = bytearray(NUM_SUBCARRIERS)
        for sc in range(NUM_SUBCARRIERS):
            i_idx = sc * 2
            q_idx = sc * 2 + 1
            
            if q_idx < len(csi_data):
                # Espressif CSI format: [Imaginary, Real, ...] per subcarrier
                Q = int(csi_data[i_idx])   # Imaginary first
                I = int(csi_data[q_idx])   # Real second
                # Handle unsigned bytes (0-255) from Python tests
                if I > 127:
                    I = I - 256
                if Q > 127:
                    Q = Q - 256
                # Calculate magnitude as uint8 (max ~181 fits in byte)
                mag = int(math.sqrt(float(I)*float(I) + float(Q)*float(Q)))
                magnitudes[sc] = min(mag, 255)
            else:
                magnitudes[sc] = 0
        
        # Write to file
        self._file.write(magnitudes)
        self._packet_count += 1
        
        # Flush periodically
        if self._packet_count % 100 == 0:
            self._file.flush()
        
        return self._packet_count
    
    def _read_packet(self, packet_idx):
        """Read a single packet from file"""
        self._file.seek(packet_idx * NUM_SUBCARRIERS)
        data = self._file.read(NUM_SUBCARRIERS)
        return list(data) if data else None
    
    def _read_window(self, start_idx, window_size):
        """Read a window of packets from file"""
        self._file.seek(start_idx * NUM_SUBCARRIERS)
        data = self._file.read(window_size * NUM_SUBCARRIERS)
        if not data:
            return []
        
        window = []
        for i in range(0, len(data), NUM_SUBCARRIERS):
            packet = list(data[i:i+NUM_SUBCARRIERS])
            if len(packet) == NUM_SUBCARRIERS:
                window.append(packet)
        return window
    
    def _prepare_for_reading(self):
        """Close write mode and reopen for reading"""
        if self._file:
            self._file.flush()
            self._file.close()
            gc.collect()
        self._file = open(BUFFER_FILE, 'rb')
    
    def _find_candidate_windows(self, current_band, window_size=200, step=50):
        """
        Find all candidate baseline windows using percentile-based detection.
        
        NO absolute threshold - adapts automatically to environment.
        """
        if self._packet_count < window_size:
            return []
        
        window_results = []
        
        for i in range(0, self._packet_count - window_size + 1, step):
            window = self._read_window(i, window_size)
            if len(window) < window_size:
                continue
            
            turbulences = []
            for packet_mags in window:
                band_mags = [packet_mags[sc] for sc in current_band if sc < len(packet_mags)]
                if band_mags:
                    mean_mag = sum(band_mags) / len(band_mags)
                    variance = sum((m - mean_mag) ** 2 for m in band_mags) / len(band_mags)
                    std = math.sqrt(variance) if variance > 0 else 0.0
                    turbulences.append(std)
            
            if turbulences:
                turb_variance = calculate_variance(turbulences)
                window_results.append((i, turb_variance))
            
            del window
            del turbulences
            if i % 200 == 0:
                gc.collect()
        
        if not window_results:
            return []
        
        variances = [w[1] for w in window_results]
        p_threshold = calculate_percentile(variances, self.percentile)
        
        candidates = [w for w in window_results if w[1] <= p_threshold]
        candidates.sort(key=lambda x: x[1])
        
        return candidates
    
    def _calculate_nbvi_weighted(self, magnitudes):
        """
        Calculate NBVI Weighted (configurable alpha, default 0.5)
        
        NBVI = alpha * (std/mean^2) + (1-alpha) * (std/mean)
        """
        if not magnitudes:
            return {'nbvi': float('inf'), 'mean': 0.0, 'std': 0.0}
        
        mean = sum(magnitudes) / len(magnitudes)
        
        if mean < 1e-6:
            return {'nbvi': float('inf'), 'mean': mean, 'std': 0.0}
        
        variance = sum((m - mean) ** 2 for m in magnitudes) / len(magnitudes)
        std = math.sqrt(variance) if variance > 0 else 0.0
        
        cv = std / mean
        nbvi_energy = std / (mean * mean)
        nbvi_weighted = self.alpha * nbvi_energy + (1 - self.alpha) * cv
        
        return {
            'nbvi': nbvi_weighted,
            'mean': mean,
            'std': std
        }
    
    def _apply_noise_gate(self, subcarrier_metrics):
        """Apply Noise Gate: exclude weak subcarriers and those with infinite NBVI"""
        # Collect valid means (exclude infinite NBVI, matching C++ implementation)
        valid_means = [m['mean'] for m in subcarrier_metrics 
                       if m['mean'] > 1.0 and m['nbvi'] != float('inf')]
        
        if not valid_means:
            print("NBVI: Noise Gate - no valid subcarriers found")
            return []
        
        threshold = calculate_percentile(valid_means, self.noise_gate_percentile)
        # Filter by mean threshold AND exclude infinite NBVI (matching C++)
        return [m for m in subcarrier_metrics 
                if m['mean'] >= threshold and m['nbvi'] != float('inf')]
    
    def _select_with_spacing(self, sorted_metrics, k=12):
        """
        Select subcarriers with spectral de-correlation
        
        Strategy:
        - Top 5: Always include (highest priority, excluding infinite NBVI)
        - Remaining 7: Select with minimum spacing
        """
        # Top 5: exclude infinite NBVI (matching C++ implementation)
        selected = []
        for m in sorted_metrics:
            if len(selected) >= 5:
                break
            if m['nbvi'] != float('inf'):
                selected.append(m['subcarrier'])
        
        for candidate in sorted_metrics[5:]:
            if len(selected) >= k:
                break
            
            sc = candidate['subcarrier']
            min_dist = min(abs(sc - s) for s in selected)
            
            if min_dist >= self.min_spacing:
                selected.append(sc)
        
        if len(selected) < k:
            for candidate in sorted_metrics:
                if len(selected) >= k:
                    break
                sc = candidate['subcarrier']
                if sc not in selected:
                    selected.append(sc)
        
        selected.sort()
        return selected
    
    def _validate_subcarriers(self, band):
        """
        Validate subcarriers by running MVS on entire buffer.
        
        Note: Hampel filter is NOT applied during calibration. Outliers are useful
        information for identifying unstable subcarriers. Hampel is only applied
        during normal operation in the CSI processor.
        
        Returns:
            tuple: (fp_rate, mv_values) where mv_values is list of moving variance values
        """
        if self._packet_count < MVS_WINDOW_SIZE:
            return 0.0, []
        
        turbulence_buffer = [0.0] * MVS_WINDOW_SIZE
        motion_count = 0
        total_packets = 0
        mv_values = []
        
        for pkt_idx in range(self._packet_count):
            packet_mags = self._read_packet(pkt_idx)
            if packet_mags is None:
                continue
            
            band_mags = [packet_mags[sc] for sc in band if sc < len(packet_mags)]
            if not band_mags:
                continue
            
            mean_mag = sum(band_mags) / len(band_mags)
            variance = sum((m - mean_mag) ** 2 for m in band_mags) / len(band_mags)
            turbulence = math.sqrt(variance) if variance > 0 else 0.0
            
            turbulence_buffer.pop(0)
            turbulence_buffer.append(turbulence)
            
            if pkt_idx < MVS_WINDOW_SIZE:
                continue
            
            mv_variance = calculate_variance(turbulence_buffer)
            mv_values.append(mv_variance)
            
            if mv_variance > MVS_THRESHOLD:
                motion_count += 1
            total_packets += 1
        
        fp_rate = motion_count / total_packets if total_packets > 0 else 0.0
        return fp_rate, mv_values
    
    def calibrate(self):
        """
        Calibrate using NBVI Weighted with percentile-based detection.
        
        Returns:
            tuple: (selected_band, mv_values) or (None, []) if failed
        """
        window_size = 200
        step = 50
        
        if self._packet_count < MVS_WINDOW_SIZE + 10:
            print("NBVI: Not enough packets for calibration")
            return None, []
        
        self._prepare_for_reading()
        
        # Use default band for finding candidate windows
        default_band = list(range(GUARD_BAND_LOW, GUARD_BAND_LOW + BAND_SIZE))
        candidates = self._find_candidate_windows(default_band, window_size, step)
        
        if not candidates:
            print("NBVI: Failed to find candidate windows")
            return None, []
        
        print(f"NBVI: Found {len(candidates)} candidate windows")
        
        best_fp_rate = 1.0
        best_band = None
        best_mv_values = []
        best_avg_nbvi = 0.0
        best_avg_mean = 0.0
        best_window_idx = 0
        
        for idx, (start_idx, window_variance) in enumerate(candidates):
            baseline_window = self._read_window(start_idx, window_size)
            if len(baseline_window) < window_size:
                continue
            
            all_metrics = []
            
            for sc in range(NUM_SUBCARRIERS):
                magnitudes = [packet_mags[sc] for packet_mags in baseline_window]
                metrics = self._calculate_nbvi_weighted(magnitudes)
                metrics['subcarrier'] = sc
                
                # Exclude guard bands and DC subcarrier
                if sc < GUARD_BAND_LOW or sc > GUARD_BAND_HIGH or sc == DC_SUBCARRIER:
                    metrics['nbvi'] = float('inf')
                elif metrics['mean'] < NULL_SUBCARRIER_THRESHOLD:
                    metrics['nbvi'] = float('inf')
                
                all_metrics.append(metrics)
            
            filtered_metrics = self._apply_noise_gate(all_metrics)
            
            if len(filtered_metrics) < BAND_SIZE:
                continue
            
            sorted_metrics = sorted(filtered_metrics, key=lambda x: x['nbvi'])
            candidate_band = self._select_with_spacing(sorted_metrics, k=BAND_SIZE)
            
            if len(candidate_band) != BAND_SIZE:
                continue
            
            fp_rate, mv_values = self._validate_subcarriers(candidate_band)
            
            if fp_rate < best_fp_rate:
                best_fp_rate = fp_rate
                best_band = candidate_band
                best_mv_values = mv_values
                best_window_idx = idx
                
                selected_metrics = [m for m in all_metrics if m['subcarrier'] in candidate_band]
                best_avg_nbvi = sum(m['nbvi'] for m in selected_metrics) / len(selected_metrics)
                best_avg_mean = sum(m['mean'] for m in selected_metrics) / len(selected_metrics)
            
            del baseline_window
            del all_metrics
            gc.collect()
        
        if best_band is None:
            print("NBVI: All candidate windows failed - using default subcarriers")
            
            # Run validation on default band to get MV values
            _, mv_values = self._validate_subcarriers(default_band)
            
            print(f"NBVI: Fallback to default band")
            
            if self._filtered_count > 0:
                print(f"  Filtered: {self._filtered_count} packets (wrong SC count)")
            
            return default_band, mv_values
        
        print(f"NBVI: Selected window {best_window_idx + 1}/{len(candidates)} with FP rate {best_fp_rate * 100:.1f}%")
        
        print(f"NBVI: Band selection successful")
        print(f"  Band: {best_band}")
        print(f"  Avg NBVI: {best_avg_nbvi:.6f}")
        print(f"  Avg magnitude: {best_avg_mean:.2f}")
        print(f"  Est. FP rate: {best_fp_rate * 100:.1f}%")
        
        if self._filtered_count > 0:
            print(f"  Filtered: {self._filtered_count} packets (wrong SC count)")
        
        return best_band, best_mv_values
    
    def free_buffer(self):
        """Free resources after calibration is complete."""
        if self._file:
            self._file.close()
            self._file = None
        
        try:
            os.remove(BUFFER_FILE)
        except OSError:
            pass
    
    def get_packet_count(self):
        """Get the number of packets currently in the buffer."""
        return self._packet_count
    
    def is_buffer_full(self):
        """Check if the buffer has collected enough packets."""
        return self._packet_count >= self.buffer_size
        
        gc.collect()
