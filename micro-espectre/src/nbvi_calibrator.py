"""
NBVI (Normalized Baseline Variability Index) Calibrator

Automatic subcarrier selection based on baseline variability analysis.
Identifies optimal subcarriers for motion detection using statistical analysis.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import math
import gc
import os

# Constants
NUM_SUBCARRIERS = 64
BUFFER_FILE = '/nbvi_buffer.bin'

# Default target mean amplitude for normalization (common reference scale)
# Optimized with 60s noisy baseline + LowPass 11Hz filter:
# TARGET=28 → Scale≈1.06, FP=2.3%, Recall=92.4%, F1=88.9%
NORMALIZATION_TARGET_MEAN_DEFAULT = 28.0

# Threshold for null subcarrier detection (mean amplitude below this = null)
# This is environment-aware: works with any chip and adapts to local RF conditions
NULL_SUBCARRIER_THRESHOLD = 1.0


def get_valid_subcarriers(chip_type=None):
    """
    Get all subcarrier indices for calibration.
    
    Null subcarrier detection is now dynamic during calibration based on
    actual signal strength (mean < NULL_SUBCARRIER_THRESHOLD = null).
    This is environment-aware and works with any chip type.
    
    Args:
        chip_type: Ignored (kept for backward compatibility)
    
    Returns:
        tuple: All subcarrier indices (0-63)
    """
    # Return all subcarriers - null detection happens during calibration
    return tuple(range(NUM_SUBCARRIERS))


class NBVICalibrator:
    """
    Automatic NBVI calibrator with percentile-based baseline detection
    
    Collects CSI packets at boot and automatically selects optimal subcarriers
    using NBVI Weighted α=0.3 algorithm with percentile-based detection.
    
    Uses file-based storage to avoid RAM limitations. Magnitudes stored as
    uint8 (max CSI magnitude ~181 fits in 1 byte). This allows collecting
    thousands of packets without memory issues.
    """
    
    def __init__(self, buffer_size=1000, percentile=10, alpha=0.3, min_spacing=3, chip_type=None,
                 normalization_enabled=True, normalization_target=None):
        """
        Initialize NBVI calibrator
        
        Args:
            buffer_size: Number of packets to collect (default: 1000)
            percentile: Percentile for baseline detection (default: 10)
            alpha: NBVI weighting factor (default: 0.3)
            min_spacing: Minimum spacing between subcarriers (default: 3)
            chip_type: Ignored (kept for backward compatibility)
            normalization_enabled: Enable/disable auto-normalization (default: True)
            normalization_target: Target mean amplitude (default: NORMALIZATION_TARGET_MEAN_DEFAULT)
        """
        self.buffer_size = buffer_size
        self.percentile = percentile
        self.alpha = alpha
        self.min_spacing = min_spacing
        self.noise_gate_percentile = 10
        self.chip_type = chip_type  # Kept for backward compatibility
        self.normalization_enabled = normalization_enabled
        self.normalization_target = normalization_target if normalization_target is not None else NORMALIZATION_TARGET_MEAN_DEFAULT
        self._packet_count = 0
        self._file = None
        
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
        
        Args:
            csi_data: CSI data array (128 bytes: 64 subcarriers × 2 I/Q)
        
        Returns:
            int: Current buffer size (progress indicator)
        """
        if self._packet_count >= self.buffer_size:
            return self.buffer_size
        
        # Extract magnitudes and write directly to file
        magnitudes = bytearray(NUM_SUBCARRIERS)
        for sc in range(NUM_SUBCARRIERS):
            i_idx = sc * 2
            q_idx = sc * 2 + 1
            
            if q_idx < len(csi_data):
                # CSI I/Q values are signed int8 (-128 to 127)
                # On MicroPython ESP32: frame[5] is array('b') = signed int8
                # On Python tests with numpy: already int8
                # On Python tests with bytes: need conversion from unsigned
                I = csi_data[i_idx]
                Q = csi_data[q_idx]
                # Handle unsigned bytes (0-255) from Python tests
                # MicroPython array('b') returns signed directly, so no conversion needed there
                if I > 127:
                    I = I - 256
                if Q > 127:
                    Q = Q - 256
                # Calculate magnitude as uint8 (max ~181 fits in byte)
                mag = int(math.sqrt(I*I + Q*Q))
                magnitudes[sc] = min(mag, 255)  # Clamp to uint8
            else:
                magnitudes[sc] = 0
        
        # Write to file
        self._file.write(magnitudes)
        self._packet_count += 1
        
        # Flush periodically to ensure data is written
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
        
        # Convert to list of lists
        window = []
        for i in range(0, len(data), NUM_SUBCARRIERS):
            packet = list(data[i:i+NUM_SUBCARRIERS])
            if len(packet) == NUM_SUBCARRIERS:
                window.append(packet)
        return window
    
    def _prepare_for_reading(self):
        """Close write mode and reopen for reading"""
        if self._file:
            self._file.flush()  # Assicura scrittura completa
            self._file.close()
            gc.collect()
        self._file = open(BUFFER_FILE, 'rb')
    
    def _calculate_variance_two_pass(self, values):
        """Two-pass variance algorithm (numerically stable)"""
        if not values:
            return 0.0
        
        n = len(values)
        if n < 2:
            return 0.0
        
        # First pass: calculate mean
        mean = sum(values) / n
        
        # Second pass: calculate variance
        sum_sq_diff = sum((x - mean) ** 2 for x in values)
        variance = sum_sq_diff / n
        
        return variance
    
    def _find_baseline_window_percentile(self, current_band, window_size=100, step=50):
        """
        Find baseline window using percentile-based detection
        
        NO absolute threshold - adapts automatically to environment
        
        Args:
            current_band: Current subcarrier band (for variance calculation)
            window_size: Size of analysis window (default: 100 packets)
            step: Step size for sliding window (default: 50 packets)
        
        Returns:
            tuple: (best_window, stats_dict) or (None, None) if not found
        """
        if self._packet_count < window_size:
            return None, None
        
        # Analyze sliding windows - store only variance and start index to save memory
        window_results = []
        
        for i in range(0, self._packet_count - window_size, step):
            # Read window from file
            window = self._read_window(i, window_size)
            if len(window) < window_size:
                continue
            
            # Calculate spatial turbulence for this window
            turbulences = []
            for packet_mags in window:
                # Extract magnitudes for current band
                band_mags = [packet_mags[sc] for sc in current_band if sc < len(packet_mags)]
                if band_mags:
                    # Spatial turbulence = std of subcarrier magnitudes
                    mean_mag = sum(band_mags) / len(band_mags)
                    variance = sum((m - mean_mag) ** 2 for m in band_mags) / len(band_mags)
                    std = math.sqrt(variance) if variance > 0 else 0.0
                    turbulences.append(std)
            
            # Calculate variance of turbulence (moving variance)
            if turbulences:
                turb_variance = self._calculate_variance_two_pass(turbulences)
                # Store only start index and variance to save memory
                window_results.append((i, turb_variance))
            
            # FIX: Move inside loop - free memory after EACH window iteration
            del window
            del turbulences
            if i % 200 == 0:  # GC every 4 iterations (step=50, so 200/50=4)
                gc.collect()
        
        if not window_results:
            return None, None
        
        # Calculate percentile threshold (adaptive!)
        variances = [w[1] for w in window_results]
        p_threshold = self._percentile(variances, self.percentile)
        
        # Find windows below percentile
        baseline_candidates = [w for w in window_results if w[1] <= p_threshold]
        
        if not baseline_candidates:
            return None, None
        
        # Use window with minimum variance
        best_result = min(baseline_candidates, key=lambda x: x[1])
        best_start = best_result[0]
        
        # Re-read the best window from file
        best_window = self._read_window(best_start, window_size)
        
        # Return window and stats for logging
        stats = {
            'variance': best_result[1],
            'threshold': p_threshold,
            'windows_analyzed': len(window_results),
            'baseline_candidates': len(baseline_candidates),
            'start_idx': best_start
        }
        
        return best_window, stats
    
    def _percentile(self, values, p):
        """Calculate percentile (simple implementation)"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        k = (n - 1) * p / 100.0
        f = int(k)
        c = f + 1
        
        if c >= n:
            return sorted_values[-1]
        
        # Linear interpolation
        d0 = sorted_values[f] * (c - k)
        d1 = sorted_values[c] * (k - f)
        return d0 + d1
    
    def _calculate_nbvi_weighted(self, magnitudes):
        """
        Calculate NBVI Weighted α=0.3
        
        NBVI = 0.3 × (σ/μ²) + 0.7 × (σ/μ)
        
        Args:
            magnitudes: List of magnitude values for a subcarrier
        
        Returns:
            dict: {'nbvi': value, 'mean': μ, 'std': σ}
        """
        if not magnitudes:
            return {'nbvi': float('inf'), 'mean': 0.0, 'std': 0.0}
        
        mean = sum(magnitudes) / len(magnitudes)
        
        if mean < 1e-6:
            return {'nbvi': float('inf'), 'mean': mean, 'std': 0.0}
        
        # Calculate std
        variance = sum((m - mean) ** 2 for m in magnitudes) / len(magnitudes)
        std = math.sqrt(variance) if variance > 0 else 0.0
        
        # NBVI Weighted α=0.3
        cv = std / mean
        nbvi_energy = std / (mean * mean)
        nbvi_weighted = self.alpha * nbvi_energy + (1 - self.alpha) * cv
        
        return {
            'nbvi': nbvi_weighted,
            'mean': mean,
            'std': std
        }
    
    def _apply_noise_gate(self, subcarrier_metrics):
        """
        Apply Noise Gate: exclude weak subcarriers
        
        Args:
            subcarrier_metrics: List of dicts with 'subcarrier' and 'mean' keys
        
        Returns:
            list: Filtered metrics (strong subcarriers only)
        """
        # Extract NON-ZERO means only (skip invalid subcarriers with mean <= 1.0)
        valid_means = [m['mean'] for m in subcarrier_metrics if m['mean'] > 1.0]
        
        if not valid_means:
            print("NBVI: Noise Gate - no valid subcarriers found")
            return []
        
        threshold = self._percentile(valid_means, self.noise_gate_percentile)
        
        filtered = [m for m in subcarrier_metrics if m['mean'] >= threshold]
        
        #print(f"NBVI: Noise Gate excluded {len(subcarrier_metrics) - len(filtered)} weak subcarriers")
        #print(f"  Threshold: {threshold:.2f} (p{self.noise_gate_percentile}, valid: {len(valid_means)})")
        
        return filtered
    
    def _select_with_spacing(self, sorted_metrics, k=12):
        """
        Select subcarriers with spectral de-correlation
        
        Strategy:
        - Top 5: Always include (highest priority)
        - Remaining 7: Select with minimum spacing Δf≥min_spacing
        
        Args:
            sorted_metrics: Subcarriers sorted by NBVI (ascending)
            k: Number to select (default: 12)
        
        Returns:
            list: Selected subcarrier indices
        """
        # Phase 1: Top 5 absolute best
        selected = [m['subcarrier'] for m in sorted_metrics[:5]]
        
        # Phase 2: Remaining 7 with spacing
        for candidate in sorted_metrics[5:]:
            if len(selected) >= k:
                break
            
            sc = candidate['subcarrier']
            
            # Check spacing with already selected
            min_dist = min(abs(sc - s) for s in selected)
            
            if min_dist >= self.min_spacing:
                selected.append(sc)
        
        # If not enough, add best remaining regardless of spacing
        if len(selected) < k:
            for candidate in sorted_metrics:
                if len(selected) >= k:
                    break
                sc = candidate['subcarrier']
                if sc not in selected:
                    selected.append(sc)
        
        selected.sort()
        
        return selected
    
    def calibrate(self, current_band, window_size=None, step=None):
        """
        Calibrate using NBVI Weighted α=0.3 with percentile-based detection
        
        Args:
            current_band: Current subcarrier band (for baseline detection)
            window_size: Window size for baseline detection (default: from config)
            step: Step size for sliding window (default: from config)
        
        Returns:
            tuple: (selected_band, normalization_scale) or (None, 1.0) if failed
                - selected_band: List of 12 subcarrier indices
                - normalization_scale: Factor to normalize CSI amplitudes across devices
        """
        # Use config values only if not provided (lazy import to avoid MicroPython deps in tests)
        if window_size is None or step is None:
            import src.config as config
            if window_size is None:
                window_size = config.NBVI_WINDOW_SIZE
            if step is None:
                step = config.NBVI_WINDOW_STEP
        
        #print("\nNBVI: Starting calibration...")
        #print(f"  Packets collected: {self._packet_count}")
        #print(f"  Window size: {window_size} packets")
        #print(f"  Step size: {step} packets")
        
        # Prepare file for reading
        self._prepare_for_reading()
        
        # Step 1: Find baseline window using percentile
        baseline_window, baseline_stats = self._find_baseline_window_percentile(current_band, window_size, step)
        
        if baseline_window is None:
            print("NBVI: Failed to find baseline window")
            return None, 1.0
                
        # Step 2: Calculate NBVI for ALL subcarriers
        # Null subcarriers are detected dynamically based on signal strength
        all_metrics = []
        null_count = 0

        for sc in range(NUM_SUBCARRIERS):
            # Extract magnitude series for this subcarrier
            magnitudes = [packet_mags[sc] for packet_mags in baseline_window]
            
            # Calculate NBVI
            metrics = self._calculate_nbvi_weighted(magnitudes)
            metrics['subcarrier'] = sc
            
            # Auto-detect null subcarriers: if mean < threshold, mark as invalid
            # This catches hardware nulls (DC, guard bands) and environment-specific issues
            if metrics['mean'] < NULL_SUBCARRIER_THRESHOLD:
                metrics['nbvi'] = float('inf')
                null_count += 1
            
            all_metrics.append(metrics)
        
        print(f"NBVI: Auto-detected {null_count} null subcarriers (threshold: {NULL_SUBCARRIER_THRESHOLD})")
        
        # Step 3: Apply Noise Gate
        filtered_metrics = self._apply_noise_gate(all_metrics)
        
        if len(filtered_metrics) < 12:
            print(f"NBVI: Not enough subcarriers after Noise Gate ({len(filtered_metrics)} < 12)")
            return None, 1.0
        
        # Step 4: Sort by NBVI (ascending - lower is better)
        sorted_metrics = sorted(filtered_metrics, key=lambda x: x['nbvi'])
        
        # Step 5: Select with spectral spacing
        selected_band = self._select_with_spacing(sorted_metrics, k=12)
        
        if len(selected_band) != 12:
            print(f"NBVI: Invalid band size ({len(selected_band)} != 12)")
            return None, 1.0
        
        # Calculate average metrics for selected band
        selected_metrics = [m for m in all_metrics if m['subcarrier'] in selected_band]
        avg_nbvi = sum(m['nbvi'] for m in selected_metrics) / len(selected_metrics)
        avg_mean = sum(m['mean'] for m in selected_metrics) / len(selected_metrics)
        
        # Calculate normalization scale to bring all devices to a common scale
        # This compensates for different CSI amplitude ranges across ESP32 variants
        if self.normalization_enabled and avg_mean > 1.0:
            normalization_scale = self.normalization_target / avg_mean
            # Clamp to reasonable range
            normalization_scale = max(0.1, min(10.0, normalization_scale))
        else:
            normalization_scale = 1.0
        
        print(f"NBVI: Calibration successful")
        print(f"  Band: {selected_band}")
        print(f"  Avg NBVI: {avg_nbvi:.6f}")
        print(f"  Avg magnitude: {avg_mean:.2f}")
        if self.normalization_enabled:
            print(f"  Normalization scale: {normalization_scale:.3f} (target: {self.normalization_target:.0f})")
        else:
            print(f"  Normalization: disabled (scale: 1.0)")
        
        return selected_band, normalization_scale
    
    def free_buffer(self):
        """
        Free resources after calibration is complete.
        Closes file and removes temporary buffer file.
        """
        if self._file:
            self._file.close()
            self._file = None
        
        # Remove buffer file
        try:
            os.remove(BUFFER_FILE)
        except OSError:
            pass
        
        gc.collect()
