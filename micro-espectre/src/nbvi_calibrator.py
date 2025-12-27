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

# Normalization target: if baseline > 0.25, attenuate to reach this value
# If baseline <= 0.25, no scaling is applied (prevents over-amplification)
NORMALIZATION_BASELINE_TARGET = 0.25

# Threshold for null subcarrier detection (mean amplitude below this = null)
# This is environment-aware: works with any chip and adapts to local RF conditions
NULL_SUBCARRIER_THRESHOLD = 1.0

# OFDM 20MHz guard band limits - these subcarriers should always be excluded
# [0-5] and [59-63] are guard bands, [32] is DC null
GUARD_BAND_LOW = 11   # First valid subcarrier (conservative, excludes edge noise)
GUARD_BAND_HIGH = 52  # Last valid subcarrier (conservative, excludes edge noise)
DC_SUBCARRIER = 32    # DC null (always excluded)


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
    using NBVI Weighted α=0.5 algorithm with percentile-based detection.
    
    Uses file-based storage to avoid RAM limitations. Magnitudes stored as
    uint8 (max CSI magnitude ~181 fits in 1 byte). This allows collecting
    thousands of packets without memory issues.
    
    Normalization is always enabled: attenuates baseline if > 0.25, otherwise
    no scaling is applied to prevent over-amplification of weak signals.
    """
    
    def __init__(self, buffer_size=700, percentile=10, alpha=0.5, min_spacing=1, 
                 noise_gate_percentile=25, chip_type=None):
        """
        Initialize NBVI calibrator
        
        Args:
            buffer_size: Number of packets to collect (default: 700)
            percentile: Percentile for baseline detection (default: 10)
            alpha: NBVI weighting factor (default: 0.5)
            min_spacing: Minimum spacing between subcarriers (default: 1)
            noise_gate_percentile: Percentile for noise gate (default: 25)
            chip_type: Ignored (kept for backward compatibility)
        """
        self.buffer_size = buffer_size
        self.percentile = percentile
        self.alpha = alpha
        self.min_spacing = min_spacing
        self.noise_gate_percentile = noise_gate_percentile
        self.chip_type = chip_type  # Kept for backward compatibility
        self._packet_count = 0
        self._file = None
        self._baseline_variance = 1.0  # Stored for normalization calculation
        
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
    
    def _calculate_baseline_variance(self, baseline_window, selected_band):
        """
        Recalculate baseline variance using the SELECTED subcarriers.
        
        This is called after NBVI selection to get accurate variance for the actual band used.
        The initial baseline_variance was calculated with current_band (default [11-22]),
        but NBVI may select different subcarriers, so we need to recalculate.
        
        Args:
            baseline_window: List of packet magnitudes from the baseline window
            selected_band: List of selected subcarrier indices from NBVI
        
        Returns:
            float: Recalculated baseline variance for the selected band
        """
        if not baseline_window or not selected_band:
            return 0.0
        
        # Calculate spatial turbulence for each packet using the SELECTED subcarriers
        turbulences = []
        for packet_mags in baseline_window:
            # Extract magnitudes for selected band
            band_mags = [packet_mags[sc] for sc in selected_band if sc < len(packet_mags)]
            if band_mags:
                # Spatial turbulence = std of subcarrier magnitudes
                mean_mag = sum(band_mags) / len(band_mags)
                variance = sum((m - mean_mag) ** 2 for m in band_mags) / len(band_mags)
                std = math.sqrt(variance) if variance > 0 else 0.0
                turbulences.append(std)
        
        # Calculate variance of turbulence (moving variance)
        if turbulences:
            return self._calculate_variance_two_pass(turbulences)
        
        return 0.0
    
    def _find_candidate_windows(self, current_band, window_size=200, step=50):
        """
        Find all candidate baseline windows using percentile-based detection.
        
        NO absolute threshold - adapts automatically to environment.
        Returns all windows below percentile threshold, sorted by variance.
        
        Args:
            current_band: Current subcarrier band (for variance calculation)
            window_size: Size of analysis window (default: 200 packets)
            step: Step size for sliding window (default: 50 packets)
        
        Returns:
            list: List of (start_idx, variance) tuples, sorted by variance (ascending)
                  Empty list if no candidates found
        """
        if self._packet_count < window_size:
            return []
        
        # Analyze sliding windows - store only variance and start index to save memory
        window_results = []
        
        for i in range(0, self._packet_count - window_size + 1, step):
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
            return []
        
        # Calculate percentile threshold (adaptive!)
        variances = [w[1] for w in window_results]
        p_threshold = self._percentile(variances, self.percentile)
        
        # Find windows below percentile and sort by variance (best first)
        candidates = [w for w in window_results if w[1] <= p_threshold]
        candidates.sort(key=lambda x: x[1])  # Sort by variance, ascending
        
        return candidates
    
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
        Calculate NBVI Weighted (configurable alpha, default 0.5)
        
        NBVI = α × (σ/μ²) + (1-α) × (σ/μ)
        
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
        
        # NBVI Weighted (configurable alpha)
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
    
    def _validate_subcarriers(self, band, mvs_window_size=50, mvs_threshold=1.0):
        """
        Validate subcarriers by running MVS on entire buffer.
        
        The calibration buffer contains ONLY baseline data (quiet room), so any
        motion detection is a false positive.
        
        Args:
            band: List of subcarrier indices to validate
            mvs_window_size: MVS window size (default: 50, production value)
            mvs_threshold: MVS threshold (default: 1.0, production value)
        
        Returns:
            float: FP rate (0.0 = perfect, 1.0 = all packets detected as motion)
        """
        if self._packet_count < mvs_window_size:
            return 0.0
        
        # Build turbulence buffer for MVS
        turbulence_buffer = [0.0] * mvs_window_size
        motion_count = 0
        total_packets = 0
        
        # Process entire buffer packet by packet
        for pkt_idx in range(self._packet_count):
            # Read single packet
            packet_mags = self._read_packet(pkt_idx)
            if packet_mags is None:
                continue
            
            # Calculate spatial turbulence with candidate subcarriers
            band_mags = [packet_mags[sc] for sc in band if sc < len(packet_mags)]
            if not band_mags:
                continue
            
            mean_mag = sum(band_mags) / len(band_mags)
            variance = sum((m - mean_mag) ** 2 for m in band_mags) / len(band_mags)
            turbulence = math.sqrt(variance) if variance > 0 else 0.0
            
            # Shift buffer and add new value (like real MVS does)
            turbulence_buffer.pop(0)
            turbulence_buffer.append(turbulence)
            
            # Skip warmup (need full window before calculating variance)
            if pkt_idx < mvs_window_size:
                continue
            
            # Calculate variance (MVS) - this is the moving variance
            mv_variance = self._calculate_variance_two_pass(turbulence_buffer)
            
            # Check if this would trigger motion detection
            if mv_variance > mvs_threshold:
                motion_count += 1
            total_packets += 1
        
        fp_rate = motion_count / total_packets if total_packets > 0 else 0.0
        return fp_rate
    
    def calibrate(self, current_band, window_size=None, step=None):
        """
        Calibrate using NBVI Weighted with percentile-based detection.
        
        Tries all candidate windows and selects the one with minimum FP rate
        (matching C++ production behavior).
        
        Args:
            current_band: Current subcarrier band (for baseline detection)
            window_size: Window size for baseline detection (default: 200)
            step: Step size for sliding window (default: 50)
        
        Returns:
            tuple: (selected_band, normalization_scale) or (None, 1.0) if failed
                - selected_band: List of 12 subcarrier indices
                - normalization_scale: Factor to normalize CSI amplitudes across devices
        """
        # Production defaults matching C++
        if window_size is None:
            window_size = 200
        if step is None:
            step = 50
        
        # Prepare file for reading
        self._prepare_for_reading()
        
        # Step 1: Find all candidate baseline windows
        candidates = self._find_candidate_windows(current_band, window_size, step)
        
        if not candidates:
            print("NBVI: Failed to find candidate windows")
            return None, 1.0
        
        print(f"NBVI: Found {len(candidates)} candidate windows")
        
        # Step 2: Try all candidate windows and select the one with minimum FP rate
        best_fp_rate = 1.0
        best_band = None
        best_baseline_window = None
        best_avg_nbvi = 0.0
        best_avg_mean = 0.0
        best_window_idx = 0
        
        for idx, (start_idx, window_variance) in enumerate(candidates):
            # Read this window
            baseline_window = self._read_window(start_idx, window_size)
            if len(baseline_window) < window_size:
                continue
            
            # Calculate NBVI for ALL subcarriers using this window
            all_metrics = []
            null_count = 0
            
            for sc in range(NUM_SUBCARRIERS):
                # Extract magnitude series for this subcarrier
                magnitudes = [packet_mags[sc] for packet_mags in baseline_window]
                
                # Calculate NBVI
                metrics = self._calculate_nbvi_weighted(magnitudes)
                metrics['subcarrier'] = sc
                
                # Exclude guard bands and DC subcarrier
                if sc < GUARD_BAND_LOW or sc > GUARD_BAND_HIGH or sc == DC_SUBCARRIER:
                    metrics['nbvi'] = float('inf')
                    null_count += 1
                # Auto-detect weak subcarriers
                elif metrics['mean'] < NULL_SUBCARRIER_THRESHOLD:
                    metrics['nbvi'] = float('inf')
                    null_count += 1
                
                all_metrics.append(metrics)
            
            # Apply Noise Gate
            filtered_metrics = self._apply_noise_gate(all_metrics)
            
            if len(filtered_metrics) < 12:
                continue
            
            # Sort by NBVI (ascending - lower is better)
            sorted_metrics = sorted(filtered_metrics, key=lambda x: x['nbvi'])
            
            # Select with spectral spacing
            candidate_band = self._select_with_spacing(sorted_metrics, k=12)
            
            if len(candidate_band) != 12:
                continue
            
            # VALIDATE: run MVS on entire buffer with selected subcarriers
            fp_rate = self._validate_subcarriers(candidate_band)
            
            # Track best result (minimum FP rate)
            if fp_rate < best_fp_rate:
                best_fp_rate = fp_rate
                best_band = candidate_band
                best_baseline_window = baseline_window
                best_window_idx = idx
                
                # Calculate average metrics for selected band
                selected_metrics = [m for m in all_metrics if m['subcarrier'] in candidate_band]
                best_avg_nbvi = sum(m['nbvi'] for m in selected_metrics) / len(selected_metrics)
                best_avg_mean = sum(m['mean'] for m in selected_metrics) / len(selected_metrics)
            
            # Clean up memory
            del baseline_window
            del all_metrics
            gc.collect()
        
        if best_band is None:
            # Fallback: keep default subcarriers but still calculate normalization
            # Use the first (best) candidate window for baseline variance
            print("NBVI: All candidate windows failed - using default subcarriers with normalization fallback")
            
            # Read first candidate window for baseline variance calculation
            first_start_idx, _ = candidates[0]
            fallback_window = self._read_window(first_start_idx, window_size)
            
            # Calculate baseline variance using current (default) subcarriers
            baseline_variance = self._calculate_baseline_variance(fallback_window, list(current_band))
            if baseline_variance < 0.01:
                baseline_variance = 1.0  # Fallback
            self._baseline_variance = baseline_variance
            
            # Calculate normalization scale
            if baseline_variance > NORMALIZATION_BASELINE_TARGET:
                normalization_scale = NORMALIZATION_BASELINE_TARGET / baseline_variance
                if normalization_scale < 0.1:
                    normalization_scale = 0.1
            else:
                normalization_scale = 1.0
            
            print(f"NBVI: Fallback calibration: default subcarriers with normalization")
            print(f"  Baseline variance: {baseline_variance:.4f}")
            print(f"  Normalization scale: {normalization_scale:.4f}")
            
            # Return default subcarriers (as list) with calculated normalization
            return list(current_band), normalization_scale
        
        print(f"NBVI: Selected window {best_window_idx + 1}/{len(candidates)} with FP rate {best_fp_rate * 100:.1f}%")
        
        # Step 3: Calculate baseline variance using the SELECTED subcarriers
        baseline_variance = self._calculate_baseline_variance(best_baseline_window, best_band)
        if baseline_variance < 0.01:
            baseline_variance = 1.0  # Fallback
        self._baseline_variance = baseline_variance
        
        # Normalize only if baseline variance is ABOVE target (0.25)
        if baseline_variance > NORMALIZATION_BASELINE_TARGET:
            normalization_scale = NORMALIZATION_BASELINE_TARGET / baseline_variance
            if normalization_scale < 0.1:
                normalization_scale = 0.1
        else:
            normalization_scale = 1.0
        
        print(f"NBVI: Calibration successful")
        print(f"  Band: {best_band}")
        print(f"  Avg NBVI: {best_avg_nbvi:.6f}")
        print(f"  Avg magnitude: {best_avg_mean:.2f}")
        print(f"  Baseline variance: {baseline_variance:.4f}")
        print(f"  Normalization scale: {normalization_scale:.4f}")
        print(f"  Validated FP rate: {best_fp_rate * 100:.1f}%")
        
        return best_band, normalization_scale
    
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
