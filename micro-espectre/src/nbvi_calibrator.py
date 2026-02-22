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

try:
    from src.config import (
        NUM_SUBCARRIERS, EXPECTED_CSI_LEN,
        GUARD_BAND_LOW, GUARD_BAND_HIGH, DC_SUBCARRIER, BAND_SIZE,
        SEG_WINDOW_SIZE, CALIBRATION_BUFFER_SIZE
    )
    from src.utils import (
        to_signed_int8, calculate_percentile,
        calculate_variance, calculate_std, calculate_moving_variance
    )
except ImportError:
    from config import (
        NUM_SUBCARRIERS, EXPECTED_CSI_LEN,
        GUARD_BAND_LOW, GUARD_BAND_HIGH, DC_SUBCARRIER, BAND_SIZE,
        SEG_WINDOW_SIZE, CALIBRATION_BUFFER_SIZE
    )
    from utils import (
        to_signed_int8, calculate_percentile,
        calculate_variance, calculate_std, calculate_moving_variance
    )

# Constants
BUFFER_FILE = '/nbvi_buffer.bin'

# Threshold for null subcarrier detection (mean amplitude below this = null)
NULL_SUBCARRIER_THRESHOLD = 1.0

# MVS parameters for validation (uses SEG_WINDOW_SIZE from config.py)
MVS_THRESHOLD = 1.0


def cleanup_buffer_file():
    """Remove any leftover buffer file from previous interrupted runs."""
    try:
        os.remove(BUFFER_FILE)
        print("NBVI: Cleaned up leftover buffer file")
    except OSError:
        pass


class NBVICalibrator:
    """
    Automatic NBVI calibrator with percentile-based baseline detection
    
    Collects CSI packets at boot and automatically selects optimal subcarriers
    using NBVI Weighted alpha=0.5 algorithm with percentile-based detection.
    
    Uses file-based storage to avoid RAM limitations. Magnitudes stored as
    uint8 (max CSI magnitude ~181 fits in 1 byte).
    
    After subcarrier selection, calculates adaptive threshold using Pxx * factor.
    """
    
    def __init__(self, buffer_size=None, mvs_window_size=None,
                 percentile=10, alpha=0.5, min_spacing=1, noise_gate_percentile=25):
        """
        Initialize NBVI calibrator
        
        Args:
            buffer_size: Number of packets to collect (default: CALIBRATION_BUFFER_SIZE from config)
            mvs_window_size: MVS window size for validation (default: SEG_WINDOW_SIZE from config)
            percentile: Percentile for baseline window detection (default: 10)
            alpha: NBVI weighting factor (default: 0.5)
            min_spacing: Minimum spacing between subcarriers (default: 1)
            noise_gate_percentile: Percentile for noise gate (default: 25)
        """
        self.buffer_size = buffer_size if buffer_size is not None else CALIBRATION_BUFFER_SIZE
        self._buffer_file = BUFFER_FILE
        self._packet_count = 0
        self._filtered_count = 0
        self._file = None
        self._initialized = False
        
        # Batch write buffer to reduce flash I/O overhead (750 writes → 8 writes)
        self._write_batch_size = 100
        self._write_buf = bytearray(self._write_batch_size * NUM_SUBCARRIERS)
        self._write_buf_idx = 0
        
        # Remove old buffer file if exists
        try:
            os.remove(BUFFER_FILE)
        except OSError:
            pass
        
        # Open file for writing
        self._file = open(BUFFER_FILE, 'wb')
        
        # NBVI parameters
        self.mvs_window_size = mvs_window_size if mvs_window_size is not None else SEG_WINDOW_SIZE
        self.percentile = percentile
        self.alpha = alpha
        self.min_spacing = min_spacing
        self.noise_gate_percentile = noise_gate_percentile
    
    # ========================================================================
    # Buffer management
    # ========================================================================
    
    def _prepare_for_reading(self):
        """Flush remaining buffer, close write mode and reopen for reading."""
        if self._file:
            # Flush any remaining packets in batch buffer
            if self._write_buf_idx > 0:
                remaining = self._write_buf_idx * NUM_SUBCARRIERS
                self._file.write(memoryview(self._write_buf)[:remaining])
                self._write_buf_idx = 0
            self._file.flush()
            self._file.close()
        # Free write buffer — no longer needed after collection phase
        self._write_buf = None
        gc.collect()
        self._file = open(self._buffer_file, 'rb')
    
    def free_buffer(self):
        """Free resources after calibration is complete."""
        if self._file:
            self._file.close()
            self._file = None
        
        # Free batch buffer
        self._write_buf = None
        
        try:
            os.remove(self._buffer_file)
        except OSError:
            pass
    
    def get_packet_count(self):
        """Get the number of packets currently in the buffer."""
        return self._packet_count
    
    def is_buffer_full(self):
        """Check if the buffer has collected enough packets."""
        return self._packet_count >= self.buffer_size
    
    # ========================================================================
    # Packet collection
    # ========================================================================
        
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
        
        # STBC packets (256 bytes) are truncated upstream before reaching here.
        # See GitHub issue #76, espressif/esp-csi#238 for details.
        if len(csi_data) != EXPECTED_CSI_LEN:
            self._filtered_count += 1
            if self._filtered_count % 50 == 1:
                print(f'[WARN] Filtered {self._filtered_count} packets with wrong SC count (got {len(csi_data)} bytes)')
            return self._packet_count
        
        # Initialize on first packet
        if not self._initialized:
            self._initialized = True
            print(f'NBVI: HT20 mode, {NUM_SUBCARRIERS} SC, guard [{GUARD_BAND_LOW}-{GUARD_BAND_HIGH}], DC={DC_SUBCARRIER}')
        
        # Extract magnitudes into batch buffer (avoids per-packet flash write)
        # Guard band and DC subcarriers are zeroed without computing sqrt —
        # they are excluded from NBVI selection anyway (marked inf in calibrate()).
        # Cache math.sqrt locally to avoid 42 global+attr lookups per packet.
        # I*I integer arithmetic avoids float() conversions (exact for I ∈ [-127,127]).
        _sqrt = math.sqrt
        buf_offset = self._write_buf_idx * NUM_SUBCARRIERS
        csi_len = len(csi_data)
        for sc in range(NUM_SUBCARRIERS):
            if sc < GUARD_BAND_LOW or sc > GUARD_BAND_HIGH or sc == DC_SUBCARRIER:
                self._write_buf[buf_offset + sc] = 0
                continue
            
            q_idx = sc * 2 + 1
            
            if q_idx < csi_len:
                # Espressif CSI format: [Imaginary, Real, ...] per subcarrier
                Q = to_signed_int8(csi_data[sc * 2])   # Imaginary first
                I = to_signed_int8(csi_data[q_idx])     # Real second
                # I*I avoids float() conversion: I ∈ [-127,127] so I*I ≤ 32258,
                # exact in float64, same result as float(I)*float(I).
                mag = int(_sqrt(I*I + Q*Q))
                self._write_buf[buf_offset + sc] = min(mag, 255)
            else:
                self._write_buf[buf_offset + sc] = 0
        
        self._write_buf_idx += 1
        self._packet_count += 1
        
        # Batch write when buffer full (reduces flash writes from 750 to ~8)
        if self._write_buf_idx >= self._write_batch_size:
            self._file.write(self._write_buf)
            self._write_buf_idx = 0
        
        return self._packet_count
    
    # ========================================================================
    # File I/O helpers
    # ========================================================================
    
    def _read_packet(self, packet_idx):
        """Read a single packet from file"""
        self._file.seek(packet_idx * NUM_SUBCARRIERS)
        data = self._file.read(NUM_SUBCARRIERS)
        return list(data) if data else None
    
    def _packet_turbulence(self, data, band):
        """Calculate spatial turbulence (std of band magnitudes) from raw packet bytes."""
        band_mags = [data[sc] for sc in band if sc < len(data)]
        if not band_mags:
            return 0.0
        mean_mag = sum(band_mags) / len(band_mags)
        variance = sum((m - mean_mag) ** 2 for m in band_mags) / len(band_mags)
        return math.sqrt(variance) if variance > 0 else 0.0
    
    # ========================================================================
    # Calibration algorithm
    # ========================================================================
    
    def _find_candidate_windows(self, current_band, window_size=200, step=50):
        """
        Find all candidate baseline windows using percentile-based detection.
        Streams packets from file one at a time to avoid large memory allocations.
        
        NO absolute threshold - adapts automatically to environment.
        """
        if self._packet_count < window_size:
            return []
        
        window_results = []
        
        for i in range(0, self._packet_count - window_size + 1, step):
            # Two-pass streaming variance of turbulences (identical to calculate_variance)
            # Pass 1: mean
            sum_turb = 0.0
            count = 0
            self._file.seek(i * NUM_SUBCARRIERS)
            for _ in range(window_size):
                data = self._file.read(NUM_SUBCARRIERS)
                if not data or len(data) < NUM_SUBCARRIERS:
                    break
                sum_turb += self._packet_turbulence(data, current_band)
                count += 1
            
            if count == 0:
                continue
            mean_turb = sum_turb / count
            
            # Pass 2: variance
            sum_sq = 0.0
            self._file.seek(i * NUM_SUBCARRIERS)
            for _ in range(window_size):
                data = self._file.read(NUM_SUBCARRIERS)
                if not data or len(data) < NUM_SUBCARRIERS:
                    break
                diff = self._packet_turbulence(data, current_band) - mean_turb
                sum_sq += diff * diff
            
            window_results.append((i, sum_sq / count))
            
            if i % 200 == 0:
                gc.collect()
        
        if not window_results:
            return []
        
        variances = [w[1] for w in window_results]
        p_threshold = calculate_percentile(variances, self.percentile)
        
        candidates = [w for w in window_results if w[1] <= p_threshold]
        candidates.sort(key=lambda x: x[1])
        
        return candidates
    
    def _calculate_nbvi_from_stats(self, mean, std):
        """
        Calculate NBVI Weighted from pre-computed mean and std.
        
        NBVI = alpha * (std/mean^2) + (1-alpha) * (std/mean)
        """
        if mean < 1e-6:
            return {'nbvi': float('inf'), 'mean': mean, 'std': std}
        
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
        if self._packet_count < self.mvs_window_size:
            return 0.0, []
        
        turbulence_buffer = [0.0] * self.mvs_window_size
        motion_count = 0
        total_packets = 0
        # Subsample mv_values at 1:5 for the adaptive threshold (P95).
        # The 750-packet buffer is needed for band selection quality, but P95
        # is statistically stable with ~140 samples. A contiguous list of 700
        # floats (2700 bytes) exceeds the available heap on ESP32-C3 after the
        # NBVI streaming phase, while 140 floats (560 bytes) fits comfortably.
        MV_SUBSAMPLE = 5
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
            
            if pkt_idx < self.mvs_window_size:
                continue
            
            mv_variance = calculate_variance(turbulence_buffer)
            if total_packets % MV_SUBSAMPLE == 0:
                mv_values.append(mv_variance)
            
            if mv_variance > MVS_THRESHOLD:
                motion_count += 1
            total_packets += 1
        
        fp_rate = motion_count / total_packets if total_packets > 0 else 0.0
        return fp_rate, mv_values
    
    def calibrate(self, hint_band=None):
        """
        Calibrate using NBVI Weighted with percentile-based detection.
        
        Args:
            hint_band: Optional band to use for candidate window search.
                       If provided, uses this band to calculate turbulence
                       when finding baseline candidate windows.
                       Matches C++ start_calibration(current_band) behavior.
        
        Returns:
            tuple: (selected_band, mv_values) or (None, []) if failed
        """
        window_size = 200
        step = 50
        
        if self._packet_count < self.mvs_window_size + 10:
            print("NBVI: Not enough packets for calibration")
            return None, []
        
        self._prepare_for_reading()
        
        # Use hint_band if provided, otherwise use default band for finding candidate windows
        # This matches C++ behavior where start_calibration() receives current_band as hint
        if hint_band is not None:
            search_band = hint_band
        else:
            search_band = list(range(GUARD_BAND_LOW, GUARD_BAND_LOW + BAND_SIZE))
        candidates = self._find_candidate_windows(search_band, window_size, step)
        
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
            # Two-pass streaming NBVI per subcarrier (~1 KB instead of ~62 KB)
            # Pass 1: accumulate sum per subcarrier for mean
            sum_sc = [0.0] * NUM_SUBCARRIERS
            count = 0
            self._file.seek(start_idx * NUM_SUBCARRIERS)
            for _ in range(window_size):
                data = self._file.read(NUM_SUBCARRIERS)
                if not data or len(data) < NUM_SUBCARRIERS:
                    break
                for sc in range(NUM_SUBCARRIERS):
                    sum_sc[sc] += data[sc]
                count += 1
            
            if count == 0:
                continue
            # Divide in-place to avoid allocating 64 new float objects (~1 KB)
            for i in range(NUM_SUBCARRIERS):
                sum_sc[i] /= count
            mean_sc = sum_sc
            
            # Pass 2: accumulate sum of squared differences for std
            sum_sq_sc = [0.0] * NUM_SUBCARRIERS
            self._file.seek(start_idx * NUM_SUBCARRIERS)
            for _ in range(window_size):
                data = self._file.read(NUM_SUBCARRIERS)
                if not data or len(data) < NUM_SUBCARRIERS:
                    break
                for sc in range(NUM_SUBCARRIERS):
                    diff = data[sc] - mean_sc[sc]
                    sum_sq_sc[sc] += diff * diff
            
            # Build metrics from streaming stats
            all_metrics = []
            for sc in range(NUM_SUBCARRIERS):
                std = math.sqrt(sum_sq_sc[sc] / count) if sum_sq_sc[sc] > 0 else 0.0
                metrics = self._calculate_nbvi_from_stats(mean_sc[sc], std)
                metrics['subcarrier'] = sc
                
                if sc < GUARD_BAND_LOW or sc > GUARD_BAND_HIGH or sc == DC_SUBCARRIER:
                    metrics['nbvi'] = float('inf')
                elif metrics['mean'] < NULL_SUBCARRIER_THRESHOLD:
                    metrics['nbvi'] = float('inf')
                
                all_metrics.append(metrics)
            
            del sum_sc
            del sum_sq_sc
            del mean_sc
            
            filtered_metrics = self._apply_noise_gate(all_metrics)
            
            if len(filtered_metrics) < BAND_SIZE:
                continue
            
            sorted_metrics = sorted(filtered_metrics, key=lambda x: x['nbvi'])
            candidate_band = self._select_with_spacing(sorted_metrics, k=BAND_SIZE)
            
            if len(candidate_band) != BAND_SIZE:
                continue
            
            # Save reporting stats before freeing all_metrics
            selected_metrics = [m for m in all_metrics if m['subcarrier'] in candidate_band]
            avg_nbvi = sum(m['nbvi'] for m in selected_metrics) / len(selected_metrics)
            avg_mean = sum(m['mean'] for m in selected_metrics) / len(selected_metrics)
            del selected_metrics
            del all_metrics
            gc.collect()
            
            fp_rate, mv_values = self._validate_subcarriers(candidate_band)
            
            if fp_rate < best_fp_rate:
                best_fp_rate = fp_rate
                best_band = candidate_band
                best_mv_values = mv_values
                best_window_idx = idx
                best_avg_nbvi = avg_nbvi
                best_avg_mean = avg_mean
        
        if best_band is None:
            print("NBVI: All candidate windows failed - using default subcarriers")
            
            # Run validation on search_band (hint_band or default) to get MV values
            _, mv_values = self._validate_subcarriers(search_band)
            
            print(f"NBVI: Fallback to default band")
            
            if self._filtered_count > 0:
                print(f"  Filtered: {self._filtered_count} packets (wrong SC count)")
            
            return search_band, mv_values
        
        print(f"NBVI: Selected window {best_window_idx + 1}/{len(candidates)} with FP rate {best_fp_rate * 100:.1f}%")
        
        print(f"NBVI: Band selection successful")
        print(f"  Band: {best_band}")
        print(f"  Avg NBVI: {best_avg_nbvi:.6f}")
        print(f"  Avg magnitude: {best_avg_mean:.2f}")
        print(f"  Est. FP rate: {best_fp_rate * 100:.1f}%")
        
        if self._filtered_count > 0:
            print(f"  Filtered: {self._filtered_count} packets (wrong SC count)")
        
        return best_band, best_mv_values
