"""
Band Calibrator - Automatic Subcarrier Band Selection

Selects optimal 12-subcarrier band for motion detection by minimizing
the P95 of moving variance during baseline. This directly optimizes
for low false positive rate.

Algorithm:
1. Collect baseline CSI packets (quiet room)
2. For each candidate band of 12 consecutive subcarriers:
   - Calculate moving variance series
   - Compute P95 (95th percentile) of moving variance
3. Select the band with LOWEST P95 (furthest from threshold)

Key insight: If P95 < threshold, the band will have near-zero FP rate.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import math
import gc
import os

# Constants
BUFFER_FILE = '/band_buffer.bin'

# Normalization target: if baseline > target, attenuate to reach this value
# For 256 SC, baseline variance is naturally higher, so we use adaptive target
NORMALIZATION_BASELINE_TARGET_64 = 0.25
NORMALIZATION_BASELINE_TARGET_256 = 0.60

# Threshold for null subcarrier detection
NULL_SUBCARRIER_THRESHOLD = 1.0

# Band size (number of subcarriers to select)
BAND_SIZE = 12

# MVS parameters for P95 calculation
MVS_WINDOW_SIZE = 50
MVS_THRESHOLD = 1.0


def calculate_guard_bands(num_subcarriers):
    """
    Calculate guard band limits optimized for motion detection.
    
    Args:
        num_subcarriers: Total number of subcarriers (64, 128, or 256)
    
    Returns:
        tuple: (guard_band_low, guard_band_high, dc_low, dc_high)
    """
    if num_subcarriers == 64:
        # HT20: Conservative guard bands (tested, optimized)
        guard_band_low = 11
        guard_band_high = 52
        dc_low = 32
        dc_high = 32
    elif num_subcarriers == 128:
        # HT40: Optimized (validated on ESP32-S3)
        # Based on grid search: best bands [50-61], [1-12], [116-127]
        # Valid zones: [7-60] and [68-120]
        guard_band_low = 7
        guard_band_high = 120
        dc_low = 61
        dc_high = 67
    elif num_subcarriers == 256:
        # HE20: Optimized (validated on ESP32-C6)
        # Excludes: edges [0-19] and [236-255], DC zone [120-136]
        # Valid zones: [20-119] and [137-235] = 199 valid subcarriers
        # Empirically validated: P95=0.56 (vs 0.60 conservative), Recall=99.7%
        guard_band_low = 20
        guard_band_high = 235
        dc_low = 120
        dc_high = 136
    else:
        # Fallback
        guard_band_low = num_subcarriers // 10
        guard_band_high = num_subcarriers - 1 - (num_subcarriers // 10)
        dc_low = num_subcarriers // 2
        dc_high = num_subcarriers // 2
    
    return guard_band_low, guard_band_high, dc_low, dc_high


def get_normalization_target(num_subcarriers):
    """Get normalization target based on subcarrier count"""
    if num_subcarriers >= 256:
        return NORMALIZATION_BASELINE_TARGET_256
    return NORMALIZATION_BASELINE_TARGET_64


class BandCalibrator:
    """
    Automatic band calibrator using P95 moving variance optimization.
    
    Selects the optimal 12-subcarrier band by finding the one with
    lowest P95 of moving variance during baseline (minimizes FP rate).
    
    Uses file-based storage to avoid RAM limitations on ESP32.
    """
    
    def __init__(self, buffer_size=700, expected_subcarriers=None):
        """
        Initialize band calibrator.
        
        Args:
            buffer_size: Number of packets to collect (default: 700)
            expected_subcarriers: Expected subcarrier count (64, 128, 256).
                                  If set, packets with different SC count are filtered.
        """
        self.buffer_size = buffer_size
        self.expected_subcarriers = expected_subcarriers
        self._packet_count = 0
        self._filtered_count = 0
        self._file = None
        self._num_subcarriers = None
        self._guard_band_low = None
        self._guard_band_high = None
        self._dc_low = None
        self._dc_high = None
        
        # Remove old buffer file
        try:
            os.remove(BUFFER_FILE)
        except OSError:
            pass
        
        self._file = open(BUFFER_FILE, 'wb')
    
    def add_packet(self, csi_data):
        """
        Add CSI packet to calibration buffer.
        
        Args:
            csi_data: CSI data array (N bytes: N/2 subcarriers × 2 I/Q)
        
        Returns:
            int: Current buffer size (progress indicator)
        """
        if self._packet_count >= self.buffer_size:
            return self.buffer_size
        
        packet_sc = len(csi_data) // 2
        
        # Filter by expected subcarrier count
        if self.expected_subcarriers is not None and packet_sc != self.expected_subcarriers:
            self._filtered_count += 1
            return self._packet_count
        
        # Initialize on first packet
        if self._num_subcarriers is None:
            self._num_subcarriers = packet_sc
            self._guard_band_low, self._guard_band_high, self._dc_low, self._dc_high = \
                calculate_guard_bands(self._num_subcarriers)
            
            if self._dc_low == self._dc_high:
                print(f'Band: {self._num_subcarriers} SC, guard [{self._guard_band_low}-{self._guard_band_high}], DC={self._dc_low}')
            else:
                print(f'Band: {self._num_subcarriers} SC, guard [{self._guard_band_low}-{self._guard_band_high}], DC [{self._dc_low}-{self._dc_high}]')
        
        # Extract magnitudes
        magnitudes = bytearray(self._num_subcarriers)
        for sc in range(self._num_subcarriers):
            i_idx = sc * 2
            q_idx = sc * 2 + 1
            
            if q_idx < len(csi_data):
                I = int(csi_data[i_idx])
                Q = int(csi_data[q_idx])
                # Handle unsigned bytes from Python tests
                if I > 127:
                    I = I - 256
                if Q > 127:
                    Q = Q - 256
                mag = int(math.sqrt(I*I + Q*Q))
                magnitudes[sc] = min(mag, 255)
            else:
                magnitudes[sc] = 0
        
        self._file.write(magnitudes)
        self._packet_count += 1
        
        if self._packet_count % 100 == 0:
            self._file.flush()
        
        return self._packet_count
    
    def _prepare_for_reading(self):
        """Close write mode and reopen for reading"""
        if self._file:
            self._file.flush()
            self._file.close()
            gc.collect()
        self._file = open(BUFFER_FILE, 'rb')
    
    def _read_all_packets(self):
        """Read all packets from file"""
        self._file.seek(0)
        data = self._file.read()
        
        packets = []
        for i in range(0, len(data), self._num_subcarriers):
            packet = list(data[i:i+self._num_subcarriers])
            if len(packet) == self._num_subcarriers:
                packets.append(packet)
        
        return packets
    
    def _calculate_spatial_turbulence(self, packet_mags, band):
        """Calculate spatial turbulence for a band"""
        band_mags = [packet_mags[sc] for sc in band if sc < len(packet_mags)]
        if not band_mags:
            return 0.0
        
        mean_mag = sum(band_mags) / len(band_mags)
        variance = sum((m - mean_mag) ** 2 for m in band_mags) / len(band_mags)
        return math.sqrt(variance) if variance > 0 else 0.0
    
    def _calculate_moving_variance(self, values, window_size=50):
        """Calculate moving variance series"""
        if len(values) < window_size:
            return []
        
        variances = []
        for i in range(window_size, len(values)):
            window = values[i-window_size:i]
            mean = sum(window) / len(window)
            var = sum((x - mean) ** 2 for x in window) / len(window)
            variances.append(var)
        
        return variances
    
    def _calculate_p95(self, values):
        """Calculate 95th percentile"""
        if not values:
            return float('inf')
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        k = int((n - 1) * 0.95)
        
        if k >= n - 1:
            return sorted_values[-1]
        
        # Linear interpolation
        frac = (n - 1) * 0.95 - k
        return sorted_values[k] * (1 - frac) + sorted_values[k + 1] * frac
    
    def _get_candidate_bands(self):
        """
        Generate valid candidate bands of 12 consecutive subcarriers.
        
        Avoids guard bands and DC zone.
        Uses step=2 for 256 SC to reduce candidates from 134 to 67 (performance optimization).
        """
        candidates = []
        
        # Use step to reduce candidate bands for 256 SC (WiFi 6)
        # step=1 for 64 SC (~40 bands), step=2 for 256 SC (~67 bands instead of 134)
        step = 2 if self._num_subcarriers >= 256 else 1
        
        # Zone before DC
        for start in range(self._guard_band_low, self._dc_low - BAND_SIZE + 1, step):
            band = list(range(start, start + BAND_SIZE))
            # Verify all subcarriers are valid
            if all(self._guard_band_low <= sc <= self._guard_band_high and 
                   not (self._dc_low <= sc <= self._dc_high) for sc in band):
                candidates.append(band)
        
        # Zone after DC
        for start in range(self._dc_high + 1, self._guard_band_high - BAND_SIZE + 2, step):
            band = list(range(start, start + BAND_SIZE))
            if all(self._guard_band_low <= sc <= self._guard_band_high and 
                   not (self._dc_low <= sc <= self._dc_high) for sc in band):
                candidates.append(band)
        
        return candidates
    
    def _evaluate_band(self, packets, band):
        """
        Evaluate a band by computing P95 of moving variance.
        
        Returns:
            dict with p95, mean_mv, fp_estimate
        """
        # Calculate turbulence series
        turbulences = [self._calculate_spatial_turbulence(pkt, band) for pkt in packets]
        
        # Calculate moving variance series
        mv_series = self._calculate_moving_variance(turbulences, MVS_WINDOW_SIZE)
        
        if not mv_series:
            return {'p95': float('inf'), 'mean_mv': 0, 'fp_estimate': 1.0}
        
        p95 = self._calculate_p95(mv_series)
        mean_mv = sum(mv_series) / len(mv_series)
        
        # Estimate FP rate (percentage of samples above threshold)
        fp_count = sum(1 for v in mv_series if v > MVS_THRESHOLD)
        fp_estimate = fp_count / len(mv_series)
        
        return {
            'p95': p95,
            'mean_mv': mean_mv,
            'fp_estimate': fp_estimate
        }
    
    def _calculate_baseline_variance(self, packets, band):
        """Calculate baseline variance for normalization"""
        turbulences = [self._calculate_spatial_turbulence(pkt, band) for pkt in packets]
        
        if not turbulences:
            return 0.0
        
        mean = sum(turbulences) / len(turbulences)
        variance = sum((t - mean) ** 2 for t in turbulences) / len(turbulences)
        
        return variance
    
    def calibrate(self):
        """
        Calibrate by selecting the optimal band based on P95 of moving variance.
        
        Strategy:
        1. Filter bands where P95 < threshold (these won't have FP)
        2. Among valid bands, select the one with HIGHEST P95
           (most "active" band that still stays under threshold)
        
        This balances FP avoidance with motion sensitivity.
        
        Returns:
            tuple: (selected_band, normalization_scale) or (None, 1.0) if failed
        """
        if self._packet_count < MVS_WINDOW_SIZE + 10:
            print("Band: Not enough packets for calibration")
            return None, 1.0
        
        self._prepare_for_reading()
        
        packets = self._read_all_packets()
        if len(packets) < MVS_WINDOW_SIZE + 10:
            print("Band: Failed to read packets")
            return None, 1.0
        
        print(f"Band: Analyzing {len(packets)} packets...")
        
        # Get all candidate bands
        candidates = self._get_candidate_bands()
        
        if not candidates:
            print("Band: No valid candidate bands found")
            return None, 1.0
        
        step = 2 if self._num_subcarriers >= 256 else 1
        print(f"Band: Evaluating {len(candidates)} candidate bands (step={step})...")
        
        # Evaluate each candidate and collect results
        band_results = []
        
        for band in candidates:
            result = self._evaluate_band(packets, band)
            result['band'] = band
            band_results.append(result)
            
            # Memory management
            if len(band_results) % 20 == 0:
                gc.collect()
        
        # Strategy: Find bands with P95 < threshold (low FP risk)
        # Then select the one with HIGHEST P95 (most active, still safe)
        safe_margin = 0.15  # Safety margin below threshold
        p95_limit = MVS_THRESHOLD - safe_margin  # 0.85 for threshold=1.0
        
        valid_bands = [r for r in band_results if r['p95'] < p95_limit]
        
        if valid_bands:
            # Select the most "active" band that's still safe
            # Higher P95 = more responsive to changes
            best_result = max(valid_bands, key=lambda r: r['p95'])
            best_band = best_result['band']
            best_p95 = best_result['p95']
            print(f"Band: Found {len(valid_bands)} safe bands (P95 < {p95_limit:.2f})")
        else:
            # Fallback: no safe bands, pick the one with lowest P95
            print(f"Band: No bands with P95 < {p95_limit:.2f}, using lowest P95")
            best_result = min(band_results, key=lambda r: r['p95'])
            best_band = best_result['band']
            best_p95 = best_result['p95']
        
        if best_band is None:
            print("Band: All candidates failed evaluation")
            return None, 1.0
        
        # Calculate normalization
        baseline_variance = self._calculate_baseline_variance(packets, best_band)
        norm_target = get_normalization_target(self._num_subcarriers)
        
        if baseline_variance > norm_target:
            normalization_scale = norm_target / baseline_variance
            normalization_scale = max(0.4, normalization_scale)
        else:
            normalization_scale = 1.0
        
        # Report results
        print(f"Band: Calibration successful")
        print(f"  Selected: [{best_band[0]}-{best_band[-1]}]")
        print(f"  P95 MV: {best_p95:.4f} (threshold: {MVS_THRESHOLD})")
        print(f"  Est. FP rate: {best_result['fp_estimate']*100:.1f}%")
        print(f"  Baseline var: {baseline_variance:.4f}")
        print(f"  Norm scale: {normalization_scale:.4f}")
        
        if self._filtered_count > 0:
            print(f"  Filtered: {self._filtered_count} packets (wrong SC count)")
        
        return best_band, normalization_scale
    
    def free_buffer(self):
        """Free resources after calibration"""
        if self._file:
            self._file.close()
            self._file = None
        
        try:
            os.remove(BUFFER_FILE)
        except OSError:
            pass
        
        gc.collect()


