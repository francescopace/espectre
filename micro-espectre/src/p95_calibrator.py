"""
P95 Calibrator - Automatic Subcarrier Band Selection

Selects optimal 12-subcarrier band for motion detection by minimizing
the P95 of moving variance during baseline. This directly optimizes
for low false positive rate.

Algorithm (P95 Band Selection):
1. Collect baseline CSI packets (quiet room)
2. For each candidate band of 12 consecutive subcarriers:
   - Calculate moving variance series
   - Compute P95 of moving variance for band ranking
3. Select the band with LOWEST P95 (furthest from threshold)

Returns (band, mv_values). Adaptive threshold is calculated externally
using threshold.py after band selection.

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
    calculate_spatial_turbulence,
    calculate_moving_variance
)

# Import interface
from src.calibrator_interface import ICalibrator

# Constants
BUFFER_FILE = '/p95_buffer.bin'

# Threshold for null subcarrier detection
NULL_SUBCARRIER_THRESHOLD = 1.0

# MVS parameters
MVS_WINDOW_SIZE = 50
MVS_THRESHOLD = 1.0

# Band selection uses fixed P95 (algorithm constant)
BAND_SELECTION_PERCENTILE = 95


def cleanup_buffer_file():
    """Remove any leftover buffer file from previous interrupted runs."""
    try:
        os.remove(BUFFER_FILE)
        print("P95: Cleaned up leftover buffer file")
    except OSError:
        pass


class P95Calibrator(ICalibrator):
    """
    Automatic band calibrator using P95 moving variance optimization.
    
    Selects the optimal 12-subcarrier band by finding the one with
    lowest P95 of moving variance during baseline (minimizes FP rate).
    
    Uses file-based storage to avoid RAM limitations on ESP32.
    HT20 only: 64 subcarriers.
    """
    
    def __init__(self, buffer_size=700):
        """
        Initialize P95 calibrator.
        
        Args:
            buffer_size: Number of packets to collect (default: 700)
        """
        self.buffer_size = buffer_size
        self._packet_count = 0
        self._filtered_count = 0
        self._file = None
        self._initialized = False
        
        # Remove old buffer file
        try:
            os.remove(BUFFER_FILE)
        except OSError:
            pass
        
        self._file = open(BUFFER_FILE, 'wb')
    
    def add_packet(self, csi_data):
        """
        Add CSI packet to calibration buffer.
        
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
            print(f'P95: HT20 mode, {NUM_SUBCARRIERS} SC, guard [{GUARD_BAND_LOW}-{GUARD_BAND_HIGH}], DC={DC_SUBCARRIER}')
        
        # Store raw I/Q data (2 bytes per subcarrier)
        iq_data = bytearray(EXPECTED_CSI_LEN)
        for sc in range(NUM_SUBCARRIERS):
            i_idx = sc * 2
            q_idx = sc * 2 + 1
            iq_data[i_idx] = int(csi_data[i_idx]) & 0xFF
            iq_data[q_idx] = int(csi_data[q_idx]) & 0xFF
        
        self._file.write(iq_data)
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
        """Read all packets from file and pre-compute magnitudes"""
        self._file.seek(0)
        data = self._file.read()
        
        num_packets = len(data) // EXPECTED_CSI_LEN
        print(f"P95: File size {len(data)} bytes = {num_packets} packets")
        
        packets_mags = []
        for pkt_idx in range(num_packets):
            offset = pkt_idx * EXPECTED_CSI_LEN
            mags = []
            for sc in range(NUM_SUBCARRIERS):
                i_idx = offset + sc * 2
                q_idx = i_idx + 1
                # Espressif CSI format: [Imaginary, Real, ...] per subcarrier
                Q = data[i_idx]      # Imaginary first
                I = data[q_idx]      # Real second
                if I > 127:
                    I = I - 256
                if Q > 127:
                    Q = Q - 256
                mags.append(math.sqrt(I * I + Q * Q))
            packets_mags.append(mags)
            
            if (pkt_idx + 1) % 50 == 0:
                gc.collect()
        
        gc.collect()
        return packets_mags
    
    def _get_candidate_bands(self):
        """Generate valid candidate bands of 12 consecutive subcarriers."""
        candidates = []
        
        # Zone before DC (subcarriers 11-31)
        for start in range(GUARD_BAND_LOW, DC_SUBCARRIER - BAND_SIZE + 1):
            band = list(range(start, start + BAND_SIZE))
            if all(GUARD_BAND_LOW <= sc <= GUARD_BAND_HIGH and sc != DC_SUBCARRIER for sc in band):
                candidates.append(band)
        
        # Zone after DC (subcarriers 33-52)
        for start in range(DC_SUBCARRIER + 1, GUARD_BAND_HIGH - BAND_SIZE + 2):
            band = list(range(start, start + BAND_SIZE))
            if all(GUARD_BAND_LOW <= sc <= GUARD_BAND_HIGH and sc != DC_SUBCARRIER for sc in band):
                candidates.append(band)
        
        return candidates
    
    def _evaluate_band(self, packets_mags, band):
        """
        Evaluate a band by computing P95 and MV values.
        
        Note: Hampel filter is NOT applied during calibration. Outliers are useful
        information for band selection (P95 is already robust to outliers).
        Hampel is only applied during normal operation in the CSI processor.
        """
        # Calculate turbulence series
        turbulences = []
        for pkt_mags in packets_mags:
            turb = calculate_spatial_turbulence(pkt_mags, band)
            turbulences.append(turb)
        
        # Calculate moving variance series
        mv_series = calculate_moving_variance(turbulences, MVS_WINDOW_SIZE)
        del turbulences
        
        if not mv_series:
            return {'p95': float('inf'), 'mv_values': [], 'fp_estimate': 1.0}
        
        # P95 for band selection (fixed)
        p95 = calculate_percentile(mv_series, BAND_SELECTION_PERCENTILE)
        
        # Estimate FP rate
        fp_count = sum(1 for v in mv_series if v > MVS_THRESHOLD)
        fp_estimate = fp_count / len(mv_series)
        
        return {
            'p95': p95,
            'mv_values': mv_series,
            'fp_estimate': fp_estimate
        }
    
    def calibrate(self):
        """
        Calibrate by selecting the optimal band.
        
        Returns:
            tuple: (selected_band, mv_values) or (None, []) if failed
        """
        if self._packet_count < MVS_WINDOW_SIZE + 10:
            print("P95: Not enough packets for calibration")
            return None, []
        
        self._prepare_for_reading()
        
        packets = self._read_all_packets()
        if len(packets) < MVS_WINDOW_SIZE + 10:
            print("P95: Failed to read packets")
            return None, []
        
        print(f"P95: Analyzing {len(packets)} packets...")
        
        candidates = self._get_candidate_bands()
        
        if not candidates:
            print("P95: No valid candidate bands found")
            return None, []
        
        print(f"P95: Evaluating {len(candidates)} candidate bands...")
        
        # Evaluate each candidate
        band_results = []
        for i, band in enumerate(candidates):
            result = self._evaluate_band(packets, band)
            result['band'] = band
            band_results.append(result)
            
            if (i + 1) % 10 == 0:
                gc.collect()
                print(f"  Evaluating... {i+1}/{len(candidates)}")
        
        # Select band with best P95
        safe_margin = 0.15
        p95_limit = MVS_THRESHOLD - safe_margin
        
        valid_bands = [r for r in band_results if r['p95'] < p95_limit]
        
        if valid_bands:
            best_result = max(valid_bands, key=lambda r: r['p95'])
            print(f"P95: Found {len(valid_bands)} safe bands (P95 < {p95_limit:.2f})")
        else:
            print(f"P95: No bands with P95 < {p95_limit:.2f}, using lowest")
            best_result = min(band_results, key=lambda r: r['p95'])
        
        best_band = best_result['band']
        if best_band is None:
            print("P95: All candidates failed evaluation")
            return None, []
        
        mv_values = best_result['mv_values']
        
        # Report results
        print(f"P95: Band selection successful")
        print(f"  Selected: [{best_band[0]}-{best_band[-1]}]")
        print(f"  P95 MV: {best_result['p95']:.4f}")
        print(f"  Est. FP rate: {best_result['fp_estimate']*100:.1f}%")
        
        if self._filtered_count > 0:
            print(f"  Filtered: {self._filtered_count} packets (wrong SC count)")
        
        return best_band, mv_values
    
    def free_buffer(self):
        """Free resources after calibration"""
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


# Backward compatibility alias
BandCalibrator = P95Calibrator
