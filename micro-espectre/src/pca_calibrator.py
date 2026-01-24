"""
Micro-ESPectre - PCA Calibrator

Calibrator for PCA-based motion detection.
Collects baseline correlation values during quiet period.

Algorithm:
1. Collect baseline CSI packets (quiet room)
2. For each packet: extract amplitudes, compute PCA, calculate correlation
3. Store correlation values (cal_values)
4. Return band (PCA subcarriers) and cal_values for threshold calculation

Threshold formula (calculated externally): (1 - min(correlation)) * PCA_SCALE

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import math
import gc

try:
    from src.calibrator_interface import ICalibrator
    from src.pca_detector import pearson_correlation, pca_power_method, PCADetector
except ImportError:
    from calibrator_interface import ICalibrator
    from pca_detector import pearson_correlation, pca_power_method, PCADetector

# PCA constants (must match pca_detector.py and C++ implementation)
PCA_WINDOW_SIZE = 10
PCA_BUFFER_SIZE = 25
PCA_SUBCARRIER_STEP = 4
PCA_NUM_SUBCARRIERS = 64 // PCA_SUBCARRIER_STEP  # 16 subcarriers


class PCACalibrator(ICalibrator):
    """
    PCA Calibrator for correlation-based motion detection.
    
    Collects correlation values during baseline for threshold calculation.
    Uses fixed subcarrier selection (every PCA_SUBCARRIER_STEP-th subcarrier).
    
    Unlike NBVI/P95, PCA doesn't select subcarriers dynamically.
    It returns the fixed PCA subcarriers and correlation values.
    """
    
    def __init__(self, buffer_size=700):
        """
        Initialize PCA calibrator.
        
        Args:
            buffer_size: Number of packets to collect (default: 700)
        """
        self.buffer_size = buffer_size
        self._packet_count = 0
        
        # CSI buffer for PCA computation
        self.csi_buffer = []
        self.csi_buffer_count = 0
        
        # PCA buffer for correlation computation
        self.pca_buffer = []
        self.pca_buffer_count = 0
        
        # Collected correlation values
        self.correlation_values = []
        
        # Fixed PCA subcarriers (every PCA_SUBCARRIER_STEP-th)
        self.selected_band = [i * PCA_SUBCARRIER_STEP for i in range(PCA_NUM_SUBCARRIERS)]
    
    def _extract_amplitudes(self, csi_data):
        """Extract amplitudes using step-based subcarrier selection."""
        amplitudes = []
        max_sc = min(64, len(csi_data) // 2)
        
        for sc_idx in range(0, max_sc, PCA_SUBCARRIER_STEP):
            i = sc_idx * 2
            if i + 1 < len(csi_data):
                # Espressif CSI format: [Imaginary, Real]
                imag = float(csi_data[i])
                real = float(csi_data[i + 1])
                amplitudes.append(math.sqrt(real * real + imag * imag))
        
        return amplitudes
    
    def _compute_pca(self):
        """Compute PCA on current CSI buffer."""
        if len(self.csi_buffer) < PCA_WINDOW_SIZE:
            return None
        
        # Use last PCA_WINDOW_SIZE packets
        window = self.csi_buffer[-PCA_WINDOW_SIZE:]
        return pca_power_method(window)
    
    def _compute_correlation(self, pca_current):
        """Calculate max correlation with past PCA vectors."""
        if len(self.pca_buffer) < 1:
            return 0.0
        
        max_corr = 0.0
        # Compare with up to 5 recent PCA vectors
        num_past = min(len(self.pca_buffer), 5)
        
        for i in range(num_past):
            past_idx = len(self.pca_buffer) - 1 - i
            pca_past = self.pca_buffer[past_idx]
            corr = abs(pearson_correlation(pca_current, pca_past))
            if corr > max_corr:
                max_corr = corr
        
        return max_corr
    
    def add_packet(self, csi_data):
        """
        Add CSI packet to calibration buffer.
        
        Args:
            csi_data: Raw CSI data (I/Q pairs)
        
        Returns:
            int: Current packet count (progress indicator)
        """
        if self._packet_count >= self.buffer_size:
            return self.buffer_size
        
        # Extract amplitudes
        amplitudes = self._extract_amplitudes(csi_data)
        
        # Add to CSI buffer
        self.csi_buffer.append(amplitudes)
        if len(self.csi_buffer) > PCA_WINDOW_SIZE:
            self.csi_buffer.pop(0)
        self.csi_buffer_count += 1
        self._packet_count += 1
        
        # Need enough data for PCA
        if self.csi_buffer_count < PCA_WINDOW_SIZE:
            if self._packet_count % 100 == 0:
                print(f"PCA: Collection progress: {self._packet_count}/{self.buffer_size}")
            return self._packet_count
        
        # Compute PCA
        pca_current = self._compute_pca()
        if pca_current is None:
            return self._packet_count
        
        # Calculate correlation with past PCA vectors
        if self.pca_buffer_count > 0:
            correlation = self._compute_correlation(pca_current)
            if correlation > 0.0:
                self.correlation_values.append(correlation)
        
        # Store PCA vector
        self.pca_buffer.append(pca_current[:])
        if len(self.pca_buffer) > PCA_BUFFER_SIZE:
            self.pca_buffer.pop(0)
        self.pca_buffer_count += 1
        
        # Log progress
        if self._packet_count % 100 == 0:
            print(f"PCA: Collection progress: {self._packet_count}/{self.buffer_size} "
                  f"(correlations: {len(self.correlation_values)})")
        
        return self._packet_count
    
    def calibrate(self):
        """
        Complete calibration and return results.
        
        Returns:
            tuple: (selected_band, correlation_values)
                - selected_band: List of PCA subcarrier indices (fixed)
                - correlation_values: List of correlation values for threshold calculation
        """
        if len(self.correlation_values) == 0:
            print("PCA: No correlation values collected")
            return None, []
        
        print(f"PCA: Calibration complete")
        print(f"  Subcarriers: every {PCA_SUBCARRIER_STEP}th (total: {PCA_NUM_SUBCARRIERS})")
        print(f"  Correlation values: {len(self.correlation_values)}")
        
        if self.correlation_values:
            min_corr = min(self.correlation_values)
            max_corr = max(self.correlation_values)
            pca_scale = PCADetector.PCA_SCALE
            suggested_threshold = (1.0 - min_corr) * pca_scale
            print(f"  Correlation range: [{min_corr:.4f}, {max_corr:.4f}]")
            print(f"  Suggested threshold: {suggested_threshold:.4f} (scaled by {pca_scale:.0f})")
        
        return self.selected_band, self.correlation_values
    
    def free_buffer(self):
        """Free resources after calibration."""
        self.csi_buffer = []
        self.pca_buffer = []
        self.correlation_values = []
        gc.collect()
    
    def get_packet_count(self):
        """Get the number of packets currently in the buffer."""
        return self._packet_count
    
    def is_buffer_full(self):
        """Check if the buffer has collected enough packets."""
        return self._packet_count >= self.buffer_size
