"""
Micro-ESPectre - PCA Detector

Principal Component Analysis detector implementation.
Based on Espressif's esp_radar algorithm.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""
import math

try:
    from src.detector_interface import IDetector, MotionState
except ImportError:
    from detector_interface import IDetector, MotionState


def pearson_correlation(a, b):
    """
    Calculate Pearson correlation coefficient between two vectors.
    
    Returns value in range [-1, 1], where:
    - 1 = perfectly correlated (identical signal shape)
    - 0 = no correlation
    - -1 = perfectly anti-correlated
    """
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    
    n = len(a)
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    
    cov_sum = 0.0
    var_sum_a = 0.0
    var_sum_b = 0.0
    
    for i in range(n):
        diff_a = a[i] - mean_a
        diff_b = b[i] - mean_b
        cov_sum += diff_a * diff_b
        var_sum_a += diff_a ** 2
        var_sum_b += diff_b ** 2
    
    denominator = math.sqrt(var_sum_a * var_sum_b)
    if denominator < 1e-10:
        return 0.0
    
    return cov_sum / denominator


def pca_power_method(data_matrix, max_iters=30, precision=0.0001):
    """
    PCA using power method to find principal eigenvector.
    Replicates Espressif's pca() function.
    
    Args:
        data_matrix: 2D list of shape (num_packets, num_subcarriers)
        max_iters: Maximum iterations for power method
        precision: Convergence threshold
    
    Returns:
        Principal component vector or None
    """
    if len(data_matrix) == 0:
        return None
    
    rows = len(data_matrix[0])  # num_subcarriers
    cols = len(data_matrix)      # num_packets
    
    if cols == 0 or rows == 0:
        return None
    
    # Transpose: subcarriers as rows, packets as columns
    matrix = [[data_matrix[j][i] for j in range(cols)] for i in range(rows)]
    
    # Compute covariance matrix (cols x cols)
    zoom_out = rows * cols
    cov_matrix = [[0.0] * cols for _ in range(cols)]
    
    for i in range(cols):
        for j in range(i + 1):
            cov_sum = 0.0
            for k in range(rows):
                cov_sum += matrix[k][i] * matrix[k][j]
            cov_matrix[i][j] = cov_sum / zoom_out
            if i != j:
                cov_matrix[j][i] = cov_matrix[i][j]
    
    # Power method to find principal eigenvector
    eigenvector = [1.0] * cols
    eigenvalue = 1.0
    eigenvalue_last = 0.0
    
    for _ in range(max_iters):
        if abs(eigenvalue - eigenvalue_last) <= precision:
            break
        
        eigenvalue_last = eigenvalue
        eigenvalue = 0.0
        
        # Multiply: eigenvalue_list = cov_matrix @ eigenvector
        eigenvalue_list = [0.0] * cols
        for i in range(cols):
            for j in range(cols):
                eigenvalue_list[i] += cov_matrix[i][j] * eigenvector[j]
            if eigenvalue_list[i] > eigenvalue:
                eigenvalue = eigenvalue_list[i]
        
        # Normalize
        if eigenvalue > 1e-10:
            eigenvector = [v / eigenvalue for v in eigenvalue_list]
    
    # Project data onto eigenvector
    output = [0.0] * rows
    for i in range(rows):
        for j in range(cols):
            output[i] += matrix[i][j] * eigenvector[j]
        output[i] /= cols
    
    return output


class PCADetector(IDetector):
    """
    PCA-based motion detector (Espressif esp_radar style).
    
    Algorithm:
    1. Collect CSI amplitudes into sliding window
    2. Apply PCA (power method) to extract principal component
    3. Calculate correlation with past PCA vectors (jitter)
    4. Calculate correlation with calibration vectors (wander)
    5. Invert values: output = 1 - correlation (high = movement)
    6. Apply count-based detection for final decision
    
    Note: Threshold must be set externally via set_threshold() after
    calibration with PCACalibrator. Formula: threshold = 1 - min(correlation)
    
    Based on esp-radar component v0.3.1 (Apache-2.0 license).
    """
    
    # Parameters (aligned with C++ implementation)
    DEFAULT_PCA_WINDOW_SIZE = 10
    DEFAULT_MOVE_BUFFER_SIZE = 5
    DEFAULT_OUTLIERS_NUM = 2
    DEFAULT_SUBCARRIER_STEP = 4
    BUFF_MAX_LEN = 25
    PCA_CALIBRATION_SAMPLES = 10  # Same as C++ PCA_CALIBRATION_SAMPLES
    
    # Scale factor for PCA values to match MVS threshold range (0.1-10.0)
    # PCA jitter is ~0.0001-0.001, scaling by 1000 brings it to ~0.1-1.0
    PCA_SCALE = 1000.0
    DEFAULT_MOVE_THRESHOLD = 10.0  # 0.01 * PCA_SCALE
    
    def __init__(self,
                 pca_window_size=DEFAULT_PCA_WINDOW_SIZE,
                 move_buffer_size=DEFAULT_MOVE_BUFFER_SIZE,
                 outliers_num=DEFAULT_OUTLIERS_NUM,
                 threshold=DEFAULT_MOVE_THRESHOLD,
                 subcarrier_step=DEFAULT_SUBCARRIER_STEP):
        """
        Initialize PCA detector.
        
        Args:
            pca_window_size: Packets for PCA computation (default: 10)
            move_buffer_size: Buffer for jitter values (default: 5)
            outliers_num: Violations needed to trigger (default: 2)
            threshold: Threshold on inverted jitter (default: 10.0, scaled by PCA_SCALE)
            subcarrier_step: Use every Nth subcarrier (default: 4)
        """
        self.pca_window_size = pca_window_size
        self.move_buffer_size = move_buffer_size
        self.outliers_num = outliers_num
        self._threshold = threshold
        self.subcarrier_step = subcarrier_step
        
        # Buffers
        self.csi_buffer = []
        self.pca_buffer = []
        self.pca_count = 0
        self.calibration_data = []  # For wander calculation (max PCA_CALIBRATION_SAMPLES)
        self.jitter_buffer = []
        
        # State
        self._state = MotionState.IDLE
        self._packet_count = 0
        self._motion_count = 0
        self._current_jitter = 0.0
        self._current_wander = 0.0
        self._threshold_externally_set = False  # Same as C++ threshold_externally_set_
        
        # Tracking
        self.jitter_history = []
        self.wander_history = []
        self.state_history = []
        self.track_data = False
    
    def _extract_amplitudes(self, csi_data):
        """Extract amplitudes using step-based subcarrier selection."""
        amplitudes = []
        max_sc = min(64, len(csi_data) // 2)
        
        for sc_idx in range(0, max_sc, self.subcarrier_step):
            i = sc_idx * 2
            if i + 1 < len(csi_data):
                # Espressif CSI format: [Imaginary, Real]
                imag = float(csi_data[i])
                real = float(csi_data[i + 1])
                amplitudes.append(math.sqrt(real * real + imag * imag))
        
        return amplitudes
    
    def _compute_waveform_jitter(self, pca_current):
        """
        Compute jitter as max correlation with RECENT PCA vectors.
        
        Jitter measures short-term changes in the signal.
        High correlation (close to 1) = stable = low jitter
        Low correlation = changing = high jitter
        
        Note: This compares with pca_buffer (recent), NOT calibration_data.
        Wander uses calibration_data for baseline comparison.
        """
        # Need at least 2 past vectors (aligned with C++ pca_buffer_count_ < 2)
        if len(self.pca_buffer) < 2:
            return 1.0  # No past data, assume max correlation (no movement)
        
        max_corr = 0.0
        # Compare with most recent past PCA vectors
        # Use MOVE_BUFFER_SIZE - 1 to match C++ behavior
        num_compare = min(self.move_buffer_size - 1, len(self.pca_buffer) - 1)
        for i in range(num_compare):
            past_idx = len(self.pca_buffer) - 1 - i  # -1 because pca_current not yet appended
            if past_idx < 0:
                break
            pca_past = self.pca_buffer[past_idx]
            corr_val = abs(pearson_correlation(pca_current, pca_past))
            if corr_val > max_corr:
                max_corr = corr_val
        
        return max_corr
    
    def _compute_waveform_wander(self, pca_current):
        """Compute wander as max correlation with calibration samples."""
        if len(self.calibration_data) == 0:
            return 1.0
        
        max_corr = 0.0
        for calib_pca in self.calibration_data:
            corr_val = abs(pearson_correlation(pca_current, calib_pca))
            if corr_val > max_corr:
                max_corr = corr_val
        
        return max_corr
    
    def process_packet(self, csi_data, selected_subcarriers=None):
        """
        Process a CSI packet.
        
        Note: PCA ignores selected_subcarriers, uses its own step-based selection.
        """
        self._packet_count += 1
        
        # Extract amplitudes
        amplitudes = self._extract_amplitudes(csi_data)
        
        # Add to CSI buffer
        self.csi_buffer.append(amplitudes)
        if len(self.csi_buffer) > self.pca_window_size:
            self.csi_buffer.pop(0)
        
        # Need enough data for PCA
        if len(self.csi_buffer) < self.pca_window_size:
            self._current_jitter = 0.0
            self._current_wander = 0.0
            return
        
        # Compute PCA
        pca_current = pca_power_method(self.csi_buffer)
        if pca_current is None:
            self._current_jitter = 0.0
            self._current_wander = 0.0
            return
        
        # Calculate waveform metrics
        jitter_corr = self._compute_waveform_jitter(pca_current)
        wander_corr = self._compute_waveform_wander(pca_current)
        
        # Invert: 1 - correlation, scaled by PCA_SCALE to match MVS range
        self._current_jitter = (1.0 - jitter_corr) * self.PCA_SCALE
        self._current_wander = (1.0 - wander_corr) * self.PCA_SCALE
        
        # Store PCA vector
        self.pca_buffer.append(pca_current[:])
        if len(self.pca_buffer) > self.BUFF_MAX_LEN:
            self.pca_buffer.pop(0)
        self.pca_count += 1
        
        # Collect calibration samples for wander (same as C++ PCA_CALIBRATION_SAMPLES)
        if len(self.calibration_data) < self.PCA_CALIBRATION_SAMPLES and self.pca_count % 5 == 0:
            self.calibration_data.append(pca_current[:])
        
        # Add to jitter buffer
        self.jitter_buffer.append(self._current_jitter)
        if len(self.jitter_buffer) > self.BUFF_MAX_LEN:
            self.jitter_buffer.pop(0)
        
        if self.track_data:
            self.jitter_history.append(self._current_jitter)
            self.wander_history.append(self._current_wander)
    
    def update_state(self):
        """Update motion state based on jitter buffer."""
        if len(self.jitter_buffer) < self.move_buffer_size:
            self._state = MotionState.IDLE
            if self.track_data:
                self.state_history.append('IDLE')
            return self._get_metrics()
        
        # Count threshold violations
        move_count = 0
        
        # Calculate median of ENTIRE jitter buffer (matches C++ behavior)
        sorted_buffer = sorted(self.jitter_buffer)
        jitter_median = sorted_buffer[len(sorted_buffer) // 2]
        
        # Check only the last move_buffer_size values
        jitter_values = self.jitter_buffer[-self.move_buffer_size:]
        for jitter_val in jitter_values:
            # Dual condition (values are scaled by PCA_SCALE)
            if (jitter_val > self._threshold or
                (jitter_val > jitter_median and jitter_val > 10.0)):
                move_count += 1
        
        # Update state
        if move_count >= self.outliers_num:
            self._state = MotionState.MOTION
            self._motion_count += 1
        else:
            self._state = MotionState.IDLE
        
        if self.track_data:
            state_str = 'MOTION' if self._state == MotionState.MOTION else 'IDLE'
            self.state_history.append(state_str)
        
        return self._get_metrics()
    
    def _get_metrics(self):
        """Get current metrics dict."""
        return {
            'jitter': self._current_jitter,
            'wander': self._current_wander,
            'threshold': self._threshold,
            'state': self._state
        }
    
    def get_state(self):
        """Get current motion state."""
        return self._state
    
    def get_motion_metric(self):
        """Get current jitter value."""
        return self._current_jitter
    
    def get_threshold(self):
        """Get current threshold."""
        return self._threshold
    
    def set_threshold(self, threshold):
        """Set detection threshold (must be called after PCACalibrator completes).
        
        Threshold is now scaled by PCA_SCALE (1000), valid range is 0.0-10.0
        matching the MVS threshold range.
        """
        if 0.0 <= threshold <= 10.0:
            self._threshold = threshold
            self._threshold_externally_set = True
            return True
        return False
    
    def is_ready(self):
        """Check if threshold has been set externally (via PCACalibrator)."""
        return self._threshold_externally_set
    
    def reset(self):
        """Reset detector state (threshold is preserved)."""
        self.csi_buffer = []
        self.pca_buffer = []
        self.pca_count = 0
        self.calibration_data = []
        self.jitter_buffer = []
        self._state = MotionState.IDLE
        self._motion_count = 0
        self._current_jitter = 0.0
        self._current_wander = 0.0
        # Note: _threshold and _threshold_externally_set are preserved across reset
        self.jitter_history = []
        self.wander_history = []
        self.state_history = []
    
    def get_name(self):
        """Get detector name."""
        return "PCA"
    
    @property
    def total_packets(self):
        """Total packets processed."""
        return self._packet_count
    
    def get_motion_count(self):
        """Get number of motion detections."""
        return self._motion_count
