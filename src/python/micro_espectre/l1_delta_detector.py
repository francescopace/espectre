"""
Micro-ESPectre - L1-Delta Detector

Normalized amplitude-profile displacement detector.

For each packet the amplitudes of the selected subcarriers are normalized by
their mean (per-packet gain invariance, same rationale as the CV turbulence
path), then compared with the profile observed ``lag`` packets earlier:

    d[n] = mean_k |A_norm[n][k] - A_norm[n - lag][k]|

The motion metric is the mean of ``d`` over the sliding window. Motion in the
room decorrelates the multipath profile, so ``d`` rises on every subcarrier
coherently, while receiver noise keeps it near a stable floor. Offline
benchmarks on the repo datasets show the quiet-level of this metric varies
<=1.3x across sessions (vs up to 14.5x for MVS moving variance), with
per-chip recall more uniform than MVS at equal aggregate F1.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""
try:
    from src.detector_interface import IDetector, MotionState
    from src.features import normalize_amplitude_profile_into
    from src.segmentation import SegmentationContext
except ImportError:
    from detector_interface import IDetector, MotionState
    from features import normalize_amplitude_profile_into
    from segmentation import SegmentationContext

# Profile comparison lag in packets (~100 ms at 100 pps): long enough for
# body motion to displace the multipath profile, short enough to track it.
L1_DELTA_LAG = 10


class L1DeltaDetector(IDetector):
    """
    L1 normalized amplitude-profile displacement detector.

    Algorithm:
    1. Compute per-packet subcarrier amplitudes (selected band)
    2. Normalize the profile by its mean (per-packet gain invariance)
    3. d = mean absolute difference vs the profile ``lag`` packets earlier
    4. Motion metric = running mean of d over the sliding window
    5. Compare to threshold for state decision

    The startup threshold uses the shared calibration flow
    (max metric during calibration x factor); the benchmark-tuned factor
    for this metric is STARTUP_THRESHOLD_FACTOR. The tight quiet floor of
    this metric also enables the calibration consistency gate in
    threshold.py, which extends a contaminated startup window instead of
    accepting a movement-inflated max.
    """
    ALGORITHM = "l1_delta"

    # Startup calibration multiplier (benchmark-tuned for this metric;
    # MVS uses the threshold.py "auto" factor instead).
    STARTUP_THRESHOLD_FACTOR = 1.1

    # Opt into the startup consistency gate (threshold.py); validated for
    # this metric only, not for MVS moving variance.
    STARTUP_GATE = True

    def __init__(self,
                 window_size=100,
                 threshold=1.0,
                 lag=L1_DELTA_LAG,
                 **_unused_filter_kwargs):
        """
        Initialize L1-Delta detector.

        Args:
            window_size: Metric averaging window in packets (default: 100)
            threshold: Motion detection threshold (default: 1.0)
            lag: Profile comparison lag in packets (default: L1_DELTA_LAG)

        Note: Hampel/low-pass filter kwargs are accepted for constructor
        compatibility with the other detectors but unused; the L1 mean is
        intrinsically robust to single-packet spikes.
        """
        self.window_size = max(2, int(window_size))
        self.threshold = threshold
        self.lag = max(1, int(lag))

        # Pre-allocated buffers: no per-packet list allocations in the hot
        # path (same rationale as SegmentationContext on MicroPython).
        profile_width = SegmentationContext.AMPLITUDE_BUFFER_SIZE
        # Ring of the last ``lag`` normalized profiles (0 length = invalid).
        self._profile_ring = [[0.0] * profile_width for _ in range(self.lag)]
        self._profile_len = [0] * self.lag
        self._current_profile = [0.0] * profile_width
        self._amplitude_buffer = [0.0] * profile_width
        # Ring of the last ``window_size`` d values with running sum.
        self._delta_ring = [0.0] * self.window_size
        self._delta_index = 0
        self._delta_count = 0
        self._delta_sum = 0.0

        self._packet_count = 0
        self._state = MotionState.IDLE
        self._current_metric = 0.0
        self.last_delta = 0.0

    def _push_delta(self, delta):
        """Add one d value to the running-mean ring."""
        if self._delta_count < self.window_size:
            self._delta_count += 1
        else:
            self._delta_sum -= self._delta_ring[self._delta_index]
        self._delta_ring[self._delta_index] = delta
        self._delta_sum += delta
        self._delta_index = (self._delta_index + 1) % self.window_size

    def process_packet(self, csi_data, selected_subcarriers=None):
        """
        Process a CSI packet.

        Args:
            csi_data: Raw CSI data (int8 I/Q pairs)
            selected_subcarriers: Subcarrier indices to use
        """
        self._packet_count += 1
        amplitude_count = SegmentationContext._fill_amplitude_buffer(
            csi_data, selected_subcarriers, self._amplitude_buffer
        )
        profile = self._current_profile
        profile_len = normalize_amplitude_profile_into(
            self._amplitude_buffer, amplitude_count, profile
        )

        ring_slot = (self._packet_count - 1) % self.lag
        reference = self._profile_ring[ring_slot]
        reference_len = self._profile_len[ring_slot]

        # Warmup or malformed packets have no comparable lagged profile.
        if profile_len > 0 and reference_len == profile_len:
            total = 0.0
            for i in range(profile_len):
                diff = profile[i] - reference[i]
                total += diff if diff >= 0 else -diff
            self.last_delta = total / profile_len
            self._push_delta(self.last_delta)

        # Store the current profile by swapping buffers (allocation-free).
        self._profile_ring[ring_slot] = profile
        self._profile_len[ring_slot] = profile_len
        self._current_profile = reference

    def update_state(self):
        """
        Update motion state from the current running mean (call at publish time).

        Returns:
            dict: Current metrics (motion_metric, l1_delta, threshold, state)
        """
        if self._delta_count >= self.window_size:
            self._current_metric = self._delta_sum / self._delta_count
        else:
            # Match MVS semantics: not ready until the window is full.
            self._current_metric = 0.0

        if self._state == MotionState.IDLE:
            if self._current_metric > self.threshold:
                self._state = MotionState.MOTION
        elif self._current_metric < self.threshold:
            self._state = MotionState.IDLE

        return {
            'motion_metric': self._current_metric,
            'l1_delta': self._current_metric,
            'threshold': self.threshold,
            'state': self._state,
        }

    def get_state(self):
        """Get current motion state."""
        return self._state

    def get_motion_metric(self):
        """Get current L1-delta running mean."""
        return self._current_metric

    def get_threshold(self):
        """Get current threshold."""
        return self.threshold

    def set_threshold(self, threshold):
        """Set detection threshold."""
        if 0.0 <= threshold <= 10.0:
            self.threshold = threshold
            return True
        return False

    def set_adaptive_threshold(self, threshold):
        """Set startup-calibrated threshold (clamped like MVS)."""
        self.threshold = max(1e-6, min(10.0, threshold))

    def is_ready(self):
        """Check if the metric window is full."""
        return self._delta_count >= self.window_size

    def reset(self):
        """Reset detector state (pre-allocated buffers are kept and reused)."""
        for i in range(self.lag):
            self._profile_len[i] = 0
        self._delta_ring = [0.0] * self.window_size
        self._delta_index = 0
        self._delta_count = 0
        self._delta_sum = 0.0
        self._packet_count = 0
        self._state = MotionState.IDLE
        self._current_metric = 0.0
        self.last_delta = 0.0

    def get_name(self):
        """Get detector name."""
        return "L1D"

    @property
    def total_packets(self):
        """Total packets processed."""
        return self._packet_count
