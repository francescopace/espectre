"""
Micro-ESPectre - Classic Detector

Vote-free, two-feature motion detector using a weighted fusion of gain-invariant
L1 profile displacement and turbulence autocorrelation. Hampel filtering is
applied independently to both per-packet streams under one shared enable flag.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""
import math

try:
    from src.detector_interface import IDetector, MotionState
    from src.features import L1_DELTA_LAG, L1DeltaTracker, calc_autocorrelation
    from src.segmentation import SegmentationContext
except ImportError:
    from detector_interface import IDetector, MotionState
    from features import L1_DELTA_LAG, L1DeltaTracker, calc_autocorrelation
    from segmentation import SegmentationContext


class ClassicDetector(IDetector):
    """Weighted ``l1_delta + turb_autocorr`` production detector."""

    ALGORITHM = "classic"
    STARTUP_GATE = True

    # Grouped, de-overlapped OOF fit, balanced by class/chip/session.
    FEATURE_CENTER = (0.03669842332601547, 0.27886947989463806)
    FEATURE_SCALE = (0.026984458789229393, 0.33479437232017517)
    FEATURE_WEIGHT = (5.572897434234619, 3.1952695846557617)
    INTERCEPT = -2.1254162788391113

    BASE_THRESHOLD = 0.6066111851930618
    TRAIN_IDLE_Q95_LOGIT = -0.6372601389884949
    STARTUP_QUANTILE = 0.95
    STARTUP_STRENGTH = 0.3
    STARTUP_SAMPLE_LIMIT = 64

    # The detector owns its startup threshold formula; the shared multiplier is
    # retained only for the generic calibrator's progress/fallback machinery.
    STARTUP_THRESHOLD_FACTOR = 1.0

    def __init__(self, window_size=100, threshold=BASE_THRESHOLD,
                 enable_lowpass=False, lowpass_cutoff=11.0,
                 enable_hampel=True, hampel_window=7, hampel_threshold=5.0,
                 **_unused):
        self._context = SegmentationContext(
            window_size=window_size,
            threshold=1.0,
            enable_lowpass=enable_lowpass,
            lowpass_cutoff=lowpass_cutoff,
            enable_hampel=enable_hampel,
            hampel_window=hampel_window,
            hampel_threshold=hampel_threshold,
        )
        self._l1 = L1DeltaTracker(
            window_size=max(2, window_size - L1_DELTA_LAG),
            lag=L1_DELTA_LAG,
            allocate_amplitude_buffer=False,
            enable_hampel=enable_hampel,
            hampel_window=hampel_window,
            hampel_threshold=hampel_threshold,
        )
        self._ordered_turbulence = [0.0] * window_size
        self._threshold = self._clamp_probability(threshold)
        self._state = MotionState.IDLE
        self._packet_count = 0
        self._current_probability = 0.0
        self._current_logit = 0.0
        self._current_l1_delta = 0.0
        self._current_turb_autocorr = 0.0
        self._startup_logits = []

    @staticmethod
    def _clamp_probability(value):
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _sigmoid(logit):
        if logit < -20.0:
            return 0.0
        if logit > 20.0:
            return 1.0
        return 1.0 / (1.0 + math.exp(-logit))

    @staticmethod
    def _quantile(values, quantile):
        if not values:
            return None
        ordered = list(values)
        ordered.sort()
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * quantile
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction

    def process_packet(self, csi_data, selected_subcarriers=None):
        self._packet_count += 1
        turbulence = self._context.calculate_spatial_turbulence(
            csi_data, selected_subcarriers
        )
        self._l1.process_amplitudes(
            self._context._amplitude_buffer,
            self._context._amplitude_count,
        )
        self._context.add_turbulence(turbulence)

    def _turb_autocorr(self):
        ctx = self._context
        count = ctx.buffer_count
        values = self._ordered_turbulence
        if count < ctx.window_size:
            for i in range(count):
                values[i] = ctx.turbulence_buffer[i]
        else:
            for i in range(count):
                values[i] = ctx.turbulence_buffer[(ctx.buffer_index + i) % count]
        mean = sum(values[:count]) / count if count else 0.0
        variance = 0.0
        for i in range(count):
            diff = values[i] - mean
            variance += diff * diff
        variance = variance / count if count else 0.0
        return calc_autocorrelation(values, count, mean=mean, variance=variance)

    def _calculate_logit(self, l1_delta, turb_autocorr):
        l1_norm = (l1_delta - self.FEATURE_CENTER[0]) / self.FEATURE_SCALE[0]
        autocorr_norm = (
            (turb_autocorr - self.FEATURE_CENTER[1]) / self.FEATURE_SCALE[1]
        )
        return (
            self.INTERCEPT
            + self.FEATURE_WEIGHT[0] * l1_norm
            + self.FEATURE_WEIGHT[1] * autocorr_norm
        )

    def update_state(self):
        if not self.is_ready():
            self._current_probability = 0.0
            self._state = MotionState.IDLE
        else:
            self._current_l1_delta = self._l1.mean()
            self._current_turb_autocorr = self._turb_autocorr()
            self._current_logit = self._calculate_logit(
                self._current_l1_delta,
                self._current_turb_autocorr,
            )
            self._current_probability = self._sigmoid(self._current_logit)
            if len(self._startup_logits) < self.STARTUP_SAMPLE_LIMIT:
                self._startup_logits.append(self._current_logit)
            self._state = (
                MotionState.MOTION
                if self._current_probability > self._threshold
                else MotionState.IDLE
            )

        return {
            "state": self._state,
            "motion_metric": self._current_probability,
            "probability": self._current_probability,
            "l1_delta": self._current_l1_delta,
            "turb_autocorr": self._current_turb_autocorr,
            "threshold": self._threshold,
        }

    def set_adaptive_threshold(self, _shared_threshold):
        session_q95 = self._quantile(self._startup_logits, self.STARTUP_QUANTILE)
        if session_q95 is None:
            self._threshold = self.BASE_THRESHOLD
            return
        base_logit = math.log(self.BASE_THRESHOLD / (1.0 - self.BASE_THRESHOLD))
        adapted_logit = base_logit + self.STARTUP_STRENGTH * (
            session_q95 - self.TRAIN_IDLE_Q95_LOGIT
        )
        self._threshold = self._sigmoid(adapted_logit)

    def on_startup_calibration_begin(self):
        """Discard stale runtime logits before a fresh calibration session."""
        self._startup_logits = []

    def set_threshold(self, threshold):
        value = float(threshold)
        if value < 0.0 or value > 1.0:
            return False
        self._threshold = value
        return True

    def get_threshold(self):
        return self._threshold

    def get_motion_metric(self):
        return self._current_probability

    def get_state(self):
        return self._state

    def is_ready(self):
        return (
            self._context.buffer_count >= self._context.window_size
            and self._l1.is_ready()
        )

    def reset(self):
        self._context.reset(full=True)
        self._l1.reset()
        self._state = MotionState.IDLE
        self._packet_count = 0
        self._current_probability = 0.0
        self._current_logit = 0.0
        self._current_l1_delta = 0.0
        self._current_turb_autocorr = 0.0
        self._startup_logits = []

    def get_name(self):
        return "Classic"

    @property
    def total_packets(self):
        return self._packet_count
