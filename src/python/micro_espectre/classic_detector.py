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
    from src.csi_features import L1_DELTA_LAG, L1DeltaTracker, calc_autocorrelation
    from src.segmentation import SegmentationContext
except ImportError:
    from detector_interface import IDetector, MotionState
    from csi_features import L1_DELTA_LAG, L1DeltaTracker, calc_autocorrelation
    from segmentation import SegmentationContext


# Lower than any reachable logit, so the first sample of a block wins.
_SETTLE_FLOOR = -1e9


class ClassicDetector(IDetector):
    """Weighted ``l1_delta + turb_autocorr`` production detector."""

    ALGORITHM = "classic"
    STARTUP_GATE = True

    # Grouped, de-overlapped OOF fit, balanced by class/chip/session.
    FEATURE_CENTER = (1.4372828727159759, 0.3899157842282158)
    FEATURE_SCALE = (0.5846221043293537, 0.3789361406116048)
    FEATURE_WEIGHT = (2.807005032259383, 4.0307753529344765)
    INTERCEPT = 0.7924447436944712

    BASE_THRESHOLD = 0.8090618336447031
    TRAIN_IDLE_Q95_LOGIT = -0.6116129330770868
    STARTUP_QUANTILE = 0.95
    STARTUP_STRENGTH = 0.75
    STARTUP_SAMPLE_LIMIT = 64

    # Settled-level rule: how long the stream has to stay quiet before the
    # startup threshold is allowed to come down, and by how much margin above
    # the level it settled at. 12 blocks of 20 evaluations is 60 s at the
    # nominal cadence. The margin is in logit units; 3.0 is the largest value
    # that still recovers the ESP32 capture, and below 2.0 the empty-room
    # recordings start to alarm.
    SETTLE_BLOCKS = 12
    SETTLE_BLOCK_EVALUATIONS = 20
    SETTLE_MARGIN_LOGITS = 3.0

    # The detector owns its startup threshold formula; the shared multiplier is
    # retained only for the generic calibrator's progress/fallback machinery.
    STARTUP_THRESHOLD_FACTOR = 1.0

    def __init__(self, window_size=100, threshold=BASE_THRESHOLD,
                 enable_lowpass=False, lowpass_cutoff=11.0,
                 enable_hampel=True, hampel_window=7, hampel_threshold=5.0,
                 lag=None, autocorr_lag=1,
                 **_unused):
        # ``lag`` is the profile-displacement distance in packets and
        # ``autocorr_lag`` the turbulence autocorrelation distance. Both default
        # to the nominal-rate constants; callers that know the measured cadence
        # pass the counts spanning L1_DELTA_LAG_US and TURB_AUTOCORR_LAG_US
        # instead, because both quantities are functions of the elapsed interval
        # rather than of how many packets happen to fall inside it.
        lag = L1_DELTA_LAG if lag is None else max(1, int(lag))
        self._lag = lag
        self._autocorr_lag = max(1, int(autocorr_lag))
        self._context = SegmentationContext(
            window_size=window_size,
            enable_lowpass=enable_lowpass,
            lowpass_cutoff=lowpass_cutoff,
            enable_hampel=enable_hampel,
            hampel_window=hampel_window,
            hampel_threshold=hampel_threshold,
        )
        self._l1 = L1DeltaTracker(
            window_size=max(2, window_size - lag),
            lag=lag,
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
        self._current_lag_ratio = 0.0
        self._current_turb_autocorr = 0.0
        self._startup_logits = []
        self._adapted_threshold_ready = False
        self._settle_blocks = []
        self._settle_block_max = _SETTLE_FLOOR
        self._settle_block_count = 0

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

    def process_packet(self, csi_data, selected_subcarriers=None, rssi_dbm=None):
        """Process one CSI packet. ``rssi_dbm`` is accepted for interface parity
        and ignored: both Classic features are already invariant to link gain."""
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
        return calc_autocorrelation(
            values, count, mean=mean, variance=variance, lag=self._autocorr_lag
        )

    def _calculate_logit(self, lag_ratio, turb_autocorr):
        l1_norm = (lag_ratio - self.FEATURE_CENTER[0]) / self.FEATURE_SCALE[0]
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
            self._current_lag_ratio = self._l1.delta_lag_ratio()
            self._current_turb_autocorr = self._turb_autocorr()
            self._current_logit = self._calculate_logit(
                self._current_lag_ratio,
                self._current_turb_autocorr,
            )
            self._current_probability = self._sigmoid(self._current_logit)
            if len(self._startup_logits) < self.STARTUP_SAMPLE_LIMIT:
                self._startup_logits.append(self._current_logit)
            self._observe_settled_level()
            self._state = (
                MotionState.MOTION
                if self._current_probability > self._threshold
                else MotionState.IDLE
            )

        return {
            "state": self._state,
            "motion_metric": self._current_probability,
            "probability": self._current_probability,
            "lag_ratio": self._current_lag_ratio,
            "turb_autocorr": self._current_turb_autocorr,
            "threshold": self._threshold,
        }

    def _observe_settled_level(self):
        """Lower the threshold once the session proves itself quieter than startup.

        Startup calibration reads the opening of a session. When that opening is
        not representative the threshold stays too high for the whole run, and
        nothing revisits it: on one ESP32 capture the prefix is 4.1x noisier than
        the rest, leaving the threshold at 3.8x the highest level the session
        ever reaches, and 4.7 points of recall on the table.

        The rule only ever lowers. It reads the median of per-block maxima, so a
        single spike cannot move it and a stretch of real motion holds it high,
        which is what keeps it from chasing the metric downward during activity.
        Blocks rather than a full history keep it to `SETTLE_BLOCKS` floats.
        """
        if not self._adapted_threshold_ready:
            return
        if self._current_logit > self._settle_block_max:
            self._settle_block_max = self._current_logit
        self._settle_block_count += 1
        if self._settle_block_count < self.SETTLE_BLOCK_EVALUATIONS:
            return

        self._settle_blocks.append(self._settle_block_max)
        self._settle_block_max = _SETTLE_FLOOR
        self._settle_block_count = 0
        if len(self._settle_blocks) > self.SETTLE_BLOCKS:
            del self._settle_blocks[0]
        if len(self._settle_blocks) < self.SETTLE_BLOCKS:
            return

        ordered = sorted(self._settle_blocks)
        settled = ordered[len(ordered) // 2]
        candidate = self._sigmoid(settled + self.SETTLE_MARGIN_LOGITS)
        if candidate < self._threshold:
            self._threshold = self._clamp_probability(candidate)

    def _reset_settled_level(self):
        """Drop the settled-level evidence, so a lowering has to be re-earned."""
        self._settle_blocks = []
        self._settle_block_max = _SETTLE_FLOOR
        self._settle_block_count = 0

    def set_adaptive_threshold(self, _shared_threshold):
        self._reset_settled_level()
        self._adapted_threshold_ready = True
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
        self._adapted_threshold_ready = False
        self._settle_blocks = []
        self._settle_block_max = _SETTLE_FLOOR
        self._settle_block_count = 0

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
        self._current_lag_ratio = 0.0
        self._current_turb_autocorr = 0.0
        self._startup_logits = []
        self._reset_settled_level()

    def get_name(self):
        return "Classic"

    @property
    def total_packets(self):
        return self._packet_count
