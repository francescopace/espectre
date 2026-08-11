"""
Micro-ESPectre - Classic Detector

Vote-free, two-feature motion detector using a weighted fusion of turbulence
autocorrelation and channel frequency-coherence curve spread. Hampel filtering
still applies independently to the turbulence and legacy L1 tracking paths under
one shared enable flag because the segmentation context continues to own both
buffers.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""
import math

try:
    from src.detector_interface import IDetector, MotionState
    from src.csi_features import L1_DELTA_LAG, L1DeltaTracker, calc_autocorrelation
    from src.ml_feature_trackers import ChannelShapeTracker
    from src.segmentation import SegmentationContext
except ImportError:
    from detector_interface import IDetector, MotionState
    from csi_features import L1_DELTA_LAG, L1DeltaTracker, calc_autocorrelation
    from ml_feature_trackers import ChannelShapeTracker
    from segmentation import SegmentationContext


# Lower than any reachable logit, so the first sample of a block wins.
_SETTLE_FLOOR = -1e9


class ClassicDetector(IDetector):
    """Weighted ``turb_autocorr + chan_freq_coh_curve_std`` detector."""

    ALGORITHM = "classic"
    STARTUP_GATE = True

    # Grouped, de-overlapped OOF fit, balanced by class/chip/session.
    FEATURE_CENTER = (0.40183487675618096, 0.013573794185191685)
    FEATURE_SCALE = (0.37890037481307803, 0.023719479149528277)
    FEATURE_WEIGHT = (5.318553379383947, 2.937413738610618)
    INTERCEPT = 0.07498562105607867

    BASE_THRESHOLD = 0.7456011395202353
    TRAIN_IDLE_Q95_LOGIT = -0.5638467984849406
    STARTUP_QUANTILE = 0.95
    STARTUP_STRENGTH = 0.5
    STARTUP_SAMPLE_LIMIT = 64

    # Settled-level rule: how long the stream has to stay quiet before the
    # startup threshold is allowed to come down, and by how much margin above
    # the level it settled at. 12 blocks of 20 evaluations is 60 s at the
    # nominal cadence. The margin is in logit units; 2.7 is the conservative
    # temporal-window operating point that clears the weak-link recall floor
    # without changing the measured normal-link or quiet-room FP tails.
    SETTLE_BLOCKS = 12
    SETTLE_BLOCK_EVALUATIONS = 20
    SETTLE_MARGIN_LOGITS = 2.7

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
        self._shape_tracker = ChannelShapeTracker(
            window_size=max(2, window_size - lag),
            lag=lag,
        )
        self._ordered_turbulence = [0.0] * window_size
        self._threshold = self._clamp_probability(threshold)
        self._state = MotionState.IDLE
        self._packet_count = 0
        self._current_probability = 0.0
        self._current_logit = 0.0
        self._current_turb_autocorr = 0.0
        self._current_chan_freq_coh_curve_std = 0.0
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

    def process_packet(self, csi_data, selected_subcarriers=None, rssi_dbm=None,
                       timestamp_us=None):
        """Process one CSI packet. ``rssi_dbm`` is accepted for interface parity
        and ignored: both Classic features are already invariant to link gain."""
        self._packet_count += 1
        del timestamp_us
        turbulence = self._context.calculate_spatial_turbulence(
            csi_data, selected_subcarriers
        )
        self._l1.process_amplitudes(
            self._context._amplitude_buffer,
            self._context._amplitude_count,
        )
        self._shape_tracker.process_packet(csi_data)
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

    def _calculate_logit(self, turb_autocorr, chan_freq_coh_curve_std):
        autocorr_norm = (
            (turb_autocorr - self.FEATURE_CENTER[0]) / self.FEATURE_SCALE[0]
        )
        curve_std_norm = (
            (chan_freq_coh_curve_std - self.FEATURE_CENTER[1])
            / self.FEATURE_SCALE[1]
        )
        return (
            self.INTERCEPT
            + self.FEATURE_WEIGHT[0] * autocorr_norm
            + self.FEATURE_WEIGHT[1] * curve_std_norm
        )

    def update_state(self):
        if not self.is_ready():
            self._current_probability = 0.0
            self._state = MotionState.IDLE
        else:
            self._current_turb_autocorr = self._turb_autocorr()
            self._current_chan_freq_coh_curve_std = (
                self._shape_tracker.frequency_coherence_curve_std()
            )
            self._current_logit = self._calculate_logit(
                self._current_turb_autocorr,
                self._current_chan_freq_coh_curve_std,
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
            "turb_autocorr": self._current_turb_autocorr,
            "chan_freq_coh_curve_std": self._current_chan_freq_coh_curve_std,
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
            and self._shape_tracker.count() >= self._context.window_size - self._lag
        )

    def reset(self):
        self._context.reset(full=True)
        self._l1.reset()
        self._shape_tracker.reset()
        self._state = MotionState.IDLE
        self._packet_count = 0
        self._current_probability = 0.0
        self._current_logit = 0.0
        self._current_turb_autocorr = 0.0
        self._current_chan_freq_coh_curve_std = 0.0
        self._startup_logits = []
        self._reset_settled_level()

    def get_name(self):
        return "Classic"

    def get_window_size(self):
        """Return the resolved detector window in samples."""
        return self._context.window_size

    @property
    def total_packets(self):
        return self._packet_count
