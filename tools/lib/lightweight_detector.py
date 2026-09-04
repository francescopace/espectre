# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - Reference Lightweight Detector

Vote-free, two-feature motion detector using turbulence autocorrelation and the
robust spread of a five-bin aggregated turbulence stream.

Author: Francesco Pace <francesco.pace@gmail.com>
"""
import math

from micro_espectre.config import DEFAULT_SUBCARRIERS
from micro_espectre.detector_interface import IDetector, MotionState

try:
    from .csi_features import TURB_IQR_AGGREGATION_WIDTH
    from .segmentation import SegmentationContext
except ImportError:
    from csi_features import TURB_IQR_AGGREGATION_WIDTH
    from segmentation import SegmentationContext


# Lower than any reachable logit, so the first sample of a block wins.
_SETTLE_FLOOR = -1e9
_SORT_SENTINEL = float("inf")
# One shared tagged-integer tuple is cheaper than two signed-byte branches for
# every magnitude in the unsigned bytes/bytearray packet path.
_SIGNED_INT8_VALUES = tuple(range(128)) + tuple(range(-128, 0))


class LightweightDetector(IDetector):
    """Weighted ``turb_autocorr + turb_iqr_over_mean_aggr`` detector."""

    ALGORITHM = "lightweight"
    STARTUP_GATE = True

    # Grouped, de-overlapped OOF fit, balanced by class/chip/session.
    FEATURE_CENTER = (0.3919344866784947, 0.24612139211074338)
    FEATURE_SCALE = (0.3798648330757351, 0.20056599613462603)
    FEATURE_WEIGHT = (5.083034533668216, 4.997501915217463)
    INTERCEPT = 1.0776769868761

    BASE_THRESHOLD = 0.6621854538596202
    TRAIN_IDLE_Q95_LOGIT = -2.253902812716911
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
                 autocorr_lag=1):
        self._autocorr_lag = max(1, int(autocorr_lag))
        self._context = SegmentationContext(
            window_size=window_size,
            enable_lowpass=enable_lowpass,
            lowpass_cutoff=lowpass_cutoff,
            enable_hampel=enable_hampel,
            hampel_window=hampel_window,
            hampel_threshold=hampel_threshold,
            track_lag1_autocorrelation=(self._autocorr_lag == 1),
        )
        self._aggregated_context = SegmentationContext(
            window_size=window_size,
            enable_lowpass=enable_lowpass,
            lowpass_cutoff=lowpass_cutoff,
            enable_hampel=enable_hampel,
            hampel_window=hampel_window,
            hampel_threshold=hampel_threshold,
            adjacent_aggregation_width=TURB_IQR_AGGREGATION_WIDTH,
        )
        self._selected_subcarriers = DEFAULT_SUBCARRIERS
        self._amplitude_plan, self._amplitude_plan_max_offset = (
            self._build_amplitude_plan(self._selected_subcarriers)
        )
        self._csi_values_signed = None
        self._valid_turbulence_scratch = [0.0] * window_size
        self._minimum_valid_samples = window_size
        self._threshold = self._clamp_probability(threshold)
        self._state = MotionState.IDLE
        self._packet_count = 0
        self._current_probability = 0.0
        self._current_logit = 0.0
        self._current_turb_autocorr = 0.0
        self._current_turb_iqr_over_mean_aggr = 0.0
        self._startup_logits = []
        self._adapted_threshold_ready = False
        self._manual_threshold_override = False
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
        return LightweightDetector._quantile_sorted(ordered, quantile)

    @staticmethod
    def _quantile_sorted(ordered, quantile, count=None):
        """Interpolate a quantile from an already sorted non-empty sequence."""
        size = len(ordered) if count is None else int(count)
        if size == 1:
            return ordered[0]
        position = (size - 1) * quantile
        lower = int(position)
        upper = min(lower + 1, size - 1)
        fraction = position - lower
        return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction

    @staticmethod
    def _build_amplitude_plan(selected_subcarriers):
        """Precompute direct and adjacent-bin offsets for the hot packet path."""
        width = TURB_IQR_AGGREGATION_WIDTH
        half = (width - 1) // 2
        plan = []
        max_offset = 0
        for selected in selected_subcarriers:
            selected = int(selected)
            low = selected - half
            high = selected + (width - 1 - half)
            if low < 4:
                low, high = 4, 4 + width - 1
            if high > 60:
                low, high = 60 - width + 1, 60

            neighbour_offsets = []
            aggregate_count = 0
            for subcarrier in range(low, high + 1):
                if subcarrier == 32:
                    continue
                aggregate_count += 1
                if subcarrier != selected:
                    offset = subcarrier * 2
                    neighbour_offsets.append(offset)
                    if offset > max_offset:
                        max_offset = offset
            selected_offset = selected * 2
            if selected_offset > max_offset:
                max_offset = selected_offset
            plan.append(
                (
                    selected_offset,
                    tuple(neighbour_offsets),
                    aggregate_count,
                    selected != 32 and low <= selected <= high,
                )
            )
        return tuple(plan), max_offset + 1

    def process_packet(self, csi_data, selected_subcarriers=None, rssi_dbm=None,
                       timestamp_us=None):
        """Process one CSI packet. ``rssi_dbm`` is accepted for interface parity
        and ignored: both Lightweight features are already invariant to link gain."""
        self._packet_count += 1
        del timestamp_us
        if selected_subcarriers is None:
            selected_subcarriers = DEFAULT_SUBCARRIERS
        if selected_subcarriers is not self._selected_subcarriers:
            selected_subcarriers = tuple(selected_subcarriers)
            if selected_subcarriers != self._selected_subcarriers:
                self._selected_subcarriers = selected_subcarriers
                self._amplitude_plan, self._amplitude_plan_max_offset = (
                    self._build_amplitude_plan(selected_subcarriers)
                )
        if len(csi_data) <= self._amplitude_plan_max_offset:
            return

        normal_values = self._context._amplitude_buffer
        aggregate_values = self._aggregated_context._amplitude_buffer
        sqrt = math.sqrt
        normal_total = 0.0
        aggregate_total = 0.0
        packet_count = 0
        if self._csi_values_signed is None:
            try:
                self._csi_values_signed = memoryview(csi_data).format == "b"
            except (AttributeError, TypeError):
                self._csi_values_signed = False
        if self._csi_values_signed:
            for (
                selected_offset,
                neighbour_offsets,
                aggregate_count,
                include_selected,
            ) in self._amplitude_plan:
                imag = int(csi_data[selected_offset])
                real = int(csi_data[selected_offset + 1])
                selected_amplitude = sqrt(real * real + imag * imag)
                normal_values[packet_count] = selected_amplitude
                normal_total += selected_amplitude
                total = selected_amplitude if include_selected else 0.0
                for offset in neighbour_offsets:
                    imag = int(csi_data[offset])
                    real = int(csi_data[offset + 1])
                    total += sqrt(real * real + imag * imag)
                aggregated_amplitude = total / aggregate_count
                aggregate_values[packet_count] = aggregated_amplitude
                aggregate_total += aggregated_amplitude
                packet_count += 1
        else:
            signed_values = _SIGNED_INT8_VALUES
            for (
                selected_offset,
                neighbour_offsets,
                aggregate_count,
                include_selected,
            ) in self._amplitude_plan:
                imag = signed_values[csi_data[selected_offset]]
                real = signed_values[csi_data[selected_offset + 1]]
                selected_amplitude = sqrt(real * real + imag * imag)
                normal_values[packet_count] = selected_amplitude
                normal_total += selected_amplitude
                total = selected_amplitude if include_selected else 0.0
                for offset in neighbour_offsets:
                    imag = signed_values[csi_data[offset]]
                    real = signed_values[csi_data[offset + 1]]
                    total += sqrt(real * real + imag * imag)
                aggregated_amplitude = total / aggregate_count
                aggregate_values[packet_count] = aggregated_amplitude
                aggregate_total += aggregated_amplitude
                packet_count += 1
        if packet_count == 0:
            return
        normal_mean = normal_total / packet_count
        aggregate_mean = aggregate_total / packet_count
        self._context._amplitude_count = packet_count
        self._context._amplitude_mean = normal_mean
        turbulence = self._context._turbulence_from_amplitude_buffer(
            normal_values, packet_count, normal_mean
        )
        self._context.add_turbulence(turbulence)
        self._aggregated_context._amplitude_count = packet_count
        self._aggregated_context._amplitude_mean = aggregate_mean
        aggregated = self._aggregated_context._turbulence_from_amplitude_buffer(
            aggregate_values,
            packet_count,
            aggregate_mean,
        )
        self._aggregated_context.add_turbulence(aggregated)

    def advance_missing_slots(self, count):
        """Preserve missing temporal slots in both feature rings."""
        for _ in range(max(0, int(count))):
            self._context.add_missing_slot()
            self._aggregated_context.add_missing_slot()

    def set_minimum_valid_samples(self, count):
        self._minimum_valid_samples = max(1, min(int(count), self._context.window_size))

    def _turb_autocorr(self):
        ctx = self._context
        if self._autocorr_lag == 1:
            return ctx.lag1_autocorrelation()
        count = ctx.buffer_count
        valid_count = ctx.valid_count
        lag = self._autocorr_lag
        if count < lag + 2 or valid_count < lag + 1:
            return 0.0

        values = ctx.turbulence_buffer
        validity = ctx.validity_buffer
        window_size = ctx.window_size
        start = ctx.buffer_index if count >= window_size else 0
        source = start
        total = 0.0
        for _ in range(count):
            if validity[source]:
                total += values[source]
            source += 1
            if source >= window_size:
                source = 0
        mean = total / valid_count

        variance_sum = 0.0
        autocovariance = 0.0
        pair_count = 0
        source = start
        for offset in range(count):
            if validity[source]:
                diff = values[source] - mean
                variance_sum += diff * diff
                if offset >= lag:
                    previous = source - lag
                    if previous < 0:
                        previous += window_size
                    if validity[previous]:
                        autocovariance += (values[previous] - mean) * diff
                        pair_count += 1
            source += 1
            if source >= window_size:
                source = 0

        variance = variance_sum / valid_count
        if variance < 1e-10 or pair_count == 0:
            return 0.0
        return (autocovariance / pair_count) / variance

    def _turb_iqr_over_mean_aggr(self):
        ctx = self._aggregated_context
        values = ctx.turbulence_buffer
        validity = ctx.validity_buffer
        count = ctx.buffer_count
        ordered = self._valid_turbulence_scratch
        window_size = ctx.window_size
        source = ctx.buffer_index if count >= window_size else 0
        valid_count = 0
        total = 0.0
        for _ in range(count):
            if validity[source]:
                value = values[source]
                ordered[valid_count] = value
                valid_count += 1
                total += value
            source += 1
            if source >= window_size:
                source = 0
        if not valid_count:
            return 0.0
        mean = total / valid_count
        for index in range(valid_count, len(ordered)):
            ordered[index] = _SORT_SENTINEL
        ordered.sort()
        q25 = self._quantile_sorted(ordered, 0.25, valid_count)
        q75 = self._quantile_sorted(ordered, 0.75, valid_count)
        return (q75 - q25) / max(abs(mean), 1e-6)

    def _calculate_logit(self, turb_autocorr, turb_iqr_over_mean_aggr):
        autocorr_norm = (
            (turb_autocorr - self.FEATURE_CENTER[0]) / self.FEATURE_SCALE[0]
        )
        iqr_norm = (
            (turb_iqr_over_mean_aggr - self.FEATURE_CENTER[1])
            / self.FEATURE_SCALE[1]
        )
        return (
            self.INTERCEPT
            + self.FEATURE_WEIGHT[0] * autocorr_norm
            + self.FEATURE_WEIGHT[1] * iqr_norm
        )

    def update_state(self):
        if not self.is_ready():
            self._current_probability = 0.0
            self._state = MotionState.IDLE
        else:
            self._current_turb_autocorr = self._turb_autocorr()
            self._current_turb_iqr_over_mean_aggr = self._turb_iqr_over_mean_aggr()
            self._current_logit = self._calculate_logit(
                self._current_turb_autocorr,
                self._current_turb_iqr_over_mean_aggr,
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
            "turb_iqr_over_mean_aggr": self._current_turb_iqr_over_mean_aggr,
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
        if not self._adapted_threshold_ready or self._manual_threshold_override:
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
        self._manual_threshold_override = False
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
        self._manual_threshold_override = False
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
        self._manual_threshold_override = True
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
            and self._context.valid_count >= self._minimum_valid_samples
            and self._aggregated_context.buffer_count
            >= self._aggregated_context.window_size
            and self._aggregated_context.valid_count >= self._minimum_valid_samples
        )

    def reset(self):
        self._context.reset(full=True)
        self._aggregated_context.reset(full=True)
        self._state = MotionState.IDLE
        self._packet_count = 0
        self._current_probability = 0.0
        self._current_logit = 0.0
        self._current_turb_autocorr = 0.0
        self._current_turb_iqr_over_mean_aggr = 0.0
        self._startup_logits = []
        self._reset_settled_level()

    def get_name(self):
        return "Lightweight"

    def get_window_size(self):
        """Return the resolved detector window in samples."""
        return self._context.window_size

    @property
    def total_packets(self):
        return self._packet_count
