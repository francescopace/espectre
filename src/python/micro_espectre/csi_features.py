"""
Micro-ESPectre - Shared Feature And L1-Delta Helpers

Pure Python implementation for MicroPython.
Provides the production ML feature extraction helpers plus the allocation-free
L1-delta tracker used by the classic detector and offline tooling.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""
import math

try:
    from src.detector_interface import MotionState
    from src.filters import HampelFilter
    from src.segmentation import SegmentationContext
except ImportError:
    from detector_interface import MotionState
    from filters import HampelFilter
    from segmentation import SegmentationContext

# Match the detector path: compare normalized profiles 10 packets apart
# (~100 ms at 100 pps).
L1_DELTA_LAG = 10
L1_DELTA_STARTUP_THRESHOLD_FACTOR = 1.1
L1_DELTA_STARTUP_GATE = True


def calc_skewness(values, count, mean, std):
    """Calculate Fisher skewness (3rd standardized moment)."""
    if count < 3 or std < 1e-10:
        return 0.0

    m3 = 0.0
    for i in range(count):
        diff = values[i] - mean
        m3 += diff * diff * diff
    m3 /= count
    return m3 / (std * std * std)


def _interpolate_sorted_percentile(sorted_values, count, percentile):
    """Calculate percentile from an already sorted list."""
    if count == 0:
        return 0.0
    if count == 1:
        return sorted_values[0]

    position = (count - 1) * (percentile / 100.0)
    lower_idx = int(position)
    upper_idx = lower_idx + 1
    if upper_idx >= count:
        return sorted_values[count - 1]

    fraction = position - lower_idx
    lower = sorted_values[lower_idx]
    upper = sorted_values[upper_idx]
    return lower * (1.0 - fraction) + upper * fraction


def calc_iqr(turbulence_buffer, buffer_count, sorted_values=None):
    """Calculate interquartile range (P75 - P25).

    Args:
        sorted_values: Pre-sorted copy to avoid redundant sorting.
    """
    if buffer_count < 2:
        return 0.0

    if sorted_values is None:
        sorted_vals = turbulence_buffer[:buffer_count]
        sorted_vals.sort()
    else:
        sorted_vals = sorted_values

    q1 = _interpolate_sorted_percentile(sorted_vals, buffer_count, 25.0)
    q3 = _interpolate_sorted_percentile(sorted_vals, buffer_count, 75.0)
    return q3 - q1


def calc_autocorrelation(turbulence_buffer, buffer_count, mean=None, variance=None, lag=1):
    """Calculate lag-k autocorrelation coefficient."""
    if buffer_count < lag + 2:
        return 0.0

    if mean is None:
        total = 0.0
        for i in range(buffer_count):
            total += turbulence_buffer[i]
        mean = total / buffer_count

    if variance is None:
        variance = 0.0
        for i in range(buffer_count):
            diff = turbulence_buffer[i] - mean
            variance += diff * diff
        variance /= buffer_count

    if variance < 1e-10:
        return 0.0

    autocovariance = 0.0
    for i in range(buffer_count - lag):
        autocovariance += (turbulence_buffer[i] - mean) * (turbulence_buffer[i + lag] - mean)
    autocovariance /= (buffer_count - lag)
    return autocovariance / variance


def calc_mad(turbulence_buffer, buffer_count, sorted_values=None):
    """Calculate median absolute deviation (MAD).

    Args:
        sorted_values: Pre-sorted copy to avoid redundant sorting.
    """
    if buffer_count < 2:
        return 0.0

    if sorted_values is None:
        sorted_vals = turbulence_buffer[:buffer_count]
        sorted_vals.sort()
    else:
        sorted_vals = sorted_values

    mid = buffer_count // 2
    if buffer_count % 2 == 0:
        median = (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
    else:
        median = sorted_vals[mid]

    for i in range(buffer_count):
        sorted_vals[i] = abs(sorted_vals[i] - median)
    sorted_vals.sort()

    if buffer_count % 2 == 0:
        return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
    return sorted_vals[mid]


def calc_waveform_length(turbulence_buffer, buffer_count):
    """Calculate waveform length as total absolute first-difference."""
    if buffer_count < 2:
        return 0.0

    total = 0.0
    prev = turbulence_buffer[0]
    for i in range(1, buffer_count):
        curr = turbulence_buffer[i]
        total += abs(curr - prev)
        prev = curr
    return total


def calc_zero_crossing_rate(values, count, center):
    """
    Calculate the crossing rate of ``values`` around ``center``.

    Shift and scale invariant when ``center`` tracks the window (median):
    white noise crosses its median almost every sample, while temporally
    coherent motion excursions stay on one side for long runs.
    """
    if count < 2:
        return 0.0
    crossings = 0
    prev_above = values[0] >= center
    for i in range(1, count):
        curr_above = values[i] >= center
        if curr_above != prev_above:
            crossings += 1
            prev_above = curr_above
    return crossings / (count - 1)


# Supported L1-delta features available to training/export flows.
L1_DELTA_FEATURES = [
    'l1_delta',
    'l1_delta_std',
    'l1_delta_waveform_length',
    'l1_delta_autocorr',
    'l1_delta_lag_ratio',
]

# Historical Core-6 production set (2026-07-07 to 2026-07-23); kept for
# reference and experiments. Superseded by Coherence-6 (see below).
CORE6_FEATURES = [
    'turb_mad_over_mean',
    'turb_skewness',
    'turb_autocorr',
    'l1_delta',
    'l1_delta_std',
    'l1_delta_waveform_length',
]

# Coherence-6 production set (2026-07-23 to 2026-07-27): Core-6 with the two
# weakest members swapped for shift/scale-invariant temporal-coherence
# statistics. On real weak-link pairs the absolute L1 features lose (or invert)
# their motion separation, while the noise floor is white in time and human
# motion is not; the coherence swap collapses seed-to-seed variance on
# out-of-sample false positives (see the temporal-coherence promotion ADR).
# Kept for reference and experiments. Superseded by Coherence-7 (see below).
COHERENCE6_FEATURES = [
    'turb_mad_over_mean',
    'turb_autocorr',
    'turb_zcr',
    'l1_delta',
    'l1_delta_std',
    'l1_delta_autocorr',
]

# Coherence-7 production set: Coherence-6 plus the lag ratio Classic adopted,
# which divides the lagged profile displacement by the adjacent one. The plain
# mean carries the link's noise floor, so it degrades and sometimes inverts on
# weak links; the ratio shares a unit with its denominator and drops the floor.
# Adding it removed five effective alarms across the reserved replays and added
# none.
COHERENCE7_FEATURES = COHERENCE6_FEATURES + [
    'l1_delta_lag_ratio',
]

# Production feature set.
DEFAULT_FEATURES = COHERENCE7_FEATURES

# Non-production features available to training experiments: the two members
# demoted from Core-6 and the never-promoted coefficient of variation.
CANDIDATE_FEATURES = [
    'turb_skewness',
    'l1_delta_waveform_length',
    'l1_delta_cv',
]

ALL_FEATURES = tuple(DEFAULT_FEATURES + CANDIDATE_FEATURES)

# Features computed from the rebuilt L1-delta series. The lag ratio is
# deliberately absent: it shares the l1_ prefix but the tracker hands it over
# ready-made, so requesting it alone must not demand a series. Mirrors
# MLFeatureSource in csi_features.h; keep the two in step.
L1_SERIES_FEATURES = frozenset({
    'l1_delta',
    'l1_delta_std',
    'l1_delta_autocorr',
    'l1_delta_waveform_length',
    'l1_delta_cv',
})

# Features that need the L1 tracker running at all. The lag ratio belongs here
# but not above: it needs the profile rings, not the rebuilt series.
L1_TRACKER_FEATURES = L1_SERIES_FEATURES | {'l1_delta_lag_ratio'}


def normalize_amplitude_profile_into(amplitudes, count, out):
    """
    Write the mean-normalized amplitude profile into ``out[:count]``.

    Shared numeric core for the L1-delta detector and the ML feature path;
    allocation-free so device hot paths can reuse pre-allocated buffers.

    Returns:
        int: Number of values written (0 when the profile is invalid).
    """
    if amplitudes is None or count < 2 or count > len(out):
        return 0
    total = 0.0
    for i in range(count):
        total += amplitudes[i]
    if total <= 0.0:
        return 0
    mean = total / count
    for i in range(count):
        out[i] = amplitudes[i] / mean
    return count


def _normalize_amplitude_profile(amplitudes):
    """Return a mean-normalized amplitude profile as a new list, or None if invalid."""
    if amplitudes is None:
        return None
    count = len(amplitudes)
    out = [0.0] * count
    if normalize_amplitude_profile_into(amplitudes, count, out) == 0:
        return None
    return out


def l1_delta_series(amplitude_history, buffer_count, lag=L1_DELTA_LAG):
    """
    Return the per-packet L1 normalized profile displacement series.

    This is the raw `d` stream the shared L1-delta tracker averages into the
    classic primary motion metric:
    1. normalize each per-packet amplitude profile by its mean
    2. compare each profile with the one `lag` packets earlier
    3. emit the mean absolute per-subcarrier displacement for that pair

    Exposing the series (not just its mean) lets the ML feature path build a
    full statistical descriptor on the L1-delta axis, mirroring what the
    turbulence path already does on the turbulence buffer. Returns an empty
    list when there is no comparable lagged profile.
    """
    if amplitude_history is None:
        return []
    n = min(int(buffer_count), len(amplitude_history))
    if n < lag + 1:
        return []

    normalized_profiles = [None] * n
    for i in range(n):
        normalized_profiles[i] = _normalize_amplitude_profile(amplitude_history[i])

    deltas = []
    for i in range(lag, n):
        current = normalized_profiles[i]
        reference = normalized_profiles[i - lag]
        if current is None or reference is None or len(current) != len(reference):
            continue

        total = 0.0
        width = len(current)
        for j in range(width):
            diff = current[j] - reference[j]
            total += diff if diff >= 0.0 else -diff
        deltas.append(total / width)

    return deltas


def calc_l1_delta(amplitude_history, buffer_count, lag=L1_DELTA_LAG):
    """
    Calculate the mean L1 normalized profile displacement over a sliding window.

    Matches the shared L1-delta motion metric used by `ClassicDetector`
    (mean of the per-packet displacement series). See `l1_delta_series` for the underlying
    stream.
    """
    deltas = l1_delta_series(amplitude_history, buffer_count, lag)
    if not deltas:
        return 0.0
    return sum(deltas) / len(deltas)


class L1DeltaTracker:
    """Allocation-free L1-delta metric tracker without detector surface."""

    def __init__(self, window_size=100, threshold=1.0, lag=L1_DELTA_LAG,
                 allocate_amplitude_buffer=True, enable_hampel=False,
                 hampel_window=7, hampel_threshold=5.0):
        self.window_size = max(2, int(window_size))
        self.threshold = threshold
        self.lag = max(1, int(lag))

        profile_width = SegmentationContext.AMPLITUDE_BUFFER_SIZE
        self._profile_ring = [[0.0] * profile_width for _ in range(self.lag)]
        self._profile_len = [0] * self.lag
        self._profile_index = 0
        self._current_profile = [0.0] * profile_width
        self._amplitude_buffer = (
            [0.0] * profile_width if allocate_amplitude_buffer else None
        )
        self._amplitude_count = 0
        self._delta_ring = [0.0] * self.window_size
        self._delta_index = 0
        self._delta_count = 0
        self._delta_sum = 0.0
        # Lag-1 displacement over the same window. Both references live in the
        # profile ring already, so the pair costs one extra running sum and no
        # extra normalization.
        self._delta1_ring = [0.0] * self.window_size
        self._delta1_index = 0
        self._delta1_count = 0
        self._delta1_sum = 0.0

        self._packet_count = 0
        self._state = MotionState.IDLE
        self._current_metric = 0.0
        self.last_delta = 0.0
        self._hampel_filter = (
            HampelFilter(hampel_window, hampel_threshold)
            if enable_hampel else None
        )
        # The ratio divides one displacement by the other, so they must be
        # filtered alike: an outlier surviving only in the denominator would
        # depress the ratio and read as less motion.
        self._hampel_filter1 = (
            HampelFilter(hampel_window, hampel_threshold)
            if enable_hampel else None
        )

    def _push_delta(self, delta):
        if self._delta_count < self.window_size:
            self._delta_count += 1
        else:
            self._delta_sum -= self._delta_ring[self._delta_index]
        self._delta_ring[self._delta_index] = delta
        self._delta_sum += delta
        self._delta_index += 1
        if self._delta_index >= self.window_size:
            self._delta_index = 0

    def _push_delta1(self, delta):
        if self._delta1_count < self.window_size:
            self._delta1_count += 1
        else:
            self._delta1_sum -= self._delta1_ring[self._delta1_index]
        self._delta1_ring[self._delta1_index] = delta
        self._delta1_sum += delta
        self._delta1_index += 1
        if self._delta1_index >= self.window_size:
            self._delta1_index = 0

    def process_packet(self, csi_data, selected_subcarriers=None):
        if self._amplitude_buffer is None:
            raise RuntimeError("amplitude buffer is disabled")
        self._amplitude_count = SegmentationContext._fill_amplitude_buffer(
            csi_data, selected_subcarriers, self._amplitude_buffer
        )
        self.process_amplitudes(self._amplitude_buffer, self._amplitude_count)

    def process_amplitudes(self, amplitudes, amplitude_count):
        """Update the L1 stream from an already extracted amplitude profile."""
        self._packet_count += 1
        profile = self._current_profile
        ring_slot = self._profile_index
        reference = self._profile_ring[ring_slot]
        reference_len = self._profile_len[ring_slot]
        # The packet before this one is the slot behind the lagged reference,
        # so the lag-1 displacement needs no storage of its own.
        prev_slot = ring_slot - 1 if ring_slot > 0 else self.lag - 1
        previous = self._profile_ring[prev_slot]
        previous_len = self._profile_len[prev_slot]
        profile_len = 0
        if amplitudes is not None and 2 <= amplitude_count <= len(profile):
            total = 0.0
            for i in range(amplitude_count):
                total += amplitudes[i]
            if total > 0.0:
                mean = total / amplitude_count
                profile_len = amplitude_count
                lagged = reference_len == profile_len
                adjacent = previous_len == profile_len
                if lagged or adjacent:
                    delta_total = 0.0
                    delta1_total = 0.0
                    for i in range(profile_len):
                        value = amplitudes[i] / mean
                        profile[i] = value
                        if lagged:
                            diff = value - reference[i]
                            delta_total += diff if diff >= 0 else -diff
                        if adjacent:
                            diff1 = value - previous[i]
                            delta1_total += diff1 if diff1 >= 0 else -diff1
                    if lagged:
                        self.last_delta = delta_total / profile_len
                        if self._hampel_filter is not None:
                            self.last_delta = self._hampel_filter.filter(self.last_delta)
                        self._push_delta(self.last_delta)
                    if adjacent:
                        delta1 = delta1_total / profile_len
                        if self._hampel_filter1 is not None:
                            delta1 = self._hampel_filter1.filter(delta1)
                        self._push_delta1(delta1)
                else:
                    for i in range(profile_len):
                        profile[i] = amplitudes[i] / mean

        self._profile_ring[ring_slot] = profile
        self._profile_len[ring_slot] = profile_len
        self._current_profile = reference
        self._profile_index += 1
        if self._profile_index >= self.lag:
            self._profile_index = 0

    def delta_lag_ratio(self):
        """Return mean(lag displacement) / mean(lag-1 displacement).

        Noise saturates the displacement immediately, so its ratio sits near
        `1.0`; real channel evolution keeps growing with the lag and lifts it.
        Both terms share the same units, so the ratio drops the noise floor that
        makes the raw mean unusable when the link is weak.
        """
        if self._delta_count == 0 or self._delta1_count == 0:
            return 1.0
        adjacent_mean = self._delta1_sum / self._delta1_count
        if adjacent_mean <= 0.0:
            return 1.0
        return (self._delta_sum / self._delta_count) / adjacent_mean

    def copy_deltas_into(self, out):
        """Copy the current delta window into ``out`` in chronological order."""
        count = min(self._delta_count, len(out))
        if count == 0:
            return 0
        if self._delta_count < self.window_size:
            start = 0
        else:
            start = self._delta_index
        source_index = start
        for i in range(count):
            out[i] = self._delta_ring[source_index]
            source_index += 1
            if source_index >= self.window_size:
                source_index = 0
        return count

    def update_metric(self):
        if self._delta_count >= self.window_size:
            self._current_metric = self._delta_sum / self._delta_count
        else:
            self._current_metric = 0.0
        return self._current_metric

    def mean(self):
        """Return the current delta-window mean without changing detector state."""
        return self._delta_sum / self._delta_count if self._delta_count else 0.0

    def update_state(self):
        metric = self.update_metric()
        if self._state == MotionState.IDLE:
            if metric > self.threshold:
                self._state = MotionState.MOTION
        elif metric < self.threshold:
            self._state = MotionState.IDLE
        return {
            "motion_metric": metric,
            "l1_delta": metric,
            "threshold": self.threshold,
            "state": self._state,
        }

    def set_threshold(self, threshold):
        if 0.0 <= threshold <= 10.0:
            self.threshold = threshold
            return True
        return False

    def set_adaptive_threshold(self, threshold):
        self.threshold = max(1e-6, min(10.0, threshold))

    def is_ready(self):
        return self._delta_count >= self.window_size

    def reset(self):
        for i in range(self.lag):
            self._profile_len[i] = 0
        self._delta_index = 0
        self._delta_count = 0
        self._delta_sum = 0.0
        self._delta1_index = 0
        self._delta1_count = 0
        self._delta1_sum = 0.0
        self._amplitude_count = 0
        self._profile_index = 0
        self._packet_count = 0
        self._state = MotionState.IDLE
        self._current_metric = 0.0
        self.last_delta = 0.0
        if self._hampel_filter is not None:
            self._hampel_filter.reset()
        if self._hampel_filter1 is not None:
            self._hampel_filter1.reset()

    def get_state(self):
        return self._state

    def get_motion_metric(self):
        return self._current_metric

    @property
    def total_packets(self):
        return self._packet_count


def extract_features_by_name(
    turbulence_buffer,
    buffer_count,
    feature_names=None,
    l1_series=None,
    l1_series_count=None,
    out=None,
    reuse_turbulence_buffer=False,
    l1_delta_lag_ratio=None,
):
    """Extract configured features from explicitly preprocessed streams."""
    if feature_names is None:
        feature_names = DEFAULT_FEATURES

    for name in feature_names:
        if name not in ALL_FEATURES:
            raise ValueError(f"Unknown feature: {name}")
    if 'l1_delta_lag_ratio' in feature_names and l1_delta_lag_ratio is None:
        raise ValueError(
            "l1_delta_lag_ratio is required when that feature is selected; "
            "pass the explicitly preprocessed tracker metric"
        )

    if out is not None and len(out) < len(feature_names):
        raise ValueError("Output feature buffer is too small")

    if buffer_count < 2:
        features = out if out is not None else [0.0] * len(feature_names)
        for i in range(len(feature_names)):
            features[i] = 0.0
        return features

    if isinstance(turbulence_buffer, list):
        turb_list = turbulence_buffer if len(turbulence_buffer) == buffer_count else turbulence_buffer[:buffer_count]
    else:
        turb_list = list(turbulence_buffer)[:buffer_count]

    n = len(turb_list)
    if n < 2:
        features = out if out is not None else [0.0] * len(feature_names)
        for i in range(len(feature_names)):
            features[i] = 0.0
        return features

    turb_mean = sum(turb_list) / n

    var_sum = 0.0
    for i in range(n):
        diff = turb_list[i] - turb_mean
        var_sum += diff * diff
    turb_var = var_sum / n
    turb_std = math.sqrt(turb_var) if turb_var > 0 else 0.0
    abs_mean = abs(turb_mean)
    mean_denom = abs_mean if abs_mean > 1e-6 else 1e-6

    turb_mad = None
    turb_skewness = None
    turb_autocorr = None
    turb_zcr = None
    l1_waveform_length = 0.0
    _l1_series = None
    _l1_n = 0
    _l1_mean = 0.0
    _l1_std = 0.0
    _l1_var = 0.0
    needs_l1 = False
    needs_mad = False
    for name in feature_names:
        if name in L1_SERIES_FEATURES:
            needs_l1 = True
        elif name == "turb_mad_over_mean":
            needs_mad = True
        elif name == "turb_skewness":
            turb_skewness = calc_skewness(turb_list, n, turb_mean, turb_std)
        elif name == "turb_autocorr":
            turb_autocorr = calc_autocorrelation(
                turb_list, n, mean=turb_mean, variance=turb_var
            )
        elif name == "turb_zcr":
            # Crossing rate needs the time-ordered series; compute it before
            # any in-place sort of the reused turbulence buffer.
            sorted_copy = sorted(turb_list)
            turb_zcr = calc_zero_crossing_rate(
                turb_list, n, sorted_copy[n // 2]
            )
    if needs_l1:
        if l1_series is None:
            raise ValueError(
                "l1_series is required for L1 features; pass the explicitly "
                "preprocessed detector stream"
            )
        _l1_series = l1_series
        _l1_n = len(l1_series) if l1_series_count is None else min(
            int(l1_series_count), len(l1_series)
        )
        if _l1_n:
            total = 0.0
            for i in range(_l1_n):
                total += _l1_series[i]
            _l1_mean = total / _l1_n
            vs = 0.0
            for i in range(_l1_n):
                value = _l1_series[i]
                d = value - _l1_mean
                vs += d * d
            _l1_var = vs / _l1_n
            _l1_std = math.sqrt(_l1_var) if _l1_var > 0 else 0.0
            l1_waveform_length = calc_waveform_length(_l1_series, _l1_n)

    if needs_mad and reuse_turbulence_buffer:
        turb_list.sort()
        turb_mad = calc_mad(turb_list, n, sorted_values=turb_list)

    features = out if out is not None else []
    for feature_index, name in enumerate(feature_names):
        if name == 'turb_mad_over_mean':
            if turb_mad is None:
                turb_mad = calc_mad(turb_list, n)
            value = turb_mad / mean_denom
        elif name == 'turb_skewness':
            value = turb_skewness
        elif name == 'turb_autocorr':
            value = turb_autocorr
        elif name == 'turb_zcr':
            value = turb_zcr
        elif name == 'l1_delta':
            value = _l1_mean
        elif name == 'l1_delta_std':
            value = _l1_std
        elif name == 'l1_delta_waveform_length':
            value = l1_waveform_length if _l1_n else 0.0
        elif name == 'l1_delta_autocorr':
            value = (
                calc_autocorrelation(_l1_series, _l1_n, mean=_l1_mean, variance=_l1_var)
                if _l1_n else 0.0
            )
        elif name == 'l1_delta_lag_ratio':
            value = l1_delta_lag_ratio
        elif name == 'l1_delta_cv':
            value = _l1_std / (_l1_mean if _l1_mean > 1e-9 else 1e-9) if _l1_n else 0.0
        else:
            raise ValueError(f"Unknown feature: {name}")
        if out is None:
            features.append(value)
        else:
            features[feature_index] = value
    return features


# Alias for backward compatibility
FEATURE_NAMES = DEFAULT_FEATURES
