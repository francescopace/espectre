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
    from src.segmentation import SegmentationContext
except ImportError:
    from detector_interface import MotionState
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
        sorted_vals = list(turbulence_buffer[:buffer_count])
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
        sorted_vals = list(turbulence_buffer[:buffer_count])
        sorted_vals.sort()
    else:
        sorted_vals = sorted_values

    mid = buffer_count // 2
    if buffer_count % 2 == 0:
        median = (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
    else:
        median = sorted_vals[mid]

    abs_devs = [abs(turbulence_buffer[i] - median) for i in range(buffer_count)]
    abs_devs.sort()

    if buffer_count % 2 == 0:
        return (abs_devs[mid - 1] + abs_devs[mid]) / 2.0
    return abs_devs[mid]


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


# Supported L1-delta features. The wider descriptor experiment was rejected;
# only the three promoted features remain available to training/export flows.
L1_DELTA_FEATURES = [
    'l1_delta',
    'l1_delta_std',
    'l1_delta_waveform_length',
]

# Core-6 production set: three gain-invariant turbulence statistics plus the
# three L1-delta profile-displacement features that survived ablation. Beats
# the former relative-8 set on all promotion gates (see docs/EXPERIMENTS.md).
CORE6_FEATURES = [
    'turb_mad_over_mean',
    'turb_skewness',
    'turb_autocorr',
]
CORE6_FEATURES.extend(L1_DELTA_FEATURES)

# Production feature set.
DEFAULT_FEATURES = CORE6_FEATURES

ALL_FEATURES = tuple(DEFAULT_FEATURES)


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

    def __init__(self, window_size=100, threshold=1.0, lag=L1_DELTA_LAG):
        self.window_size = max(2, int(window_size))
        self.threshold = threshold
        self.lag = max(1, int(lag))

        profile_width = SegmentationContext.AMPLITUDE_BUFFER_SIZE
        self._profile_ring = [[0.0] * profile_width for _ in range(self.lag)]
        self._profile_len = [0] * self.lag
        self._current_profile = [0.0] * profile_width
        self._amplitude_buffer = [0.0] * profile_width
        self._delta_ring = [0.0] * self.window_size
        self._delta_index = 0
        self._delta_count = 0
        self._delta_sum = 0.0

        self._packet_count = 0
        self._state = MotionState.IDLE
        self._current_metric = 0.0
        self.last_delta = 0.0

    def _push_delta(self, delta):
        if self._delta_count < self.window_size:
            self._delta_count += 1
        else:
            self._delta_sum -= self._delta_ring[self._delta_index]
        self._delta_ring[self._delta_index] = delta
        self._delta_sum += delta
        self._delta_index = (self._delta_index + 1) % self.window_size

    def process_packet(self, csi_data, selected_subcarriers=None):
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

        if profile_len > 0 and reference_len == profile_len:
            total = 0.0
            for i in range(profile_len):
                diff = profile[i] - reference[i]
                total += diff if diff >= 0 else -diff
            self.last_delta = total / profile_len
            self._push_delta(self.last_delta)

        self._profile_ring[ring_slot] = profile
        self._profile_len[ring_slot] = profile_len
        self._current_profile = reference

    def update_metric(self):
        if self._delta_count >= self.window_size:
            self._current_metric = self._delta_sum / self._delta_count
        else:
            self._current_metric = 0.0
        return self._current_metric

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
        self._delta_ring = [0.0] * self.window_size
        self._delta_index = 0
        self._delta_count = 0
        self._delta_sum = 0.0
        self._packet_count = 0
        self._state = MotionState.IDLE
        self._current_metric = 0.0
        self.last_delta = 0.0

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
    amplitude_history=None,
):
    """Extract configured feature vector from turbulence buffer."""
    if feature_names is None:
        feature_names = DEFAULT_FEATURES

    if buffer_count < 2:
        return [0.0] * len(feature_names)

    if isinstance(turbulence_buffer, list):
        turb_list = turbulence_buffer if len(turbulence_buffer) == buffer_count else turbulence_buffer[:buffer_count]
    else:
        turb_list = list(turbulence_buffer)[:buffer_count]

    n = len(turb_list)
    if n < 2:
        return [0.0] * len(feature_names)

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
    l1_waveform_length = None
    # L1-delta state is computed lazily so turbulence-only requests do not pay
    # the profile-history cost.
    _l1_series = None
    _l1_ready = False
    _l1_n = 0
    _l1_mean = 0.0
    _l1_std = 0.0
    def _ensure_l1_series():
        nonlocal _l1_series, _l1_ready, _l1_n, _l1_mean, _l1_std
        if _l1_ready:
            return
        _l1_ready = True
        _l1_series = l1_delta_series(amplitude_history, n)
        _l1_n = len(_l1_series)
        if _l1_n == 0:
            return
        _l1_mean = sum(_l1_series) / _l1_n
        vs = 0.0
        for value in _l1_series:
            d = value - _l1_mean
            vs += d * d
        variance = vs / _l1_n
        _l1_std = math.sqrt(variance) if variance > 0 else 0.0

    features = []
    for name in feature_names:
        if name == 'turb_mad_over_mean':
            if turb_mad is None:
                turb_mad = calc_mad(turb_list, n)
            features.append(turb_mad / mean_denom)
        elif name == 'turb_skewness':
            features.append(calc_skewness(turb_list, n, turb_mean, turb_std))
        elif name == 'turb_autocorr':
            features.append(calc_autocorrelation(turb_list, n, mean=turb_mean, variance=turb_var))
        elif name == 'l1_delta':
            _ensure_l1_series()
            features.append(_l1_mean)
        elif name == 'l1_delta_std':
            _ensure_l1_series()
            features.append(_l1_std)
        elif name == 'l1_delta_waveform_length':
            _ensure_l1_series()
            if l1_waveform_length is None and _l1_n:
                l1_waveform_length = calc_waveform_length(_l1_series, _l1_n)
            features.append(l1_waveform_length if _l1_n else 0.0)
        else:
            raise ValueError(f"Unknown feature: {name}")
    return features


# Alias for backward compatibility
FEATURE_NAMES = DEFAULT_FEATURES
