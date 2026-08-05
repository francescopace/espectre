"""
Micro-ESPectre - Production ML Feature Trackers

Allocation-aware, MicroPython-friendly trackers for the promoted production ML
features that go beyond turbulence and L1-delta statistics.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import math

try:
    from src.csi_features import L1_DELTA_LAG
except ImportError:
    from csi_features import L1_DELTA_LAG


HT20_CSI_LEN = 128
HT20_LIVE_BINS = tuple(range(4, 32)) + tuple(range(33, 61))
HT20_LIVE_WIDTH = len(HT20_LIVE_BINS)
HT20_COHERENCE_SUBBANDS = (
    tuple(range(4, 18)),
    tuple(range(18, 32)),
    tuple(range(33, 47)),
    tuple(range(47, 61)),
)
_LIVE_BIN_TO_INDEX = {}
for _idx, _bin in enumerate(HT20_LIVE_BINS):
    _LIVE_BIN_TO_INDEX[_bin] = _idx
_ADJACENT_PAIRS = tuple(
    i for i in range(HT20_LIVE_WIDTH - 1)
    if HT20_LIVE_BINS[i + 1] - HT20_LIVE_BINS[i] == 1
)
_ADJACENT_LEFT = tuple(i + 1 for i in _ADJACENT_PAIRS)
_ADJACENT_RIGHT = tuple(_ADJACENT_PAIRS)
_BIN_INDEX = tuple(float(bin_index) for bin_index in HT20_LIVE_BINS)
_SUBBAND_PROFILE_INDICES = tuple(
    tuple(_LIVE_BIN_TO_INDEX[bin_index] for bin_index in subband)
    for subband in HT20_COHERENCE_SUBBANDS
)
_SUBBAND_BIN_INDICES = tuple(
    tuple(float(bin_index) for bin_index in subband)
    for subband in HT20_COHERENCE_SUBBANDS
)
FREQUENCY_COHERENCE_OFFSETS = (2, 4, 12)
# The DC null splits the live band into two runs that are contiguous in both
# bin number and profile index, so a pair separated by `offset` bins is always
# `left + offset` inside one run. Deriving the split from the bin table keeps
# that equivalence honest if the layout ever changes.
_LIVE_BAND_SPLIT = HT20_LIVE_WIDTH
for _i in range(1, HT20_LIVE_WIDTH):
    if HT20_LIVE_BINS[_i] - HT20_LIVE_BINS[_i - 1] != 1:
        _LIVE_BAND_SPLIT = _i
        break
_FREQUENCY_COHERENCE_SPANS = (
    (0, _LIVE_BAND_SPLIT),
    (_LIVE_BAND_SPLIT, HT20_LIVE_WIDTH),
)


def complex_profile(csi_data, out=None):
    """Return the centered HT20 complex profile over the live band."""
    if csi_data is None or len(csi_data) < HT20_CSI_LEN:
        if out is None:
            return [0j] * HT20_LIVE_WIDTH
        for i in range(HT20_LIVE_WIDTH):
            out[i] = 0j
        return out
    profile = out if out is not None else [0j] * HT20_LIVE_WIDTH
    for i, sc_idx in enumerate(HT20_LIVE_BINS):
        imag = csi_data[sc_idx * 2]
        real = csi_data[sc_idx * 2 + 1]
        imag = float(imag if imag < 128 else imag - 256)
        real = float(real if real < 128 else real - 256)
        profile[i] = complex(real, imag)
    return profile


def _delay_compensated_coherence_band(current, reference, indices, bin_index):
    total = 0.0
    cross = [0j] * len(indices)
    for pos, index in enumerate(indices):
        value = current[index] * reference[index].conjugate()
        cross[pos] = value
        total += abs(value)
    if total <= 0.0:
        return 0.0
    ramp_sum = 0j
    for pos in range(1, len(cross)):
        ramp_sum += cross[pos] * cross[pos - 1].conjugate()
    ramp = math.atan2(ramp_sum.imag, ramp_sum.real)
    aligned = 0j
    for pos, value in enumerate(cross):
        angle = -ramp * bin_index[pos]
        aligned += value * complex(math.cos(angle), math.sin(angle))
    return abs(aligned) / total


def delay_compensated_coherence(current, reference):
    """Coherence between two HT20 live-band profiles with delay removed."""
    total = 0.0
    cross = [0j] * HT20_LIVE_WIDTH
    for i in range(HT20_LIVE_WIDTH):
        value = current[i] * reference[i].conjugate()
        cross[i] = value
        total += abs(value)
    if total <= 0.0:
        return 0.0
    ramp_sum = 0j
    for left, right in zip(_ADJACENT_LEFT, _ADJACENT_RIGHT):
        ramp_sum += cross[left] * cross[right].conjugate()
    ramp = math.atan2(ramp_sum.imag, ramp_sum.real)
    aligned = 0j
    for i in range(HT20_LIVE_WIDTH):
        angle = -ramp * _BIN_INDEX[i]
        aligned += cross[i] * complex(math.cos(angle), math.sin(angle))
    return abs(aligned) / total


def subband_coherences(current, reference):
    values = [0.0] * len(_SUBBAND_PROFILE_INDICES)
    for i, indices in enumerate(_SUBBAND_PROFILE_INDICES):
        values[i] = _delay_compensated_coherence_band(
            current, reference, indices, _SUBBAND_BIN_INDICES[i]
        )
    return values


def normalized_amplitude_profile(profile):
    """Return the channel magnitude divided by its packet L2 norm."""
    amplitudes = [abs(value) for value in profile]
    total = 0.0
    for value in amplitudes:
        total += value * value
    norm = math.sqrt(total)
    if norm <= 0.0:
        return [0.0] * len(amplitudes)
    return [value / norm for value in amplitudes]


def motion_participation(energy):
    """Normalized participation ratio of motion energy across subcarriers."""
    total = 0.0
    squared = 0.0
    count = len(energy)
    if count == 0:
        return 0.0
    for value in energy:
        total += value
        squared += value * value
    if total <= 0.0 or squared <= 0.0:
        return 0.0
    return (total * total) / (count * squared)


def new_frequency_coherence_squares():
    """Return the reusable per-packet squared-magnitude buffer."""
    return [0.0] * HT20_LIVE_WIDTH


def _fill_frequency_coherence_squares(profile, squares):
    """Cache the squared magnitude of every live subcarrier once per packet.

    Every offset reads the same magnitudes, so computing them here replaces the
    repeated `abs(value) * abs(value)` that each pair used to redo.
    """
    for i in range(HT20_LIVE_WIDTH):
        value = profile[i]
        real = value.real
        imag = value.imag
        squares[i] = real * real + imag * imag


def _frequency_coherence_from_squares(profile, squares, offset):
    """Coherence at one offset, reusing the cached squared magnitudes.

    Pairs are visited in ascending left index, exactly as the original pair
    table listed them, so the numerator accumulates in an unchanged order.
    """
    numerator = 0j
    left_norm = 0.0
    right_norm = 0.0
    for start, stop in _FREQUENCY_COHERENCE_SPANS:
        for left in range(start, stop - offset):
            right = left + offset
            numerator += profile[left].conjugate() * profile[right]
            left_norm += squares[left]
            right_norm += squares[right]
    denominator = math.sqrt(left_norm) * math.sqrt(right_norm)
    if denominator <= 0.0:
        return 0.0
    return abs(numerator) / denominator


def frequency_coherences(profile, out=None, squares=None):
    """Return the offset 2, 4, and 12 coherences for one packet.

    Passing the caller's `out` and `squares` buffers keeps the per-packet path
    free of allocations.
    """
    if out is None:
        out = [0.0] * len(FREQUENCY_COHERENCE_OFFSETS)
    if squares is None:
        squares = new_frequency_coherence_squares()
    _fill_frequency_coherence_squares(profile, squares)
    for i in range(len(FREQUENCY_COHERENCE_OFFSETS)):
        out[i] = _frequency_coherence_from_squares(
            profile, squares, FREQUENCY_COHERENCE_OFFSETS[i]
        )
    return out


def frequency_coherence(profile, offset=4):
    """Normalized within-packet coherence at a fixed subcarrier separation."""
    if int(offset) not in FREQUENCY_COHERENCE_OFFSETS:
        return 0.0
    squares = new_frequency_coherence_squares()
    _fill_frequency_coherence_squares(profile, squares)
    return _frequency_coherence_from_squares(profile, squares, int(offset))


class ChannelShapeTracker:
    """Track gain-free amplitude-shape and frequency-coherence dynamics."""

    def __init__(self, window_size=90, lag=L1_DELTA_LAG):
        self.window_size = max(2, int(window_size))
        self.lag = max(1, int(lag))
        self._ring = [[0.0] * HT20_LIVE_WIDTH for _ in range(self.lag)]
        self._ring_filled = [False] * self.lag
        self._index = 0
        self._previous = [0.0] * HT20_LIVE_WIDTH
        self._has_previous = False
        self._lag_distance_ring = [0.0] * self.window_size
        self._adjacent_distance_ring = [0.0] * self.window_size
        self._lag_distance_slot = 0
        self._lag_distance_count = 0
        self._lag_distance_sum = 0.0
        self._adjacent_distance_slot = 0
        self._adjacent_distance_count = 0
        self._adjacent_distance_sum = 0.0
        self._motion_energy = [0.0] * HT20_LIVE_WIDTH
        self._motion_energy_ring = [
            [0.0] * HT20_LIVE_WIDTH for _ in range(self.window_size)
        ]
        self._motion_energy_slot = 0
        self._motion_energy_count = 0
        self._frequency_coherence_ring = [0.0] * self.window_size
        self._frequency_coherence_slot = 0
        self._frequency_coherence_count = 0
        self._frequency_coherence_sum = 0.0
        self._frequency_coherence_square_sum = 0.0
        self._frequency_curve_ring = [0.0] * self.window_size
        self._frequency_curve_slot = 0
        self._frequency_curve_count = 0
        self._frequency_curve_sum = 0.0
        self._frequency_curve_square_sum = 0.0
        self._complex_profile = [0j] * HT20_LIVE_WIDTH
        self._coherence_squares = new_frequency_coherence_squares()
        self._coherence_values = [0.0] * len(FREQUENCY_COHERENCE_OFFSETS)

    def _push_scalar(self, value, ring, slot, count, total):
        if count < self.window_size:
            count += 1
        else:
            total -= ring[slot]
        ring[slot] = value
        total += value
        slot = (slot + 1) % self.window_size
        return slot, count, total

    def _push_motion_energy(self, values):
        if self._motion_energy_count < self.window_size:
            self._motion_energy_count += 1
        else:
            old = self._motion_energy_ring[self._motion_energy_slot]
            for i in range(HT20_LIVE_WIDTH):
                self._motion_energy[i] -= old[i]
        slot = self._motion_energy_ring[self._motion_energy_slot]
        for i in range(HT20_LIVE_WIDTH):
            slot[i] = values[i]
            self._motion_energy[i] += values[i]
        self._motion_energy_slot = (self._motion_energy_slot + 1) % self.window_size

    def _push_frequency_coherence(self, value):
        if self._frequency_coherence_count < self.window_size:
            self._frequency_coherence_count += 1
        else:
            old = self._frequency_coherence_ring[self._frequency_coherence_slot]
            self._frequency_coherence_sum -= old
            self._frequency_coherence_square_sum -= old * old
        self._frequency_coherence_ring[self._frequency_coherence_slot] = value
        self._frequency_coherence_sum += value
        self._frequency_coherence_square_sum += value * value
        self._frequency_coherence_slot = (
            self._frequency_coherence_slot + 1
        ) % self.window_size

    def _push_frequency_curve(self, value):
        if self._frequency_curve_count < self.window_size:
            self._frequency_curve_count += 1
        else:
            old = self._frequency_curve_ring[self._frequency_curve_slot]
            self._frequency_curve_sum -= old
            self._frequency_curve_square_sum -= old * old
        self._frequency_curve_ring[self._frequency_curve_slot] = value
        self._frequency_curve_sum += value
        self._frequency_curve_square_sum += value * value
        self._frequency_curve_slot = (self._frequency_curve_slot + 1) % self.window_size

    def process_packet(self, csi_data):
        complex_values = complex_profile(csi_data, self._complex_profile)
        profile = normalized_amplitude_profile(complex_values)
        slot = self._index
        if self._ring_filled[slot]:
            delta = [0.0] * HT20_LIVE_WIDTH
            squared = 0.0
            for i in range(HT20_LIVE_WIDTH):
                diff = profile[i] - self._ring[slot][i]
                delta[i] = diff * diff
                squared += delta[i]
            (
                self._lag_distance_slot,
                self._lag_distance_count,
                self._lag_distance_sum,
            ) = self._push_scalar(
                math.sqrt(squared),
                self._lag_distance_ring,
                self._lag_distance_slot,
                self._lag_distance_count,
                self._lag_distance_sum,
            )
            self._push_motion_energy(delta)
        if self._has_previous:
            squared = 0.0
            for i in range(HT20_LIVE_WIDTH):
                diff = profile[i] - self._previous[i]
                squared += diff * diff
            (
                self._adjacent_distance_slot,
                self._adjacent_distance_count,
                self._adjacent_distance_sum,
            ) = self._push_scalar(
                math.sqrt(squared),
                self._adjacent_distance_ring,
                self._adjacent_distance_slot,
                self._adjacent_distance_count,
                self._adjacent_distance_sum,
            )
        short_coherence, base_coherence, long_coherence = frequency_coherences(
            complex_values, self._coherence_values, self._coherence_squares
        )
        self._push_frequency_coherence(base_coherence)
        coherence_sum = short_coherence + long_coherence
        curve_contrast = (
            (short_coherence - long_coherence) / coherence_sum
            if coherence_sum > 0.0 else 0.0
        )
        self._push_frequency_curve(curve_contrast)
        for i in range(HT20_LIVE_WIDTH):
            self._previous[i] = profile[i]
            self._ring[slot][i] = profile[i]
        self._has_previous = True
        self._ring_filled[slot] = True
        self._index = (self._index + 1) % self.lag

    def count(self):
        return self._lag_distance_count

    def shape_spread(self):
        return motion_participation(self._motion_energy)

    def frequency_coherence_cv(self):
        if self._frequency_coherence_count == 0:
            return 0.0
        mean = self._frequency_coherence_sum / self._frequency_coherence_count
        variance = max(
            0.0,
            self._frequency_coherence_square_sum / self._frequency_coherence_count
            - mean * mean,
        )
        if mean <= 0.0:
            return 0.0
        return math.sqrt(variance) / mean

    def frequency_coherence_curve_std(self):
        if self._frequency_curve_count == 0:
            return 0.0
        mean = self._frequency_curve_sum / self._frequency_curve_count
        variance = max(
            0.0,
            self._frequency_curve_square_sum / self._frequency_curve_count
            - mean * mean,
        )
        return math.sqrt(variance)

    def reset(self):
        for i in range(self.lag):
            self._ring_filled[i] = False
        self._index = 0
        self._has_previous = False
        self._lag_distance_slot = 0
        self._lag_distance_count = 0
        self._lag_distance_sum = 0.0
        self._adjacent_distance_slot = 0
        self._adjacent_distance_count = 0
        self._adjacent_distance_sum = 0.0
        for i in range(HT20_LIVE_WIDTH):
            self._motion_energy[i] = 0.0
        self._motion_energy_slot = 0
        self._motion_energy_count = 0
        self._frequency_coherence_slot = 0
        self._frequency_coherence_count = 0
        self._frequency_coherence_sum = 0.0
        self._frequency_coherence_square_sum = 0.0
        self._frequency_curve_slot = 0
        self._frequency_curve_count = 0
        self._frequency_curve_sum = 0.0
        self._frequency_curve_square_sum = 0.0


class ChannelCoherenceTracker:
    """Track delay-compensated channel coherence metrics for ML production."""

    def __init__(self, window_size=90, lag=L1_DELTA_LAG):
        self.window_size = max(2, int(window_size))
        self.lag = max(1, int(lag))
        self._ring = [[0j] * HT20_LIVE_WIDTH for _ in range(self.lag)]
        self._ring_filled = [False] * self.lag
        self._index = 0
        self._previous = [0j] * HT20_LIVE_WIDTH
        self._has_previous = False
        self._lag_sum = 0.0
        self._lag_count = 0
        self._adjacent_sum = 0.0
        self._adjacent_count = 0
        self._lag_ring = [0.0] * self.window_size
        self._adjacent_ring = [0.0] * self.window_size
        self._lag_slot = 0
        self._adjacent_slot = 0
        subband_count = len(_SUBBAND_PROFILE_INDICES)
        self._subband_lag_sum = [0.0] * subband_count
        self._subband_adjacent_sum = [0.0] * subband_count
        self._subband_lag_ring = [
            [0.0] * subband_count for _ in range(self.window_size)
        ]
        self._subband_adjacent_ring = [
            [0.0] * subband_count for _ in range(self.window_size)
        ]
        self._subband_lag_slot = 0
        self._subband_lag_count = 0
        self._subband_adjacent_slot = 0
        self._subband_adjacent_count = 0

    def _push(self, value, ring, slot, count, total):
        if count < self.window_size:
            count += 1
        else:
            total -= ring[slot]
        ring[slot] = value
        total += value
        slot = (slot + 1) % self.window_size
        return slot, count, total

    def _push_subbands(self, values, ring, slot, count, total):
        if count < self.window_size:
            count += 1
        else:
            old = ring[slot]
            for i in range(len(total)):
                total[i] -= old[i]
        dest = ring[slot]
        for i in range(len(total)):
            dest[i] = values[i]
            total[i] += values[i]
        slot = (slot + 1) % self.window_size
        return slot, count, total

    def process_packet(self, csi_data):
        profile = complex_profile(csi_data)
        slot = self._index
        if self._ring_filled[slot]:
            lag_value = delay_compensated_coherence(profile, self._ring[slot])
            self._lag_slot, self._lag_count, self._lag_sum = self._push(
                lag_value, self._lag_ring, self._lag_slot, self._lag_count,
                self._lag_sum,
            )
            lag_subbands = subband_coherences(profile, self._ring[slot])
            (
                self._subband_lag_slot,
                self._subband_lag_count,
                self._subband_lag_sum,
            ) = self._push_subbands(
                lag_subbands, self._subband_lag_ring, self._subband_lag_slot,
                self._subband_lag_count, self._subband_lag_sum,
            )
        if self._has_previous:
            adjacent_value = delay_compensated_coherence(profile, self._previous)
            (
                self._adjacent_slot,
                self._adjacent_count,
                self._adjacent_sum,
            ) = self._push(
                adjacent_value, self._adjacent_ring, self._adjacent_slot,
                self._adjacent_count, self._adjacent_sum,
            )
            adjacent_subbands = subband_coherences(profile, self._previous)
            (
                self._subband_adjacent_slot,
                self._subband_adjacent_count,
                self._subband_adjacent_sum,
            ) = self._push_subbands(
                adjacent_subbands, self._subband_adjacent_ring,
                self._subband_adjacent_slot, self._subband_adjacent_count,
                self._subband_adjacent_sum,
            )
        for i in range(HT20_LIVE_WIDTH):
            self._previous[i] = profile[i]
            self._ring[slot][i] = profile[i]
        self._has_previous = True
        self._ring_filled[slot] = True
        self._index = (self._index + 1) % self.lag

    def count(self):
        return self._lag_count

    def coherence_gap(self):
        if self._lag_count == 0 or self._adjacent_count == 0:
            return 0.0
        return (
            self._adjacent_sum / self._adjacent_count
            - self._lag_sum / self._lag_count
        )

    def coherence_subband_gap_median(self):
        if self._subband_lag_count == 0 or self._subband_adjacent_count == 0:
            return 0.0
        gaps = [0.0] * len(self._subband_lag_sum)
        for i in range(len(gaps)):
            gaps[i] = (
                self._subband_adjacent_sum[i] / self._subband_adjacent_count
                - self._subband_lag_sum[i] / self._subband_lag_count
            )
        gaps.sort()
        mid = len(gaps) // 2
        if len(gaps) % 2 == 0:
            return 0.5 * (gaps[mid - 1] + gaps[mid])
        return gaps[mid]

    def reset(self):
        for i in range(self.lag):
            self._ring_filled[i] = False
        self._index = 0
        self._has_previous = False
        self._lag_sum = 0.0
        self._lag_count = 0
        self._adjacent_sum = 0.0
        self._adjacent_count = 0
        self._lag_slot = 0
        self._adjacent_slot = 0
        for i in range(len(self._subband_lag_sum)):
            self._subband_lag_sum[i] = 0.0
            self._subband_adjacent_sum[i] = 0.0
        self._subband_lag_slot = 0
        self._subband_lag_count = 0
        self._subband_adjacent_slot = 0
        self._subband_adjacent_count = 0
