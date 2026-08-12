# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
Micro-ESPectre - Production ML Feature Trackers

Allocation-aware, MicroPython-friendly trackers for the promoted production ML
features that go beyond turbulence and L1-delta statistics.

Author: Francesco Pace <francesco.pace@gmail.com>
"""

import math
import time

try:
    from src.csi_features import L1_DELTA_LAG
except ImportError:
    from csi_features import L1_DELTA_LAG


HT20_CSI_LEN = 128
HT20_LIVE_BINS = tuple(range(4, 32)) + tuple(range(33, 61))
HT20_LIVE_WIDTH = len(HT20_LIVE_BINS)
FREQUENCY_COHERENCE_OFFSETS = (4, 12)
CHANNEL_SHAPE_SUBBAND_COUNT = 8
CHANNEL_SHAPE_SUBBAND_WIDTH = HT20_LIVE_WIDTH // CHANNEL_SHAPE_SUBBAND_COUNT
CHANNEL_SHAPE_BIN_US = 80_000
CHANNEL_SHAPE_WINDOW_US = 1_000_000
CHANNEL_SHAPE_MAX_PROFILES_PER_BIN = 32
_CHANNEL_SHAPE_DCT = (
    (0.3535533906, 0.4903926402, 0.4619397663, 0.4157348062, 0.3535533906, 0.2777851165, 0.1913417162, 0.0975451610),
    (0.3535533906, 0.4157348062, 0.1913417162, -0.0975451610, -0.3535533906, -0.4903926402, -0.4619397663, -0.2777851165),
    (0.3535533906, 0.2777851165, -0.1913417162, -0.4903926402, -0.3535533906, 0.0975451610, 0.4619397663, 0.4157348062),
    (0.3535533906, 0.0975451610, -0.4619397663, -0.2777851165, 0.3535533906, 0.4157348062, -0.1913417162, -0.4903926402),
    (0.3535533906, -0.0975451610, -0.4619397663, 0.2777851165, 0.3535533906, -0.4157348062, -0.1913417162, 0.4903926402),
    (0.3535533906, -0.2777851165, -0.1913417162, 0.4903926402, -0.3535533906, -0.0975451610, 0.4619397663, -0.4157348062),
    (0.3535533906, -0.4157348062, 0.1913417162, 0.0975451610, -0.3535533906, 0.4903926402, -0.4619397663, 0.2777851165),
    (0.3535533906, -0.4903926402, 0.4619397663, -0.4157348062, 0.3535533906, -0.2777851165, 0.1913417162, -0.0975451610),
)
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
    """Return the offset 4 and 12 coherences used by Classic.

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


def frequency_coherences_from_csi(csi_data, out, squares):
    """Compute both Classic coherences directly from raw I/Q bytes."""
    if csi_data is None or len(csi_data) < HT20_CSI_LEN:
        for i in range(len(FREQUENCY_COHERENCE_OFFSETS)):
            out[i] = 0.0
        return out
    for i, subcarrier in enumerate(HT20_LIVE_BINS):
        imag = int(csi_data[subcarrier * 2])
        real = int(csi_data[subcarrier * 2 + 1])
        imag = imag if imag < 128 else imag - 256
        real = real if real < 128 else real - 256
        squares[i] = real * real + imag * imag

    for output_index, offset in enumerate(FREQUENCY_COHERENCE_OFFSETS):
        numerator_real = 0.0
        numerator_imag = 0.0
        left_norm = 0.0
        right_norm = 0.0
        for start, stop in _FREQUENCY_COHERENCE_SPANS:
            for left in range(start, stop - offset):
                right = left + offset
                left_subcarrier = HT20_LIVE_BINS[left]
                right_subcarrier = HT20_LIVE_BINS[right]
                left_imag = int(csi_data[left_subcarrier * 2])
                left_real = int(csi_data[left_subcarrier * 2 + 1])
                right_imag = int(csi_data[right_subcarrier * 2])
                right_real = int(csi_data[right_subcarrier * 2 + 1])
                left_imag = left_imag if left_imag < 128 else left_imag - 256
                left_real = left_real if left_real < 128 else left_real - 256
                right_imag = right_imag if right_imag < 128 else right_imag - 256
                right_real = right_real if right_real < 128 else right_real - 256
                numerator_real += left_real * right_real + left_imag * right_imag
                numerator_imag += left_real * right_imag - left_imag * right_real
                left_norm += squares[left]
                right_norm += squares[right]
        denominator = math.sqrt(left_norm) * math.sqrt(right_norm)
        out[output_index] = (
            math.sqrt(
                numerator_real * numerator_real
                + numerator_imag * numerator_imag
            ) / denominator
            if denominator > 0.0 else 0.0
        )
    return out


def frequency_coherence(profile, offset=4):
    """Normalized within-packet coherence at a fixed subcarrier separation."""
    if int(offset) not in FREQUENCY_COHERENCE_OFFSETS:
        return 0.0
    squares = new_frequency_coherence_squares()
    _fill_frequency_coherence_squares(profile, squares)
    return _frequency_coherence_from_squares(profile, squares, int(offset))


def _median_prefix_in_place(values, count):
    """Return the median after sorting only the populated prefix in place."""
    if count <= 0:
        return 0.0
    for i in range(1, count):
        value = values[i]
        j = i
        while j > 0 and values[j - 1] > value:
            values[j] = values[j - 1]
            j -= 1
        values[j] = value
    middle = count // 2
    if count % 2 == 0:
        return 0.5 * (values[middle - 1] + values[middle])
    return values[middle]


class ChannelShapeTrajectoryTracker:
    """Track gain-invariant channel-shape geometry in physical-time bins."""

    def __init__(self, window_duration_us=CHANNEL_SHAPE_WINDOW_US,
                 bin_us=CHANNEL_SHAPE_BIN_US):
        self.window_duration_us = max(3, int(window_duration_us))
        self.bin_us = max(1, int(bin_us))
        self._window_bins = max(
            3,
            (self.window_duration_us + self.bin_us - 1) // self.bin_us,
        )
        self._median_values = [0.0] * CHANNEL_SHAPE_MAX_PROFILES_PER_BIN
        self._median_profile_buffer = [0.0] * CHANNEL_SHAPE_SUBBAND_COUNT
        self._current_modes = [0.0] * CHANNEL_SHAPE_SUBBAND_COUNT
        self._spread_energy = [0.0] * CHANNEL_SHAPE_SUBBAND_COUNT
        self._innovation_samples = [0.0] * self._window_bins
        self._excess_samples = [0.0] * self._window_bins
        self._current_profiles = [
            [0.0] * CHANNEL_SHAPE_SUBBAND_COUNT
            for _ in range(CHANNEL_SHAPE_MAX_PROFILES_PER_BIN)
        ]
        self._bin_indices = [0] * self._window_bins
        self._bin_modes = [
            [0.0] * CHANNEL_SHAPE_SUBBAND_COUNT
            for _ in range(self._window_bins)
        ]
        self._previous_raw = bytearray(HT20_CSI_LEN)
        self.reset()

    def _timestamp_us(self, timestamp_us):
        if timestamp_us is not None:
            return int(timestamp_us)
        try:
            return int(time.ticks_us())
        except AttributeError:
            return int(time.monotonic() * 1_000_000)

    def _fill_profile(self, csi_data, energy, subcarrier_energies=None,
                      subcarrier_count=0):
        for i in range(CHANNEL_SHAPE_SUBBAND_COUNT):
            energy[i] = 0.0
        if csi_data is None or len(csi_data) < HT20_CSI_LEN:
            return energy
        total = 0.0
        for live_index, subcarrier in enumerate(HT20_LIVE_BINS):
            if subcarrier_energies is not None and subcarrier < subcarrier_count:
                value = subcarrier_energies[subcarrier]
            else:
                imag = csi_data[subcarrier * 2]
                real = csi_data[subcarrier * 2 + 1]
                imag = float(imag if imag < 128 else imag - 256)
                real = float(real if real < 128 else real - 256)
                value = real * real + imag * imag
            energy[live_index // CHANNEL_SHAPE_SUBBAND_WIDTH] += value
            total += value
        if total <= 0.0:
            return energy
        for i in range(CHANNEL_SHAPE_SUBBAND_COUNT):
            energy[i] = math.sqrt(energy[i] / total)
        return energy

    def _median_profile(self, profiles, count, profile=None):
        if profile is None:
            profile = [0.0] * CHANNEL_SHAPE_SUBBAND_COUNT
        values = self._median_values
        for i in range(CHANNEL_SHAPE_SUBBAND_COUNT):
            for row_index in range(count):
                values[row_index] = profiles[row_index][i]
            profile[i] = _median_prefix_in_place(values, count)
        norm_squared = 0.0
        for value in profile:
            norm_squared += value * value
        norm = math.sqrt(norm_squared)
        if norm > 0.0:
            for i in range(CHANNEL_SHAPE_SUBBAND_COUNT):
                profile[i] /= norm
        return profile

    def _finalize_current_bin(self):
        if self._current_bin is None or self._current_profile_count == 0:
            return
        profile = self._median_profile(
            self._current_profiles,
            self._current_profile_count,
            self._median_profile_buffer,
        )
        if self._bin_count < self._window_bins:
            slot = self._bin_start + self._bin_count
            if slot >= self._window_bins:
                slot -= self._window_bins
            self._bin_count += 1
        else:
            slot = self._bin_start
            self._bin_start += 1
            if self._bin_start >= self._window_bins:
                self._bin_start = 0
        self._bin_indices[slot] = self._current_bin
        self._modes(profile, self._bin_modes[slot])

    def _trim(self, current_bin):
        first_bin = int(current_bin) - self._window_bins + 1
        while (
            self._bin_count > 0
            and self._bin_indices[self._bin_start] < first_bin
        ):
            self._bin_start += 1
            if self._bin_start >= self._window_bins:
                self._bin_start = 0
            self._bin_count -= 1

    def _bin_at(self, index):
        slot = self._bin_start + index
        if slot >= self._window_bins:
            slot -= self._window_bins
        return self._bin_indices[slot], self._bin_modes[slot]

    def process_packet(self, csi_data, timestamp_us=None,
                       subcarrier_energies=None, subcarrier_count=0):
        if csi_data is None or len(csi_data) < HT20_CSI_LEN:
            return
        if isinstance(csi_data, (bytes, bytearray, memoryview)):
            duplicate = self._has_previous_raw and self._previous_raw == csi_data
            self._previous_raw[:] = csi_data
        else:
            duplicate = self._has_previous_raw
            for i in range(HT20_CSI_LEN):
                value = int(csi_data[i]) & 0xFF
                if not self._has_previous_raw or self._previous_raw[i] != value:
                    duplicate = False
                self._previous_raw[i] = value
        self._has_previous_raw = True
        if duplicate:
            return
        bin_index = max(0, self._timestamp_us(timestamp_us)) // self.bin_us
        if self._current_bin is None:
            self._current_bin = bin_index
        elif bin_index != self._current_bin:
            self._finalize_current_bin()
            self._current_bin = bin_index
            self._current_profile_count = 0
            self._trim(bin_index)
        if self._current_profile_count >= CHANNEL_SHAPE_MAX_PROFILES_PER_BIN:
            return
        self._fill_profile(
            csi_data,
            self._current_profiles[self._current_profile_count],
            subcarrier_energies,
            subcarrier_count,
        )
        self._current_profile_count += 1

    @staticmethod
    def _modes(values, modes=None):
        if modes is None:
            modes = [0.0] * CHANNEL_SHAPE_SUBBAND_COUNT
        for mode in range(CHANNEL_SHAPE_SUBBAND_COUNT):
            total = 0.0
            for i in range(CHANNEL_SHAPE_SUBBAND_COUNT):
                total += values[i] * _CHANNEL_SHAPE_DCT[i][mode]
            modes[mode] = total
        return modes

    def trajectory_features_with_spread(self):
        """Return all trajectory readouts from cached orthonormal DCT modes."""
        bin_count = self._bin_count
        has_current = self._current_profile_count > 0
        count = bin_count + (1 if has_current else 0)
        if count < 2:
            return 0.0, 0.0, 0.0
        if has_current:
            current_profile = self._median_profile(
                self._current_profiles,
                self._current_profile_count,
                self._median_profile_buffer,
            )
            self._modes(current_profile, self._current_modes)

        spread_energy = self._spread_energy
        for subband in range(CHANNEL_SHAPE_SUBBAND_COUNT):
            spread_energy[subband] = 0.0
        previous_bin, previous_modes = self._bin_at(0)
        for index in range(1, count):
            if index < bin_count:
                current_bin, current_modes = self._bin_at(index)
            else:
                current_bin, current_modes = self._current_bin, self._current_modes
            if current_bin - previous_bin == 1:
                for subband in range(CHANNEL_SHAPE_SUBBAND_COUNT):
                    delta = 0.0
                    for mode in range(CHANNEL_SHAPE_SUBBAND_COUNT):
                        delta += (
                            current_modes[mode] - previous_modes[mode]
                        ) * _CHANNEL_SHAPE_DCT[subband][mode]
                    spread_energy[subband] += delta * delta
            previous_bin = current_bin
            previous_modes = current_modes

        first_bin, first_modes = self._bin_at(0)
        if bin_count > 1:
            middle_bin, middle_modes = self._bin_at(1)
        else:
            middle_bin, middle_modes = self._current_bin, self._current_modes
        innovation_count = 0
        excess_count = 0
        innovation_samples = self._innovation_samples
        excess_samples = self._excess_samples
        for index in range(2, count):
            if index < bin_count:
                last_bin, last_modes = self._bin_at(index)
            else:
                last_bin, last_modes = self._current_bin, self._current_modes
            previous_dt = middle_bin - first_bin
            current_dt = last_bin - middle_bin
            first_norm_squared = 0.0
            second_norm_squared = 0.0
            chord_norm_squared = 0.0
            first_high_squared = 0.0
            second_high_squared = 0.0
            chord_high_squared = 0.0
            innovation_low_squared = 0.0
            innovation_high_squared = 0.0
            ratio = current_dt / previous_dt if previous_dt > 0 else 0.0
            for mode in range(CHANNEL_SHAPE_SUBBAND_COUNT):
                first_delta = middle_modes[mode] - first_modes[mode]
                second_delta = last_modes[mode] - middle_modes[mode]
                chord_delta = last_modes[mode] - first_modes[mode]
                first_norm_squared += first_delta * first_delta
                second_norm_squared += second_delta * second_delta
                chord_norm_squared += chord_delta * chord_delta
                if mode >= 4:
                    first_high_squared += first_delta * first_delta
                    second_high_squared += second_delta * second_delta
                    chord_high_squared += chord_delta * chord_delta
                if previous_dt > 0 and current_dt > 0 and mode > 0:
                    residual = second_delta - ratio * first_delta
                    if mode < 4:
                        innovation_low_squared += residual * residual
                    else:
                        innovation_high_squared += residual * residual
            if previous_dt > 0 and current_dt > 0:
                innovation_samples[innovation_count] = max(
                    0.0,
                    innovation_low_squared - innovation_high_squared,
                )
                innovation_count += 1
            # Parseval preserves the full-profile L2 distance in DCT space.
            raw = (
                math.sqrt(first_norm_squared)
                + math.sqrt(second_norm_squared)
                - math.sqrt(chord_norm_squared)
            )
            high = (
                math.sqrt(first_high_squared)
                + math.sqrt(second_high_squared)
                - math.sqrt(chord_high_squared)
            )
            excess_samples[excess_count] = max(0.0, raw - max(0.0, high))
            excess_count += 1
            first_bin = middle_bin
            middle_bin = last_bin
            first_modes = middle_modes
            middle_modes = last_modes
        return (
            _median_prefix_in_place(innovation_samples, innovation_count),
            _median_prefix_in_place(excess_samples, excess_count),
            motion_participation(spread_energy),
        )

    def trajectory_features(self):
        """Return the two historical geometry readouts."""
        innovation, excess, _spread = self.trajectory_features_with_spread()
        return innovation, excess

    def coherent_innovation_energy(self):
        return self.trajectory_features()[0]

    def excess_path(self):
        return self.trajectory_features()[1]

    def shape_spread_subband(self):
        return self.trajectory_features_with_spread()[2]

    def reset(self):
        self._bin_start = 0
        self._bin_count = 0
        self._current_bin = None
        self._current_profile_count = 0
        self._has_previous_raw = False


class FrequencyCoherenceTracker:
    """Track the Classic detector's frequency-coherence curve."""

    def __init__(self, window_size=90):
        self.window_size = max(2, int(window_size))
        self._frequency_curve_ring = [0.0] * self.window_size
        self._frequency_curve_slot = 0
        self._frequency_curve_count = 0
        self._frequency_curve_sum = 0.0
        self._frequency_curve_square_sum = 0.0
        self._coherence_squares = new_frequency_coherence_squares()
        self._coherence_values = [0.0] * len(FREQUENCY_COHERENCE_OFFSETS)

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
        self._frequency_curve_slot += 1
        if self._frequency_curve_slot >= self.window_size:
            self._frequency_curve_slot = 0

    def process_packet(self, csi_data):
        short_coherence, long_coherence = frequency_coherences_from_csi(
            csi_data, self._coherence_values, self._coherence_squares
        )
        coherence_sum = short_coherence + long_coherence
        self._push_frequency_curve(
            (short_coherence - long_coherence) / coherence_sum
            if coherence_sum > 0.0 else 0.0
        )

    def count(self):
        return self._frequency_curve_count

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
        self._frequency_curve_slot = 0
        self._frequency_curve_count = 0
        self._frequency_curve_sum = 0.0
        self._frequency_curve_square_sum = 0.0
