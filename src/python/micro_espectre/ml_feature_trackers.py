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
