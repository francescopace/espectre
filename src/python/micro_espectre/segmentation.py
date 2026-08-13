# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
Micro-ESPectre - Shared Turbulence Context

Pure Python implementation compatible with both MicroPython and standard Python.
Provides the shared turbulence buffer, amplitude scratch, and filters used by
LightweightDetector and ML feature extraction.
Uses two-pass variance helpers for numerical stability (matches C++).

Author: Francesco Pace <francesco.pace@gmail.com>
"""
import math

try:
    from src.device_utils import to_signed_int8, calculate_variance
except ImportError:
    from device_utils import to_signed_int8, calculate_variance


class SegmentationContext:
    """
    Shared turbulence context for Lightweight/High Accuracy feature streams.

    Production detectors use this for spatial turbulence, filtering, and the
    circular turbulence/amplitude buffers.

    Uses two-pass variance calculation for numerical stability.
    This matches the C++ implementation and avoids catastrophic cancellation
    that can occur with running variance on float32.

    Two-pass variance formula: Var(X) = Σ(x - μ)² / n

    All configuration is passed as parameters (dependency injection),
    making this class usable in both MicroPython and standard Python.
    """

    # Pre-allocated amplitude scratch (12 selected subcarriers, matches C++ path)
    AMPLITUDE_BUFFER_SIZE = 12

    def __init__(self,
                 window_size=100,
                 enable_lowpass=False,
                 lowpass_cutoff=11.0,
                 enable_hampel=True,
                 hampel_window=7,
                 hampel_threshold=5.0,
                 allocate_amplitude_buffer=True,
                 adjacent_aggregation_width=None):
        """
        Initialize segmentation context

        Args:
            window_size: Turbulence buffer / window size (default: 100, matches C++ DETECTOR_DEFAULT_WINDOW_SIZE)
            enable_lowpass: Enable low-pass filter for noise reduction (default: False)
            lowpass_cutoff: Low-pass filter cutoff frequency in Hz (default: 11.0)
            enable_hampel: Enable Hampel filter for outlier removal (default: True)
            hampel_window: Hampel filter window size (default: 7)
            hampel_threshold: Hampel filter threshold in MAD units (default: 5.0)
        """
        self.window_size = window_size
        self.adjacent_aggregation_width = adjacent_aggregation_width

        # Turbulence circular buffer (pre-allocated)
        self.turbulence_buffer = [0.0] * window_size
        self.buffer_index = 0
        self.buffer_count = 0

        self.last_turbulence = 0.0

        # Last amplitudes (stored for external use)
        self.last_amplitudes = None
        self._amplitude_buffer = (
            [0.0] * self.AMPLITUDE_BUFFER_SIZE
            if allocate_amplitude_buffer
            else None
        )
        self._amplitude_count = 0

        # Initialize low-pass filter if enabled
        self.lowpass_filter = None
        if enable_lowpass:
            try:
                # Try MicroPython path first, then standard Python path
                try:
                    from src.filters import LowPassFilter
                except ImportError:
                    from filters import LowPassFilter
                self.lowpass_filter = LowPassFilter(
                    cutoff_hz=lowpass_cutoff,
                    sample_rate_hz=100.0,
                    enabled=True
                )
            except Exception as e:
                print(f"[ERROR] Failed to initialize LowPassFilter: {e}")
                self.lowpass_filter = None

        # Initialize Hampel filter if enabled
        self.hampel_filter = None
        if enable_hampel:
            try:
                # Try MicroPython path first, then standard Python path
                try:
                    from src.filters import HampelFilter
                except ImportError:
                    from filters import HampelFilter
                self.hampel_filter = HampelFilter(
                    window_size=hampel_window,
                    threshold=hampel_threshold
                )
            except Exception as e:
                print(f"[ERROR] Failed to initialize HampelFilter: {e}")
                self.hampel_filter = None

    @staticmethod
    def compute_variance_two_pass(values):
        """
        Calculate variance using two-pass algorithm (numerically stable) - static version

        Delegates to utils.calculate_variance() to avoid code duplication.

        Args:
            values: List or array of float values

        Returns:
            float: Variance (0.0 if empty)
        """
        return calculate_variance(values)

    @staticmethod
    def _amplitude_at_subcarrier(csi_data, sc_idx):
        """Return amplitude for one subcarrier, or None if CSI payload is too short."""
        i = sc_idx * 2
        if i + 1 >= len(csi_data):
            return None
        imag = float(to_signed_int8(csi_data[i]))
        real = float(to_signed_int8(csi_data[i + 1]))
        return math.sqrt(real * real + imag * imag)

    @staticmethod
    def _fill_amplitude_buffer(csi_data, selected_subcarriers, out_buffer):
        """
        Fill a pre-allocated amplitude buffer (no per-packet list allocations).

        Returns:
            int: Number of valid amplitudes written
        """
        n = 0
        max_slots = len(out_buffer)
        csi_len = len(csi_data)

        if selected_subcarriers is None:
            max_values = min(128, csi_len)
            for sc_idx in range(0, max_values // 2):
                if n >= max_slots:
                    break
                i = sc_idx * 2
                imag = csi_data[i]
                real = csi_data[i + 1]
                imag = float(imag if imag < 128 else imag - 256)
                real = float(real if real < 128 else real - 256)
                out_buffer[n] = math.sqrt(real * real + imag * imag)
                n += 1
        else:
            for sc_idx in selected_subcarriers:
                if n >= max_slots:
                    break
                i = sc_idx * 2
                if i + 1 >= csi_len:
                    continue
                imag = csi_data[i]
                real = csi_data[i + 1]
                imag = float(imag if imag < 128 else imag - 256)
                real = float(real if real < 128 else real - 256)
                out_buffer[n] = math.sqrt(real * real + imag * imag)
                n += 1
        return n

    @staticmethod
    def fill_subcarrier_energy_buffer(csi_data, out_buffer):
        """Extract one squared magnitude per CSI bin into reusable storage."""
        count = min(len(out_buffer), len(csi_data) // 2)
        for sc_idx in range(count):
            i = sc_idx * 2
            imag = csi_data[i]
            real = csi_data[i + 1]
            imag = float(imag if imag < 128 else imag - 256)
            real = float(real if real < 128 else real - 256)
            out_buffer[sc_idx] = real * real + imag * imag
        for sc_idx in range(count, len(out_buffer)):
            out_buffer[sc_idx] = 0.0
        return count

    @staticmethod
    def energies_to_amplitudes_in_place(values, count):
        """Convert a reusable squared-magnitude frame to magnitudes in place."""
        for i in range(min(int(count), len(values))):
            values[i] = math.sqrt(values[i])

    @staticmethod
    def _fill_amplitude_buffer_from_subcarrier_amplitudes(
        subcarrier_amplitudes,
        subcarrier_count,
        selected_subcarriers,
        out_buffer,
    ):
        """Select detector-band amplitudes from a shared per-bin frame."""
        if selected_subcarriers is None:
            selected_subcarriers = range(subcarrier_count)
        n = 0
        for sc_idx in selected_subcarriers:
            if n >= len(out_buffer):
                break
            if 0 <= sc_idx < subcarrier_count:
                out_buffer[n] = subcarrier_amplitudes[sc_idx]
                n += 1
        return n

    @staticmethod
    def _fill_adjacent_aggregated_amplitude_buffer(
        csi_data,
        selected_subcarriers,
        out_buffer,
        width,
    ):
        """Fill amplitudes by averaging adjacent live-bin magnitudes."""
        if selected_subcarriers is None or width is None or width < 1:
            return 0

        n = 0
        max_slots = len(out_buffer)
        csi_len = len(csi_data)
        half = (width - 1) // 2
        for subcarrier in selected_subcarriers:
            if n >= max_slots:
                break
            low = subcarrier - half
            high = subcarrier + (width - 1 - half)
            if low < 4:
                low, high = 4, 4 + width - 1
            if high > 60:
                low, high = 60 - width + 1, 60

            total = 0.0
            count = 0
            for sc_idx in range(low, high + 1):
                if sc_idx == 32:
                    continue
                i = sc_idx * 2
                if i + 1 >= csi_len:
                    continue
                imag = csi_data[i]
                real = csi_data[i + 1]
                imag = float(imag if imag < 128 else imag - 256)
                real = float(real if real < 128 else real - 256)
                total += math.sqrt(real * real + imag * imag)
                count += 1
            if count:
                out_buffer[n] = total / count
                n += 1
        return n

    @staticmethod
    def _fill_adjacent_aggregated_amplitude_buffer_from_subcarrier_amplitudes(
        subcarrier_amplitudes,
        subcarrier_count,
        selected_subcarriers,
        out_buffer,
        width,
    ):
        """Build adjacent-bin means from a shared per-bin magnitude frame."""
        if selected_subcarriers is None or width is None or width < 1:
            return 0

        n = 0
        half = (width - 1) // 2
        for subcarrier in selected_subcarriers:
            if n >= len(out_buffer):
                break
            low = subcarrier - half
            high = subcarrier + (width - 1 - half)
            if low < 4:
                low, high = 4, 4 + width - 1
            if high > 60:
                low, high = 60 - width + 1, 60

            total = 0.0
            count = 0
            for sc_idx in range(low, high + 1):
                if sc_idx == 32 or sc_idx < 0 or sc_idx >= subcarrier_count:
                    continue
                total += subcarrier_amplitudes[sc_idx]
                count += 1
            if count:
                out_buffer[n] = total / count
                n += 1
        return n

    @staticmethod
    def _turbulence_from_amplitude_buffer(amplitude_buffer, count):
        """Compute gain-invariant spatial turbulence from amplitudes."""
        if count < 2:
            return 0.0

        total = 0.0
        for i in range(count):
            total += amplitude_buffer[i]
        mean = total / count

        var_sum = 0.0
        for i in range(count):
            diff = amplitude_buffer[i] - mean
            var_sum += diff * diff
        variance = var_sum / count

        return math.sqrt(variance) / mean if mean > 0 else 0.0

    @staticmethod
    def calculate_turbulence_from_amplitudes(amplitude_buffer, count):
        """Calculate turbulence from a previously extracted amplitude profile."""
        return SegmentationContext._turbulence_from_amplitude_buffer(
            amplitude_buffer, count
        )

    @staticmethod
    def compute_spatial_turbulence(csi_data, selected_subcarriers=None):
        """
        Calculate spatial turbulence from CSI subcarrier amplitudes

        The runtime always returns gain-invariant turbulence as `std/mean`.

        Args:
            csi_data: array of int8 I/Q values (alternating real, imag)
            selected_subcarriers: list of subcarrier indices to use (default: all up to 64)

        Returns:
            tuple: (turbulence, amplitudes) - turbulence value and amplitude list
        """
        if len(csi_data) < 2:
            return 0.0, []

        scratch = [0.0] * 64
        count = SegmentationContext._fill_amplitude_buffer(
            csi_data, selected_subcarriers, scratch
        )
        turbulence = SegmentationContext._turbulence_from_amplitude_buffer(scratch, count)
        return turbulence, scratch[:count]

    def _compute_spatial_turbulence_in_buffer(self, csi_data, selected_subcarriers=None):
        """Fast instance path: reuse pre-allocated amplitude buffer."""
        if self._amplitude_buffer is None:
            raise RuntimeError("Amplitude extraction buffer is disabled")
        if len(csi_data) < 2:
            self._amplitude_count = 0
            return 0.0

        if self.adjacent_aggregation_width is None:
            self._amplitude_count = self._fill_amplitude_buffer(
                csi_data, selected_subcarriers, self._amplitude_buffer
            )
        else:
            self._amplitude_count = self._fill_adjacent_aggregated_amplitude_buffer(
                csi_data,
                selected_subcarriers,
                self._amplitude_buffer,
                self.adjacent_aggregation_width,
            )
        return self._turbulence_from_amplitude_buffer(self._amplitude_buffer, self._amplitude_count)

    def calculate_spatial_turbulence(self, csi_data, selected_subcarriers=None, return_amplitudes=False):
        """
        Calculate spatial turbulence and store amplitudes for features

        Uses the instance's normalized turbulence path.

        Args:
            csi_data: array of int8 I/Q values (alternating real, imag)
            selected_subcarriers: list of subcarrier indices to use (default: all up to 64)
            return_amplitudes: if True, return (turbulence, amplitudes) tuple

        Returns:
            float: Gain-invariant turbulence value
            OR tuple (turbulence, amplitudes) if return_amplitudes=True

        Note: Stores last amplitudes only when return_amplitudes=True (legacy callers).
        """
        turbulence = self._compute_spatial_turbulence_in_buffer(
            csi_data, selected_subcarriers
        )
        if return_amplitudes:
            self.last_amplitudes = self._amplitude_buffer[:self._amplitude_count]
            return turbulence, self.last_amplitudes
        self.last_amplitudes = None
        return turbulence

    def calculate_spatial_turbulence_from_subcarrier_amplitudes(
        self,
        subcarrier_amplitudes,
        subcarrier_count,
        selected_subcarriers=None,
    ):
        """Calculate turbulence from a packet-wide magnitude frame.

        The caller may reuse one frame for the normal band, adjacent-bin
        aggregation, L1 displacement, and channel-shape tracking.
        """
        if self._amplitude_buffer is None:
            raise RuntimeError("Amplitude extraction buffer is disabled")
        if self.adjacent_aggregation_width is None:
            self._amplitude_count = (
                self._fill_amplitude_buffer_from_subcarrier_amplitudes(
                    subcarrier_amplitudes,
                    subcarrier_count,
                    selected_subcarriers,
                    self._amplitude_buffer,
                )
            )
        else:
            self._amplitude_count = (
                self._fill_adjacent_aggregated_amplitude_buffer_from_subcarrier_amplitudes(
                    subcarrier_amplitudes,
                    subcarrier_count,
                    selected_subcarriers,
                    self._amplitude_buffer,
                    self.adjacent_aggregation_width,
                )
            )
        self.last_amplitudes = None
        return self._turbulence_from_amplitude_buffer(
            self._amplitude_buffer,
            self._amplitude_count,
        )

    def add_turbulence(self, turbulence):
        """
        Add turbulence value to the circular buffer.

        Filter chain: raw → hampel → low-pass → buffer

        Args:
            turbulence: Spatial turbulence value
        """
        # Apply Hampel filter first (removes outliers/spikes)
        filtered_turbulence = turbulence
        if self.hampel_filter is not None:
            try:
                filtered_turbulence = self.hampel_filter.filter(filtered_turbulence)
            except Exception as e:
                print(f"[ERROR] Hampel filter failed: {e}")

        # Apply low-pass filter (removes high-frequency noise)
        if self.lowpass_filter is not None:
            try:
                filtered_turbulence = self.lowpass_filter.filter(filtered_turbulence)
            except Exception as e:
                print(f"[ERROR] LowPass filter failed: {e}")

        self.last_turbulence = filtered_turbulence

        # Store value in circular buffer
        self.turbulence_buffer[self.buffer_index] = filtered_turbulence
        self.buffer_index += 1
        if self.buffer_index >= self.window_size:
            self.buffer_index = 0
        if self.buffer_count < self.window_size:
            self.buffer_count += 1

    def reset(self, full=False):
        """
        Reset context state.

        Args:
            full: If True, also reset buffer (cold start).
                  If False (default), keep buffer warm.
        """
        if full:
            self.buffer_index = 0
            self.buffer_count = 0
            self.last_turbulence = 0.0
            self.last_amplitudes = None

            # Reset filters
            if self.lowpass_filter is not None:
                self.lowpass_filter.reset()
            if self.hampel_filter is not None:
                self.hampel_filter.reset()
