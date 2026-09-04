# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""MicroPython facade over the ESPectre core temporal CSI sampler."""

try:
    import espectre_native_features as _native_core
except ImportError:
    _native_core = None

MICROSECONDS_PER_SECOND = 1_000_000
MINIMUM_COVERAGE_NUMERATOR = 7
MINIMUM_COVERAGE_DENOMINATOR = 10
SLOT_HALF_DENOMINATOR = 2


def temporal_window_slots(target_pps, window_size_ms):
    """Return the number of target-rate slots covering ``window_size_ms``."""
    rate = int(target_pps)
    duration = int(window_size_ms)
    if rate <= 0:
        raise ValueError("target_pps must be positive")
    if duration <= 0:
        raise ValueError("window_size_ms must be positive")
    return max(1, (rate * duration + 999) // 1000)


def minimum_valid_slots(window_slots):
    """Return the shared seven-tenths occupancy floor, rounded up."""
    slots = max(1, int(window_slots))
    return (
        slots * MINIMUM_COVERAGE_NUMERATOR
        + MINIMUM_COVERAGE_DENOMINATOR
        - 1
    ) // MINIMUM_COVERAGE_DENOMINATOR


def minimum_sample_spacing_us(target_pps):
    """Return half a target-rate slot, rounded up to whole microseconds."""
    rate = int(target_pps)
    if rate <= 0:
        raise ValueError("target_pps must be positive")
    denominator = rate * SLOT_HALF_DENOMINATOR
    return max(1, (MICROSECONDS_PER_SECOND + denominator - 1) // denominator)


class TemporalCsiSampler:
    """Allocation-safe MicroPython facade over the core SDK sampler."""

    def __init__(self, target_pps, window_size_ms):
        if (
            _native_core is None
            or not hasattr(_native_core, "TemporalCsiSampler")
        ):
            raise RuntimeError(
                "Micro-ESPectre requires a compatible espectre core sampler"
            )
        self._native = _native_core.TemporalCsiSampler(
            target_pps,
            window_size_ms,
        )
        self.target_pps = self._native.get_u32(0)
        self.window_size_ms = self._native.get_u32(1)
        self.window_size_us = self.window_size_ms * 1000
        self.window_slots = self._native.get_u32(2)
        self.minimum_valid_slots = self._native.get_u32(3)
        self.minimum_sample_spacing_us = self._native.get_u32(4)

    def reset(self):
        self._native.reset()

    def clear_history(self):
        self._native.clear_history()

    def clear_window_preserving_phase(self):
        self._native.clear_window_preserving_phase()

    def admit(self, timestamp_us, now_us=None):
        return self._native.admit(timestamp_us, now_us)

    def flush(self):
        return self._native.flush()

    @property
    def current_slot(self):
        return self._native.get_u64(0)

    @property
    def has_pending_candidate(self):
        return self._native.get_flag(3)

    @property
    def occupancy_slots(self):
        return self._native.get_u32(5)

    @property
    def occupancy_ratio(self):
        return self._native.occupancy_ratio()

    @property
    def is_ready(self):
        return self._native.get_flag(0)

    @property
    def accepted(self):
        return self._native.get_flag(1)

    @property
    def selected_current(self):
        return self._native.get_flag(2)

    @property
    def reset_required(self):
        return self._native.get_flag(4)

    @property
    def gap_reset_required(self):
        return self._native.get_flag(5)

    @property
    def slots_advanced(self):
        return self._native.get_u64(1)

    @property
    def missing_slots_before(self):
        return self._native.get_u64(2)

    @property
    def accepted_packets(self):
        return self._native.get_u64(3)

    @property
    def excess_packets(self):
        return self._native.get_u64(4)

    @property
    def duplicate_packets(self):
        return self._native.get_u64(5)

    @property
    def out_of_order_packets(self):
        return self._native.get_u64(6)

    @property
    def stale_packets(self):
        return self._native.get_u64(7)

    @property
    def missing_timestamp_packets(self):
        return self._native.get_u64(8)

    @property
    def missing_slots(self):
        return self._native.get_u64(9)

    @property
    def gap_resets(self):
        return self._native.get_u64(10)

    def close(self):
        native_sampler = getattr(self, "_native", None)
        if native_sampler is not None:
            native_sampler.deinit()
            self._native = None

    def __del__(self):
        self.close()
