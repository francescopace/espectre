# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Fixed-grid temporal CSI admission shared by device and host workflows."""

import sys

_native_core = None
if getattr(sys.implementation, "name", "") == "micropython":
    try:
        import espectre_native_features as _native_core
    except ImportError:
        raise RuntimeError(
            "Micro-ESPectre requires the espectre_native_features core module"
        )

MICROSECONDS_PER_SECOND = 1_000_000
UINT32_MODULUS = 1 << 32
UINT32_HALF_RANGE = 1 << 31
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
    numerator = MICROSECONDS_PER_SECOND
    denominator = rate * SLOT_HALF_DENOMINATOR
    return max(1, (numerator + denominator - 1) // denominator)


class TemporalCsiSampler:
    """Admit at most one CSI packet in each configured temporal slot.

    The packet timestamp is an unsigned 32-bit microsecond clock. Decisions are
    exposed through allocation-free properties so the same class can run in a
    MicroPython CSI loop and in CPython replay or training code.
    """

    def __init__(self, target_pps, window_size_ms):
        self.target_pps = int(target_pps)
        self.window_size_ms = int(window_size_ms)
        self.window_size_us = self.window_size_ms * 1000
        self.window_slots = temporal_window_slots(
            self.target_pps, self.window_size_ms
        )
        self.minimum_valid_slots = minimum_valid_slots(self.window_slots)
        self.minimum_sample_spacing_us = minimum_sample_spacing_us(
            self.target_pps
        )
        self._slot_ids = [-1] * self.window_slots
        self.reset()

    def reset(self):
        """Clear temporal history and diagnostic counters."""
        self._clear_window()
        self._last_timestamp = None
        self._elapsed_us = 0
        self._last_admitted_slot = None
        self._last_admitted_elapsed_us = None
        self._window_origin_slot = None
        self._reported_slot = None
        self._active_slot = None
        self._pending_slot = None
        self._pending_elapsed_us = None
        self._pending_center_error = None
        self._pending_reset_required = False
        self.accepted = False
        self.selected_current = False
        self.reset_required = False
        self.gap_reset_required = False
        self.slots_advanced = 0
        self.missing_slots_before = 0
        self.accepted_packets = 0
        self.excess_packets = 0
        self.duplicate_packets = 0
        self.out_of_order_packets = 0
        self.stale_packets = 0
        self.missing_timestamp_packets = 0
        self.missing_slots = 0
        self.gap_resets = 0

    def clear_history(self):
        """Start a new temporal epoch while preserving lifetime counters."""
        self._clear_window()
        self._last_timestamp = None
        self._elapsed_us = 0
        self._last_admitted_slot = None
        self._last_admitted_elapsed_us = None
        self._window_origin_slot = None
        self._reported_slot = None
        self._active_slot = None
        self._pending_slot = None
        self._pending_elapsed_us = None
        self._pending_center_error = None
        self._pending_reset_required = False
        self.accepted = False
        self.selected_current = False
        self.reset_required = False
        self.gap_reset_required = False
        self.slots_advanced = 0
        self.missing_slots_before = 0

    def clear_window_preserving_phase(self):
        """Clear window data without changing the temporal grid phase."""
        self._clear_window()
        self._window_origin_slot = self._active_slot
        self._pending_slot = None
        self._pending_elapsed_us = None
        self._pending_center_error = None
        self._pending_reset_required = False
        self.accepted = False
        self.selected_current = False
        self.reset_required = False
        self.gap_reset_required = False
        self.slots_advanced = 0
        self.missing_slots_before = 0

    def _clear_window(self):
        for index in range(len(self._slot_ids)):
            self._slot_ids[index] = -1
        self._occupancy = 0

    @staticmethod
    def _forward_delta(current, previous):
        return (int(current) - int(previous)) % UINT32_MODULUS

    def _drop(self):
        self.accepted = False
        self.selected_current = False
        self.reset_required = False
        self.gap_reset_required = False
        self.slots_advanced = 0
        self.missing_slots_before = 0
        return False

    def _slot_for_elapsed(self, elapsed_us):
        return (
            int(elapsed_us) * self.target_pps
            + MICROSECONDS_PER_SECOND // 2
        ) // MICROSECONDS_PER_SECOND

    def _select_candidate(self, slot, elapsed_us, reset_required=False):
        scaled_elapsed = int(elapsed_us) * self.target_pps
        scaled_center = int(slot) * MICROSECONDS_PER_SECOND
        center_error = abs(scaled_elapsed - scaled_center)
        if (
            self._last_admitted_slot is not None
            and slot <= self._last_admitted_slot
        ):
            self.excess_packets += 1
            return False
        if (
            self._last_admitted_elapsed_us is not None
            and elapsed_us - self._last_admitted_elapsed_us
            < self.minimum_sample_spacing_us
        ):
            self.excess_packets += 1
            return False
        if (
            self._pending_slot is not None
            and center_error >= self._pending_center_error
        ):
            self.excess_packets += 1
            return False
        if self._pending_slot is not None:
            self.excess_packets += 1
        self._pending_slot = int(slot)
        self._pending_elapsed_us = int(elapsed_us)
        self._pending_center_error = int(center_error)
        self._pending_reset_required = bool(reset_required)
        self.selected_current = True
        return True

    def _commit_candidate(self):
        if self._pending_slot is None:
            return False
        slot = self._pending_slot
        slots_advanced = (
            slot - self._last_admitted_slot
            if self._last_admitted_slot is not None
            else 0
        )
        missing_before = (
            max(0, slots_advanced - 1)
            if self._last_admitted_slot is not None
            else 0
        )
        if self._last_admitted_slot is not None:
            if slots_advanced >= self.window_slots:
                self._clear_window()
            else:
                first = self._last_admitted_slot + 1
                for expired_slot in range(first, slot + 1):
                    index = expired_slot % self.window_slots
                    if self._slot_ids[index] >= 0:
                        self._slot_ids[index] = -1
                        self._occupancy -= 1

        index = slot % self.window_slots
        if self._slot_ids[index] < 0:
            self._occupancy += 1
        self._slot_ids[index] = slot
        self._last_admitted_slot = slot
        self._reported_slot = slot
        self._last_admitted_elapsed_us = self._pending_elapsed_us
        self.accepted = True
        self.reset_required = self._pending_reset_required
        self.slots_advanced = slots_advanced
        self.missing_slots_before = missing_before
        self.accepted_packets += 1
        self.missing_slots += missing_before
        self._pending_slot = None
        self._pending_elapsed_us = None
        self._pending_center_error = None
        self._pending_reset_required = False
        return True

    def admit(self, timestamp_us, now_us=None):
        """Observe a timestamp and return ``True`` when a prior slot is emitted.

        ``now_us`` is optional and must use the same unsigned 32-bit clock. It
        lets live runtimes reject a packet that sat in a processing backlog for
        at least one detector window; deterministic replay normally omits it.
        """
        self._drop()
        if timestamp_us is None:
            self.missing_timestamp_packets += 1
            return False

        timestamp = int(timestamp_us) % UINT32_MODULUS
        if now_us is not None:
            age = self._forward_delta(int(now_us) % UINT32_MODULUS, timestamp)
            if self.window_size_us <= age < UINT32_HALF_RANGE:
                self.stale_packets += 1
                return False

        if self._last_timestamp is None:
            self._last_timestamp = timestamp
            self._elapsed_us = 0
            slot = self._slot_for_elapsed(0)
            self._active_slot = slot
            self._window_origin_slot = slot
            self._select_candidate(slot, 0)
            return False

        delta = self._forward_delta(timestamp, self._last_timestamp)
        if delta == 0:
            self.duplicate_packets += 1
            return False
        if delta >= UINT32_HALF_RANGE:
            self.out_of_order_packets += 1
            return False

        self._last_timestamp = timestamp
        if delta >= self.window_size_us:
            self.gap_resets += 1
            emitted = self._commit_candidate()
            self._clear_window()
            self._elapsed_us = 0
            self._last_admitted_slot = None
            self._last_admitted_elapsed_us = None
            slot = self._slot_for_elapsed(0)
            self._active_slot = slot
            self._window_origin_slot = slot
            self._select_candidate(slot, 0, reset_required=True)
            self.gap_reset_required = True
            return emitted

        self._elapsed_us += delta
        # Center bins on their ideal sampling instant. Flooring makes ordinary
        # +/- scheduling jitter pathological: 0, 9, 20, 29 ms at 100 pps maps
        # to 0, 0, 2, 2 instead of four independent samples.
        slot = self._slot_for_elapsed(self._elapsed_us)
        if self._active_slot is not None and slot < self._active_slot:
            self.excess_packets += 1
            return False
        emitted = False
        if self._active_slot is None or slot > self._active_slot:
            emitted = self._commit_candidate()
            self._active_slot = slot
        self._select_candidate(slot, self._elapsed_us)
        return emitted

    def flush(self):
        """Emit the final buffered slot at the end of a finite replay."""
        self._drop()
        return self._commit_candidate()

    @property
    def current_slot(self):
        return self._reported_slot

    @property
    def has_pending_candidate(self):
        return self._pending_slot is not None

    @property
    def occupancy_slots(self):
        return self._occupancy

    @property
    def occupancy_ratio(self):
        return self._occupancy / float(self.window_slots)

    @property
    def is_ready(self):
        return (
            self._window_origin_slot is not None
            and self._last_admitted_slot is not None
            and self._last_admitted_slot >= self._window_origin_slot
            and self._last_admitted_slot - self._window_origin_slot + 1
            >= self.window_slots
            and self._occupancy >= self.minimum_valid_slots
        )


if _native_core is not None:
    if not hasattr(_native_core, "TemporalCsiSampler"):
        raise RuntimeError(
            "Micro-ESPectre requires a compatible espectre core sampler"
        )

    class TemporalCsiSampler:
        """Allocation-safe MicroPython facade over the core SDK sampler."""

        def __init__(self, target_pps, window_size_ms):
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
