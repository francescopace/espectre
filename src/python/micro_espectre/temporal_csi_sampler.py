# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Fixed-grid temporal CSI admission shared by device and host workflows."""

MICROSECONDS_PER_SECOND = 1_000_000
UINT32_MODULUS = 1 << 32
UINT32_HALF_RANGE = 1 << 31
MINIMUM_COVERAGE_NUMERATOR = 4
MINIMUM_COVERAGE_DENOMINATOR = 5


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
    """Return the shared four-fifths occupancy floor, rounded up."""
    slots = max(1, int(window_slots))
    return (
        slots * MINIMUM_COVERAGE_NUMERATOR
        + MINIMUM_COVERAGE_DENOMINATOR
        - 1
    ) // MINIMUM_COVERAGE_DENOMINATOR


def minimum_sample_spacing_us(target_pps):
    """Return the minimum separation for two independent target-rate samples."""
    rate = int(target_pps)
    if rate <= 0:
        raise ValueError("target_pps must be positive")
    numerator = MICROSECONDS_PER_SECOND * MINIMUM_COVERAGE_NUMERATOR
    denominator = rate * MINIMUM_COVERAGE_DENOMINATOR
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
        self.accepted = False
        self.reset_required = False
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
        """Invalidate the active window while preserving lifetime counters."""
        self._clear_window()
        self._last_timestamp = None
        self._elapsed_us = 0
        self._last_admitted_slot = None
        self._last_admitted_elapsed_us = None
        self.accepted = False
        self.reset_required = False
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
        self.reset_required = False
        self.slots_advanced = 0
        self.missing_slots_before = 0
        return False

    def _accept_slot(self, slot, slots_advanced, missing_before):
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
        self._last_admitted_elapsed_us = self._elapsed_us
        self.accepted = True
        self.slots_advanced = slots_advanced
        self.missing_slots_before = missing_before
        self.accepted_packets += 1
        self.missing_slots += missing_before
        return True

    def admit(self, timestamp_us, now_us=None):
        """Return ``True`` when the timestamp owns a new temporal slot.

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
            return self._accept_slot(0, 0, 0)

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
            self.reset_required = True
            self._clear_window()
            self._elapsed_us = 0
            self._last_admitted_slot = None
            self._last_admitted_elapsed_us = None
            return self._accept_slot(0, 0, 0)

        self._elapsed_us += delta
        # Center bins on their ideal sampling instant. Flooring makes ordinary
        # +/- scheduling jitter pathological: 0, 9, 20, 29 ms at 100 pps maps
        # to 0, 0, 2, 2 instead of four independent samples.
        slot = (
            self._elapsed_us * self.target_pps
            + MICROSECONDS_PER_SECOND // 2
        ) // MICROSECONDS_PER_SECOND
        if slot <= self._last_admitted_slot:
            self.excess_packets += 1
            return False
        if (
            self._elapsed_us - self._last_admitted_elapsed_us
            < self.minimum_sample_spacing_us
        ):
            self.excess_packets += 1
            return False

        advanced = slot - self._last_admitted_slot
        missing = max(0, advanced - 1)
        return self._accept_slot(slot, advanced, missing)

    @property
    def current_slot(self):
        return self._last_admitted_slot

    @property
    def occupancy_slots(self):
        return self._occupancy

    @property
    def occupancy_ratio(self):
        return self._occupancy / float(self.window_slots)

    @property
    def is_ready(self):
        return (
            self._last_admitted_slot is not None
            and self._last_admitted_slot + 1 >= self.window_slots
            and self._occupancy >= self.minimum_valid_slots
        )
