# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
Micro-ESPectre - Wrap-aware serial sequence tracking

Author: Francesco Pace <francesco.pace@gmail.com>
"""

UINT32_MASK = 0xFFFFFFFF
UINT32_HALF_RANGE = 0x80000000

ORDER_FIRST = "first"
ORDER_FORWARD = "forward"
ORDER_DUPLICATE = "duplicate"
ORDER_BACKWARD = "backward"
ORDER_RESYNC = "resync"


class WrappingTimestampOrder:
    """Order wrapping 32-bit timestamps without stalling after a long silence.

    Mirrors ``core/wrapping_timestamp_order.h``. A forward delta of half the
    range or more reads as backward; rejected values that keep advancing
    among themselves for ``resync_span`` restart the sequence, while an
    isolated late value stays rejected. A repeated value inside such a run is
    a duplicate and leaves the run intact.
    """

    def __init__(self, resync_span):
        self.set_resync_span(resync_span)
        self.reset()

    def set_resync_span(self, resync_span):
        self.resync_span = max(1, int(resync_span))

    def reset(self):
        self.last = None
        self.delta = 0
        self._run_origin = None
        self._run_last = 0

    def observe(self, value):
        """Classify ``value`` and accept it unless it is a duplicate or backward."""
        value = int(value) & UINT32_MASK
        self.delta = 0
        if self.last is None:
            self.last = value
            self._run_origin = None
            return ORDER_FIRST
        delta = (value - self.last) & UINT32_MASK
        if delta == 0:
            return ORDER_DUPLICATE
        if delta < UINT32_HALF_RANGE:
            self.last = value
            self.delta = delta
            self._run_origin = None
            return ORDER_FORWARD
        if self._run_origin is not None:
            run_step = (value - self._run_last) & UINT32_MASK
            # A repeated value neither advances nor breaks the run.
            if run_step == 0:
                return ORDER_DUPLICATE
            if run_step < UINT32_HALF_RANGE:
                self._run_last = value
                if ((value - self._run_origin) & UINT32_MASK) < self.resync_span:
                    return ORDER_BACKWARD
                self.last = value
                self._run_origin = None
                return ORDER_RESYNC
        # Start a run, or restart it when it steps back.
        self._run_origin = value
        self._run_last = value
        return ORDER_BACKWARD


class SerialSequenceTracker:
    """Track a wrapping 32-bit serial without advancing on stale values."""

    def __init__(self):
        self.reset()

    def observe(self, value):
        """Return missing values before a fresh serial, or -1 when stale."""
        current = int(value) & 0xFFFFFFFF
        if not self._initialized:
            self._last = current
            self._initialized = True
            return 0

        delta = (current - self._last) & 0xFFFFFFFF
        if delta == 0 or delta >= 0x80000000:
            return -1

        self._last = current
        return delta - 1

    def seed(self, value):
        self._last = int(value) & 0xFFFFFFFF
        self._initialized = True

    def reset(self):
        self._last = 0
        self._initialized = False
