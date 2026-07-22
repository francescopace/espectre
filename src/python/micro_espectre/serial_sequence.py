"""
Micro-ESPectre - Wrap-aware serial sequence tracking

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""


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
