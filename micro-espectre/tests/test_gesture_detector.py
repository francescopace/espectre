"""
Micro-ESPectre - Gesture Detector Tests

Unit and integration tests for gesture_detector.py.
Tests the ring buffer, event lifecycle, and inference routing.

Since gesture_weights.py may not exist yet (no model trained),
inference tests gracefully skip when weights are unavailable.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import sys
import math
from pathlib import Path

import pytest
import numpy as np

src_path = str(Path(__file__).parent.parent / 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from gesture_detector import (
    GestureDetector,
    RING_BUFFER_LEN,
    MAX_EVENT_LEN,
    GESTURE_PREROLL_LEN,
)


# ============================================================================
# Helpers
# ============================================================================

def make_fake_csi(n_bytes=128, amplitude=100):
    """Generate fake CSI bytes with known amplitude."""
    csi = bytes([amplitude & 0x7F] * n_bytes)
    return csi


def feed_packets(gd, n, csi=None):
    """Feed n packets to a GestureDetector."""
    if csi is None:
        csi = make_fake_csi()
    for _ in range(n):
        gd.process_packet(csi, None)


# ============================================================================
# Initialization
# ============================================================================

class TestGestureDetectorInit:
    def test_initial_state_inactive(self):
        gd = GestureDetector()
        assert not gd.is_active

    def test_initial_event_len_zero(self):
        gd = GestureDetector()
        assert gd.event_len == 0

    def test_initial_last_gesture_none(self):
        gd = GestureDetector()
        assert gd.get_last_gesture() is None


# ============================================================================
# Ring buffer
# ============================================================================

class TestRingBuffer:
    def test_partial_fill_preroll(self):
        gd = GestureDetector()
        feed_packets(gd, 30)
        gd.start_detection()
        # Pre-roll should be 30 packets
        assert gd.event_len == 30

    def test_full_ring_preroll(self):
        gd = GestureDetector()
        feed_packets(gd, RING_BUFFER_LEN + 50)
        gd.start_detection()
        # Pre-roll should be capped to fixed pre-roll length.
        assert gd.event_len == GESTURE_PREROLL_LEN

    def test_ring_fills_incrementally(self):
        gd = GestureDetector()
        for n in [10, 20, 50]:
            gd2 = GestureDetector()
            feed_packets(gd2, n)
            gd2.start_detection()
            assert gd2.event_len == n

    def test_clear_ring(self):
        gd = GestureDetector()
        feed_packets(gd, 50)
        gd.start_detection()
        gd.clear_ring()
        assert not gd.is_active
        assert gd.event_len == 0

        # After clear, pre-roll should be 0
        gd.start_detection()
        assert gd.event_len == 0


# ============================================================================
# Event lifecycle
# ============================================================================

class TestEventLifecycle:
    def test_active_after_motion_start(self):
        gd = GestureDetector()
        gd.start_detection()
        assert gd.is_active

    def test_inactive_after_motion_end(self):
        gd = GestureDetector()
        feed_packets(gd, 30)
        gd.start_detection()
        feed_packets(gd, 20)
        gd.finalize_detection()
        assert not gd.is_active

    def test_event_accumulation_while_active(self):
        gd = GestureDetector()
        feed_packets(gd, 20)
        gd.start_detection()
        preroll = gd.event_len

        feed_packets(gd, 30)
        assert gd.event_len == preroll + 30

    def test_no_accumulation_when_inactive(self):
        gd = GestureDetector()
        feed_packets(gd, 50)
        # Not started yet: feeding should not accumulate
        assert gd.event_len == 0

    def test_event_cleared_after_end(self):
        gd = GestureDetector()
        feed_packets(gd, 20)
        gd.start_detection()
        feed_packets(gd, 20)
        gd.finalize_detection()
        assert gd.event_len == 0

    def test_max_event_len_capped(self):
        gd = GestureDetector()
        gd.start_detection()
        feed_packets(gd, MAX_EVENT_LEN + 100)
        assert gd.event_len == MAX_EVENT_LEN

    def test_short_event_returns_none(self):
        gd = GestureDetector()
        # Start with empty ring → no pre-roll
        gd.start_detection()
        feed_packets(gd, 5)  # < 10 minimum
        result = gd.finalize_detection()
        assert result is None

    def test_motion_start_resets_active_event(self):
        gd = GestureDetector()
        feed_packets(gd, 10)
        gd.start_detection()
        feed_packets(gd, 10)

        # Second start (re-trigger) should reset event buffer
        gd.start_detection()
        # Event should now contain fresh pre-roll from ring (not from old event)
        # The ring now has RING_BUFFER_LEN available packets at most
        assert gd.event_len <= RING_BUFFER_LEN


# ============================================================================
# Inference (weights-conditional)
# ============================================================================

class TestInference:
    def test_finalize_detection_no_weights_or_result(self):
        """finalize_detection should return None for incomplete (<200 pkt) windows."""
        gd = GestureDetector()
        feed_packets(gd, 50)
        gd.start_detection()
        feed_packets(gd, 50)
        result = gd.finalize_detection()
        assert result is None

    def test_last_gesture_updated(self):
        """get_last_gesture() returns the last classification result."""
        gd = GestureDetector()
        feed_packets(gd, 50)
        gd.start_detection()
        feed_packets(gd, 50)
        result = gd.finalize_detection()

        if result is not None:
            assert gd.get_last_gesture() == result
        else:
            # No classification happened, last_gesture may be None or stale
            pass

    def test_weights_available_property(self):
        """weights_available is a boolean."""
        gd = GestureDetector()
        assert isinstance(gd.weights_available, bool)


# ============================================================================
# Reset
# ============================================================================

class TestReset:
    def test_reset_clears_event(self):
        gd = GestureDetector()
        feed_packets(gd, 20)
        gd.start_detection()
        feed_packets(gd, 20)
        gd.reset()
        assert not gd.is_active
        assert gd.event_len == 0
        assert gd.get_last_gesture() is None

    def test_reset_does_not_clear_ring(self):
        gd = GestureDetector()
        feed_packets(gd, RING_BUFFER_LEN)
        gd.reset()
        # Ring is preserved: start_detection should still have fixed pre-roll.
        gd.start_detection()
        assert gd.event_len == GESTURE_PREROLL_LEN


# ============================================================================
# Integration: full event cycle
# ============================================================================

class TestFullCycle:
    def test_full_idle_to_motion_to_idle_cycle(self):
        gd = GestureDetector()

        # Simulate idle period: fill ring buffer
        feed_packets(gd, RING_BUFFER_LEN)
        assert not gd.is_active

        # IDLE → MOTION transition
        gd.start_detection()
        assert gd.is_active
        assert gd.event_len == GESTURE_PREROLL_LEN  # fixed pre-roll

        # Accumulate event packets
        feed_packets(gd, 50)
        assert gd.event_len == GESTURE_PREROLL_LEN + 50

        # MOTION → IDLE transition
        result = gd.finalize_detection()
        assert not gd.is_active
        assert gd.event_len == 0

        # result is None (no weights) or a gesture name
        if result is not None:
            assert isinstance(result, str)

    def test_multiple_events_in_sequence(self):
        gd = GestureDetector()

        for _ in range(3):
            feed_packets(gd, 50)
            gd.start_detection()
            feed_packets(gd, 30)
            result = gd.finalize_detection()
            assert not gd.is_active
            assert gd.event_len == 0

    def test_channel_change_resets_state(self):
        gd = GestureDetector()
        feed_packets(gd, 50)
        gd.start_detection()
        assert gd.is_active

        gd.clear_ring()  # simulate channel change

        assert not gd.is_active
        gd.start_detection()
        assert gd.event_len == 0  # no pre-roll after clear


# ============================================================================
# Additional edge case tests (for cross-platform validation)
# ============================================================================

class TestEdgeCases:
    def test_double_motion_start(self):
        """Second start_detection should reset event buffer."""
        gd = GestureDetector()
        feed_packets(gd, 50)
        gd.start_detection()
        first_len = gd.event_len
        feed_packets(gd, 20)
        second_len = gd.event_len

        # Second start_detection: resets event buffer with fresh pre-roll
        gd.start_detection()
        assert gd.is_active
        # Event should contain pre-roll from ring (at most RING_BUFFER_LEN)
        assert gd.event_len <= RING_BUFFER_LEN

    def test_very_long_event(self):
        """Event longer than MAX_EVENT_LEN should be capped."""
        gd = GestureDetector()
        gd.start_detection()
        feed_packets(gd, MAX_EVENT_LEN + 200)
        assert gd.event_len == MAX_EVENT_LEN

    def test_event_with_varying_amplitude(self):
        """Event with varying CSI amplitude should produce valid turbulence."""
        gd = GestureDetector()

        # Feed packets with increasing amplitude
        for amp in range(50, 110):
            csi = make_fake_csi(amplitude=amp)
            gd.process_packet(csi, None)

        gd.start_detection()
        # Should have some pre-roll
        assert gd.event_len > 0

        feed_packets(gd, 30)
        result = gd.finalize_detection()
        assert not gd.is_active
        # Result depends on weights availability

    def test_ring_buffer_wraparound(self):
        """Ring buffer should correctly wrap around when full."""
        gd = GestureDetector()

        # Fill ring buffer multiple times to ensure wraparound
        total_packets = RING_BUFFER_LEN * 3
        feed_packets(gd, total_packets)

        # Start event: should get exactly fixed pre-roll
        gd.start_detection()
        assert gd.event_len == GESTURE_PREROLL_LEN

    def test_finalize_detection_without_start(self):
        """finalize_detection without prior start should handle gracefully."""
        gd = GestureDetector()
        feed_packets(gd, 50)

        # Call finalize_detection without start_detection
        # This is an edge case - event buffer is empty
        result = gd.finalize_detection()
        assert result is None
        assert not gd.is_active

    def test_consecutive_events_independence(self):
        """Each event should be independent of previous ones."""
        gd = GestureDetector()
        results = []

        for i in range(3):
            feed_packets(gd, 40)
            gd.start_detection()
            feed_packets(gd, 30)
            result = gd.finalize_detection()
            results.append(result)
            assert not gd.is_active
            assert gd.event_len == 0

        # All events should complete successfully
        # (results may be None if weights unavailable)

    def test_process_packet_with_custom_subcarriers(self):
        """process_packet should accept custom subcarrier list."""
        gd = GestureDetector()
        csi = make_fake_csi()

        # Use custom subcarriers
        custom_sc = [10, 20, 30, 40]
        gd.process_packet(csi, custom_sc)

        # Should work without error
        assert True

    def test_minimum_event_length_boundary(self):
        """Test the boundary condition for minimum event length (10 packets)."""
        gd = GestureDetector()

        # Exactly 10 packets (should be accepted)
        gd.start_detection()
        feed_packets(gd, 10)
        assert gd.event_len == 10
        result = gd.finalize_detection()
        # Should attempt classification (result depends on weights)

        # Now test with 9 packets (should be rejected)
        gd2 = GestureDetector()
        gd2.start_detection()
        feed_packets(gd2, 9)
        assert gd2.event_len == 9
        result2 = gd2.finalize_detection()
        assert result2 is None  # Too short
