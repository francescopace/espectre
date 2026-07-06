"""
Micro-ESPectre - L1-Delta Detector Tests

Unit tests for the normalized amplitude-profile displacement detector.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

from detector_interface import MotionState
from l1_delta_detector import L1DeltaDetector

BAND = [14, 17, 20, 23, 26, 29, 35, 38, 41, 44, 47, 50]


def make_csi(amplitudes_by_subcarrier):
    """Build a 64-subcarrier int8 I/Q payload with the given real amplitudes."""
    csi = [0] * 128
    for sc_idx, amplitude in amplitudes_by_subcarrier.items():
        csi[sc_idx * 2] = 0  # imag
        csi[sc_idx * 2 + 1] = int(amplitude)  # real
    return csi


def band_profile(scale=1):
    """A frequency-selective band profile (distinct amplitude per subcarrier)."""
    return {sc: (20 + 3 * (i % 5)) * scale for i, sc in enumerate(BAND)}


def shifted_profile(scale=1):
    """A different profile shape (multipath changed by motion)."""
    return {sc: (20 + 3 * ((i + 2) % 5)) * scale for i, sc in enumerate(BAND)}


def run_packets(detector, payloads):
    metrics = None
    for payload in payloads:
        detector.process_packet(payload, BAND)
        metrics = detector.update_state()
    return metrics


def test_quiet_profile_stays_idle_with_zero_metric():
    detector = L1DeltaDetector(window_size=20, threshold=0.05, lag=5)
    payload = make_csi(band_profile())
    metrics = run_packets(detector, [payload] * 40)

    assert detector.is_ready()
    assert metrics["motion_metric"] == 0.0
    assert metrics["state"] == MotionState.IDLE


def test_profile_change_raises_metric_and_triggers_motion():
    detector = L1DeltaDetector(window_size=20, threshold=0.05, lag=5)
    quiet = make_csi(band_profile())
    moved = make_csi(shifted_profile())

    run_packets(detector, [quiet] * 30)
    # Alternate profiles: with an odd lag every comparison crosses a change.
    payloads = [quiet if i % 2 == 0 else moved for i in range(40)]
    metrics = run_packets(detector, payloads)

    assert metrics["motion_metric"] > 0.05
    assert metrics["state"] == MotionState.MOTION


def test_metric_is_invariant_to_packet_gain():
    payload_sequence = []
    for i in range(60):
        profile = band_profile() if i % 3 else shifted_profile()
        payload_sequence.append(profile)

    detector_1x = L1DeltaDetector(window_size=20, threshold=1.0, lag=5)
    detector_2x = L1DeltaDetector(window_size=20, threshold=1.0, lag=5)
    metrics_1x = run_packets(detector_1x, [make_csi(p) for p in payload_sequence])
    metrics_2x = run_packets(
        detector_2x,
        [make_csi({sc: amp * 2 for sc, amp in p.items()}) for p in payload_sequence],
    )

    assert abs(metrics_1x["motion_metric"] - metrics_2x["motion_metric"]) < 1e-12


def test_not_ready_until_window_full():
    detector = L1DeltaDetector(window_size=20, threshold=0.05, lag=5)
    payload = make_csi(band_profile())

    metrics = run_packets(detector, [payload] * (5 + 20 - 1))
    assert not detector.is_ready()
    assert metrics["motion_metric"] == 0.0

    run_packets(detector, [payload])
    assert detector.is_ready()


def test_startup_threshold_factor_and_adaptive_clamp():
    detector = L1DeltaDetector()
    assert L1DeltaDetector.STARTUP_THRESHOLD_FACTOR == 1.1
    assert L1DeltaDetector.STARTUP_GATE is True

    detector.set_adaptive_threshold(0.0)
    assert detector.get_threshold() == 1e-6
    detector.set_adaptive_threshold(42.0)
    assert detector.get_threshold() == 10.0
    detector.set_adaptive_threshold(0.03)
    assert detector.get_threshold() == 0.03


def test_reset_clears_state():
    detector = L1DeltaDetector(window_size=20, threshold=0.05, lag=5)
    quiet = make_csi(band_profile())
    moved = make_csi(shifted_profile())
    run_packets(detector, [quiet if i % 2 else moved for i in range(60)])
    assert detector.is_ready()

    detector.reset()
    assert not detector.is_ready()
    assert detector.get_motion_metric() == 0.0
    assert detector.get_state() == MotionState.IDLE
    assert detector.total_packets == 0
