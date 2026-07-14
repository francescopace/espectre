"""
Micro-ESPectre - Classic Detector Tests

Unit tests for the L1-Delta + variance fusion detector: contract/registry, threshold
delegation to the L1-Delta primary, the fused decision (band x vote x
dispersion gate), and the frozen dispersion gate.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

from detector_interface import (
    MotionState,
    load_detector_class,
    get_detector_algorithm,
    detector_needs_startup_calibration,
)
from classic_detector import ClassicDetector
from features import (
    L1_DELTA_STARTUP_GATE,
    L1_DELTA_STARTUP_THRESHOLD_FACTOR,
)

BAND = [14, 17, 20, 23, 26, 29, 35, 38, 41, 44, 47, 50]


def make_csi(amplitudes_by_subcarrier):
    """Build a 64-subcarrier int8 I/Q payload with the given real amplitudes."""
    csi = [0] * 128
    for sc_idx, amplitude in amplitudes_by_subcarrier.items():
        csi[sc_idx * 2] = 0
        csi[sc_idx * 2 + 1] = int(amplitude)
    return csi


def band_profile(scale=1):
    return {sc: (20 + 3 * (i % 5)) * scale for i, sc in enumerate(BAND)}


def shifted_profile(scale=1):
    return {sc: (20 + 3 * ((i + 2) % 5)) * scale for i, sc in enumerate(BAND)}


class _FakeSub:
    """Minimal stand-in for a sub-detector, with controllable metric/state."""

    def __init__(self, metric=0.0, threshold=0.1, ready=True):
        self._m = metric
        self._thr = threshold
        self._ready = ready
        self.current_moving_variance = metric

    def process_packet(self, *args, **kwargs):
        pass

    def update_state(self):
        return {}

    def get_motion_metric(self):
        return self._m

    def get_threshold(self):
        return self._thr

    def is_ready(self):
        return self._ready

    def set_threshold(self, threshold):
        self._thr = threshold
        return True

    def set_adaptive_threshold(self, threshold):
        self._thr = threshold

    def reset(self):
        pass


def _decide(l1_metric, moving_variance, threshold=0.1, floor=1.0, vote=True,
            recovery_vote_configured=True):
    """Drive one fused decision with controlled sub-detector metrics."""
    det = ClassicDetector(
        window_size=10,
        threshold=threshold,
        enable_recovery_vote=recovery_vote_configured,
    )
    det._l1 = _FakeSub(l1_metric, threshold)
    det._variance_ctx = _FakeSub(moving_variance)
    det._floor_frozen = True                   # skip floor collection; use manual state
    det._variance_floor = floor
    det._recovery_vote_enabled = vote
    det.update_state()
    return det.get_state()


# --- contract / registry ---------------------------------------------------

def test_registry_exposes_classic():
    assert load_detector_class("classic") is ClassicDetector
    assert detector_needs_startup_calibration("classic")


def test_algorithm_and_name():
    det = ClassicDetector(window_size=100, threshold=1.0)
    assert ClassicDetector.ALGORITHM == "classic"
    assert det.get_name() == "Classic"
    assert get_detector_algorithm(det) == "classic"


def test_delegates_startup_gate_contract_to_l1_delta():
    assert ClassicDetector.STARTUP_THRESHOLD_FACTOR == L1_DELTA_STARTUP_THRESHOLD_FACTOR
    assert ClassicDetector.STARTUP_GATE == L1_DELTA_STARTUP_GATE


def test_adaptive_threshold_sets_primary_threshold():
    det = ClassicDetector(window_size=100, threshold=1.0)
    det.set_adaptive_threshold(0.05)
    assert abs(det.get_threshold() - 0.05) < 1e-9


# --- fused decision matrix -------------------------------------------------

def test_primary_l1_delta_fires_above_threshold():
    assert _decide(l1_metric=0.2, moving_variance=0.0) == MotionState.MOTION


def test_recovery_vote_recovers_in_band():
    # l1 in band (0.06 < 0.08 <= 0.1), variance above K*floor, vote enabled -> MOTION
    assert _decide(l1_metric=0.08, moving_variance=5.0) == MotionState.MOTION


def test_no_vote_when_variance_below_ratio():
    # l1 in band but variance not elevated -> IDLE
    assert _decide(l1_metric=0.08, moving_variance=2.0) == MotionState.IDLE


def test_no_vote_when_gate_disabled():
    # l1 in band, variance elevated, but the dispersion gate disabled the vote -> IDLE
    assert _decide(l1_metric=0.08, moving_variance=5.0, vote=False) == MotionState.IDLE


def test_no_vote_when_disabled_by_configuration():
    assert _decide(
        l1_metric=0.08,
        moving_variance=5.0,
        recovery_vote_configured=False,
    ) == MotionState.IDLE


def test_disabled_vote_does_not_allocate_variance_context():
    det = ClassicDetector(window_size=10, enable_recovery_vote=False)
    assert det._variance_ctx is None


def test_disabled_vote_can_update_state_without_variance_context():
    det = ClassicDetector(window_size=10, enable_recovery_vote=False)

    metrics = det.update_state()

    assert metrics["state"] == MotionState.IDLE
    assert metrics["moving_variance"] == 0.0


def test_no_vote_below_band():
    # deep-quiet l1 (below BAND_ALPHA*thr) never triggers the vote -> IDLE
    assert _decide(l1_metric=0.03, moving_variance=5.0) == MotionState.IDLE


# --- integration on crafted CSI --------------------------------------------

def test_quiet_profile_stays_idle():
    det = ClassicDetector(window_size=20, threshold=0.05)
    det._l1.lag = 5
    payload = make_csi(band_profile())
    for _ in range(60):
        det.process_packet(payload, BAND)
        det.update_state()
    assert det.is_ready()
    assert det.get_state() == MotionState.IDLE


def test_strong_motion_triggers_via_primary():
    det = ClassicDetector(window_size=20, threshold=0.05)
    quiet = make_csi(band_profile())
    moved = make_csi(shifted_profile())
    for _ in range(30):
        det.process_packet(quiet, BAND)
        det.update_state()
    saw_motion = False
    for i in range(60):
        det.process_packet(quiet if i % 2 == 0 else moved, BAND)
        det.update_state()
        if det.get_state() == MotionState.MOTION:
            saw_motion = True
    assert saw_motion


def test_reset_preserves_frozen_floor_state():
    det = ClassicDetector(window_size=20, threshold=0.05)
    det.apply_startup_floor(1.0, True, 400)
    det._floor_frozen = True
    det.reset()
    assert det._variance_floor == 1.0
    assert det._recovery_vote_enabled
    assert det._floor_frozen
    assert det.get_state() == MotionState.IDLE
    assert det.total_packets == 0


def test_classic_does_not_allocate_legacy_floor_ring():
    det = ClassicDetector(window_size=20)

    assert not hasattr(det, "_variance_floor_ring")


def test_apply_startup_floor_disables_vote_when_snapshot_too_small():
    det = ClassicDetector(window_size=20, threshold=0.05)
    det.apply_startup_floor(1.0, True, 10)
    assert det._variance_floor == 1.0
    assert not det._recovery_vote_enabled


def test_apply_startup_floor_keeps_configured_vote_disabled():
    det = ClassicDetector(window_size=20, enable_recovery_vote=False)
    det.apply_startup_floor(1.0, True, 400)
    assert det._floor_count == 0
    assert det._variance_floor is None
    assert not det._recovery_vote_enabled
