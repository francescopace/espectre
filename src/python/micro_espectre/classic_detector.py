"""
Micro-ESPectre - Classic Detector

Non-ML fusion detector: L1-Delta primary with a variance recovery vote.

L1-Delta drives detection with its stable, cross-session quiet floor and its
startup gate. Moving variance is consulted only as a recovery vote in the
ambiguous band just below the L1-Delta threshold, where L1-Delta and variance
complementary failure modes: gentle motion that barely displaces the normalized
profile can still raise the variance. The vote is threshold-free (a relative
test against the variance metric's own session floor) and self-gated by that floor's
dispersion, so the variance path is used only where its floor is tight enough to
trust. On rooms where L1-Delta already fires, or where the variance floor is bursty
(the S3 quiet-run failure mode), the vote stays out and the detector reduces to
L1-Delta alone.

Fusion decision per packet (l1 = L1-Delta metric, thr = L1-Delta threshold,
variance = moving variance, base = variance session floor):

    l1 > thr                                          -> MOTION  (L1-Delta)
    BAND_ALPHA*thr < l1 <= thr and vote and variance>K*base -> MOTION  (variance recovery)
    else                                              -> IDLE

    The variance floor is frozen from startup-validated quiet samples supplied by the
    shared calibrator. This keeps motion-first startup from poisoning the floor while
    preserving the existing low-contrast recovery vote when enough quiet samples were
    observed after the useful motion segment.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""
try:
    from src.detector_interface import IDetector, MotionState
    from src.features import (
        L1DeltaTracker,
        L1_DELTA_STARTUP_GATE,
        L1_DELTA_STARTUP_THRESHOLD_FACTOR,
    )
    from src.segmentation import SegmentationContext
except ImportError:
    from detector_interface import IDetector, MotionState
    from features import (
        L1DeltaTracker,
        L1_DELTA_STARTUP_GATE,
        L1_DELTA_STARTUP_THRESHOLD_FACTOR,
    )
    from segmentation import SegmentationContext


class ClassicDetector(IDetector):
    """
    L1-Delta + variance fusion detector (the default non-ML detector).

    Uses an embedded L1-delta tracker as the primary path and a shared moving-
    variance context as the support vote. The primary threshold and startup
    calibration are L1-delta's; the variance side is self-maintained online without
    exposing standalone detector classes.
    """
    ALGORITHM = "classic"

    # Delegate the startup-gate contract to L1-Delta (see threshold.py).
    STARTUP_THRESHOLD_FACTOR = L1_DELTA_STARTUP_THRESHOLD_FACTOR
    STARTUP_GATE = L1_DELTA_STARTUP_GATE

    # Fusion parameters (benchmark-tuned on the paired + long-quiet sets;
    # see the ClassicDetector promotion ADR).
    BAND_ALPHA = 0.6            # lower edge of the ambiguous band (x threshold)
    RECOVERY_VOTE_RATIO = 3.0       # variance must exceed K x its own session floor
    VARIANCE_FLOOR_MIN = 300        # samples before the vote may enable

    def __init__(self,
                 window_size=100,
                 threshold=1.0,
                 enable_lowpass=False,
                 lowpass_cutoff=11.0,
                 enable_hampel=True,
                 hampel_window=7,
                 hampel_threshold=5.0,
                 enable_recovery_vote=True,
                 **_unused):
        """
        Initialize the fusion detector.

        Args mirror the shared detector factory contract. Filter kwargs are
        forwarded to the variance support detector; L1-Delta ignores them.
        """
        self._l1 = L1DeltaTracker(window_size=window_size, threshold=threshold)
        self._recovery_vote_configured = bool(enable_recovery_vote)
        # Do not allocate or update the variance path in L1-only mode.
        self._variance_ctx = None
        if self._recovery_vote_configured:
            self._variance_ctx = SegmentationContext(
                window_size=window_size,
                threshold=10.0,
                enable_lowpass=enable_lowpass,
                lowpass_cutoff=lowpass_cutoff,
                enable_hampel=enable_hampel,
                hampel_window=hampel_window,
                hampel_threshold=hampel_threshold,
                allocate_amplitude_buffer=False,
            )

        self._floor_count = 0
        self._variance_floor = None           # median supplied by startup calibration
        self._recovery_vote_enabled = False   # dispersion gate result
        self._floor_frozen = False            # set once startup calibration completes

        self._state = MotionState.IDLE
        self._packet_count = 0
        self._last_moving_variance = 0.0

    # -- hot path -----------------------------------------------------------

    def process_packet(self, csi_data, selected_subcarriers=None):
        """Feed the packet to both the primary and the support detector."""
        self._packet_count += 1
        self._l1.process_packet(csi_data, selected_subcarriers)
        if self._recovery_vote_configured and self._variance_ctx is not None:
            turbulence = self._variance_ctx.calculate_turbulence_from_amplitudes(
                self._l1._amplitude_buffer,
                self._l1._amplitude_count,
            )
            self._variance_ctx.add_turbulence(turbulence)

    def update_state(self):
        """
        Update the fused motion state (call at publish time).

        Returns:
            dict: shared `motion_metric` (L1-Delta) plus fusion diagnostics.
        """
        if hasattr(self._l1, "update_metric"):
            self._l1.update_metric()
        else:
            self._l1.update_state()
        l1v = self._l1.get_motion_metric()
        moving_variance = 0.0
        thr = self._l1.threshold if hasattr(self._l1, "threshold") else self._l1.get_threshold()
        recovery_band = l1v > self.BAND_ALPHA * thr and l1v <= thr
        variance_ctx = self._variance_ctx
        variance_ready = (
            variance_ctx is not None
            and (
                not hasattr(variance_ctx, "buffer_count")
                or not hasattr(variance_ctx, "window_size")
                or variance_ctx.buffer_count >= variance_ctx.window_size
            )
        )
        variance_needed = (
            self._recovery_vote_configured
            and variance_ready
            and (not self._floor_frozen or recovery_band)
        )
        if variance_needed:
            variance_ctx.update_state()
            if hasattr(variance_ctx, "current_moving_variance"):
                moving_variance = variance_ctx.current_moving_variance
            else:
                moving_variance = variance_ctx.get_motion_metric()
        self._last_moving_variance = moving_variance

        ready = self._l1.is_ready()
        band_low = self.BAND_ALPHA * thr
        motion = False
        if ready:
            if l1v > thr:
                motion = True
            elif (self._recovery_vote_configured
                  and self._recovery_vote_enabled and self._variance_floor is not None
                  and l1v > band_low
                  and moving_variance > self.RECOVERY_VOTE_RATIO * self._variance_floor):
                motion = True
        self._state = MotionState.MOTION if motion else MotionState.IDLE

        return {
            'motion_metric': l1v,
            'l1_delta': l1v,
            'moving_variance': moving_variance,
            'variance_floor': self._variance_floor if self._variance_floor is not None else 0.0,
            'recovery_vote_enabled': self._recovery_vote_enabled,
            'threshold': thr,
            'state': self._state,
        }

    # -- delegated primary contract ----------------------------------------

    def get_state(self):
        """Get the current fused motion state."""
        return self._state

    def get_motion_metric(self):
        """Primary metric is L1-Delta's (used by startup calibration)."""
        return self._l1.get_motion_metric()

    def get_threshold(self):
        """Primary threshold is L1-Delta's."""
        return self._l1.threshold if hasattr(self._l1, "threshold") else self._l1.get_threshold()

    def set_threshold(self, threshold):
        """Set the primary (L1-Delta) threshold."""
        return self._l1.set_threshold(threshold)

    def set_adaptive_threshold(self, threshold):
        """Set the startup-calibrated primary threshold.

        The shared startup calibrator now owns the validated-quiet selection and
        passes any frozen floor snapshot via ``apply_startup_floor`` before this call.
        If startup did not yield enough quiet variance samples the floor stays unset
        and the detector runs as L1-Delta alone (safe default).
        """
        self._l1.set_adaptive_threshold(threshold)
        self._floor_frozen = self._recovery_vote_configured

    def get_last_moving_variance(self):
        """Expose the latest variance metric to the shared startup calibrator."""
        return self._last_moving_variance

    def apply_startup_floor(self, variance_floor, recovery_vote_enabled, sample_count):
        """Freeze one validated startup floor snapshot supplied by the calibrator."""
        if not self._recovery_vote_configured:
            self._floor_count = 0
            self._variance_floor = None
            self._recovery_vote_enabled = False
            return

        count = max(0, int(sample_count))
        self._floor_count = count
        self._variance_floor = float(variance_floor) if count > 0 else None
        self._recovery_vote_enabled = (
            self._recovery_vote_configured
            and bool(recovery_vote_enabled)
            and count >= self.VARIANCE_FLOOR_MIN
        )

    def is_ready(self):
        """Detection readiness follows the primary detector."""
        return self._l1.is_ready()

    def reset(self):
        """Reset runtime state while preserving frozen calibration when present."""
        preserve_frozen_floor = self._recovery_vote_configured and self._floor_frozen
        preserved_floor = self._variance_floor
        preserved_vote = self._recovery_vote_enabled
        self._l1.reset()
        if self._recovery_vote_configured and self._variance_ctx is not None:
            self._variance_ctx.reset(full=True)
        if preserve_frozen_floor:
            self._variance_floor = preserved_floor
            self._recovery_vote_enabled = preserved_vote
        else:
            self._floor_count = 0
            self._variance_floor = None
            self._recovery_vote_enabled = False
        self._floor_frozen = preserve_frozen_floor
        self._state = MotionState.IDLE
        self._packet_count = 0
        self._last_moving_variance = 0.0

    def get_name(self):
        """Get detector name."""
        return "Classic"

    @property
    def total_packets(self):
        """Total packets processed."""
        return self._packet_count
