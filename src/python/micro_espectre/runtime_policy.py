from collections.abc import Mapping as MappingABC

"""
Micro-ESPectre - Runtime Policy

Keeps detector evaluation cadence and motion hit filtering aligned with the
ESPHome/C++ runtime behavior.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

try:
    from src.detector_interface import MotionState
except ImportError:
    from detector_interface import MotionState

from config import (
    EVALUATION_INTERVAL_US,
    L1_DELTA_LAG_US,
    L1_DELTA_LAG_MAX,
    MIN_PLAUSIBLE_PACKET_INTERVAL_US,
    RATE_ADAPTATION_DEAD_BAND,
    SEG_WINDOW_MAX,
    SEG_WINDOW_MIN,
    SEG_WINDOW_SIZE,
    SEG_WINDOW_US,
    TURB_AUTOCORR_LAG_US,
)


DEFAULT_GAP_RESET_RATIO = 4.0
# Medians are refreshed on this stride instead of on every packet.
RATE_ESTIMATOR_REFRESH_STRIDE = 16
DEFAULT_GAP_RESET_SEQ_THRESHOLD = 3
DEFAULT_GAP_RESET_MIN_US = 250_000
_UINT32_MODULUS = 1 << 32

# Rolling sample count used to estimate the effective packet cadence. One
# second of packets at the nominal rate is enough to be stable without making
# the estimate slow to follow a genuine rate change.
DEFAULT_RATE_ESTIMATOR_SAMPLES = 64
# Samples required before a rate-derived rule is allowed to fire. Until then
# the estimator has not seen enough of the stream to tell a slower cadence
# from packet loss, and guessing either way is worse than not acting.
DEFAULT_RATE_ESTIMATOR_WARMUP = 16


def nominal_packet_interval_us(window_packets):
    """Return the nominal packet interval implied by one window per second."""
    packets = max(1, int(window_packets))
    return max(1, int(round(1_000_000.0 / float(packets))))


def _median(values):
    """Return the median of a non-empty sequence without importing statistics."""
    ordered = sorted(values)
    count = len(ordered)
    middle = count // 2
    if count % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


class PacketRateEstimator:
    """Track the effective packet cadence from observed inter-packet deltas.

    The estimate is a rolling median rather than a mean: a stream with holes
    contains a few very large deltas, and a mean would let those dominate the
    cadence the rest of the pipeline derives its windows from.
    """

    def __init__(
        self,
        nominal_interval_us,
        samples=DEFAULT_RATE_ESTIMATOR_SAMPLES,
        warmup=DEFAULT_RATE_ESTIMATOR_WARMUP,
    ):
        self.nominal_interval_us = max(1, int(nominal_interval_us))
        self.samples = max(1, int(samples))
        self.warmup = max(1, int(warmup))
        self.reset()

    def reset(self):
        """Forget the observed cadence and fall back to the nominal interval."""
        self._deltas = []
        self._seq_steps = []
        self._interval_cache = None
        self._seq_cache = None
        self._since_refresh = 0

    def observe_interval(self, delta_us):
        """Record one inter-packet interval."""
        if delta_us is None:
            return
        delta = int(delta_us)
        if delta <= 0:
            return
        self._deltas.append(delta)
        if len(self._deltas) > self.samples:
            del self._deltas[0]
        # Sorting on every packet would put an O(n log n) step in the hot path
        # for an estimate that cannot move quickly anyway, so the medians are
        # refreshed on a fixed stride and served from cache in between.
        self._since_refresh += 1
        if self._since_refresh >= RATE_ESTIMATOR_REFRESH_STRIDE:
            self._since_refresh = 0
            self._interval_cache = None
            self._seq_cache = None

    def observe_sequence_step(self, seq_step):
        """Record one observed advance of the packet sequence counter."""
        if seq_step is None:
            return
        step = int(seq_step)
        if step <= 0:
            return
        self._seq_steps.append(step)
        if len(self._seq_steps) > self.samples:
            del self._seq_steps[0]

    @property
    def ready(self):
        """True once enough intervals have been seen to trust the estimate."""
        return len(self._deltas) >= self.warmup

    @property
    def interval_us(self):
        """Return the effective packet interval, or the nominal one until ready."""
        if not self.ready:
            return self.nominal_interval_us
        if self._interval_cache is None:
            self._interval_cache = max(1, int(round(_median(self._deltas))))
        return self._interval_cache

    @property
    def sequence_established(self):
        """True once the stream's own sequence step has been observed enough."""
        return len(self._seq_steps) >= self.warmup

    @property
    def sequence_step(self):
        """Return the cadence-normal sequence advance, or 1 until established.

        A stream that natively runs slower than the nominal rate advances its
        sequence counter by more than one per delivered packet. That is the
        stream's own step, not loss, so loss has to be measured against it.
        """
        if not self.sequence_established:
            return 1
        if self._seq_cache is None:
            self._seq_cache = max(1, int(round(_median(self._seq_steps))))
        return self._seq_cache


def derive_detector_timing(interval_us, window_override=None):
    """Resolve a candidate detector timing for replay analysis.

    Deployed runtimes deliberately keep the production lags at their nominal
    packet offsets. Scaling only the numerator of the L1 lag ratio changes the
    feature definition and regresses low-rate recall; scaling both offsets would
    require a Classic refit and an ML retrain. The host replay path keeps this
    helper, mirrored in ``core/detector_timing.h``, so future timing candidates
    can be measured against shipped behavior.

    The lags describe how far the channel has moved over an interval, so they
    have to track that interval. This is decisive at high rates: on the 1000 pps
    diagnostic pair, restoring the turbulence-autocorrelation lag to its 10 ms
    scale takes false positives from 17.8-32.7% down to 0.0%, because at under a
    millisecond of separation consecutive packets are almost perfectly
    correlated and the feature leaves the range its coefficients were fitted
    over.

    The window is different. Its features are estimator averages, so their
    sampling behaviour depends on how many samples they average, not on the time
    those samples span. Holding a one-second span at 25 pps leaves 25 samples,
    the estimates get noisier, startup calibration answers the wider quiet
    distribution by lifting the threshold, and recall collapses to 60% while
    false positives stay low. Holding the sample count instead keeps recall at
    99% at the same cadence.

    ``window_override`` supplies the window in packets instead of deriving it,
    still bounded by the measured floor and ceiling. Replay analyses use it to
    isolate lag behavior: a sweep over the 22 normal-link paired recordings at
    their native rate puts the worst-session recall at 91.9% for an 80-sample
    window and 94.2% at 90, both under the 95% production target, against 96.8%
    at 100. Leaving it ``None`` evaluates the fully derived candidate timing.
    """
    interval = max(1, int(interval_us))

    # Near the nominal cadence, adapting buys nothing and costs homogeneity.
    # Rounding a duration into packets flips between neighbouring counts across
    # streams that all run at essentially the nominal rate, so a single
    # coefficient set has to cover slightly different feature definitions.
    # Measured on the training corpus, adapting inside this band costs 3.8
    # points of fitted recall at the same false-positive ceiling.
    nominal = nominal_packet_interval_us(SEG_WINDOW_SIZE)
    if abs(interval - nominal) <= RATE_ADAPTATION_DEAD_BAND * nominal:
        interval = nominal

    def packets_for(duration_us, minimum=1):
        return max(minimum, int(round(float(duration_us) / float(interval))))

    derived_window = (
        packets_for(SEG_WINDOW_US) if window_override is None else int(window_override)
    )
    window_packets = min(max(derived_window, SEG_WINDOW_MIN), SEG_WINDOW_MAX)
    # Both lags must leave a usable series inside the window, and the L1 lag
    # also respects the bound the firmware profile ring is sized for.
    lag_ceiling = min(max(1, window_packets // 2), L1_DELTA_LAG_MAX)
    return {
        "interval_us": interval,
        "window_packets": window_packets,
        "lag": min(packets_for(L1_DELTA_LAG_US), lag_ceiling),
        "autocorr_lag": min(packets_for(TURB_AUTOCORR_LAG_US), lag_ceiling),
        "evaluation_interval": packets_for(EVALUATION_INTERVAL_US),
    }


def equivalent_packet_weight(elapsed_us, nominal_interval_us, fallback_packets=1):
    """Convert elapsed clean time to one packet-equivalent coverage weight."""
    fallback = max(1, int(fallback_packets))
    nominal = max(1, int(nominal_interval_us))
    if elapsed_us is None:
        return fallback
    elapsed = int(elapsed_us)
    if elapsed <= 0:
        return fallback
    return max(1, int(round(float(elapsed) / float(nominal))))


def _packet_field(packet, key):
    """Return one field from a dict-like packet or packet object."""
    if isinstance(packet, MappingABC):
        return _mapping_field(packet, key)
    return getattr(packet, key, None)


def _mapping_field(packet, key):
    """Return one field from a packet already known to be dict-like."""
    value = packet.get(key)
    if value is None and key == "seq_num":
        return packet.get("stream_seq_num")
    return value


def _packet_fields(packet, keys, out):
    """Return several fields from one packet, checking its shape once.

    The abstract base class check does not depend on the key, and it is the
    expensive part: `Mapping.__instancecheck__` walks the ABC registry, while
    the lookup that follows is a plain dict hit. Every packet reads three
    fields, so hoisting the check out of the per-field call removes two thirds
    of that cost from the timing path.
    """
    if isinstance(packet, MappingABC):
        for i in range(len(keys)):
            out[i] = _mapping_field(packet, keys[i])
    else:
        for i in range(len(keys)):
            out[i] = getattr(packet, keys[i], None)
    return out


_TIMING_PACKET_FIELDS = ("seq_num", "device_ticks_us", "wifi_rx_ts_us")


def _unsigned_delta(current, previous, modulus):
    """Unsigned modular delta between two monotonic counters."""
    return (int(current) - int(previous)) % int(modulus)


class PacketTimingTracker:
    """Track packet timing and gap contamination from stream metadata."""

    def __init__(
        self,
        nominal_packet_interval_us,
        *,
        gap_reset_ratio=DEFAULT_GAP_RESET_RATIO,
        sequence_gap_reset=DEFAULT_GAP_RESET_SEQ_THRESHOLD,
        gap_reset_min_us=DEFAULT_GAP_RESET_MIN_US,
        rate_estimator=None,
    ):
        self.nominal_packet_interval_us = max(1, int(nominal_packet_interval_us))
        self.gap_reset_ratio = float(gap_reset_ratio)
        self.sequence_gap_reset = max(1, int(sequence_gap_reset))
        self.gap_reset_min_us = max(1, int(gap_reset_min_us))
        self.rate = (
            PacketRateEstimator(self.nominal_packet_interval_us)
            if rate_estimator is None
            else rate_estimator
        )
        self._field_values = [None] * len(_TIMING_PACKET_FIELDS)
        self.reset()

    def reset(self):
        """Forget prior packet positions, but keep the learned stream cadence.

        A hole says nothing about how fast the stream runs, and discarding the
        cadence here is self-defeating: the next packet would be judged against
        a step of one again, read as loss, and reset the tracker once more.
        """
        self._last_seq_num = None
        self._last_device_ticks_us = None
        self._last_wifi_rx_ts_us = None

    def _gap_threshold_us(self):
        """Return the elapsed-time hole threshold for the observed cadence.

        Scaling with the measured interval is what keeps a genuinely slower
        stream from reading as one continuous hole; the absolute floor keeps a
        fast stream from calling ordinary jitter a gap.
        """
        return max(
            self.gap_reset_min_us,
            int(round(self.rate.interval_us * self.gap_reset_ratio)),
        )

    def observe_packet(self, packet):
        """Return timing metadata for one packet."""
        seq_num, device_ticks_us, wifi_rx_ts_us = _packet_fields(
            packet, _TIMING_PACKET_FIELDS, self._field_values
        )

        # Loss is measured against the cadence the stream has actually shown,
        # not against a hardcoded step of one. A capture that natively delivers
        # every fourth packet advances the counter by four with nothing missing,
        # and treating that as loss contaminates every packet in the stream.
        seq_step = None
        missing_seq = 0
        if seq_num is not None and self._last_seq_num is not None:
            seq_step = (int(seq_num) - int(self._last_seq_num)) & 0xFFFFFFFF
            if not 0 < seq_step < 0x80000000:
                seq_step = None
            else:
                # Record before judging, so the packet that completes the warmup
                # is measured against the step it just helped establish rather
                # than against the placeholder step of one.
                self.rate.observe_sequence_step(seq_step)
                missing_seq = max(0, int(seq_step) - self.rate.sequence_step)

        delta_us = None
        source = "nominal"
        if (
            device_ticks_us is not None
            and self._last_device_ticks_us is not None
            and int(device_ticks_us) >= int(self._last_device_ticks_us)
        ):
            delta_us = int(device_ticks_us) - int(self._last_device_ticks_us)
            source = "device_ticks_us"
        elif (
            wifi_rx_ts_us is not None
            and self._last_wifi_rx_ts_us is not None
        ):
            candidate = _unsigned_delta(
                int(wifi_rx_ts_us),
                int(self._last_wifi_rx_ts_us),
                _UINT32_MODULUS,
            )
            if 0 < candidate < (_UINT32_MODULUS // 2):
                delta_us = int(candidate)
                source = "wifi_rx_ts_us"

        if delta_us is None:
            # Without timestamps, elapsed time is inferred from how many packets
            # the counter says are missing, measured at the observed cadence.
            delta_us = self.rate.interval_us * max(1, missing_seq + 1)
            source = "nominal"

        # As with the sequence step, the interval is recorded before the packet
        # is judged. The median is what makes this safe: an occasional hole
        # contributes one large sample and is rejected, while gating the
        # estimate on contamination would deadlock, because a slower stream
        # reads as contaminated until its own cadence is established and can
        # never establish it while being discarded.
        if source != "nominal":
            self.rate.observe_interval(delta_us)

        contaminated = (
            (
                seq_num is not None
                and self._last_seq_num is not None
                and self.rate.sequence_established
                and missing_seq >= self.sequence_gap_reset
            )
            or (
                delta_us >= self._gap_threshold_us()
                and (
                    self._last_device_ticks_us is not None
                    or self._last_wifi_rx_ts_us is not None
                    or self._last_seq_num is not None
                )
            )
        )

        if seq_num is not None:
            self._last_seq_num = int(seq_num)
        if device_ticks_us is not None:
            self._last_device_ticks_us = int(device_ticks_us)
        if wifi_rx_ts_us is not None:
            self._last_wifi_rx_ts_us = int(wifi_rx_ts_us)

        return {
            "delta_us": int(delta_us),
            "coverage_us": 0 if contaminated else int(delta_us),
            "missing_seq": int(missing_seq),
            "source": source,
            "contaminated": bool(contaminated),
        }


class RuntimeMotionPolicy:
    """Central runtime policy for evaluation cadence and hit filtering."""

    def __init__(
        self,
        evaluation_interval=25,
        motion_on_hits=4,
        motion_off_hits=3,
        evaluation_interval_us=None,
    ):
        self.evaluation_interval = max(1, int(evaluation_interval))
        self.motion_on_hits = max(1, int(motion_on_hits))
        self.motion_off_hits = max(1, int(motion_off_hits))
        self.evaluation_interval_us = (
            None
            if evaluation_interval_us is None
            else max(1, int(evaluation_interval_us))
        )
        self._rate = PacketRateEstimator(
            nominal_packet_interval_us(SEG_WINDOW_SIZE)
        )
        self._last_arrival_us = None
        self.reset()

    def reset(self):
        """Reset cadence counters and effective motion state."""
        self.packets_since_evaluation = 0
        self.elapsed_us_since_evaluation = 0
        self.effective_state = MotionState.IDLE
        self.pending_state = MotionState.IDLE
        self.pending_hits = 0

    def note_arrival(self, timestamp_us):
        """Record one packet from its Wi-Fi RX arrival timestamp.

        Arrival time rather than loop time: the loop clock measures how fast
        packets are processed, which matches arrival on hardware but not on
        replay, and it would make the cadence depend on host scheduling. The
        timestamp is an input, so the cadence is reproducible.

        Falls back to counting packets while the cadence is unknown or
        implausible, which covers warmup and sources that report no timestamp.
        """
        elapsed_us = None
        if timestamp_us:
            if self._last_arrival_us is not None:
                delta = (int(timestamp_us) - self._last_arrival_us) % _UINT32_MODULUS
                # Past half the range the counter went backwards rather than a
                # very long gap having elapsed.
                if 0 < delta < (_UINT32_MODULUS // 2):
                    self._rate.observe_interval(delta)
                    if (
                        self._rate.ready
                        and self._rate.interval_us >= MIN_PLAUSIBLE_PACKET_INTERVAL_US
                    ):
                        elapsed_us = delta
            self._last_arrival_us = int(timestamp_us)
        self.note_packet(elapsed_us=elapsed_us)

    @property
    def packet_interval_us(self):
        """Effective packet interval seen so far, for rate-derived sizing."""
        return self._rate.interval_us

    def note_packet(self, elapsed_us=None):
        """Record that one new CSI packet has been processed."""
        self.packets_since_evaluation += 1
        if elapsed_us is not None:
            self.elapsed_us_since_evaluation += max(0, int(elapsed_us))

    def should_evaluate(self, should_publish=False):
        """Check whether the detector should be evaluated now."""
        if should_publish:
            return True
        if (
            self.evaluation_interval_us is not None
            and self.elapsed_us_since_evaluation > 0
        ):
            return self.elapsed_us_since_evaluation >= self.evaluation_interval_us
        return self.packets_since_evaluation >= self.evaluation_interval

    def after_evaluation(self):
        """Reset the cadence counter after an evaluation."""
        self.packets_since_evaluation = 0
        self.elapsed_us_since_evaluation = 0

    def note_evaluation_tick(self, elapsed_us=None, should_publish=False):
        """Record one packet and return True when an evaluation is due."""
        self.note_packet(elapsed_us=elapsed_us)
        if not self.should_evaluate(should_publish=should_publish):
            return False
        self.after_evaluation()
        return True

    def equivalent_packets_since_evaluation(self, nominal_packet_interval_us):
        """Return elapsed coverage as nominal packet-equivalent weight."""
        return equivalent_packet_weight(
            self.elapsed_us_since_evaluation,
            nominal_packet_interval_us,
            fallback_packets=self.packets_since_evaluation,
        )

    def apply_state(self, detector_state):
        """
        Apply hit filtering to the raw detector state.

        Returns:
            tuple: (effective_state, state_changed)
        """
        previous_state = self.effective_state

        if detector_state == self.effective_state:
            self.pending_state = self.effective_state
            self.pending_hits = 0
            return self.effective_state, False

        if detector_state != self.pending_state:
            self.pending_state = detector_state
            self.pending_hits = 1
        else:
            self.pending_hits += 1

        required_hits = (
            self.motion_on_hits
            if self.pending_state == MotionState.MOTION
            else self.motion_off_hits
        )
        if self.pending_hits >= required_hits:
            self.effective_state = self.pending_state
            self.pending_hits = 0

        return self.effective_state, self.effective_state != previous_state


def make_evaluation_cadence(evaluation_interval=25, evaluation_interval_us=None):
    """Return a runtime policy used only for evaluation-interval cadence."""
    return RuntimeMotionPolicy(
        evaluation_interval=evaluation_interval,
        evaluation_interval_us=evaluation_interval_us,
    )
