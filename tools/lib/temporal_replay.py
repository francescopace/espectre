# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Host orchestration around the production Micro-ESPectre temporal sampler."""

from dataclasses import dataclass

from .bootstrap import setup_paths

setup_paths()

try:
    from temporal_csi_sampler import TemporalCsiSampler
except ImportError:
    from src.temporal_csi_sampler import TemporalCsiSampler


@dataclass(frozen=True)
class TemporalAdmission:
    """One packet admitted by the production sampler and its slot metadata."""

    packet_index: int
    packet: object
    timestamp_us: int
    slot_index: int
    coverage_us: int
    missing_slots_before: int
    reset_required: bool
    context: object = None


class TemporalReplayController:
    """Stateful adapter for tools that receive replay packets one at a time."""

    def __init__(self, target_pps, window_size_ms, fallback_interval_us=None):
        self.target_pps = max(1, int(target_pps))
        self.window_size_ms = max(1, int(window_size_ms))
        self.fallback_interval_us = fallback_interval_us
        self.nominal_interval_us = max(
            1,
            int(round(1_000_000.0 / self.target_pps)),
        )
        self.sampler = TemporalCsiSampler(
            self.target_pps,
            self.window_size_ms,
        )
        self.packet_index = 0
        self._pending = None

    def reset(self):
        self.sampler.reset()
        self.packet_index = 0
        self._pending = None

    def clear_history(self):
        """Drop buffered data and start a new sampler temporal epoch."""
        self.sampler.clear_history()
        self._pending = None

    def clear_window_preserving_phase(self):
        """Drop buffered data while retaining the sampler grid phase."""
        self.sampler.clear_window_preserving_phase()
        self._pending = None

    def _build_admission(self, pending):
        if pending is None:
            return None
        packet_index, packet, timestamp_us, context = pending
        return TemporalAdmission(
            packet_index=packet_index,
            packet=packet,
            timestamp_us=int(timestamp_us),
            slot_index=int(self.sampler.current_slot),
            coverage_us=(
                int(self.sampler.slots_advanced) * self.nominal_interval_us
            ),
            missing_slots_before=int(self.sampler.missing_slots_before),
            reset_required=bool(self.sampler.reset_required),
            context=context,
        )

    def admit(self, packet, context=None):
        packet_index = self.packet_index
        self.packet_index += 1
        timestamp_us = packet_timestamp_us(
            packet,
            fallback_index=packet_index,
            fallback_interval_us=self.fallback_interval_us,
        )
        emitted = self.sampler.admit(timestamp_us)
        admission = self._build_admission(self._pending) if emitted else None
        if emitted:
            self._pending = None
        if self.sampler.selected_current:
            self._pending = (packet_index, packet, int(timestamp_us), context)
        return admission

    def finish(self):
        """Emit the final selected slot after a finite packet sequence."""
        if not self.sampler.flush():
            return None
        admission = self._build_admission(self._pending)
        self._pending = None
        return admission


def packet_timestamp_us(packet, fallback_index=None, fallback_interval_us=None):
    """Return a packet's device timestamp, with an explicit replay fallback."""
    getter = packet.get if hasattr(packet, "get") else None
    for field in ("wifi_rx_ts_us", "device_ticks_us"):
        value = getter(field) if getter is not None else getattr(packet, field, None)
        if value is not None:
            return int(value)
    if fallback_index is None or fallback_interval_us is None:
        return None
    return int(fallback_index) * max(1, int(fallback_interval_us))


def target_pps_from_interval(interval_us):
    """Resolve the legacy-capture target when provenance has no target PPS."""
    return max(1, int(round(1_000_000.0 / max(1, int(interval_us)))))


def target_pps_for_packets(packets, fallback_interval_us):
    """Prefer recorded target provenance and fall back for legacy captures."""
    if packets:
        first = packets[0]
        getter = first.get if hasattr(first, "get") else None
        value = (
            getter("csi_target_pps")
            if getter is not None
            else getattr(first, "csi_target_pps", None)
        )
        if value is not None and int(value) > 0:
            return int(value)
    return target_pps_from_interval(fallback_interval_us)


def iter_temporal_admissions(
    packets,
    *,
    target_pps,
    window_size_ms,
    fallback_interval_us=None,
):
    """Yield packets admitted by the single production temporal sampler."""
    controller = TemporalReplayController(
        target_pps,
        window_size_ms,
        fallback_interval_us,
    )
    for packet in packets:
        admission = controller.admit(packet)
        if admission is not None:
            yield admission
    admission = controller.finish()
    if admission is not None:
        yield admission


def apply_temporal_admission(detector, admission):
    """Apply reset and missing-slot semantics before consuming a packet."""
    if admission.reset_required:
        detector.reset()
    if (
        admission.missing_slots_before
        and hasattr(detector, "advance_missing_slots")
    ):
        detector.advance_missing_slots(admission.missing_slots_before)
