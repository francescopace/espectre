# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Deterministic occupancy thinning for reserved replay gates and sweeps."""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

from .bootstrap import setup_paths

setup_paths()

from config import CSI_TARGET_PPS, SEGMENTATION_WINDOW_SIZE_MS  # noqa: E402
from tools.lib.temporal_csi_sampler import (  # noqa: E402
    MINIMUM_COVERAGE_DENOMINATOR,
    MINIMUM_COVERAGE_NUMERATOR,
    MICROSECONDS_PER_SECOND,
    UINT32_MODULUS,
    temporal_window_slots,
)
from tools.lib.temporal_replay import (  # noqa: E402
    iter_temporal_admissions,
)

OCCUPANCY_GATE_PERCENT = (
    100 * MINIMUM_COVERAGE_NUMERATOR // MINIMUM_COVERAGE_DENOMINATOR
)
OCCUPANCY_THIN_SEED = 20260807
OCCUPANCY_GATE_TRANSFORM = "occupancy_gate_thin_v1"
TARGET_PPS = int(CSI_TARGET_PPS)


def copy_packet(packet: Any, *, target_pps: int) -> dict[str, Any]:
    """Return a mapping copy with the sensing target stamped on the packet."""
    copied = dict(packet)
    copied["csi_target_pps"] = int(target_pps)
    return copied


def admit_packets(
    packets: Sequence[dict[str, Any]],
    *,
    target_pps: int = TARGET_PPS,
) -> tuple[dict[str, Any], ...]:
    """Admit CSI onto the production temporal grid before thinning."""
    admitted: list[dict[str, Any]] = []
    for admission in iter_temporal_admissions(
        packets,
        target_pps=target_pps,
        window_size_ms=SEGMENTATION_WINDOW_SIZE_MS,
    ):
        admitted.append(copy_packet(admission.packet, target_pps=target_pps))
    return tuple(admitted)


def thin_packets(
    packets: Sequence[dict[str, Any]],
    *,
    keep_ratio: float,
) -> tuple[dict[str, Any], ...]:
    """Keep a uniform stride of admitted packets. Timestamps stay original."""
    if keep_ratio >= 1.0 or len(packets) <= 1:
        return tuple(packets)
    stride = 1.0 / float(keep_ratio)
    kept: list[dict[str, Any]] = []
    cursor = 0.0
    while True:
        index = int(round(cursor))
        if index >= len(packets):
            break
        kept.append(packets[index])
        cursor += stride
    return tuple(kept)


def capture_seed(dataset_id: str, occupancy_percent: int, *, offset: int = 0) -> int:
    """Return a stable identity seed for one thinned capture."""
    payload = f"{OCCUPANCY_THIN_SEED}:{occupancy_percent}:{dataset_id}:{offset}".encode(
        "utf-8"
    )
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little")


def mean_window_occupancy(
    packets: Sequence[dict[str, Any]],
    *,
    target_pps: int = TARGET_PPS,
) -> float:
    """Mean occupancy at every grid slot after the initial window fills.

    Empty slots, including stream outages, have the same weight as occupied
    ones. The interval ends at the last admitted slot. After a gap resets the
    sampler's phase, place its new epoch at the nearest capture-wide slot.
    """
    if not packets:
        return 0.0
    window_slots = temporal_window_slots(target_pps, SEGMENTATION_WINDOW_SIZE_MS)
    occupied_slots: list[int] = []
    last_timestamp = None
    elapsed_us = 0
    epoch_slot = 0
    for admission in iter_temporal_admissions(
        packets,
        target_pps=target_pps,
        window_size_ms=SEGMENTATION_WINDOW_SIZE_MS,
        fallback_interval_us=max(1, int(round(MICROSECONDS_PER_SECOND / target_pps))),
    ):
        if last_timestamp is not None:
            elapsed_us += (admission.timestamp_us - last_timestamp) % UINT32_MODULUS
        last_timestamp = admission.timestamp_us
        if admission.reset_required:
            epoch_slot = (
                elapsed_us * target_pps + MICROSECONDS_PER_SECOND // 2
            ) // MICROSECONDS_PER_SECOND
        occupied_slots.append(epoch_slot + admission.slot_index)

    first_window_end = window_slots - 1
    if not occupied_slots or occupied_slots[-1] < first_window_end:
        return 0.0
    last_window_end = occupied_slots[-1]
    # Each admitted sample contributes to at most one window's worth of ends.
    # Summing those overlaps avoids allocating or iterating through long gaps.
    occupied_total = sum(
        max(0, min(last_window_end, slot + window_slots - 1)
            - max(first_window_end, slot) + 1)
        for slot in occupied_slots
    )
    window_count = last_window_end - first_window_end + 1
    return occupied_total / float(window_count * window_slots)


def thin_to_occupancy(
    packets: Sequence[dict[str, Any]],
    *,
    occupancy_percent: int = OCCUPANCY_GATE_PERCENT,
    dataset_id: str,
    offset: int = 0,
    target_pps: int = TARGET_PPS,
) -> tuple[tuple[dict[str, Any], ...], float, int]:
    """Admit, then thin toward ``occupancy_percent`` of the production grid.

    Returns the thinned admitted packets, the keep ratio, and the identity seed.
    """
    admitted = admit_packets(packets, target_pps=target_pps)
    admitted_occupancy = mean_window_occupancy(admitted, target_pps=target_pps)
    target_occupancy = occupancy_percent / 100.0
    keep_ratio = (
        1.0
        if admitted_occupancy <= 0.0
        else min(1.0, target_occupancy / admitted_occupancy)
    )
    seed = capture_seed(dataset_id, occupancy_percent, offset=offset)
    thinned = thin_packets(admitted, keep_ratio=keep_ratio)
    return thinned, float(keep_ratio), int(seed)
