# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - Adjacent subcarrier aggregation helpers

Research-only helpers that replace the production amplitude-buffer fill with an
adjacent-bin averaging variant for the duration of one host-side experiment.

Author: Francesco Pace <francesco.pace@gmail.com>
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Callable, Dict, Iterator, Sequence, Tuple

from .bootstrap import setup_paths

setup_paths()

from tools.lib.segmentation import SegmentationContext  # noqa: E402

# Guard-band and DC limits of the centered HT20 convention. Clamping matters:
# bins 3 and 61 are guard nulls, so an unclamped 3-wide window around the edge
# tones would average a hard zero into two of the twelve profile entries.
GUARD_LOW = 4
GUARD_HIGH = 60
DC_BIN = 32

BASELINE_FILL = SegmentationContext._fill_amplitude_buffer


def aggregation_groups(
    band: Sequence[int],
    width: int,
) -> Tuple[Tuple[int, ...], ...]:
    """Bins averaged for each selected subcarrier at one group width."""
    groups = []
    for subcarrier in band:
        half = (width - 1) // 2
        low = subcarrier - half
        high = subcarrier + (width - 1 - half)
        if low < GUARD_LOW:
            low, high = GUARD_LOW, GUARD_LOW + width - 1
        if high > GUARD_HIGH:
            low, high = GUARD_HIGH - width + 1, GUARD_HIGH
        groups.append(tuple(bin_index for bin_index in range(low, high + 1) if bin_index != DC_BIN))
    return tuple(groups)


def make_aggregating_fill(width: int, coherent: bool = False) -> Callable[..., int]:
    """Build a drop-in `_fill_amplitude_buffer` that averages adjacent bins."""
    cache: Dict[Tuple[int, ...], Tuple[Tuple[int, ...], ...]] = {}

    def fill(csi_data, selected_subcarriers, out_buffer) -> int:
        if selected_subcarriers is None:
            return BASELINE_FILL(csi_data, selected_subcarriers, out_buffer)
        if not coherent:
            return SegmentationContext._fill_adjacent_aggregated_amplitude_buffer(
                csi_data,
                selected_subcarriers,
                out_buffer,
                width,
            )
        key = tuple(selected_subcarriers)
        groups = cache.get(key)
        if groups is None:
            groups = aggregation_groups(key, width)
            cache[key] = groups

        written = 0
        max_slots = len(out_buffer)
        csi_len = len(csi_data)
        for bins in groups:
            if written >= max_slots:
                break
            acc_real = acc_imag = acc_magnitude = 0.0
            count = 0
            for sc_idx in bins:
                index = sc_idx * 2
                if index + 1 >= csi_len:
                    continue
                imag = csi_data[index]
                real = csi_data[index + 1]
                imag = float(imag if imag < 128 else imag - 256)
                real = float(real if real < 128 else real - 256)
                if coherent:
                    acc_real += real
                    acc_imag += imag
                else:
                    acc_magnitude += math.sqrt(real * real + imag * imag)
                count += 1
            if count == 0:
                continue
            if coherent:
                out_buffer[written] = math.sqrt(
                    acc_real * acc_real + acc_imag * acc_imag
                ) / count
            else:
                out_buffer[written] = acc_magnitude / count
            written += 1
        return written

    return fill


@contextmanager
def aggregated_amplitudes(
    width: int | None,
    coherent: bool = False,
) -> Iterator[None]:
    """Replace the production amplitude fill for the duration of the block."""
    try:
        if width is not None:
            SegmentationContext._fill_amplitude_buffer = staticmethod(
                make_aggregating_fill(width, coherent)
            )
        yield
    finally:
        SegmentationContext._fill_amplitude_buffer = staticmethod(BASELINE_FILL)
