"""
ESPectre - Host-Side Candidate Features

Evaluation-only feature candidates for the ML feature-set experiments.

These features are deliberately **not** part of the production surface in
`csi_features.py`, which stays exactly the exported set; see
`docs/adr/2026-07-27-reduce-the-feature-surface-to-the-production-set.md`. A
candidate lives here until a promotion decision is taken, and only then does it
earn a calc function in both languages, an `MLFeatureId`, and a `CPP_FEATURE_IDS`
entry. Until that happens the trainer refuses to export a model that uses one.

Every candidate must obey the same membership rule as the production set: it is
a ratio, a correlation, or a crossing rate, so the unrecorded per-packet int8
scaling factor cancels. `test/python/test_candidate_features.py` asserts it.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .bootstrap import setup_paths

setup_paths()

try:
    from csi_features import L1_DELTA_LAG
except ImportError:  # pragma: no cover
    from src.csi_features import L1_DELTA_LAG

# HT20 data and pilot bins in the centered convention the loaders normalize to:
# DC sits at bin 32 and the guard bands at 0..3 and 61..63. Coherence reads the
# whole live band rather than the twelve classic tones, because the delay
# estimate below needs frequency span to be well conditioned.
HT20_LIVE_BINS: Tuple[int, ...] = tuple(range(4, 32)) + tuple(range(33, 61))

# Index pairs inside HT20_LIVE_BINS that are genuinely adjacent in frequency.
# The DC gap splits the band, so the pair that straddles it is excluded from the
# delay estimator while both halves still contribute to the coherent sum.
_ADJACENT_PAIRS: Tuple[int, ...] = tuple(
    i for i in range(len(HT20_LIVE_BINS) - 1)
    if HT20_LIVE_BINS[i + 1] - HT20_LIVE_BINS[i] == 1
)

_LIVE_LEFT = [i + 1 for i in _ADJACENT_PAIRS]
_LIVE_RIGHT = list(_ADJACENT_PAIRS)
_BIN_INDEX = np.asarray(HT20_LIVE_BINS, dtype=np.float64)

COHERENCE_GAP_LOW_THRESHOLD = 0.02

CHANNEL_COHERENCE_FEATURES = (
    'chan_coh_lag_ratio',
    'chan_coh_mean',
    'chan_coh_gap',
    'chan_coh_gap_low_frac',
    'chan_coh_gap_q20',
)
CANDIDATE_FEATURES: Tuple[str, ...] = CHANNEL_COHERENCE_FEATURES


def complex_profile(csi_data, out=None) -> np.ndarray:
    """Return the complex channel over the live band.

    Payload layout matches `SegmentationContext._fill_amplitude_buffer`: each
    subcarrier is an ``(imag, real)`` int8 pair.
    """
    raw = np.asarray(csi_data, dtype=np.int8)
    if raw.ndim != 1 or raw.size < 2 * (HT20_LIVE_BINS[-1] + 1):
        return np.zeros(len(HT20_LIVE_BINS), dtype=np.complex128)
    imag = raw[0::2][list(HT20_LIVE_BINS)].astype(np.float64)
    real = raw[1::2][list(HT20_LIVE_BINS)].astype(np.float64)
    if out is None:
        return real + 1j * imag
    out[:] = real + 1j * imag
    return out


def delay_compensated_coherence(current: np.ndarray, reference: np.ndarray) -> float:
    """Coherence between two complex profiles, with the packet delay removed.

    ``|sum_k c_k e^{-j k d}| / sum_k |c_k|`` for ``c_k = H_k current * conj(H_k
    reference)``, with ``d`` estimated from the products of adjacent bins.

    The magnitude of the sum drops any common phase, so a per-packet carrier
    offset cancels; ``d`` absorbs the sampling-time offset that would otherwise
    read as decorrelation. Numerator and denominator are both quadratic in the
    channel, so the unrecorded int8 scaling factor cancels and the result stays
    in ``[0, 1]``: near 1 when the channel is unchanged, lower as movement
    decorrelates it.
    """
    cross = current * np.conj(reference)
    magnitude = np.abs(cross)
    total = magnitude.sum()
    if total <= 0.0:
        return 0.0
    ramp = np.angle(np.sum(cross[_LIVE_LEFT] * np.conj(cross[_LIVE_RIGHT])))
    aligned = cross * np.exp(-1j * ramp * _BIN_INDEX)
    return float(abs(aligned.sum()) / total)


class ChannelCoherenceTracker:
    """Running delay-compensated coherence at the profile lag and at lag 1.

    Mirrors the shape of `L1DeltaTracker`: the same window of packets, the same
    pair of lags, and a ratio between them so no absolute magnitude survives.
    """

    def __init__(self, window_size: int = 90, lag: int = L1_DELTA_LAG):
        self.window_size = max(2, int(window_size))
        self.lag = max(1, int(lag))
        width = len(HT20_LIVE_BINS)
        self._ring = [np.zeros(width, dtype=np.complex128) for _ in range(self.lag)]
        self._ring_filled = [False] * self.lag
        self._index = 0
        self._previous = np.zeros(width, dtype=np.complex128)
        self._has_previous = False
        self._lag_sum = 0.0
        self._lag_count = 0
        self._adjacent_sum = 0.0
        self._adjacent_count = 0
        self._lag_ring = [0.0] * self.window_size
        self._adjacent_ring = [0.0] * self.window_size
        self._gap_ring = [0.0] * self.window_size
        self._lag_slot = 0
        self._adjacent_slot = 0
        self._gap_slot = 0
        self._gap_count = 0

    def _push(self, value, ring, slot, count, total):
        if count < self.window_size:
            count += 1
        else:
            total -= ring[slot]
        ring[slot] = value
        total += value
        slot += 1
        if slot >= self.window_size:
            slot = 0
        return slot, count, total

    def process_packet(self, csi_data) -> None:
        """Consume one raw CSI payload."""
        profile = complex_profile(csi_data)
        slot = self._index
        lag_value = None
        adjacent_value = None
        if self._ring_filled[slot]:
            lag_value = delay_compensated_coherence(profile, self._ring[slot])
            self._lag_slot, self._lag_count, self._lag_sum = self._push(
                lag_value, self._lag_ring, self._lag_slot, self._lag_count, self._lag_sum
            )
        if self._has_previous:
            adjacent_value = delay_compensated_coherence(profile, self._previous)
            (self._adjacent_slot, self._adjacent_count,
             self._adjacent_sum) = self._push(
                adjacent_value, self._adjacent_ring, self._adjacent_slot,
                self._adjacent_count, self._adjacent_sum
            )
        if lag_value is not None and adjacent_value is not None:
            self._gap_slot, self._gap_count, _ = self._push(
                adjacent_value - lag_value,
                self._gap_ring,
                self._gap_slot,
                self._gap_count,
                0.0,
            )
        self._previous = profile.copy()
        self._has_previous = True
        self._ring[slot] = profile
        self._ring_filled[slot] = True
        self._index += 1
        if self._index >= self.lag:
            self._index = 0

    def mean_coherence(self) -> float:
        """Mean lag-``lag`` coherence over the current window."""
        if self._lag_count == 0:
            return 1.0
        return self._lag_sum / self._lag_count

    def coherence_lag_ratio(self) -> float:
        """Lag-``lag`` coherence divided by lag-1 coherence.

        Both terms carry the same link and hardware conditions, so the ratio
        reports how much faster the channel decorrelates over the profile lag
        than between neighbouring packets. It sits near 1 on a still channel and
        falls as motion decorrelates the longer lag first.
        """
        if self._lag_count == 0 or self._adjacent_count == 0:
            return 1.0
        adjacent_mean = self._adjacent_sum / self._adjacent_count
        if adjacent_mean <= 0.0:
            return 1.0
        return (self._lag_sum / self._lag_count) / adjacent_mean

    def coherence_gap(self) -> float:
        """Adjacent-packet coherence minus lag-``lag`` coherence.

        Still channels keep both terms near one, so the gap stays near zero.
        Motion lowers the longer-lag coherence first, pushing the gap positive.
        """
        if self._lag_count == 0 or self._adjacent_count == 0:
            return 0.0
        return (self._adjacent_sum / self._adjacent_count) - (
            self._lag_sum / self._lag_count
        )

    def _gap_values(self) -> np.ndarray:
        if self._gap_count == 0:
            return np.zeros(0, dtype=np.float64)
        return np.asarray(self._gap_ring[:self._gap_count], dtype=np.float64)

    def coherence_gap_low_frac(self, threshold: float = COHERENCE_GAP_LOW_THRESHOLD) -> float:
        """Fraction of window entries whose coherence gap exceeds ``threshold``.

        Isolated noisy packets leave this close to zero, while sustained motion
        keeps the long-lag coherence consistently below the adjacent coherence.
        """
        values = self._gap_values()
        if values.size == 0:
            return 0.0
        return float(np.mean(values > float(threshold)))

    def coherence_gap_q20(self) -> float:
        """20th percentile of the coherence-gap window.

        This stays near zero unless a large share of the window shows a positive
        adjacent-minus-lag margin, making it more robust to short noisy bursts.
        """
        values = self._gap_values()
        if values.size == 0:
            return 0.0
        return float(np.quantile(values, 0.20))

    def reset(self) -> None:
        for i in range(self.lag):
            self._ring_filled[i] = False
        self._index = 0
        self._has_previous = False
        self._lag_sum = 0.0
        self._lag_count = 0
        self._adjacent_sum = 0.0
        self._adjacent_count = 0
        self._lag_slot = 0
        self._adjacent_slot = 0
        self._gap_slot = 0
        self._gap_count = 0


def needs_channel_coherence(feature_names: Iterable[str]) -> bool:
    """Return whether any requested feature needs the coherence tracker."""
    return any(name in CHANNEL_COHERENCE_FEATURES for name in feature_names)


def split_feature_names(
    feature_names: Iterable[str],
) -> Tuple[List[str], List[str]]:
    """Split a requested set into production names and candidate names."""
    names = list(feature_names)
    candidates = [name for name in names if name in CANDIDATE_FEATURES]
    production = [name for name in names if name not in CANDIDATE_FEATURES]
    return production, candidates


def candidate_values(
    feature_names: Iterable[str],
    coherence_tracker: ChannelCoherenceTracker = None,
) -> Dict[str, float]:
    """Evaluate the requested candidates from their preprocessed trackers."""
    values: Dict[str, float] = {}
    for name in feature_names:
        if name not in CANDIDATE_FEATURES:
            continue
        if coherence_tracker is None:
            raise ValueError(
                f"{name} needs the channel coherence tracker; pass the "
                f"explicitly preprocessed stream"
            )
        if name == 'chan_coh_lag_ratio':
            values[name] = coherence_tracker.coherence_lag_ratio()
        elif name == 'chan_coh_mean':
            values[name] = coherence_tracker.mean_coherence()
        elif name == 'chan_coh_gap':
            values[name] = coherence_tracker.coherence_gap()
        elif name == 'chan_coh_gap_low_frac':
            values[name] = coherence_tracker.coherence_gap_low_frac()
        elif name == 'chan_coh_gap_q20':
            values[name] = coherence_tracker.coherence_gap_q20()
    return values


def assemble_feature_vector(
    feature_names: Sequence[str],
    production_names: Sequence[str],
    production_values: Sequence[float],
    candidates: Dict[str, float],
) -> List[float]:
    """Reorder production and candidate values into the requested order."""
    position = {name: index for index, name in enumerate(production_names)}
    vector = []
    for name in feature_names:
        if name in position:
            vector.append(production_values[position[name]])
        else:
            vector.append(candidates[name])
    return vector
