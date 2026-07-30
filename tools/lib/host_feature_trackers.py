"""
ESPectre - Host-Side Feature Trackers

Shared NumPy-based HT20 feature primitives and trackers used by the host-side
training and evaluation flows. Production and candidate feature registries build
on top of these helpers.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .bootstrap import setup_paths

setup_paths()

try:
    from csi_features import L1_DELTA_LAG  # pyright: ignore[reportMissingImports]
except ImportError:  # pragma: no cover
    from src.csi_features import L1_DELTA_LAG  # pyright: ignore[reportMissingImports]

# HT20 data and pilot bins in the centered convention the loaders normalize to:
# DC sits at bin 32 and the guard bands at 0..3 and 61..63. Coherence reads the
# whole live band rather than the twelve classic tones, because the delay
# estimate below needs frequency span to be well conditioned.
HT20_LIVE_BINS: Tuple[int, ...] = tuple(range(4, 32)) + tuple(range(33, 61))
HT20_COHERENCE_SUBBANDS: Tuple[Tuple[int, ...], ...] = (
    tuple(range(4, 18)),
    tuple(range(18, 32)),
    tuple(range(33, 47)),
    tuple(range(47, 61)),
)

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
_SUBBAND_PROFILE_INDICES: Tuple[np.ndarray, ...] = tuple(
    np.asarray([HT20_LIVE_BINS.index(sc) for sc in subband], dtype=np.intp)
    for subband in HT20_COHERENCE_SUBBANDS
)
_SUBBAND_BIN_INDICES: Tuple[np.ndarray, ...] = tuple(
    np.asarray(subband, dtype=np.float64)
    for subband in HT20_COHERENCE_SUBBANDS
)

COHERENCE_GAP_LOW_THRESHOLD = 0.02

CHANNEL_COHERENCE_FEATURES = (
    'chan_coh_lag_ratio',
)
PROMOTED_CHANNEL_COHERENCE_FEATURES = (
    'chan_coh_gap',
    'chan_coh_subband_gap_median',
)
SUBBAND_COHERENCE_FEATURES = (
    'chan_coh_subband_gap_median',
)
SPECTRAL_FEATURES = ()
PHASE_FEATURES = (
    'phase_resid_lag_ratio',
    'phase_closure_var_std',
)
CHANNEL_SHAPE_FEATURES = ()
PROMOTED_CHANNEL_SHAPE_FEATURES = (
    'chan_shape_spread',
    'chan_freq_coh_cv',
    'chan_freq_coh_curve_std',
)
COMPOSITE_FEATURES = ()
CANDIDATE_FEATURES: Tuple[str, ...] = (
    CHANNEL_COHERENCE_FEATURES
    + SPECTRAL_FEATURES
    + PHASE_FEATURES
    + CHANNEL_SHAPE_FEATURES
    + COMPOSITE_FEATURES
)


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


def _contiguous_delay_compensated_coherence(
    current: np.ndarray,
    reference: np.ndarray,
    bin_index: np.ndarray,
) -> float:
    """Delay-compensated coherence for one contiguous frequency band."""
    cross = current * np.conj(reference)
    magnitude = np.abs(cross)
    total = magnitude.sum()
    if total <= 0.0:
        return 0.0
    ramp = np.angle(np.sum(cross[1:] * np.conj(cross[:-1])))
    aligned = cross * np.exp(-1j * ramp * bin_index)
    return float(abs(aligned.sum()) / total)


def subband_coherences(current: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return delay-compensated coherence for four contiguous HT20 bands."""
    values = np.empty(len(HT20_COHERENCE_SUBBANDS), dtype=np.float64)
    for i, (indices, bin_index) in enumerate(
        zip(_SUBBAND_PROFILE_INDICES, _SUBBAND_BIN_INDICES)
    ):
        values[i] = _contiguous_delay_compensated_coherence(
            current[indices],
            reference[indices],
            bin_index,
        )
    return values


def turbulence_band_power_ratio(series: Sequence[float]) -> float:
    """Fraction of non-DC turbulence power below 0.1 cycles per sample.

    At the nominal 100 pps capture rate this is the historical 0.5-10 Hz
    movement band, rounded to the FFT bins available in the live window. The
    numerator and denominator scale quadratically together, so the statistic
    is invariant to the magnitude of the input series.
    """
    values = np.asarray(series, dtype=np.float64)
    if values.size < 4:
        return 0.0
    centered = values - float(np.mean(values))
    power = np.abs(np.fft.rfft(centered * np.hanning(values.size))) ** 2
    total = float(power[1:].sum())
    if total <= 0.0:
        return 0.0
    upper_bin = max(1, min(len(power) - 1, int(np.floor(values.size * 0.10))))
    return float(power[1:upper_bin + 1].sum() / total)


def sanitized_phase_profile(profile: np.ndarray) -> np.ndarray:
    """Return adjacent-bin phase residuals with CFO and STO removed.

    Adjacent-bin products cancel the common phase offset from CFO. A sampling
    delay becomes one common rotation across those products, which is removed
    by their circular mean. The remaining unit phasors describe only the
    frequency-selective phase shape, without using amplitude.
    """
    adjacent = profile[_LIVE_LEFT] * np.conj(profile[_LIVE_RIGHT])
    magnitude = np.abs(adjacent)
    residual = np.zeros(len(adjacent), dtype=np.complex128)
    valid = magnitude > 0.0
    residual[valid] = adjacent[valid] / magnitude[valid]
    common = residual[valid].sum()
    if abs(common) > 0.0:
        residual[valid] *= np.conj(common) / abs(common)
    return residual


def phase_profile_distance(current: np.ndarray, reference: np.ndarray) -> float:
    """Mean wrapped displacement between two sanitized phase profiles."""
    valid = (np.abs(current) > 0.0) & (np.abs(reference) > 0.0)
    if not np.any(valid):
        return 0.0
    delta = current[valid] * np.conj(reference[valid])
    return float(np.mean(np.abs(np.angle(delta))) / np.pi)


_PHASE_CLOSURE_TRIPLETS: Tuple[Tuple[int, int, int], ...] = tuple(
    (index - 1, index, index + 1)
    for index in range(1, len(HT20_LIVE_BINS) - 1)
    if (
        HT20_LIVE_BINS[index] - HT20_LIVE_BINS[index - 1] == 1
        and HT20_LIVE_BINS[index + 1] - HT20_LIVE_BINS[index] == 1
    )
)
_PHASE_CLOSURE_LEFT = np.asarray(
    [left for left, _, _ in _PHASE_CLOSURE_TRIPLETS],
    dtype=np.intp,
)
_PHASE_CLOSURE_CENTER = np.asarray(
    [center for _, center, _ in _PHASE_CLOSURE_TRIPLETS],
    dtype=np.intp,
)
_PHASE_CLOSURE_RIGHT = np.asarray(
    [right for _, _, right in _PHASE_CLOSURE_TRIPLETS],
    dtype=np.intp,
)


def local_phase_closure_variance(
    profile: np.ndarray,
    relative_floor: float = 0.02,
) -> float:
    """Circular variance of local phase curvature across frequency.

    ``angle(H[k-1] H[k+1] conj(H[k])^2)`` is the second phase difference, so
    common phase and a linear phase ramp cancel exactly. Triplets containing a
    bin below the packet-relative floor are excluded.
    """
    profile = np.asarray(profile, dtype=np.complex128)
    if profile.size != len(HT20_LIVE_BINS):
        return 0.0
    amplitude = np.abs(profile)
    maximum = float(np.max(amplitude))
    if maximum <= 0.0:
        return 0.0
    threshold = max(0.0, float(relative_floor)) * maximum
    valid = (
        (amplitude[_PHASE_CLOSURE_LEFT] > threshold)
        & (amplitude[_PHASE_CLOSURE_CENTER] > threshold)
        & (amplitude[_PHASE_CLOSURE_RIGHT] > threshold)
    )
    if np.count_nonzero(valid) < 4:
        return 0.0
    closure = (
        profile[_PHASE_CLOSURE_LEFT[valid]]
        * profile[_PHASE_CLOSURE_RIGHT[valid]]
        * np.conj(profile[_PHASE_CLOSURE_CENTER[valid]]) ** 2
    )
    magnitude = np.abs(closure)
    nonzero = magnitude > 0.0
    if np.count_nonzero(nonzero) < 4:
        return 0.0
    unit = closure[nonzero] / magnitude[nonzero]
    return float(1.0 - abs(np.mean(unit)))


def normalized_amplitude_profile(profile: np.ndarray) -> np.ndarray:
    """Return the channel magnitude divided by its packet L2 norm."""
    amplitude = np.abs(profile)
    norm = float(np.linalg.norm(amplitude))
    if norm <= 0.0:
        return np.zeros(len(amplitude), dtype=np.float64)
    return amplitude / norm


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return zero-based ranks, averaging positions occupied by ties."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind='stable')
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def rank_profile_distance(
    current: np.ndarray,
    reference: np.ndarray,
    relative_floor: float = 0.02,
) -> float:
    """Return bounded Spearman distance between two amplitude profiles.

    Bins below ``relative_floor`` times either packet maximum are excluded from
    that comparison. This prevents quantization around null tones from creating
    arbitrary rank turnover. The remaining ranks are invariant to any positive
    packet gain; ``(1 - rho) / 2`` maps correlation to ``[0, 1]``.
    """
    current = np.asarray(current, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if current.shape != reference.shape or current.size == 0:
        return 0.0
    current_max = float(np.max(current))
    reference_max = float(np.max(reference))
    if current_max <= 0.0 or reference_max <= 0.0:
        return 0.0
    floor = max(0.0, float(relative_floor))
    valid = (
        (current > floor * current_max)
        & (reference > floor * reference_max)
    )
    if np.count_nonzero(valid) < 4:
        return 0.0
    current_ranks = _average_ranks(current[valid])
    reference_ranks = _average_ranks(reference[valid])
    current_ranks -= float(np.mean(current_ranks))
    reference_ranks -= float(np.mean(reference_ranks))
    denominator = float(
        np.linalg.norm(current_ranks) * np.linalg.norm(reference_ranks)
    )
    if denominator <= 0.0:
        return 0.0
    rho = float(np.dot(current_ranks, reference_ranks) / denominator)
    return 0.5 * (1.0 - float(np.clip(rho, -1.0, 1.0)))


def motion_participation(energy: np.ndarray) -> float:
    """Normalized participation ratio of motion energy across subcarriers.

    A value near ``1 / K`` means one live subcarrier dominates the change,
    while one means that the change is spread uniformly over the live band.
    """
    values = np.asarray(energy, dtype=np.float64)
    total = float(values.sum())
    squared = float(np.dot(values, values))
    if total <= 0.0 or squared <= 0.0:
        return 0.0
    return total * total / (values.size * squared)


def _frequency_coherence_pairs(offset: int) -> Tuple[Tuple[int, int], ...]:
    return tuple(
        (left, right)
        for left, left_bin in enumerate(HT20_LIVE_BINS)
        for right, right_bin in enumerate(HT20_LIVE_BINS)
        if right_bin - left_bin == offset
        and not (left_bin < 32 < right_bin)
    )


_FREQUENCY_COHERENCE_INDICES = {
    offset: (
        np.asarray(
            [left for left, _ in _frequency_coherence_pairs(offset)],
            dtype=np.intp,
        ),
        np.asarray(
            [right for _, right in _frequency_coherence_pairs(offset)],
            dtype=np.intp,
        ),
    )
    for offset in (2, 4, 12)
}
_FREQUENCY_COHERENCE_LEFT = np.asarray(
    _FREQUENCY_COHERENCE_INDICES[4][0],
)
_FREQUENCY_COHERENCE_RIGHT = np.asarray(
    _FREQUENCY_COHERENCE_INDICES[4][1],
)


def cross_subcarrier_ratio_distance(
    current: np.ndarray,
    reference: np.ndarray,
    relative_floor: float = 0.02,
) -> float:
    """Return robust change in guarded cross-subcarrier log ratios.

    Fixed four-bin pairs avoid corpus-trained subcarrier selection. A pair is
    retained only when both of its bins clear the relative floor in both
    packets. The median of ``abs(delta log-ratio) / (1 + abs(delta
    log-ratio))`` is bounded to ``[0, 1]`` and exactly cancels positive
    per-packet gain.
    """
    current = np.asarray(current, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if current.shape != reference.shape or current.size == 0:
        return 0.0
    current_max = float(np.max(current))
    reference_max = float(np.max(reference))
    if current_max <= 0.0 or reference_max <= 0.0:
        return 0.0
    floor = max(0.0, float(relative_floor))
    threshold_current = floor * current_max
    threshold_reference = floor * reference_max
    left = _FREQUENCY_COHERENCE_LEFT
    right = _FREQUENCY_COHERENCE_RIGHT
    valid = (
        (current[left] > threshold_current)
        & (current[right] > threshold_current)
        & (reference[left] > threshold_reference)
        & (reference[right] > threshold_reference)
    )
    if np.count_nonzero(valid) < 4:
        return 0.0
    current_ratio = np.log(current[left[valid]]) - np.log(current[right[valid]])
    reference_ratio = (
        np.log(reference[left[valid]]) - np.log(reference[right[valid]])
    )
    delta = np.abs(current_ratio - reference_ratio)
    return float(np.median(delta / (1.0 + delta)))


def frequency_coherence(profile: np.ndarray, offset: int = 4) -> float:
    """Normalized within-packet coherence at a fixed subcarrier separation.

    Common gain cancels in the normalized complex correlation. CFO contributes
    one common phase and STO contributes one common rotation for the fixed
    frequency separation; taking the magnitude removes both. The statistic is
    therefore a compact proxy for frequency coherence, and hence multipath
    delay structure, without requiring a stable absolute phase.
    """
    indices = _FREQUENCY_COHERENCE_INDICES.get(int(offset))
    if indices is None:
        return 0.0
    left = profile[indices[0]]
    right = profile[indices[1]]
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0.0:
        return 0.0
    return float(abs(np.vdot(left, right)) / denominator)


class ChannelShapeTracker:
    """Track gain-free amplitude-shape and frequency-coherence dynamics."""

    def __init__(self, window_size: int = 90, lag: int = L1_DELTA_LAG):
        self.window_size = max(2, int(window_size))
        self.lag = max(1, int(lag))
        width = len(HT20_LIVE_BINS)
        self._ring = [np.zeros(width, dtype=np.float64) for _ in range(self.lag)]
        self._ring_filled = [False] * self.lag
        self._index = 0
        self._previous = np.zeros(width, dtype=np.float64)
        self._has_previous = False
        self._lag_distance_ring = [0.0] * self.window_size
        self._adjacent_distance_ring = [0.0] * self.window_size
        self._lag_distance_slot = 0
        self._lag_distance_count = 0
        self._lag_distance_sum = 0.0
        self._adjacent_distance_slot = 0
        self._adjacent_distance_count = 0
        self._adjacent_distance_sum = 0.0
        self._motion_energy = np.zeros(width, dtype=np.float64)
        self._motion_energy_ring = np.zeros(
            (self.window_size, width),
            dtype=np.float64,
        )
        self._motion_energy_slot = 0
        self._motion_energy_count = 0
        self._frequency_coherence_ring = [0.0] * self.window_size
        self._frequency_coherence_slot = 0
        self._frequency_coherence_count = 0
        self._frequency_coherence_sum = 0.0
        self._frequency_coherence_square_sum = 0.0
        self._frequency_curve_ring = [0.0] * self.window_size
        self._frequency_curve_slot = 0
        self._frequency_curve_count = 0
        self._frequency_curve_sum = 0.0
        self._frequency_curve_square_sum = 0.0
        self._rank_lag_ring = [0.0] * self.window_size
        self._rank_lag_slot = 0
        self._rank_lag_count = 0
        self._rank_lag_sum = 0.0
        self._rank_adjacent_ring = [0.0] * self.window_size
        self._rank_adjacent_slot = 0
        self._rank_adjacent_count = 0
        self._rank_adjacent_sum = 0.0
        self._ratio_lag_ring = [0.0] * self.window_size
        self._ratio_lag_slot = 0
        self._ratio_lag_count = 0
        self._ratio_lag_sum = 0.0
        self._ratio_adjacent_ring = [0.0] * self.window_size
        self._ratio_adjacent_slot = 0
        self._ratio_adjacent_count = 0
        self._ratio_adjacent_sum = 0.0

    def _push_scalar(self, value, ring, slot, count, total):
        if count < self.window_size:
            count += 1
        else:
            total -= ring[slot]
        ring[slot] = value
        total += value
        slot = (slot + 1) % self.window_size
        return slot, count, total

    def _push_motion_energy(self, values):
        if self._motion_energy_count < self.window_size:
            self._motion_energy_count += 1
        else:
            self._motion_energy -= self._motion_energy_ring[
                self._motion_energy_slot
            ]
        self._motion_energy_ring[self._motion_energy_slot] = values
        self._motion_energy += values
        self._motion_energy_slot = (
            self._motion_energy_slot + 1
        ) % self.window_size

    def _push_frequency_coherence(self, value):
        if self._frequency_coherence_count < self.window_size:
            self._frequency_coherence_count += 1
        else:
            old = self._frequency_coherence_ring[
                self._frequency_coherence_slot
            ]
            self._frequency_coherence_sum -= old
            self._frequency_coherence_square_sum -= old * old
        self._frequency_coherence_ring[self._frequency_coherence_slot] = value
        self._frequency_coherence_sum += value
        self._frequency_coherence_square_sum += value * value
        self._frequency_coherence_slot = (
            self._frequency_coherence_slot + 1
        ) % self.window_size

    def _push_frequency_curve(self, value):
        if self._frequency_curve_count < self.window_size:
            self._frequency_curve_count += 1
        else:
            old = self._frequency_curve_ring[self._frequency_curve_slot]
            self._frequency_curve_sum -= old
            self._frequency_curve_square_sum -= old * old
        self._frequency_curve_ring[self._frequency_curve_slot] = value
        self._frequency_curve_sum += value
        self._frequency_curve_square_sum += value * value
        self._frequency_curve_slot = (
            self._frequency_curve_slot + 1
        ) % self.window_size

    def process_packet(self, csi_data) -> None:
        complex_values = complex_profile(csi_data)
        profile = normalized_amplitude_profile(complex_values)
        slot = self._index
        if self._ring_filled[slot]:
            delta = profile - self._ring[slot]
            distance = float(np.linalg.norm(delta))
            (
                self._lag_distance_slot,
                self._lag_distance_count,
                self._lag_distance_sum,
            ) = self._push_scalar(
                distance,
                self._lag_distance_ring,
                self._lag_distance_slot,
                self._lag_distance_count,
                self._lag_distance_sum,
            )
            self._push_motion_energy(delta * delta)
            (
                self._rank_lag_slot,
                self._rank_lag_count,
                self._rank_lag_sum,
            ) = self._push_scalar(
                rank_profile_distance(profile, self._ring[slot]),
                self._rank_lag_ring,
                self._rank_lag_slot,
                self._rank_lag_count,
                self._rank_lag_sum,
            )
            (
                self._ratio_lag_slot,
                self._ratio_lag_count,
                self._ratio_lag_sum,
            ) = self._push_scalar(
                cross_subcarrier_ratio_distance(profile, self._ring[slot]),
                self._ratio_lag_ring,
                self._ratio_lag_slot,
                self._ratio_lag_count,
                self._ratio_lag_sum,
            )
        if self._has_previous:
            distance = float(np.linalg.norm(profile - self._previous))
            (
                self._adjacent_distance_slot,
                self._adjacent_distance_count,
                self._adjacent_distance_sum,
            ) = self._push_scalar(
                distance,
                self._adjacent_distance_ring,
                self._adjacent_distance_slot,
                self._adjacent_distance_count,
                self._adjacent_distance_sum,
            )
            (
                self._rank_adjacent_slot,
                self._rank_adjacent_count,
                self._rank_adjacent_sum,
            ) = self._push_scalar(
                rank_profile_distance(profile, self._previous),
                self._rank_adjacent_ring,
                self._rank_adjacent_slot,
                self._rank_adjacent_count,
                self._rank_adjacent_sum,
            )
            (
                self._ratio_adjacent_slot,
                self._ratio_adjacent_count,
                self._ratio_adjacent_sum,
            ) = self._push_scalar(
                cross_subcarrier_ratio_distance(profile, self._previous),
                self._ratio_adjacent_ring,
                self._ratio_adjacent_slot,
                self._ratio_adjacent_count,
                self._ratio_adjacent_sum,
            )
        self._push_frequency_coherence(frequency_coherence(complex_values))
        short_coherence = frequency_coherence(complex_values, offset=2)
        long_coherence = frequency_coherence(complex_values, offset=12)
        coherence_sum = short_coherence + long_coherence
        curve_contrast = (
            (short_coherence - long_coherence) / coherence_sum
            if coherence_sum > 0.0
            else 0.0
        )
        self._push_frequency_curve(curve_contrast)
        self._previous = profile.copy()
        self._has_previous = True
        self._ring[slot] = profile
        self._ring_filled[slot] = True
        self._index = (self._index + 1) % self.lag

    def shape_lag_ratio(self) -> float:
        """Lagged normalized-shape displacement over adjacent displacement."""
        if (
            self._lag_distance_count == 0
            or self._adjacent_distance_count == 0
        ):
            return 1.0
        adjacent_mean = (
            self._adjacent_distance_sum / self._adjacent_distance_count
        )
        if adjacent_mean <= 0.0:
            return 1.0
        return (
            self._lag_distance_sum / self._lag_distance_count
        ) / adjacent_mean

    def shape_spread(self) -> float:
        """Participation ratio of lagged motion energy over the live band."""
        return motion_participation(self._motion_energy)

    def frequency_coherence_cv(self) -> float:
        """Temporal CV of the gain- and offset-free frequency coherence."""
        if self._frequency_coherence_count == 0:
            return 0.0
        count = self._frequency_coherence_count
        mean = self._frequency_coherence_sum / count
        variance = max(
            0.0,
            self._frequency_coherence_square_sum / count - mean * mean,
        )
        if mean <= 0.0:
            return 0.0
        return float(np.sqrt(variance) / mean)

    def frequency_coherence_curve_std(self) -> float:
        """Temporal standard deviation of short-versus-long coherence."""
        if self._frequency_curve_count == 0:
            return 0.0
        count = self._frequency_curve_count
        mean = self._frequency_curve_sum / count
        variance = max(
            0.0,
            self._frequency_curve_square_sum / count - mean * mean,
        )
        return float(np.sqrt(variance))

    def rank_gap(self) -> float:
        """Mean lagged rank distance minus adjacent-packet rank distance."""
        if self._rank_lag_count == 0 or self._rank_adjacent_count == 0:
            return 0.0
        return (
            self._rank_lag_sum / self._rank_lag_count
        ) - (
            self._rank_adjacent_sum / self._rank_adjacent_count
        )

    def ratio_gap(self) -> float:
        """Mean lagged ratio distance minus adjacent-packet ratio distance."""
        if self._ratio_lag_count == 0 or self._ratio_adjacent_count == 0:
            return 0.0
        return (
            self._ratio_lag_sum / self._ratio_lag_count
        ) - (
            self._ratio_adjacent_sum / self._ratio_adjacent_count
        )

    def reset(self) -> None:
        for i in range(self.lag):
            self._ring_filled[i] = False
        self._index = 0
        self._has_previous = False
        self._lag_distance_slot = 0
        self._lag_distance_count = 0
        self._lag_distance_sum = 0.0
        self._adjacent_distance_slot = 0
        self._adjacent_distance_count = 0
        self._adjacent_distance_sum = 0.0
        self._motion_energy.fill(0.0)
        self._motion_energy_slot = 0
        self._motion_energy_count = 0
        self._frequency_coherence_slot = 0
        self._frequency_coherence_count = 0
        self._frequency_coherence_sum = 0.0
        self._frequency_coherence_square_sum = 0.0
        self._frequency_curve_slot = 0
        self._frequency_curve_count = 0
        self._frequency_curve_sum = 0.0
        self._frequency_curve_square_sum = 0.0
        self._rank_lag_slot = 0
        self._rank_lag_count = 0
        self._rank_lag_sum = 0.0
        self._rank_adjacent_slot = 0
        self._rank_adjacent_count = 0
        self._rank_adjacent_sum = 0.0
        self._ratio_lag_slot = 0
        self._ratio_lag_count = 0
        self._ratio_lag_sum = 0.0
        self._ratio_adjacent_slot = 0
        self._ratio_adjacent_count = 0
        self._ratio_adjacent_sum = 0.0


class PhaseResidualTracker:
    """Running lag/adjacent displacement of sanitized phase profiles."""

    def __init__(self, window_size: int = 90, lag: int = L1_DELTA_LAG):
        self.window_size = max(2, int(window_size))
        self.lag = max(1, int(lag))
        width = len(_ADJACENT_PAIRS)
        self._ring = [np.zeros(width, dtype=np.complex128) for _ in range(self.lag)]
        self._ring_filled = [False] * self.lag
        self._index = 0
        self._previous = np.zeros(width, dtype=np.complex128)
        self._has_previous = False
        self._lag_ring = [0.0] * self.window_size
        self._adjacent_ring = [0.0] * self.window_size
        self._lag_slot = 0
        self._lag_count = 0
        self._lag_sum = 0.0
        self._adjacent_slot = 0
        self._adjacent_count = 0
        self._adjacent_sum = 0.0
        self._closure_ring = [0.0] * self.window_size
        self._closure_slot = 0
        self._closure_count = 0
        self._closure_sum = 0.0
        self._closure_square_sum = 0.0

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
        complex_values = complex_profile(csi_data)
        profile = sanitized_phase_profile(complex_values)
        slot = self._index
        if self._ring_filled[slot]:
            value = phase_profile_distance(profile, self._ring[slot])
            self._lag_slot, self._lag_count, self._lag_sum = self._push(
                value,
                self._lag_ring,
                self._lag_slot,
                self._lag_count,
                self._lag_sum,
            )
        if self._has_previous:
            value = phase_profile_distance(profile, self._previous)
            (
                self._adjacent_slot,
                self._adjacent_count,
                self._adjacent_sum,
            ) = self._push(
                value,
                self._adjacent_ring,
                self._adjacent_slot,
                self._adjacent_count,
                self._adjacent_sum,
            )
        self._previous = profile.copy()
        self._has_previous = True
        self._ring[slot] = profile
        self._ring_filled[slot] = True
        self._index = (self._index + 1) % self.lag
        closure = local_phase_closure_variance(complex_values)
        if self._closure_count < self.window_size:
            self._closure_count += 1
        else:
            old = self._closure_ring[self._closure_slot]
            self._closure_sum -= old
            self._closure_square_sum -= old * old
        self._closure_ring[self._closure_slot] = closure
        self._closure_sum += closure
        self._closure_square_sum += closure * closure
        self._closure_slot = (self._closure_slot + 1) % self.window_size

    def phase_residual_lag_ratio(self) -> float:
        if self._lag_count == 0 or self._adjacent_count == 0:
            return 1.0
        adjacent_mean = self._adjacent_sum / self._adjacent_count
        if adjacent_mean <= 0.0:
            return 1.0
        return (self._lag_sum / self._lag_count) / adjacent_mean

    def phase_closure_variance_std(self) -> float:
        """Temporal standard deviation of local closure circular variance."""
        if self._closure_count == 0:
            return 0.0
        mean = self._closure_sum / self._closure_count
        variance = max(
            0.0,
            self._closure_square_sum / self._closure_count - mean * mean,
        )
        return float(np.sqrt(variance))

    def reset(self) -> None:
        for i in range(self.lag):
            self._ring_filled[i] = False
        self._index = 0
        self._has_previous = False
        self._lag_slot = 0
        self._lag_count = 0
        self._lag_sum = 0.0
        self._adjacent_slot = 0
        self._adjacent_count = 0
        self._adjacent_sum = 0.0
        self._closure_slot = 0
        self._closure_count = 0
        self._closure_sum = 0.0
        self._closure_square_sum = 0.0


class ChannelCoherenceTracker:
    """Running delay-compensated coherence at the profile lag and at lag 1.

    Mirrors the shape of `L1DeltaTracker`: the same window of packets, the same
    pair of lags, and a ratio between them so no absolute magnitude survives.
    """

    def __init__(
        self,
        window_size: int = 90,
        lag: int = L1_DELTA_LAG,
        track_subbands: bool = False,
    ):
        self.window_size = max(2, int(window_size))
        self.lag = max(1, int(lag))
        self.track_subbands = bool(track_subbands)
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
        subband_count = len(HT20_COHERENCE_SUBBANDS)
        self._subband_lag_sum = np.zeros(subband_count, dtype=np.float64)
        self._subband_adjacent_sum = np.zeros(subband_count, dtype=np.float64)
        self._subband_lag_ring = np.zeros(
            (self.window_size, subband_count), dtype=np.float64
        )
        self._subband_adjacent_ring = np.zeros(
            (self.window_size, subband_count), dtype=np.float64
        )
        self._subband_lag_slot = 0
        self._subband_lag_count = 0
        self._subband_adjacent_slot = 0
        self._subband_adjacent_count = 0
        self._subband_lag_median_sum = 0.0
        self._subband_adjacent_median_sum = 0.0
        self._subband_lag_median_ring = [0.0] * self.window_size
        self._subband_adjacent_median_ring = [0.0] * self.window_size
        self._subband_lag_median_slot = 0
        self._subband_lag_median_count = 0
        self._subband_adjacent_median_slot = 0
        self._subband_adjacent_median_count = 0
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

    def _push_subbands(self, values, ring, slot, count, total):
        if count < self.window_size:
            count += 1
        else:
            total -= ring[slot]
        ring[slot] = values
        total += values
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
            if self.track_subbands:
                lag_subbands = subband_coherences(profile, self._ring[slot])
                (
                    self._subband_lag_slot,
                    self._subband_lag_count,
                    self._subband_lag_sum,
                ) = self._push_subbands(
                    lag_subbands,
                    self._subband_lag_ring,
                    self._subband_lag_slot,
                    self._subband_lag_count,
                    self._subband_lag_sum,
                )
                (
                    self._subband_lag_median_slot,
                    self._subband_lag_median_count,
                    self._subband_lag_median_sum,
                ) = self._push(
                    float(np.median(lag_subbands)),
                    self._subband_lag_median_ring,
                    self._subband_lag_median_slot,
                    self._subband_lag_median_count,
                    self._subband_lag_median_sum,
                )
        if self._has_previous:
            adjacent_value = delay_compensated_coherence(profile, self._previous)
            (self._adjacent_slot, self._adjacent_count,
             self._adjacent_sum) = self._push(
                adjacent_value, self._adjacent_ring, self._adjacent_slot,
                self._adjacent_count, self._adjacent_sum
            )
            if self.track_subbands:
                adjacent_subbands = subband_coherences(profile, self._previous)
                (
                    self._subband_adjacent_slot,
                    self._subband_adjacent_count,
                    self._subband_adjacent_sum,
                ) = self._push_subbands(
                    adjacent_subbands,
                    self._subband_adjacent_ring,
                    self._subband_adjacent_slot,
                    self._subband_adjacent_count,
                    self._subband_adjacent_sum,
                )
                (
                    self._subband_adjacent_median_slot,
                    self._subband_adjacent_median_count,
                    self._subband_adjacent_median_sum,
                ) = self._push(
                    float(np.median(adjacent_subbands)),
                    self._subband_adjacent_median_ring,
                    self._subband_adjacent_median_slot,
                    self._subband_adjacent_median_count,
                    self._subband_adjacent_median_sum,
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

    def coherence_subband_median_gap(self) -> float:
        """Gap after taking the median subband coherence for every pair."""
        if (
            self._subband_lag_median_count == 0
            or self._subband_adjacent_median_count == 0
        ):
            return 0.0
        return (
            self._subband_adjacent_median_sum
            / self._subband_adjacent_median_count
        ) - (
            self._subband_lag_median_sum
            / self._subband_lag_median_count
        )

    def coherence_subband_gap_median(self) -> float:
        """Median across subbands of their adjacent-minus-lag mean gaps."""
        if self._subband_lag_count == 0 or self._subband_adjacent_count == 0:
            return 0.0
        gaps = (
            self._subband_adjacent_sum / self._subband_adjacent_count
        ) - (
            self._subband_lag_sum / self._subband_lag_count
        )
        return float(np.median(gaps))

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
        self._subband_lag_sum.fill(0.0)
        self._subband_adjacent_sum.fill(0.0)
        self._subband_lag_slot = 0
        self._subband_lag_count = 0
        self._subband_adjacent_slot = 0
        self._subband_adjacent_count = 0
        self._subband_lag_median_sum = 0.0
        self._subband_adjacent_median_sum = 0.0
        self._subband_lag_median_slot = 0
        self._subband_lag_median_count = 0
        self._subband_adjacent_median_slot = 0
        self._subband_adjacent_median_count = 0

