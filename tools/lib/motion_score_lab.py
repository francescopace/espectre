"""
Research-only library for the independent motion-score benchmark.

This module is intentionally separate from the production detector code
and from `tools/12_benchmark_motion_features.py` (owned by a different,
concurrent research track). It re-uses the *read-only* production
primitives (amplitude extraction, spatial turbulence, Hampel filter) so
that every candidate score is built from the exact same conditioned
signal the runtime would see, but all windowing, scoring, and evaluation
logic below is new and does not touch `src/`.

Nothing here is imported by production code paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .bootstrap import setup_paths

setup_paths()

try:
    import config
except ImportError:
    import src.config as config

try:
    from filters import HampelFilter
except ImportError:  # pragma: no cover
    from src.filters import HampelFilter


DEFAULT_SUBCARRIERS: Tuple[int, ...] = tuple(config.DEFAULT_SUBCARRIERS)
NUM_SUBCARRIERS: int = int(config.NUM_SUBCARRIERS)
SEG_WINDOW_SIZE: int = int(config.SEG_WINDOW_SIZE)
HAMPEL_WINDOW: int = int(config.HAMPEL_WINDOW)
HAMPEL_THRESHOLD: float = float(config.HAMPEL_THRESHOLD)
EPS = 1e-9


# ---------------------------------------------------------------------------
# Amplitude extraction (vectorized, mirrors SegmentationContext byte layout)
# ---------------------------------------------------------------------------


def amplitudes_from_csi_batch(csi_batch: np.ndarray) -> np.ndarray:
    """Return (N, 64) amplitude matrix from an (N, 128) int8 I/Q batch.

    Matches the production byte layout: even index = Q (imag), odd index =
    I (real), i.e. amplitude[sc] = sqrt(I[sc]^2 + Q[sc]^2).
    """
    csi = np.asarray(csi_batch, dtype=np.float32)
    q = csi[:, 0::2]
    i = csi[:, 1::2]
    return np.sqrt(i * i + q * q)


def packets_to_csi_matrix(packets: Sequence[Dict[str, Any]]) -> np.ndarray:
    """Stack a list of packet dicts into one (N, 128) int8 matrix."""
    return np.stack([np.asarray(p["csi_data"], dtype=np.int8) for p in packets], axis=0)


def raw_turbulence(amp_selected: np.ndarray) -> np.ndarray:
    """Per-packet CV turbulence: std/mean across the selected-subcarrier axis."""
    mean = amp_selected.mean(axis=1)
    std = amp_selected.std(axis=1)
    turbulence = np.divide(std, mean, out=np.zeros_like(std), where=mean > 0)
    return turbulence.astype(np.float64)


def hampel_filter_series(x: np.ndarray, window: int = HAMPEL_WINDOW, threshold: float = HAMPEL_THRESHOLD) -> np.ndarray:
    """Vectorized causal Hampel filter, numerically equivalent to filters.HampelFilter.

    For i >= window - 1 the window is [i-window+1, i] (vectorized). The first
    `window - 1` samples use a real growing-window causal filter (delegated
    to the exact production HampelFilter for correctness).
    """
    n = len(x)
    out = np.array(x, dtype=np.float64, copy=True)
    if n == 0:
        return out

    warmup = min(window - 1, n)
    live = HampelFilter(window_size=window, threshold=threshold)
    for i in range(warmup):
        out[i] = live.filter(float(x[i]))

    if n <= warmup:
        return out

    windows = np.lib.stride_tricks.sliding_window_view(x, window)  # (n-window+1, window)
    scaled_threshold = threshold * 1.4826
    median = np.median(windows, axis=1)
    mad = np.median(np.abs(windows - median[:, None]), axis=1)
    current = windows[:, -1]
    deviation = np.divide(
        np.abs(current - median), mad, out=np.zeros_like(median), where=mad > 1e-6
    )
    is_outlier = (mad > 1e-6) & (deviation > scaled_threshold)
    filtered = np.where(is_outlier, median, current)
    out[window - 1 :] = filtered
    return out


def estimate_sample_rate_hz(entry: Dict[str, Any], num_packets: int) -> float:
    """Estimate the effective packet rate from dataset_info duration metadata."""
    duration_ms = float(entry.get("duration_ms") or 0.0)
    if duration_ms > 0 and num_packets > 1:
        return float(num_packets - 1) / (duration_ms / 1000.0)
    return 100.0


# ---------------------------------------------------------------------------
# Windowed candidate scores
# ---------------------------------------------------------------------------


def _sliding_1d(x: np.ndarray, window: int, hop: int) -> np.ndarray:
    view = np.lib.stride_tricks.sliding_window_view(x, window)
    return view[::hop]


def _sliding_2d_time_major(x: np.ndarray, window: int, hop: int) -> np.ndarray:
    """x: (N, C) -> (W, window, C), subsampled every `hop` windows."""
    view = np.lib.stride_tricks.sliding_window_view(x, window, axis=0)  # (N-window+1, C, window)
    view = view[::hop]
    return np.ascontiguousarray(np.transpose(view, (0, 2, 1)))


def window_end_indices(n: int, window: int, hop: int) -> np.ndarray:
    total = n - window + 1
    if total <= 0:
        return np.array([], dtype=np.int64)
    starts = np.arange(0, total, hop)
    return starts + window - 1


def _relative_iqr(turb_windows: np.ndarray, mean_abs: np.ndarray) -> np.ndarray:
    q75 = np.percentile(turb_windows, 75, axis=1)
    q25 = np.percentile(turb_windows, 25, axis=1)
    return (q75 - q25) / mean_abs


def _relative_mad(turb_windows: np.ndarray, mean_abs: np.ndarray) -> np.ndarray:
    median = np.median(turb_windows, axis=1, keepdims=True)
    mad = np.median(np.abs(turb_windows - median), axis=1)
    return mad / mean_abs


def _lag1_decorrelation(turb_windows: np.ndarray) -> np.ndarray:
    mean = turb_windows.mean(axis=1, keepdims=True)
    centered = turb_windows - mean
    var = (centered * centered).mean(axis=1)
    cov = (centered[:, :-1] * centered[:, 1:]).mean(axis=1)
    autocorr = np.divide(cov, var, out=np.zeros_like(var), where=var > EPS)
    return 1.0 - autocorr


def _band_power_and_entropy(turb_windows: np.ndarray, fs: float, band=(0.5, 10.0)) -> Tuple[np.ndarray, np.ndarray]:
    window = turb_windows.shape[1]
    hann = np.hanning(window)
    centered = turb_windows - turb_windows.mean(axis=1, keepdims=True)
    spectrum = np.fft.rfft(centered * hann[None, :], axis=1)
    power = (spectrum.real ** 2 + spectrum.imag ** 2)
    freqs = np.fft.rfftfreq(window, d=1.0 / fs)

    # Exclude the DC bin (index 0) from all spectral ratios.
    power_ac = power[:, 1:]
    freqs_ac = freqs[1:]
    total_power = power_ac.sum(axis=1)

    band_mask = (freqs_ac >= band[0]) & (freqs_ac <= band[1])
    band_power = power_ac[:, band_mask].sum(axis=1)
    band_ratio = np.divide(band_power, total_power, out=np.zeros_like(total_power), where=total_power > EPS)

    probs = np.divide(power_ac, total_power[:, None], out=np.zeros_like(power_ac), where=total_power[:, None] > EPS)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_probs = np.where(probs > 0, np.log(probs), 0.0)
    entropy = -(probs * log_probs).sum(axis=1)
    norm_entropy = entropy / np.log(max(power_ac.shape[1], 2))
    return band_ratio, norm_entropy


def _pca_eigen_spread(amp_windows: np.ndarray) -> np.ndarray:
    """amp_windows: (W, window, C). Returns 1 - (top eigenvalue / trace)."""
    row_mean = amp_windows.mean(axis=2, keepdims=True)
    normalized = amp_windows / np.maximum(row_mean, EPS)
    centered = normalized - normalized.mean(axis=1, keepdims=True)
    window = amp_windows.shape[1]
    cov = np.einsum("wtc,wtd->wcd", centered, centered) / window
    eigvals = np.linalg.eigvalsh(cov)  # ascending order, shape (W, C)
    trace = eigvals.sum(axis=1)
    top = eigvals[:, -1]
    return 1.0 - np.divide(top, trace, out=np.ones_like(trace), where=trace > EPS)


def _subcarrier_decorrelation(amp_windows: np.ndarray) -> np.ndarray:
    """amp_windows: (W, window, C). Returns 1 - mean pairwise Pearson correlation."""
    mean = amp_windows.mean(axis=1, keepdims=True)
    std = amp_windows.std(axis=1, keepdims=True)
    z = (amp_windows - mean) / np.maximum(std, EPS)
    window = amp_windows.shape[1]
    corr = np.einsum("wtc,wtd->wcd", z, z) / window
    num_sc = corr.shape[-1]
    iu = np.triu_indices(num_sc, k=1)
    mean_offdiag = corr[:, iu[0], iu[1]].mean(axis=1)
    return 1.0 - mean_offdiag


CANDIDATE_NAMES: Tuple[str, ...] = (
    "mvs_var",
    "turb_mean",
    "turb_iqr_rel",
    "turb_mad_rel",
    "turb_decorr",
    "band_power_ratio",
    "spectral_entropy",
    "pca_eigen_spread",
    "subcarrier_decorr",
    "ctrl_rssi_level",
    "ctrl_mean_amp_level",
)

# Physically-expected sign: True if the score is expected to increase with
# motion. Verified numerically per LOCO fold, not assumed at evaluation time.
EXPECTED_INCREASE_WITH_MOTION: Dict[str, bool] = {
    "mvs_var": True,
    "turb_mean": True,
    "turb_iqr_rel": True,
    "turb_mad_rel": True,
    "turb_decorr": True,
    "band_power_ratio": True,
    "spectral_entropy": True,
    "pca_eigen_spread": True,
    "subcarrier_decorr": True,
    "ctrl_rssi_level": False,
    "ctrl_mean_amp_level": False,
}


@dataclass
class WindowScores:
    end_index: np.ndarray
    scores: Dict[str, np.ndarray]


def compute_window_scores(
    turb_filtered: np.ndarray,
    amp_selected: np.ndarray,
    amp_all: np.ndarray,
    *,
    window: int = SEG_WINDOW_SIZE,
    hop: int = 10,
    fs: float = 100.0,
) -> WindowScores:
    """Compute all candidate scores over sliding windows of one continuous stream."""
    n = len(turb_filtered)
    end_idx = window_end_indices(n, window, hop)
    if len(end_idx) == 0:
        return WindowScores(end_index=end_idx, scores={name: np.array([]) for name in CANDIDATE_NAMES})

    turb_windows = _sliding_1d(turb_filtered, window, hop)
    mean_signed = turb_windows.mean(axis=1)
    mean_abs = np.maximum(np.abs(mean_signed), 1e-6)

    scores: Dict[str, np.ndarray] = {}
    scores["mvs_var"] = turb_windows.var(axis=1)
    scores["turb_mean"] = mean_signed
    scores["turb_iqr_rel"] = _relative_iqr(turb_windows, mean_abs)
    scores["turb_mad_rel"] = _relative_mad(turb_windows, mean_abs)
    scores["turb_decorr"] = _lag1_decorrelation(turb_windows)
    band_ratio, entropy = _band_power_and_entropy(turb_windows, fs)
    scores["band_power_ratio"] = band_ratio
    scores["spectral_entropy"] = entropy

    amp_sel_windows = _sliding_2d_time_major(amp_selected, window, hop)
    scores["pca_eigen_spread"] = _pca_eigen_spread(amp_sel_windows)
    scores["subcarrier_decorr"] = _subcarrier_decorrelation(amp_sel_windows)
    scores["ctrl_mean_amp_level"] = amp_sel_windows.mean(axis=(1, 2))

    amp_all_windows = _sliding_2d_time_major(amp_all, window, hop)
    scores["ctrl_rssi_level"] = amp_all_windows.mean(axis=(1, 2))

    return WindowScores(end_index=end_idx, scores=scores)


# ---------------------------------------------------------------------------
# Synthetic perturbations for AGC / RF-noise robustness stress tests
# ---------------------------------------------------------------------------


def apply_gain_shift(amp_all: np.ndarray, amp_selected: np.ndarray, factor: float) -> Tuple[np.ndarray, np.ndarray]:
    """Uniformly scale all amplitudes by `factor` (simulated AGC/gain drift)."""
    return amp_all * factor, amp_selected * factor


def inject_spike_noise(
    amp_all: np.ndarray,
    amp_selected: np.ndarray,
    *,
    rng: np.random.Generator,
    spike_rate: float = 0.005,
    spike_factor_range: Tuple[float, float] = (3.0, 8.0),
    selected_indices: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Inject single-subcarrier, single-packet multiplicative spikes.

    Mirrors the documented real failure mode (EXPERIMENTS.md): isolated
    frame-scale amplitude jumps on individual packets/subcarriers, not a
    sustained gain shift.
    """
    n, num_sc_all = amp_all.shape
    out_all = amp_all.copy()
    out_sel = amp_selected.copy()
    num_spikes = int(n * spike_rate)
    if num_spikes <= 0:
        return out_all, out_sel

    packet_idx = rng.integers(0, n, size=num_spikes)
    sc_idx_all = rng.integers(0, num_sc_all, size=num_spikes)
    factors = rng.uniform(spike_factor_range[0], spike_factor_range[1], size=num_spikes)
    out_all[packet_idx, sc_idx_all] *= factors

    if selected_indices is not None:
        selected_indices = list(selected_indices)
        # Re-derive the effect on the selected-subcarrier view directly from
        # the perturbed full matrix to keep the two views consistent.
        out_sel = out_all[:, selected_indices]
    return out_all, out_sel
