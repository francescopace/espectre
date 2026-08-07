"""
ESPectre - Host-Side Candidate Features

Registry and evaluator for the still-active host-only candidate features.
Shared tracker math and HT20 feature primitives live in
`host_feature_trackers.py`, which the current production feature set also uses
during host-side training and validation.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .host_feature_trackers import (
    AGGREGATED_SPECTRAL_FEATURES,
    CHANNEL_COHERENCE_FEATURES,
    CHANNEL_SHAPE_FEATURES,
    COMPOSITE_FEATURES,
    PHASE_FEATURES,
    PROMOTED_CHANNEL_COHERENCE_FEATURES,
    PROMOTED_CHANNEL_SHAPE_FEATURES,
    SPECTRAL_FEATURES,
    SUBBAND_COHERENCE_FEATURES,
    ChannelCoherenceTracker,
    ChannelShapeTracker,
    PhaseResidualTracker,
)

CANDIDATE_FEATURES: Tuple[str, ...] = (
    CHANNEL_COHERENCE_FEATURES
    + SPECTRAL_FEATURES
    + AGGREGATED_SPECTRAL_FEATURES
    + PHASE_FEATURES
    + CHANNEL_SHAPE_FEATURES
    + COMPOSITE_FEATURES
)


def needs_channel_coherence(feature_names: Iterable[str]) -> bool:
    """Return whether any requested feature needs the coherence tracker."""
    return any(
        name in CHANNEL_COHERENCE_FEATURES
        or name in PROMOTED_CHANNEL_COHERENCE_FEATURES
        or name in COMPOSITE_FEATURES
        for name in feature_names
    )


def needs_subband_coherence(feature_names: Iterable[str]) -> bool:
    """Return whether any requested feature needs per-subband coherence."""
    return any(name in SUBBAND_COHERENCE_FEATURES for name in feature_names)


def needs_turbulence_series(feature_names: Iterable[str]) -> bool:
    """Return whether any requested candidate reads the turbulence window."""
    return any(name in SPECTRAL_FEATURES for name in feature_names)


def needs_aggregated_turbulence(feature_names: Iterable[str]) -> bool:
    """Return whether any requested candidate reads the aggregated window."""
    return any(name in AGGREGATED_SPECTRAL_FEATURES for name in feature_names)


def needs_phase_residual(feature_names: Iterable[str]) -> bool:
    """Return whether any requested feature needs sanitized phase tracking."""
    return any(name in PHASE_FEATURES for name in feature_names)


def needs_channel_shape(feature_names: Iterable[str]) -> bool:
    """Return whether any requested feature needs normalized channel shape."""
    return any(
        name in CHANNEL_SHAPE_FEATURES
        or name in PROMOTED_CHANNEL_SHAPE_FEATURES
        or name in COMPOSITE_FEATURES
        for name in feature_names
    )


def split_feature_names(
    feature_names: Iterable[str],
) -> Tuple[List[str], List[str]]:
    """Split a requested set into production names and candidate names."""
    names = list(feature_names)
    candidates = [name for name in names if name in CANDIDATE_FEATURES]
    production = [name for name in names if name not in CANDIDATE_FEATURES]
    return production, candidates


def assemble_feature_vector(
    feature_names: Sequence[str],
    production_names: Sequence[str],
    production_values: Sequence[float],
    candidate_feature_values: Mapping[str, float],
) -> List[float]:
    """Rebuild the full feature vector in the caller's requested order."""
    production_lookup = dict(zip(production_names, production_values))
    return [
        candidate_feature_values[name]
        if name in candidate_feature_values
        else production_lookup[name]
        for name in feature_names
    ]


def candidate_values(
    feature_names: Iterable[str],
    coherence_tracker: ChannelCoherenceTracker = None,
    turbulence_series: Sequence[float] = None,
    aggregated_turbulence_series: Sequence[float] = None,
    phase_tracker: PhaseResidualTracker = None,
    shape_tracker: ChannelShapeTracker = None,
) -> Dict[str, float]:
    """Evaluate the requested candidates from their preprocessed trackers."""
    values: Dict[str, float] = {}
    turbulence = None
    if turbulence_series is not None:
        turbulence = np.asarray(turbulence_series, dtype=np.float64)
    aggregated_turbulence = None
    if aggregated_turbulence_series is not None:
        aggregated_turbulence = np.asarray(
            aggregated_turbulence_series,
            dtype=np.float64,
        )
    mean_denom = None
    iqr = None
    q95 = None
    aggregated_mean_denom = None
    aggregated_mad = None
    aggregated_q95 = None

    for name in feature_names:
        if name not in CANDIDATE_FEATURES:
            continue
        if name in SPECTRAL_FEATURES:
            if turbulence is None:
                raise ValueError(
                    f"{name} needs the turbulence window; pass the explicitly "
                    f"preprocessed stream"
                )
            if len(turbulence) < 4:
                values[name] = 0.0
                continue
            mean = float(np.mean(turbulence))
            if mean_denom is None:
                mean_denom = abs(mean) if abs(mean) > 1e-6 else 1e-6
            if name == 'turb_iqr_over_mean':
                if iqr is None:
                    q25, q75 = np.percentile(turbulence, [25, 75])
                    iqr = float(q75 - q25)
                values[name] = iqr / mean_denom
                continue
            if name == 'turb_p95_over_mean':
                if q95 is None:
                    q95 = float(np.percentile(turbulence, 95))
                values[name] = q95 / mean_denom
                continue
        if name in AGGREGATED_SPECTRAL_FEATURES:
            if aggregated_turbulence is None:
                raise ValueError(
                    f"{name} needs the aggregated turbulence window; pass the "
                    f"explicitly preprocessed aggregated stream"
                )
            if len(aggregated_turbulence) < 4:
                values[name] = 0.0
                continue
            aggregated_mean = float(np.mean(aggregated_turbulence))
            if aggregated_mean_denom is None:
                aggregated_mean_denom = (
                    abs(aggregated_mean)
                    if abs(aggregated_mean) > 1e-6
                    else 1e-6
                )
            if name == 'turb_mad_over_mean_aggr':
                if aggregated_mad is None:
                    median = float(np.median(aggregated_turbulence))
                    aggregated_mad = float(
                        np.median(np.abs(aggregated_turbulence - median))
                    )
                values[name] = aggregated_mad / aggregated_mean_denom
                continue
            if name == 'turb_p95_over_mean_aggr':
                if aggregated_q95 is None:
                    aggregated_q95 = float(np.percentile(aggregated_turbulence, 95))
                values[name] = aggregated_q95 / aggregated_mean_denom
                continue
        if name in CHANNEL_COHERENCE_FEATURES and coherence_tracker is None:
            raise ValueError(
                f"{name} needs the channel coherence tracker; pass the "
                f"explicitly preprocessed stream"
            )
        if (
            name in SUBBAND_COHERENCE_FEATURES
            and not coherence_tracker.track_subbands
        ):
            raise ValueError(
                f"{name} needs subband coherence tracking; construct the "
                f"tracker with track_subbands=True"
            )
        if name == 'chan_coh_lag_ratio':
            values[name] = coherence_tracker.coherence_lag_ratio()
        elif name == 'phase_resid_lag_ratio':
            if phase_tracker is None:
                raise ValueError(
                    f"{name} needs the sanitized phase tracker"
                )
            values[name] = phase_tracker.phase_residual_lag_ratio()
        elif name == 'phase_closure_var_std':
            if phase_tracker is None:
                raise ValueError(
                    f"{name} needs the sanitized phase tracker"
                )
            values[name] = phase_tracker.phase_closure_variance_std()
    return values
