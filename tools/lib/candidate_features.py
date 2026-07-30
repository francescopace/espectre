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

from .host_feature_trackers import (
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
    phase_tracker: PhaseResidualTracker = None,
    shape_tracker: ChannelShapeTracker = None,
) -> Dict[str, float]:
    """Evaluate the requested candidates from their preprocessed trackers."""
    values: Dict[str, float] = {}
    for name in feature_names:
        if name not in CANDIDATE_FEATURES:
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
