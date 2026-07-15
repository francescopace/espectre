"""
Shared CSI analysis helpers for tool-side workflows.
"""

from __future__ import annotations

from .bootstrap import setup_paths

setup_paths()

try:
    import config
except ImportError:
    import src.config as config

try:
    from segmentation import SegmentationContext
except ImportError:  # pragma: no cover
    from src.segmentation import SegmentationContext


def calculate_spatial_turbulence(csi_data, selected_subcarriers=None) -> float:
    """
    Calculate spatial turbulence from CSI data using the normalized AGC-active path.
    """
    band = config.DEFAULT_SUBCARRIERS if selected_subcarriers is None else selected_subcarriers
    turbulence, _ = SegmentationContext.compute_spatial_turbulence(csi_data, band)
    return turbulence


def calculate_variance_two_pass(values) -> float:
    """Calculate variance using the shared numerically stable implementation."""
    return SegmentationContext.compute_variance_two_pass(values)
