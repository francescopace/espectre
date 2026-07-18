"""
ESPectre - CSI Analysis

Shared CSI analysis helpers for tool-side workflows.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

from __future__ import annotations

import numpy as np

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


def extract_amplitudes_matrix(csi_matrix) -> np.ndarray:
    """Extract amplitudes for all packets at once using numpy.

    CSI format: [Q0, I0, Q1, I1, ...] per packet (128 int8 values for 64
    subcarriers).  Amplitude = sqrt(I^2 + Q^2).  Values are upcast to int16
    before squaring to avoid int8 overflow.

    Args:
        csi_matrix: numpy array of shape (num_packets, 2 * num_subcarriers)

    Returns:
        numpy array of shape (num_packets, num_subcarriers), dtype float64
    """
    data = np.asarray(csi_matrix).astype(np.int16)
    q_values = data[:, 0::2]
    i_values = data[:, 1::2]
    return np.sqrt((i_values * i_values + q_values * q_values).astype(np.float64))
