"""
Micro-ESPectre - Analysis Tools

Collection of analysis and optimization scripts for CSI data processing.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

# Make common tooling helpers available at package level.
from .lib.csi_io import CSICollector
from .lib.dataset_metadata import get_dataset_stats

__all__ = ['CSICollector', 'get_dataset_stats']

