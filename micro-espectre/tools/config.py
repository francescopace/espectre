"""
Analysis Configuration

Centralized configuration for CSI analysis tools.
Contains optimal parameters found by comprehensive grid search.

Optimal parameters found by comprehensive grid search (2_comprehensive_grid_search.py)
Performance: FP=0, TP=29, Score=29.00

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

# Optimal MVS parameters (from full grid search)
#SELECTED_SUBCARRIERS = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
SELECTED_SUBCARRIERS = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
WINDOW_SIZE = 50
THRESHOLD = 1

# Hampel filter parameters (outlier/spike removal)
HAMPEL_WINDOW = 7          # Window size for median calculation
HAMPEL_THRESHOLD = 4.0     # MAD threshold to flag as outlier

# Low-pass filter parameters
LOWPASS_CUTOFF = 11.0      # Cutoff frequency in Hz
