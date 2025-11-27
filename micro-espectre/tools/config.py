"""
Analysis Configuration
Centralized configuration for CSI analysis tools

Optimal parameters found by comprehensive grid search (2_comprehensive_grid_search.py)
Performance: FP=0, TP=29, Score=29.00

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

# Optimal MVS parameters (from full grid search)
SELECTED_SUBCARRIERS = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
WINDOW_SIZE = 50
THRESHOLD = 1
