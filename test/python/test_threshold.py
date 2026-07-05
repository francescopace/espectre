"""
Tests for `src/python/micro_espectre/threshold.py`.
"""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_SRC = REPO_ROOT / "src" / "python" / "micro_espectre"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

from threshold import calculate_adaptive_threshold, calculate_startup_threshold_from_max


def test_calculate_startup_threshold_from_max_uses_auto_factor() -> None:
    threshold, formula = calculate_startup_threshold_from_max(0.25, "auto")

    assert threshold == 0.25 * 1.3
    assert formula == "max x 1.3"


def test_calculate_adaptive_threshold_handles_empty_iterable() -> None:
    threshold, formula = calculate_adaptive_threshold([], "auto")

    assert threshold == 0.0
    assert formula == "max x 1.3"
