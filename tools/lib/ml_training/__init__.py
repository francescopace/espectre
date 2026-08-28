# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Implementation package for the ML training command."""

from pathlib import Path


def implementation_paths() -> tuple[Path, ...]:
    """Return the Python sources that define the training workflow."""
    return tuple(sorted(Path(__file__).resolve().parent.glob("*.py")))
