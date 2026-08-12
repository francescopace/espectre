# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - Repo Paths

Repository path helpers for ESPectre host-side tooling.

Author: Francesco Pace <francesco.pace@gmail.com>
"""

from pathlib import Path


def repo_root() -> Path:
    """Return the ESPectre repository root."""
    path = Path(__file__).resolve()
    if path.parent.name == "tools":
        return path.parent.parent
    return path.parents[2]


def micro_espectre_root() -> Path:
    """Return the legacy Micro-ESPectre project root."""
    return repo_root() / "micro-espectre"


def python_src_dir() -> Path:
    """Return the current host-side Python source directory."""
    root = repo_root()
    new_layout = root / "src" / "python" / "micro_espectre"
    if new_layout.exists():
        return new_layout
    return micro_espectre_root() / "src"


def python_tests_dir() -> Path:
    """Return the current host-side Python tests directory."""
    root = repo_root()
    new_layout = root / "test" / "python"
    if new_layout.exists():
        return new_layout
    return micro_espectre_root() / "tests"


def tools_dir() -> Path:
    """Return the current tools directory."""
    root = repo_root()
    new_layout = root / "tools"
    if new_layout.exists():
        return new_layout
    return micro_espectre_root() / "tools"


def data_dir() -> Path:
    """Return the current shared data directory."""
    root = repo_root()
    new_layout = root / "data"
    if new_layout.exists():
        return new_layout
    return micro_espectre_root() / "data"


def generated_data_dir() -> Path:
    """Return the current shared generated-data directory."""
    return data_dir() / "auto_generated"


def cpp_core_dir() -> Path:
    """Return the current shared C++ core directory."""
    root = repo_root()
    new_layout = root / "src" / "cpp" / "core"
    if new_layout.exists():
        return new_layout
    return root / "src" / "core"


def requirements_file() -> Path:
    """Return the current Python requirements file."""
    root = repo_root()
    new_layout = root / "requirements.txt"
    if new_layout.exists():
        return new_layout
    return micro_espectre_root() / "requirements.txt"

