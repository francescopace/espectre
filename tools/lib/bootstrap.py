# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - Tools Bootstrap

Runtime bootstrap helpers for tool-side imports.

Author: Francesco Pace <francesco.pace@gmail.com>
"""

from __future__ import annotations

import sys

from .repo_paths import python_src_dir, repo_root, tools_dir


def setup_paths() -> None:
    """
    Add repository-root-relative source directories to ``sys.path``.

    Safe to call multiple times.
    """
    src_path = str(python_src_dir())
    current_tools_path = str(tools_dir())
    root_path = str(repo_root())

    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    if current_tools_path not in sys.path:
        sys.path.insert(0, current_tools_path)
    if root_path not in sys.path:
        sys.path.insert(0, root_path)


setup_paths()

