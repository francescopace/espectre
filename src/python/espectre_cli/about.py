# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - CLI About

About and version helpers for the ESPectre repository CLI.

Author: Francesco Pace <francesco.pace@gmail.com>
"""

from __future__ import annotations

import subprocess

from .common import Fore, REPO_ROOT, Style
from micro_espectre.branding import ASCII_BANNER


def cli_version_label() -> str:
    """Return a human-friendly version label for this workspace checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return "workspace checkout"

    revision = result.stdout.strip()
    return f"git {revision}" if revision else "workspace checkout"


def print_about(_args=None) -> int:
    """Print project and CLI information."""
    print(f"{Fore.MAGENTA}{ASCII_BANNER}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}ESPectre repository CLI for device, host, and frontend workflows{Style.RESET_ALL}")
    print()
    print(f"Version: {cli_version_label()}")
    print("GitHub: github.com/francescopace/espectre")
    return 0


def print_version(_args=None) -> int:
    """Print the CLI version label."""
    print(f"ESPectre CLI {cli_version_label()}")
    return 0
