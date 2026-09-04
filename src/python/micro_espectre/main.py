# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Minimal Micro-ESPectre entry point that reserves Wi-Fi memory first."""

from src.wifi_bootstrap import main


if __name__ == "__main__":
    main()
