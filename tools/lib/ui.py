# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
ESPectre - Tools UI

Shared UI helpers for tool scripts.

Author: Francesco Pace <francesco.pace@gmail.com>
"""


def show_plot_window(plot_module, *, cancel_message: str = "Plot display cancelled.") -> bool:
    """
    Show a matplotlib window and handle Ctrl-C gracefully.

    Returns:
        bool: True when the plot closed normally, False when cancelled.
    """
    try:
        plot_module.show()
        return True
    except KeyboardInterrupt:
        try:
            plot_module.close("all")
        except Exception:
            pass
        print(f"\n{cancel_message}")
        return False

