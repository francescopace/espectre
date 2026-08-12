# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
Micro-ESPectre - MicroPython CSI Motion Detection

WiFi CSI-based motion detection for ESP32-C6.
Main package for the Micro-ESPectre system.

Author: Francesco Pace <francesco.pace@gmail.com>
"""

__version__ = "1.0.0"
__author__ = "Francesco Pace"

# Note: No automatic imports here to avoid MicroPython-specific dependencies
# (umqtt, etc.) breaking pytest. Import modules explicitly where needed.
