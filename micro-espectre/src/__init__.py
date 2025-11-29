"""
Micro-ESPectre - MicroPython CSI Motion Detection

WiFi CSI-based motion detection for ESP32-C6.
Main package for the Micro-ESPectre system.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

__version__ = "1.0.0"
__author__ = "Francesco Pace"

from .segmentation import SegmentationContext
from .mqtt.handler import MQTTHandler
from .traffic_generator import TrafficGenerator
from .nvs_storage import NVSStorage
