"""
Micro-ESPectre Configuration

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

# WiFi Configuration (override in wifi_config.py)
WIFI_SSID = "YourSSID"
WIFI_PASSWORD = "YourPassword"

# MQTT Configuration
MQTT_BROKER = "homeassistant.local"  # Your MQTT broker IP
MQTT_PORT = 1883
MQTT_CLIENT_ID = "espectre-lite"
MQTT_TOPIC = "home/espectre/node1"
MQTT_USERNAME = "mqtt"
MQTT_PASSWORD = "mqtt"

# CSI Configuration
CSI_BUFFER_SIZE = 10  # Circular buffer size (used to store csi packets until processed)

# Selected subcarriers for turbulence calculation (from C version)
# These are the most informative subcarriers identified through analysis
SELECTED_SUBCARRIERS = [47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58]

# Segmentation Configuration
# Values calibrated for int8 CSI data with 12 selected subcarriers
SEG_WINDOW_SIZE = 30      # Moving variance window (packets)
SEG_MIN_LENGTH = 10       # Min motion segment length (packets)
SEG_MAX_LENGTH = 60       # Max motion segment length (packets)
SEG_THRESHOLD = 3.0       # Motion detection threshold (Lower values = more sensitive to motion)

# Publishing Configuration
# If SMART_PUBLISHING = False, messages are published every 1 secon
SMART_PUBLISHING = False     # Only publish on significant changes
DELTA_THRESHOLD = 0.05      # Minimum change to trigger publish (0.05 = 5%)
MAX_PUBLISH_INTERVAL_MS = 5000  # Max time between publishes - heartbeat (5 seconds)

# Traffic Generator Configuration
# Generates WiFi traffic (ICMP ping) to ensure continuous CSI data
TRAFFIC_GENERATOR_RATE = 20  # Packets per second (0=disabled, 1-50, recommended: 20)
