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
CSI_BUFFER_SIZE = 16  # Circular buffer size (used to store csi packets until processed)

# Selected subcarriers for turbulence calculation
# These are the most informative subcarriers identified through analysis
# Can be automatically calibrated using NBVI algorithm at boot
SELECTED_SUBCARRIERS = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

# NBVI Auto-Calibration Configuration
NBVI_ENABLED = True           # Enable automatic subcarrier selection at boot
NBVI_BUFFER_SIZE = 500        # Packets to collect for calibration (5s @ 100Hz)
NBVI_WINDOW_SIZE = 100        # Window size for baseline detection (1s @ 100Hz)
NBVI_WINDOW_STEP = 50         # Step size for sliding window analysis
NBVI_PERCENTILE = 10          # Percentile for baseline detection (10 = p10)
NBVI_ALPHA = 0.3              # NBVI weighting factor (0.3 = optimal)
NBVI_MIN_SPACING = 3          # Minimum spacing between subcarriers (Δf≥3)
NBVI_NOISE_GATE_PERCENTILE = 10  # Exclude weak subcarriers below this percentile

# Segmentation Configuration
# Limits (matching segmentation.h)
SEG_WINDOW_SIZE_MIN = 10
SEG_WINDOW_SIZE_MAX = 200
SEG_THRESHOLD_MIN = 0.0
SEG_THRESHOLD_MAX = 10.0

# Defaults
SEG_WINDOW_SIZE = 50     # Moving variance window (packets)
SEG_THRESHOLD = 1.0       # Motion detection threshold (Lower values = more sensitive to motion)

# Turbulence Filtering Configuration
ENABLE_HAMPEL_FILTER = True    # Enable/disable Hampel outlier filter
HAMPEL_WINDOW = 7              # Window size for median calculation (3-9 recommended)
HAMPEL_THRESHOLD = 4.0         # Outlier detection threshold in MAD units (2.0-4.0 recommended)
                               # Higher values = less aggressive filtering

# Subcarrier limits
SUBCARRIER_INDEX_MIN = 0
SUBCARRIER_INDEX_MAX = 63

# Publishing Configuration
# If SMART_PUBLISHING = False, messages are published every 1 secon
SMART_PUBLISHING = False     # Only publish on significant changes
DELTA_THRESHOLD = 0.05      # Minimum change to trigger publish (0.05 = 5%)
MAX_PUBLISH_INTERVAL_MS = 5000  # Max time between publishes - heartbeat (5 seconds)

# Traffic Generator Configuration
# Generates WiFi traffic to ensure continuous CSI data
TRAFFIC_RATE_MIN = 0          # Minimum rate (0=disabled)
TRAFFIC_RATE_MAX = 1000       # Maximum rate (packets per second)
TRAFFIC_GENERATOR_RATE = 100  # Default rate (packets per second, recommended: 100)
