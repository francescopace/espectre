"""
Micro-ESPectre - Configuration

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""
import sys

# WiFi Configuration
WIFI_SSID = "YourSSID"
WIFI_PASSWORD = "YourPassword"
# Optional AP lock for mesh/repeater environments.
# Format: "AA:BB:CC:DD:EE:FF" (or without separators).
# WIFI_BSSID = "AA:BB:CC:DD:EE:FF"

# MQTT Configuration
MQTT_ENABLED = True
MQTT_BROKER = "homeassistant.local"  # Your MQTT broker IP
MQTT_PORT = 1883
MQTT_CLIENT_ID = "micro-espectre"
MQTT_TOPIC_PREFIX = "espectre/v1/devices"
MQTT_USERNAME = "mqtt"
MQTT_PASSWORD = "mqtt"

# Traffic Generator Configuration
# Generates WiFi traffic to ensure continuous CSI data
TRAFFIC_GENERATOR_RATE = 100  # Target valid CSI rate (packets per second)
TRAFFIC_GENERATOR_ADAPTIVE = True  # Adjust send pacing from CSI feedback and socket errors
TRAFFIC_GENERATOR_MODE = "ping"  # Default mode: "ping" or "dns"
PUBLISH_INTERVAL = 100        # Packets between periodic MQTT/log updates
EVALUATION_INTERVAL = 25      # Packets between internal detector evaluations
MOTION_ON_HITS = 4            # Consecutive evaluated hits required for IDLE -> MOTION
MOTION_OFF_HITS = 3           # Consecutive evaluated hits required for MOTION -> IDLE

# CSI Configuration
CSI_BUFFER_SIZE = 8  # Circular buffer size (used to store csi packets until processed)

# Fixed subcarriers shared by Classic and ML detectors.
DEFAULT_SUBCARRIERS = (14, 17, 20, 23, 26, 29, 35, 38, 41, 44, 47, 50)

# Detection Algorithm
# "classic" (default): weighted L1-delta + turbulence-autocorrelation fusion
# "ml": Neural Network - learned patterns, trained default threshold
DETECTION_ALGORITHM = "classic"

# Threshold bootstrap configuration (fixed subcarriers, no disk I/O)
CALIBRATION_NUM_WINDOWS = 10   # Number of windows worth of packets to collect
# CALIBRATION_BUFFER_SIZE calculated after SEG_WINDOW_SIZE is defined

# Segmentation Parameters
SEG_WINDOW_SIZE = 100         # Shared detector window (packets) - used by Classic and Features

# Calibration buffer size = number of windows * window size
CALIBRATION_BUFFER_SIZE = CALIBRATION_NUM_WINDOWS * SEG_WINDOW_SIZE

# Low-pass filter (removes high-frequency noise, reduces false positives)
ENABLE_LOWPASS_FILTER = False   # Recommended: reduces FP in noisy environments
LOWPASS_CUTOFF = 11.0          # Cutoff frequency in Hz (11 Hz: 2.3% FP, 92.4% Recall)
                               # Human movement is typically 0.5-10 Hz, RF noise is >15 Hz

# Hampel filter (removes outliers from turbulence and L1-delta streams)
ENABLE_HAMPEL_FILTER = True    # Enable/disable Hampel preprocessing for all detector feature streams
HAMPEL_WINDOW = 7             # Window size for median calculation (3-11)
HAMPEL_THRESHOLD = 5.0        # Outlier detection threshold in MAD units (2.0-6.0 recommended)
                              # Higher values = less aggressive filtering

# HT20 Constants (64 subcarriers - do not change)
NUM_SUBCARRIERS = 64           # HT20: 64 subcarriers
EXPECTED_CSI_LEN = 128         # 64 SC × 2 bytes (I/Q pairs)
GUARD_BAND_LOW = 11            # First valid subcarrier
GUARD_BAND_HIGH = 52           # Last valid subcarrier  
DC_SUBCARRIER = 32             # DC null subcarrier
BAND_SIZE = len(DEFAULT_SUBCARRIERS)  # Selected subcarriers for motion detection

# Optional local overrides (config_local.py is gitignored)
# Skip local overrides only under pytest to keep tests hermetic.
if "pytest" not in sys.modules:
    try:
        import src.config_local as _local
    except ImportError:
        try:
            import config_local as _local
        except ImportError:
            _local = None
    if _local is not None:
        for _name in dir(_local):
            if _name.isupper() and _name != "DEFAULT_SUBCARRIERS":
                globals()[_name] = getattr(_local, _name)
