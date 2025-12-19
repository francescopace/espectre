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
MQTT_CLIENT_ID = "micro-espectre"
MQTT_TOPIC = "home/espectre/node1"
MQTT_USERNAME = "mqtt"
MQTT_PASSWORD = "mqtt"

# Traffic Generator Configuration
# Generates WiFi traffic to ensure continuous CSI data
TRAFFIC_GENERATOR_RATE = 100  # Default rate (packets per second, recommended: 100)

# CSI Configuration
CSI_BUFFER_SIZE = 8  # Circular buffer size (used to store csi packets until processed)

# Selected subcarriers for turbulence calculation. None (or comment out) to auto-calibrate at boot using NBVI algorithm.
#SELECTED_SUBCARRIERS = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

# CSI Amplitude Normalization
# Automatically calculated during NBVI calibration
# If baseline > 0.25: scale = 0.25 / baseline_variance (attenuate)
# If baseline <= 0.25: scale = 1.0 (no amplification, prevents extreme values)
NORMALIZATION_SCALE = 1.0      # Current scale (calculated during calibration, default: 1.0)

# NBVI Auto-Calibration Configuration (used when SELECTED_SUBCARRIERS is None)
NBVI_BUFFER_SIZE = 700        # Packets to collect for calibration (7s @ 100Hz, after 3s gain lock)
NBVI_WINDOW_SIZE = 200        # Window size for baseline detection (2s @ 100Hz)
NBVI_WINDOW_STEP = 50         # Step size for sliding window analysis
NBVI_PERCENTILE = 10          # Percentile for baseline detection (10 = p10)
NBVI_ALPHA = 0.5              # NBVI weighting factor (0.5 = balanced)
NBVI_MIN_SPACING = 1          # Minimum spacing between subcarriers (1 = adjacent allowed)
NBVI_NOISE_GATE_PERCENTILE = 10  # Exclude weak subcarriers below this percentile

# Segmentation Window and Threshold
SEG_WINDOW_SIZE = 50          # Moving variance window (packets) - used by both MVS and Features
SEG_THRESHOLD = 1.0           # Motion detection threshold (Lower values = more sensitive to motion)

# Low-pass filter (removes high-frequency noise, reduces false positives)
ENABLE_LOWPASS_FILTER = False   # Recommended: reduces FP in noisy environments
LOWPASS_CUTOFF = 11.0          # Cutoff frequency in Hz (11 Hz: 2.3% FP, 92.4% Recall)
                               # Human movement is typically 0.5-10 Hz, RF noise is >15 Hz

# Hampel filter (removes outliers/spikes in turbulence)
ENABLE_HAMPEL_FILTER = False   # Enable/disable Hampel outlier filter (spikes in turbulence)
HAMPEL_WINDOW = 7             # Window size for median calculation (3-9 recommended)
HAMPEL_THRESHOLD = 4.0        # Outlier detection threshold in MAD units (2.0-4.0 recommended)
                              # Higher values = less aggressive filtering

# CSI Feature Extraction Configuration
ENABLE_FEATURES = True        # Enable/disable CSI feature extraction and confidence calculation