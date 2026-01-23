"""
Micro-ESPectre Configuration

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

# WiFi Configuration
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

# Selected subcarriers for turbulence calculation. None to auto-calibrate at boot.
#SELECTED_SUBCARRIERS = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

# Gain Lock Configuration
# Controls AGC/FFT gain locking for stable CSI amplitudes
# Modes: "auto" (skip if signal too strong), "enabled" (always lock), "disabled" (never lock)
GAIN_LOCK_MODE = "auto"       # Recommended: "auto" - skips gain lock if AGC < 30
GAIN_LOCK_MIN_SAFE_AGC = 30   # Minimum safe AGC value (below this, gain lock is skipped in auto mode)

# Detection Algorithm
# "mvs" (default): Moving Variance Segmentation - fast, good accuracy
# "pca": Principal Component Analysis (Espressif-style) - better noise rejection
DETECTION_ALGORITHM = "mvs"

# Band Calibration Configuration (used when SELECTED_SUBCARRIERS is None)
# For MVS: "nbvi" (default) or "p95"
#   - nbvi: NBVI weighted algorithm (normalization scale, fixed threshold)
#   - p95: P95 moving variance optimization (adaptive threshold)
# For PCA: calibration uses PCACalibrator (fixed subcarrier step)
CALIBRATION_ALGORITHM = "nbvi"
CALIBRATION_BUFFER_SIZE = 700  # Packets to collect for calibration (7s @ 100Hz, after 3s gain lock)

# Segmentation Parameters
# SEG_THRESHOLD can be:
#   - "auto" (default): P95 × 1.4 - minimizes false positives
#   - "min": P100 × 1.0 - maximum sensitivity (may have false positives)
#   - a number (0.1-10.0): fixed manual threshold
SEG_THRESHOLD = "auto"
SEG_WINDOW_SIZE = 50          # Moving variance window (packets) - used by both MVS and Features

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

# HT20 Constants (64 subcarriers - do not change)
NUM_SUBCARRIERS = 64           # HT20: 64 subcarriers
EXPECTED_CSI_LEN = 128         # 64 SC × 2 bytes (I/Q pairs)
GUARD_BAND_LOW = 11            # First valid subcarrier
GUARD_BAND_HIGH = 52           # Last valid subcarrier  
DC_SUBCARRIER = 32             # DC null subcarrier
BAND_SIZE = 12                 # Selected subcarriers for motion detection

# Optional local overrides (config_local.py is gitignored)
try:
    import src.config_local as _local
    for _name in dir(_local):
        if _name.isupper():
            globals()[_name] = getattr(_local, _name)
except ImportError:
    pass