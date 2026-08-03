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
MQTT_HA_DISCOVERY_ENABLED = False
MQTT_HA_DISCOVERY_PREFIX = "homeassistant"

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
# Subcarriers +/-4, +/-9, +/-14, +/-19, +/-24, +/-28. Spans the full usable range
# because the motion perturbation stays coherent over ~10 subcarriers (3.1 MHz)
# while quiet noise is nearly per-tone independent, so span is what buys
# independent looks. Stops short of |sc| <= 3, where relative jitter rises ~10%.
# See docs/adr/2026-07-25-select-the-classic-band-from-channel-coherence.md.
DEFAULT_SUBCARRIERS = (4, 8, 13, 18, 23, 28, 36, 41, 46, 51, 56, 60)

# Detection Algorithm
# "classic" (default): weighted L1-delta + turbulence-autocorrelation fusion
# "ml": Neural Network - learned patterns, trained default threshold
DETECTION_ALGORITHM = "classic"

# Threshold bootstrap configuration (fixed subcarriers, no disk I/O)
CALIBRATION_NUM_WINDOWS = 10   # Number of windows worth of packets to collect
# CALIBRATION_BUFFER_SIZE calculated after SEG_WINDOW_SIZE is defined

# Segmentation Parameters
SEG_WINDOW_SIZE = 100         # Shared detector window (packets) - used by Classic and Features

# Detector timing contract, in microseconds. The packet counts above and in
# csi_features.py are the values these durations resolve to at the nominal
# 100 pps; on a stream that runs faster or slower, the counts are re-derived
# from the measured cadence so a window keeps spanning the same physical time.
# Feature values depend on the interval they are measured over, not on how many
# packets happen to land in it, so the durations are the contract.
SEG_WINDOW_US = 1_000_000          # Detector window span
EVALUATION_INTERVAL_US = 250_000   # Time between detector evaluations
L1_DELTA_LAG_US = 100_000          # Profile-displacement lag
TURB_AUTOCORR_LAG_US = 10_000      # Turbulence autocorrelation lag (1 packet at 100 pps)
L1_DELTA_LAG_MAX = 32              # Firmware sizes the profile ring statically
# Window bounds, shared by configured and rate-derived windows alike. The
# floor is measured: under ~100 samples the window features get noisy enough
# that startup calibration lifts the threshold and recall collapses (a
# 25-sample window scores 60.2% against 98.7% at 100). The ceiling is a stack
# bound in the C++ runtime, where the autocorrelation and MAD scratch arrays
# live on the CSI callback stack.
SEG_WINDOW_MIN = 100
SEG_WINDOW_MAX = 200
RATE_ADAPTATION_DEAD_BAND = 0.25   # Treat 80-133 pps as nominal and do not adapt
# A cadence faster than this is not a CSI stream, it is a batch delivered
# faster than real time. Elapsed time is only trusted above it; below, the
# packet-count fallback covers. There is no upper bound because a stream
# slower than one window is already handled as a hole by SEG_WINDOW_US.
MIN_PLAUSIBLE_PACKET_INTERVAL_US = 200      # 5000 pps

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
GUARD_BAND_LOW = 4             # First valid subcarrier (-28)
GUARD_BAND_HIGH = 60           # Last valid subcarrier (+28)
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
