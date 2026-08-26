# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
Micro-ESPectre - Configuration

Author: Francesco Pace <francesco.pace@gmail.com>
"""
import sys

# WiFi Configuration
WIFI_SSID = "YourSSID"
WIFI_PASSWORD = "YourPassword"
# Optional AP lock for mesh/repeater environments.
# Format: "AA:BB:CC:DD:EE:FF" (or without separators).
# WIFI_BSSID = "AA:BB:CC:DD:EE:FF"

# Traffic Generator Configuration
# Generates WiFi traffic to ensure continuous CSI data
CSI_TARGET_PPS = 100  # Requested temporal sensing grid and managed traffic target
TRAFFIC_GENERATOR_ENABLED = True  # False expects an external CSI traffic source
PUBLISH_INTERVAL_MS = 1000    # Time between periodic Direct HTTP/log updates
EVALUATION_INTERVAL_MS = 250  # Time between internal detector evaluations
DEBUG_TELEMETRY = False       # Periodic benchmark-only heap and timing logs
MOTION_HITS_MIN = 1
MOTION_HITS_MAX = 20
MOTION_ON_HITS = 4            # Consecutive evaluated hits required for IDLE -> MOTION
MOTION_OFF_HITS = 3           # Consecutive evaluated hits required for MOTION -> IDLE

# CSI Configuration
CSI_BUFFER_SIZE = 8  # Circular buffer size (used to store csi packets until processed)

# Fixed subcarriers used by the Lightweight detector.
# Subcarriers +/-4, +/-9, +/-14, +/-19, +/-24, +/-28. Spans the full usable range
# because the motion perturbation stays coherent over ~10 subcarriers (3.1 MHz)
# while quiet noise is nearly per-tone independent, so span is what buys
# independent looks. Stops short of |sc| <= 3, where relative jitter rises ~10%.
# See docs/adr/2026-07-25-select-the-classic-band-from-channel-coherence.md.
DEFAULT_SUBCARRIERS = (4, 8, 13, 18, 23, 28, 36, 41, 46, 51, 56, 60)

# Threshold bootstrap configuration (fixed subcarriers, no disk I/O)
CALIBRATION_NUM_WINDOWS = 10

# Segmentation parameters. The runtime resolves this duration to a fixed slot
# count from CSI_TARGET_PPS; observed network jitter never resizes the detector.
SEGMENTATION_WINDOW_SIZE_MS = 1000
CALIBRATION_DURATION_MS = CALIBRATION_NUM_WINDOWS * SEGMENTATION_WINDOW_SIZE_MS

# Detector timing contract, in microseconds. Slot-relative feature lags retain
# these temporal meanings even when individual slots are missing.
L1_DELTA_LAG_US = 100_000          # Profile-displacement lag
TURB_AUTOCORR_LAG_US = 10_000      # Turbulence autocorrelation lag (1 packet at 100 pps)
L1_DELTA_LAG_MAX = 32              # Firmware sizes the profile ring statically
# Storage bounds for the slot count resolved from the temporal window.
SEG_WINDOW_MIN = 1
SEG_WINDOW_MAX = 1000
# Legacy capture diagnostics ignore intervals faster than this when estimating
# an effective source rate. Live detector geometry never depends on that
# estimate; TemporalCsiSampler follows CSI_TARGET_PPS and packet timestamps.
MIN_PLAUSIBLE_PACKET_INTERVAL_US = 200      # 5000 pps

# Low-pass filter (removes high-frequency noise, reduces false positives)
ENABLE_LOWPASS_FILTER = False   # Recommended: reduces FP in noisy environments
LOWPASS_CUTOFF = 11.0          # Cutoff frequency in Hz (11 Hz: 2.3% FP, 92.4% Recall)
                               # Human movement is typically 0.5-10 Hz, RF noise is >15 Hz

# Hampel filter (removes outliers from turbulence and L1-delta streams)
ENABLE_HAMPEL_FILTER = True   # Enabled by default; disable only for explicit resource experiments
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
