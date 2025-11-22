# 🛜 ESPectre 👻 - Changelog

All notable changes to this project will be documented in this file.

---

## [1.3.1] - in progress

### 🎵 Added - Wi-Fi Theremin Sonification Client

**Real-time audio sonification of Wi-Fi CSI data**

New `espectre-theremin.html` transforms CSI data into sound using Web Audio API:

**Core Features:**
- **Three modes**: Continuous (glissando), Quantized (musical notes), Hybrid
- **Musical scales**: Pentatonic, Major, Minor, Chromatic
- **60Hz interpolation**: Smooth audio between 1Hz MQTT updates
- **MQTT WebSocket**: Real-time data streaming

**Feature-Based Modulation:**
All 10 CSI features modulate audio parameters in real-time:
- **Waveform** (entropy): Sine/square/sawtooth selection
- **Vibrato** (temporal_delta_mean + spatial_variance): Speed and depth
- **Filter** (kurtosis + skewness): Cutoff and resonance
- **Stereo Pan** (spatial_correlation): Spatial positioning
- **Tremolo** (temporal_delta_variance + spatial_gradient): Volume modulation
- **Auto Scale** (entropy + variance): Dynamic scale selection

**Technical:**
- Web Audio API with oscillator, gain, filter, panner nodes
- Musical note detection (MIDI conversion)
- Single HTML file, no dependencies

### 🔧 Changed - Feature Extraction Always Active

**Simplified feature extraction logic**

Modified feature extraction behavior to be always active when enabled, regardless of segmentation state:

**Previous behavior:**
- Features extracted only during MOTION state
- IDLE state published without features

**New behavior:**
- Features extracted continuously when `features_enabled=true`
- Features published in both IDLE and MOTION states
- Provides continuous data stream for analysis and ML applications

**Benefits:**
- ✅ Continuous feature data for machine learning dataset collection
- ✅ Better baseline characterization during IDLE periods

### 🔄 Migrate ESPectre CLI from Bash to Python

**Cross-platform Python implementation with enhanced user experience**

Completely rewrote the ESPectre CLI tool, migrating from Bash to Python for better cross-platform compatibility and improved user experience:

**New Python Implementation (`espectre-cli.py`):**
- **Cross-platform support**: Works on Linux, macOS, and Windows without platform-specific dependencies
- **Modern UI**: Interactive prompt using `prompt_toolkit` with:
  * Styled command prompt with color-coded output
  * Nested autocompletion for commands and parameters
  * Syntax highlighting for YAML responses using Pygments
- **Pure Python MQTT**: Uses `paho-mqtt` library instead of external `mosquitto_pub/sub` tools


## [1.3.0] - 2025-11-22

### 🚀 ESP32-C6 Platform Support

**Complete multi-platform support with ESP32-C6 and WiFi 6**

Added comprehensive support for ESP32-C6 alongside ESP32-S3, enabling WiFi 6 (802.11ax):

**ESP32-C6 Implementation:**
- **CSI Configuration**: Implemented `wifi_csi_acquire_config_t` structure
  * `.acquire_csi_legacy = 1` - Captures CSI from 802.11a/g packets (L-LTF)
  * `.acquire_csi_ht20 = 1` - Captures CSI from 802.11n HT20 packets (64 subcarriers)
  * `.acquire_csi_ht40 = 0` - Captures CSI from 802.11n HT40 packets (128 subcarriers) - to be tested
  * `.acquire_csi_su = 1` - Captures CSI from WiFi 6 Single-User packets
  * Critical: Both `acquire_csi_legacy` and `acquire_csi_ht20` required for callback invocation

- **WiFi 6 Support**: Enabled 802.11ax protocol for ESP32-C6
  * Automatic negotiation with WiFi 6 routers
  * Backward compatible with WiFi 4/5 routers
  * Improved packet scheduling and efficiency

**Configuration Files:**
- `sdkconfig.defaults.esp32c6`: ESP32-C6 specific configuration
  * DIO flash mode (more stable than QIO)
  * USB stability improvements (PM disabled)
  * Simplified CSI API configuration

**Critical Fix - CSI Callback Issue:**
- **Root cause**: Incomplete CSI configuration (only `.enable = 1` was insufficient)
- **Solution**: Added all required `acquire_csi_*` fields
- **Result**: CSI callback now working, 50-100+ packets/second

**Testing Infrastructure:**
- **Added `real_csi_data_esp32_c6.h`**: 2015 lines of real CSI test data
- **Renamed `real_csi_data.h`** → `real_csi_data_eps32_s3.h` for clarity
- **Added `segmentation_analysis_c6.png`**: Visual analysis of C6 performance
- **Rewrote `test_segmentation_local.py`**: Complete rewrite for better parameter analysis
  * Grid search optimization
  * Visual segmentation analysis
  * Platform-specific parameter validation

**Documentation:**
- Removed ESP32-S3 specific references from user-facing documentation
- Updated platform badges and hardware requirements
- Added platform comparison tables
- Clarified platform-specific features and limitations
- Updated **ESP32-PLATFORM-SUPPORT.md** with platform-specific defaults

**Reference:** ESP-IDF Issue #14271 - https://github.com/espressif/esp-idf/issues/14271

### 🔧 Runtime-Configurable Parameters

**Major system configurability improvements**

Transformed hardcoded parameters into runtime-configurable settings, enabling fine-tuning without recompilation:

**Segmentation Parameters (MQTT-configurable):**
- **Threshold**: Direct value setting (0.5-10.0)
- **K factor**: Threshold sensitivity multiplier (0.5-5.0)
  * Higher values = less sensitive (fewer false positives)
  * Lower values = more sensitive (better detection of subtle movements)
- **Window size**: Moving variance window (3-50 packets)
  * Smaller = more reactive, larger = more stable
- **Min segment length**: Minimum motion duration (5-100 packets)
- **Max segment length**: Maximum motion duration (10-200 packets)
- **Platform-specific defaults**: Optimized separately for ESP32-S3 and ESP32-C6

**Subcarrier Selection (runtime-configurable):**
- Dynamic subcarrier selection for feature extraction
- New API: `csi_set_subcarrier_selection()`
- Configurable via MQTT with array of indices (0-63)
- Allows optimization for different environments and interference patterns

**New MQTT Commands:**
- `segmentation_threshold <value>` - Set detection threshold
- `segmentation_k_factor <value>` - Set threshold sensitivity
- `segmentation_window_size <value>` - Set moving variance window
- `segmentation_min_length <value>` - Set minimum segment length
- `segmentation_max_length <value>` - Set maximum segment length

**Enhanced Stats Command:**
- Now displays all configurable parameters
- Shows current subcarrier selection
- Provides real-time configuration overview

**Benefits:**
- ✅ No recompilation needed for parameter tuning
- ✅ Easy optimization for different environments
- ✅ Platform-specific defaults (ESP32-S3 vs ESP32-C6)
- ✅ All parameters saved to NVS automatically
- ✅ Simplified testing and validation workflow

### ✨ Added - System Resource Monitoring

**Real-time CPU and RAM usage in stats command**

- Added `cpu_usage_percent` and `heap_usage_percent` fields to `stats` command response
- Calculated using FreeRTOS runtime statistics and ESP-IDF heap APIs
- Minimal overhead (< 0.1% CPU, ~150 bytes RAM)
- Requires FreeRTOS runtime stats enabled in sdkconfig (added to all platform configs)
- Web UI updated to display CPU and RAM in statistics modal

**Real-world performance (ESP32-C6 with all filters + features):**
- CPU: 5.4%, Heap: 22.3% - confirms excellent resource efficiency

### 🌐 Web-Based Real-Time Monitor

**Modern web interface for ESPectre monitoring and configuration**

New `espectre-monitor.html` provides a comprehensive web-based alternative to the CLI shell:

**Features:**
- **Real-time visualization**: Live chart of movement and threshold values
- **Interactive metrics**: State, movement, threshold, and segment counters
- **Complete configuration**: All detection parameters and filters controllable via web UI
  * Segmentation threshold (0.5-10.0)
  * Traffic generator rate (0-50 pps)
  * Features extraction toggle
  * Smart publishing toggle
  * All filters (Hampel, Savitzky-Golay, Butterworth, Wavelet) with parameters
- **Device information**: Displays ESP32 IP address
- **Statistics viewer**: Runtime statistics in modal popup
- **Auto-sync**: Automatically loads current configuration on connection
- **Factory reset**: Web-based factory reset with confirmation

**Technical details:**
- Single HTML file (no dependencies except CDN libraries)
- MQTT.js for WebSocket communication
- Chart.js for real-time data visualization
- Responsive design with collapsible sections
- Toast notifications for command feedback

**Benefits:**
- ✅ No terminal required - works in any modern browser
- ✅ Visual feedback and easier parameter tuning
- ✅ Multi-device support (can monitor multiple ESPectre nodes)
- ✅ Cross-platform (works on desktop, tablet, mobile)
- ✅ Can replace `espectre-cli.sh` for most use cases

### 🔧 MQTT Data Structure Optimization

**Simplified and standardized MQTT message format**

Optimized MQTT data structure for consistency and reduced bandwidth:

**Periodic Data (published during detection):**
- ❌ Removed `segments_total` (not needed for motion detection)
- ✅ Kept essential fields: `movement`, `threshold`, `state`, `features` (optional), `timestamp`

**Stats Command Response:**
- ❌ Removed entire `segments` object (total, active, last_completed)
- ✅ Renamed `moving_variance` → `movement` (consistent with periodic data)
- ✅ Renamed `adaptive_threshold` → `threshold` (consistent with periodic data)
- ✅ Kept `turbulence` for diagnostics
- ✅ Simplified to essential runtime metrics only

**Code Cleanup:**
- ❌ Removed `segment_t` structure from `segmentation.h`
- ❌ Removed segment array and tracking logic from `segmentation.c` (~150 lines)
- ❌ Removed functions: `segmentation_get_num_segments()`, `segmentation_get_segment()`, `segmentation_clear_segments()`, `segmentation_get_active_segments_count()`, `segmentation_get_last_completed_segment()`
- ✅ Simplified state machine to focus only on IDLE ↔ MOTION transitions

**MQTT Handler Simplification:**
- Removed `mqtt_publish_calibration_status()` function
- Removed `mqtt_publish_calibration_complete()` function
- Added `mqtt_publish_binary()` for CSI raw data collection
- Simplified API focused on segmentation-only approach

**Benefits:**
- 📉 Reduced message size and memory usage
- 🔄 Consistent field naming between periodic data and stats
- 🎯 Cleaner API focused on motion detection
- 🧹 Simpler codebase (~200 lines removed)
- 🐛 Fixed bug where last_completed_segment showed stale data after 10 segments

### 🛠️ Enhanced Tools

**Web Monitor:**
- Added controls for all new segmentation parameters
- Real-time parameter adjustment with visual feedback
- Improved configuration synchronization

**CLI (`espectre-cli.sh`):**
- Added commands for segmentation parameter configuration
- Improved interactive menu with new options
- Better parameter validation and feedback

### 📚 Documentation & Cleanup

**Documentation Updates:**
- **CALIBRATION.md**: Added runtime parameter configuration section
- **SETUP.md**: Updated with new MQTT commands and examples
- **ESP32-PLATFORM-SUPPORT.md**: Clarified platform-specific defaults

**Code Cleanup:**
- Removed `.DS_Store` file
- Updated `.gitignore` with better patterns
- Removed obsolete `convert_csi_to_header.py` script

---

## [1.2.1] - 2025-11-17

### 🚀 Improved - Wi-Fi Configuration Optimization

**ESP-IDF best practices implementation for optimal CSI performance**

Based on comprehensive analysis of ESP32-S3 Wi-Fi driver documentation, implemented all recommended optimizations:

- **Power Management**: Disabled Wi-Fi power save mode (`WIFI_PS_NONE`)
  * Minimizes latency in CSI packet reception
  * Critical for real-time movement detection
  * Ensures consistent packet capture rate

- **Country Code Configuration**: Added configurable regulatory domain
  * New `CONFIG_WIFI_COUNTRY_CODE` option in menuconfig
  * Default: "IT" (Italy) in `sdkconfig.defaults`
  * Automatic channel adaptation via `WIFI_COUNTRY_POLICY_AUTO`
  * Driver automatically configures correct channel range per country:
    - US: channels 1-11
    - EU/IT: channels 1-13
    - JP: channels 1-14

- **Protocol Mode**: Explicitly configured 802.11a/g/n
  * Ensures predictable performance on 2.4GHz band
  * Optimal for CSI data collection

- **Bandwidth Configuration**: Set to HT20 (20MHz)
  * Provides stability in high-interference environments
  * Can be changed to HT40 for more subcarriers if needed

**Benefits:**
- ✅ Minimal latency for real-time CSI capture
- ✅ Regulatory compliance for any country
- ✅ Predictable and stable Wi-Fi performance
- ✅ Easy country configuration via menuconfig

**Configuration:**
```bash
idf.py menuconfig
# Navigate: ESPectre Configuration → WiFi Country Code
# Change from "IT" to your country code (US, GB, DE, FR, ES, JP, CN, etc.)
```

**Technical details:**
- Power save mode: `WIFI_PS_NONE` (no modem sleep)
- Country policy: `WIFI_COUNTRY_POLICY_AUTO` (driver adapts channels)
- Protocol: `WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N`
- Bandwidth: `WIFI_BW_HT20` (20MHz)

---

## [1.2.0] - 2025-11-16

### 🏗️ Major Refactoring - Simplified Architecture

**Architectural simplification: Focus on segmentation-only approach**

Removed the complex calibration/detection system in favor of a simpler, more maintainable segmentation-only architecture:

- **Removed modules** (6 files, ~2,140 lines):
  * `calibration.c/h` - Complex calibration system
  * `detection_engine.c/h` - Multi-state detection engine
  * `statistics.c/h` - Statistics buffer and analysis

- **Simplified core files**:
  * `espectre.c`: 900 → 557 lines (-38%, -343 lines)
  * `mqtt_commands.c`: 966 → 604 lines (-37%, -362 lines)
  * Total reduction: ~2,845 lines removed

- **New architecture**:
  ```
  CSI Packet → Segmentation (always) → IF MOTION && features_enabled:
                                          → Extract Features
                                          → Publish with features
                                       ELSE:
                                          → Publish without features
  ```

- **Removed MQTT commands** (8 commands):
  * `detection_threshold` - No longer needed
  * `calibrate` - Calibration system removed
  * `analyze` - Statistics analysis removed
  * `features` - Feature display removed
  * `stats` - Statistics buffer removed
  * `debounce` - Detection debouncing removed
  * `persistence` - Detection persistence removed
  * `hysteresis` - Detection hysteresis removed

- **Added MQTT commands**:
  * `features_enable <on|off>` - Toggle feature extraction during MOTION state

- **Simplified configuration**:
  * Removed: `debounce_count`, `hysteresis_ratio`, `persistence_timeout`, `feature_weights[]`
  * Removed: `threshold_high`, `threshold_low` (detection thresholds)
  * Added: `features_enabled` (bool, default: true)
  * NVS version incremented to 6

**Benefits:**
- ✅ Simpler codebase (~2,845 lines removed, -60% complexity)
- ✅ Easier to understand and maintain
- ✅ Focus on proven MVS algorithm
- ✅ Optional feature extraction (performance optimization)
- ✅ Cleaner MQTT API (10 commands instead of 18)

**MQTT Publishing:**

*IDLE or MOTION without features:*
```json
{
  "movement": 1.85,
  "threshold": 2.20,
  "state": "idle",
  "timestamp": 1730066405
}
```

*MOTION with features_enabled=true:*
```json
{
  "movement": 2.45,
  "threshold": 2.20,
  "state": "motion",
  "features": {
    "variance": 315.5,
    "skewness": 0.85,
    ...
  },
  "timestamp": 1730066405
}
```

### ✨ Added - Moving Variance Segmentation (MVS) Module

**Real-time motion segment detection and analysis**

- **Segmentation module**: New `segmentation.c/h` module for motion segment extraction
  * Implements Moving Variance Segmentation (MVS) algorithm with adaptive threshold
  * Extracts motion segments from CSI data in real-time
  * Calculates statistical features per segment (duration, avg/max turbulence)
  * Circular buffer maintains up to 10 recent segments
  
- **Spatial turbulence calculation**: New `csi_calculate_spatial_turbulence()` function
  * Calculates standard deviation of subcarrier amplitudes
  * Uses selective subcarrier filtering (47-58) matching Python implementation
  * Integrated into main CSI processing loop
  
- **Automatic segment logging**: Real-time segment detection feedback
  * Logs segment number, start index, length, duration
  * Reports average and maximum turbulence values
  * Foundation for advanced motion classification

**Benefits:**
- ✅ Real-time motion segment extraction
- ✅ Adaptive threshold based on moving variance
- ✅ Statistical features per segment for classification
- ✅ Foundation for advanced motion pattern recognition

### 🚀 Major - Amplitude-Based Skewness & Kurtosis

**Performance breakthrough: +151% separation improvement**

Based on analysis of 6 scientific papers on Wi-Fi CSI sensing, implemented amplitude-based approach for skewness and kurtosis:

- **Amplitude-based pipeline**:
  1. Convert raw bytes (I+jQ) → amplitude |h| = √(I² + Q²) for each subcarrier
  2. Aggregate all subcarriers → single amplitude value per packet
  3. Maintain 20-packet circular buffer for temporal analysis
  4. Calculate statistical moments (m2, m3, m4) on amplitude time series
  
- **Shared buffer optimization**:
  * Skewness and kurtosis share same `amplitude_moments_buffer`
  * Cached moments (m2, m4) reused between features
  * Zero memory overhead, ~5% CPU overhead
  
- **Results**:
  * **Skewness**: 2.91x separation (vs 1.16x previous), 82.3% accuracy, 0% false positives
  * **Kurtosis**: 2.47x separation (+79% vs raw kurtosis)
  * **Combined**: Calibrator selects both as dominant features (68% total weight)

### 🚀 Improved - Traffic Generator

**Reliable CSI packet generation with bidirectional traffic**

- **ICMP ping-based**: Replaced UDP broadcast with ICMP Echo Request/Reply
- **ESP-IDF ping component**: Uses official `ping/ping_sock.h` API
- **Bidirectional traffic**: Guaranteed request + reply for CSI generation
- **Auto-discovery**: Automatically targets WiFi gateway
- **Robust implementation**: Thread-safe, tested, maintained by Espressif
- **Statistics**: Success/timeout tracking with callbacks

**Benefits:**
- ✅ Reliable CSI packet generation on every ping reply
- ✅ No external dependencies (uses gateway)
- ✅ Simpler code (~200 lines vs manual ICMP implementation)
- ✅ Better error handling and logging

**Technical details:**
- Previous: UDP broadcast (no reply, unreliable CSI generation)
- Current: ICMP ping to gateway (bidirectional, reliable CSI on reply)
- Configurable rate: 1-50 pps
- Automatic gateway IP discovery from network interface

### 🚀 Improved - CSI Subcarrier Optimization

**Maximum spatial information: Reading ALL available subcarriers**

Based on ESP32-S3 Wi-Fi documentation analysis, optimized CSI data collection to capture complete channel information:

- **Channel filter disabled**: Changed `channel_filter_en` from `true` to `false`
  * Now receives ALL 64 subcarriers instead of 52 (+23% spatial information)
  * Includes edge subcarriers (-32 to -27 and +27 to +32) previously filtered
  * More complete frequency response of the channel
  
**Benefits:**
- ✅ +23% more spatial information (64 vs 52 subcarriers)
- ✅ Better movement detection accuracy
- ✅ More data for calibration optimization
- ✅ Higher spatial resolution

### ✨ Added - Temporal Features

**Enhanced feature set: Expanded from 8 to 10 features**

- **New temporal features**: Added 2 temporal features that track changes between consecutive CSI packets
  - `temporal_delta_mean`: Average absolute difference from previous packet
  - `temporal_delta_variance`: Variance of differences from previous packet
- **Improved detection**: Temporal features capture movement dynamics over time
- **Backward compatible**: Existing calibrations continue to work with the expanded feature set

**Feature set now includes:**
- **Statistical** (5): variance, skewness, kurtosis, entropy, iqr
- **Spatial** (3): spatial_variance, spatial_correlation, spatial_gradient
- **Temporal** (2): temporal_delta_mean, temporal_delta_variance

### 🔧 Changed - Modified Fisher Criterion

**Improved feature selection algorithm**

- **Modified Fisher Score**: Changed from standard Fisher `(μ₁ - μ₂)² / (σ₁² + σ₂²)` to Modified Fisher `(μ₁ - μ₂)² / √(σ₁² + σ₂²)`
- **Pre-normalization**: All features normalized to [0,1] before Fisher calculation
  * Eliminates bias towards features with large absolute values
  * Ensures fair comparison between features
  * Skewness/kurtosis now correctly selected as top features
- **Benefits**: 
  - Less penalty for features with high variance
  - Better selection of features with strong signal separation
  - More robust in noisy environments
- **Configurable**: Can be toggled via `USE_MODIFIED_FISHER` flag in `calibration.c`

### 🧪 Added - Local Segmentation Test Script

**Python tool for rapid parameter tuning without device flashing**

- **test_segmentation_local.py**: Replicates ESP32 segmentation algorithm locally
  * Implements Moving Variance Segmentation (MVS) with adaptive threshold
  * Calculates spatial turbulence from CSI packets
  * Extracts statistical features from motion segments
  * Includes Random Forest classifier for validation
  * Interactive visualization of segmentation results

**Features:**
- **Parameter optimization**: Grid search over 300 combinations (--optimize flag)
- **Configurable parameters**: K_FACTOR, WINDOW_SIZE, MIN_SEGMENT, MAX_SEGMENT at top of file
- **Batch mode**: Skip visualization with --no-plot flag
- **Comprehensive documentation**: Usage examples and parameter descriptions

**Benefits:**
- ✅ Fast iteration without ESP32 flashing
- ✅ Visual feedback for parameter tuning
- ✅ Automatic optimal parameter discovery
- ✅ Validates C code implementation in Python

**Usage:**
```bash
python test_app/test_segmentation_local.py              # Run with defaults
python test_app/test_segmentation_local.py --optimize   # Find optimal parameters
python test_app/test_segmentation_local.py --no-plot    # Skip visualization
```

### ✨ Added - CSI Raw Data Collection

**Dataset generation for testing and analysis**
- **Calibration data export**: Extended `calibrate` command to print CSI raw data during calibration

**Usage:**
```bash
espectre> calibrate start 100 verbose
```

### 🗑️ Removed - Adaptive Normalizer

**Code simplification: Removed adaptive normalizer filter**

The adaptive normalizer has been removed to simplify the codebase and reduce computational overhead:

- **Simplified filter pipeline**:
  ```
  Butterworth (ON) → Wavelet (OFF) → Hampel (OFF) → Savitzky-Golay (ON)
  ```
**Rationale:**
The adaptive normalizer was primarily used for monitoring/debugging and did not directly affect signal processing or motion detection. Its removal simplifies the system while maintaining all core functionality.

---

## [1.1.0] - 2025-11-08

### 🤖 Enhanced - Intelligent Automatic Calibration System

**Major enhancement: Complete system auto-calibration with intelligent filter optimization**

- **Auto-feature selection**: Automatically selects the 4-6 most discriminant features from 8 available
- **Optimal weight calculation**: Uses Fisher's criterion to calculate weights proportional to separability
- **Optimal threshold calculation**: Fisher's optimal threshold minimizes classification error
- **Intelligent filter analysis**: Analyzes signal characteristics to determine optimal filter configuration
- **Automatic filter application**: Applies optimal filters with calculated parameters
- **Sample-based collection**: Uses `duration × traffic_rate` for deterministic sample count
- **Automatic application**: Applies all parameters without manual intervention

**New commands:**
- `calibrate start [duration]` - Start automatic calibration
- `calibrate stop` - Stop calibration
- `calibrate status` - Check calibration progress
- `factory_reset` - Restore all settings to defaults

**Performance improvements:**
- ⚡ 30-40% CPU savings (extracts only 4-6 features instead of 8 after calibration)
- 💾 Reduced RAM usage (no history buffer needed)
- 🎯 Environment-specific optimization

**Algorithms implemented:**
- Welford's algorithm for online statistics
- Fisher's criterion for feature selection
- Fisher's optimal threshold for threshold calculation

### 🔧 Changed

- **Simplified feature set**: Reduced from 15 to 8 features (removed 6 problematic temporal features)
- **Removed history buffer**: No longer needed without temporal features
- **Removed direction analysis**: Required history buffer (not critical for basic detection)
- **Sample-based calibration**: Uses sample count instead of time duration for more reliable collection
- **Simplified weight management**: Removed manual weight modification commands
- **Documentation**: Updated all guides to reflect new calibration system and simplified features

### ✨ Added - Butterworth Low-Pass Filter

**Signal processing improvement from scientific papers**

- **Butterworth IIR filter**: Order 4, cutoff 8Hz (human movement: 0.5-8Hz)
- **Pre-calculated coefficients**: Optimized for ~100 packets/sec sampling rate
- **Default enabled**: Significantly reduces false positives
- **Configurable**: Can be toggled via MQTT/CLI

### ✨ Added - Wavelet Filter (Daubechies db4)

**Advanced denoising for high-noise environments**

- **Daubechies db4 wavelet transform**: Removes low-frequency persistent noise
- **Streaming mode**: Real-time processing with circular buffer (32 samples)
- **Configurable parameters**: Level (1-3), threshold (0.5-2.0), method (soft/hard)
- **Optimized for ESP32**: Minimal memory footprint (~4KB flash, ~2KB RAM)
- **Default disabled**: Enable manually for high-noise environments (variance >500)

### 🛠️ Tools

**CLI Improvements:**
- **Interactive mode**: `espectre-cli.sh` now features an interactive menu-driven interface
- **Easier navigation**: Browse and execute commands without memorizing syntax
- **User-friendly**: Ideal for quick testing and configuration

### 🗑️ Removed

- Manual weight modification commands (`weight_variance`, `weight_spatial_gradient`, `weight_variance_short`, `weight_iqr`)

### 💾 Added - NVS Persistent Storage

**Configuration and calibration persistence**

- **Automatic loading**: All parameters loaded from NVS at boot
- **Automatic saving**: Configuration saved after every MQTT command
- **Calibration persistence**: Calibration results survive reboots

**Benefits:**
- 🔄 No need to recalibrate after reboot
- ⚙️ Configuration persists across power cycles
- 🛡️ Validated data loading prevents corruption
- 🏭 Easy factory reset for troubleshooting

**Data persisted:**
- All calibration results (features, weights, threshold)
- All runtime parameters (filters, thresholds, timeouts)
- All MQTT-configurable settings

### 🏗️ Refactored - Modular Architecture

**Code restructuring: Monolithic file split into specialized modules**

- **Before**: Single `espectre.c`
- **After**: `espectre.c` + 10 specialized modules

**New modules:**
- `mqtt_handler.c/h` - MQTT client and event handling
- `mqtt_commands.c/h` - MQTT command handlers
- `wifi_manager.c/h` - WiFi connection management
- `config_manager.c/h` - Runtime configuration
- `csi_processor.c/h` - CSI feature extraction
- `detection_engine.c/h` - Movement detection logic
- `filters.c/h` - Signal filtering pipeline
- `statistics.c/h` - Statistical analysis
- `nvs_storage.c/h` - NVS persistence
- `calibration.c/h` - Calibration system

### 📚 Documentation

- Updated `CALIBRATION.md` with automatic calibration section
- Updated `SETUP.md` with calibrate command
- Updated `README.md` to mention auto-calibration
- Added `CHANGELOG.md` to track changes

---

## [1.0.0] - 2025-11-01

### 🎉 Initial Release

**Complete CSI-based movement detection system for ESP32-S3**

### ✨ Features

**Signal Processing Pipeline:**
- **Hampel filter**: Outlier removal using MAD (Median Absolute Deviation)
- **Savitzky-Golay filter**: Polynomial smoothing
- **Adaptive normalization**: Running statistics with Welford's algorithm

**Feature Extraction (15 features):**
- **Time-domain** (6): Mean, Variance, Skewness, Kurtosis, Entropy, IQR
- **Spatial** (3): Spatial variance, correlation, gradient
- **Temporal** (3): Autocorrelation, zero-crossing rate, peak rate
- **Multi-window** (3): Variance at short/medium/long time scales

**Detection System:**
- **4-state detection**: IDLE, MICRO, DETECTED, INTENSE
- **Debouncing**: Configurable consecutive detections
- **Persistence**: Configurable timeout before state downgrade
- **Hysteresis**: Prevents state flickering

**Communication:**
- **MQTT publishing**: JSON messages with movement data
- **Smart publishing**: Reduces traffic by publishing only significant changes
- **Runtime configuration**: All parameters adjustable via MQTT commands

**Tools:**
- **CLI script** (`espectre-cli.sh`): Easy command-line control
- **MQTT commands**: Complete remote configuration
- **Serial monitoring**: Real-time debugging

### 📊 Performance

- **CSI capture rate**: 10-100 packets/second
- **Processing latency**: <50ms per packet
- **MQTT bandwidth**: ~0.2-1 KB/s
- **Power consumption**: ~500mW typical
- **Detection range**: 3-8 meters optimal

### 🛠️ Technical Stack

- **Framework**: ESP-IDF v6.1
- **Language**: C
- **Target**: ESP32-S3 (16MB Flash, 8MB PSRAM)
- **Protocol**: MQTT over Wi-Fi 2.4GHz
