# 🛜 ESPectre 👻 - Changelog

All notable changes to this project will be documented in this file.

---

## [1.2.0] - In Progress

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
- **Benefits**: 
  - Less penalty for features with high variance
  - Better selection of features with strong signal separation
  - More robust in noisy environments
- **Configurable**: Can be toggled via `USE_MODIFIED_FISHER` flag in `calibration.c`

### 🔧 Changed

- **Feature extraction**: Updated to support all 10 features
- **Calibration system**: Now analyzes all features using Modified Fisher criterion
- **Documentation**: Updated all references from 8 to 10 features

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
