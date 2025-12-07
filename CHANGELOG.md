# 🛜 ESPectre 👻 - Changelog

All notable changes to this project will be documented in this file.

---

## [2.0.1] - in progress

### ⚡ Performance Optimization

**Unified variance algorithm and optimized Hampel filter across both platforms**

This release focuses on code uniformity between MicroPython and C++ implementations, improving numerical stability and performance.

#### Algorithm Uniformity
- **Two-pass variance**: Both platforms now use the same numerically stable algorithm
  - Formula: `Var(X) = Σ(x - μ)² / n` (more stable than `E[X²] - E[X]²`)
  - Eliminates catastrophic cancellation risk with float32
  - Identical behavior between MicroPython and C++

#### Hampel Filter Optimization
- **C++ (ESPHome)**: Eliminated dynamic memory allocation
  - Pre-allocated static buffers in `hampel_turbulence_state_t`
  - Insertion sort replaces `qsort()` for small arrays (N=3-11)
  - **~20-25μs saved per packet** (no malloc/free overhead)
  
- **MicroPython**: Pre-allocated buffers and circular buffer
  - Eliminated list creation per call
  - Insertion sort for small arrays
  - **~120μs saved per packet**

#### Validation
- New test script `16_test_optimization_equivalence.py` using real CSI data
- Verified with 2000 real CSI packets (baseline + movement)
- Maximum variance difference: 9.41e-14 (effectively zero)

| Change | C++ Impact | MicroPython Impact |
|--------|------------|-------------------|
| Two-pass variance | Unchanged (already used) | +25μs (acceptable) |
| Hampel optimization | -20-25μs | -120μs |
| **Net improvement** | **-20-25μs/pkt** | **-95μs/pkt** |

### 🧪 Test Suite & Code Coverage

- **140 test cases** (+72 from 2.0.0) with real CSI data
- **Full device testing**: All tests run on both native and ESP32-C6 via `IWiFiCSI` dependency injection
- **Codecov integration**: Coverage badge, PR comments, 80% threshold
- **82% line coverage**, 98% function coverage
- **Refactoring**: Shared utilities in `utils.h`, configurable `CalibrationManager`

---

## [2.0.0] - 2025-12-06

### 🚀 Major - ESPHome Native Integration

**Complete platform migration from ESP-IDF to ESPHome**

This release represents a major architectural shift from standalone ESP-IDF firmware to a native ESPHome component, enabling seamless Home Assistant integration.

> ⚠️ **Note**: Extensively tested on ESP32-S3 and ESP32-C6, but bugs may still exist. Community contributions, bug reports, and support for additional ESP32 variants are welcome!

### 🎯 Two-Platform Strategy

**ESPectre now follows a dual-platform development model:**

| Platform | Role | Focus | Target |
|----------|------|-------|--------|
| **ESPectre** (ESPHome - C++) | Production | Motion detection only | Smart home users, Home Assistant |
| **Micro-ESPectre** (Micro Python) | R&D | Features + Filters for research | Researchers, developers, academics |

**ESPectre** focuses on core motion detection for Home Assistant integration.
**Micro-ESPectre** provides features (variance, skewness, kurtosis, entropy, IQR, spatial_*, temporal_*) and advanced filters (Butterworth, Wavelet, Savitzky-Golay) for research/ML applications.

**New Architecture:**
- **Native ESPHome component**: Full C++ implementation as ESPHome external component
- **Home Assistant auto-discovery**: Automatic device and sensor registration via Native API
- **YAML configuration**: All parameters configurable via simple YAML files
- **OTA updates**: Wireless firmware updates via ESPHome

**Implementation:**
- `components/espectre/`: Complete ESPHome component with Python config and C++ implementation
- Modular C++ architecture: `calibration_manager`, `csi_manager`, `sensor_publisher`, etc.
- Binary sensor for motion detection
- Movement score sensor
- Adjustable threshold (number entity) - controllable from Home Assistant

### 🔄 Micro-ESPectre

**R&D Platform for Wi-Fi CSI Motion Detection - Pure Python implementation for MicroPython**

Micro-ESPectre is the research and development platform of the ESPectre project, designed for rapid prototyping, algorithmic experimentation, and academic/industrial research. It implements motion detection algorithms in pure Python, enabling fast iterations without compilation overhead.

**Key Features:**
- ⚡ **Instant Deploy**: ~5 seconds to update code (no compilation)
- 🔧 **MQTT Integration**: Runtime configuration via MQTT commands
- 🧬 **Auto Calibration Algorithm**: Automatic subcarrier selection (F1=97.1%)
- 📊 **Analysis Tools**: Complete suite for CSI analysis and algorithm optimization
- 🧮 **Feature Extraction**: Statistical features (variance, skewness, kurtosis, entropy, IQR)
- 🎯 **Confidence Score**: Experimental motion detection confidence estimation
- 💾 **NVS Persistence**: Persistent configuration on filesystem

**Advanced Applications (ML/DL ready):**
- 🔬 People counting
- 🏃 Activity recognition (walking, falling, sitting, sleeping)
- 📍 Localization and tracking
- 👋 Gesture recognition

**Dependencies:** 
- [`micropython-esp32-csi`](https://github.com/francescopace/micropython-esp32-csi) - Custom MicroPython fork with native CSI support for ESP32 family 
- MQTT broker (e.g., Mosquitto)

### 🧪 Test Suite Refactoring

**Migration from Unity (ESP-IDF) to PlatformIO Unity for ESPHome consistency**

The test suite has been migrated from ESP-IDF's Unity framework to PlatformIO Unity, aligning with the ESPHome ecosystem and enabling native (desktop) test execution without hardware.

**Complete test suite with 68 test cases organized in 5 suites and Memory leak detection:**

| Suite | Tests | Focus |
|-------|-------|-------|
| `test_csi_processor` | 19 | API, initialization, validation, memory management |
| `test_hampel_filter` | 16 | Outlier removal filter behavior |
| `test_calibration` | 21 | NBVI algorithm, variance, percentile calculations |
| `test_calibration_file_storage` | 9 | Calibration persistence and file I/O |
| `test_motion_detection` | 3 | MVS performance with real CSI data (2000 packets) |

```bash
# Run tests locally
cd test && pio test -e native
```

### 🔄 CI/CD Pipeline

**GitHub Actions integration for automated quality assurance**

- **Automated testing**: Runs on push to `main`/`develop` and pull requests
- **ESPHome build verification**: Compiles `espectre.yaml` to validate component
- **Status badge**: Real-time CI status displayed in README
- **Path filtering**: Only triggers on changes to `components/espectre/` or `test/`

---

## [1.5.0] - 2025-12-03

### 🧬 Automatic Subcarrier Selection
- Zero-configuration subcarrier selection using NBVI (Normalized Baseline Variability Index) algorithm. 
- Auto-calibration at boot, re-calibration after factory_reset.
- Formula: `NBVI = 0.3 × (σ/μ²) + 0.7 × (σ/μ)`. 
- Achieves F1=97.1% (-0.2% gap to manual). 

---

## [1.4.0] - 2025-11-28

### 🏗️ Major Refactoring
- **Feature extraction module**: Extracted to `csi_features.c/h`, reduced `csi_processor.c` by 50%
- **Configuration centralization**: All defaults in `espectre.h`, validation in `validation.h/c`
- **Two-pass variance**: Numerically stable calculation
- **Traffic generator**: Max rate 1000 pps (was 50), default 100 pps
- **CLI migration**: Bash → Python (cross-platform)
- **Wi-Fi Theremin**: `espectre-theremin.html` for CSI sonification
- **Removed**: Redundant segmentation parameters (min_length, max_length, k_factor)

---

## [1.3.0] - 2025-11-22

### 🚀 ESP32-C6 Platform Support
- **WiFi 6 (802.11ax)** support with proper CSI configuration
- **Runtime-configurable parameters**: threshold, window_size via MQTT
- **Web Monitor**: `espectre-monitor.html` with real-time visualization
- **System monitoring**: CPU/RAM usage in stats command
- **MQTT optimization**: Simplified message format, removed segment tracking

---

## [1.2.1] - 2025-11-17

### 🔧 Wi-Fi Optimization
ESP-IDF best practices: disabled power save (`WIFI_PS_NONE`), configurable country code, HT20 bandwidth.

---

## [1.2.0] - 2025-11-16

### 🏗️ Simplified Architecture
- **MVS algorithm**: Moving Variance Segmentation with adaptive threshold
- **Amplitude-based features**: +151% separation improvement for skewness/kurtosis
- **Traffic generator**: ICMP ping-based (was UDP broadcast)
- **64 subcarriers**: All available (was 52 filtered)
- **10 features**: Added temporal_delta_mean, temporal_delta_variance

---

## [1.1.0] - 2025-11-08

### 🤖 Auto-Calibration System
- **Fisher's criterion**: Automatic feature selection (4-6 from 8)
- **Butterworth filter**: Order 4, cutoff 8Hz
- **Wavelet filter**: Daubechies db4 for high-noise environments
- **NVS persistence**: Configuration survives reboots
- **Modular architecture**: Split into 10 specialized modules

---

## [1.0.0] - 2025-11-01

### 🎉 Initial Release
CSI-based movement detection for ESP32-S3. Hampel + Savitzky-Golay filters, 15 features, 4-state detection (IDLE/MICRO/DETECTED/INTENSE), MQTT publishing, CLI tool. 10-100 pps, <50ms latency, 3-8m range.

---

## 📄 License

GPLv3 - See [LICENSE](LICENSE) for details.
