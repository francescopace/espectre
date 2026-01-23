# Changelog

All notable changes to this project will be documented in this file.

---

## [2.4.0] - in progress

### Algorithm Changes

#### Dual Detection Algorithms (MVS + PCA)

Two motion detection algorithms are now available via the `IDetector` interface:

| Algorithm | Method | Subcarrier Selection | Best For |
|-----------|--------|---------------------|----------|
| **MVS** | Moving Variance of Turbulence | Calibrated (12 best) | Default, most environments |
| **PCA** | PCA + Correlation (Espressif style) | Fixed step (16) | High-noise environments |

**Configuration:**

ESPHome (YAML):
```yaml
espectre:
  detection_algorithm: mvs  # default, or "pca"
```

Python (Micro-ESPectre):
```python
from mvs_detector import MVSDetector
from pca_detector import PCADetector

detector = MVSDetector(window_size=50, threshold=1.0)
# or
detector = PCADetector()
```

**Architecture:**
- New `IDetector` interface (`detector_interface.h/.py`) for polymorphic detection
- `MVSDetector` and `PCADetector` implement the interface
- Legacy `csi_processor` module removed in favor of modular detectors
- `csi_manager` now uses `IDetector*` for algorithm-agnostic processing

#### Dual Band Selection Algorithms (P95 + NBVI)

Two automatic subcarrier selection algorithms are now available:

| Algorithm | Selection | Best For |
|-----------|-----------|----------|
| **NBVI** | 12 non-consecutive subcarriers | Default, spectral diversity |
| **P95** | 12 consecutive subcarriers | Simpler, consistent selection |

Both algorithms achieve similar performance (~95% Recall, <1% FP Rate).

**Configuration:**

ESPHome (YAML):
```yaml
espectre:
  segmentation_calibration: nbvi  # default, or "p95"
```

Python (Micro-ESPectre):
```python
# In config.py
CALIBRATION_ALGORITHM = "nbvi"  # default, or "p95"
```

#### Centralized Adaptive Threshold

The adaptive threshold calculation is now centralized in `threshold.py` (Python) and `threshold.h` (C++):

- Calibrators return `(selected_band, mv_values)` instead of threshold
- Threshold is calculated as `Pxx × factor` after band selection
- Cleaner separation of concerns: calibrators do band selection only

**Threshold modes:**
| Mode | Formula | Description |
|------|---------|-------------|
| `auto` | P95 × 1.4 | Minimizes false positives (default) |
| `min` | P100 × 1.0 | Maximum sensitivity |

#### HT20-Only Mode (64 Subcarriers)

Simplified to WiFi 4 (802.11n) HT20 mode exclusively for stable 64 subcarriers across all ESP32 variants.

- Fixed 64 SC for consistent performance on all chips (C3, C6, S3, original ESP32)
- Centralized constants in `csi_processor.h` (C++) and `config.py` (Python)
- Reduced memory footprint with optimized buffer sizes

### Testing

#### Multi-Chip Test Suite (C6 + S3)

Unit tests now run on both ESP32-C6 and ESP32-S3 with dedicated datasets and chip-specific parameters:

| Parameter | ESP32-C6 | ESP32-S3 |
|-----------|----------|----------|
| Window Size | 50 | 100 |
| Subcarriers | [11-22] | [48-59] |
| Hampel Filter | OFF | ON |
| FP Rate Target | <10% | <15% |

**Results:**
- **C6:** 98.8% Recall, 0.0% FP Rate, 99.4% F1-Score
- **S3:** 99.1% Recall, 14.3% FP Rate, 92.9% F1-Score

S3 dataset requires Hampel filter and larger window to handle higher baseline noise. Both C++ (`pio test`) and Python (`pytest`) test suites validate performance.

### Features

#### Optional Segmentation Threshold

The `segmentation_threshold` parameter is now optional in YAML configuration.

- **Default**: Adaptive threshold calculated as `P95 × 1.4` during calibration
- **Manual override**: Specify value in YAML to disable adaptive threshold

#### Calibrate Switch

New Home Assistant switch for triggering band recalibration without reflashing.

- `switch.espectre_calibrate`: Turn ON to start, automatically turns OFF when complete
- Useful for recalibrating after room layout changes

#### Other Changes

- **ESP32-C3 Development Config**: Added `espectre-c3-dev.yaml`
- **Lower Threshold Minimum**: Lowered from 0.5 to 0.1 for high-sensitivity applications

### Micro-ESPectre (R&D Platform)

#### Refactored Calibrator Architecture

The calibrator modules have been reorganized for better maintainability:

| File | Purpose |
|------|---------|
| `p95_calibrator.py` | P95 band selection (renamed from `band_calibrator.py`) |
| `nbvi_calibrator.py` | NBVI band selection (restored) |
| `threshold.py` | Centralized adaptive threshold calculation |
| `utils.py` | Common utility functions (percentile, variance, etc.) |

#### SHA256 Firmware Verification

New `verify` command to detect outdated firmware causing CSI collection failures.

- **Automatic hash check**: Compares deployed vs. source file hashes
- **Clear warnings**: Identifies mismatched files that need redeployment

#### CSI Stream Protocol v2

- **Chip type in header**: Auto-detected from device, no manual `--chip` option needed
- **Contributor tracking**: Auto-detected from `git config user.name`
- **File naming**: `{label}_{chip}_{num_sc}sc_{timestamp}.npz` format

### Bug Fixes

- **ESP32-C3 boot crash**: Fixed duplicate `register_component` calls
- **USB Serial JTAG**: Use correct ESPHome macro for detection
- **CSI data overflow**: Limit to 128 bytes to prevent `num_sc` overflow
- **Game mobile view**: Fixed visualization issues on mobile devices

### CI/CD

#### NPZ Data Loading with cnpy

C++ tests now load CSI data directly from NPZ files using [cnpy](https://github.com/rogersce/cnpy), eliminating the need for duplicate data in C header files.

- **Same data source**: Both Python and C++ tests use the same NPZ files in `micro-espectre/data/`
- **ZIP64 support**: Added ZIP64 extended information parsing to cnpy for NumPy compatibility
- **Compressed NPZ**: Added `npz_save_compressed()` function with zlib deflate compression
- **Automatic path detection**: Test loader finds files from any working directory

> Note: cnpy improvements submitted upstream as [PR](https://github.com/rogersce/cnpy/pull/103)

#### Multi-Chip Testing

Tests run with real CSI data from multiple ESP32 chips (C6, S3) using 64 SC datasets.

- **Parameterized tests**: Both C++ and Python tests iterate over available chip datasets
- **Centralized test data**: `csi_test_data.h` manages chip-specific data loading
- **Guard band validation**: Uses centralized `HT20_*` constants from `csi_processor.h`

#### Smoke Tests (QEMU)

Added automated smoke tests using QEMU emulation to catch firmware crashes early.

- **Supported chips**: ESP32-S3, ESP32-C3, ESP32-C6
- **Detection**: Kernel panics, Guru Meditation errors, assertion failures, stack smashing
- **Stack trace analysis**: Automatic `addr2line` decoding on crash for easier debugging
- **Architecture coverage**: Both Xtensa (S3) and RISC-V (C3, C6)
- **IDF 5.5**: Upgraded Docker image for smoke tests

New UART configurations in `examples/uart/` for boards with USB-UART bridges (CH340, CP2102, CH343) and QEMU smoke tests.

> Note: ESP32 original is excluded from QEMU testing as the emulator doesn't correctly emulate PHY/radio registers, causing false positive crashes.

#### Other CI Improvements

- **Stale bot**: Auto-close inactive issues/PRs after 30+7 days
- **Build caching**: Faster CI builds with PlatformIO/ESPHome cache
- **Coverage threshold**: CI fails if coverage drops below 80%
- **Dependency updates**: `actions/stale@10`, `actions/upload-artifact@6`

### Documentation

- Added Part 2 Medium article link to README
- Updated bug report template with crash debug files section

---

## [2.3.0] - 2025-12-31 

### ESPectre - The Game 

As a thank you to the community, I'm closing the year with something fun: a browser-based reaction game where your physical movement controls the gameplay through WiFi sensing. No controller needed!

Beyond the fun, it's actually useful for threshold tuning, calibration verification, and coverage testing. Uses Web Serial API (Chrome/Edge) with real-time CSI streaming at ~100 Hz.

**Play now:** [espectre.dev/game](https://espectre.dev/game)

### Features

#### Sensor Entity Customization

Full control over exposed sensor entities with standard ESPHome options: `internal`, `icon`, `filters`, `disabled_by_default`.

```yaml
espectre:
  movement_sensor:
    name: "Movement"
    internal: true
    filters:
      - multiply: 100
  motion_sensor:
    icon: "mdi:motion-sensor"
```

See [#51](https://github.com/francescopace/espectre/issues/51).

#### External Traffic Mode

Support for multi-device deployments with reduced network overhead.

- **`traffic_generator_rate: 0`**: Disable internal traffic generator and rely on external WiFi traffic
- **`publish_interval`**: Control sensor update rate independently from traffic source
- **UDP Listener**: Opens port 5555 to receive external UDP packets for CSI generation
- **95% less network overhead**: One broadcast source feeds all ESPectre devices

External traffic source: [`espectre_traffic_generator.py`](examples/espectre_traffic_generator.py) - standalone script with daemon mode and Home Assistant integration.

See [#50](https://github.com/francescopace/espectre/issues/50).

#### Traffic Generator Ping Mode

New `traffic_generator_mode: ping` option using ICMP echo requests instead of DNS queries. Use when DNS mode has low packet rates (~18 pps instead of ~100 pps) due to routers not responding to root domain queries.

See [#48](https://github.com/francescopace/espectre/issues/48).

#### Gain Lock Mode

New `gain_lock` option to control AGC/FFT gain locking behavior:

- **`auto`** (default): Enable gain lock but skip if signal too strong (AGC < 30)
- **`enabled`**: Always force gain lock (may freeze if too close to AP)
- **`disabled`**: Never lock gain (less stable CSI but works at any distance)

Solves the issue where devices too close to the AP (RSSI > -40 dB) would freeze during calibration. See [TUNING.md](TUNING.md) for AGC threshold details.

### Improvements

#### WiFi Channel Change Detection

Automatic detection and buffer reset when AP switches channel (auto-channel, roaming), preventing false positives from CSI data spikes.

Fixes [#46](https://github.com/francescopace/espectre/issues/46).

#### Multi-Window NBVI Calibration

Optimized subcarrier selection with multi-window validation, gain lock phase exclusion (first 300 packets), and updated noise gate percentile (10% → 25%).

#### Calibration Fallback with Normalization

When NBVI calibration fails, normalization is still calculated and default subcarriers [11-22] are used, preventing 2000%+ motion values from missing normalization.

### Platform Support

#### ESP32-C3 Super Mini Tested

Added example configuration `espectre-c3.yaml`. Full feature support including gain lock and NBVI calibration.

#### ESP32 (Original/WROOM-32) Tested

Tested on ESP32-WROOM-32D Mini (CH340). Fixed NBVI calibration not starting on platforms without gain lock.

**Known limitations** (ESP32 original only):
- AGC/FFT gain lock not available
- External traffic generator must start **after** ESP32 connects to WiFi
- Broadcast mode not supported; use unicast instead

See [espressif/esp-csi#247](https://github.com/espressif/esp-csi/issues/247).

---

## [2.2.0] - 2025-12-19

### Gain Lock (AGC/FFT Stabilization)

Automatic gain control locking for stable CSI measurements, based on [Espressif esp-csi](https://github.com/espressif/esp-csi) recommendations.

- **Two-phase calibration**: Gain Lock (3s, 300 pkt) → NBVI (7s, 700 pkt)
- Gain lock happens BEFORE NBVI calibration to ensure clean data
- Eliminates amplitude variations caused by automatic gain control
- Supported on ESP32-S3, C3, C5, C6 (not available on ESP32, S2)
- New files: `gain_controller.h`, `gain_controller.cpp`

### Baseline Variance Normalization

Automatic attenuation for consistent thresholds across devices and environments.

- **Always enabled** - no configuration needed
- During calibration, calculates baseline variance using selected subcarriers
- If baseline > 0.25: attenuate with `scale = 0.25 / baseline_variance`
- If baseline ≤ 0.25: no scaling needed (scale = 1.0)
- Prevents over-amplification of weak signals while taming strong ones
- Removed `normalization_enabled` and `normalization_target` parameters

### Low-Pass Filter

New 1st order Butterworth IIR filter to reduce high-frequency RF noise.

- Cutoff frequency: 11 Hz (human movement: 0.5-10 Hz, RF noise: >15 Hz)
- Reduces false positives in noisy environments (51% → 2%)
- Disabled by default: enable with `lowpass_enabled: true`
- Processing pipeline: Normalization → Hampel → **Low-Pass** → Buffer

### NBVI Improvements

Optimized parameters and restricted search range for better subcarrier selection:

| Parameter | Old | New | Effect |
|-----------|-----|-----|--------|
| `alpha` | 0.3 | 0.5 | Balanced weight between signal strength and stability |
| `min_spacing` | 2 | 1 | Allow adjacent subcarriers for better quality selection |
| `window_size` | 100 | 200 | Larger window (2s) for more stable baseline detection |
| `GUARD_BAND_LOW` | 6 | 11 | Exclude noisy edge subcarriers |
| `GUARD_BAND_HIGH` | 58 | 52 | Exclude noisy edge subcarriers |

Dynamic null subcarrier detection replaces hardcoded lists - adapts to local RF conditions.

### Performance

**Lazy Variance Evaluation**: Moving variance calculated only at publish time.
- ~99% CPU savings for variance calculation
- New API: `csi_processor_update_state()` (C++), `seg.update_state()` (Python)

### Automatic sdkconfig

The ESPHome component now auto-configures all required sdkconfig options:
- `CONFIG_ESP_WIFI_CSI_ENABLED`, `CONFIG_PM_ENABLE`, AMPDU settings, buffer sizes, tick rate
- YAML files only need platform-specific options (WiFi 6, CPU frequency, PSRAM)

### ML Data Collection

New infrastructure for building labeled CSI datasets (groundwork for 3.x):
- `me collect` CLI subcommand for recording labeled samples
- `.npz` format for ML-ready datasets
- `csi_utils.py` module with `CSIReceiver`, `CSICollector`, `MVSDetector`

### Configuration Changes

**Removed options** (now automatic):
- `normalization_enabled`, `normalization_target`, manual sdkconfig options

**Default values**: All filters disabled, normalization always active.

**Enhanced logging**: Movement logs now include WiFi channel and RSSI:

```
[I][espectre]: [######--|----] 43% | mvmt:0.43 thr:1.00 | IDLE | 101 pkt/s | ch:3 rssi:-47
```

### Testing & Documentation

- **324 pytest tests** with CI integration (`test-python` job)
- Python coverage uploaded to Codecov
- New `micro-espectre/ALGORITHMS.md` with scientific documentation of MVS, NBVI, Hampel filter

---

## [2.1.0] - 2025-12-10

### Made for ESPHome Compliance

**All example configurations now meet "Made for ESPHome" requirements**

#### WiFi Provisioning
- **BLE provisioning**: `esp32_improv` for easy setup via ESPHome/HA Companion app
- **USB provisioning**: `improv_serial` for web.esphome.io configuration (not yet supported on ESP32-C5)
- **Captive Portal**: Fallback AP "ESPectre Fallback" for WiFi configuration
- **No hardcoded credentials**: Removed `YOUR_WIFI_SSID` placeholders

#### Dashboard Adoption
- **`dashboard_import`**: One-click adoption from ESPHome Dashboard
- **`project` metadata**: Version tracking for firmware updates

#### Code Cleanup
- Renamed `espectre_component.cpp/.h` → `espectre.cpp/.h`
- Component ID standardized to `espectre_csi`
- Updated `me` CLI: `erase_flash` → `erase-flash` (esptool deprecation fix)

### Performance Optimization

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

### Test Suite & Code Coverage

- **140 test cases** (+72 from 2.0.0) with real CSI data
- **Full device testing**: All tests run on both native and ESP32-C6 via `IWiFiCSI` dependency injection
- **Codecov integration**: Coverage badge, PR comments, 80% threshold
- **84% line coverage**, 94% function coverage
- **Refactoring**: Shared utilities in `utils.h`, configurable `CalibrationManager`

---

## [2.0.0] - 2025-12-06

### Major - ESPHome Native Integration

**Complete platform migration from ESP-IDF to ESPHome**

This release represents a major architectural shift from standalone ESP-IDF firmware to a native ESPHome component, enabling seamless Home Assistant integration.

> ⚠️ **Note**: Extensively tested on ESP32-S3 and ESP32-C6, but bugs may still exist. Community contributions, bug reports, and support for additional ESP32 variants are welcome!

### Two-Platform Strategy

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

### Micro-ESPectre

**R&D Platform for Wi-Fi CSI Motion Detection - Pure Python implementation for MicroPython**

Micro-ESPectre is the research and development platform of the ESPectre project, designed for rapid prototyping, algorithmic experimentation, and academic/industrial research. It implements motion detection algorithms in pure Python, enabling fast iterations without compilation overhead.

**Key Features:**
- **Instant Deploy**: ~5 seconds to update code (no compilation)
- **MQTT Integration**: Runtime configuration via MQTT commands
- **Auto Calibration Algorithm**: Automatic subcarrier selection (F1=97.6%)
- **Analysis Tools**: Complete suite for CSI analysis and algorithm optimization
- **Feature Extraction**: Statistical features (variance, skewness, kurtosis, entropy, IQR)
- **Confidence Score**: Experimental motion detection confidence estimation
- **NVS Persistence**: Persistent configuration on filesystem

**Advanced Applications (ML/DL ready):**
- People counting
- Activity recognition (walking, falling, sitting, sleeping)
- Localization and tracking
- Gesture recognition

**Dependencies:** 
- [`micropython-esp32-csi`](https://github.com/francescopace/micropython-esp32-csi) - Custom MicroPython fork with native CSI support for ESP32 family 
- MQTT broker (e.g., Mosquitto)

### Test Suite Refactoring

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
# Run tests locally (native is the default environment)
cd test && pio test
```

### CI/CD Pipeline

**GitHub Actions integration for automated quality assurance**

- **Automated testing**: Runs on push to `main`/`develop` and pull requests
- **ESPHome build verification**: Compiles `espectre.yaml` to validate component
- **Status badge**: Real-time CI status displayed in README
- **Path filtering**: Only triggers on changes to `components/espectre/` or `test/`

---

## [1.5.0] - 2025-12-03

### Automatic Subcarrier Selection
- Zero-configuration subcarrier selection using NBVI (Normalized Baseline Variability Index) algorithm. 
- Auto-calibration at boot, re-calibration after factory_reset.
- Formula: `NBVI = 0.3 × (σ/μ²) + 0.7 × (σ/μ)`. 
- Achieves F1=97.6% (Recall 95.3%, Precision 100%, FP 0%). 

---

## [1.4.0] - 2025-11-28

### Major Refactoring
- **Feature extraction module**: Extracted to `csi_features.c/h`, reduced `csi_processor.c` by 50%
- **Configuration centralization**: All defaults in `espectre.h`, validation in `validation.h/c`
- **Two-pass variance**: Numerically stable calculation
- **Traffic generator**: Max rate 1000 pps (was 50), default 100 pps
- **CLI migration**: Bash → Python (cross-platform)
- **Wi-Fi Theremin**: `espectre-theremin.html` for CSI sonification
- **Removed**: Redundant segmentation parameters (min_length, max_length, k_factor)

---

## [1.3.0] - 2025-11-22

### ESP32-C6 Platform Support
- **WiFi 6 (802.11ax)** support with proper CSI configuration
- **Runtime-configurable parameters**: threshold, window_size via MQTT
- **Web Monitor**: `espectre-monitor.html` with real-time visualization
- **System monitoring**: CPU/RAM usage in stats command
- **MQTT optimization**: Simplified message format, removed segment tracking

---

## [1.2.1] - 2025-11-17

### Wi-Fi Optimization
ESP-IDF best practices: disabled power save (`WIFI_PS_NONE`), configurable country code, HT20 bandwidth.

---

## [1.2.0] - 2025-11-16

### Simplified Architecture
- **MVS algorithm**: Moving Variance Segmentation with adaptive threshold
- **Amplitude-based features**: +151% separation improvement for skewness/kurtosis
- **Traffic generator**: ICMP ping-based (was UDP broadcast)
- **64 subcarriers**: All available (was 52 filtered)
- **10 features**: Added temporal_delta_mean, temporal_delta_variance

---

## [1.1.0] - 2025-11-08

### Auto-Calibration System
- **Fisher's criterion**: Automatic feature selection (4-6 from 8)
- **Butterworth filter**: Order 4, cutoff 8Hz
- **Wavelet filter**: Daubechies db4 for high-noise environments
- **NVS persistence**: Configuration survives reboots
- **Modular architecture**: Split into 10 specialized modules

---

## [1.0.0] - 2025-11-01

### Initial Release
CSI-based movement detection for ESP32-S3. Hampel + Savitzky-Golay filters, 15 features, 4-state detection (IDLE/MICRO/DETECTED/INTENSE), MQTT publishing, CLI tool. 10-100 pps, <50ms latency, 3-8m range.

---

## License

GPLv3 - See [LICENSE](LICENSE) for details.
