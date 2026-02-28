# Changelog

All notable changes to this project will be documented in this file.

---

## [2.6.0] - in progress - Gesture Recognition (experimental)

### Highlights

- **Gesture detection**: ML detector now classifies motion into named gesture classes (e.g. `wave`) in addition to generic `motion` and `idle`. Both C++ and Python logs report the detected gesture name on each motion event.
- **Multiclass-only ML pipeline**: Removed all binary classification code. The training script, weights format, and inference engine are now multiclass-only (softmax output, sparse categorical crossentropy).
- **Consistent test infrastructure**: C++ and Python tests now use identical hardcoded dataset files. CV normalization is derived from per-file NPZ metadata (`gain_locked` field) instead of chip-type heuristics.

### Added

- **Gesture classification**: `MLDetector::update_state()` logs the detected class label on motion start: `Motion started (prob=0.86, class=wave)`. The gesture name comes from `ML_CLASS_LABELS` in `ml_weights.h`, generated at training time.
- **Gesture logging in Python**: `main.py` appends the gesture name to the status bar when a non-idle, non-generic class is detected (e.g. `[wave]`).
- **`gain_locked` metadata in test data loader** (`csi_test_data.h`): `CsiData` now stores `gain_locked` / `has_gain_locked` read from the NPZ file. `needs_cv_normalization()` uses this field when available, falling back to chip-based heuristics for older files.
- **`read_gain_locked()` helper** (`csi_utils.py`): Reads the `gain_locked` field from an NPZ file; returns `None` for files that predate the field.
- **Hardcoded dataset paths in Python tests**: `get_available_datasets()` now uses the same explicit file paths as `csi_test_data.h`, ensuring C++ and Python always test on identical data.

### Changed

- **Micro-ESPectre default algorithm**: `DETECTION_ALGORITHM` in `config.py` changed from `"mvs"` to `"ml"`. The ML detector requires no calibration wait and is now the recommended default for Micro-ESPectre.
- **ML architecture**: Hidden layer size changed from `[16, 8]` to `[24]` (single hidden layer). The simpler architecture matches the performance of the deeper one on current data while reducing MACs from 416 to 360.
- **ML output**: Output layer uses softmax over N classes (was sigmoid over 1). `MLDetector::predict()` returns `1 - prob[idle]` for compatibility with the existing threshold-based state machine.
- **`use_cv_normalization` fixture** (`test_validation_real_data.py`): Now reads `gain_locked` from the baseline NPZ instead of a per-chip hardcoded value, matching the C++ `needs_cv_normalization()` logic.
- **`ml_test_data.npz` format**: `expected_outputs` field changed from `int32` class IDs to `float32` probabilities (`1 - prob[idle]`), matching what `MLDetector::predict()` returns. This enables the C++ unit test to numerically verify the full inference pipeline.
- **`dataset_info.json`**: Removed the deprecated `id` field from label entries. Only `class_id`, `description`, and `color` are kept.
- **`me collect --info`**: Replaced the `id` column with `Class` (showing `class_id`) in the dataset info table.

### Removed

- **Binary classification mode**: `predict()`, `is_motion()`, and `sigmoid()` removed from `ml_detector.py`. `predict_class()` is now the sole public inference function, returning `(class_id, class_name, confidence)`.
- **`evaluate_model()` (binary)**: Replaced by `evaluate_model_multiclass()` throughout the training script.
- **`experiment_architectures` binary branch**: Architecture comparison now always uses multiclass metrics.

### Performance

Results verified on C++ (PlatformIO) and Python (pytest) with identical datasets. See [PERFORMANCE.md](PERFORMANCE.md) for full tables.

| Chip | Algorithm | Recall | FP Rate |
|------|-----------|--------|---------|
| ESP32-C3 | ML (multiclass) | 99.8% | 0.3% |
| ESP32-C6 | ML (multiclass) | 100.0% | 0.0% |
| ESP32-S3 | ML (multiclass) | 99.5% | 2.2% |
| ESP32 | ML (multiclass) | 100.0% | 0.0% |

---

## [2.5.1] - 2026-02-23 - HT STBC Multi-Antenna Router Fix

### Fixed

- **ESP32-C5/C6 STBC multi-antenna router fix**: Multi-antenna routers with STBC TX send two HT training fields per frame (HT-LTF1 + HT-LTF2), causing the CSI callback to receive 256-byte packets instead of the expected 128 bytes for HT20. On ESP32-C5/C6, `wifi_csi_acquire_config_t` has no field to disable HT STBC capture (unlike older chips). ESPectre now accepts these packets and takes the first 64 subcarriers (HT-LTF1), which is a valid channel estimate (#76, espressif/esp-csi#238)
- **Micro-ESPectre NBVI calibration on ESP32-C3**: Fixed OOM crashes during calibration caused by large in-memory allocations in the streaming NBVI computation phase; calibration now completes successfully on C3 with ~59 KB free heap
- **ESPHome 2026.2.0+ compatibility**: Ensure SPIFFS inclusion for newer ESPHome versions (#87)
- **CI develop branch**: Use local component configs instead of main branch for CI builds on develop

### Changed

- **Micro-ESPectre NBVI calibration speed**: Packet collection rate improved ~3x on ESP32-C3 (28 → 80 pps) by skipping sqrt on guard band subcarriers (excluded from NBVI selection), caching `math.sqrt` locally, and using integer arithmetic in the magnitude loop

### Added

- **Performance documentation**: RAM, Flash, and detection timing benchmarks for ESP32-C3 and ESP32-C6 in PERFORMANCE.md
- **Performance logging**: Lightweight DEBUG-level logging for heap usage (startup/post-calibration) and detection time (~10s interval)
- **git_ref substitution**: All example YAML files now use a `git_ref` substitution, making it easy to switch between branches, tags, or commits
- **Snapshot builds**: Automated pre-release builds on every push to main, providing pre-compiled firmware for testing fixes before official releases

### Documentation

- Clarify that NBVI calibration applies only to MVS mode
- Add media section in README

---

## [2.5.0] - 2026-02-15 - ML Detector, Training Pipeline & Pre-built Firmware

### Highlights

- **ML Detector (Experimental)**: Neural network-based motion detection with ~3s boot time (no calibration needed)
- **Training Pipeline**: Collect data, train, and export models for both platforms
- **Pre-built Firmware**: Ready-to-flash binaries for all 6 ESP32 variants via GitHub Releases
- **PCA & P95 Removed**: Simplified to MVS + ML detectors, NBVI-only calibration

### ML Detector

First experimental release of a neural network-based motion detector, available in both C++ (ESPHome) and Python (Micro-ESPectre).

- **Configuration**: `detection_algorithm: ml` in YAML
- **Performance**: ~97-100% F1 score depending on chip
- **Boot time**: ~3 seconds (vs ~10s for MVS) — no band calibration needed
- **Zero dependencies**: Manual MLP inference, no TFLite required

```yaml
espectre:
  detection_algorithm: ml
```

The pre-trained model shipped with this release was trained on a limited dataset collected in a single environment. It performs well in initial testing, but **we need your help to make it better**. If you try the ML detector, consider contributing baseline (empty room) and movement recordings from your environment — the more diverse the training data, the more robust the model becomes. See [ML_DATA_COLLECTION.md](micro-espectre/ML_DATA_COLLECTION.md) for how to collect and submit data via pull request.

For architecture and feature details, see [ALGORITHMS.md](micro-espectre/ALGORITHMS.md#csi-features-for-ml).

#### Training Pipeline

Collect labeled data, train a model, and export weights for both platforms:

```bash
./me collect --label <name> --duration <sec>   # Collect data
python tools/10_train_ml_model.py               # Train model
```

Exports `ml_weights.py` (Python), `ml_weights.h` (C++), and TFLite checkpoint. See [ML_DATA_COLLECTION.md](micro-espectre/ML_DATA_COLLECTION.md) for the full workflow.

### Pre-built Firmware & Easy Install

New `release.yml` workflow builds and publishes firmware for all supported chips (ESP32, S2, S3, C3, C5, C6) on every tagged release.

SETUP.md now offers two installation paths:

- **Option A: Web Flash** — Download from Releases, flash via [ESPConnect](https://thelastoutpostworkshop.github.io/ESPConnect/) in Chrome
- **Option B: ESPHome CLI** — Traditional `esphome run` workflow

Based on PR #77 by [@WLaoDuo](https://github.com/WLaoDuo).

### Removed

**PCA Detection Algorithm**: While PCA itself is a well-known statistical technique, our implementation was based on Espressif's open-source esp-radar library. Since Espressif has transitioned this library to closed source, we have removed our PCA implementation to ensure full compliance with our GPLv3 license.

MVS remains the recommended algorithm with excellent performance. Future development will focus on the ML detector, which shows very promising results in early testing and requires no initial calibration.

**P95 Calibrator**: NBVI (Normalized Band Variance Index) is now the sole calibration algorithm. NBVI consistently outperforms P95 by selecting non-consecutive subcarriers for better spectral diversity and resilience to narrowband interference.

### Improvements

- **Gain lock**: Median-based calibration (replaces mean), signed FFT gain fix, CV normalization when gain lock is skipped
- **Bug fixes**: Double amplitude calculation fix, stack allocation in Hampel filter

### Micro-ESPectre (R&D Platform)

- **Extended hardware support**: `me` CLI now supports ESP32, C3, S3, C6 with auto-detection and SHA256 firmware verification
- **ML detector filter support**: ML detector now accepts low-pass and Hampel filter parameters, matching C++ implementation
- **Import standardization**: All `src/` modules now use `try/except ImportError` pattern for MicroPython/CPython compatibility
- **Bug fixes**: Signed int8 CSI parsing, ESP32 flash offset corrected, LowPass default cutoff aligned to 11.0 Hz

### For Contributors

<details>
<summary>Architecture changes and internal improvements</summary>

#### Architecture

- **Calibrator simplification**: Removed `ICalibrator` interface and `BaseCalibrator` base class, merging all functionality into `NBVICalibrator`
- **Window size centralization**: `segmentation_window_size` (default: 75) is now defined in a single source of truth (`DETECTOR_DEFAULT_WINDOW_SIZE` in C++, `SEG_WINDOW_SIZE` in Python) and passed to both detector and calibrator
- **Calibration buffer size**: Now calculated as `10 × window_size` (default: 750 packets), automatically adapts if window size changes
- **CV normalization**: Both platforms use CV normalization consistently when gain lock is skipped
- **Unified window size in tools**: All analysis tools now use `SEG_WINDOW_SIZE` from `config.py` instead of hardcoded values

</details>

---

## [2.4.0] - 2026-01-24 - Live Recalibration, Adaptive Threshold & PCA

### Highlights

- **Two detection algorithms**: Choose between MVS (default) and PCA (Principal Component Analysis)
- **Recalibrate from Home Assistant**: New switch to trigger recalibration without reflashing
- **Adaptive threshold by default**: No manual tuning needed - works out of the box

### New Features

#### Dual Detection Algorithms

Two motion detection algorithms are now available:

| Algorithm | Configuration |
|-----------|---------------|
| **MVS** (default) | `detection_algorithm: mvs` |
| **PCA** (experimental) | `detection_algorithm: pca` |

#### Calibrate Switch

New Home Assistant switch for triggering recalibration without reflashing:

- `switch.espectre_calibrate`: Turn ON to recalibrate, auto-turns OFF when complete
- Useful after room layout changes or furniture moves

#### Adaptive Threshold

The `segmentation_threshold` parameter is now optional:

- **Default**: Adaptive threshold calculated as P95 × 1.1 during calibration
- **Manual override**: Specify value in YAML to use fixed threshold

### Improvements

#### Dual Band Selection Algorithms

Two automatic subcarrier selection algorithms:

| Algorithm | Method | Configuration |
|-----------|--------|---------------|
| **NBVI** (default) | 12 non-consecutive subcarriers | `segmentation_calibration: nbvi` |
| **P95** | 12 consecutive subcarriers | `segmentation_calibration: p95` |

#### HT20-Only Mode

Simplified to WiFi 4 (802.11n) HT20 mode for stable 64 subcarriers across all ESP32 variants:

- Consistent performance on C3, C6, S3, and original ESP32
- Reduced memory footprint

#### Other Improvements

- **Lower threshold minimum**: 0.1 (was 0.5) for high-sensitivity applications
- **ESP32-C3 dev config**: Added `espectre-c3-dev.yaml`

### Bug Fixes

- **ESP32-C3 boot crash**: Fixed duplicate `register_component` calls
- **USB Serial JTAG**: Correct ESPHome macro for detection
- **CSI data overflow**: Limit to 128 bytes to prevent overflow

### Micro-ESPectre (R&D Platform)

- **SHA256 firmware verification**: New `verify` command detects outdated firmware
- **CSI Stream Protocol v2**: Auto-detected chip type, contributor tracking
- **Refactored calibrators**: Cleaner architecture with `p95_calibrator.py`, `nbvi_calibrator.py`, `threshold.py`

### For Contributors

<details>
<summary>Architecture changes, testing, and CI/CD improvements</summary>

#### Architecture

- New `IDetector` interface for polymorphic detection (`MVSDetector`, `PCADetector`)
- Centralized threshold calculation in `threshold.h` / `threshold.py`
- Legacy `csi_processor` module removed

#### Multi-Chip Test Suite

Tests run on ESP32-C6 and ESP32-S3 with real CSI data:

| Chip | Recall | FP Rate | F1-Score |
|------|--------|---------|----------|
| C6 | 98.8% | 0.0% | 99.4% |
| S3 | 99.1% | 14.3% | 92.9% |

#### CI/CD

- **NPZ data loading**: C++ tests use same NPZ files as Python via [cnpy](https://github.com/rogersce/cnpy)
- **QEMU smoke tests**: Catches crashes on S3, C3, C6 before deploy
- **Stale bot**: Auto-close inactive issues after 30+7 days
- **Coverage threshold**: CI fails below 80%

</details>

### Documentation

- **Roadmap update**: Added 3D Localization in v4.x (30-50cm indoor tracking with phase-coherent antenna array)
- Added Part 2 Medium article link to README
- Updated bug report template with crash debug section
- Added Home Assistant dashboard screenshot to README

---

## [2.3.0] - 2025-12-31 - End of Year Edition

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

## [2.2.0] - 2025-12-19 - Gain Lock, Low-Pass Filter & ML Data Collection

### Gain Lock (AGC/FFT Stabilization)

Automatic gain control locking for stable CSI measurements, based on [Espressif esp-csi](https://github.com/espressif/esp-csi) recommendations.

- **Two-phase calibration**: Gain Lock (3s, 300 pkt) → NBVI (~7.5s, 10 × window_size pkt)
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

## [2.1.0] - 2025-12-10 - Made for ESPHome Compliance

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

## [2.0.0] - 2025-12-06 - ESPHome Native Integration

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

## [1.5.0] - 2025-12-03 - Automatic Subcarrier Selection

### Automatic Subcarrier Selection
- Zero-configuration subcarrier selection using NBVI (Normalized Baseline Variability Index) algorithm. 
- Auto-calibration at boot, re-calibration after factory_reset.
- Formula: `NBVI = 0.3 × (σ/μ²) + 0.7 × (σ/μ)`. 
- Achieves F1=97.6% (Recall 95.3%, Precision 100%, FP 0%). 

---

## [1.4.0] - 2025-11-28 - Major Refactoring & Technical Debt Reduction

### Major Refactoring
- **Feature extraction module**: Extracted to `csi_features.c/h`, reduced `csi_processor.c` by 50%
- **Configuration centralization**: All defaults in `espectre.h`, validation in `validation.h/c`
- **Two-pass variance**: Numerically stable calculation
- **Traffic generator**: Max rate 1000 pps (was 50), default 100 pps
- **CLI migration**: Bash → Python (cross-platform)
- **Wi-Fi Theremin**: `espectre-theremin.html` for CSI sonification
- **Removed**: Redundant segmentation parameters (min_length, max_length, k_factor)

---

## [1.3.0] - 2025-11-22 - ESP32-C6 Platform Support

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

## [1.2.0] - 2025-11-16 - Simplified Architecture & MVS Segmentation

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
