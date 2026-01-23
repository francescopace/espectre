# 🛜 Micro-ESPectre 👻

**R&D Platform for Wi-Fi CSI Motion Detection - Pure Python implementation for MicroPython**

Micro-ESPectre is the **research and development platform** of the ESPectre project, designed for fast prototyping, algorithm experimentation, and academic/industrial research. It implements the core motion detection algorithms in pure Python, enabling rapid iteration without compilation overhead.

## What is Micro-ESPectre?

Micro-ESPectre implements the ESPectre motion-detection algorithms entirely in Python and serves as the **innovation lab** where new approaches and parameters are developed and validated before being migrated to the production ESPHome component.

### Role in the ESPectre Ecosystem

Micro-ESPectre is part of a **two-platform strategy**:

| Platform | Purpose | Target Users |
|----------|---------|--------------|
| **[ESPectre](https://github.com/francescopace/espectre)** (C++) | Production deployment | Smart home users, Home Assistant |
| **[Micro-ESPectre](https://github.com/francescopace/espectre/tree/main/micro-espectre)** (Python) | R&D and prototyping | Researchers, developers, academics |

**Why MQTT instead of Native API?**
Micro-ESPectre uses MQTT for maximum flexibility - it's not tied to Home Assistant and can integrate with:
- **Industrial systems** (SCADA, PLCs, factory automation)
- **Academic research** (data collection, algorithm validation)
- **Custom applications** (any MQTT-compatible platform)

### Innovation Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    INNOVATION CYCLE                         │
├─────────────────────────────────────────────────────────────┤
│  Micro-ESPectre (Python)          ESPectre (ESPHome)        │
│  ┌─────────────────────┐          ┌─────────────────────┐   │
│  │ • Fast prototyping  │ ──────▶  │ • Production ready  │   │
│  │ • Algorithm testing │  Port    │ • Home Assistant    │   │
│  │ • Parameter tuning  │ ──────▶  │ • Native API        │   │
│  │ • Research/academic │          │ • OTA updates       │   │
│  └─────────────────────┘          └─────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Key Benefits for R&D:**
- **Instant deployment**: No compilation, ~5 seconds to update
- **Easy experimentation**: Modify parameters and test immediately
- **Quick validation**: Test algorithms and configurations rapidly
- **Proven patterns**: Validated algorithms flow to production C++ code

### What is micropython-esp32-csi?

[micropython-esp32-csi](https://github.com/francescopace/micropython-esp32-csi) is a MicroPython fork that I wrote to expose ESP32's CSI (Channel State Information) capabilities to Python. 
This fork makes CSI-based applications accessible to Python developers and enables rapid prototyping of WiFi sensing applications.

## Comparison with C++ Version (ESPHome)

### Feature Comparison

| Feature | ESPHome (C++) | Python (MicroPython) | Status |
|---------|---------------|----------------------|--------|
| **Core Algorithm** |
| MVS Detector | ✅ | ✅ | Aligned |
| PCA Detector | ✅ | ✅ | Aligned |
| MVS Segmentation | ✅ | ✅ | Aligned |
| Spatial Turbulence | ✅ | ✅ | Aligned |
| Moving Variance | ✅ | ✅ | Aligned |
| **Gain Lock (AGC/FFT)** |
| Gain Lock | ✅ | ✅ | Aligned (S3/C3/C5/C6) |
| Read AGC/FFT values | ✅ | ✅ | Implemented |
| Force AGC/FFT values | ✅ | ✅ | Implemented |
| **WiFi Traffic Generator** |
| Traffic Generation | ✅ | ✅ | Implemented |
| Configurable Rate | ✅ | ✅ | Implemented |
| **Configuration** |
| YAML Configuration | ✅ | ❌ | ESPHome only |
| MQTT Commands | ❌ | ✅ | Micro-ESPectre only |
| Runtime Config | ✅ (via HA) | ✅ (via MQTT) | Different methods |
| **Automatic Subcarrier Selection** |
| NBVI Calibration | ✅ | ✅ | Default algorithm |
| P95 Band Selection | ✅ | ✅ | Alternative algorithm |
| Percentile-based Detection | ✅ | ✅ | Implemented |
| Noise Gate | ✅ | ✅ | ✅ Implemented |
| Spectral De-correlation | ✅ | ✅ | Implemented |
| **Filters** |
| Low-Pass Filter | ✅ | ✅ | Butterworth 1st order, reduces high-freq noise (11 Hz default) |
| Hampel Filter | ✅ | ✅ | Outlier removal, applied to turbulence (disabled by default) |
| **CSI Features** |
| `features_enable` | ❌ | ✅ | `ENABLE_FEATURES = True` in config.py |
| CSI Features | ❌ | ✅ | entropy_turb, iqr_turb, variance_turb, skewness, kurtosis |
| Feature Extraction | ❌| ✅ | Publish-time calculation (no buffer, 92% memory saved) |

### Performance Comparison

| Metric | ESPHome (C++) | Python (MicroPython) |
|--------|---------------|----------------------|
| Performance | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Memory Usage | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Ease of Use | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Deployment | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Build Time | ~15 seconds | Instant (no build) |
| Update Time | ~15 seconds (OTA) | ~5 seconds |
| HA Integration | ⭐⭐⭐⭐⭐ (Native) | ⭐⭐⭐ (MQTT) |

For detailed performance metrics (confusion matrix, F1-score, benchmarks), see [PERFORMANCE.md](../PERFORMANCE.md).

### When to Use Which Version?

**Use Micro-ESPectre (Python) if you want:**
- Quick prototyping and experimentation
- Easy deployment and updates (~5 seconds)
- Core motion detection functionality
- Simple Python-based development
- MQTT-based runtime configuration
- Automatic subcarrier selection

**Use ESPectre (ESPHome) if you need:**
- Native Home Assistant integration (auto-discovery)
- Maximum performance and efficiency
- Advanced CSI feature extraction
- Multiple filtering algorithms
- Production-grade stability
- YAML-based configuration

## Requirements

### Hardware
- ESP32 with CSI support (S3/C6 recommended, other variants supported)
- 2.4GHz WiFi router

### Software
- MicroPython with esp32-csi module installed
- MQTT broker (Home Assistant, Mosquitto, etc.)
- Python 3.12 (Recommended for deployment scripts, CLI, and analysis tools)

## CLI Tool Overview

Micro-ESPectre includes a unified command-line tool called **`me`** that simplifies all device operations. This tool will be used throughout the Quick Start guide and beyond.

### Main Commands

The `me` CLI provides these essential commands:

| Command | Description | Usage Example |
|---------|-------------|---------------|
| `flash` | Flash MicroPython firmware to device | `./me flash --erase` |
| `deploy` | Deploy Python code to device | `./me deploy` |
| `run` | Run the application | `./me run` |
| `stream` | Stream raw CSI data via UDP | `./me stream --ip 192.168.1.100` |
| `collect` | Collect labeled CSI data for ML | `./me collect --label movement --duration 30` |
| `verify` | Verify firmware installation | `./me verify` |
| *(interactive)* | Interactive MQTT control | `./me` |

### Key Features

- **Auto-detection**: Automatically detects serial port and chip type
- **Fast deployment**: Updates code in ~5 seconds (no compilation)
- **Simple syntax**: Intuitive commands for all operations
- **Manual override**: Specify port/chip manually if needed

**Example workflow:**
```bash
./me flash --erase    # Flash firmware (first time only)
./me deploy           # Deploy code
./me run              # Run application
./me                  # Interactive MQTT control

# For real-time CSI streaming (gesture detection, research)
./me stream --ip 192.168.1.100  # Stream to PC
```

> **Note**: The interactive mode (`./me` without arguments) provides advanced MQTT control features and is covered in detail in the [Interactive CLI (Advanced)](#interactive-cli-advanced) section.

## Quick Start

Get started in just **6 simple steps** - no compilation required!

### 0. Setup Python Environment

If you've already set up the main ESPectre project, you can reuse that virtual environment. Otherwise, create a new one:

```bash
# Clone the repository
git clone https://github.com/francescopace/espectre.git
cd espectre/micro-espectre

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate   # On Windows

# Your prompt should now show (venv) prefix
```

**Why use a virtual environment?**
- Isolates project dependencies from system Python
- Prevents version conflicts with other projects
- Makes the project portable and reproducible

**Note:** Remember to activate the virtual environment (`source venv/bin/activate`) every time you open a new terminal session to work with this project.

### 1. Install Dependencies

```bash
# Install Python dependencies (venv should be active)
pip install -r requirements.txt
```

This installs all required tools including `esptool` (for flashing firmware) and `mpremote` (for deploying code).

### 2. Flash MicroPython Firmware

**⚠️ Required for first-time setup** (only once per device)

The precompiled firmware with CSI support is automatically downloaded from [micropython-esp32-csi releases](https://github.com/francescopace/micropython-esp32-csi/releases).

**Auto-detect mode** (recommended - detects port and chip automatically):
```bash
./me flash --erase
```

The CLI will:
- Auto-detect your serial port
- Auto-detect your chip type
- Download the correct firmware (cached locally)
- Flash it to your device

**Manual mode** (if auto-detect fails):
```bash
# Specify chip and/or port manually
./me flash --chip s3 --port /dev/ttyUSB0 --erase
```

**Verify the installation:**
```bash
./me verify
```

### 3. Configure WiFi and MQTT

```bash
# Create configuration file
cp src/config_local.py.example src/config_local.py

# Edit with your credentials
vi src/config_local.py  # or use your preferred editor
```

Update these settings:
```python
WIFI_SSID = "YourWiFiSSID"
WIFI_PASSWORD = "YourWiFiPassword"
MQTT_BROKER = "homeassistant.local"  # or IP address
MQTT_USERNAME = "mqtt"
MQTT_PASSWORD = "mqtt"
```

### 4. Deploy and Run

```bash
# Deploy code (auto-detect port)
./me deploy

# Run application (auto-detect port)
./me run
```

That's it! The device will now:
- Connect to WiFi
- Connect to MQTT broker
- Start publishing motion detection data
- Automatically calibrate subcarriers (NBVI or P95 algorithm)

### 5. Monitor and Control

**Option A: Interactive CLI (MQTT control)**
```bash
./me
```

**Option B: Home Assistant**

Add to your `configuration.yaml`:
```yaml
mqtt:
  binary_sensor:
    - name: "ESPectre Motion"
      state_topic: "home/espectre/node1"
      value_template: "{{ value_json.state }}"
      payload_on: "motion"
      payload_off: "idle"
      device_class: motion
```

## Project Structure

```
micro-espectre/
├── data/                      # Collected CSI datasets (.npz files)
├── firmware/                  # Downloaded firmware cache (gitignored)
├── src/                       # Main package
├── tests/                     # Pytest test suite
├── tools/                     # Analysis and optimization tools
├── requirements.txt           # Python dependencies
├── espectre-monitor.html      # Web Monitor: real-time analysis & configuration
├── espectre-theremin.html     # Audio sonification tool (experimental)
├── me                         # Unified CLI tool (flash/deploy/run/stream/collect/MQTT)
├── ML_DATA_COLLECTION.md      # Guide for ML data collection
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

### Key Files

- **`me`**: Main CLI tool for flashing firmware, deploying code, running app, streaming, collect, and MQTT control
- **`firmware/`**: Downloaded firmware cache (auto-created on first flash)
- **`src/`**: Core Python implementation of motion detection algorithms
- **`src/csi_streamer.py`**: UDP streaming module for real-time CSI data
- **`tests/`**: Pytest test suite for all core modules
- **`tools/`**: Analysis scripts for algorithm development and validation
- **`tools/csi_utils.py`**: CSI utilities (receiver, collector, detectors) for PC-side processing
- **`ML_DATA_COLLECTION.md`**: Guide for collecting labeled CSI datasets for ML

## Testing

Micro-ESPectre includes a comprehensive test suite using pytest.

### Running Tests

```bash
# Activate virtual environment
source venv/bin/activate

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_filters.py -v

# Run specific test class or function
pytest tests/test_segmentation.py::TestStateMachine -v
```

### Test Suites

| Suite | Type | Data | Focus |
|-------|------|------|-------|
| `test_config` | Unit | — | Configuration constants, guard bands |
| `test_filters` | Unit | Synthetic | Hampel, low-pass filters |
| `test_features` | Unit | Synthetic | Feature extraction (entropy, skewness, kurtosis) |
| `test_segmentation` | Unit | Synthetic | MVS state machine, variance calculation |
| `test_p95_calibrator` | Unit | **Real** | P95 band selection, magnitude, turbulence |
| `test_nbvi_calibrator` | Unit | **Real** | NBVI subcarrier selection |
| `test_mqtt` | Unit | Synthetic | MQTT handler and commands |
| `test_traffic_generator` | Unit | Synthetic | Rate limiting, error handling |
| `test_running_variance` | Unit | Synthetic | O(1) vs two-pass variance comparison |
| `test_optimization_equivalence` | Unit | Synthetic | Optimization correctness |
| `test_validation_real_data` | Integration | **Real** | End-to-end with real CSI data (C6, S3) |

### CI Integration

Tests run automatically on every push/PR via GitHub Actions. See `.github/workflows/ci.yml`.

## Configuration

All configuration is in `config.py`. The detection pipeline follows this order:

```
Boot → Gain Lock → Calibration → Detection Loop (with optional filters)
```

### 1. Gain Lock (Hardware Stabilization)

Locks AGC/FFT gain values for stable CSI amplitudes.

```python
GAIN_LOCK_MODE = "auto"       # "auto", "enabled", or "disabled"
GAIN_LOCK_MIN_SAFE_AGC = 30   # Minimum safe AGC (used in auto mode)
```

| Mode | Description |
|------|-------------|
| `auto` (default) | Lock gain, skip if signal too strong (AGC < 30) |
| `enabled` | Always force gain lock (may freeze if too close to AP) |
| `disabled` | Never lock gain (less stable CSI, works at any distance) |

### 2. Detection Algorithm

Choose the motion detection algorithm.

```python
DETECTION_ALGORITHM = "mvs"   # "mvs" (default) or "pca"
```

| Algorithm | Method |
|-----------|--------|
| **MVS** (default) | Moving Variance Segmentation of Turbulence |
| **PCA** | Principal Component Analysis |

### 3. Calibration Algorithm (MVS only)

Selects which subcarriers to use for detection. Ignored when using PCA.

```python
CALIBRATION_ALGORITHM = "nbvi"  # "nbvi" (default) or "p95"
```

| Algorithm | Selection | Best For |
|-----------|-----------|----------|
| **NBVI** (default) | 12 non-consecutive subcarriers | Spectral diversity, ~3x faster |
| **P95** | 12 consecutive subcarriers | Simpler logic, contiguous bands |

### 4. Detection Parameters (MVS only)

```python
SEG_THRESHOLD = "auto"     # "auto" (P95×1.4), "min" (P100), or 0.1-10.0
SEG_WINDOW_SIZE = 50       # Moving variance window (10-200 packets)
```

### 5. Filters (Optional)

Applied to turbulence values before motion detection.

```python
# Low-pass filter (reduces high-frequency noise)
ENABLE_LOWPASS_FILTER = False
LOWPASS_CUTOFF = 11.0          # Cutoff frequency in Hz

# Hampel filter (outlier/spike removal)
ENABLE_HAMPEL_FILTER = False
HAMPEL_WINDOW = 7
HAMPEL_THRESHOLD = 4.0
```

### 6. Feature Extraction (Optional, MVS only)

```python
ENABLE_FEATURES = False    # Enable CSI feature extraction for ML
```

For detailed parameter tuning, see [TUNING.md](../TUNING.md).

### Published Data (MQTT Payload)

The system publishes JSON payloads to the configured MQTT topic (default: `home/espectre/node1`):

**Basic payload** (always published):
```json
{
  "movement": 0.0234,            // Current moving variance
  "threshold": 1.0,              // Current threshold
  "state": "idle",               // "idle" or "motion"
  "packets_processed": 100,      // Packets since last publish
  "packets_dropped": 0,          // Packets dropped since last publish
  "pps": 105,                    // Packets per second (calculated with ms precision)
  "timestamp": 1700000000        // Unix timestamp
}
```

**Extended payload** (when `ENABLE_FEATURES = True`):
```json
{
  "movement": 1.2345,
  "threshold": 1.0,
  "state": "motion",
  "packets_processed": 100,
  "packets_dropped": 0,
  "pps": 105,
  "timestamp": 1700000000,
  
  "features": {
    "entropy_turb": 2.97,
    "iqr_turb": 4.56,
    "variance_turb": 4.71,
    "skewness": 0.23,
    "kurtosis": -1.16
  },
  
  "confidence": 0.85,
  
  "triggered": ["entropy_turb", "iqr_turb", "variance_turb", "skewness"]
}
```

| Field | Description |
|-------|-------------|
| `features` | CSI features calculated at publish time |
| `confidence` | Detection confidence (0.0-1.0) based on weighted feature voting |
| `triggered` | List of features that exceeded their motion thresholds |

**Top 5 Features (publish-time, no buffer needed, tested with SEG_WINDOW_SIZE=50):**
| Feature | Fisher J | Type | Description |
|---------|----------|------|-------------|
| `iqr_turb` | 3.56 | Turbulence buffer | IQR approximation (range × 0.5) |
| `skewness` | 2.54 | W=1 (current pkt) | Distribution asymmetry |
| `kurtosis` | 2.24 | W=1 (current pkt) | Distribution tailedness |
| `entropy_turb` | 2.08 | Turbulence buffer | Shannon entropy of turbulence distribution |
| `variance_turb` | 1.21 | Turbulence buffer | Moving variance (already calculated by MVS!) |

**Confidence interpretation:**
| Confidence | Meaning |
|------------|---------|
| 0.0 - 0.3 | Very likely IDLE |
| 0.3 - 0.5 | Uncertain / slight movement |
| 0.5 - 0.7 | Probable movement |
| 0.7 - 1.0 | Confident motion detection |

## Analysis Tools

The `tools/` directory contains Python scripts for CSI data analysis and algorithm validation.

See [tools/README.md](tools/README.md) for complete script documentation.

## Automatic Subcarrier Selection

Micro-ESPectre implements automatic subcarrier selection with two algorithms:

- **NBVI** (default): Selects 12 non-consecutive subcarriers based on baseline variability index
- **P95**: Selects 12 consecutive subcarriers by minimizing P95 moving variance

Both algorithms achieve high performance (>90% recall, <15% FP rate) with **zero manual configuration**.

> ⚠️ **IMPORTANT**: Keep the room **quiet and still** for 10 seconds after device boot. The auto-calibration runs during this time and movement will affect detection accuracy.

For complete algorithm documentation, see [ALGORITHMS.md](ALGORITHMS.md#automatic-subcarrier-selection).

## Machine Learning & Advanced Applications

Micro-ESPectre is the **R&D platform** for advanced CSI-based applications. While the core focuses on motion detection using mathematical algorithms (MVS + NBVI/P95 calibration), the platform provides infrastructure for ML-based features planned for release 3.x:

- **Gesture recognition**
- **Human Activity Recognition (HAR)**
- **People counting**
- **Localization and tracking**

### Getting Started with ML

👉 **[ML_DATA_COLLECTION.md](ML_DATA_COLLECTION.md)** - Complete guide for data collection, labeling, and dataset format.

### Available Features

Micro-ESPectre extracts **5 CSI features** for ML applications:

| Feature | Fisher J | Description |
|---------|----------|-------------|
| **iqr_turb** | 3.56 | IQR of turbulence buffer |
| **skewness** | 2.54 | Distribution asymmetry |
| **kurtosis** | 2.24 | Distribution tailedness |
| **entropy_turb** | 2.08 | Shannon entropy |
| **variance_turb** | 1.21 | Moving variance (from MVS) |

See `tests/test_features.py` and `tests/test_validation_real_data.py` for feature validation.

<details>
<summary>Standardized Wi-Fi Sensing (IEEE 802.11bf) (click to expand)</summary>

Currently, only a limited number of Wi-Fi chipsets support CSI extraction, which restricts hardware options for Wi-Fi sensing applications. However, the **IEEE 802.11bf (Wi-Fi Sensing)** standard should significantly improve this situation by making CSI extraction a standardized feature.

### IEEE 802.11bf - Wi-Fi Sensing

The **802.11bf** standard was **[officially published on September 26, 2025](https://standards.ieee.org/ieee/802.11bf/11574/)**, introducing **Wi-Fi Sensing** as a native feature of the Wi-Fi protocol. Main characteristics:

- **Native sensing**: Detection of movements, gestures, presence, and vital signs
- **Interoperability**: Standardized support across different vendors
- **Optimizations**: Specific protocols to reduce overhead and power consumption
- **Privacy by design**: Privacy protection mechanisms integrated into the standard
- **Greater precision**: Improvements in temporal and spatial granularity
- **Existing infrastructure**: Works with already present Wi-Fi infrastructure

### Adoption Status (2025)

**Market**: The Wi-Fi Sensing market is in its early stages and is expected to experience significant growth in the coming years as the 802.11bf standard enables native sensing capabilities in consumer devices.

**Hardware availability**:
- ⚠️ **Consumer routers**: Currently **there are no widely available consumer routers** with native 802.11bf support
- 🏢 **Commercial/industrial**: Experimental devices and integrated solutions already in use
- 🔧 **Hardware requirements**: Requires multiple antennas, Wi-Fi 6/6E/7 support, and AI algorithms for signal processing

**Expected timeline**:
- **2025-2026**: First implementations in enterprise and premium smart home devices
- **2027-2028**: Diffusion in high-end consumer routers
- **2029+**: Mainstream adoption in consumer devices

### Future Benefits for Wi-Fi Sensing

When 802.11bf is widely adopted, applications like this project will become:
- **More accessible**: No need for specialized hardware or modified firmware
- **More reliable**: Standardization ensures predictable behavior
- **More efficient**: Protocols optimized for continuous sensing
- **More secure**: Privacy mechanisms integrated at the standard level
- **More powerful**: Ability to detect even vital signs (breathing, heartbeat)

**Perspective**: In the next 3-5 years, routers and consumer devices will natively support Wi-Fi Sensing, making projects like this implementable without specialized hardware or firmware modifications. This will open new possibilities for smart home, elderly care, home security, health monitoring, and advanced IoT applications.

**For now**: Solutions like this project based on **ESP32 CSI API** remain the most accessible and economical way to experiment with Wi-Fi Sensing.

</details>

## Interactive CLI (Advanced)

Beyond the basic commands covered in the [CLI Tool Overview](#cli-tool-overview), the `me` tool provides an **interactive mode** for advanced device control and monitoring via MQTT.

**Prerequisites**: Make sure you have completed the [Python Environment Setup](#0-setup-python-environment) before using the CLI.

### Usage

```bash
# Make sure virtual environment is active
# (your prompt should show (venv) prefix)
# If not: source venv/bin/activate  # On macOS/Linux

# Run the interactive CLI:
./me

# Connect to specific broker
./me --broker 192.168.1.100 --port 1883

# With authentication
./me --broker homeassistant.local --username mqtt --password mqtt
```

### Features

- **Interactive prompt** with autocompletion (TAB) and history search (Ctrl+R)
- **All MQTT commands** available (see table below)
- **Web UI launcher**: `webui` command opens `espectre-monitor.html` in browser
- **YAML-formatted responses** for easy reading
- **Environment variables** support via `.env` file

### CLI-Only Commands

| Command | Description |
|---------|-------------|
| `webui` | Open web monitor in browser |
| `clear` | Clear screen |
| `help` | Show all commands |
| `about` | Show about information |
| `exit` | Exit CLI |

## Web Monitor

Micro-ESPectre includes a powerful **Web-based monitoring dashboard** for real-time analysis and configuration. This tool is essential for parameter tuning, algorithm validation, and live visualization of motion detection.

### Features

| Feature | Description |
|---------|-------------|
| **MQTT Connection** | Direct WebSocket connection to your MQTT broker |
| **Device Info** | View device model, IP, MAC, WiFi protocol, bandwidth, and channel |
| **Live Configuration** | Adjust detection parameters (response speed, threshold) in real-time |
| **Real-Time Chart** | Live visualization of movement, threshold, packets/sec, and dropped packets |
| **Runtime Statistics** | Memory usage, loop timing, and Traffic Generator diagnostics |
| **Factory Reset** | Reset device to default configuration and re-calibrate |

### Screenshots

**Real-Time Chart View**

![Web Monitor Chart](../images/web_monitor_chart.png)

The dashboard displays:
- **State**: Current detection state (MOTION in red, IDLE in green)
- **Confidence**: Detection confidence percentage with progress bar
- **Movement**: Current moving variance value
- **Last Update**: Timestamp of last MQTT message

The chart shows:
- **Movement** (red line): Current moving variance output
- **Threshold** (dashed blue): Detection threshold level
- **Packets/sec** (green): CSI packet rate from traffic generator
- **Dropped** (orange): Dropped packets count

**Configuration Panel**

![Web Monitor Configuration](../images/web_monitor_configurations.png)

**Device Configuration** section shows:
- Model, IP address, MAC address, WiFi protocol
- Bandwidth (HT20), Channel, CSI status

**Detection Parameters** (adjustable via sliders):
- **Response Speed** (10-200): How fast the system reacts to changes (window size)
- **Motion Threshold** (0.5-10.0): Minimum threshold to detect movement
- **Active Subcarriers**: WiFi frequency channels used for detection

**Action Buttons**:
- **RELOAD INFO**: Refresh device information
- **STATISTICS**: View runtime statistics
- **FACTORY RESET**: Reset device to default configuration

### Usage

**Launch from CLI** (recommended):
```bash
./me          # Start interactive mode
webui         # Open web monitor in browser
```

The CLI automatically serves the HTML file and opens it in your default browser.

**Manual launch**:
Open `micro-espectre/espectre-monitor.html` directly in your browser and configure the MQTT connection manually.

### Browser Compatibility

> ⚠️ **Chrome Users**: If MQTT WebSocket connection fails, you need to enable local network WebSocket access:
> 1. Open `chrome://flags/#local-network-access-check-websockets`
> 2. Set to **"Enabled"**
> 3. Restart Chrome
>
> This is required because Chrome blocks WebSocket connections to private network addresses (like `homeassistant.local`) by default.

### Additional Tools

- **`espectre-theremin.html`**: Audio sonification of CSI data (experimental) - converts motion data to sound for auditory feedback

## MQTT Integration

Micro-ESPectre uses MQTT for communication with Home Assistant and runtime configuration.

> **Note**: The main ESPectre component (ESPHome) uses **Native API** instead of MQTT for Home Assistant integration. Micro-ESPectre retains MQTT support for flexibility and compatibility with non-Home Assistant setups.

### Available MQTT Commands

Publish JSON commands to `home/espectre/node1/cmd`:

| Command | Example Payload | Description |
|---------|-----------------|-------------|
| `info` | `{"cmd": "info"}` | Get system information (network, device, config) |
| `stats` | `{"cmd": "stats"}` | Get runtime statistics (memory, state, metrics) |
| `segmentation_threshold` | `{"cmd": "segmentation_threshold", "value": 1.5}` | Set detection threshold (0.0-10.0) |
| `segmentation_window_size` | `{"cmd": "segmentation_window_size", "value": 100}` | Set window size (10-200 packets) |
| `factory_reset` | `{"cmd": "factory_reset"}` | Reset to defaults and re-calibrate |

### Command Responses

**`info` command** returns system information:
```json
{
  "network": {
    "ip_address": "192.168.1.28",
    "mac_address": "7C:2C:67:42:BB:AC",
    "channel": {"primary": 4, "secondary": 0},
    "protocol": "802.11b/g/n/ax",
    "bandwidth": "HT20",
    "csi_enabled": true,
    "traffic_generator_rate": 100
  },
  "device": {"type": "esp32"},
  "mqtt": {
    "base_topic": "home/espectre/node1",
    "cmd_topic": "home/espectre/node1/cmd",
    "response_topic": "home/espectre/node1/response"
  },
  "segmentation": {"threshold": 1.0, "window_size": 50},
  "subcarriers": {"indices": [6, 9, 10, 15, 18, 19, 30, 33, 36, 40, 49, 52]}
}
```

**`stats` command** returns runtime statistics:
```json
{
  "timestamp": 1733250000,
  "uptime": "2h 15m 30s",
  "free_memory_kb": 8090.2,
  "loop_time_ms": 0.97,
  "state": "idle",
  "turbulence": 1.8608,
  "movement": 0.0824,
  "threshold": 1.0,
  "traffic_generator": {
    "running": true,
    "target_pps": 100,
    "actual_pps": 99.8,
    "avg_loop_ms": 1.25,
    "packets_sent": 125000,
    "errors": 0
  }
}
```

- `free_memory_kb`: Available heap memory (higher on S3 with PSRAM ~8MB)
- `loop_time_ms`: Main loop execution time in milliseconds (the smaller, the better)
- `traffic_generator`: Traffic generator diagnostics
  - `running`: Whether the generator is active
  - `target_pps`: Configured packets per second
  - `actual_pps`: Actual packets per second achieved
  - `avg_loop_ms`: Average loop time (should be ≤10ms for 100pps)
  - `packets_sent`: Total packets sent since start
  - `errors`: Socket errors count

### Runtime Configuration

Configuration changes made via MQTT commands are **session-only** and reset on reboot. The adaptive threshold (P95 × 1.4) is recalculated automatically at each boot for optimal performance.

## Home Assistant Integration

For seamless Home Assistant integration with auto-discovery, consider using the main [ESPectre ESPHome component](https://github.com/francescopace/espectre) instead.
However, if you need to integrate Micro-ESPectre with Home Assistant, you can do it via MQTT.
Add these sensors to your `configuration.yaml`:

```yaml
mqtt:
  sensor:
    - name: "ESPectre Movement"
      state_topic: "home/espectre/node1"
      value_template: "{{ value_json.movement }}"
      unit_of_measurement: ""
      
    - name: "ESPectre State"
      state_topic: "home/espectre/node1"
      value_template: "{{ value_json.state }}"

  binary_sensor:
    - name: "ESPectre Motion"
      state_topic: "home/espectre/node1"
      value_template: "{{ value_json.state }}"
      payload_on: "motion"
      payload_off: "idle"
      device_class: motion
```

## References

For scientific references and algorithm documentation, see [ALGORITHMS.md](ALGORITHMS.md).

## Related Projects

- [ESPectre](../README.md) - Main project with native Home Assistant integration
- [micropython-esp32-csi](https://github.com/francescopace/micropython-esp32-csi) - MicroPython CSI module

## License

GPLv3 - See [LICENSE](../LICENSE) for details.

---

## Author

**Francesco Pace**  
Email: [francesco.pace@espectre.dev](mailto:francesco.pace@espectre.dev)  
LinkedIn: [linkedin.com/in/francescopace](https://www.linkedin.com/in/francescopace/)
