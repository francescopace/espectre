# 🛜 Micro-ESPectre 👻

**R&D Platform for Wi-Fi CSI Motion Detection - Pure Python implementation for MicroPython**

Micro-ESPectre is the **research and development platform** of the ESPectre project, designed for fast prototyping, algorithm experimentation, and academic/industrial research. It implements the core motion detection algorithms in pure Python, enabling rapid iteration without compilation overhead.

## 🎯 What is Micro-ESPectre?

Micro-ESPectre implements the ESPectre motion-detection algorithms entirely in Python and serves as the **innovation lab** where new approaches and parameters are developed and validated before being migrated to the production ESPHome component.

### 🔬 Role in the ESPectre Ecosystem

Micro-ESPectre is part of a **two-platform strategy**:

| Platform | Purpose | Target Users |
|----------|---------|--------------|
| **[ESPectre](https://github.com/francescopace/espectre)** (ESPHome) | Production deployment | Smart home users, Home Assistant |
| **Micro-ESPectre** (Python) | R&D and prototyping | Researchers, developers, academics |

**Why MQTT instead of Native API?**
Micro-ESPectre uses MQTT for maximum flexibility - it's not tied to Home Assistant and can integrate with:
- 🏭 **Industrial systems** (SCADA, PLCs, factory automation)
- 🎓 **Academic research** (data collection, algorithm validation)
- 🔧 **Custom applications** (any MQTT-compatible platform)

### 💡 Innovation Flow

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
- ⚡ **Instant deployment**: No compilation, ~5 seconds to update
- 🔧 **Easy experimentation**: Modify parameters and test immediately
- 📊 **Quick validation**: Test algorithms and configurations rapidly
- 🔄 **Proven patterns**: Validated algorithms flow to production C++ code

### What is micropython-esp32-csi?

[micropython-esp32-csi](https://github.com/francescopace/micropython-esp32-csi) is a MicroPython fork that I wrote to expose ESP32's CSI (Channel State Information) capabilities to Python. 
This fork makes CSI-based applications accessible to Python developers and enables rapid prototyping of WiFi sensing applications.

## 🆚 Comparison with C++ Version (ESPHome)

### Feature Comparison

| Feature | ESPHome (C++) | Python (MicroPython) | Status |
|---------|---------------|----------------------|--------|
| **Core Algorithm** |
| MVS Segmentation | ✅ | ✅ | ✅ Aligned |
| Spatial Turbulence | ✅ | ✅ | ✅ Aligned |
| Moving Variance | ✅ | ✅ | ✅ Aligned |
| **WiFi Traffic Generator** |
| Traffic Generation | ✅ | ✅ | ✅ Implemented |
| Configurable Rate | ✅ | ✅ | ✅ Implemented |
| **Configuration** |
| YAML Configuration | ✅ | ❌ | ESPHome only |
| MQTT Commands | ❌ | ✅ | Micro-ESPectre only |
| Runtime Config | ✅ (via HA) | ✅ (via MQTT) | Different methods |
| **Storage** |
| NVS Persistence | ✅ | ✅ | ✅ Implemented |
| Auto-save on config change | ✅ | ✅ | ✅ Implemented |
| Auto-load on startup | ✅ | ✅ | ✅ Implemented |
| **Automatic Subcarrier Selection** |
| NBVI Algorithm | ✅ | ✅ | ✅ Implemented |
| Percentile-based Detection | ✅ | ✅ | ✅ Implemented |
| Noise Gate | ✅ | ✅ | ✅ Implemented |
| Spectral De-correlation | ✅ | ✅ | ✅ Implemented |
| **CSI Features** |
| `features_enable` | ❌ | ✅ | `ENABLE_FEATURES = True` in config.py |
| 5 CSI Features | ❌ | ✅ | entropy_turb, iqr_turb, variance_turb, skewness, kurtosis |
| Feature Extraction | ❌| ✅ | Publish-time calculation (no buffer, 92% memory saved) |
| Hampel Filter | ❌ | ✅ | Applied to turbulence (configurable) |

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

📊 **For detailed ESPectre performance metrics and test methodology**, see [PERFORMANCE.md](../PERFORMANCE.md) in the main repository.

### When to Use Which Version?

**Use Micro-ESPectre (Python) if you want:**
- ✅ Quick prototyping and experimentation
- ✅ Easy deployment and updates (~5 seconds)
- ✅ Core motion detection functionality
- ✅ Simple Python-based development
- ✅ MQTT-based runtime configuration
- ✅ Automatic subcarrier selection

**Use ESPectre (ESPHome) if you need:**
- ✅ Native Home Assistant integration (auto-discovery)
- ✅ Maximum performance and efficiency
- ✅ Advanced CSI feature extraction
- ✅ Multiple filtering algorithms
- ✅ Production-grade stability
- ✅ YAML-based configuration

## 📋 Requirements

### Hardware
- ESP32 with CSI support (S3/C6 recommended, other variants supported)
- 2.4GHz WiFi router

### Software
- MicroPython with esp32-csi module installed
- MQTT broker (Home Assistant, Mosquitto, etc.)
- Python 3.12 (Recommended for deployment scripts, CLI, and analysis tools)

## 🔧 CLI Tool Overview

Micro-ESPectre includes a unified command-line tool called **`me`** that simplifies all device operations. This tool will be used throughout the Quick Start guide and beyond.

### Main Commands

The `me` CLI provides these essential commands:

| Command | Description | Usage Example |
|---------|-------------|---------------|
| `flash` | Flash MicroPython firmware to device | `./me flash --erase` |
| `deploy` | Deploy Python code to device | `./me deploy` |
| `run` | Run the application | `./me run` |
| `verify` | Verify firmware installation | `./me verify` |
| *(interactive)* | Interactive MQTT control | `./me` |

### Key Features

- 🔍 **Auto-detection**: Automatically detects serial port and chip type
- ⚡ **Fast deployment**: Updates code in ~5 seconds (no compilation)
- 🎯 **Simple syntax**: Intuitive commands for all operations
- 🔧 **Manual override**: Specify port/chip manually if needed

**Example workflow:**
```bash
./me flash --erase    # Flash firmware (first time only)
./me deploy           # Deploy code
./me run              # Run application
./me                  # Interactive MQTT control
```

> **Note**: The interactive mode (`./me` without arguments) provides advanced MQTT control features and is covered in detail in the [Interactive CLI (Advanced)](#-interactive-cli-advanced) section.

## 🚀 Quick Start

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
- 🔍 Auto-detect your serial port
- 🔍 Auto-detect your chip type
- 📦 Download the correct firmware (cached locally)
- ⚡ Flash it to your device

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

That's it! 🎉 The device will now:
- Connect to WiFi
- Connect to MQTT broker
- Start publishing motion detection data
- Automatically calibrate subcarriers (NBVI algorithm)

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

## 🔧 Additional Commands

### Data Collection (for analysis/research)

```bash
# Collect baseline CSI data (auto-detect port)
./me run --collect-baseline

# Collect movement CSI data (auto-detect port)
./me run --collect-movement
```

Data files are saved to `tools/` directory for analysis with the Python scripts.

### Update Code (during development)

```bash
# Deploy updated code (auto-detect port)
./me deploy

# Run application (auto-detect port)
./me run
```

## 📁 Project Structure

```
micro-espectre/
├── firmware/                  # Downloaded firmware cache (gitignored)
├── src/                       # Main package
│   ├── main.py                # Main application entry point
│   └── ...
├── tools/                     # Analysis and optimization tools
│   ├── 1_analyze_raw_data.py  # Raw CSI data visualization
│   └── ...
├── requirements.txt           # Python dependencies
├── espectre-monitor.html      # Web-based monitoring dashboard
├── espectre-theremin.html     # Audio sonification tool (experimental)
├── me                         # Unified CLI tool (flash/deploy/run/verify/MQTT)
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

### Key Files

- **`me`**: Main CLI tool for flashing firmware, deploying code, running app, and MQTT control
- **`firmware/`**: Downloaded firmware cache (auto-created on first flash)
- **`src/`**: Core Python implementation of motion detection algorithms
- **`tools/`**: Analysis scripts for algorithm development and validation

## ⚙️ Configuration

### Segmentation Parameters (config.py)

```python
SEG_WINDOW_SIZE = 50       # Moving variance window (10-200 packets)
                          # Larger = smoother, slower response
                          # Smaller = faster response, more noise

SEG_THRESHOLD = 1.0       # Motion detection threshold (0.0-10.0)
                          # Lower values = more sensitive to motion
```

### Feature Extraction Parameters (config.py)

```python

ENABLE_FEATURES = False       # Enable/disable feature extraction and MQTT publishing
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ENABLE_FEATURES` | `False` | Enable 10-feature extraction and MQTT publishing |
| `SEG_WINDOW_SIZE` | `50` | Shared window size for both MVS and features |
| `ENABLE_HAMPEL_FILTER` | `True` | Shared Hampel filter enable (used by both MVS and features) |
| `HAMPEL_WINDOW` | `7` | Shared Hampel window size |
| `HAMPEL_THRESHOLD` | `4.0` | Shared Hampel threshold (MAD multiplier) |

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
| `features` | 5 CSI features calculated at publish time |
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

## 🔧 Analysis Tools

The `tools/` directory contains a comprehensive suite of Python scripts for CSI data analysis, algorithm optimization, and subcarrier selection. These tools were instrumental in developing and validating the MVS algorithm and the breakthrough **NBVI (Normalized Baseline Variability Index)** automatic subcarrier selection method.

### Quick Start

```bash
# Collect CSI data samples (auto-detect port)
./me run --collect-baseline
./me run --collect-movement

# Run analysis
cd tools
python 2_analyze_system_tuning.py --quick
python 11_test_nbvi_selection.py
```

### Available Tools

The tools directory includes **14 analysis scripts** covering:
- 📊 Raw data visualization and system tuning
- 🔬 MVS algorithm validation and optimization
- 🎨 I/Q constellation analysis
- 🧬 **NBVI automatic subcarrier selection** (F1=97.1%)
- 🔍 Ring geometry analysis (23+ strategies tested)
- 📈 Detection methods comparison
- 🧮 **CSI features extraction and analysis**

**For complete documentation**, see **[tools/README.md](tools/README.md)** which includes:
- Detailed description of all 12 scripts
- Usage examples and options
- NBVI algorithm explanation and results
- Performance comparisons and scientific findings

### 🧬 NBVI: Breakthrough in Automatic Subcarrier Selection

**NBVI (Normalized Baseline Variability Index)** achieves **F1=97.1%** (pure data) and **F1=91.2%** (mixed data) with **zero manual configuration** - the best automatic method tested among 23+ strategies.

**Key Results**:
- ✅ Gap to manual optimization: only **-0.2%**
- ✅ Outperforms variance-only by **+4.7%** (pure), **∞** (mixed - variance fails)
- ✅ **Percentile-based**: NO threshold configuration needed
- ✅ **Production-ready**: Validated on real CSI data

For complete NBVI documentation, algorithm details, and performance analysis, see **[tools/README.md](tools/README.md)**.

## 🧬 Automatic Subcarrier Selection (NBVI)

Micro-ESPectre implements the **NBVI (Normalized Baseline Variability Index)** algorithm for automatic subcarrier selection, achieving near-optimal performance (F1=97.1%) with **zero manual configuration**.

NBVI automatically selects the optimal 12 subcarriers from the 64 available in WiFi CSI by analyzing their stability and signal strength during a baseline period. The calibration runs automatically:
- **At first boot** (if no saved configuration exists)
- **After factory_reset** command

For complete NBVI documentation, algorithm details, performance analysis, and configuration parameters, see **[tools/README.md](tools/README.md)**.

## 🤖 Advanced Applications & Machine Learning

Micro-ESPectre is the **R&D platform** for advanced CSI-based applications requiring feature extraction and machine learning.

While [ESPectre (ESPHome)](https://github.com/francescopace/espectre) focuses on **production-ready motion detection** using mathematical algorithms (MVS + NBVI), Micro-ESPectre provides the tools and features needed for advanced applications:

- 🔬 **People counting**
- 🏃 **Activity recognition** (walking, falling, sitting, sleeping)
- 📍 **Localization and tracking**
- 👋 **Gesture recognition**

### Available Features for ML Training

Micro-ESPectre extracts **5 CSI features** at publish time for motion detection and ML applications:

| Feature | Fisher J | Type | Description |
|---------|----------|------|-------------|
| **iqr_turb** | 3.56 | Turbulence buffer | IQR approximation (range × 0.5) |
| **skewness** | 2.54 | W=1 (current pkt) | Distribution asymmetry |
| **kurtosis** | 2.24 | W=1 (current pkt) | Distribution tailedness |
| **entropy_turb** | 2.08 | Turbulence buffer | Shannon entropy of turbulence distribution |
| **variance_turb** | 1.21 | Turbulence buffer | Moving variance (reused from MVS) |

> **Note**: Fisher J values tested with `SEG_WINDOW_SIZE=50`. Features are calculated **at publish time only** (not per-packet), saving 92% memory. The analysis tool `12_test_csi_features.py` tests all 10 features for research purposes.

### Tools & Resources

- ✅ **Analysis tools** for dataset collection (see [tools/README.md](tools/README.md))
- ✅ **Feature extraction** pipeline ready for ML training
- ✅ **MQTT-based** data streaming for real-time ML inference
- ✅ **Python-based** for rapid prototyping and experimentation

<details>
<summary>📚 Machine Learning and Deep Learning (click to expand)</summary>

The current implementation uses an **advanced mathematical approach** with 5 features (entropy_turb, iqr_turb, variance_turb, skewness, kurtosis) and multi-criteria detection to identify movement patterns. While this provides excellent results without requiring ML training, scientific research has shown that **Machine Learning** and **Deep Learning** techniques can extract even richer information from CSI data for complex tasks like people counting, activity recognition, and gesture detection.

### Advanced Applications with ML/DL

#### 1. **People Counting**
Classification or regression models can estimate the number of people present in an environment by analyzing complex patterns in CSI.

**References:**
- *Wang et al.* (2017) - "Device-Free Crowd Counting Using WiFi Channel State Information" - IEEE INFOCOM
- *Xi et al.* (2016) - "Electronic Frog Eye: Counting Crowd Using WiFi" - IEEE INFOCOM

#### 2. **Activity Recognition**
Neural networks (CNN, LSTM, Transformer) can classify human activities like walking, falling, sitting, sleeping.

**References:**
- *Wang et al.* (2015) - "Understanding and Modeling of WiFi Signal Based Human Activity Recognition" - ACM MobiCom
- *Yousefi et al.* (2017) - "A Survey on Behavior Recognition Using WiFi Channel State Information" - IEEE Communications Magazine
- *Zhang et al.* (2019) - "WiFi-Based Indoor Robot Positioning Using Deep Neural Networks" - IEEE Access

#### 3. **Localization and Tracking**
Deep learning algorithms can estimate position and trajectory of moving people.

**References:**
- *Wang et al.* (2016) - "CSI-Based Fingerprinting for Indoor Localization: A Deep Learning Approach" - IEEE Transactions on Vehicular Technology
- *Chen et al.* (2018) - "WiFi CSI Based Passive Human Activity Recognition Using Attention Based BLSTM" - IEEE Transactions on Mobile Computing

#### 4. **Gesture Recognition**
Models trained on CSI temporal sequences can recognize hand gestures for touchless control.

**References:**
- *Abdelnasser et al.* (2015) - "WiGest: A Ubiquitous WiFi-based Gesture Recognition System" - IEEE INFOCOM
- *Jiang et al.* (2020) - "Towards Environment Independent Device Free Human Activity Recognition" - ACM MobiCom

### Available Public Datasets

- **UT-HAR**: Human Activity Recognition dataset (University of Texas)
- **Widar 3.0**: Gesture recognition dataset with CSI
- **SignFi**: Sign language recognition dataset
- **FallDeFi**: Fall detection dataset

</details>

<details>
<summary>🛜 Standardized Wi-Fi Sensing (IEEE 802.11bf) (click to expand)</summary>

Currently, only a limited number of Wi-Fi chipsets support CSI extraction, which restricts hardware options for Wi-Fi sensing applications. However, the **IEEE 802.11bf (Wi-Fi Sensing)** standard should significantly improve this situation by making CSI extraction a standardized feature.

### IEEE 802.11bf - Wi-Fi Sensing

The **802.11bf** standard was **[officially published on September 26, 2025](https://standards.ieee.org/ieee/802.11bf/11574/)**, introducing **Wi-Fi Sensing** as a native feature of the Wi-Fi protocol. Main characteristics:

🔹 **Native sensing**: Detection of movements, gestures, presence, and vital signs
🔹 **Interoperability**: Standardized support across different vendors
🔹 **Optimizations**: Specific protocols to reduce overhead and power consumption
🔹 **Privacy by design**: Privacy protection mechanisms integrated into the standard
🔹 **Greater precision**: Improvements in temporal and spatial granularity
🔹 **Existing infrastructure**: Works with already present Wi-Fi infrastructure

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

## 🖥️ Interactive CLI (Advanced)

Beyond the basic commands covered in the [CLI Tool Overview](#-cli-tool-overview), the `me` tool provides an **interactive mode** for advanced device control and monitoring via MQTT.

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

### Web Tools

Two HTML-based tools are included for visualization:

- **`espectre-monitor.html`**: Real-time motion monitoring dashboard with charts
- **`espectre-theremin.html`**: Audio sonification of CSI data (experimental)

**Usage**: Run `webui` command from the interactive CLI to open the web monitor. The CLI starts a local HTTP server automatically.

> ⚠️ **Chrome Users**: If MQTT WebSocket connection fails, you need to enable local network WebSocket access:
> 1. Open `chrome://flags/#local-network-access-check-websockets`
> 2. Set to **"Enabled"**
> 3. Restart Chrome
>
> This is required because Chrome blocks WebSocket connections to private network addresses (like `homeassistant.local`) by default.

## 📡 MQTT Integration

Micro-ESPectre uses MQTT for communication with Home Assistant and runtime configuration.

> **Note**: The main ESPectre component (ESPHome) uses **Native API** instead of MQTT for Home Assistant integration. Micro-ESPectre retains MQTT support for flexibility and compatibility with non-Home Assistant setups.

### Available MQTT Commands

Publish commands to `home/espectre/node1/cmd`:

| Command | Example | Description |
|---------|---------|-------------|
| `info` | `info` | Get system information (network, device, config) |
| `stats` | `stats` | Get runtime statistics (memory, state, metrics) |
| `segmentation_threshold` | `segmentation_threshold 1.5` | Set detection threshold |
| `segmentation_window_size` | `segmentation_window_size 100` | Set window size |
| `subcarrier_selection` | `subcarrier_selection 11,12,13,...` | Set subcarriers |
| `factory_reset` | `factory_reset` | Reset to defaults |

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

### Configuration Persistence

All configuration changes made via MQTT commands are **automatically saved** to a JSON file (`espectre_config.json`) on the ESP32 filesystem and **automatically loaded** on startup, ensuring settings persist across reboots.

## 🏠 Home Assistant Integration

Micro-ESPectre integrates with Home Assistant via MQTT. Add these sensors to your `configuration.yaml`:

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

> **Tip**: For seamless Home Assistant integration with auto-discovery, consider using the main [ESPectre ESPHome component](https://github.com/francescopace/espectre) instead.

## 📚 Scientific References

This section contains extensive research references in Wi-Fi sensing and CSI-based movement detection. These academic works and theses provide valuable insights into mathematical signal processing approaches and machine learning techniques for human activity recognition using Wi-Fi Channel State Information.

### University Theses

1. **Wi-Fi Sensing per Human Identification attraverso CSI**  
   University thesis (in Italian) covering CSI data collection for human recognition through Wi-Fi signal analysis, with in-depth exploration of mathematical signal processing methods.  
   📄 [Read thesis](https://amslaurea.unibo.it/id/eprint/29166/1/tesi.pdf)

2. **Channel State Information (CSI) Features Collection in Wi-Fi**  
   Detailed analysis of CSI feature collection and processing in Wi-Fi environments, with methods for extraction and analysis suitable for mathematical processing.  
   📄 [Read thesis](https://www.politesi.polimi.it/handle/10589/196727)

3. **Wi-Fi CSI for Human Activity Recognition** - UBC  
   Baseline detection and calibration-free approaches for activity recognition.  
   📄 [Read thesis](https://open.library.ubc.ca/media/stream/pdf/24/1.0365967/4)

### Scientific Papers

4. **Indoor Motion Detection Using Wi-Fi Channel State Information (2018)**  
   Scientific article describing indoor movement detection using CSI with approaches based on signal mathematics and physics. False positive reduction and sensitivity optimization.  
   📄 [Read paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6068568/)

5. **WiFi Motion Detection: A Study into Efficacy and Performance (2019)**  
   Study using CSI data collected from standard devices to detect movements, with analysis of signal processing methods to extract movement events without relying on ML.  
   📄 [Read paper](https://arxiv.org/abs/1908.08476)

6. **CSI-HC: A WiFi-Based Indoor Complex Human Motion Recognition Using Channel State Information (2020)**  
   Recognition of complex indoor movements through CSI with methods based on mathematical signal features, ideal for projects with signal-based analysis without advanced ML.  
   📄 [Read paper](https://onlinelibrary.wiley.com/doi/10.1155/2020/3185416)

7. **Location Intelligence System for People Estimation in Indoor Environment During Emergency Operation (2022)**  
   Demonstrates the use of ESP32 with wavelet filtering (Daubechies db4) for people detection in emergency scenarios. This paper directly influenced ESPectre's wavelet filter implementation.  
   📄 [Read paper](https://scholarspace.manoa.hawaii.edu/server/api/core/bitstreams/a2d2de7c-7697-485b-97c5-62f4bf1260d0/content)

8. **Mitigation of CSI Temporal Phase Rotation** - PMC  
   B2B calibration methods for phase analysis in CSI-based sensing.  
   📄 [Read paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6263436/)

9. **CSI-based Passive Intrusion Detection** - NIH  
   Multipath components and subcarrier sensitivity analysis for intrusion detection.  
   📄 [Read paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11630712/)

10. **Time-Selective RNN for Multi-Room Detection** - arXiv  
    Environment-dependent channel optimization for multi-room scenarios.  
    📄 [Read paper](https://arxiv.org/html/2304.13107v2)

11. **CIRSense: Rethinking WiFi Sensing** - arXiv  
    SSNR (Sensing Signal-to-Noise Ratio) optimization for sensing applications.  
    📄 [Read paper](https://arxiv.org/html/2510.11374v1)

12. **MVS Segmentation** - ResearchGate  
    Moving Variance Segmentation algorithm: the fused CSI stream and corresponding moving variance sequence.  
    📄 [Read paper](https://www.researchgate.net/figure/MVS-segmentation-a-the-fused-CSI-stream-b-corresponding-moving-variance-sequence_fig6_326244454)

13. **CSI-F: Feature Fusion Method** - MDPI  
    Hampel filter and statistical robustness for CSI feature extraction.  
    📄 [Read paper](https://www.mdpi.com/1424-8220/24/3/862)

14. **Linear-Complexity Subcarrier Selection** - ResearchGate  
    Computational efficiency strategies for embedded systems.  
    📄 [Read paper](https://www.researchgate.net/publication/397240630)

15. **Passive Indoor Localization** - PMC  
    SNR considerations and noise gate strategies for indoor localization.  
    📄 [Read paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6412876/)

16. **Subcarrier Selection for Indoor Localization** - ResearchGate  
    Spectral de-correlation and feature diversity for optimal subcarrier selection.  
    📄 [Read paper](https://www.researchgate.net/publication/326195991)

These references demonstrate that effective Wi-Fi sensing can be achieved through both mathematical and machine learning approaches, supporting Micro-ESPectre's role as the R&D platform for algorithm development and validation.

## Related Projects

- [ESPectre (ESPHome)](https://github.com/francescopace/espectre) - Main project with native Home Assistant integration
- [micropython-esp32-csi](https://github.com/francescopace/micropython-esp32-csi) - MicroPython CSI module

## 📄 License

GPLv3 - See [LICENSE](../LICENSE) for details.
