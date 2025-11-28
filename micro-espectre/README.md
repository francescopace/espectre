# 🛜 Micro-ESPectre 👻

**Motion detection system based on Wi-Fi CSI (Channel State Information) - Pure Python implementation for MicroPython**

Micro-ESPectre is a lightweight Python port of [ESPectre](https://github.com/francescopace/espectre) designed to run on MicroPython-enabled ESP32 devices. It brings the core motion detection capabilities of ESPectre to resource-constrained environments with easy deployment and no compilation required.

## 🎯 What is Micro-ESPectre?

Micro-ESPectre implements the **MVS (Moving Variance Segmentation)** algorithm from ESPectre in pure Python. It focuses on the essential motion detection functionality while maintaining full backward compatibility with ESPectre's MQTT command interface.

### 🔬 Role in Development

Micro-ESPectre serves a dual purpose:

1. **Production Use**: Lightweight motion detection for resource-constrained environments
2. **Development Tool**: Rapid prototyping and parameter tuning platform

The Python implementation enables **fast iteration cycles** for testing configurations and algorithms without the overhead of C compilation. Successful patterns and optimized parameters discovered in Micro-ESPectre are then ported back to the C firmware with confidence. This approach significantly accelerated the development of ESPectre v1.4.0's refactoring and optimization work.

**Key Benefits for Development:**
- ⚡ **Instant deployment**: No compilation, ~5 seconds to update
- 🔧 **Easy experimentation**: Modify parameters and test immediately
- 📊 **Quick validation**: Test algorithms and configurations rapidly
- 🔄 **Bidirectional sync**: Proven patterns flow back to C implementation

### What is esp32-microcsi?

[esp32-microcsi](https://github.com/francescopace/esp32-microcsi) is a MicroPython module that I wrote to expose ESP32's CSI (Channel State Information) capabilities to Python. This module makes CSI-based applications accessible to Python developers and enables rapid prototyping of WiFi sensing applications.

## 🆚 Comparison with C Version

### Feature Comparison

| Feature | C (ESP-IDF) | Python (MicroPython) | Status |
|---------|-------------|----------------------|--------|
| **Core Algorithm** |
| MVS Segmentation | ✅ | ✅ | ✅ Aligned |
| Spatial Turbulence | ✅ | ✅ | ✅ Aligned |
| Moving Variance | ✅ | ✅ | ✅ Aligned |
| **WiFi Traffic Generator** |
| Traffic Generation | ✅ (ICMP ping) | ✅ (DNS/UDP) | ✅ Implemented |
| Configurable Rate | ✅ | ✅ | ✅ Implemented |
| **MQTT Commands** |
| `info` | ✅ | ✅ | ✅ Implemented |
| `stats` | ✅ | ✅ | ✅ Implemented |
| `segmentation_threshold` | ✅ | ✅ | ✅ Implemented |
| `segmentation_window_size` | ✅ | ✅ | ✅ Implemented |
| `subcarrier_selection` | ✅ | ✅ | ✅ Implemented |
| `traffic_generator_rate` | ✅ | ✅ | ✅ Implemented |
| `smart_publishing` | ✅ | ✅ | ✅ Implemented |
| `factory_reset` | ✅ | ✅ | ✅ Implemented |
| **Storage** |
| NVS Persistence | ✅ | ✅ (JSON file) | ✅ Implemented |
| Auto-save on config change | ✅ | ✅ | ✅ Implemented |
| Auto-load on startup | ✅ | ✅ | ✅ Implemented |
| **Automatic Subcarrier Selection** |
| NBVI Algorithm | ✅ | ✅ | ✅ Implemented |
| Percentile-based Detection | ✅ | ✅ | ✅ Implemented |
| Noise Gate | ✅ | ✅ | ✅ Implemented |
| Spectral De-correlation | ✅ | ✅ | ✅ Implemented |
| **CSI Features** |
| `features_enable` | ✅ | ❌ | Not implemented |
| 10 CSI Features | ✅ | ❌ | Not implemented |
| Feature Extraction | ✅ | ❌ | Not implemented |
| Hampel Filter | ✅ | ❌ | Not implemented |
| Savitzky-Golay Filter | ✅ | ❌ | Not implemented |
| Butterworth Filter | ✅ | ❌ | Not implemented |
| Wavelet Filter | ✅ | ❌ | Not implemented |

### Performance Comparison

| Metric | C (ESP-IDF) | Python (MicroPython) |
|--------|-------------|----------------------|
| Performance | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Memory Usage | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Ease of Use | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Deployment | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Build Time | ~5 minutes | Instant (no build) |
| Update Time | ~5 minutes | ~5 seconds |

### When to Use Which Version?

**Use Micro-ESPectre (Python) if you want:**
- ✅ Quick prototyping and experimentation
- ✅ Easy deployment and updates
- ✅ Core motion detection functionality
- ✅ Simple Python-based development
- ✅ MQTT-based runtime configuration
- ✅ Automatic subcarrier selection

**Use ESPectre (C) if you need:**
- ✅ Maximum performance and efficiency
- ✅ Advanced CSI feature extraction
- ✅ Multiple filtering algorithms
- ✅ Raw CSI data capture
- ✅ Production-grade stability

## 📋 Requirements

### Hardware
- ESP32-S3 or ESP32-C6 board
- 2.4GHz WiFi router

### Software
- MicroPython with esp32-microcsi module installed
- MQTT broker (Home Assistant, Mosquitto, etc.)

## 🚀 Quick Start

### 1. Install MicroPython with CSI Support 

**Note:** This step is only required once to flash the patched MicroPython firmware with esp32-microcsi module to your device.

Follow the instructions at [esp32-microcsi](https://github.com/francescopace/esp32-microcsi):

```bash
# Clone esp32-microcsi repository
git clone https://github.com/francescopace/esp32-microcsi
cd esp32-microcsi

# Setup environment
./scripts/setup_env.sh

# Integrate CSI module
./scripts/integrate_csi.sh

# Build and flash (ESP32-S3)
./scripts/build_flash.sh -b ESP32_GENERIC_S3

# Or for ESP32-C6
./scripts/build_flash.sh -b ESP32_GENERIC_C6
```

### 2. Configure WiFi and MQTT

Create `config_local.py` from the template:

```bash
cp config_local.py.example config_local.py
```

Edit `config_local.py` with your credentials:

```python
# WiFi Configuration
WIFI_SSID = "YourWiFiSSID"
WIFI_PASSWORD = "YourWiFiPassword"

# MQTT Configuration
MQTT_BROKER = "homeassistant.local"  # Your MQTT broker IP or hostname
MQTT_PORT = 1883
MQTT_USERNAME = "username"
MQTT_PASSWORD = "password"
```

**Note**: `config_local.py` overrides the defaults in `config.py`. You can also customize other settings like topic, buffer size, etc.

### 3. Upload Files to ESP32

Use the deployment script:

```bash
# Deploy only (upload files)
./deploy.sh /dev/cu.usbmodem*

# Deploy and run main application
./deploy.sh /dev/cu.usbmodem* --run

# Deploy and collect baseline data (for testing/analysis)
./deploy.sh /dev/cu.usbmodem* --collect-baseline

# Deploy and collect movement data (for testing/analysis)
./deploy.sh /dev/cu.usbmodem* --collect-movement
```

**Data Collection:**
The `--collect-baseline` and `--collect-movement` flags are used to collect CSI data samples for algorithm testing and parameter tuning. The collected binary files are automatically downloaded to the `tools/` directory and can be analyzed with the Python analysis scripts.

### 4. Run

```bash
# Run main application
mpremote connect /dev/cu.usbmodem* run src/main.py

# Or connect to REPL and run
mpremote connect /dev/cu.usbmodem*
>>> from src import main
>>> main.main()
```

## 📁 Project Structure

```
micro-espectre/
├── src/                       # Main package
│   ├── __init__.py            # Package initialization
│   ├── main.py                # Main application entry point
│   ├── config.py              # Default configuration
│   ├── segmentation.py        # MVS segmentation logic
│   ├── traffic_generator.py   # WiFi traffic generator
│   ├── nvs_storage.py         # JSON-based config persistence
│   ├── filters.py             # Signal filtering (Hampel filter)
│   ├── nbvi_calibrator.py     # NBVI automatic subcarrier selection
│   ├── data_collector.py      # CSI data collection for testing
│   └── mqtt/                  # MQTT sub-package
│       ├── __init__.py        # MQTT package initialization
│       ├── handler.py         # MQTT connection and publishing
│       └── commands.py        # MQTT command processing
├── tools/                     # Analysis and optimization tools
│   └── ...
├── config_local.py            # Local config override (gitignored)
├── config_local.py.example    # Configuration template
├── deploy.sh                  # Deployment script
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

## ⚙️ Configuration

### Segmentation Parameters (config.py)

```python
SEG_WINDOW_SIZE = 50       # Moving variance window (10-200 packets)
                          # Larger = smoother, slower response
                          # Smaller = faster response, more noise

SEG_THRESHOLD = 1.0       # Motion detection threshold (0.0-10.0)
                          # Lower values = more sensitive to motion
```

### Published Data (same as ESPectre)

The system publishes JSON payloads to the configured MQTT topic (default: `home/espectre/node1`):

```json
{
  "movement": 0.0234,            // Current moving variance
  "threshold": 1.0,              // Current threshold
  "state": "idle",               // "idle" or "motion"
  "packets_processed": 42,       // Packets since last publish
  "packets_dropped": 0,          // Packets dropped since last publish
  "timestamp": 1700000000        // Unix timestamp
}
```

## 🔧 Analysis Tools

The `tools/` directory contains a comprehensive suite of Python scripts for CSI data analysis, algorithm optimization, and subcarrier selection. These tools were instrumental in developing and validating the MVS algorithm and the breakthrough **NBVI (Normalized Baseline Variability Index)** automatic subcarrier selection method.

### Quick Start

```bash
# Collect CSI data samples
./deploy.sh /dev/cu.usbmodem* --collect-baseline
./deploy.sh /dev/cu.usbmodem* --collect-movement

# Run analysis
cd tools
python 2_analyze_system_tuning.py --quick
python 11_test_nbvi_selection.py
```

### Available Tools

The tools directory includes **11 analysis scripts** covering:
- 📊 Raw data visualization and system tuning
- 🔬 MVS algorithm validation and optimization
- 🎨 I/Q constellation analysis
- 🧬 **NBVI automatic subcarrier selection** (F1=97.1%)
- 🔍 Ring geometry analysis (23+ strategies tested)
- 📈 Detection methods comparison

**For complete documentation**, see **[tools/README.md](tools/README.md)** which includes:
- Detailed description of all 11 scripts
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

## 📡 MQTT Integration

Micro-ESPectre maintains **full backward compatibility** with ESPectre's MQTT command interface.

For detailed documentation on MQTT commands, payloads, and usage examples, see the [ESPectre SETUP.md - MQTT Commands Reference](https://github.com/francescopace/espectre/blob/main/SETUP.md#mqtt-commands-reference).

### Configuration Persistence

All configuration changes made via MQTT commands are **automatically saved** to a JSON file (`espectre_config.json`) on the ESP32 filesystem and **automatically loaded** on startup, ensuring settings persist across reboots.

## 🏠 Home Assistant Integration

Micro-ESPectre uses the same MQTT topics and data format as ESPectre, so the Home Assistant configuration is identical.

For detailed Home Assistant integration instructions, see the [ESPectre SETUP.md - Home Assistant section](https://github.com/francescopace/espectre/blob/main/SETUP.md#home-assistant).

## 📚 References

- [ESPectre (C/ESP-IDF)](https://github.com/francescopace/espectre)
- [esp32-microcsi](https://github.com/francescopace/esp32-microcsi)
- [MicroPython](https://micropython.org/)

## 📄 License

GPLv3 - See ESPEctre LICENSE file for details

## 👤 Author

**Francesco Pace**  
📧 Email: francesco.pace@gmail.com  
💼 LinkedIn: [linkedin.com/in/francescopace](https://www.linkedin.com/in/francescopace/)
