# Micro-ESPectre

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

The `tools/` directory contains Python scripts for analyzing CSI data and optimizing system parameters. These tools were instrumental in developing and tuning the MVS algorithm.

### Data Collection

First, collect CSI data samples using the deployment script:

```bash
# Collect baseline data (no movement)
./deploy.sh /dev/cu.usbmodem* --collect-baseline

# Collect movement data (with movement)
./deploy.sh /dev/cu.usbmodem* --collect-movement
```

This creates `baseline_data.bin` and `movement_data.bin` in the `tools/` directory.

### Analysis Scripts

**1. Raw Data Analysis** (`1_analyze_raw_data.py`)
- Visualizes raw CSI amplitude data
- Analyzes subcarrier patterns and noise characteristics
- Helps identify most informative subcarriers
- **Options**: None

**2. System Tuning** (`2_analyze_system_tuning.py`)
- Comprehensive grid search for optimal MVS parameters
- Tests subcarrier clusters, thresholds, and window sizes
- Always shows confusion matrix for best configuration
- **Options**: 
  - `--quick`: Reduced parameter space (faster)

**3. MVS Visualization** (`3_analyze_moving_variance_segmentation.py`)
- Visualizes MVS algorithm behavior with current configuration
- Shows moving variance, threshold, and detection states
- Always displays confusion matrix and performance metrics
- **Options**:
  - `--plot`: Show MVS graphs (baseline + movement)

**4. Filter Location Analysis** (`4_analyze_filter_location.py`)
- Compares filter placement in processing pipeline
- Tests pre-filtering vs post-filtering approaches
- Evaluates impact on motion detection accuracy
- **Options**:
  - `--plot`: Show comparison visualizations

**5. Filter Turbulence Analysis** (`5_analyze_filter_turbulence.py`)
- Analyzes turbulence calculation with different filters
- Compares filtered vs unfiltered turbulence
- Validates Hampel filter effectiveness
- **Options**:
  - `--plot`: Show filter comparison plots
  - `--optimize-filters`: Optimize filter parameters

**6. Hampel Parameter Optimization** (`6_optimize_hampel_parameters.py`)
- Grid search over Hampel filter parameters
- Tests window sizes (3-9) and thresholds (2.0-4.0)
- Finds optimal outlier detection configuration
- **Options**: None

**7. Variance Algorithm Comparison** (`7_analyze_variance_algo.py`)
- Compares one-pass vs two-pass variance algorithms
- Analyzes numerical stability with large values
- Validates algorithm selection for production
- **Options**: None

**8. Detection Methods Comparison** (`8_compare_detection_methods.py`)
- Compares RSSI, Mean Amplitude, Turbulence, and MVS
- Demonstrates MVS superiority for movement detection
- Shows why MVS provides best separation
- **Options**:
  - `--plot`: Show comparison visualization (4×2 layout)

![Detection Methods Comparison](../images/detection_method_comparison.png)
*Comparison of detection methods (RSSI, Mean Amplitude, Turbulence, MVS) demonstrating MVS superiority with lowest false positives and highest recall*

**9. Retroactive Auto-Calibration Test** (`9_test_retroactive_calibration.py`)
- Tests automatic subcarrier selection using retrospective baseline detection
- Validates adaptive threshold for universal deployment
- Tests with random starting bands and realistic mixed data
- **Key Findings**:
  - Works with pure baseline blocks (F1=96.7%)
  - Adaptive threshold enables calibration from ANY starting band (6/6 success)
  - Fails with realistic mixed data (80% baseline, 20% movement scattered)
  - Conclusion: Fully automatic calibration difficult; recommend robust default + optional manual calibration
- **Options**: None (runs all tests automatically)

**10. I/Q Constellation Plotter** (`10_plot_constellation.py`)
- Visualizes I/Q constellation diagrams for CSI subcarriers
- Compares baseline (stable) vs movement (dispersed) patterns
- Shows all 64 subcarriers (top row) and selected subcarriers (bottom row)
- Useful for understanding signal quality and subcarrier behavior
- **Options**:
  - `--packets N`: Number of contiguous packets to plot (default: 100)
  - `--offset N`: Starting packet index (default: 0)
  - `--subcarriers LIST`: Comma-separated subcarrier indices (default: from config.py)
  - `--grid`: Use grid layout (one subplot per subcarrier)

### Usage Example

```bash
cd tools

# Analyze raw CSI data
python 1_analyze_raw_data.py

# Optimize segmentation parameters (shows confusion matrix)
python 2_analyze_system_tuning.py

# Quick mode (faster)
python 2_analyze_system_tuning.py --quick

# Plot I/Q constellations (100 packets)
python 10_plot_constellation.py

# Plot more packets with offset
python 10_plot_constellation.py --packets 200 --offset 50

# Plot specific subcarriers
python 10_plot_constellation.py --subcarriers 47,48,49,50

# Use grid layout (one subplot per subcarrier)
python 10_plot_constellation.py --grid
```
**Benefits:**
- 📊 Visual feedback on algorithm performance
- 🎯 Data-driven parameter optimization
- 🔬 Scientific validation of design choices
- ⚡ Faster iteration than C firmware testing

## 📡 MQTT Integration

Micro-ESPectre maintains **full backward compatibility** with ESPectre's MQTT command interface. Every MQTT commands are supported:

- System monitoring: `info`, `stats`
- Segmentation tuning: `segmentation_threshold`, `segmentation_window_size`
- Configuration: `subcarrier_selection`, `traffic_generator_rate`, `smart_publishing`
- Maintenance: `factory_reset`

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
