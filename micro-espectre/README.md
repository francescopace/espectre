# Micro-ESPectre

**Motion detection system based on Wi-Fi CSI (Channel State Information) - Pure Python implementation for MicroPython**

Micro-ESPectre is a lightweight Python port of [ESPectre](https://github.com/francescopace/espectre) designed to run on MicroPython-enabled ESP32 devices. It brings the core motion detection capabilities of ESPectre to resource-constrained environments with easy deployment and no compilation required.

## 🎯 What is Micro-ESPectre?

Micro-ESPectre implements the **MVS (Moving Variance Segmentation)** algorithm from ESPectre in pure Python. It focuses on the essential motion detection functionality while maintaining full backward compatibility with ESPectre's MQTT command interface.

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
| `csi_raw_capture` | ✅ | ❌ | Not implemented |
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

# Deploy and run debug script (diagnose CSI issues)
./deploy.sh /dev/cu.usbmodem* --debug
```

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
│   ├── traffic_generator.py   # UDP traffic generator
│   ├── nvs_storage.py         # JSON-based config persistence
│   └── mqtt/                  # MQTT sub-package
│       ├── __init__.py        # MQTT package initialization
│       ├── handler.py         # MQTT connection and publishing
│       └── commands.py        # MQTT command processing
├── config_local.py            # Local config override (gitignored)
├── config_local.py.example    # Configuration template
├── deploy.sh                  # Deployment script
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

## ⚙️ Configuration

### Segmentation Parameters (config.py)

```python
SEG_WINDOW_SIZE = 100      # Moving variance window (3-50 packets)
                          # Larger = smoother, slower response
                          # Smaller = faster response, more noise

SEG_THRESHOLD = 3.0       # Base threshold 
                          # Lower values = more sensitive to motion
```

### Publishing Parameters

```python
SMART_PUBLISHING = True   # Only publish on significant changes
DELTA_THRESHOLD = 0.05    # Minimum variance change to trigger publish
MAX_PUBLISH_INTERVAL_MS = 5000  # Max time between publishes (heartbeat)
```
### Published Data (same as ESPectre)

The system publishes JSON payloads to the configured MQTT topic (default: `home/espectre/node1`):

```json
{
  "movement": 0.0234,            // Current moving variance
  "threshold": 3.0,              // Current threshold
  "state": "idle",               // "idle" or "motion"
  "packets_processed": 42,       // Packets since last publish
  "packets_dropped": 0,          // Packets dropped since last publish
  "timestamp": 1700000000        // Unix timestamp
}
```

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
